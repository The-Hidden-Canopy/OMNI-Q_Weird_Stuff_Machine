# Portions derived from *The Hidden Canopy LLC* —
# [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
# Vendored from `src/ida_train/models/ida_lattice/multiscale_memory.py`; body unmodified except as recorded in ../SOURCE.md.

from __future__ import annotations

import math

import torch
from torch import nn

from .tensor_contracts import (
    TENSOR_CONTRACT_BACKEND_EINSUM,
    TENSOR_CONTRACT_BACKEND_MATMUL,
    contract_anchor_bank,
)


class MultiscaleMemoryBank(nn.Module):
    """GPU-resident multiscale temporal anchor contraction.

    For J time scales and A stored anchors:

        tau_j      = exp(log_tau_j).clamp(tau_min, tau_max)        learnable
        decay[j,a] = exp(-elapsed[a] / tau_j)
        W[j,b,a]   = softmax_a( relevance[b,a] + log(decay[j,a]) )
        M_j[b,h]   = einsum('jba,ah->jbh', W, anchor_bank)
        hat_m[b,h] = sum_j scale_w_j * M_j[b,h]   (scale_w = softmax of learned weights)
        output     = current + sigmoid(gate([current; hat_m])) * hat_m

    FLOPs for AI body (J=8, A=32, H=2048):
        relevance:    B * A * H  = B * 32 * 2048  matmul
        contraction:  J * B * A * H = 8 * B * 32 * 2048 = 1.049M per sample
    This is ~0.026% of a full AI decode step (4.034B FLOPs).
    """

    def __init__(
        self,
        hidden_size: int,
        num_scales: int = 8,
        num_anchors: int = 32,
        tau_min: float = 1.0,
        tau_max: float = 64.0,
        contract_backend: str = TENSOR_CONTRACT_BACKEND_EINSUM,
        contract_fallback: str = TENSOR_CONTRACT_BACKEND_MATMUL,
        recall_only: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_scales = max(1, int(num_scales))
        self.num_anchors = max(1, int(num_anchors))
        self._tau_min = float(tau_min)
        self._tau_max = float(max(tau_min + 1.0, tau_max))
        self.contract_backend = (
            str(contract_backend or TENSOR_CONTRACT_BACKEND_EINSUM).strip().lower()
            or TENSOR_CONTRACT_BACKEND_EINSUM
        )
        self.contract_fallback = (
            str(contract_fallback or TENSOR_CONTRACT_BACKEND_MATMUL).strip().lower()
            or TENSOR_CONTRACT_BACKEND_MATMUL
        )
        self._last_contract_backend_resolved = self.contract_backend

        log_tau_init = torch.linspace(
            math.log(max(self._tau_min, 1e-3)),
            math.log(self._tau_max),
            self.num_scales,
        )
        self.log_tau = nn.Parameter(log_tau_init)
        self.scale_weights = nn.Parameter(torch.zeros(self.num_scales))

        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_gate = None if recall_only else nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        current_hidden: torch.Tensor,     # [B, H]
        anchor_bank: torch.Tensor,        # [A, H]
        elapsed_times: torch.Tensor,      # [A]  steps since each anchor (≥1)
    ) -> dict[str, torch.Tensor]:
        B, H = current_hidden.shape
        A = anchor_bank.size(0)
        J = self.num_scales

        # Relevance [B, A]
        q = self.query_proj(current_hidden)                              # [B, H]
        k = self.key_proj(anchor_bank)                                   # [A, H]
        relevance = torch.matmul(q, k.t()) * (H ** -0.5)                # [B, A]

        # Time-decay log-weights [J, A]
        tau = self.log_tau.exp().clamp(self._tau_min, self._tau_max)    # [J]
        elapsed = elapsed_times.to(device=tau.device, dtype=tau.dtype)  # [A]
        log_decay = -(elapsed.unsqueeze(0) / tau.unsqueeze(1))          # [J, A]

        # Per-scale softmax attention [J, B, A]
        per_scale_attn = torch.softmax(
            relevance.unsqueeze(0) + log_decay.unsqueeze(1),            # [J, B, A]
            dim=-1,
        )

        # Weighted anchor contraction [J, B, H]
        scale_reconstructions, resolved_backend = contract_anchor_bank(
            per_scale_attn,
            anchor_bank,
            backend=self.contract_backend,
            fallback=self.contract_fallback,
        )
        if self.output_gate is not None:
            self._last_contract_backend_resolved = resolved_backend

        # Scale mixture [B, H]
        scale_w = torch.softmax(self.scale_weights, dim=0)              # [J]
        hat_m = (scale_reconstructions * scale_w.view(J, 1, 1)).sum(0) # [B, H]

        # Gated blend
        if self.output_gate is None:
            return {"recall": hat_m}
        gate_input = torch.cat([current_hidden, hat_m], dim=-1)         # [B, 2H]
        blend_gate = torch.sigmoid(self.output_gate(gate_input))        # [B, H]
        output = current_hidden + blend_gate * hat_m                    # [B, H]

        # Diagnostics
        weight_entropy = (
            -(per_scale_attn * per_scale_attn.clamp_min(1e-9).log())
            .sum(dim=-1)
            .mean()
        )

        return {
            "recall": hat_m,
            "reconstructed": output,                                          # [B, H]
            "blend_gate_mean": blend_gate.mean(),                             # scalar
            "reconstruction_norm": output.norm(dim=-1).mean(),               # scalar
            "weight_entropy": weight_entropy,                                 # scalar
            "scale_concentration": scale_w.max(),                            # scalar
            "anchor_count": output.new_tensor(float(A)),                     # scalar
        }
