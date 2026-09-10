# Portions derived from *The Hidden Canopy LLC* —
# [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
# Vendored from `src/ida_train/models/ida_lattice/constitutional_router.py`; body unmodified except as recorded in ../SOURCE.md.

from __future__ import annotations

import torch
from torch import nn


class ConstitutionalRouter(nn.Module):
    """Sparse router from cognitive pressure into expert participation."""

    def __init__(
        self,
        hidden_size: int,
        num_routes: int,
        top_k: int,
        *,
        pressure_size: int | None = None,
    ) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, num_routes)
        self.num_routes = int(num_routes)
        self.top_k = max(1, min(int(top_k), self.num_routes))

        resolved_pressure_size = int(pressure_size or num_routes)

        self.pressure_to_routes = (
            nn.Identity()
            if resolved_pressure_size == self.num_routes
            else nn.Linear(
                resolved_pressure_size,
                self.num_routes,
                bias=False,
            )
        )

    def forward(
        self,
        hidden: torch.Tensor,
        pressure: torch.Tensor,
        *,
        tokenwise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routed_pressure = self.pressure_to_routes(pressure)
        if tokenwise:
            if hidden.ndim != 3:
                raise ValueError("tokenwise routing requires hidden with shape [batch, sequence, hidden]")
            logits = self.score(hidden) + routed_pressure.unsqueeze(1)
        else:
            pooled = hidden.mean(dim=1)
            logits = self.score(pooled) + routed_pressure
        scores = torch.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(
            scores,
            k=min(self.top_k, self.num_routes),
            dim=-1,
        )
        sparse = torch.zeros_like(scores)
        sparse.scatter_(-1, top_idx, top_vals)
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return sparse, logits
