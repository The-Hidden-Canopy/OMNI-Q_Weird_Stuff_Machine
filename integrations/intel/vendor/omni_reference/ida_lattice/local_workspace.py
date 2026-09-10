# Portions derived from *The Hidden Canopy LLC* —
# [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
# Vendored from `src/ida_train/models/ida_lattice/local_workspace.py`; body unmodified except as recorded in ../SOURCE.md.

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from .projections import build_projection, run_projection

# Module-level LRU cache for local-causal attention masks.
# Key: (device_type, device_index, seq_len, window)
# Value: bool mask of shape (seq_len, seq_len), True = block this position
#
# Semantics match nn.MultiheadAttention's attn_mask convention.
# Built once per (device, seq_len, window) combination; reused across all
# layers and all forward passes with the same shape.  Bounded at 8 entries
# to handle multiple GPUs, multiple curriculum sequence lengths, and
# any window variants without unbounded growth.
_MASK_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()
_MASK_CACHE_MAX = 8


def get_local_causal_mask(
    seq_len: int,
    window: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Return the (seq_len, seq_len) local causal attention mask.

    True = block this position (nn.MultiheadAttention convention).
    Positions are blocked when they are in the future (causal) or
    further than `window` tokens in the past (local).

    The mask is built on the first call for a given (device, seq_len, window)
    and served from cache on every subsequent call.
    """
    key = (device.type, device.index, seq_len, window)
    if key in _MASK_CACHE:
        _MASK_CACHE.move_to_end(key)
        return _MASK_CACHE[key]

    q = torch.arange(seq_len, device=device).unsqueeze(1)
    k = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = (k > q) | ((q - k) >= window)

    _MASK_CACHE[key] = mask
    while len(_MASK_CACHE) > _MASK_CACHE_MAX:
        _MASK_CACHE.popitem(last=False)

    return mask


class LocalAttentionWorkspace(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        local_attention_window: int = 128,
        *,
        fp8_enabled: bool = False,
        fp8_backend: str = "off",
        fp8_scope: list[str] | tuple[str, ...] | None = None,
        fp8_fallback: str = "native_scaled_mm",
        fp8_recipe: str = "delayed_hybrid",
        fp8_amax_history_len: int = 16,
        fp8_amax_compute_algo: str = "max",
        fp8_weight_cache: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = build_projection(
            self.hidden_size,
            self.hidden_size,
            fp8_enabled=fp8_enabled,
            fp8_backend=fp8_backend,
            fp8_scope=fp8_scope,
            fp8_fallback=fp8_fallback,
            fp8_recipe=fp8_recipe,
            fp8_amax_history_len=fp8_amax_history_len,
            fp8_amax_compute_algo=fp8_amax_compute_algo,
            fp8_weight_cache=fp8_weight_cache,
            surface="workspace",
            module_name="workspace.q_proj",
        )
        self.k_proj = build_projection(
            self.hidden_size,
            self.hidden_size,
            fp8_enabled=fp8_enabled,
            fp8_backend=fp8_backend,
            fp8_scope=fp8_scope,
            fp8_fallback=fp8_fallback,
            fp8_recipe=fp8_recipe,
            fp8_amax_history_len=fp8_amax_history_len,
            fp8_amax_compute_algo=fp8_amax_compute_algo,
            fp8_weight_cache=fp8_weight_cache,
            surface="workspace",
            module_name="workspace.k_proj",
        )
        self.v_proj = build_projection(
            self.hidden_size,
            self.hidden_size,
            fp8_enabled=fp8_enabled,
            fp8_backend=fp8_backend,
            fp8_scope=fp8_scope,
            fp8_fallback=fp8_fallback,
            fp8_recipe=fp8_recipe,
            fp8_amax_history_len=fp8_amax_history_len,
            fp8_amax_compute_algo=fp8_amax_compute_algo,
            fp8_weight_cache=fp8_weight_cache,
            surface="workspace",
            module_name="workspace.v_proj",
        )
        self.out_proj = build_projection(
            self.hidden_size,
            self.hidden_size,
            fp8_enabled=fp8_enabled,
            fp8_backend=fp8_backend,
            fp8_scope=fp8_scope,
            fp8_fallback=fp8_fallback,
            fp8_recipe=fp8_recipe,
            fp8_amax_history_len=fp8_amax_history_len,
            fp8_amax_compute_algo=fp8_amax_compute_algo,
            fp8_weight_cache=fp8_weight_cache,
            surface="workspace",
            module_name="workspace.out_proj",
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.local_attention_window = max(1, int(local_attention_window))

    def build_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Kept for backward compatibility. Prefer get_local_causal_mask() for caching."""
        query_positions = torch.arange(seq_len, device=device).unsqueeze(1)
        key_positions = torch.arange(seq_len, device=device).unsqueeze(0)
        future_mask = key_positions > query_positions
        distance = query_positions - key_positions
        local_mask = distance >= self.local_attention_window
        return future_mask | local_mask

    def _shape_heads(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        return hidden.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def precision_runtime_summary(self) -> dict[str, object]:
        backend_counts: dict[str, int] = {}
        for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            backend = str(getattr(module, "_fp8_backend", "off") or "off")
            backend_counts[backend] = int(backend_counts.get(backend, 0)) + 1
        resolved_backend = "mixed"
        if len(backend_counts) == 1:
            resolved_backend = next(iter(backend_counts))
        elif not backend_counts:
            resolved_backend = "off"
        return {
            "resolved_backend": resolved_backend,
            "backend_counts": backend_counts,
            "total_projection_surfaces": 4,
        }

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        del attention_mask
        attn_mask = get_local_causal_mask(hidden.size(1), self.local_attention_window, hidden.device)
        attn_bias = torch.zeros(
            (1, 1, hidden.size(1), hidden.size(1)),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        attn_bias = attn_bias.masked_fill(attn_mask.view(1, 1, hidden.size(1), hidden.size(1)), float("-inf"))
        # key_padding_mask is intentionally omitted: padding positions that fall
        # beyond the local window have no valid keys when key_padding_mask is
        # applied, producing softmax([-inf,...]) = NaN that propagates through
        # the entire batch via fp8 amax().  Causal masking ensures padding
        # positions cannot corrupt valid-token gradients, so omitting the mask
        # is safe for packed-sequence training.
        q = self._shape_heads(
            run_projection(
                self.q_proj,
                hidden,
                is_first_microbatch=is_first_microbatch,
            )
        )
        k = self._shape_heads(
            run_projection(
                self.k_proj,
                hidden,
                is_first_microbatch=is_first_microbatch,
            )
        )
        v = self._shape_heads(
            run_projection(
                self.v_proj,
                hidden,
                is_first_microbatch=is_first_microbatch,
            )
        )
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(hidden.size(0), hidden.size(1), self.hidden_size)
        attn_out = run_projection(
            self.out_proj,
            attn_out,
            is_first_microbatch=is_first_microbatch,
        )
        return self.norm(hidden + attn_out)
