# Portions derived from *The Hidden Canopy LLC* —
# [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
# Vendored from `src/ida_train/models/ida_lattice/tensor_contracts.py`; body unmodified except as recorded in ../SOURCE.md.

from __future__ import annotations

import importlib

import torch

TENSOR_CONTRACT_BACKEND_EINSUM = "einsum"
TENSOR_CONTRACT_BACKEND_MATMUL = "matmul"
TENSOR_CONTRACT_BACKEND_CUTENSOR = "cutensor"


def _contract_einsum(
    per_scale_attn: torch.Tensor,
    anchor_bank: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("jba,ah->jbh", per_scale_attn, anchor_bank)


def _contract_matmul(
    per_scale_attn: torch.Tensor,
    anchor_bank: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(per_scale_attn, anchor_bank)


def _contract_cutensor(
    per_scale_attn: torch.Tensor,
    anchor_bank: torch.Tensor,
) -> torch.Tensor:
    cutensor_torch = importlib.import_module("cutensor.torch")
    cutensor_einsum = getattr(cutensor_torch, "einsum", None)
    if cutensor_einsum is None:
        raise RuntimeError("cutensor.torch.einsum is unavailable")
    return cutensor_einsum("jba,ah->jbh", per_scale_attn, anchor_bank)


def contract_anchor_bank(
    per_scale_attn: torch.Tensor,
    anchor_bank: torch.Tensor,
    *,
    backend: str = TENSOR_CONTRACT_BACKEND_EINSUM,
    fallback: str = TENSOR_CONTRACT_BACKEND_MATMUL,
) -> tuple[torch.Tensor, str]:
    requested = (
        str(backend or TENSOR_CONTRACT_BACKEND_EINSUM).strip().lower()
        or TENSOR_CONTRACT_BACKEND_EINSUM
    )
    fallback_backend = (
        str(fallback or TENSOR_CONTRACT_BACKEND_MATMUL).strip().lower()
        or TENSOR_CONTRACT_BACKEND_MATMUL
    )

    implementations = {
        TENSOR_CONTRACT_BACKEND_EINSUM: _contract_einsum,
        TENSOR_CONTRACT_BACKEND_MATMUL: _contract_matmul,
        TENSOR_CONTRACT_BACKEND_CUTENSOR: _contract_cutensor,
    }

    tried: list[str] = []
    for candidate in (requested, fallback_backend):
        if candidate in tried:
            continue
        tried.append(candidate)
        impl = implementations.get(candidate)
        if impl is None:
            continue
        try:
            return impl(per_scale_attn, anchor_bank), candidate
        except Exception:
            continue

    return _contract_einsum(per_scale_attn, anchor_bank), TENSOR_CONTRACT_BACKEND_EINSUM
