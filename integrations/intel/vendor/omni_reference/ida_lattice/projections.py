# Portions derived from *The Hidden Canopy LLC* —
# [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
# Vendored from `src/ida_train/models/ida_lattice/projections.py`; body unmodified except as recorded in ../SOURCE.md.

from __future__ import annotations

import importlib

import torch
from torch import nn

from .fp8_linear import Fp8Linear

FP8_BACKEND_OFF = "off"
FP8_BACKEND_NATIVE = "native_scaled_mm"
FP8_BACKEND_TRANSFORMER_ENGINE = "transformer_engine"


class TransformerEngineLinearAdapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        params_dtype: torch.dtype,
        fp8_recipe: str,
        fp8_amax_history_len: int,
        fp8_amax_compute_algo: str,
        fp8_weight_cache: bool,
    ) -> None:
        super().__init__()
        self._te = importlib.import_module("transformer_engine.pytorch")
        recipe_mod = importlib.import_module("transformer_engine.common.recipe")
        format_name = "HYBRID" if "hybrid" in str(fp8_recipe).lower() else "E4M3"
        fp8_format = getattr(recipe_mod.Format, format_name)
        self._recipe = recipe_mod.DelayedScaling(
            fp8_format=fp8_format,
            amax_history_len=max(1, int(fp8_amax_history_len)),
            amax_compute_algo=str(fp8_amax_compute_algo or "max").strip().lower() or "max",
        )
        self._weight_cache_enabled = bool(fp8_weight_cache)
        self._last_is_first_microbatch: bool | None = None
        self.linear = self._te.Linear(
            in_features,
            out_features,
            bias=bias,
            params_dtype=params_dtype,
        )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        # Set by build_projection() via _annotate_backend() right after
        # construction; mutated in forward() below if TE turns out to be
        # unusable at actual call time (see the comment there).
        self._fp8_backend = FP8_BACKEND_TRANSFORMER_ENGINE

    @property
    def weight(self) -> torch.Tensor | nn.Parameter:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | nn.Parameter | None:
        return getattr(self.linear, "bias", None)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        call_kwargs: dict[str, object] = {}
        if self._weight_cache_enabled and is_first_microbatch is not None:
            call_kwargs["is_first_microbatch"] = bool(is_first_microbatch)
        self._last_is_first_microbatch = (
            bool(is_first_microbatch) if is_first_microbatch is not None else None
        )
        try:
            with self._te.autocast(enabled=self.training, recipe=self._recipe):
                return self.linear(hidden, **call_kwargs)
        except RuntimeError as exc:
            if "needs cuda" not in str(exc).lower():
                raise
            # TE requires its parameters (and inputs) to actually live on a
            # CUDA device, which it only validates lazily at forward time --
            # the constructor above happily builds on CPU. Production always
            # constructs the model, THEN calls model.to(device) before any
            # real forward runs (see scripts/canary_compile.py), so this
            # branch is not reachable there. It exists for CPU-only callers
            # (unit tests, structural-only tooling) that never move the
            # module to CUDA: fall back to a plain linear compute instead of
            # crashing, and record the downgrade so telemetry (which reads
            # _fp8_backend, see local_workspace.py/student_experts.py
            # precision_runtime_summary()) reports what actually ran.
            self._fp8_backend = FP8_BACKEND_NATIVE
            # TE's own Linear places its weight on the current CUDA device
            # by default (unlike plain nn.Linear, which defaults to CPU) --
            # match it to whatever device 'hidden' is actually on rather
            # than assuming either side.
            weight = self.linear.weight.to(device=hidden.device, dtype=hidden.dtype)
            bias = self.bias
            if bias is not None:
                bias = bias.to(device=hidden.device, dtype=hidden.dtype)
            return torch.nn.functional.linear(hidden, weight, bias)


def _scope_enabled(scope: list[str] | tuple[str, ...] | None, surface: str) -> bool:
    enabled = {str(item).strip().lower() for item in (scope or []) if str(item).strip()}
    return str(surface).strip().lower() in enabled


def _annotate_backend(module: nn.Module, *, backend: str, module_name: str | None) -> nn.Module:
    setattr(module, "_fp8_backend", backend)
    if module_name:
        setattr(module, "_fp8_module_name", module_name)
    return module


def _build_transformer_engine_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    params_dtype: torch.dtype,
    fp8_recipe: str,
    fp8_amax_history_len: int,
    fp8_amax_compute_algo: str,
    fp8_weight_cache: bool,
    module_name: str | None,
) -> nn.Module:
    linear = TransformerEngineLinearAdapter(
        in_features,
        out_features,
        bias=bias,
        params_dtype=params_dtype,
        fp8_recipe=fp8_recipe,
        fp8_amax_history_len=fp8_amax_history_len,
        fp8_amax_compute_algo=fp8_amax_compute_algo,
        fp8_weight_cache=fp8_weight_cache,
    )
    return _annotate_backend(
        linear,
        backend=FP8_BACKEND_TRANSFORMER_ENGINE,
        module_name=module_name,
    )


def build_projection(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    fp8_enabled: bool = False,
    fp8_backend: str = FP8_BACKEND_OFF,
    fp8_scope: list[str] | tuple[str, ...] | None = None,
    fp8_fallback: str = FP8_BACKEND_NATIVE,
    fp8_recipe: str = "delayed_hybrid",
    fp8_amax_history_len: int = 16,
    fp8_amax_compute_algo: str = "max",
    fp8_weight_cache: bool = False,
    surface: str = "sensitive",
    module_name: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    backend = str(fp8_backend or FP8_BACKEND_OFF).strip().lower() or FP8_BACKEND_OFF
    fallback = str(fp8_fallback or FP8_BACKEND_NATIVE).strip().lower() or FP8_BACKEND_NATIVE
    linear_kwargs: dict[str, object] = {"bias": bias}
    if device is not None:
        linear_kwargs["device"] = device
    if dtype is not None:
        linear_kwargs["dtype"] = dtype
    if not fp8_enabled or backend == FP8_BACKEND_OFF or not _scope_enabled(fp8_scope, surface):
        return _annotate_backend(
            nn.Linear(in_features, out_features, **linear_kwargs),
            backend=FP8_BACKEND_OFF,
            module_name=module_name,
        )

    if backend == FP8_BACKEND_NATIVE:
        return _annotate_backend(
            Fp8Linear(
                in_features,
                out_features,
                **linear_kwargs,
            ),
            backend=FP8_BACKEND_NATIVE,
            module_name=module_name,
        )

    if backend == FP8_BACKEND_TRANSFORMER_ENGINE:
        try:
            return _build_transformer_engine_linear(
                in_features,
                out_features,
                bias=bias,
                params_dtype=dtype or torch.bfloat16,
                fp8_recipe=fp8_recipe,
                fp8_amax_history_len=fp8_amax_history_len,
                fp8_amax_compute_algo=fp8_amax_compute_algo,
                fp8_weight_cache=fp8_weight_cache,
                module_name=module_name,
            )
        except Exception:
            if fallback == FP8_BACKEND_NATIVE:
                return _annotate_backend(
                    Fp8Linear(
                        in_features,
                        out_features,
                        **linear_kwargs,
                    ),
                    backend=FP8_BACKEND_NATIVE,
                    module_name=module_name,
                )
            return _annotate_backend(
                nn.Linear(in_features, out_features, **linear_kwargs),
                backend=FP8_BACKEND_OFF,
                module_name=module_name,
            )

    return _annotate_backend(
        nn.Linear(in_features, out_features, **linear_kwargs),
        backend=FP8_BACKEND_OFF,
        module_name=module_name,
    )


def run_projection(
    module: nn.Module,
    hidden: torch.Tensor,
    *,
    is_first_microbatch: bool | None = None,
) -> torch.Tensor:
    if getattr(module, "_fp8_backend", None) == FP8_BACKEND_TRANSFORMER_ENGINE:
        return module(hidden, is_first_microbatch=is_first_microbatch)
    return module(hidden)
