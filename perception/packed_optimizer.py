"""Packed MXFP8 Lion optimizer state for the YOLO fine-tune path.

The packed payload and UE8M0 block scales are the authoritative weight state;
BF16 momentum is the only persistent floating-point optimizer state.  The
ordinary PyTorch parameter remains as a floating-point compute view because
the stock YOLO Conv2d/Linear graph and autograd require floating parameters.
It is refreshed from the packed state after every successful optimizer step.

This is therefore a packed-*optimizer-state* path, not a claim of zero
floating-point model residency.  Native fused packed operators are required
to remove the compute view as well.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer

from integrations.qualcomm.lowbit.vendor.mxfp_scales import (
    _E4M3_UNPACK_LUT,
)

BLOCK_SIZE = 32
FP8_E4M3_MAX = 448.0
TINY_SCALE = 1.0e-30
STATE_FORMAT = "mxfp8_e4m3_ue8m0_k32"

_E4M3_LUT_NP = np.asarray(_E4M3_UNPACK_LUT, dtype=np.float32)
_E4M3_GRID_NP = _E4M3_LUT_NP[:127]

_NATIVE_EXTENSION = None
_NATIVE_ATTEMPTED = False
_NATIVE_FAILURE = None
_NATIVE_STEP_USED = False


def _load_native_extension():
    """Load the optional fused CUDA step once, leaving the reference fallback intact."""
    global _NATIVE_ATTEMPTED, _NATIVE_EXTENSION, _NATIVE_FAILURE
    if _NATIVE_ATTEMPTED:
        return _NATIVE_EXTENSION
    _NATIVE_ATTEMPTED = True
    if not torch.cuda.is_available():
        _NATIVE_FAILURE = "CUDA is unavailable"
        if os.getenv("OMNIQ_REQUIRE_NATIVE_FUSION", "").lower() in {"1", "true", "yes"}:
            raise RuntimeError("native packed Lion fusion is required but CUDA is unavailable")
        return None
    try:
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parents[1]
        vendor = root / "integrations" / "qualcomm" / "lowbit" / "vendor" / "ida_train_v2"
        _NATIVE_EXTENSION = load(
            name="omniq_packed_lion_v1",
            sources=[str(Path(__file__).with_name("native_packed_optimizer.cu"))],
            extra_include_paths=[str(root), str(vendor.parent)],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        print("packed Lion native fusion loaded: omniq_packed_lion_v1")
    except Exception as exc:  # pragma: no cover - hardware/toolchain dependent
        _NATIVE_FAILURE = f"{type(exc).__name__}: {exc}"
        print(f"packed Lion native fusion unavailable; using torch fallback: {exc}")
        if os.getenv("OMNIQ_REQUIRE_NATIVE_FUSION", "").lower() in {"1", "true", "yes"}:
            raise RuntimeError(
                "native packed Lion fusion is required but failed to load"
            ) from exc
    return _NATIVE_EXTENSION


def _e4m3_lut(device: torch.device) -> Tensor:
    return torch.as_tensor(_E4M3_LUT_NP, dtype=torch.float32, device=device)


def _e4m3_grid(device: torch.device) -> Tensor:
    return torch.as_tensor(_E4M3_GRID_NP, dtype=torch.float32, device=device)


def _safe_scales(codes: Tensor) -> Tensor:
    exponents = codes.to(dtype=torch.int32) - 127
    powers = torch.ldexp(
        torch.ones(codes.shape, dtype=torch.float32, device=codes.device),
        exponents,
    )
    return torch.where(
        codes == 0,
        torch.full_like(powers, TINY_SCALE),
        powers.clamp_min(TINY_SCALE),
    )


def _encode_e4m3(normalized: Tensor) -> Tensor:
    """Encode finite/non-finite normalized values with RNE E4M3 semantics."""
    flat = normalized.reshape(-1).to(dtype=torch.float32)
    magnitude = flat.abs()
    finite = torch.isfinite(flat)
    grid = _e4m3_grid(flat.device)

    insertion = torch.bucketize(magnitude, grid)
    lo_index = (insertion - 1).clamp(0, 125)
    hi_index = insertion.clamp(1, 126)
    lo = grid.index_select(0, lo_index)
    hi = grid.index_select(0, hi_index)
    midpoint = (lo + hi) * 0.5
    take_hi = (magnitude > midpoint) | (
        (magnitude == midpoint) & ((lo_index & 1) == 1)
    )
    chosen = torch.where(take_hi, hi_index, lo_index)
    chosen = torch.where(magnitude == 0, torch.zeros_like(chosen), chosen)
    chosen = torch.where(magnitude >= FP8_E4M3_MAX, torch.full_like(chosen, 126), chosen)
    chosen = torch.where(finite, chosen, torch.full_like(chosen, 127))

    sign = torch.signbit(flat).to(dtype=torch.uint8)
    payload = chosen.to(dtype=torch.uint8) | (sign << 7)
    return payload.reshape(normalized.shape)


def decode_mxfp8(payload: Tensor, scales: Tensor, numel: int | None = None) -> Tensor:
    """Decode packed E4M3 payload plus UE8M0 K32 scales to float32."""
    n = int(payload.numel() if numel is None else numel)
    if n == 0:
        return torch.empty(0, dtype=torch.float32, device=payload.device)
    values = _e4m3_lut(payload.device).index_select(0, payload.reshape(-1).long())
    applied = _safe_scales(scales.reshape(-1))
    return values[:n] * applied.repeat_interleave(BLOCK_SIZE)[:n]


def pack_mxfp8(values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Pack a tensor and return ``(payload, scales, decoded_compute_view)``."""
    source = values.detach().to(dtype=torch.float32).reshape(-1)
    n = int(source.numel())
    if n == 0:
        empty = torch.empty(0, dtype=torch.uint8, device=source.device)
        return empty, empty.clone(), source.reshape(values.shape)

    blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded = torch.zeros(blocks * BLOCK_SIZE, dtype=torch.float32, device=source.device)
    padded[:n].copy_(source)
    block_view = padded.reshape(blocks, BLOCK_SIZE)
    finite = torch.isfinite(block_view)
    amax = torch.where(finite, block_view.abs(), torch.zeros_like(block_view)).amax(dim=1)
    scale_value = amax / FP8_E4M3_MAX
    positive = torch.isfinite(scale_value) & (scale_value > 0)
    log_input = torch.where(positive, scale_value, torch.ones_like(scale_value))
    exponent = torch.ceil(torch.log2(log_input)).clamp(-126, 127).to(torch.int32)
    scales = torch.where(
        positive,
        exponent + 127,
        torch.zeros_like(exponent),
    ).to(dtype=torch.uint8)
    applied = _safe_scales(scales)
    normalized = padded / applied.repeat_interleave(BLOCK_SIZE)
    payload = _encode_e4m3(normalized)
    decoded = decode_mxfp8(payload, scales, n).reshape(values.shape)
    return payload[:n].contiguous(), scales.contiguous(), decoded


class PackedMXFP8Lion(Optimizer):
    """Lion whose persistent weight state is MXFP8 payload/scales.

    The update matches the IDA-TRAIN-V2 packed Lion equation:

    ``direction = beta1 * momentum + (1-beta1) * grad``
    ``momentum = beta2 * momentum + (1-beta2) * grad``
    ``weight -= lr * (sign(direction) + weight_decay * weight)``

    Repacking happens inside ``step()``, so callers cannot observe a successful
    optimizer step without the packed state and compute view being updated
    together.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"invalid Lion betas: {betas}")
        if weight_decay < 0:
            raise ValueError(f"invalid weight decay: {weight_decay}")
        super().__init__(
            params,
            defaults={"lr": lr, "betas": betas, "weight_decay": weight_decay},
        )

    def _initialize_parameter(self, parameter: Tensor) -> None:
        if not parameter.is_floating_point():
            raise TypeError("PackedMXFP8Lion requires floating-point parameters")
        state = self.state[parameter]
        if "payload" in state:
            self._validate_state(parameter, state)
            return
        payload, scales, decoded = pack_mxfp8(parameter.data)
        state.update(
            {
                "format": STATE_FORMAT,
                "momentum_format": "bfloat16",
                "step": 0,
                "payload": payload,
                "scales": scales,
                "momentum": torch.zeros(
                    parameter.numel(), dtype=torch.bfloat16, device=parameter.device
                ),
            }
        )
        parameter.data.copy_(decoded)

    @staticmethod
    def _validate_state(parameter: Tensor, state: dict) -> None:
        expected_scales = (parameter.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
        payload = state.get("payload")
        scales = state.get("scales")
        momentum = state.get("momentum")
        if not isinstance(payload, Tensor) or payload.dtype != torch.uint8:
            raise ValueError("invalid MXFP8 optimizer payload state")
        if not isinstance(scales, Tensor) or scales.dtype != torch.uint8:
            raise ValueError("invalid MXFP8 optimizer scale state")
        if not isinstance(momentum, Tensor) or momentum.dtype != torch.bfloat16:
            raise ValueError("invalid BF16 Lion momentum state")
        if payload.numel() != parameter.numel() or scales.numel() != expected_scales:
            raise ValueError("MXFP8 optimizer state shape does not match parameter")
        if momentum.numel() != parameter.numel():
            raise ValueError("Lion momentum shape does not match parameter")
        if state.get("format") not in {None, STATE_FORMAT}:
            raise ValueError(f"unsupported optimizer state format: {state.get('format')!r}")

    def initialize_state(self) -> None:
        """Project initial floating compute views and allocate packed state."""
        with torch.no_grad():
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.requires_grad:
                        self._initialize_parameter(parameter)

    @torch.no_grad()
    def step(self, closure=None):
        global _NATIVE_STEP_USED
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        native_extension = (
            _load_native_extension()
            if any(parameter.is_cuda for group in self.param_groups
                   for parameter in group["params"])
            else None
        )

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            native_payloads = []
            native_scales = []
            native_momenta = []
            native_gradients = []
            native_computes = []
            native_parameters = []
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("PackedMXFP8Lion does not support sparse gradients")
                self._initialize_parameter(parameter)
                state = self.state[parameter]
                self._validate_state(parameter, state)

                gradient_view = parameter.grad.detach().reshape(-1)
                compute_view = parameter.data.reshape(-1)
                if (
                    native_extension is not None
                    and parameter.data.dtype == torch.float32
                    and gradient_view.dtype == torch.float32
                    and parameter.data.is_contiguous()
                    and parameter.grad.is_contiguous()
                    and compute_view.is_contiguous()
                    and gradient_view.is_contiguous()
                ):
                    if hasattr(native_extension, "lion_step_multi"):
                        native_payloads.append(state["payload"])
                        native_scales.append(state["scales"])
                        native_momenta.append(state["momentum"])
                        native_gradients.append(gradient_view)
                        native_computes.append(compute_view)
                        native_parameters.append(parameter)
                    else:
                        # Keep an already-loaded older extension on its native
                        # path while a long-lived process transitions to the
                        # multi-tensor entry point.
                        if not _NATIVE_STEP_USED:
                            print("packed Lion native fusion active for FP32 compute/gradient views")
                            _NATIVE_STEP_USED = True
                        native_extension.lion_step(
                            state["payload"],
                            state["scales"],
                            state["momentum"],
                            gradient_view,
                            compute_view,
                            lr,
                            beta1,
                            beta2,
                            weight_decay,
                        )
                        state["step"] = int(state["step"]) + 1
                    continue

                weight = decode_mxfp8(
                    state["payload"], state["scales"], parameter.numel()
                )
                momentum = state["momentum"].to(dtype=torch.float32)
                gradient = gradient_view.to(dtype=torch.float32)
                direction = beta1 * momentum + (1.0 - beta1) * gradient
                next_momentum = beta2 * momentum + (1.0 - beta2) * gradient
                next_weight = weight - lr * (
                    direction.sign() + weight_decay * weight
                )
                payload, scales, decoded = pack_mxfp8(next_weight)
                state["payload"] = payload
                state["scales"] = scales
                state["momentum"] = next_momentum.to(dtype=torch.bfloat16)
                state["step"] = int(state["step"]) + 1
                parameter.data.copy_(decoded.reshape_as(parameter))

            if native_payloads:
                if not _NATIVE_STEP_USED:
                    print("packed Lion native fusion active for FP32 compute/gradient views")
                    _NATIVE_STEP_USED = True
                native_extension.lion_step_multi(
                    native_payloads,
                    native_scales,
                    native_momenta,
                    native_gradients,
                    native_computes,
                    lr,
                    beta1,
                    beta2,
                    weight_decay,
                )
                for parameter in native_parameters:
                    self.state[parameter]["step"] = (
                        int(self.state[parameter]["step"]) + 1
                    )
        # The selected-layer packed-compute pilot keeps its FP4 payloads in
        # sync with the authoritative MXFP8/BF16-Lion update.  This callback
        # is intentionally optional so the default MXFP8 path is unchanged.
        for module in getattr(self, "packed_compute_modules", ()):
            module.refresh_packed()
        return loss

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict() performs the parameter-ID remap we need,
        # but its recursive tensor cast assumes ordinary floating optimizer
        # state.  That would turn the uint8 MXFP8 payload/scales and BF16
        # momentum into the parameter dtype before the packed contract can be
        # checked.  Restore the saved tensors by parameter order after the
        # generic remap, preserving their serialized dtypes.
        super().load_state_dict(state_dict)
        saved_groups = state_dict.get("param_groups", [])
        saved_state = state_dict.get("state", {})
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("packed optimizer parameter-group count changed")
        with torch.no_grad():
            for saved_group, group in zip(saved_groups, self.param_groups):
                saved_ids = saved_group.get("params", [])
                parameters = group["params"]
                if len(saved_ids) != len(parameters):
                    raise ValueError("packed optimizer parameter-group size changed")
                for saved_id, parameter in zip(saved_ids, parameters):
                    saved_entry = saved_state.get(saved_id, {})
                    current_entry = self.state[parameter]
                    for key, value in saved_entry.items():
                        current_entry[key] = (
                            value.to(device=parameter.device, non_blocking=True)
                            if isinstance(value, Tensor) else value
                        )
                    if parameter.requires_grad:
                        self._validate_state(parameter, current_entry)
                        parameter.data.copy_(
                            decode_mxfp8(
                                current_entry["payload"],
                                current_entry["scales"],
                                parameter.numel(),
                            ).reshape_as(parameter)
                        )


__all__ = ["PackedMXFP8Lion", "decode_mxfp8", "pack_mxfp8"]
