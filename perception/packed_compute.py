"""Selected-layer packed MXFP4 compute pilot for the YOLO trainer.

This is the portable part of the IDA-TRAIN-V2 Hopper recipe: retain a packed
E2M1/UE8M0 K32 operand and decode it at the consumer boundary.  The stock
YOLO convolution still performs its arithmetic through PyTorch's normal
Conv2d/autocast path; this module does not claim Hopper WGMMA or native
block-scaled MXFP4 MMA on Ada.

The trainable parameter remains the floating-point straight-through view.  A
selected convolution uses the packed FP4 decode for its forward value, while
the optimizer can continue to own the authoritative MXFP8 master and BF16
Lion momentum.  The packed payload/scales are refreshed after each optimizer
step, so the pilot is resumable and does not mutate the default path.
"""

from __future__ import annotations

import re
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

BLOCK_SIZE = 32
FP4_MAX = 6.0
_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_TINY_SCALE = 1.0e-30


def _safe_scales(codes: Tensor) -> Tensor:
    exponents = codes.to(dtype=torch.int32) - 127
    powers = torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents)
    return torch.where(
        codes == 0,
        torch.full_like(powers, _TINY_SCALE),
        powers.clamp_min(_TINY_SCALE),
    )


def _pack_nibbles(codes: Tensor) -> Tensor:
    flat = codes.reshape(-1).to(dtype=torch.uint8)
    if flat.numel() % 2:
        flat = torch.cat((flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)))
    return (flat[0::2] | (flat[1::2] << 4)).contiguous()


def _unpack_nibbles(payload: Tensor, numel: int) -> Tensor:
    flat = payload.reshape(-1).to(dtype=torch.uint8)
    codes = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=flat.device)
    codes[0::2] = flat & 0x0F
    codes[1::2] = flat >> 4
    return codes[:numel]


def pack_mxfp4(values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return packed nibbles, UE8M0 K32 scales, and the decoded view."""
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
    scale_value = amax / FP4_MAX
    positive = torch.isfinite(scale_value) & (scale_value > 0)
    log_input = torch.where(positive, scale_value, torch.ones_like(scale_value))
    exponent = torch.ceil(torch.log2(log_input)).clamp(-126, 127).to(torch.int32)
    scales = torch.where(positive, exponent + 127, torch.zeros_like(exponent)).to(torch.uint8)
    applied = _safe_scales(scales)
    normalized = padded / applied.repeat_interleave(BLOCK_SIZE)

    magnitudes = torch.as_tensor(_MAGNITUDES, dtype=torch.float32, device=source.device)
    magnitude = normalized.abs()
    insertion = torch.bucketize(magnitude, magnitudes)
    hi_index = insertion.clamp(1, len(_MAGNITUDES) - 1)
    lo_index = hi_index - 1
    lo = magnitudes.index_select(0, lo_index.reshape(-1)).reshape_as(magnitude)
    hi = magnitudes.index_select(0, hi_index.reshape(-1)).reshape_as(magnitude)
    midpoint = (lo + hi) * 0.5
    take_hi = (magnitude > midpoint) | (
        (magnitude == midpoint) & ((hi_index & 1) == 0)
    )
    code = torch.where(take_hi, hi_index, lo_index).to(torch.uint8)
    code = torch.where(magnitude == 0, torch.zeros_like(code), code)
    code = torch.where(magnitude >= FP4_MAX, torch.full_like(code, 7), code)
    code = torch.where(torch.isfinite(normalized), code, torch.zeros_like(code))
    code = code | (torch.signbit(normalized).to(torch.uint8) << 3)
    payload = _pack_nibbles(code[:n])
    decoded = decode_mxfp4(payload, scales, n, values.shape)
    return payload, scales.contiguous(), decoded


def decode_mxfp4(
    payload: Tensor,
    scales: Tensor,
    numel: int,
    shape: tuple[int, ...] | torch.Size | None = None,
) -> Tensor:
    """Decode packed MXFP4 payload/scales without materialising FP32 state."""
    codes = _unpack_nibbles(payload, int(numel))
    magnitudes = torch.as_tensor(_MAGNITUDES, dtype=torch.float32, device=payload.device)
    values = magnitudes.index_select(0, (codes & 0x07).long())
    values = torch.where((codes & 0x08) != 0, -values, values)
    block_values = _safe_scales(scales.reshape(-1)).repeat_interleave(BLOCK_SIZE)
    decoded = values * block_values[:numel]
    return decoded.reshape(tuple(shape) if shape is not None else (numel,))


class PackedFP4Conv2d(nn.Conv2d):
    """Conv2d with a packed FP4 consumer view and STE gradients."""

    precision_contract = "mxfp4_e2m1_ue8m0_k32"
    arithmetic_contract = "torch_conv_autocast"

    def __init__(self, source: nn.Conv2d) -> None:
        super().__init__(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            source.stride,
            source.padding,
            source.dilation,
            source.groups,
            source.bias is not None,
            source.padding_mode,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        with torch.no_grad():
            self.weight.copy_(source.weight)
            if self.bias is not None and source.bias is not None:
                self.bias.copy_(source.bias)
        blocks = (self.weight.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
        self.register_buffer(
            "packed_payload",
            torch.empty((self.weight.numel() + 1) // 2, dtype=torch.uint8),
        )
        self.register_buffer("packed_scales", torch.empty(blocks, dtype=torch.uint8))
        self.refresh_packed()

    @classmethod
    def from_conv(cls, source: nn.Conv2d) -> "PackedFP4Conv2d":
        return cls(source)

    @torch.no_grad()
    def refresh_packed(self) -> None:
        payload, scales, _ = pack_mxfp4(self.weight)
        self.packed_payload.copy_(payload)
        self.packed_scales.copy_(scales)

    def forward(self, input: Tensor) -> Tensor:
        decoded = decode_mxfp4(
            self.packed_payload,
            self.packed_scales,
            self.weight.numel(),
            self.weight.shape,
        ).to(dtype=self.weight.dtype)
        # Straight-through estimator: forward uses the packed decode, while
        # gradients continue to update the floating trainable parameter.
        effective_weight = self.weight + (decoded - self.weight).detach()
        return F.conv2d(
            input,
            effective_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def _replace_child(root: nn.Module, full_name: str, replacement: nn.Module) -> None:
    parent_name, child_name = full_name.rsplit(".", 1) if "." in full_name else ("", full_name)
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


def install_packed_fp4_pilot(root: nn.Module, pattern: str) -> list[str]:
    """Replace matching Conv2d modules and return their stable module names."""
    matcher = re.compile(pattern)
    selected: list[str] = []
    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Conv2d) or isinstance(module, PackedFP4Conv2d):
            continue
        if not matcher.search(name):
            continue
        _replace_child(root, name, PackedFP4Conv2d.from_conv(module))
        selected.append(name)
    if not selected:
        raise ValueError(f"packed FP4 pilot pattern matched no Conv2d modules: {pattern!r}")
    return selected


def packed_fp4_modules(root: nn.Module) -> Iterable[PackedFP4Conv2d]:
    return (module for module in root.modules() if isinstance(module, PackedFP4Conv2d))


@torch.no_grad()
def refresh_packed_fp4_modules(root: nn.Module) -> None:
    for module in packed_fp4_modules(root):
        module.refresh_packed()


__all__ = [
    "BLOCK_SIZE",
    "PackedFP4Conv2d",
    "decode_mxfp4",
    "install_packed_fp4_pilot",
    "pack_mxfp4",
    "refresh_packed_fp4_modules",
]
