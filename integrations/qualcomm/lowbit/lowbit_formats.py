"""Low-bit weight formats for CPU devices — storage packing + tensor API.

Sits on top of the vendored 2-bit codec (`vendor/`, MXFP2 + NVINT2) and adds:

- **4×2-bit-per-byte payload packing** for on-disk / in-RAM storage (the
  vendored codec works in per-element uint8 codes; the on-disk format packs
  four codes per byte).
- **MXFP4** (E2M1 payload × UE8M0 K32) as the 4-bit ladder reference arm —
  implemented locally against the same vendored `encode_scale`/`safe_scale`
  helpers, kept format-symmetric with the 2-bit arms ("a format is a format").
- `quantize_tensor` / `dequantize_tensor` — one-call weight compress/restore
  with error stats, the surface the YOLO evaluation tiers consume.

Attribution: the scale helpers and 2-bit ladders are vendored from The
Hidden Canopy LLC's IDA-TRAIN-V2 (see vendor/SOURCE.md); the packing, MXFP4
arm, and tensor API are OMNI-Q's own layer on top.

Portions derived from *The Hidden Canopy LLC* —
[`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2).
Used with permission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .vendor.mxfp2_codec import (
    BLOCK_SIZE_MXFP2,
    BLOCK_SIZE_NVINT2,
    decode_tensor as decode_tensor_2bit,
    encode_tensor as encode_tensor_2bit,
)
from .vendor.mxfp_scales import encode_scale, safe_scale

FORMATS = ("mxfp4", "nvint2", "mxfp2")
BLOCK_SIZES = {"mxfp4": 32, "nvint2": BLOCK_SIZE_NVINT2, "mxfp2": BLOCK_SIZE_MXFP2}

# MXFP4 (E2M1) payload ladder, magnitudes only; sign is bit 0x08.
_MXFP4_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
_MXFP4_MAX = 6.0


# --------------------------------------------------------------------------- #
# 4-codes-per-byte packing (OMNI-Q storage layer; codecs stay per-element)      #
# --------------------------------------------------------------------------- #
def pack_codes(codes: np.ndarray) -> bytes:
    """Pack per-element 2-bit codes (uint8 0..3) four per byte, little-endian
    within the byte (first element in bits 0-1)."""
    c = np.asarray(codes, dtype=np.uint8).reshape(-1)
    pad = (-c.size) % 4
    if pad:
        c = np.concatenate([c, np.zeros(pad, dtype=np.uint8)])
    out = (
        c[0::4] | (c[1::4] << 2) | (c[2::4] << 4) | (c[3::4] << 6)
    ).astype(np.uint8)
    return out.tobytes()


def unpack_codes(payload: bytes | np.ndarray, numel: int) -> np.ndarray:
    """Inverse of :func:`pack_codes`; returns the first ``numel`` codes."""
    b = np.frombuffer(payload, dtype=np.uint8).reshape(-1)
    codes = np.empty(b.size * 4, dtype=np.uint8)
    codes[0::4] = b & 0x03
    codes[1::4] = (b >> 2) & 0x03
    codes[2::4] = (b >> 4) & 0x03
    codes[3::4] = (b >> 6) & 0x03
    return codes[:numel]


# --------------------------------------------------------------------------- #
# MXFP4 ladder arm (local implementation on the vendored UE8M0 helpers)         #
# --------------------------------------------------------------------------- #
def _mxfp4_pack_rne(normalized: np.ndarray) -> np.ndarray:
    v = np.asarray(normalized, dtype=np.float64)
    sign = np.signbit(v)
    mag = np.clip(np.abs(v), 0.0, _MXFP4_MAX)
    idx = np.searchsorted(_MXFP4_MAGNITUDES, mag, side="left")
    idx = np.clip(idx, 1, len(_MXFP4_MAGNITUDES) - 1)
    lo = _MXFP4_MAGNITUDES[idx - 1]
    hi = _MXFP4_MAGNITUDES[idx]
    # tie to even code, matching e2m1_pack_rne
    take_hi = (mag > 0.5 * (lo + hi)) | (
        (mag == 0.5 * (lo + hi)) & ((idx & 1) == 0)
    )
    chosen = np.where(take_hi, idx, idx - 1).astype(np.uint8)
    chosen[np.isnan(v)] = 0
    return chosen | (sign.astype(np.uint8) << 3)


def _mxfp4_unpack(codes: np.ndarray) -> np.ndarray:
    c = np.asarray(codes, dtype=np.uint8)
    mag = _MXFP4_MAGNITUDES[c & 0x07]
    return np.where((c & 0x08) != 0, -mag, mag).astype(np.float32)


def _encode_mxfp4(values: np.ndarray, block: int = 32):
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    n = flat.size
    nblocks = (n + block - 1) // block
    pad = nblocks * block - n
    block_view = np.concatenate([flat, np.zeros(pad)]).reshape(nblocks, block)
    finite = np.isfinite(block_view)
    amax = np.max(np.where(finite, np.abs(block_view), 0.0), axis=1)
    scale_codes = encode_scale(amax / _MXFP4_MAX).reshape(nblocks)
    applied = safe_scale(scale_codes).reshape(nblocks, 1)
    normalized = (block_view / applied).reshape(-1)[:n]
    return _mxfp4_pack_rne(normalized).astype(np.uint8), np.asarray(
        scale_codes, dtype=np.uint8)


def _decode_mxfp4(payload, scales, numel=None, block: int = 32):
    payload = np.asarray(payload, dtype=np.uint8).reshape(-1)
    scales = np.asarray(scales, dtype=np.uint8).reshape(-1)
    n = payload.size if numel is None else int(numel)
    expect = (n + block - 1) // block
    if scales.size < expect:
        raise ValueError(f"need {expect} scale bytes for numel {n}")
    values = _mxfp4_unpack(payload[:n])
    per_elem = safe_scale(scales[:expect]).repeat(block)[:n]
    return (values * per_elem).astype(np.float32)


# --------------------------------------------------------------------------- #
# Tensor API                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PackedTensor:
    """A weight tensor compressed to a low-bit master."""

    fmt: str                                  # mxfp4 | nvint2 | mxfp2
    shape: tuple[int, ...]
    numel: int
    payload: bytes                            # 4 x 2-bit codes per byte
    scales: bytes                             # block scale bytes
    tensor_scale: float = 1.0                 # nvint2 only
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def bytes_total(self) -> int:
        return len(self.payload) + len(self.scales) + (4 if self.fmt == "nvint2" else 0)

    @property
    def bits_per_weight(self) -> float:
        return 8.0 * self.bytes_total / max(self.numel, 1)


def _rel_cos(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(b, dtype=np.float64)
    approx = np.asarray(a, dtype=np.float64)
    denom = np.linalg.norm(ref)
    cos = float(np.dot(approx, ref) / (np.linalg.norm(approx) * denom)) if denom else 1.0
    rel = float(np.linalg.norm(approx - ref) / denom) if denom else 0.0
    return rel, cos


def quantize_tensor(values: np.ndarray, fmt: str, *, mode: str = "rne") -> PackedTensor:
    """Compress a weight tensor to a low-bit master; returns stats vs original."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r} (want one of {FORMATS})")
    arr = np.ascontiguousarray(values, dtype=np.float32)
    flat = arr.reshape(-1)
    if mode != "rne":
        raise ValueError("storage path supports rne only (sr is training-side)")
    if fmt == "mxfp4":
        codes, scales = _encode_mxfp4(flat)
        tensor_scale = 1.0
    elif fmt == "mxfp2":
        codes, scales = encode_tensor_2bit(flat, fmt="mxfp2", mode="rne")
        tensor_scale = 1.0
    else:
        codes, scales, tensor_scale = encode_tensor_2bit(flat, fmt="nvint2", mode="rne")
    restored = dequantize_tensor(PackedTensor(
        fmt=fmt, shape=arr.shape, numel=flat.size,
        payload=pack_codes(codes), scales=scales.tobytes(),
        tensor_scale=float(tensor_scale)))
    rel, cos = _rel_cos(restored.reshape(-1), flat.astype(np.float64))
    return PackedTensor(
        fmt=fmt, shape=arr.shape, numel=flat.size,
        payload=pack_codes(codes), scales=scales.tobytes(),
        tensor_scale=float(tensor_scale),
        stats={"weight_rel_err": rel, "weight_cosine": cos,
               "block_size": BLOCK_SIZES[fmt], "mode": mode,
               "source_dtype": str(arr.dtype)},
    )


def dequantize_tensor(packed: PackedTensor) -> np.ndarray:
    """Restore a float32 weight tensor from its low-bit master."""
    codes = unpack_codes(packed.payload, packed.numel)
    scales = np.frombuffer(packed.scales, dtype=np.uint8)
    if packed.fmt == "mxfp4":
        flat = _decode_mxfp4(codes, scales, packed.numel)
    elif packed.fmt in ("mxfp2", "nvint2"):
        flat = decode_tensor_2bit(codes, scales, packed.numel,
                                  fmt=packed.fmt,
                                  tensor_scale=packed.tensor_scale)
    else:
        raise ValueError(f"unknown format {packed.fmt!r}")
    return flat.reshape(packed.shape)
