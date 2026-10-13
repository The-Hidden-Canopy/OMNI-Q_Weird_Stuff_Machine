"""Public low-bit weight formats for CPU devices.

The public CPU path provides:

- **MXFP8** (E4M3fn payload × UE8M0 K32) as the resident top tier of the
  compression slider — per the OCP MX specification.
- **MXFP4** (E2M1 payload × UE8M0 K32) as the 4-bit ladder reference arm —
  implemented locally against the public scale helpers.
- `quantize_tensor` / `dequantize_tensor` — one-call weight compress/restore
  with error stats, the surface the YOLO evaluation tiers consume.

Experimental or deployment-specific low-bit formats are intentionally absent
from the public API until they have an independently releasable implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .scale_codec import e4m3_pack, e4m3_unpack, encode_scale, safe_scale

FORMATS = ("mxfp8", "mxfp4")
BLOCK_SIZES = {"mxfp8": 32, "mxfp4": 32}
CODE_BITS = {"mxfp8": 8, "mxfp4": 4}

# MXFP8 (E4M3fn) payload: finite-only byte codec, max normal 448; the NaN
# codes 0x7F/0xFF are reserved for non-finite inputs. Per OCP MX spec.
E4M3_MAX = 448.0

# MXFP4 (E2M1) payload ladder, magnitudes only; sign is bit 0x08.
_MXFP4_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
_MXFP4_MAX = 6.0


# --------------------------------------------------------------------------- #
# 4-bit storage packing (two nibbles per byte).                              #
# --------------------------------------------------------------------------- #
def pack_nibbles(codes: np.ndarray) -> bytes:
    """Pack per-element 4-bit codes (uint8 0..15) two per byte, low nibble
    first (4-bit formats: mxfp4)."""
    c = np.asarray(codes, dtype=np.uint8).reshape(-1)
    pad = (-c.size) % 2
    if pad:
        c = np.concatenate([c, np.zeros(pad, dtype=np.uint8)])
    out = (c[0::2] | (c[1::2] << 4)).astype(np.uint8)
    return out.tobytes()


def unpack_nibbles(payload: bytes | np.ndarray, numel: int) -> np.ndarray:
    """Inverse of :func:`pack_nibbles`; returns the first ``numel`` codes."""
    b = np.frombuffer(payload, dtype=np.uint8).reshape(-1)
    codes = np.empty(b.size * 2, dtype=np.uint8)
    codes[0::2] = b & 0x0F
    codes[1::2] = (b >> 4) & 0x0F
    return codes[:numel]


def _pack_payload(codes: np.ndarray, fmt: str) -> bytes:
    if CODE_BITS[fmt] == 8:
        return np.asarray(codes, dtype=np.uint8).tobytes()
    if CODE_BITS[fmt] == 4:
        return pack_nibbles(codes)
    raise ValueError(f"format {fmt!r} has no public storage codec")


def _unpack_payload(payload: bytes | np.ndarray, numel: int, fmt: str) -> np.ndarray:
    if CODE_BITS[fmt] == 8:
        return np.frombuffer(payload, dtype=np.uint8)[:numel]
    if CODE_BITS[fmt] == 4:
        return unpack_nibbles(payload, numel)
    raise ValueError(f"format {fmt!r} has no public storage codec")


# --------------------------------------------------------------------------- #
# MXFP8 arm (OCP MX: E4M3fn payload on the public UE8M0 block scales)           #
# --------------------------------------------------------------------------- #
def _encode_mxfp8(values: np.ndarray, block: int = 32):
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    n = flat.size
    nblocks = (n + block - 1) // block
    pad = nblocks * block - n
    block_view = np.concatenate([flat, np.zeros(pad)]).reshape(nblocks, block)
    finite = np.isfinite(block_view)
    amax = np.max(np.where(finite, np.abs(block_view), 0.0), axis=1)
    scale_codes = encode_scale(amax / E4M3_MAX).reshape(nblocks)
    applied = safe_scale(scale_codes).reshape(nblocks, 1)
    normalized = (block_view / applied).reshape(-1)[:n]
    return e4m3_pack(normalized).astype(np.uint8), np.asarray(
        scale_codes, dtype=np.uint8)


def _decode_mxfp8(payload, scales, numel=None, block: int = 32):
    payload = np.asarray(payload, dtype=np.uint8).reshape(-1)
    scales = np.asarray(scales, dtype=np.uint8).reshape(-1)
    n = payload.size if numel is None else int(numel)
    expect = (n + block - 1) // block
    if scales.size < expect:
        raise ValueError(f"need {expect} scale bytes for numel {n}")
    values = e4m3_unpack(payload[:n])
    per_elem = safe_scale(scales[:expect]).repeat(block)[:n]
    return (values * per_elem).astype(np.float32)


# --------------------------------------------------------------------------- #
# MXFP4 ladder arm (local implementation on the public UE8M0 helpers)           #
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

    fmt: str                                  # mxfp8 | mxfp4
    shape: tuple[int, ...]
    numel: int
    payload: bytes                            # 1 byte (mxfp8) / 2 x 4-bit (mxfp4)
    scales: bytes                             # block scale bytes
    tensor_scale: float = 1.0                 # reserved for future public formats
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def bytes_total(self) -> int:
        return len(self.payload) + len(self.scales)

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
    if fmt == "mxfp8":
        codes, scales = _encode_mxfp8(flat)
        tensor_scale = 1.0
    elif fmt == "mxfp4":
        codes, scales = _encode_mxfp4(flat)
        tensor_scale = 1.0
    else:
        raise ValueError(f"format {fmt!r} is not enabled in the public build")
    restored = dequantize_tensor(PackedTensor(
        fmt=fmt, shape=arr.shape, numel=flat.size,
        payload=_pack_payload(codes, fmt), scales=scales.tobytes(),
        tensor_scale=float(tensor_scale)))
    rel, cos = _rel_cos(restored.reshape(-1), flat.astype(np.float64))
    return PackedTensor(
        fmt=fmt, shape=arr.shape, numel=flat.size,
        payload=_pack_payload(codes, fmt), scales=scales.tobytes(),
        tensor_scale=float(tensor_scale),
        stats={"weight_rel_err": rel, "weight_cosine": cos,
               "block_size": BLOCK_SIZES[fmt], "mode": mode,
               "source_dtype": str(arr.dtype)},
    )


def dequantize_tensor(packed: PackedTensor) -> np.ndarray:
    """Restore a float32 weight tensor from its low-bit master."""
    codes = _unpack_payload(packed.payload, packed.numel, packed.fmt)
    scales = np.frombuffer(packed.scales, dtype=np.uint8)
    if packed.fmt == "mxfp8":
        flat = _decode_mxfp8(codes, scales, packed.numel)
    elif packed.fmt == "mxfp4":
        flat = _decode_mxfp4(codes, scales, packed.numel)
    else:
        raise ValueError(f"unknown format {packed.fmt!r}")
    return flat.reshape(packed.shape)
