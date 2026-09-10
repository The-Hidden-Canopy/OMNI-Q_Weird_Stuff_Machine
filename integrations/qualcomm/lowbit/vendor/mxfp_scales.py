"""UE8M0 block scales + E4M3 byte codec — helpers shared by the vendored
2-bit codec (`mxfp2_codec.py`) and the local MXFP4 ladder arm.

Vendored numerical twin of the helpers in
`IDA-TRAIN-V2/src/ida_train/native/mxfp8_codec.py`
(`FP8_E4M3_MAX`, `e4m3_pack`, `e4m3_unpack`, `encode_scale`, `safe_scale`),
inlined here so this vendor directory is self-contained (no cross-repo
imports). See SOURCE.md for provenance.

Portions derived from *The Hidden Canopy LLC* —
[`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2).
Used with permission.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FP8_E4M3_MAX",
    "e4m3_pack",
    "e4m3_unpack",
    "encode_scale",
    "safe_scale",
]

FP8_E4M3_MAX = 448.0
_TINY_SCALE = 1.0e-30


# --------------------------------------------------------------------------- #
# E4M3 payload  (finite NVIDIA encoding: exp 0 subnormal, 0x7f = NaN, max 448) #
# --------------------------------------------------------------------------- #
def _build_e4m3_unpack_lut() -> np.ndarray:
    """256-entry byte -> float32 table, matching fp8_e4m3::unpack exactly."""
    lut = np.empty(256, dtype=np.float32)
    for bits in range(256):
        sign = -1.0 if (bits & 0x80) else 1.0
        exponent = (bits >> 3) & 0x0F
        mantissa = bits & 0x07
        if exponent == 0:
            lut[bits] = sign * (mantissa * (2.0 ** -9))
        elif exponent == 15 and mantissa == 7:
            lut[bits] = np.nan
        elif exponent == 15:
            lut[bits] = sign * (1.0 + mantissa / 8.0) * 256.0
        else:
            lut[bits] = sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
    return lut


_E4M3_UNPACK_LUT = _build_e4m3_unpack_lut()


def e4m3_unpack(payload: np.ndarray) -> np.ndarray:
    """Decode E4M3 bytes (uint8) to float32 via the exact-match LUT."""
    return _E4M3_UNPACK_LUT[np.asarray(payload, dtype=np.uint8)]


def _e4m3_pack_via_lut(values: np.ndarray) -> np.ndarray:
    """Vectorised E4M3 pack: snap each float to the nearest LUT value under
    round-nearest-even, reproducing fp8_e4m3::pack's monotone rounding."""
    # Ordered list of (representable magnitude, byte) for the positive
    # finite E4M3 grid, plus the midpoints where RNE flips.
    mags = []
    for b in range(0x7F):  # 0x00..0x7E, positive, finite
        v = float(_E4M3_UNPACK_LUT[b])
        mags.append((abs(v), b))
    mags.sort()
    grid_mag = np.array([m for m, _ in mags], dtype=np.float64)
    grid_byte = np.array([b for _, b in mags], dtype=np.uint8)

    v = np.asarray(values, dtype=np.float64)
    out = np.zeros(v.shape, dtype=np.uint8)
    sign = np.signbit(v)
    mag = np.abs(v)

    nan_inf = ~np.isfinite(v)
    over = mag >= 448.0
    normal = ~nan_inf & ~over

    idx = np.searchsorted(grid_mag, mag)
    idx = np.clip(idx, 1, len(grid_mag) - 1)
    lo = grid_mag[idx - 1]
    hi = grid_mag[idx]
    # round-nearest-even at the exact midpoint
    mid = 0.5 * (lo + hi)
    take_hi = (mag > mid) | ((mag == mid) & ((grid_byte[idx - 1] & 1) == 1))
    chosen = np.where(take_hi, idx, idx - 1)
    out[normal] = grid_byte[chosen[normal]]
    out[over] = 0x7E
    out[nan_inf] = 0x7F
    out |= (sign.astype(np.uint8) << 7)
    # exact zero stays 0x00 / 0x80
    out[(mag == 0.0)] &= np.uint8(0x80)
    return out


def e4m3_pack(values: np.ndarray) -> np.ndarray:
    """Encode float32 -> E4M3 bytes (vectorised LUT snap)."""
    return _e4m3_pack_via_lut(np.asarray(values, dtype=np.float64))


# --------------------------------------------------------------------------- #
# UE8M0 block scale                                                           #
# --------------------------------------------------------------------------- #
def encode_scale(value: float | np.ndarray) -> np.ndarray:
    """mxfp8_encode_scale: 0 for non-positive/non-finite, else
    clamp(ceil(log2(value)), -126, 127) + 127 as a uint8."""
    v = np.asarray(value, dtype=np.float64)
    out = np.zeros(v.shape, dtype=np.uint8)
    ok = np.isfinite(v) & (v > 0.0)
    exp = np.ceil(np.log2(np.where(ok, v, 1.0)))
    exp = np.clip(exp, -126, 127) + 127
    out[ok] = exp[ok].astype(np.uint8)
    return out if out.shape else np.uint8(out)


def safe_scale(code: np.ndarray) -> np.ndarray:
    """mxfp8_safe_scale: code 0 -> 1e-30, else max(2^(code-127), 1e-30)."""
    c = np.asarray(code, dtype=np.int32)
    val = np.where(c == 0, _TINY_SCALE, np.ldexp(1.0, c - 127))
    return np.maximum(val, _TINY_SCALE).astype(np.float32)
