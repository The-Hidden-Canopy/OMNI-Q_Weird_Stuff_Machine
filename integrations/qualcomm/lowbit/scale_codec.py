"""Public MX floating-point scale and E4M3 codecs.

These standalone numerical utilities implement the public finite E4M3 byte
grid and UE8M0 block-scale rule. They do not load model code, checkpoints, or
deployment-specific kernels.
"""

from __future__ import annotations

import numpy as np

FP8_E4M3_MAX = 448.0
_TINY_SCALE = 1.0e-30


def _build_e4m3_unpack_lut() -> np.ndarray:
    lut = np.empty(256, dtype=np.float32)
    for bits in range(256):
        sign = -1.0 if bits & 0x80 else 1.0
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
    """Decode finite E4M3 bytes to float32 values."""
    return _E4M3_UNPACK_LUT[np.asarray(payload, dtype=np.uint8)]


def e4m3_pack(values: np.ndarray) -> np.ndarray:
    """Round values to the nearest finite E4M3 value, ties to even."""
    grid = sorted(
        (abs(float(_E4M3_UNPACK_LUT[bits])), bits)
        for bits in range(0x7F)
    )
    magnitudes = np.asarray([item[0] for item in grid], dtype=np.float64)
    codes = np.asarray([item[1] for item in grid], dtype=np.uint8)
    value = np.asarray(values, dtype=np.float64)
    magnitude = np.abs(value)
    output = np.zeros(value.shape, dtype=np.uint8)
    finite = np.isfinite(value)
    over = finite & (magnitude >= FP8_E4M3_MAX)
    normal = finite & ~over
    index = np.clip(np.searchsorted(magnitudes, magnitude), 1, len(magnitudes) - 1)
    lower = magnitudes[index - 1]
    upper = magnitudes[index]
    midpoint = 0.5 * (lower + upper)
    choose_upper = (magnitude > midpoint) | (
        (magnitude == midpoint) & ((codes[index - 1] & 1) == 1)
    )
    chosen = np.where(choose_upper, index, index - 1)
    output[normal] = codes[chosen[normal]]
    output[over] = 0x7E
    output[~finite] = 0x7F
    output |= np.signbit(value).astype(np.uint8) << 7
    output[magnitude == 0.0] &= np.uint8(0x80)
    return output


def encode_scale(value: float | np.ndarray) -> np.ndarray:
    """Encode positive scales as UE8M0 exponent bytes."""
    array = np.asarray(value, dtype=np.float64)
    output = np.zeros(array.shape, dtype=np.uint8)
    valid = np.isfinite(array) & (array > 0.0)
    exponent = np.ceil(np.log2(np.where(valid, array, 1.0)))
    exponent = np.clip(exponent, -126, 127) + 127
    output[valid] = exponent[valid].astype(np.uint8)
    return output


def safe_scale(code: np.ndarray | int) -> np.ndarray:
    """Decode UE8M0 bytes, retaining a finite floor for zero codes."""
    value = np.asarray(code, dtype=np.uint8)
    return np.maximum(_TINY_SCALE, np.exp2(value.astype(np.float64) - 127.0))


__all__ = ["FP8_E4M3_MAX", "e4m3_pack", "e4m3_unpack", "encode_scale", "safe_scale"]
