"""Python twins of two 2-bit master-weight codecs: MXFP2 and NVINT2.

Vendored numerical twin of
`IDA-TRAIN-V2/src/ida_train/native/mxfp2_codec.py` (numpy only), with the
only change being the import of the shared scale/E4M3 helpers from the local
`mxfp_scales.py` instead of `ida_train.native.mxfp8_codec`. Behavior is
byte-for-byte identical; an optional parity test diffs this module against
the source repo when `IDA_TRAIN_V2_ROOT` is set. See SOURCE.md.

* **MXFP2** -- ``MXFP2_INT2_UE8M0_K32``: a ternary ``{-1, 0, +1}`` payload
  (1 sign bit + 1 magnitude bit; the 4th code is ``-0.0``) times one
  **UE8M0** power-of-two scale byte per 32-element block. This is the OCP-MX
  scale shape — the one AMD CDNA4 microscaling and Blackwell MXFP* consume
  natively.

* **NVINT2** -- ``NVINT2_INT2_E4M3_K16``: a symmetric 4-level uniform INT2
  payload ``{-1.5, -0.5, +0.5, +1.5}`` (raw 2-bit code minus 1.5; no zero)
  times one **E4M3** scale byte per 16-element block, plus one **FP32
  per-tensor** second-level scale. This is the NVFP4-shaped two-level scale.

NUMERICAL twin, not a storage-layout twin: the payload is one ``uint8`` code
(0..3) per element. The 4-elements-per-byte 2-bit packing the on-disk format
implies lives in ``lowbit_formats.py`` (this repo's storage layer).

Two payload pack modes each:
* ``rne`` -- deterministic nearest, explicit tie-to-even-code rule.
* ``sr``  -- bracketing-pair stochastic rounding with a caller-supplied
  seeded numpy ``Generator``.

Portions derived from *The Hidden Canopy LLC* —
[`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2).
Used with permission.
"""

from __future__ import annotations

import numpy as np

from .mxfp_scales import (
    FP8_E4M3_MAX,
    e4m3_pack,
    e4m3_unpack,
    encode_scale,
    safe_scale,
)

__all__ = [
    "MXFP2_WEIGHTS_DTYPE",
    "NVINT2_WEIGHTS_DTYPE",
    "BLOCK_SIZE_MXFP2",
    "BLOCK_SIZE_NVINT2",
    "TERNARY_MAX",
    "INT2_MAX",
    "TERNARY_MAGNITUDES",
    "INT2_LEVELS",
    "E4M3_SCALE_MIN",
    "ternary_unpack",
    "ternary_pack_rne",
    "ternary_pack_sr",
    "int2_unpack",
    "int2_pack_rne",
    "int2_pack_sr",
    "encode_scale",
    "safe_scale",
    "decode_tensor",
    "encode_tensor",
]

MXFP2_WEIGHTS_DTYPE = "MXFP2_INT2_UE8M0_K32"
NVINT2_WEIGHTS_DTYPE = "NVINT2_INT2_E4M3_K16"
BLOCK_SIZE_MXFP2 = 32
BLOCK_SIZE_NVINT2 = 16

# MXFP2 ternary payload. code & 0x01 -> magnitude index; code & 0x02 -> sign.
TERNARY_MAGNITUDES = np.array([0.0, 1.0], dtype=np.float64)
TERNARY_MAX = 1.0

# NVINT2 symmetric 4-level payload. Signed value IS the code table entry
# (raw code 0..3 -> level); uniform spacing 1.0, no zero rung.
INT2_LEVELS = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
INT2_MAX = 1.5

# Smallest positive E4M3 value (min subnormal, 2^-9). NVINT2 factors a per-
# tensor FP32 scale so block E4M3 scales stay well above this; the clamp is
# the belt-and-braces floor for a tensor whose block dynamic range exceeds
# 448 / 2^-9.
E4M3_SCALE_MIN = 2.0**-9
_TINY = 1.0e-30


# --------------------------------------------------------------------------- #
# MXFP2 ternary payload                                                        #
# --------------------------------------------------------------------------- #
def ternary_unpack(codes: np.ndarray) -> np.ndarray:
    """Decode per-element ternary codes (uint8, 0..3) to float32.

    ``code & 1`` indexes ``TERNARY_MAGNITUDES``; ``code & 2`` is the sign, so
    code ``0x2`` decodes to ``-0.0`` (signed-zero evidence preserved)."""
    c = np.asarray(codes, dtype=np.uint8)
    mag = TERNARY_MAGNITUDES[(c & 0x01)]
    return np.where((c & 0x02) != 0, -mag, mag).astype(np.float32)


def ternary_pack_rne(values: np.ndarray) -> np.ndarray:
    """Encode float -> ternary codes (uint8, 0..3), round to nearest.

    ``|v|`` is clamped to 1.0 then snapped onto ``{0, 1}``: the tie at exactly
    0.5 prefers the EVEN code (magnitude 0). NaN encodes as ``+0.0`` (code 0)."""
    v = np.asarray(values, dtype=np.float64)
    sign = np.signbit(v)
    mag = np.clip(np.abs(v), 0.0, TERNARY_MAX)
    bit = (mag > 0.5).astype(np.uint8)  # strict >: 0.5 -> even code 0
    out = bit | (sign.astype(np.uint8) << 1)
    out[np.isnan(v)] = 0
    return out


def ternary_pack_sr(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stochastic-nearest ternary pack. Between the rungs ``0 < |v| < 1`` the
    magnitude bit is set with probability ``|v|`` (``span == 1``), making the
    expectation exact. ``|v| >= 1`` saturates to magnitude 1; non-finite
    encodes as ``+0.0``."""
    v = np.asarray(values, dtype=np.float64)
    sign = np.signbit(v)
    mag = np.clip(np.abs(v), 0.0, TERNARY_MAX)  # +-inf -> magnitude 1, sign kept
    draw = rng.uniform(size=mag.shape)
    bit = (draw < mag).astype(np.uint8)
    out = bit | (sign.astype(np.uint8) << 1)
    out[np.isnan(v)] = 0  # NaN only; inf saturates above
    return out


# --------------------------------------------------------------------------- #
# NVINT2 symmetric 4-level payload                                             #
# --------------------------------------------------------------------------- #
def int2_unpack(codes: np.ndarray) -> np.ndarray:
    """Decode per-element INT2 codes (uint8, 0..3) to float32 via
    ``INT2_LEVELS``."""
    return INT2_LEVELS[np.asarray(codes, dtype=np.uint8) & 0x03].astype(np.float32)


def _int2_bracket(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clamp to the ladder span and return (lo_index, lo_value) for the
    largest level at or below ``v``. ``lo_index`` in ``[0, 2]``."""
    vc = np.clip(v, INT2_LEVELS[0], INT2_LEVELS[-1])
    lo_i = np.clip(np.searchsorted(INT2_LEVELS, vc, side="right") - 1, 0, 2)
    return lo_i, INT2_LEVELS[lo_i]


def int2_pack_rne(values: np.ndarray) -> np.ndarray:
    """Encode float -> INT2 codes (uint8, 0..3), round to nearest with the
    tie going to the EVEN code. Values outside ``[-1.5, +1.5]`` saturate.
    NaN encodes to code 2 (``+0.5``, the level nearest zero)."""
    v = np.asarray(values, dtype=np.float64)
    lo_i, lo = _int2_bracket(v)
    hi = INT2_LEVELS[lo_i + 1]
    mid = 0.5 * (lo + hi)
    vc = np.clip(v, INT2_LEVELS[0], INT2_LEVELS[-1])
    take_hi = (vc > mid) | ((vc == mid) & (((lo_i + 1) & 1) == 0))
    chosen = np.where(take_hi, lo_i + 1, lo_i).astype(np.uint8)
    chosen[np.isnan(v)] = 2
    return chosen


def int2_pack_sr(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stochastic-nearest INT2 pack. The ladder is uniform (span 1.0), so
    ``frac`` is ``v - lo``; round to the upper level with probability
    ``frac``. ``+-inf`` saturates to ``+-1.5`` (matching the RNE path via the
    bracket clamp); NaN encodes to code 2 (``+0.5``, the level nearest zero)."""
    v = np.asarray(values, dtype=np.float64)
    lo_i, lo = _int2_bracket(v)  # clamps to [-1.5, +1.5]; +-inf -> the end rungs
    frac = np.clip(v - lo, 0.0, 1.0)
    draw = rng.uniform(size=v.shape)
    chosen = (lo_i + (draw < frac).astype(np.int64)).astype(np.uint8)
    chosen[v >= INT2_LEVELS[-1]] = 3
    chosen[v <= INT2_LEVELS[0]] = 0
    chosen[np.isnan(v)] = 2
    return chosen


# --------------------------------------------------------------------------- #
# Tensor-level decode / encode                                                 #
# --------------------------------------------------------------------------- #
def _fmt_params(fmt: str) -> tuple[int, float]:
    if fmt == "mxfp2":
        return BLOCK_SIZE_MXFP2, TERNARY_MAX
    if fmt == "nvint2":
        return BLOCK_SIZE_NVINT2, INT2_MAX
    raise ValueError(f"unknown 2-bit format {fmt!r} (want 'mxfp2' or 'nvint2')")


def encode_tensor(
    values: np.ndarray,
    *,
    fmt: str = "mxfp2",
    mode: str = "rne",
    rng: np.random.Generator | None = None,
):
    """Encode a float vector to a 2-bit master.

    ``fmt="mxfp2"``  -> ``(payload_u8, ue8m0_scale_u8)``; per-block scale is
        ``encode_scale(amax / 1.0)`` (K32), payload packed against the ternary
        ladder.
    ``fmt="nvint2"`` -> ``(payload_u8, e4m3_scale_u8, tensor_scale_f32)``; a
        per-tensor FP32 scale is factored so the largest block's E4M3 scale
        lands at the finite E4M3 max, then per-block scale is
        ``e4m3(amax / 1.5 / tensor_scale)`` (K16), payload packed against the
        4-level INT2 ladder.

    ``mode`` is ``"rne"`` or ``"sr"``; ``"sr"`` needs a seeded ``rng``. Block
    padding matches the MXFP8/MXFP4 twins: trailing partial block zero-padded
    for the amax pass.
    """
    block, ladder_max = _fmt_params(fmt)
    if mode == "sr" and rng is None:
        raise ValueError("mode='sr' requires a seeded numpy Generator")

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    n = flat.size
    nblocks = (n + block - 1) // block
    pad = nblocks * block - n
    block_view = np.concatenate([flat, np.zeros(pad)]).reshape(nblocks, block)
    finite = np.isfinite(block_view)
    amax = np.max(np.where(finite, np.abs(block_view), 0.0), axis=1)

    if fmt == "mxfp2":
        scale_codes = encode_scale(amax / ladder_max).reshape(nblocks)
        applied = safe_scale(scale_codes).reshape(nblocks, 1)
        normalized = (block_view / applied).reshape(-1)[:n]
        payload = (
            ternary_pack_rne(normalized)
            if mode == "rne"
            else ternary_pack_sr(normalized, rng)
        )
        return payload.astype(np.uint8), np.asarray(scale_codes, dtype=np.uint8)

    # nvint2: two-level scale
    gmax = float(np.max(np.abs(flat[np.isfinite(flat)])) if n else 0.0)
    tensor_scale = np.float32(max(gmax, _TINY) / (ladder_max * FP8_E4M3_MAX))
    blk_scale_real = amax / ladder_max / float(tensor_scale)
    scale_codes = e4m3_pack(np.maximum(blk_scale_real, E4M3_SCALE_MIN)).reshape(nblocks)
    applied = (
        np.maximum(e4m3_unpack(scale_codes), np.float32(_TINY)).astype(np.float64)
        * float(tensor_scale)
    ).reshape(nblocks, 1)
    normalized = (block_view / applied).reshape(-1)[:n]
    payload = (
        int2_pack_rne(normalized) if mode == "rne" else int2_pack_sr(normalized, rng)
    )
    return (
        payload.astype(np.uint8),
        np.asarray(scale_codes, dtype=np.uint8),
        tensor_scale,
    )


def decode_tensor(
    payload: np.ndarray,
    scales: np.ndarray,
    numel: int | None = None,
    *,
    fmt: str = "mxfp2",
    tensor_scale: float = 1.0,
) -> np.ndarray:
    """Decode a flat 2-bit payload + block scales to float32.

    ``fmt="mxfp2"``  : ``value = ternary_unpack(p) * safe_scale(scale[i//32])``.
    ``fmt="nvint2"`` : ``value = int2_unpack(p) * e4m3_unpack(scale[i//16]) *
        tensor_scale`` (pass the ``tensor_scale`` from ``encode_tensor``).
    """
    block, _ = _fmt_params(fmt)
    payload = np.asarray(payload, dtype=np.uint8).reshape(-1)
    scales = np.asarray(scales, dtype=np.uint8).reshape(-1)
    n = payload.size if numel is None else int(numel)
    if n > payload.size:
        raise ValueError(f"numel {n} exceeds payload size {payload.size}")
    expect_scales = (n + block - 1) // block
    if scales.size < expect_scales:
        raise ValueError(
            f"need {expect_scales} scale bytes for numel {n}, got {scales.size}"
        )
    if fmt == "mxfp2":
        values = ternary_unpack(payload[:n])
        per_elem = safe_scale(scales[:expect_scales]).repeat(block)[:n]
        return (values * per_elem).astype(np.float32)
    values = int2_unpack(payload[:n])
    per_elem = (
        np.maximum(e4m3_unpack(scales[:expect_scales]), np.float32(_TINY))
        * np.float32(tensor_scale)
    ).repeat(block)[:n]
    return (values * per_elem).astype(np.float32)
