"""Vendored 2-bit weight codecs (MXFP2 / NVINT2) + scale helpers.

Self-contained numerical twins of the IDA-TRAIN-V2 reference codecs — see
SOURCE.md for provenance. Import from here, never from the source repo.
"""

from .mxfp_scales import (
    FP8_E4M3_MAX,
    e4m3_pack,
    e4m3_unpack,
    encode_scale,
    safe_scale,
)
from .mxfp2_codec import (
    BLOCK_SIZE_MXFP2,
    BLOCK_SIZE_NVINT2,
    E4M3_SCALE_MIN,
    INT2_LEVELS,
    INT2_MAX,
    MXFP2_WEIGHTS_DTYPE,
    NVINT2_WEIGHTS_DTYPE,
    TERNARY_MAGNITUDES,
    TERNARY_MAX,
    decode_tensor,
    encode_tensor,
    int2_pack_rne,
    int2_pack_sr,
    int2_unpack,
    ternary_pack_rne,
    ternary_pack_sr,
    ternary_unpack,
)

__all__ = [
    "FP8_E4M3_MAX",
    "BLOCK_SIZE_MXFP2",
    "BLOCK_SIZE_NVINT2",
    "MXFP2_WEIGHTS_DTYPE",
    "NVINT2_WEIGHTS_DTYPE",
    "encode_tensor",
    "decode_tensor",
    "encode_scale",
    "safe_scale",
    "e4m3_pack",
    "e4m3_unpack",
]
