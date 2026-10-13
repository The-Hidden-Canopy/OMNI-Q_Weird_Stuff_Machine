"""Public low-bit weight formats for CPU devices.

Public API:
    quantize_tensor(values, fmt) / dequantize_tensor(packed)  -- tensor level
    pack_nibbles / unpack_nibbles                             -- 2x4-bit storage

Formats: "mxfp8" (E4M3 × UE8M0 resident top tier) and "mxfp4" (4-bit
reference ladder). Experimental formats are not shipped by the public build.
"""

from .lowbit_formats import (
    FORMATS,
    BLOCK_SIZES,
    PackedTensor,
    dequantize_tensor,
    pack_nibbles,
    quantize_tensor,
    unpack_nibbles,
)

__all__ = [
    "FORMATS",
    "BLOCK_SIZES",
    "PackedTensor",
    "quantize_tensor",
    "dequantize_tensor",
    "pack_nibbles",
    "unpack_nibbles",
]
