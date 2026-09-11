"""Low-bit weight formats for CPU devices (OMNI-Q's own layer + vendored codecs).

Public API:
    quantize_tensor(values, fmt) / dequantize_tensor(packed)  -- tensor level
    pack_codes / unpack_codes                                 -- 4x2-bit storage

Formats: "mxfp8" (E4M3 × UE8M0 resident top tier), "mxfp4" (4-bit ladder
reference), "nvint2", "mxfp2" — evaluated symmetrically; no preassigned
verdicts. Provenance: vendor/SOURCE.md.
"""

from .lowbit_formats import (
    FORMATS,
    BLOCK_SIZES,
    PackedTensor,
    dequantize_tensor,
    pack_codes,
    quantize_tensor,
    unpack_codes,
)

__all__ = [
    "FORMATS",
    "BLOCK_SIZES",
    "PackedTensor",
    "quantize_tensor",
    "dequantize_tensor",
    "pack_codes",
    "unpack_codes",
]
