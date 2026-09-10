"""Low-bit codec acceptance — OMNI-Q's storage layer over the vendored 2-bit codec.

Covers pack/unpack round-trips (odd sizes + padding), quantize/dequantize
per format (shape, dtype, stats, bits-per-weight), the MXFP4 E2M1 known-value
and tie-to-even rules, the weight-cosine ordering across the three arms on a
fixed-seed gaussian, and — when ``IDA_TRAIN_V2_ROOT`` is set — an exact
encode/decode parity diff of the vendored twins against the source repo.

Portions derived from *The Hidden Canopy LLC* —
[`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2).
Used with permission.
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

from integrations.qualcomm.lowbit import (
    FORMATS,
    dequantize_tensor,
    quantize_tensor,
)
from integrations.qualcomm.lowbit.lowbit_formats import (
    BLOCK_SIZES,
    CODE_BITS,
    pack_codes,
    pack_nibbles,
    unpack_codes,
    unpack_nibbles,
)


def _gaussian(numel: int = 4096, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(numel, dtype=np.float32)


# --------------------------------------------------------------------------- #
# pack / unpack round-trips                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 33, 63, 1000])
def test_pack_codes_roundtrip_odd_sizes(n: int):
    rng = np.random.default_rng(n)
    codes = rng.integers(0, 4, size=n, dtype=np.uint8)
    payload = pack_codes(codes)
    assert isinstance(payload, bytes)
    assert len(payload) == (n + 3) // 4
    restored = unpack_codes(payload, n)
    np.testing.assert_array_equal(restored, codes)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 33, 1000])
def test_pack_nibbles_roundtrip_odd_sizes(n: int):
    rng = np.random.default_rng(10_000 + n)
    codes = rng.integers(0, 16, size=n, dtype=np.uint8)
    payload = pack_nibbles(codes)
    assert isinstance(payload, bytes)
    assert len(payload) == (n + 1) // 2
    restored = unpack_nibbles(payload, n)
    np.testing.assert_array_equal(restored, codes)


def test_pack_codes_padding_bits_are_zero():
    codes = np.array([3, 1, 2], dtype=np.uint8)  # 3 elements -> 1 byte, 1 pad slot
    payload = pack_codes(codes)
    assert len(payload) == 1
    restored = unpack_codes(payload, 4)
    assert restored[3] == 0, "padding slots decode to code 0"


# --------------------------------------------------------------------------- #
# quantize / dequantize per format                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("shape", [(4096,), (64, 64), (24, 16, 8)])
def test_quantize_dequantize_roundtrip_shape_dtype_stats(fmt: str, shape: tuple):
    w = _gaussian(int(np.prod(shape))).reshape(shape)
    packed = quantize_tensor(w, fmt)
    assert packed.fmt == fmt
    assert packed.shape == shape
    assert packed.numel == w.size
    restored = dequantize_tensor(packed)
    assert restored.shape == shape
    assert restored.dtype == np.float32
    assert packed.stats["weight_cosine"] > 0.5
    assert "weight_rel_err" in packed.stats
    assert packed.stats["block_size"] == BLOCK_SIZES[fmt]
    # payload alone carries exactly CODE_BITS per weight once packing
    # granularity (4 codes/byte for 2-bit, 2 for 4-bit) divides numel;
    # scale bytes add the small per-block overhead.
    granularity = 8 // CODE_BITS[fmt]
    if w.size % granularity == 0:
        payload_bits = 8 * len(packed.payload) / packed.numel
        assert payload_bits == pytest.approx(CODE_BITS[fmt], abs=1e-9)
    assert packed.bits_per_weight > CODE_BITS[fmt]
    assert packed.bits_per_weight <= CODE_BITS[fmt] + 8.0 / BLOCK_SIZES[fmt] + 40.0 / packed.numel


def test_weight_cosine_ordering_mxfp4_gt_nvint2_gt_mxfp2():
    w = _gaussian()
    cos = {fmt: quantize_tensor(w, fmt).stats["weight_cosine"] for fmt in FORMATS}
    assert cos["mxfp4"] > cos["nvint2"] > cos["mxfp2"], cos


# --------------------------------------------------------------------------- #
# MXFP4 E2M1 known values + tie-to-even                                       #
# --------------------------------------------------------------------------- #
def _mxfp4_block_with_amax(probes: np.ndarray, amax: float = 6.0) -> np.ndarray:
    """One 32-element block whose amax is exactly ``amax`` (first slot)."""
    block = np.zeros(32, dtype=np.float32)
    block[0] = amax
    block[1 : 1 + probes.size] = probes
    return block


def test_mxfp4_e2m1_known_values_roundtrip_exactly():
    probes = np.array([0.0, 0.5, -0.5, 6.0, -6.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0],
                      dtype=np.float32)
    w = _mxfp4_block_with_amax(probes)
    # amax/6 = 1.0 -> UE8M0 scale code 127 -> scale exactly 1.0, so the
    # E2M1 ladder values themselves are the representable grid.
    restored = dequantize_tensor(quantize_tensor(w, "mxfp4"))
    np.testing.assert_array_equal(restored[1 : 1 + probes.size], probes)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.75, 1.0),   # tie between 0.5 (code 1, odd) and 1.0 (code 2, even) -> even
        (1.25, 1.0),   # tie between 1.0 (code 2, even) and 1.5 (code 3, odd) -> even
        (2.5, 2.0),    # tie between 2.0 (code 4, even) and 3.0 (code 5, odd) -> even
        (5.0, 4.0),    # tie between 4.0 (code 6, even) and 6.0 (code 7, odd) -> even
    ],
)
def test_mxfp4_e2m1_tie_goes_to_even_code(value: float, expected: float):
    w = _mxfp4_block_with_amax(np.array([value], dtype=np.float32))
    restored = dequantize_tensor(quantize_tensor(w, "mxfp4"))
    assert restored[1] == pytest.approx(expected, abs=1e-6), (
        f"tie at {value} must snap to the even code ({expected})"
    )


# --------------------------------------------------------------------------- #
# Parity vs the source repo (IDA_TRAIN_V2_ROOT)                               #
# --------------------------------------------------------------------------- #
def _load_source_codec():
    root = os.environ.get("IDA_TRAIN_V2_ROOT")
    if not root:
        pytest.skip("IDA_TRAIN_V2_ROOT not set; source repo is only a test oracle")
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module("ida_train.native.mxfp2_codec")


@pytest.mark.parametrize("fmt", ["mxfp2", "nvint2"])
def test_parity_with_source_codec_encode_decode(fmt: str):
    source = _load_source_codec()
    vendored = importlib.import_module(
        "integrations.qualcomm.lowbit.vendor.mxfp2_codec")

    rng = np.random.default_rng(20260910)
    vectors = [
        rng.standard_normal(1024, dtype=np.float32),
        rng.standard_normal(4096, dtype=np.float32) * np.exp2(rng.integers(-8, 8, 4096)),
        np.array([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 6.0], dtype=np.float32),
        rng.standard_normal(33, dtype=np.float32),  # partial trailing block
    ]
    for vec in vectors:
        ours = vendored.encode_tensor(vec, fmt=fmt, mode="rne")
        theirs = source.encode_tensor(vec, fmt=fmt, mode="rne")
        for a, b in zip(ours, theirs):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        ours_dec = vendored.decode_tensor(*ours[:2], numel=vec.size, fmt=fmt,
                                           tensor_scale=ours[2] if fmt == "nvint2" else 1.0)
        theirs_dec = source.decode_tensor(*theirs[:2], numel=vec.size, fmt=fmt,
                                          tensor_scale=theirs[2] if fmt == "nvint2" else 1.0)
        np.testing.assert_array_equal(ours_dec, theirs_dec)
