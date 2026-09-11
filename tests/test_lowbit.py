"""Low-bit codec acceptance — OMNI-Q's storage layer over the vendored 2-bit codec.

Covers pack/unpack round-trips (odd sizes + padding), quantize/dequantize
per format (shape, dtype, stats, bits-per-weight), the MXFP8 E4M3fn and
MXFP4 E2M1 known-value and tie-to-even rules, the weight-cosine ordering
across the arms on a fixed-seed gaussian, and — when ``IDA_TRAIN_V2_ROOT``
is set — an exact encode/decode parity diff of the vendored twins against
the source repo.

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
from integrations.qualcomm.lowbit.vendor.mxfp_scales import safe_scale


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
# MXFP8 (E4M3fn + UE8M0 K32) — resident top tier of the compression slider      #
# --------------------------------------------------------------------------- #
def _mxfp8_block_with_amax(probes: np.ndarray, amax: float = 448.0) -> np.ndarray:
    """One 32-element block whose amax is exactly ``amax`` (first slot)."""
    block = np.zeros(32, dtype=np.float32)
    block[0] = amax
    block[1 : 1 + probes.size] = probes
    return block


def test_mxfp8_e4m3_known_values_roundtrip_exactly():
    probes = np.array(
        [1.0, -1.0, 2.0, -2.0, 4.0, 8.0, 16.0, 256.0, 448.0, -448.0,
         3.0, 5.0, 6.0, 7.0, 9.0, 12.0, 15.0, 384.0,
         0.5, 0.25, -0.125,
         2.0 ** -9, 3.0 * 2.0 ** -9, 7.0 * 2.0 ** -9, -2.0 ** -9],
        dtype=np.float32,
    )
    w = _mxfp8_block_with_amax(probes)
    # amax/448 = 1.0 -> UE8M0 scale code 127 -> scale exactly 1.0, so the
    # E4M3fn grid values themselves are the representable set.
    restored = dequantize_tensor(quantize_tensor(w, "mxfp8"))
    np.testing.assert_array_equal(restored[1 : 1 + probes.size], probes)


def test_mxfp8_monotone_input_stays_monotone():
    w = np.linspace(-448.0, 448.0, 32, dtype=np.float32)  # one block, amax 448
    restored = dequantize_tensor(quantize_tensor(w, "mxfp8"))
    assert np.all(np.diff(restored) >= 0)


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.0625, 1.0),       # tie between 1.0 (code 0x38, even) and 1.125 (0x39, odd)
        (1.1875, 1.25),      # tie between 1.125 (code 0x39, odd) and 1.25 (0x3A, even)
        (2.0 ** -10, 0.0),   # tie between 0 (code 0x00, even) and 2**-9 (0x01, odd)
    ],
)
def test_mxfp8_e4m3_tie_goes_to_even_code(value: float, expected: float):
    w = _mxfp8_block_with_amax(np.array([value], dtype=np.float32))
    restored = dequantize_tensor(quantize_tensor(w, "mxfp8"))
    assert restored[1] == pytest.approx(expected, abs=1e-6), (
        f"tie at {value} must snap to the even code ({expected})"
    )


def test_mxfp8_nan_input_roundtrips_as_nan():
    w = _mxfp8_block_with_amax(np.array([np.nan], dtype=np.float32))
    packed = quantize_tensor(w, "mxfp8")
    # mxfp8 payload is one uint8 per element; NaN takes the finite-only
    # codec's NaN code 0x7F (finite amax comes from slot 0, NaN is ignored).
    assert packed.payload[1] == 0x7F
    restored = dequantize_tensor(packed)
    assert np.isnan(restored[1])
    np.testing.assert_array_equal(restored[2:], np.zeros(30, dtype=np.float32))


def test_mxfp8_zero_block_uses_zero_scale_code():
    w = np.zeros(64, dtype=np.float32)
    packed = quantize_tensor(w, "mxfp8")
    # amax == 0 -> encode_scale(0) == 0 (safe_scale maps code 0 to 1e-30,
    # same contract the other formats rely on); zeros survive untouched.
    assert packed.payload == bytes(64)
    assert packed.scales == bytes(2)
    np.testing.assert_array_equal(dequantize_tensor(packed), w)


def test_mxfp8_normal_range_relative_error_bounded():
    w = _gaussian()
    packed = quantize_tensor(w, "mxfp8")
    restored = dequantize_tensor(packed).reshape(-1)
    scales = safe_scale(np.frombuffer(packed.scales, dtype=np.uint8))
    per_elem = scales.repeat(BLOCK_SIZES["mxfp8"])[: w.size]
    # e4m3fn min normal is 2**-6 (times the block scale); elements at or
    # above it round within the normal grid, where 3 mantissa bits bound the
    # relative error at 2**-4 (half ULP) — comfortably under 2**-3.
    mask = np.abs(w) >= (2.0 ** -6) * per_elem
    rel = np.abs(restored[mask] - w[mask]) / np.abs(w[mask])
    assert np.max(rel) <= 2.0 ** -3


@pytest.mark.parametrize("numel", [32, 64, 4096, 4128])
def test_mxfp8_byte_accounting(numel: int):
    w = _gaussian(numel)
    packed = quantize_tensor(w, "mxfp8")
    nblocks = (numel + BLOCK_SIZES["mxfp8"] - 1) // BLOCK_SIZES["mxfp8"]
    assert len(packed.payload) == numel          # one uint8 per code
    assert len(packed.scales) == nblocks         # one UE8M0 byte per block
    assert packed.bytes_total == numel + nblocks
    assert packed.bits_per_weight == pytest.approx(
        8.0 * (numel + nblocks) / numel, abs=1e-9)


def test_mxfp8_mean_relative_error_well_below_mxfp4():
    w = _gaussian()
    means = {}
    for fmt in ("mxfp8", "mxfp4"):
        restored = dequantize_tensor(quantize_tensor(w, fmt)).reshape(-1)
        nz = w != 0
        means[fmt] = float(np.mean(np.abs(restored[nz] - w[nz]) / np.abs(w[nz])))
    assert means["mxfp8"] < 0.5 * means["mxfp4"], means


# --------------------------------------------------------------------------- #
# Parity vs the source repo (IDA_TRAIN_V2_ROOT)                               #
# --------------------------------------------------------------------------- #
def _load_source_module(name: str):
    root = os.environ.get("IDA_TRAIN_V2_ROOT")
    if not root:
        pytest.skip("IDA_TRAIN_V2_ROOT not set; source repo is only a test oracle")
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(name)


def _load_source_codec():
    return _load_source_module("ida_train.native.mxfp2_codec")


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


def test_parity_with_source_mxfp8_codec():
    source = _load_source_module("ida_train.native.mxfp8_codec")
    vendored = importlib.import_module(
        "integrations.qualcomm.lowbit.vendor.mxfp_scales")

    rng = np.random.default_rng(20260911)
    vectors = [
        rng.standard_normal(1024, dtype=np.float32)
        * np.exp2(rng.integers(-12, 9, 1024)),
        np.array([0.0, 448.0, -448.0, 2.0 ** -9, 7 * 2.0 ** -9, np.nan],
                 dtype=np.float32),
    ]
    for vec in vectors:
        codes = vendored.e4m3_pack(vec)
        np.testing.assert_array_equal(codes, source.e4m3_pack(vec))
        np.testing.assert_array_equal(vendored.e4m3_unpack(codes),
                                      source.e4m3_unpack(codes))
        for val in [0.0, 1.0, 448.0, 1e-30, 1e30]:
            assert vendored.encode_scale(val) == source.encode_scale(val)
            c = vendored.encode_scale(val)
            assert vendored.safe_scale(c) == source.safe_scale(c)

    # end-to-end: our mxfp8 tensor arm against a reference built from the
    # source helpers composed block-wise
    w = (rng.standard_normal(4096, dtype=np.float32)
         * np.exp2(rng.integers(-8, 8, 4096)))
    ours = quantize_tensor(w, "mxfp8")
    blocks = w.reshape(-1, 32)
    amax = np.max(np.abs(blocks), axis=1)
    ref_scales = source.encode_scale(amax / 448.0)
    ref_codes = source.e4m3_pack(
        blocks / source.safe_scale(ref_scales)[:, None]).reshape(-1)
    np.testing.assert_array_equal(
        np.frombuffer(ours.scales, dtype=np.uint8), ref_scales)
    np.testing.assert_array_equal(
        np.frombuffer(ours.payload, dtype=np.uint8), ref_codes)
