"""Tests for the fused MXFP8 RNE re-projection fast path (OQ-029 follow-up).

`rne_reproject_tensor` must be bit-exact against the codec round-trip
``dequantize_tensor(quantize_tensor(x, "mxfp8"))`` on every distribution —
the masters on HuggingFace are sacred, so the fused path may not drift by a
single ULP. Covers: random tensors at many scales (incl. near-block-boundary
values, zeros, denormals, values > 448, NaN/Inf), the real table_yolo_v2
Conv2d weights, packed optimizer-state wiring, and a printed (non-asserting)
slow-vs-fast benchmark.

Portions derived from *The Hidden Canopy LLC* — [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2).
Used with permission.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

OMNIQ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OMNIQ_ROOT / "perception"))
sys.path.insert(0, str(OMNIQ_ROOT))

import finetune  # noqa: E402
from integrations.qualcomm.lowbit import (  # noqa: E402
    dequantize_tensor,
    quantize_tensor,
)
from integrations.qualcomm.lowbit.vendor.mxfp_scales import (  # noqa: E402
    _E4M3_UNPACK_LUT,
)
from package_quant_weights import (  # noqa: E402
    load_state_dict,
    package,
    rne_reproject_tensor,
)

torch = pytest.importorskip("torch")


def _slow_roundtrip(x: np.ndarray) -> np.ndarray:
    return dequantize_tensor(quantize_tensor(x, "mxfp8"))


def _assert_bit_exact(x: np.ndarray, label: str) -> None:
    want = _slow_roundtrip(x)
    got = rne_reproject_tensor(x)
    assert got.dtype == np.float32, label
    assert got.shape == want.shape, label
    same = (got == want) | (np.isnan(got) & np.isnan(want))
    if not bool(same.all()):
        idx = np.nonzero(~same)[0][:5]
        raise AssertionError(
            f"{label}: {int((~same).sum())} mismatches; "
            f"x={x.reshape(-1)[idx].tolist()} "
            f"want={want.reshape(-1)[idx].tolist()} "
            f"got={got.reshape(-1)[idx].tolist()}")


# --------------------------------------------------------------------------- #
# Bit-exactness: random tensors across scales and hostile value classes       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e3, 1e-20, 1e20, 1e-38, 3e38])
def test_bit_exact_random_scales(scale):
    rng = np.random.default_rng(20260911)
    x = (rng.standard_normal(300_001) * scale).astype(np.float32)
    _assert_bit_exact(x, f"randn*{scale}")


def test_bit_exact_block_boundary_sizes():
    rng = np.random.default_rng(7)
    for n in (1, 31, 32, 33, 63, 4095, 4096, 4097):
        _assert_bit_exact(rng.standard_normal(n).astype(np.float32), f"n={n}")


def test_bit_exact_e4m3_midpoints_and_ladder():
    # Every midpoint between adjacent e4m3 magnitudes (RNE tie cases), each
    # grid point itself, and boundary values, at several block scales.
    grid = np.unique(np.abs(_E4M3_UNPACK_LUT[np.isfinite(_E4M3_UNPACK_LUT)]))
    mids = np.array([0.5 * (a + b) for a, b in zip(grid[:-1], grid[1:])],
                    dtype=np.float32)
    special = np.array([0.0, -0.0, 448.0, -448.0, 447.9999, 448.0001,
                        2**-6, -(2**-6), 2**-9, 1e-30], dtype=np.float32)
    base = np.concatenate([grid, mids, special, -grid, -mids])
    rng = np.random.default_rng(3)
    for scale in (1.0, 1e6, 1e-8, 1e-25):
        x = np.concatenate([base, rng.standard_normal(97).astype(np.float32)])
        _assert_bit_exact((x * np.float32(scale)).astype(np.float32),
                          f"midpoints*{scale}")


def test_bit_exact_zeros_denormals_huge_nan_inf():
    rng = np.random.default_rng(11)
    x = rng.standard_normal(2048).astype(np.float32)
    x[0::8] = 0.0
    x[1::8] = -0.0
    x[2::8] = np.nan
    x[3::8] = np.inf
    x[4::8] = -np.inf
    x[5::8] = np.float32(1e-42)          # f32 denormal
    x[6::8] = np.float32(-1e-44)
    x[7::8] = np.float32(3.4e38)
    _assert_bit_exact(x, "zeros/denormals/huge/nan/inf")


def test_bit_exact_floored_scale_blocks():
    # amax < ~3.5e-28 -> UE8M0 codes < 28 -> safe_scale floored to 1e-30
    # (not a power of two): exercises the exact f64 fallback branch. NaN does
    # not move amax, so floored blocks can still contain NaN/Inf/zeros.
    rng = np.random.default_rng(13)
    for scale in (1e-40, 1e-41, 1e-44, 1e-28, 1e-27):
        x = (rng.uniform(-2, 2, 4096) * np.float32(scale)).astype(np.float32)
        x[5] = np.nan
        x[100] = -np.inf
        x[200] = 0.0
        x[300] = -0.0
        _assert_bit_exact(x, f"floored*{scale}")
    # zero blocks (all-nonpositive amax): only 0/NaN/Inf allowed
    for fill in (0.0, -0.0, np.nan, np.inf, -np.inf):
        _assert_bit_exact(np.full(96, fill, dtype=np.float32), f"all-{fill}")


def test_reprojection_is_idempotent_on_grid():
    rng = np.random.default_rng(17)
    x = rng.standard_normal(100_003).astype(np.float32)
    once = rne_reproject_tensor(x)
    twice = rne_reproject_tensor(once)
    np.testing.assert_array_equal(once.view(np.uint32), twice.view(np.uint32))


# --------------------------------------------------------------------------- #
# Bit-exactness on the real model weights                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not (OMNIQ_ROOT / "models" / "table_yolo_v2_ft_2026-09-11.pt").is_file(),
    reason="table_yolo_v2_ft_2026-09-11.pt not present")
def test_bit_exact_real_model_conv_weights():
    state = load_state_dict(OMNIQ_ROOT / "models" / "table_yolo_v2_ft_2026-09-11.pt")
    checked = 0
    for name, tensor in state.items():
        if not tensor.is_floating_point():
            continue
        w = tensor.detach().cpu().numpy().astype(np.float32)
        want = _slow_roundtrip(w)
        got = rne_reproject_tensor(w)
        same = (got == want) | (np.isnan(got) & np.isnan(want))
        assert bool(same.all()), f"{name}: {int((~same).sum())} mismatches"
        checked += 1
    assert checked > 0


# --------------------------------------------------------------------------- #
# Finetune packed optimizer wiring                                             #
# --------------------------------------------------------------------------- #

def _tiny_trainable() -> torch.nn.Module:
    torch.manual_seed(7)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3),
        torch.nn.BatchNorm2d(4),
        torch.nn.Linear(4, 2),
    )


def test_packed_optimizer_initialization_matches_codec_roundtrip():
    from packed_optimizer import PackedMXFP8Lion

    model = _tiny_trainable()
    with torch.no_grad():
        model[0].weight.mul_(1.7)
        model[2].weight.mul_(0.31)
    before = {n: p.clone() for n, p in model.named_parameters()}

    optimizer = PackedMXFP8Lion(model.parameters(), lr=1e-3)
    optimizer.initialize_state()
    for name, param in model.named_parameters():
        if param.requires_grad:
            want = _slow_roundtrip(before[name].detach().cpu().numpy())
            torch.testing.assert_close(
                param, torch.from_numpy(want).reshape(param.shape))


class _FakeInnerModel:
    training = True

    def load_state_dict(self, state, strict=True):
        pass


class _FakeYOLO:
    def __init__(self, base):
        self.model = _FakeInnerModel()
        self.callbacks = {}

    def add_callback(self, event, fn):
        self.callbacks[event] = fn

    def train(self, **kwargs):
        pass


@pytest.fixture()
def fake_yolo(monkeypatch):
    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", lambda base: _FakeYOLO(base))


def test_cli_note_reports_active_path(tmp_path, fake_yolo, capsys):
    weights = tmp_path / "tiny.pt"
    torch.save({"w": torch.randn(4, 4)}, weights)
    package(weights, ["mxfp8"], tmp_path)
    finetune.main(["--base", str(tmp_path / "tiny.mxfp8.npz"),
                   "--w-master", "mxfp8", "--epochs", "1"])
    out = capsys.readouterr().out
    assert "packed optimizer-state training" in out
    assert "packed optimizer-state training" in out


# --------------------------------------------------------------------------- #
# Benchmark (printed only — not an assertion)                                 #
# --------------------------------------------------------------------------- #

def _min_ms(fn, reps=3) -> float:
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def test_benchmark_slow_vs_fast(capsys):
    rng = np.random.default_rng(20260911)
    x = rng.standard_normal(3_000_000).astype(np.float32)
    slow_ms = _min_ms(lambda: _slow_roundtrip(x))
    fast_ms = _min_ms(lambda: rne_reproject_tensor(x))
    with capsys.disabled():
        print(f"\n[rne-bench] 3M params: slow {slow_ms:.1f} ms/tensor, "
              f"fast {fast_ms:.1f} ms/tensor ({slow_ms / fast_ms:.1f}x)")
