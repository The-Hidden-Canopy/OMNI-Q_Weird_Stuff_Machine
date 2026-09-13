"""Tests for perception/package_quant_weights.py — packed low-bit masters.

Covers: npz + manifest production for all slider tiers on a synthetic
state_dict, manifest totals/ratio math, ratio ordering (mxfp8 > mxfp4 > mxfp2),
passthrough marking + byte-exact loader round-trip, and the real cross-check —
the emitted standalone loader's decoder must match the repo codec
(integrations/qualcomm/lowbit.dequantize_tensor) bit-for-bit on the same
PackedTensor. A skipif-gated smoke test packages the real ftv2 weights (mxfp8)
and predicts on a tiny random image with the loader-restored state_dict.

Portions derived from *The Hidden Canopy LLC* — [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2).
Used with permission.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

OMNIQ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OMNIQ_ROOT / "perception"))

from integrations.qualcomm.lowbit import (  # noqa: E402
    BLOCK_SIZES,
    FORMATS,
    PackedTensor,
    dequantize_tensor,
)
from integrations.qualcomm.lowbit.lowbit_formats import CODE_BITS  # noqa: E402
from package_quant_weights import LOADER_SOURCE, package  # noqa: E402

torch = pytest.importorskip("torch")

FORMATS_TIERED = ("mxfp8", "mxfp4", "mxfp2")
REAL_WEIGHTS = OMNIQ_ROOT / "models" / "table_yolo_v2_ft_2026-09-11.pt"


def _synthetic_state_dict() -> dict:
    rng = np.random.default_rng(20260911)
    return {
        "model.0.conv.weight": torch.from_numpy(
            rng.standard_normal((8, 3, 3, 3), dtype=np.float32)),
        "model.1.bn.weight": torch.from_numpy(
            rng.standard_normal((16,), dtype=np.float32)),
        "model.2.cv.conv.weight": torch.from_numpy(
            rng.standard_normal((12, 8, 1, 1), dtype=np.float32)),
        "model.1.bn.num_batches_tracked": torch.arange(3, dtype=torch.int64),
    }


@pytest.fixture()
def packaged(tmp_path):
    weights = tmp_path / "tiny.pt"
    torch.save(_synthetic_state_dict(), weights)
    out_dir = tmp_path / "out"
    results = package(weights, list(FORMATS_TIERED), out_dir, loader=True)
    return {"weights": weights, "out_dir": out_dir, "results": results,
            "stem": "tiny"}


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz_meta(npz_path: Path):
    with np.load(npz_path) as npz:
        meta = json.loads(npz["meta_json"].item())
        payload = npz["payload"].tobytes()
        scales = npz["scales"].tobytes()
    return meta, payload, scales


def _import_loader(loader_path: Path):
    spec = importlib.util.spec_from_file_location("standalone_loader_under_test",
                                                  loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_formats_produce_npz_and_manifest(packaged):
    for fmt in FORMATS_TIERED:
        assert (packaged["out_dir"] / f"tiny.{fmt}.npz").is_file()
        manifest = _load_manifest(packaged["out_dir"] / f"tiny.{fmt}.manifest.json")
        assert manifest["format"] == fmt
        assert manifest["block_size"] == BLOCK_SIZES[fmt]
        assert manifest["code_bits"] == CODE_BITS[fmt]
        assert fmt in FORMATS


def test_manifest_totals_and_ratio_math(packaged):
    state = _synthetic_state_dict()
    expected_params = sum(int(t.numel()) for t in state.values())
    for fmt in FORMATS_TIERED:
        manifest = _load_manifest(packaged["out_dir"] / f"tiny.{fmt}.manifest.json")
        totals = manifest["totals"]
        per_tensor_sum = sum(t["packed_bytes"] for t in manifest["tensors"])
        assert totals["packed_bytes"] == per_tensor_sum
        assert totals["params"] == expected_params
        assert totals["fp32_bytes"] == expected_params * 4
        assert totals["ratio"] == pytest.approx(totals["packed_bytes"]
                                                / totals["fp32_bytes"])
        cosines = [t["weight_cosine"] for t in manifest["tensors"]
                   if "dtype_passthrough" not in t]
        assert totals["mean_weight_cosine"] == pytest.approx(float(np.mean(cosines)))
        assert manifest["model"]["param_count"] == expected_params
        assert manifest["model"]["source_file"] == "tiny.pt"
        assert "mxfp4/mxfp2 collapse" in manifest["honesty"]["label"]
        assert manifest["honesty"]["arms_ref"].startswith(
            "evidence/benchmark_results/yolo_2bit_map_20260911_071815")


def test_npz_packed_bytes_match_manifest_per_tensor(packaged):
    for fmt in FORMATS_TIERED:
        manifest = _load_manifest(packaged["out_dir"] / f"tiny.{fmt}.manifest.json")
        meta, payload, scales = _load_npz_meta(packaged["out_dir"] / f"tiny.{fmt}.npz")
        assert len(meta["tensors"]) == len(manifest["tensors"])
        for m_entry, j_entry in zip(meta["tensors"], manifest["tensors"]):
            assert m_entry["name"] == j_entry["name"]
            packed_bytes = j_entry["packed_bytes"]
            assert packed_bytes == m_entry["payload_len"] + m_entry["scales_len"]
            assert len(payload[m_entry["payload_off"]:
                               m_entry["payload_off"] + m_entry["payload_len"]]
                       ) == m_entry["payload_len"]
            assert len(scales[m_entry["scales_off"]:
                              m_entry["scales_off"] + m_entry["scales_len"]]
                       ) == m_entry["scales_len"]
            if m_entry["kind"] == "quant":
                assert m_entry["fmt"] == fmt
                assert m_entry["block_size"] == BLOCK_SIZES[fmt]


def test_ratio_ordering_mxfp8_gt_mxfp4_gt_mxfp2(packaged):
    ratios = {}
    for fmt in FORMATS_TIERED:
        manifest = _load_manifest(packaged["out_dir"] / f"tiny.{fmt}.manifest.json")
        ratios[fmt] = manifest["totals"]["ratio"]
    assert ratios["mxfp8"] > ratios["mxfp4"] > ratios["mxfp2"]


def test_passthrough_marked_and_byte_exact_via_loader(packaged):
    loader = _import_loader(packaged["out_dir"] / "tiny_loader.py")
    state = _synthetic_state_dict()
    for fmt in FORMATS_TIERED:
        manifest = _load_manifest(packaged["out_dir"] / f"tiny.{fmt}.manifest.json")
        passthroughs = [t for t in manifest["tensors"] if "dtype_passthrough" in t]
        assert len(passthroughs) == 1
        assert passthroughs[0]["dtype_passthrough"] == "int64"
        restored = loader.load_state_dict(str(packaged["out_dir"]
                                              / f"tiny.{fmt}.npz"))
        original = state["model.1.bn.num_batches_tracked"].numpy()
        np.testing.assert_array_equal(
            restored["model.1.bn.num_batches_tracked"], original)
        assert restored["model.1.bn.num_batches_tracked"].tobytes() == original.tobytes()


@pytest.mark.parametrize("fmt", FORMATS_TIERED)
def test_standalone_loader_matches_repo_codec(packaged, fmt):
    """The emitted loader's decoder must equal lowbit.dequantize_tensor exactly."""
    loader = _import_loader(packaged["out_dir"] / "tiny_loader.py")
    meta, payload, scales = _load_npz_meta(packaged["out_dir"] / f"tiny.{fmt}.npz")
    for entry in meta["tensors"]:
        if entry["kind"] != "quant":
            continue
        p = payload[entry["payload_off"]:entry["payload_off"] + entry["payload_len"]]
        s = scales[entry["scales_off"]:entry["scales_off"] + entry["scales_len"]]
        got = loader.decode_payload(entry, p, s).reshape(tuple(entry["shape"]))
        packed = PackedTensor(
            fmt=fmt,
            shape=tuple(entry["shape"]),
            numel=entry["numel"],
            payload=p,
            scales=s,
            tensor_scale=entry.get("tensor_scale", 1.0),
        )
        want = dequantize_tensor(packed)
        np.testing.assert_array_equal(got, want)


def test_loader_source_has_no_repo_deps(packaged):
    loader_path = packaged["out_dir"] / "tiny_loader.py"
    text = loader_path.read_text(encoding="utf-8")
    assert "OMNI-Q_Weird_Stuff_Machine" in text  # attribution header
    for banned in ("integrations", "sys.path", "import ultralytics"):
        assert banned not in text


def test_package_unknown_format_rejected(tmp_path):
    weights = tmp_path / "tiny.pt"
    torch.save(_synthetic_state_dict(), weights)
    with pytest.raises(ValueError, match="unknown format"):
        package(weights, ["fp64"], tmp_path / "out")


@pytest.mark.skipif(not REAL_WEIGHTS.is_file(), reason="real ftv2 weights not present")
def test_real_weights_mxfp8_smoke(tmp_path):
    """Package the real model (mxfp8), restore via loader, predict on noise."""
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    out_dir = tmp_path / "out"
    results = package(REAL_WEIGHTS, ["mxfp8"], out_dir, loader=True)
    manifest = results["mxfp8"]
    assert manifest["totals"]["mean_weight_cosine"] > 0.999

    loader = _import_loader(out_dir / f"{REAL_WEIGHTS.stem}_loader.py")
    restored = loader.load_state_dict(str(out_dir / f"{REAL_WEIGHTS.stem}.mxfp8.npz"))
    restored_pt = tmp_path / "restored.pt"
    torch.save({k: torch.from_numpy(np.ascontiguousarray(v).copy())
                for k, v in restored.items()}, restored_pt)

    model = YOLO(str(REAL_WEIGHTS))
    model.model.load_state_dict(torch.load(restored_pt, map_location="cpu"))
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    preds = model.predict(source=image, verbose=False)
    assert len(preds) == 1
