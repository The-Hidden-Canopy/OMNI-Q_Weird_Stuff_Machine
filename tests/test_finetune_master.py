"""Tests for perception/finetune.py packed-master training (OQ-008 / OQ-029).

Covers: npz -> state_dict round-trip through finetune's loader path (bit-exact
vs the emitted standalone loader), packed MXFP8 Lion optimizer state and
update semantics, refusal of sim-only mxfp4/mxfp2 masters, the resume-gate
format contract, and CLI wiring of --base/*.npz + --w-master with a fake YOLO
(no real weights, no training).

Design source for the residency semantics: the IDA-TRAIN-V2 MXFP4-master sim
ablation (docs/mxfp4-master-sim-ablation-2026-09-09.md in IDA-TRAIN-V2) — RNE
re-encode of the master each step is stable; SR on the master is a refuted
random walk; mxfp4/mxfp2 masters are sim-only and refused at admission.

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
sys.path.insert(0, str(OMNIQ_ROOT))

import finetune  # noqa: E402
from integrations.qualcomm.lowbit import (  # noqa: E402
    dequantize_tensor,
    quantize_tensor,
)
from package_quant_weights import decode_npz, package  # noqa: E402

torch = pytest.importorskip("torch")


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
def packaged_mxfp8(tmp_path):
    weights = tmp_path / "tiny.pt"
    torch.save(_synthetic_state_dict(), weights)
    out_dir = tmp_path / "out"
    package(weights, ["mxfp8"], out_dir, loader=True)
    return out_dir / "tiny.mxfp8.npz"


def _import_loader(loader_path: Path):
    spec = importlib.util.spec_from_file_location("standalone_loader_under_test",
                                                  loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _codec_roundtrip(weight: torch.Tensor) -> torch.Tensor:
    packed = quantize_tensor(weight.detach().cpu().numpy(), "mxfp8")
    return torch.from_numpy(dequantize_tensor(packed))


def _tiny_trainable() -> torch.nn.Module:
    torch.manual_seed(7)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3),
        torch.nn.BatchNorm2d(4),
        torch.nn.Linear(4, 2),
    )


# --------------------------------------------------------------------------- #
# npz -> state_dict round-trip via finetune's loader path
# --------------------------------------------------------------------------- #

def test_decode_master_state_dict_roundtrip_bitexact_with_loader(packaged_mxfp8,
                                                                 tmp_path):
    state = finetune.decode_master_state_dict(packaged_mxfp8)
    original = _synthetic_state_dict()
    assert set(state) == set(original)

    loader = _import_loader(packaged_mxfp8.with_name("tiny_loader.py"))
    via_loader = loader.decode_npz(str(packaged_mxfp8))
    for name, arr in state.items():
        np.testing.assert_array_equal(arr.numpy(), via_loader[name])
        if not original[name].is_floating_point():
            assert arr.dtype == original[name].dtype

    assert decode_npz(packaged_mxfp8)["model.1.bn.num_batches_tracked"].tobytes() \
        == original["model.1.bn.num_batches_tracked"].numpy().tobytes()


def test_repo_decode_npz_matches_emitted_loader(packaged_mxfp8):
    loader = _import_loader(packaged_mxfp8.with_name("tiny_loader.py"))
    for name, arr in decode_npz(packaged_mxfp8).items():
        np.testing.assert_array_equal(arr, loader.decode_npz(str(packaged_mxfp8))[name])


# --------------------------------------------------------------------------- #
# Packed MXFP8 Lion optimizer
# --------------------------------------------------------------------------- #

def test_packed_optimizer_owns_payload_scales_and_bf16_momentum():
    from packed_optimizer import PackedMXFP8Lion, decode_mxfp8

    parameter = torch.nn.Parameter(torch.linspace(-2.0, 2.0, 65))
    optimizer = PackedMXFP8Lion([parameter], lr=0.1)
    optimizer.initialize_state()
    state = optimizer.state[parameter]

    assert state["format"] == "mxfp8_e4m3_ue8m0_k32"
    assert state["payload"].dtype == torch.uint8
    assert state["scales"].dtype == torch.uint8
    assert state["momentum"].dtype == torch.bfloat16
    assert state["payload"].numel() == parameter.numel()
    assert state["scales"].numel() == 3
    assert state["momentum"].numel() == parameter.numel()
    assert not any(
        torch.is_tensor(value) and value.dtype == torch.float32
        for value in state.values()
    )
    torch.testing.assert_close(
        parameter, decode_mxfp8(state["payload"], state["scales"], parameter.numel())
    )


def test_packed_optimizer_step_matches_decode_update_repack():
    from packed_optimizer import PackedMXFP8Lion

    parameter = torch.nn.Parameter(torch.linspace(-2.0, 2.0, 65))
    optimizer = PackedMXFP8Lion([parameter], lr=0.1, betas=(0.9, 0.99))
    optimizer.initialize_state()
    initial = parameter.detach().clone()
    gradient = torch.linspace(-1.0, 1.0, parameter.numel())
    parameter.grad = gradient.clone()

    optimizer.step()

    projected = _codec_roundtrip(initial)
    direction = 0.1 * gradient  # beta1 * 0 + (1 - beta1) * grad
    expected_unpacked = projected - 0.1 * direction.sign()
    expected = _codec_roundtrip(expected_unpacked)
    torch.testing.assert_close(parameter, expected)
    assert optimizer.state[parameter]["step"] == 1


# --------------------------------------------------------------------------- #
# Sim-only format refusal + resume-gate contract
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fmt", ["mxfp4", "mxfp2"])
def test_w_master_sim_only_formats_refused(fmt):
    with pytest.raises(SystemExit) as exc:
        finetune.validate_w_master(fmt)
    msg = str(exc.value)
    assert "refused" in msg
    assert "mxfp4-master-sim-ablation-2026-09-09" in msg
    assert "random walk" in msg


def test_w_master_accepts_none_and_mxfp8():
    assert finetune.validate_w_master("none") == "none"
    assert finetune.validate_w_master("mxfp8") == "mxfp8"


def test_master_format_contract_match_passes(packaged_mxfp8):
    manifest = finetune.check_master_format_contract(packaged_mxfp8, "mxfp8")
    assert manifest["format"] == "mxfp8"


def test_master_format_contract_mismatch_refused(packaged_mxfp8):
    manifest_path = finetune.manifest_path_for(packaged_mxfp8)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = "mxfp4"  # simulate a wrong-residency master
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="contract violated"):
        finetune.check_master_format_contract(packaged_mxfp8, "mxfp8")


def test_master_format_contract_absent_manifest_is_noop(tmp_path):
    orphan = tmp_path / "lonely.mxfp8.npz"
    orphan.write_bytes(b"not a real npz; contract must not read it")
    assert finetune.check_master_format_contract(orphan, "mxfp8") == {}


# --------------------------------------------------------------------------- #
# CLI wiring (fake YOLO — no real weights, no training)
# --------------------------------------------------------------------------- #

class _FakeInnerModel:
    training = True

    def __init__(self):
        self.loaded = None

    def load_state_dict(self, state, strict=True):
        self.loaded = dict(state)


class _FakeYOLO:
    def __init__(self, base):
        self.base = base
        self.model = _FakeInnerModel()
        self.callbacks = {}
        self.train_kwargs = None

    def add_callback(self, event, fn):
        self.callbacks[event] = fn

    def train(self, **kwargs):
        self.train_kwargs = kwargs


@pytest.fixture()
def fake_yolo(monkeypatch):
    instances = []
    monkeypatch.setattr("ultralytics.YOLO", lambda base: instances.append(
        _FakeYOLO(base)) or instances[-1])
    return instances


def test_cli_npz_base_with_w_master_registers_projection(packaged_mxfp8, fake_yolo,
                                                         capsys):
    finetune.main(["--base", str(packaged_mxfp8), "--w-master", "mxfp8",
                   "--epochs", "1"])
    model = fake_yolo[-1]
    assert model.base == "yolov8n.yaml"
    assert set(model.model.loaded) == set(_synthetic_state_dict())
    assert model.callbacks == {}
    assert model.train_kwargs["trainer"].__name__ == "PackedMasterDetectionTrainer"
    out = capsys.readouterr().out
    assert "packed master: format mxfp8" in out
    assert "mean weight_cosine" in out
    assert "packed optimizer-state training" in out
    assert model.train_kwargs["epochs"] == 1
    assert model.train_kwargs["lr0"] == finetune.PACKED_LION_DEFAULT_LR


def test_cli_pt_base_without_w_master_no_callback(fake_yolo, capsys):
    finetune.main(["--base", "some.pt", "--epochs", "1"])
    model = fake_yolo[-1]
    assert model.base == "some.pt"
    assert model.callbacks == {}
    assert "projected-resident" not in capsys.readouterr().out


def test_cli_refuses_packed_compute_without_mxfp8_master(fake_yolo):
    with pytest.raises(SystemExit, match="requires --w-master mxfp8"):
        finetune.main([
            "--base", "some.pt", "--packed-compute", "mxfp4", "--epochs", "1"
        ])
    assert fake_yolo == []


def test_cli_resume_wires_total_epochs_and_new_save_dir(packaged_mxfp8, fake_yolo,
                                                        tmp_path, capsys):
    finetune.main([
        "--base", "last.pt", "--w-master", "mxfp8", "--resume", "last.pt",
        "--epochs", "120", "--project", str(tmp_path), "--name", "continuation",
    ])
    model = fake_yolo[-1]
    assert model.train_kwargs["resume"] == "last.pt"
    assert model.train_kwargs["epochs"] == 120
    assert model.train_kwargs["save_dir"] == str(tmp_path / "continuation")
    assert "packed optimizer-state training" in capsys.readouterr().out


@pytest.mark.parametrize("fmt", ["mxfp4", "mxfp2"])
def test_cli_refuses_sim_only_w_master(fake_yolo, fmt):
    with pytest.raises(SystemExit, match="refused"):
        finetune.main(["--base", "some.pt", "--w-master", fmt])
    assert fake_yolo == []  # refused before any model is built


def test_cli_refuses_manifest_format_mismatch(packaged_mxfp8, fake_yolo,
                                              monkeypatch):
    manifest_path = finetune.manifest_path_for(packaged_mxfp8)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = "mxfp2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="contract violated"):
        finetune.main(["--base", str(packaged_mxfp8), "--w-master", "mxfp8"])
    assert fake_yolo == []


def test_cli_lower_tier_npz_loads_as_plain_fp32_init(tmp_path, fake_yolo,
                                                    capsys):
    """Residency is refused, not the bytes: a sim-only-tier master may still
    seed an ordinary fp32 fine-tune (degraded init, no on-grid claim)."""
    weights = tmp_path / "tiny.pt"
    torch.save(_synthetic_state_dict(), weights)
    out_dir = tmp_path / "out"
    package(weights, ["mxfp4"], out_dir)
    npz = out_dir / "tiny.mxfp4.npz"
    finetune.main(["--base", str(npz), "--w-master", "none", "--epochs", "1"])
    model = fake_yolo[-1]
    assert model.callbacks == {}
    out = capsys.readouterr().out
    assert "packed master: format mxfp4" in out
    assert "projected-resident" not in out
