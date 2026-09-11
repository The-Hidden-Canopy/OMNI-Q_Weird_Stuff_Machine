"""Smoke test for the mAP eval script wiring (no model load)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

OMNIQ_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = OMNIQ_ROOT / "integrations" / "qualcomm" / "scripts" / "eval_yolo_2bit_map.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_yolo_2bit_map", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_module_imports_and_exposes_arms():
    mod = _load_module()
    assert mod.ARMS == ("fp32", "mxfp4", "nvint2", "mxfp2")
    assert mod.FMTS == ("mxfp4", "nvint2", "mxfp2")


def test_script_has_argparse_main_wiring_and_defaults():
    mod = _load_module()
    assert callable(mod.main)
    assert Path(mod.MODEL_DEFAULT).name == "best.pt"
    assert "table_yolo" in mod.DATA_DEFAULT
    assert "data.yaml" in mod.DATA_DEFAULT


def test_script_exposes_helper_functions():
    mod = _load_module()
    for fn in ("load_model", "quantize_arm", "run_val", "render_readme",
               "count_val_images", "count_val_instances"):
        assert callable(getattr(mod, fn)), fn
