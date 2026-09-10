"""Real MuJoCo smoke coverage for the Intel Online adapter.

Skipped in the pure-Python core environment; run with the project venv after
installing ``integrations/intel/requirements.txt`` for this hardware-adjacent
coverage.
"""

from __future__ import annotations

import pytest


mujoco = pytest.importorskip("mujoco")

from omni_q.intel_sim import build_intel_sim_engine, load_dual_so101_model


def test_dual_so101_scene_loads_two_actuated_arms_and_table_cameras():
    model = load_dual_so101_model()

    assert model.nu == 12
    assert model.nq >= 12
    assert model.ncam >= 2


def test_intel_table_setting_steps_real_mujoco_time_on_both_arms():
    engine = build_intel_sim_engine()

    receipt = engine.run("set the table")

    devices = {action["device"] for action in receipt.actions if action["device"]}
    assert {"intel.left_arm", "intel.right_arm"} <= devices
    assert receipt.metrics["resolved"] is True
    assert receipt.inputs["mode"] == "simulation-scripted-manipulation"
    assert engine.world.simulation_summary()["time"] > 0.0
    assert all(
        action["result"].get("simulation_mode") == "simulation-scripted-manipulation"
        for action in receipt.actions
        if action["op"] in {"PICK", "MOVE"}
    )
