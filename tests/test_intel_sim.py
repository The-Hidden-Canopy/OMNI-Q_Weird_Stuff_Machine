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
    """Real IK + contact grasp (OQ-010) replaced the old kinematic teleport,
    but a reliable pinch needs orientation-aware IK this adapter doesn't have
    yet -- position-only IK converges (see test_intel_sim_primitives.py) but
    the jaws aren't guaranteed to be angled onto the object, so grasps
    currently fail more than they succeed. This does NOT assert
    resolved is True: that would be asserting a success rate the system
    doesn't actually have. What must hold regardless of grasp success: both
    arms get real work, the sim genuinely steps (time advances), the receipt
    is honest about the mode, and repeated failure drives real replans to a
    clean HOLD rather than a crash or a false "done"."""
    engine = build_intel_sim_engine()

    receipt = engine.run("set the table")

    devices = {action["device"] for action in receipt.actions if action["device"]}
    assert {"intel.left_arm", "intel.right_arm"} <= devices
    assert receipt.inputs["mode"] == "simulation-scripted-manipulation"
    assert engine.world.simulation_summary()["time"] > 0.0
    assert all(
        action["result"].get("simulation_mode") == "simulation-scripted-manipulation"
        for action in receipt.actions
        if action["op"] in {"PICK", "MOVE"}
    )
    if not receipt.metrics["resolved"]:
        assert receipt.metrics["mode"] == "HOLD"  # failed cleanly, not silently
