"""Real MuJoCo Intel sim + bimanual scheduler, wired together zero-touch.

Skipped in the pure-Python core environment; run with the project venv after
installing the ``intel`` extra (``pip install -e ".[dev,intel]"``).
"""

from __future__ import annotations

import pytest


mujoco = pytest.importorskip("mujoco")

from omni_q.intel_sim import build_intel_sim_engine
from omni_q.scheduler import ScheduledPlanner


def test_scheduled_planner_wraps_intel_planner_and_still_steps_real_physics():
    engine = build_intel_sim_engine()
    scheduled = ScheduledPlanner(engine.planner)
    engine.planner = scheduled

    receipt = engine.run("set the table")

    assert scheduled.last_error is None
    assert scheduled.last_schedule is not None
    arms_used = {arm for arm in scheduled.last_schedule.assignment.values() if arm}
    assert arms_used == {"left", "right"}
    assert receipt.metrics["resolved"] is True
    assert engine.world.simulation_summary()["time"] > 0.0
