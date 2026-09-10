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


def test_distinct_object_zones_let_both_arms_work_concurrently():
    """OQ-007 regression: a shared literal zone for every tableware item
    used to collapse the schedule to max_parallelism=1 regardless of arm
    (see integrations/intel/README.md). Distinct start zones + real region
    x-centers (scheduler.DEFAULT_LAYOUT) should let left/right run waves
    together with no false workspace conflicts."""
    engine = build_intel_sim_engine()
    scheduled = ScheduledPlanner(engine.planner)
    engine.planner = scheduled

    engine.run("set the table")

    sch = scheduled.last_schedule
    assert sch is not None
    assert sch.metrics["max_parallelism"] >= 2
    assert sch.metrics["serialized_conflicts"] == 0
    assert not any(b.kind == "reach" for b in sch.barriers)
