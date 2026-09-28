"""Intel-suite hook: OmniPlanner composed over the real MuJoCo engine.

Mirrors demo_intel_reasoner.py's composition (ScheduledPlanner wrap, no
edits to intel_sim.py) and verifies the reasoner advice channel survives
contact with the Intel table world: decisions carry backend labels, MuJoCo
executes the validated graph, and rationale events reach the bus.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco", reason="Intel MuJoCo stack not installed")

from omni_q.events import EventBus
from omni_q.intel_sim import IntelSimulationUnavailable, build_intel_sim_engine
from omni_q.omni_planner import OmniPlanner
from omni_q.omni_reasoner import MockReasoner
from omni_q.scheduler import ScheduledPlanner

GOAL = "set the table"


def _engine(reasoner_text=None):
    try:
        engine = build_intel_sim_engine()
    except IntelSimulationUnavailable:
        pytest.skip("Intel MuJoCo stack not installed")
    scheduled = ScheduledPlanner(engine.planner)
    planner = OmniPlanner(MockReasoner(reasoner_text), fallback=scheduled)
    engine.planner = planner
    return engine, planner, scheduled


def test_reasoner_advises_intel_suite_with_truthful_labels():
    engine, planner, _ = _engine()  # default rationale-only -> validated fallback
    reasons = []
    engine.bus.subscribe(lambda e: reasons.append(e.data["reason"])
                         if e.kind == "plan.decision" and e.data.get("reason")
                         else None)
    receipt = engine.run(GOAL)
    assert receipt.decisions
    assert any("[mock]" in d["reason"] for d in receipt.decisions)
    assert any("fallback" in d["reason"] for d in receipt.decisions)
    assert any("[mock]" in r for r in reasons), "rationale reached the event bus (UI channel)"


def test_intel_world_prompt_lists_intel_objects():
    engine, planner, _ = _engine()
    planner.reasoner.calls.clear()
    engine.run(GOAL)
    assert planner.reasoner.calls, "reasoner was consulted"
    _, prompt = planner.reasoner.calls[0]
    assert "goal: set the table" in prompt
    for token in ("objects:", "known_zones:", "constraints:"):
        assert token in prompt
