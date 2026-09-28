"""Adversarial coverage for bounded runtime-generated expression."""

from __future__ import annotations

import pytest

from omni_q import actions as A, build_mock_engine
from omni_q.contracts import PlanGraph, Step
from omni_q.expressive import (
    EXPRESSIVE_PRIMITIVES,
    ExpressiveWindow,
    ExpressiveWindowRejected,
    ExpressionProposal,
    attach_expression,
    generate_expression_plan,
    validate_proposal,
)
from omni_q.fakes import RulePlanner
from omni_q.scheduler import schedule
from omni_q.scheduler import ScheduledPlanner
from omni_q.world import MockWorld


def _window(**overrides) -> ExpressiveWindow:
    values = {
        "window_id": "window_1",
        "org_id": "local-demo",
        "session_id": "session_1",
        "base_world_revision": 0,
        "start_ms": 1000,
        "deadline_ms": 3800,
        "return_margin_ms": 250,
        "allowed_arms": ("left", "right"),
        "allowed_regions": ("safe_free_volume",),
        "allowed_primitives": ("ARC", "OSCILLATE", "MIRROR", "LEAD", "FOLLOW"),
        "max_amplitude": 0.15,
    }
    values.update(overrides)
    return ExpressiveWindow(**values)


def test_expression_uses_generic_primitives_not_named_gestures():
    window = _window()
    plan = generate_expression_plan(window, MockWorld.sample().state(), seed=7)

    assert plan.proposals
    assert all(p.primitive in EXPRESSIVE_PRIMITIVES for p in plan.proposals)
    assert all("BOW" not in p.primitive and "HIGH_FIVE" not in p.primitive
               for p in plan.proposals)
    assert all("object" not in p.as_args() for p in plan.proposals)
    assert plan.span_ms <= window.return_budget_ms


def test_same_window_and_seed_reproduce_policy_evidence():
    window = _window(objective_weights=(("coordination", 0.0), ("novelty", 1.0)))
    first = generate_expression_plan(window, seed=11)
    second = generate_expression_plan(window, seed=11)

    assert first.as_dict() == second.as_dict()
    assert first.window_digest == window.digest()


def test_cross_org_and_stale_world_are_rejected():
    world = MockWorld.sample().state()
    with pytest.raises(ExpressiveWindowRejected, match="organization scope"):
        generate_expression_plan(_window(org_id="other-org"), world)

    with pytest.raises(ExpressiveWindowRejected, match="revision"):
        generate_expression_plan(_window(base_world_revision=1), world)


def test_contact_requires_explicit_contact_allowance():
    window = _window(allowed_primitives=("SOFT_CONTACT",))
    proposal = ExpressionProposal(
        proposal_id="p1", window_id=window.window_id, org_id=window.org_id,
        session_id=window.session_id, base_world_revision=window.base_world_revision,
        arm="left", primitive="SOFT_CONTACT", start_ms=window.start_ms,
        duration_ms=100,
    )
    with pytest.raises(ExpressiveWindowRejected, match="contact allowance"):
        validate_proposal(window, proposal)


def test_proposal_cannot_escape_arm_region_or_return_deadline():
    window = _window()
    base = dict(
        proposal_id="p1", window_id=window.window_id, org_id=window.org_id,
        session_id=window.session_id, base_world_revision=window.base_world_revision,
        primitive="ARC", start_ms=window.start_ms, duration_ms=100,
    )
    with pytest.raises(ExpressiveWindowRejected, match="outside the window"):
        validate_proposal(window, ExpressionProposal(**base, arm="third"))
    with pytest.raises(ExpressiveWindowRejected, match="outside the window"):
        validate_proposal(window, ExpressionProposal(**base, arm="left", region_id="table"))
    late = dict(base)
    late.update(arm="left", start_ms=window.return_deadline_ms - 50, duration_ms=100)
    with pytest.raises(ExpressiveWindowRejected, match="return deadline"):
        validate_proposal(window, ExpressionProposal(**late))


def test_semantic_gesture_and_unknown_primitive_cannot_enter_window():
    with pytest.raises(ExpressiveWindowRejected, match="unsupported"):
        _window(allowed_primitives=("BOW",))

    assert A.known("EXPRESS")
    assert not A.is_style("EXPRESS")


def test_scheduler_replaces_legacy_named_flourish_path_when_window_is_explicit():
    world = MockWorld.sample().state()
    graph = PlanGraph(goal="inspect")
    graph.steps = [
        Step("work", "manipulate", "MOVE",
             args={"object": "connector_2", "to": "bin"}),
        Step("verify", "verify", "VERIFY", deps=("work",)),
    ]
    window = _window(objective_weights=(("novelty", 1.0),))
    scheduled = schedule(graph, world, expressive_window=window, expression_seed=3)

    assert scheduled.expression_plan is not None
    assert scheduled.metrics["expressions_scheduled"] >= 1
    assert scheduled.metrics["idle_flourishes"] == 0
    assert not any(ss.op in {"BOW", "HIGH_FIVE", "CALL_AND_RESPONSE"}
                   for wave in scheduled.waves for ss in wave.steps)


def test_expression_steps_are_authorized_leaf_steps_before_terminal_verify():
    world = MockWorld.sample().state()
    graph = PlanGraph(goal="tidy")
    graph.steps = [
        Step("work", "manipulate", "MOVE", args={"object": "connector_2", "to": "bin"}),
        Step("verify", "verify", "VERIFY", deps=("work",)),
    ]
    window = _window(allowed_arms=("left",), objective_weights=(("novelty", 1.0),))
    plan = generate_expression_plan(window)
    attached = attach_expression(graph, plan)

    assert attached.by_id("express_" + plan.proposals[0].proposal_id).op == "EXPRESS"
    assert all(step.op != "BOW" for step in attached.steps)
    assert attached.steps.index(next(s for s in attached.steps if s.op == "EXPRESS")) \
        < attached.steps.index(next(s for s in attached.steps if s.op == "VERIFY"))
    assert not any(
        expression.id in verify.deps
        for expression in attached.steps if expression.op == "EXPRESS"
        for verify in attached.steps if verify.op == "VERIFY"
    )


def test_scheduler_expression_execution_is_opt_in():
    world = MockWorld.sample().state()
    graph = PlanGraph(goal="inspect")
    graph.steps = [
        Step("work", "manipulate", "MOVE",
             args={"object": "connector_2", "to": "bin"}),
        Step("verify", "verify", "VERIFY", deps=("work",)),
    ]
    window = _window(objective_weights=(("novelty", 1.0),))
    scheduled = schedule(graph, world, expressive_window=window, expression_seed=4)

    display = scheduled.annotate(graph)
    execute = scheduled.annotate(graph, execute_expression=True)
    assert not any(step.op == "EXPRESS" for step in display.steps)
    assert any(step.op == "EXPRESS" for step in execute.steps)
    execute.topo_order()


def test_scheduled_planner_can_execute_the_explicit_window_path():
    world = MockWorld.sample().state()
    graph = PlanGraph(goal="inspect")
    graph.steps = [
        Step("work", "manipulate", "MOVE",
             args={"object": "connector_2", "to": "bin"}),
        Step("verify", "verify", "VERIFY", deps=("work",)),
    ]
    window = _window(allowed_arms=("left",), objective_weights=(("novelty", 1.0),))
    planner = ScheduledPlanner(
        RulePlanner(), expressive_window=window, execute_expression=True)
    scheduled = planner._apply(graph, world)

    assert any(step.op == "EXPRESS" for step in scheduled.steps)
    assert all(step.args.get("window_id") == window.window_id
               for step in scheduled.steps if step.op == "EXPRESS")


def test_engine_denies_forged_expressive_step_without_window_authority():
    engine = build_mock_engine()
    step = Step(
        "express_forged", "manipulate", "EXPRESS",
        args={
            "window_id": "not-authorized",
            "primitive": "ARC",
            "duration_ms": 100,
            "region": "safe_free_volume",
        }, arm="left",
    )
    engine._run_id = "test"
    engine.graph = PlanGraph(goal="inspect")
    authorization = engine._authorize(step, engine.world.state())

    assert authorization.verdict.value == "DENY"
    assert "authorized expressive window" in authorization.reason


def test_engine_accepts_window_bound_expression_step():
    world = MockWorld.sample()
    state = world.state()
    window = _window(allowed_arms=("left",), objective_weights=(("novelty", 1.0),))
    planner = ScheduledPlanner(
        RulePlanner(), expressive_window=window, execute_expression=True)
    plan = generate_expression_plan(window)
    step = plan.to_steps()[0]
    engine = build_mock_engine(planner=planner)
    engine.world = world
    engine._run_id = "test"
    engine.graph = PlanGraph(goal="inspect")

    authorization = engine._authorize(step, state)

    assert authorization.verdict.value == "ALLOW"


def test_expressive_window_and_seed_participate_in_run_identity():
    window = _window(allowed_arms=("left",), objective_weights=(("novelty", 1.0),))
    engine = build_mock_engine(planner=ScheduledPlanner(
        RulePlanner(), expressive_window=window, expression_seed=19))

    context = engine._run_config("inspect")["planner_context"]
    assert context["expressive_window"]["digest"] == window.digest()
    assert context["expression_seed"] == 19
