"""Adversarial fleet-fault integration tests for the governed engine loop."""

from __future__ import annotations

import pytest

from omni_q.contracts import (
    AutonomyMode,
    Detection,
    MissionEnvelope,
    PlanGraph,
    Step,
)
from omni_q.devices import default_devices
from omni_q.engine import OmniQ
from omni_q.events import validate_event_chain
from omni_q.fakes import FakeManipulator, FakeObserver, FakeRecorder, FakeVerifier, RulePlanner
from omni_q.fleet import FleetError, FleetFault, ManipulationFleet, ManipulatorSpec
from omni_q.scheduler import ScheduledPlanner
from omni_q.world import MockWorld


CAPS = frozenset({"PICK", "MOVE", "PLACE", "CO_ROTATE"})
REGIONS = frozenset({"A", "bin", "B", "tray", "setting_1"})


def _fleet() -> ManipulationFleet:
    return ManipulationFleet.from_specs(
        [
            ManipulatorSpec("left", CAPS, REGIONS),
            ManipulatorSpec("right", CAPS, REGIONS),
        ],
        preferred_pairs=(("left", "right"),),
    )


def _engine(planner, *, world: MockWorld | None = None, manipulator=None) -> OmniQ:
    world = world or MockWorld.sample()
    return OmniQ(
        world=world,
        observer=FakeObserver(),
        planner=planner,
        manipulator=manipulator or FakeManipulator(world),
        verifier=FakeVerifier(),
        device=default_devices(),
        recorder=FakeRecorder(),
        envelope=MissionEnvelope(mission_id="fleet-test"),
    )


def _fault(*, fault_id: str = "fault-1", org_id: str = "local-demo") -> FleetFault:
    return FleetFault(
        fault_id,
        ("left",),
        "left arm health unavailable",
        1234,
        org_id=org_id,
    )


class _RecordingManipulator(FakeManipulator):
    def __init__(self, world):
        super().__init__(world)
        self.calls: list[tuple[str, str | None]] = []

    def execute(self, step, world):
        self.calls.append((step.id, step.arm))
        return super().execute(step, world)


class _ReportAfterMoveVerifier(FakeVerifier):
    def __init__(self, engine: OmniQ):
        self.engine = engine
        self.reported = False

    def check(self, step, observation):
        result = super().check(step, observation)
        if (
            not self.reported
            and step.op == "MOVE"
            and step.args.get("object") == "connector_2"
        ):
            self.reported = True
            self.engine.report_fleet_fault(_fault())
        return result


def test_mid_run_fault_replans_from_current_world_and_never_reaches_new_calls_on_failed_arm():
    fleet = _fleet()
    planner = ScheduledPlanner(RulePlanner(), fleet=fleet)
    manipulator = _RecordingManipulator(MockWorld.sample())
    engine = _engine(planner, manipulator=manipulator)
    engine.verifier = _ReportAfterMoveVerifier(engine)

    receipt = engine.run("inspect and correct the workspace")

    assert receipt.metrics["resolved"] is True
    assert receipt.metrics["revisions"] >= 1
    assert planner.fleet is not fleet
    assert planner.fleet is not None
    assert planner.fleet.spec("left").online is False

    observed = next(e for e in engine.bus.log if e.kind == "fleet.fault.observed")
    reallocated = next(e for e in engine.bus.log if e.kind == "fleet.reallocated")
    recompiles = [e for e in engine.bus.log if e.kind == "graph.recompiled"]
    assert recompiles
    recompiled = recompiles[-1]
    assert observed.state_revision == reallocated.state_revision == recompiled.state_revision
    assert recompiled.graph_revision > 0
    assert engine.mode is AutonomyMode.DEGRADED
    assert engine.graph is not None
    assert all(
        step.arm != "left"
        for step in engine.graph.steps
        if step.contract == "manipulate" and step.state == "pending"
    )

    move_index = next(
        index for index, (step_id, _arm) in enumerate(manipulator.calls)
        if step_id == "move_connector_2"
    )
    assert all(arm != "left" for _step_id, arm in manipulator.calls[move_index + 1:])

    assert reallocated.data == next(
        decision for decision in receipt.decisions
        if decision.get("kind") == "fleet.reallocation"
    )
    validate_event_chain(engine.bus.log)


def test_fault_queue_has_no_immediate_fleet_graph_or_hardware_side_effect():
    fleet = _fleet()
    planner = ScheduledPlanner(RulePlanner(), fleet=fleet)
    world = MockWorld.sample()
    manipulator = _RecordingManipulator(world)
    engine = _engine(planner, world=world, manipulator=manipulator)

    engine.report_fleet_fault(_fault())

    assert planner.fleet is fleet
    assert engine.graph is None
    assert manipulator.calls == []
    assert engine.bus.log == []
    assert world.revision == 0


def test_cross_org_fault_is_rejected_without_fleet_or_graph_mutation():
    fleet = _fleet()
    planner = ScheduledPlanner(RulePlanner(), fleet=fleet)
    engine = _engine(planner)
    engine.report_fleet_fault(_fault(org_id="other-org"))

    receipt = engine.run("inspect and correct the workspace")

    assert receipt.metrics["resolved"] is True
    assert planner.fleet is fleet
    assert engine.graph is not None and engine.graph.revision == 0
    assert not [e for e in engine.bus.log if e.kind == "fleet.reallocated"]
    assert not [e for e in engine.bus.log if e.kind == "graph.recompiled"]
    rejection = next(e for e in engine.bus.log if e.kind == "fleet.fault.rejected")
    assert rejection.data["world_org_id"] == "local-demo"
    validate_event_chain(engine.bus.log)


def test_duplicate_fault_is_rejected_and_reallocated_exactly_once():
    planner = ScheduledPlanner(RulePlanner(), fleet=_fleet())
    engine = _engine(planner)
    engine.report_fleet_fault(_fault())
    engine.report_fleet_fault(_fault())

    receipt = engine.run("inspect and correct the workspace")

    assert receipt.metrics["resolved"] is True
    assert len([e for e in engine.bus.log if e.kind == "fleet.reallocated"]) == 1
    assert len([
        e for e in engine.bus.log
        if e.kind == "fleet.fault.rejected"
        and "duplicate" in e.data["reason"]
    ]) == 1
    validate_event_chain(engine.bus.log)


class _BimanualPlanner:
    last_decision = None

    def plan(self, goal, world):
        return PlanGraph(goal=goal, steps=[
            Step("co_rotate", "manipulate", "CO_ROTATE", args={"object": "plate_1"}),
            Step("verify", "verify", "VERIFY", deps=("co_rotate",)),
        ])

    def replan(self, current, world, reason):
        return PlanGraph(goal=current.goal, steps=list(current.steps), revision=current.revision + 1)


def test_unresolvable_fleet_replan_enters_hold_without_executing_stale_graph():
    planner = ScheduledPlanner(_BimanualPlanner(), fleet=_fleet())
    manipulator = _RecordingManipulator(MockWorld.sample())
    engine = _engine(planner, manipulator=manipulator)
    engine.report_fleet_fault(_fault())

    receipt = engine.run("coordinate the table")

    assert receipt.metrics["mode"] == AutonomyMode.HOLD.value
    assert receipt.metrics["steps_executed"] == 0
    assert manipulator.calls == []
    assert planner.last_schedule is None
    assert not [e for e in engine.bus.log if e.kind == "graph.recompiled"]
    assert any(e.kind == "fleet.replan.failed" for e in engine.bus.log)
    validate_event_chain(engine.bus.log)


def test_planner_without_fleet_seam_fails_closed_instead_of_reusing_old_graph():
    world = MockWorld.sample()
    manipulator = _RecordingManipulator(world)
    engine = _engine(RulePlanner(), world=world, manipulator=manipulator)
    engine.report_fleet_fault(FleetFault("fault-1", ("left",), "health unavailable", 5))

    receipt = engine.run("inspect and correct the workspace")

    assert receipt.metrics["mode"] == AutonomyMode.HOLD.value
    assert receipt.metrics["steps_executed"] == 0
    assert manipulator.calls == []
    assert not [e for e in engine.bus.log if e.kind == "fleet.reallocated"]
    assert any(e.kind == "fleet.replan.failed" for e in engine.bus.log)
    validate_event_chain(engine.bus.log)


def test_scheduled_planner_run_context_keeps_initial_fleet_digest_and_latest_reallocation_metadata():
    fleet = _fleet()
    planner = ScheduledPlanner(RulePlanner(), fleet=fleet)
    initial = planner.run_context()

    updated, evidence = planner.reallocate_fleet(_fault())
    context = planner.run_context()

    assert initial["fleet_digest"] == fleet.digest()
    assert context["fleet_digest"] == fleet.digest()
    assert context["current_fleet_digest"] == updated.digest()
    assert context["last_reallocation"] == evidence.as_dict()


@pytest.mark.parametrize(
    "case",
    [
        {"source": ""},
        {"source": "   "},
        {"org_id": ""},
        {"observed_ms": -1},
        {"observed_ms": 1.5},
    ],
)
def test_fault_contract_rejects_invalid_provenance_and_timestamp(case):
    values = {
        "fault_id": "fault-1",
        "resource_ids": ("left",),
        "reason": "health unavailable",
        "observed_ms": 1,
    }
    values.update(case)
    with pytest.raises(FleetError):
        FleetFault(**values)


def test_malformed_reallocation_requests_are_rejected_before_queueing():
    engine = _engine(ScheduledPlanner(RulePlanner(), fleet=_fleet()))

    with pytest.raises(FleetError, match="ReallocationRequest"):
        engine.report_fleet_fault(_fault(), requests=({"lease_id": "bad"},))
    assert engine._pending_fleet_faults == []


def test_event_order_and_parent_chain_are_valid_for_a_successful_queued_fault():
    planner = ScheduledPlanner(RulePlanner(), fleet=_fleet())
    engine = _engine(planner)
    engine.report_fleet_fault(_fault())
    engine.run("inspect and correct the workspace")

    kinds = [event.kind for event in engine.bus.log]
    observed_index = kinds.index("fleet.fault.observed")
    reallocated_index = kinds.index("fleet.reallocated")
    recompiled_index = kinds.index("graph.recompiled")
    assert observed_index < reallocated_index < recompiled_index
    assert engine.bus.log[reallocated_index].parent_id == engine.bus.log[reallocated_index - 1].event_id
    validate_event_chain(engine.bus.log)


def test_legacy_omniq_run_without_fleet_remains_unchanged():
    engine = _engine(RulePlanner())

    receipt = engine.run("inspect and correct the workspace")

    assert receipt.metrics["resolved"] is True
    assert receipt.metrics["revisions"] == 0
    assert not [event for event in engine.bus.log if event.kind.startswith("fleet.")]
