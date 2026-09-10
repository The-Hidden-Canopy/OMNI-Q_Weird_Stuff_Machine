"""Adversarial boundaries for the governed Omni Q execution path."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from omni_q import build_mock_engine
from omni_q.contracts import (
    AuthorizationVerdict,
    ConstraintValidationError,
    DataStatus,
    Detection,
    MissionEnvelope,
    PlanGraph,
    Step,
    TransitionRejected,
    TransitionRequest,
)
from omni_q.devices import default_devices
from omni_q.evidence import EvidenceLedger
from omni_q.engine import OmniQ
from omni_q.events import validate_event_chain
from omni_q.fakes import FakeManipulator, FakeObserver, FakeRecorder, FakeVerifier, RulePlanner
from omni_q.world import MockWorld


class InjectedPlanner:
    """A deliberately hostile planner that ignores every stated restriction."""

    last_decision = None

    def plan(self, goal, world):
        return PlanGraph(goal=goal, steps=[
            Step(
                "injected_move",
                "manipulate",
                "MOVE",
                args={"object": "connector_2", "to": "bin"},
                rationale="untrusted planner injection",
            )
        ])

    def replan(self, current, world, reason):
        return PlanGraph(goal=current.goal, steps=current.steps, revision=current.revision + 1)


def _engine(world: MockWorld, *, envelope: MissionEnvelope | None = None, recorder=None) -> OmniQ:
    return OmniQ(
        world=world,
        observer=FakeObserver(),
        planner=InjectedPlanner(),
        manipulator=FakeManipulator(world),
        verifier=FakeVerifier(),
        device=default_devices(),
        recorder=recorder or FakeRecorder(),
        envelope=envelope or MissionEnvelope(mission_id="adversarial", max_revisions=0),
    )


def test_world_rejects_a_stale_transition_revision():
    world = MockWorld.sample()
    before = world.state()
    request = TransitionRequest(
        step_id="pick", op="PICK", args={"object": "connector_2"},
        expected_revision=before.revision, actor="arduino.left_arm", org_id=before.org_id,
    )
    result = world.apply_transition(request)
    assert result.state_revision == before.revision + 1

    with pytest.raises(TransitionRejected, match="expected revision"):
        world.apply_transition(request)


def test_executor_cannot_mutate_the_world_without_a_transition():
    world = MockWorld.sample()
    before = world.state()
    result = FakeManipulator(world).execute(
        Step("move", "manipulate", "MOVE", args={"object": "connector_2", "to": "bin"}),
        before,
    )

    assert result.ok is True
    assert world.state().objects["connector_2"].zone == "A"
    assert world.state().revision == before.revision


def test_cross_org_request_is_blocked_before_any_mutation():
    world = MockWorld.sample()
    engine = _engine(world, envelope=MissionEnvelope(
        mission_id="scope", org_id="other-org", max_revisions=0,
    ))

    receipt = engine.run("inspect and correct the workspace")

    assert world.state().objects["connector_2"].zone == "A"
    assert receipt.actions[0]["state"] == "denied"
    assert engine.recorder.authorizations[0].verdict is AuthorizationVerdict.DENY
    assert "organization scope" in engine.recorder.authorizations[0].reason


def test_envelope_blocks_an_untrusted_planner_from_moving_a_forbidden_object():
    world = MockWorld.sample()
    engine = _engine(world, envelope=MissionEnvelope(
        mission_id="restricted", forbidden_objects=("connector_2",), max_revisions=0,
    ))

    receipt = engine.run("inspect and correct the workspace")

    assert world.state().objects["connector_2"].zone == "A"
    assert receipt.actions[0]["state"] == "denied"
    assert engine.recorder.authorizations[0].verdict is AuthorizationVerdict.DENY
    assert "forbidden" in engine.recorder.authorizations[0].reason


def test_missing_justification_and_ai_constraint_mutation_are_blocked():
    engine = build_mock_engine()
    with pytest.raises(ConstraintValidationError, match="requires justification"):
        engine.add_constraint("keep_local")
    with pytest.raises(ConstraintValidationError, match="only an operator"):
        engine.add_constraint(
            "keep_local", source="ai", justification="model suggested local routing",
        )


def test_fallback_observation_cannot_be_authorized_by_an_untrusted_planner():
    world = MockWorld.sample()
    world._objects["connector_2"] = Detection(
        "connector_2", "connector", zone="A", target_zone="bin",
        status=DataStatus.FALLBACK,
    )
    engine = _engine(world)

    engine.run("inspect and correct the workspace")

    assert world.state().objects["connector_2"].zone == "A"
    assert engine.recorder.authorizations[0].verdict is AuthorizationVerdict.REQUIRE_APPROVAL


def test_each_manipulation_is_authorized_then_reobserved_and_verified():
    engine = build_mock_engine()
    engine.run("inspect and correct the workspace")

    events = engine.bus.log
    validate_event_chain(events)
    manip_finished = [
        event for event in events
        if event.kind == "step.finished" and event.data["contract"] == "manipulate"
    ]
    verified = [event for event in events if event.kind == "verified"]
    assert len(verified) >= len(manip_finished)

    for finished in manip_finished:
        step_id = finished.data["id"]
        authorized = [
            event for event in events
            if event.kind == "step.authorized"
            and event.data["authorization"]["step_id"] == step_id
        ]
        assert authorized and authorized[0].seq < finished.seq


def test_authorization_receipt_failure_fails_closed_before_driver_execution():
    class FailingRecorder(FakeRecorder):
        def authorize(self, authorization):
            raise OSError("ledger unavailable")

    world = MockWorld.sample()
    recorder = FailingRecorder()
    engine = _engine(world, recorder=recorder)
    calls: list[str] = []
    original_execute = engine.manipulator.execute

    def watched_execute(step, state):
        calls.append(step.id)
        return original_execute(step, state)

    engine.manipulator.execute = watched_execute
    receipt = engine.run("inspect and correct the workspace")

    assert calls == []
    assert receipt.actions[0]["state"] == "denied"
    assert "authorization.failed" in [event.kind for event in engine.bus.log]


def test_durable_ledger_writes_authorizations_before_a_parent_chained_receipt():
    # Some locked-down Windows runners deny pytest's global temp root.  Keep
    # this isolated under the repository-owned receipt directory instead.
    root = Path("evidence") / "receipts" / f"test-{uuid4().hex}"
    world = MockWorld.sample()
    ledger = EvidenceLedger(root)
    engine = OmniQ(
        world=world,
        observer=FakeObserver(),
        planner=RulePlanner(),
        manipulator=FakeManipulator(world),
        verifier=FakeVerifier(),
        device=default_devices(),
        recorder=ledger,
    )

    try:
        receipt = engine.run("inspect and correct the workspace")

        run_dir = root / "runs" / receipt.run_id
        assert (run_dir / "authorizations.jsonl").exists()
        assert (run_dir / "receipt.json").exists()
        manifest = (root / "manifest.jsonl").read_text(encoding="utf-8")
        assert receipt.content_hash in manifest
        assert receipt.inputs["mode"] == "mock"
    finally:
        shutil.rmtree(root)
