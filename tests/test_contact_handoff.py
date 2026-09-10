"""OQ-010/OQ-011 contact-handoff gates and adversarial boundaries."""

from __future__ import annotations

import inspect
import json
import shutil
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest


mujoco = pytest.importorskip("mujoco")

from omni_q.contracts import (  # noqa: E402
    AuthorizationVerdict,
    DataStatus,
    Detection,
    MissionEnvelope,
    ConstraintValidationError,
    Step,
    TransitionRejected,
    TransitionRequest,
)
from omni_q.intel_sim import (  # noqa: E402
    CONTACT_HANDOFF_MODE,
    ContactHandoffConfig,
    ContactHandoffRejected,
    IntelContactHandoffWorld,
    IntelContactHandoffManipulator,
    _ContactHandoffController,
    build_intel_contact_handoff_engine,
    contact_handoff_xml,
    run_contact_handoff,
    run_randomized_contact_handoff_report,
    verify_contact_handoff_receipt,
    write_contact_handoff_receipt,
)


def test_contact_scene_is_named_and_has_no_attachment_or_scripted_route():
    xml = contact_handoff_xml()
    assert "cup_1_contact" in xml
    assert 'name="handoff_floor"' in xml
    assert "left_fixed_jaw_pad_4" in xml
    assert "right_fixed_jaw_pad_4" in xml
    assert "<equality" not in xml
    assert "weld" not in xml.lower()
    assert "simulation-scripted-manipulation" not in xml


def test_deterministic_contact_gate_is_ten_of_ten():
    outcomes = [run_contact_handoff(ContactHandoffConfig(seed=19)) for _ in range(10)]
    assert all(receipt.success for receipt in outcomes)
    assert all(receipt.failure_reason is None for receipt in outcomes)
    assert all(receipt.final_stable for receipt in outcomes)
    assert all(receipt.final_owner is None for receipt in outcomes)
    assert all(receipt.mode == CONTACT_HANDOFF_MODE for receipt in outcomes)
    assert all(receipt.contact_transitions for receipt in outcomes)


def test_governed_engine_admits_only_verified_contact_transition():
    engine = build_intel_contact_handoff_engine(ContactHandoffConfig(seed=19))
    receipt = engine.run("transfer cup_1 from left arm to right arm")

    assert receipt.inputs["mode"] == CONTACT_HANDOFF_MODE
    assert engine.world.state().objects["cup_1"].zone == "right_table"
    assert engine.world.state().ownership["cup_1"] is None
    assert len(engine.world.contact_events) == 1
    assert engine.world.contact_events[0]["kind"] == "contact_handoff.admitted"
    physical = next(action for action in receipt.actions if action["op"] == "HANDOFF")
    assert physical["result"]["contact_handoff"]["success"] is True


def test_legacy_scripted_engine_never_emits_contact_handoff():
    from omni_q.intel_sim import build_intel_sim_engine

    engine = build_intel_sim_engine()
    receipt = engine.run("set the table")
    assert receipt.inputs["mode"] == "simulation-scripted-manipulation"
    assert all(
        action["result"].get("simulation_mode") != CONTACT_HANDOFF_MODE
        for action in receipt.actions
    )


def test_controller_does_not_write_cup_freejoint_or_use_attachment():
    source = inspect.getsource(_ContactHandoffController)
    assert "cup_1_free" not in source
    assert "<equality" not in source.lower()
    # The only qpos writes are arm-joint slices used by IK/reset; no fixed
    # free-joint address or body-pose assignment is present.
    assert "qpos[offset:offset + 5]" in source
    assert "qpos[12" not in source


def test_missing_contact_blocks_ownership_admission():
    world = IntelContactHandoffWorld()
    controller = _ContactHandoffController(world)
    with pytest.raises(ContactHandoffRejected, match="missing contact"):
        controller._require_owner("left", "missing_contact")


def test_premature_release_without_right_owner_is_blocked():
    world = IntelContactHandoffWorld()
    controller = _ContactHandoffController(world)
    controller._ownership = lambda: None  # type: ignore[method-assign]
    with pytest.raises(ContactHandoffRejected, match="missing contact"):
        controller._require_owner("right", "premature_release")


def test_dual_gripper_ownership_ambiguity_is_blocked():
    world = IntelContactHandoffWorld()
    controller = _ContactHandoffController(world)
    controller._ownership = lambda: "dual"  # type: ignore[method-assign]
    with pytest.raises(ContactHandoffRejected, match="ambiguous ownership"):
        controller._require_owner("right", "left_release")


def test_controller_timeout_is_recorded_and_does_not_mutate_authoritative_state():
    world = IntelContactHandoffWorld(ContactHandoffConfig(motion_timeout_steps=0))
    before = world.state().objects["cup_1"]
    receipt = _ContactHandoffController(world).run(world.revision)
    assert receipt.success is False
    assert "timeout" in (receipt.failure_reason or "")
    assert world.state().objects["cup_1"] == before
    assert world.revision == 0


def test_failed_engine_attempt_is_not_reentered_on_dirty_physics_state():
    engine = build_intel_contact_handoff_engine(
        ContactHandoffConfig(motion_timeout_steps=0)
    )
    receipt = engine.run("transfer cup_1 from left arm to right arm")

    handoff_actions = [a for a in receipt.actions if a["op"] == "HANDOFF"]
    assert len(handoff_actions) == 1
    assert handoff_actions[0]["state"] == "failed"
    assert not receipt.metrics["resolved"]
    assert engine.world.pending_contact_receipt is not None
    assert engine.planner.last_decision is not None
    assert any("fresh contact world" in item["reason"] for item in receipt.rejected)


def test_manipulator_rejects_direct_reentry_without_advancing_mujoco():
    world = IntelContactHandoffWorld(ContactHandoffConfig(motion_timeout_steps=0))
    manipulator = IntelContactHandoffManipulator(world)
    step = Step(
        "one_shot", "manipulate", "HANDOFF",
        args={"object": "cup_1", "to_actor": "intel.right_arm"},
        arm="left",
    )
    first = manipulator.execute(step, world.state())
    before_time = float(world.data.time)

    second = manipulator.execute(step, world.state())

    assert not first.ok
    assert not second.ok
    assert "one-shot" in second.detail["error"]
    assert second.detail["contact_handoff"]["content_hash"] == (
        first.detail["contact_handoff"]["content_hash"]
    )
    assert float(world.data.time) == pytest.approx(before_time)


def test_successful_engine_cannot_be_reused_for_a_second_physical_handoff():
    engine = build_intel_contact_handoff_engine(ContactHandoffConfig(seed=19))
    first = engine.run("transfer cup_1 from left arm to right arm")
    assert first.metrics["resolved"]
    assert len(engine.world.contact_events) == 1
    before_events = list(engine.world.contact_events)

    second = engine.run("transfer cup_1 from left arm to right arm")

    assert second.metrics["resolved"]
    assert not second.actions
    assert engine.world.contact_events == before_events


def test_unsafe_shared_workspace_entry_fails_closed():
    world = IntelContactHandoffWorld(ContactHandoffConfig(shared_workspace_x_limit_m=0.001))
    receipt = _ContactHandoffController(world).run(world.revision)
    assert receipt.success is False
    assert receipt.failure_reason == "unsafe shared-workspace entry"
    assert world.state().objects["cup_1"].zone == "left_table"


def test_stale_fallback_data_cannot_plan_a_contact_handoff():
    engine = build_intel_contact_handoff_engine()
    engine.world._objects["cup_1"] = replace(
        engine.world._objects["cup_1"], status=DataStatus.FALLBACK
    )
    receipt = engine.run("transfer cup_1 from left arm to right arm")
    assert not receipt.actions
    assert engine.world.state().objects["cup_1"].zone == "left_table"
    assert engine.planner.last_decision is not None
    assert engine.planner.last_decision.candidates_feasible == 0


def test_wrong_role_tier_and_cross_org_are_denied_before_driver():
    wrong_role = build_intel_contact_handoff_engine()
    wrong_role.envelope = MissionEnvelope(
        mission_id="wrong-role", permitted_ops=("PICK",), max_revisions=0
    )
    denied = wrong_role.run("transfer cup_1 from left arm to right arm")
    assert denied.actions[0]["state"] == "denied"
    assert wrong_role.recorder.authorizations[0].verdict is AuthorizationVerdict.DENY

    cross_org = build_intel_contact_handoff_engine()
    cross_org.envelope = MissionEnvelope(
        mission_id="cross-org", org_id="other-org", max_revisions=0
    )
    denied_scope = cross_org.run("transfer cup_1 from left arm to right arm")
    assert denied_scope.actions[0]["state"] == "denied"
    assert "organization scope" in cross_org.recorder.authorizations[0].reason


def test_missing_operator_justification_is_blocked_on_contact_engine():
    engine = build_intel_contact_handoff_engine()
    with pytest.raises(ConstraintValidationError, match="requires justification"):
        engine.add_constraint("keep_local")


def test_invalid_transition_and_unaudited_background_mutation_are_rejected():
    world = IntelContactHandoffWorld()
    with pytest.raises(TransitionRejected, match="only admits HANDOFF"):
        world.apply_transition(TransitionRequest(
            step_id="bad", op="PICK", args={"object": "cup_1"},
            expected_revision=world.revision, actor="intel.left_arm", org_id=world.org_id,
        ))
    with pytest.raises(TransitionRejected, match="recorded contact domain event"):
        world.move_object("cup_1", "right_table")


def test_stale_state_revision_is_rejected_after_physics_proposal():
    world = IntelContactHandoffWorld()
    receipt = _ContactHandoffController(world).run(world.revision)
    world.pending_contact_receipt = receipt
    world.start_mission("revision changes before admission")
    with pytest.raises(TransitionRejected, match="expected revision"):
        world.apply_transition(TransitionRequest(
            step_id="stale", op="HANDOFF",
            args={"object": "cup_1", "to_actor": "intel.right_arm"},
            expected_revision=0, actor="intel.left_arm", org_id=world.org_id,
        ))


def test_receipt_tampering_is_rejected():
    receipt = run_contact_handoff(ContactHandoffConfig(seed=19))
    tampered = receipt.as_dict()
    tampered["final_cup_pose"]["position"][0] += 0.5
    with pytest.raises(ContactHandoffRejected, match="content hash mismatch"):
        verify_contact_handoff_receipt(tampered)


def test_receipt_writer_refuses_to_overwrite_existing_evidence():
    root = Path("tmp") / f"contact-receipt-{uuid4().hex}"
    root.mkdir(parents=True)
    path = root / "receipt.json"
    try:
        receipt = run_contact_handoff(ContactHandoffConfig(seed=19))
        path.write_text("operator-preserved-evidence\n", encoding="utf-8")
        with pytest.raises(ContactHandoffRejected, match="refusing to overwrite"):
            write_contact_handoff_receipt(receipt, path)
        assert path.read_text(encoding="utf-8") == "operator-preserved-evidence\n"
    finally:
        shutil.rmtree(root)


def test_randomized_report_retains_all_twenty_receipts():
    # Keep the artifact under the repository-owned tmp directory; some locked
    # Windows runners deny pytest's global temp root.
    root = Path("tmp") / f"contact-report-{uuid4().hex}"
    try:
        report = run_randomized_contact_handoff_report(root, trials=20, seed=700)
        assert report["mode"] == CONTACT_HANDOFF_MODE
        assert report["kind"].endswith("not a promotion claim")
        assert len(report["receipts"]) == 20
        assert sum(report["outcomes"].values()) == 20
        receipt_files = sorted(root.glob("trial-*.json"))
        assert len(receipt_files) == 20
        for path in receipt_files:
            persisted = json.loads(path.read_text(encoding="utf-8"))
            verify_contact_handoff_receipt(persisted)
    finally:
        shutil.rmtree(root)
