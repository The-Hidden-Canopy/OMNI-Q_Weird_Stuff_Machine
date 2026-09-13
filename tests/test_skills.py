from __future__ import annotations

from dataclasses import replace

import pytest

from omni_q import (
    ActionAuthorization,
    AuthorizationVerdict,
    ControllerType,
    PromotionEvidence,
    ProposedAction,
    SkillAuthority,
    SkillControl,
    SkillExecutionStatus,
    SkillManifest,
    SkillObservation,
    SkillRegistry,
    SkillRegistryError,
    SkillRequest,
    SkillRuntime,
    SkillStage,
    SkillSupervisor,
    SkillUnavailable,
    SupervisorDecision,
)


ARTIFACT = "sha256:skill-grasp-v1"


def manifest(*, skill_id: str = "right_arm.rl_grasp.v1",
             stage: SkillStage = SkillStage.TRAINED) -> SkillManifest:
    return SkillManifest(
        skill_id=skill_id,
        org_id="org_a",
        capability="grasp",
        controller_type=ControllerType.RESIDUAL_RL,
        artifact_digest=ARTIFACT,
        arm_ids=("arm_1",),
        input_fields=("object_pose", "tcp_pose", "contact_force"),
        preconditions=("rigid_object", "reachable"),
        control=SkillControl(
            max_delta_mm=4.0,
            max_rotation_deg=3.0,
            max_gripper_delta=0.05,
            workspace_mm=(
                ("x", -10.0, 10.0),
                ("y", -10.0, 10.0),
                ("z", 0.0, 20.0),
            ),
            max_speed_mm_s=100.0,
            max_contact_force_n=20.0,
            residual_scale_max=0.15,
        ),
        termination=("grasp_verified", "timeout_ms=5000"),
        fallback_skill_id="right_arm.ik_grasp.v1",
        verification=("vision", "contact"),
        authority=SkillAuthority(may_move_arm=True),
        stage=stage,
    )


def authorization(revision: int = 3, *, org_id: str = "org_a") -> ActionAuthorization:
    return ActionAuthorization(
        run_id="run_1",
        step_id="step_1",
        op="PICK",
        verdict=AuthorizationVerdict.ALLOW,
        reason="mission envelope allows PICK",
        state_revision=revision,
        envelope_digest="sha256:envelope",
    )


def request(*, skill_id: str | None = None, revision: int = 3,
            org_id: str = "org_a") -> SkillRequest:
    return SkillRequest(
        request_id="request_1",
        org_id=org_id,
        capability="grasp",
        operation="PICK",
        arm_id="arm_1",
        expected_world_revision=revision,
        authorization=authorization(revision, org_id=org_id),
        skill_id=skill_id,
    )


def observation(*, revision: int = 3, force: float = 0.0,
                safety_stop: bool = False) -> SkillObservation:
    return SkillObservation(
        org_id="org_a",
        arm_id="arm_1",
        world_revision=revision,
        tcp_position_mm=(0.0, 0.0, 5.0),
        contact_force_n=force,
        safety_stop=safety_stop,
    )


def active_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(manifest())
    evidence = PromotionEvidence(
        receipt_id="promotion-1",
        artifact_digest=ARTIFACT,
        checks=(("passed", True),),
    )
    for _ in range(6):
        registry.promote("right_arm.rl_grasp.v1", evidence)
    assert registry.get("right_arm.rl_grasp.v1").stage is SkillStage.ACTIVE
    return registry


class ResidualController:
    backend = "sac-grasp-v1"

    def __init__(self, skill_id: str = "right_arm.rl_grasp.v1") -> None:
        self.skill_id = skill_id
        self.calls = 0

    def propose(self, request, _observation):
        self.calls += 1
        return ProposedAction.from_mapping(
            request.request_id,
            self.skill_id,
            {"dx_mm": 10.0, "dy_mm": 1.0},
            self.backend,
            residual_scale=0.15,
        )


class RecordingActuator:
    def __init__(self) -> None:
        self.calls: list[dict[str, float]] = []

    def apply(self, action, _observation):
        self.calls.append(dict(action))
        return {"applied": True}


def test_skill_cannot_become_active_without_adjacent_promotion_evidence() -> None:
    registry = SkillRegistry()
    registry.register(manifest())

    with pytest.raises(SkillRegistryError):
        registry.register(replace(manifest(), stage=SkillStage.ACTIVE))

    registry.promote(
        "right_arm.rl_grasp.v1",
        PromotionEvidence("sim", ARTIFACT, (("sim", True),)),
    )
    assert registry.get("right_arm.rl_grasp.v1").stage is SkillStage.SIM_EVAL
    with pytest.raises(SkillUnavailable):
        registry.select(request())


def test_active_skill_selection_is_org_and_arm_scoped() -> None:
    registry = active_registry()

    selected = registry.select(request())
    assert selected.manifest.skill_id == "right_arm.rl_grasp.v1"
    assert selected.fallback_skill_id == "right_arm.ik_grasp.v1"

    with pytest.raises(SkillUnavailable, match="organization"):
        registry.select(request(org_id="org_b"))

    with pytest.raises(SkillUnavailable, match="arm"):
        registry.select(replace(request(), arm_id="arm_2"))


def test_residual_policy_is_scaled_and_clamped_before_actuation() -> None:
    registry = active_registry()
    actuator = RecordingActuator()
    result = SkillRuntime(registry).step(
        request(),
        observation(),
        ResidualController(),
        actuator,
    )

    assert result.status is SkillExecutionStatus.EXECUTED
    assert result.supervisor is not None
    assert result.supervisor.decision is SupervisorDecision.CLAMP
    assert result.supervisor.safe_action == (("dx_mm", 1.5), ("dy_mm", 0.15))
    assert actuator.calls == [{"dx_mm": 1.5, "dy_mm": 0.15}]


def test_controller_cannot_select_another_skill_or_bypass_supervisor() -> None:
    registry = active_registry()
    actuator = RecordingActuator()
    controller = ResidualController(skill_id="unregistered-or-other-skill")

    result = SkillRuntime(registry).step(
        request(), observation(), controller, actuator,
    )

    assert result.status is SkillExecutionStatus.DENIED
    assert result.supervisor is not None
    assert "another skill" in result.supervisor.reason
    assert actuator.calls == []


def test_workspace_violation_is_denied_and_force_or_stop_is_stopped() -> None:
    registry = active_registry()
    actuator = RecordingActuator()
    runtime = SkillRuntime(registry)

    # x=0 plus 1.5 is safe, but z=5 plus no z action remains in bounds. Use a
    # direct proposal to exercise the envelope without involving controller IO.
    selected = registry.select(request()).manifest
    workspace = runtime.supervisor.evaluate(
        request(),
        selected,
        observation(),
        ProposedAction.from_mapping(
            "request_1", selected.skill_id,
            {"dz_mm": 200.0}, "test", residual_scale=0.15,
        ),
    )
    assert workspace.decision is SupervisorDecision.DENY
    assert "workspace" in workspace.reason

    stopped = runtime.step(
        request(), observation(force=21.0), ResidualController(), actuator,
    )
    assert stopped.status is SkillExecutionStatus.STOPPED
    assert stopped.supervisor and stopped.supervisor.decision is SupervisorDecision.STOP
    assert actuator.calls == []

    emergency = runtime.step(
        request(), observation(safety_stop=True), ResidualController(), actuator,
    )
    assert emergency.status is SkillExecutionStatus.STOPPED
    assert actuator.calls == []


def test_stale_authorization_is_denied_without_automatic_fallback() -> None:
    registry = active_registry()
    actuator = RecordingActuator()
    result = SkillRuntime(registry).step(
        request(revision=3),
        observation(revision=4),
        ResidualController(),
        actuator,
    )

    assert result.status is SkillExecutionStatus.DENIED
    assert result.fallback_skill_id == "right_arm.ik_grasp.v1"
    assert actuator.calls == []
