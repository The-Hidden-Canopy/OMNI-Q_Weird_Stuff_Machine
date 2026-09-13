from __future__ import annotations

from omni_q import (
    GraspEvidence,
    GraspVerificationPolicy,
    ProposedAction,
    RLGraspController,
    SkillObservation,
    verify_grasp,
)


class Policy:
    def __init__(self) -> None:
        self.seen = []

    def predict(self, observation):
        self.seen.append(observation)
        return {"dx_mm": 1.0, "gripper_delta": -0.02}


def evidence(**overrides):
    values = {
        "org_id": "org_a",
        "arm_id": "arm_1",
        "object_id": "cup_1",
        "world_revision": 4,
        "contact_pad_count": 2,
        "object_lift_mm": 21.0,
        "object_relative_distance_mm": 12.0,
        "gripper_closed": True,
        "max_contact_force_n": 4.0,
    }
    values.update(overrides)
    return GraspEvidence(**values)


def test_rl_grasp_controller_only_emits_a_proposal() -> None:
    policy = Policy()
    controller = RLGraspController("arm_1.rl_grasp.v1", policy)
    observation = SkillObservation(
        org_id="org_a",
        arm_id="arm_1",
        world_revision=4,
        tcp_position_mm=(10.0, 20.0, 30.0),
    )

    proposal = controller.propose(
        type("Request", (), {"request_id": "request-1"})(),
        observation,
    )

    assert isinstance(proposal, ProposedAction)
    assert proposal.skill_id == "arm_1.rl_grasp.v1"
    assert dict(proposal.values) == {"dx_mm": 1.0, "gripper_delta": -0.02}
    assert policy.seen == [observation]
    assert not hasattr(controller, "apply")


def test_grasp_requires_measured_lift_not_only_contact() -> None:
    verified = verify_grasp(evidence())
    contact_only = verify_grasp(evidence(object_lift_mm=0.0))

    assert verified.ok
    assert contact_only.ok is False
    assert "lift" in contact_only.reason


def test_grasp_verifier_preserves_failed_evidence_and_force_boundary() -> None:
    policy = GraspVerificationPolicy(max_contact_force_n=5.0)
    result = verify_grasp(evidence(max_contact_force_n=5.1), policy)

    assert result.ok is False
    assert dict(result.checks)["force"] is False
    assert result.as_dict()["checks"]["force"] is False
