"""Bounded policy adapter for grasp proposals.

This module deliberately contains no torch, MuJoCo, or actuator dependency.
The learned policy sees a read-only :class:`SkillObservation` and returns a
proposal.  :class:`omni_q.skills.supervisor.SkillSupervisor` remains the only
component allowed to turn that proposal into actuator input.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from ..contracts import ProposedAction, SkillObservation, SkillRequest

__all__ = ["GraspPolicy", "RLGraspController"]


@runtime_checkable
class GraspPolicy(Protocol):
    """Inference-only policy boundary used by the controller adapter."""

    def predict(self, observation: SkillObservation) -> Mapping[str, float]: ...


class RLGraspController:
    """Turn a policy prediction into a proposal for one registered skill.

    The controller owns neither skill selection nor safety limits.  It always
    emits its configured ``skill_id`` and leaves action-field validation,
    residual scaling, workspace checks, and force checks to the supervisor.
    """

    def __init__(
        self,
        skill_id: str,
        policy: GraspPolicy,
        *,
        backend: str = "rl-grasp",
        residual_scale: float = 1.0,
    ) -> None:
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise ValueError("skill_id must be non-empty")
        if not isinstance(policy, GraspPolicy):
            raise TypeError("policy must implement GraspPolicy.predict")
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError("backend must be non-empty")
        self.skill_id = skill_id.strip()
        self.policy = policy
        self.backend = backend.strip()
        self.residual_scale = residual_scale

    def propose(
        self,
        request: SkillRequest,
        observation: SkillObservation,
    ) -> ProposedAction:
        predicted = self.policy.predict(observation)
        if not isinstance(predicted, Mapping):
            raise TypeError("grasp policy must return a mapping of action proposals")
        return ProposedAction.from_mapping(
            request.request_id,
            self.skill_id,
            predicted,
            self.backend,
            residual_scale=self.residual_scale,
        )
