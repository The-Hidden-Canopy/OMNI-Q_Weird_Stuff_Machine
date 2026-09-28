"""One-step skill execution after selection and supervision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from ..events import EventBus
from .contracts import (
    ProposedAction,
    SkillExecutionStatus,
    SkillObservation,
    SkillRequest,
    SupervisorDecision,
    SupervisorReceipt,
)
from .registry import SkillRegistry
from .supervisor import SkillSupervisor

__all__ = [
    "SkillActuator",
    "SkillController",
    "SkillExecutionResult",
    "SkillRuntime",
]


@runtime_checkable
class SkillController(Protocol):
    """Proposal-only controller contract; no actuator method is exposed."""

    backend: str

    def propose(
        self,
        request: SkillRequest,
        observation: SkillObservation,
    ) -> ProposedAction: ...


@runtime_checkable
class SkillActuator(Protocol):
    """Physical or simulated actuator that receives supervisor output only."""

    def apply(
        self,
        action: Mapping[str, float],
        observation: SkillObservation,
    ) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True)
class SkillExecutionResult:
    request_id: str
    skill_id: str
    status: SkillExecutionStatus
    supervisor: SupervisorReceipt | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    fallback_skill_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "skill_id": self.skill_id,
            "status": self.status.value,
            "supervisor": self.supervisor.as_dict() if self.supervisor else None,
            "detail": dict(self.detail),
            "fallback_skill_id": self.fallback_skill_id,
        }


class SkillRuntime:
    """Select a promoted skill, supervise its proposal, then actuate safely."""

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        supervisor: SkillSupervisor | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.registry = registry
        self.supervisor = supervisor or SkillSupervisor()
        self.bus = bus or registry.bus

    def step(
        self,
        request: SkillRequest,
        observation: SkillObservation,
        controller: SkillController,
        actuator: SkillActuator,
    ) -> SkillExecutionResult:
        selection = self.registry.select(request)
        manifest = selection.manifest
        self.bus.publish(
            "skill.selected",
            source="skill-runtime",
            request_id=request.request_id,
            skill_id=manifest.skill_id,
            org_id=request.org_id,
            fallback_skill_id=selection.fallback_skill_id,
        )

        try:
            proposal = controller.propose(request, observation)
            if not isinstance(proposal, ProposedAction):
                raise TypeError("controller must return ProposedAction")
        except Exception as exc:  # noqa: BLE001 - controller boundary
            detail = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self.bus.publish(
                "skill.controller.failed",
                source="skill-runtime",
                request_id=request.request_id,
                skill_id=manifest.skill_id,
                **detail,
            )
            return SkillExecutionResult(
                request.request_id,
                manifest.skill_id,
                SkillExecutionStatus.CONTROLLER_FAILED,
                detail=detail,
                fallback_skill_id=selection.fallback_skill_id,
            )

        self.bus.publish(
            "skill.proposed",
            source="skill-runtime",
            request_id=request.request_id,
            skill_id=manifest.skill_id,
            proposal=proposal.as_dict(),
        )
        receipt = self.supervisor.evaluate(request, manifest, observation, proposal)
        self.bus.publish(
            "skill.supervised",
            source="skill-runtime",
            receipt=receipt.as_dict(),
        )
        if receipt.decision is SupervisorDecision.STOP:
            return SkillExecutionResult(
                request.request_id,
                manifest.skill_id,
                SkillExecutionStatus.STOPPED,
                supervisor=receipt,
                fallback_skill_id=selection.fallback_skill_id,
            )
        if receipt.decision is SupervisorDecision.DENY:
            return SkillExecutionResult(
                request.request_id,
                manifest.skill_id,
                SkillExecutionStatus.DENIED,
                supervisor=receipt,
                fallback_skill_id=selection.fallback_skill_id,
            )

        try:
            detail = actuator.apply(
                dict(receipt.safe_action),
                observation,
            )
            detail_dict = dict(detail or {})
        except Exception as exc:  # noqa: BLE001 - actuator boundary
            failure = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self.bus.publish(
                "skill.actuator.failed",
                source="skill-runtime",
                request_id=request.request_id,
                skill_id=manifest.skill_id,
                **failure,
            )
            return SkillExecutionResult(
                request.request_id,
                manifest.skill_id,
                SkillExecutionStatus.ACTUATOR_FAILED,
                supervisor=receipt,
                detail=failure,
                fallback_skill_id=selection.fallback_skill_id,
            )

        self.bus.publish(
            "skill.executed",
            source="skill-runtime",
            request_id=request.request_id,
            skill_id=manifest.skill_id,
            supervisor=receipt.as_dict(),
            detail=detail_dict,
        )
        return SkillExecutionResult(
            request.request_id,
            manifest.skill_id,
            SkillExecutionStatus.EXECUTED,
            supervisor=receipt,
            detail=detail_dict,
            fallback_skill_id=selection.fallback_skill_id,
        )
