"""Value contracts for governed deterministic and learned skills.

The policy/controller boundary is intentionally proposal-only.  A controller
can return a bounded action proposal, but it cannot authorize itself, change a
goal, alter a constraint, select a different skill, or write an actuator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping

from ..contracts import ActionAuthorization, AuthorizationVerdict

__all__ = [
    "ControllerType",
    "PromotionEvidence",
    "ProposedAction",
    "SkillContractError",
    "SkillAuthority",
    "SkillControl",
    "SkillExecutionStatus",
    "SkillManifest",
    "SkillObservation",
    "SkillRequest",
    "SkillStage",
    "SupervisorDecision",
    "SupervisorReceipt",
]


class SkillContractError(ValueError):
    """A skill contract or bounded execution value is invalid."""


class ControllerType(str, Enum):
    DETERMINISTIC = "deterministic"
    SCRIPTED = "scripted"
    RL = "rl"
    RESIDUAL_RL = "residual_rl"
    VLA = "vla"


class SkillStage(str, Enum):
    TRAINED = "TRAINED"
    SIM_EVAL = "SIM_EVAL"
    DOMAIN_RANDOMIZATION = "DOMAIN_RANDOMIZATION"
    SHADOW = "SHADOW"
    SUPERVISED_HARDWARE = "SUPERVISED_HARDWARE"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"


class SupervisorDecision(str, Enum):
    ALLOW = "ALLOW"
    CLAMP = "CLAMP"
    DENY = "DENY"
    STOP = "STOP"


class SkillExecutionStatus(str, Enum):
    EXECUTED = "executed"
    DENIED = "denied"
    STOPPED = "stopped"
    CONTROLLER_FAILED = "controller_failed"
    ACTUATOR_FAILED = "actuator_failed"


_PROMOTION_ORDER = (
    SkillStage.TRAINED,
    SkillStage.SIM_EVAL,
    SkillStage.DOMAIN_RANDOMIZATION,
    SkillStage.SHADOW,
    SkillStage.SUPERVISED_HARDWARE,
    SkillStage.VALIDATED,
    SkillStage.ACTIVE,
)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SkillContractError(f"{name} must be a non-empty string")
    return value.strip()


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SkillContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        bound = f" >= {minimum}" if minimum is not None else ""
        raise SkillContractError(f"{name} must be finite{bound}")
    return result


def _unique(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise SkillContractError(f"{name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise SkillContractError(f"{name} must not contain duplicates")
    return tuple(value.strip() for value in values)


@dataclass(frozen=True)
class SkillAuthority:
    """Authority a controller may exercise inside its skill envelope."""

    may_move_arm: bool = True
    may_change_goal: bool = False
    may_change_constraints: bool = False
    may_select_other_skills: bool = False

    def __post_init__(self) -> None:
        for name in (
            "may_move_arm",
            "may_change_goal",
            "may_change_constraints",
            "may_select_other_skills",
        ):
            if not isinstance(getattr(self, name), bool):
                raise SkillContractError(f"{name} must be a bool")
        if self.may_change_goal or self.may_change_constraints or self.may_select_other_skills:
            raise SkillContractError(
                "a skill controller may not change goals, constraints, or skill selection"
            )

    def as_dict(self) -> dict[str, bool]:
        return {
            "may_move_arm": self.may_move_arm,
            "may_change_goal": self.may_change_goal,
            "may_change_constraints": self.may_change_constraints,
            "may_select_other_skills": self.may_select_other_skills,
        }


@dataclass(frozen=True)
class SkillControl:
    """Hard control envelope applied after every controller proposal."""

    action_space: str = "cartesian_delta"
    max_delta_mm: float = 4.0
    max_rotation_deg: float = 3.0
    max_gripper_delta: float = 0.05
    workspace_mm: tuple[tuple[str, float, float], ...] = ()
    max_speed_mm_s: float | None = 100.0
    max_contact_force_n: float | None = 20.0
    residual_scale_max: float = 1.0
    action_interval_ms: int = 50
    timeout_ms: int = 5_000

    def __post_init__(self) -> None:
        if self.action_space != "cartesian_delta":
            raise SkillContractError(
                "only cartesian_delta is supported by the initial skill supervisor"
            )
        _finite(self.max_delta_mm, "max_delta_mm", minimum=0.0)
        _finite(self.max_rotation_deg, "max_rotation_deg", minimum=0.0)
        _finite(self.max_gripper_delta, "max_gripper_delta", minimum=0.0)
        _finite(self.residual_scale_max, "residual_scale_max", minimum=0.0)
        if self.max_speed_mm_s is not None:
            _finite(self.max_speed_mm_s, "max_speed_mm_s", minimum=0.0)
        if self.max_contact_force_n is not None:
            _finite(self.max_contact_force_n, "max_contact_force_n", minimum=0.0)
        if (isinstance(self.action_interval_ms, bool)
                or not isinstance(self.action_interval_ms, int)
                or self.action_interval_ms <= 0):
            raise SkillContractError("action_interval_ms must be a positive integer")
        if (isinstance(self.timeout_ms, bool)
                or not isinstance(self.timeout_ms, int)
                or self.timeout_ms <= 0):
            raise SkillContractError("timeout_ms must be a positive integer")

        axes: set[str] = set()
        normalized_workspace: list[tuple[str, float, float]] = []
        for axis, lower, upper in self.workspace_mm:
            axis = _text(axis, "workspace axis")
            if axis not in {"x", "y", "z"} or axis in axes:
                raise SkillContractError("workspace must contain unique x/y/z axes")
            lower = _finite(lower, f"workspace {axis} lower bound")
            upper = _finite(upper, f"workspace {axis} upper bound")
            if lower >= upper:
                raise SkillContractError(f"workspace {axis} lower bound must be below upper bound")
            axes.add(axis)
            normalized_workspace.append((axis, lower, upper))
        object.__setattr__(self, "workspace_mm", tuple(normalized_workspace))

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_space": self.action_space,
            "max_delta_mm": self.max_delta_mm,
            "max_rotation_deg": self.max_rotation_deg,
            "max_gripper_delta": self.max_gripper_delta,
            "workspace_mm": [list(item) for item in self.workspace_mm],
            "max_speed_mm_s": self.max_speed_mm_s,
            "max_contact_force_n": self.max_contact_force_n,
            "residual_scale_max": self.residual_scale_max,
            "action_interval_ms": self.action_interval_ms,
            "timeout_ms": self.timeout_ms,
        }


@dataclass(frozen=True)
class SkillManifest:
    """Promotion-gated identity and operating contract for one skill."""

    skill_id: str
    org_id: str
    capability: str
    controller_type: ControllerType
    artifact_digest: str
    arm_ids: tuple[str, ...] = ()
    input_fields: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    control: SkillControl = SkillControl()
    termination: tuple[str, ...] = ()
    fallback_skill_id: str | None = None
    verification: tuple[str, ...] = ()
    authority: SkillAuthority = SkillAuthority()
    stage: SkillStage = SkillStage.TRAINED

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_id", _text(self.skill_id, "skill_id"))
        object.__setattr__(self, "org_id", _text(self.org_id, "org_id"))
        object.__setattr__(self, "capability", _text(self.capability, "capability"))
        if not isinstance(self.controller_type, ControllerType):
            raise SkillContractError("controller_type must be a ControllerType")
        object.__setattr__(self, "artifact_digest", _text(self.artifact_digest, "artifact_digest"))
        if self.controller_type in {
            ControllerType.RL,
            ControllerType.RESIDUAL_RL,
            ControllerType.VLA,
        }:
            if not self.artifact_digest.startswith("sha256:"):
                raise SkillContractError("learned skills require a sha256 artifact digest")
            if not self.verification:
                raise SkillContractError("learned skills require verification channels")
        object.__setattr__(self, "arm_ids", _unique(tuple(self.arm_ids), "arm_ids"))
        object.__setattr__(self, "input_fields", _unique(tuple(self.input_fields), "input_fields"))
        object.__setattr__(self, "preconditions", _unique(tuple(self.preconditions), "preconditions"))
        object.__setattr__(self, "termination", _unique(tuple(self.termination), "termination"))
        object.__setattr__(self, "verification", _unique(tuple(self.verification), "verification"))
        if self.fallback_skill_id is not None:
            fallback = _text(self.fallback_skill_id, "fallback_skill_id")
            if fallback == self.skill_id:
                raise SkillContractError("a skill cannot fall back to itself")
            object.__setattr__(self, "fallback_skill_id", fallback)
        if not isinstance(self.authority, SkillAuthority):
            raise SkillContractError("authority must be a SkillAuthority")
        if not isinstance(self.control, SkillControl):
            raise SkillContractError("control must be a SkillControl")
        if not isinstance(self.stage, SkillStage):
            raise SkillContractError("stage must be a SkillStage")
        if self.authority.may_move_arm:
            axes = {axis for axis, _, _ in self.control.workspace_mm}
            if axes != {"x", "y", "z"}:
                raise SkillContractError(
                    "moving skills require a complete x/y/z workspace envelope"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "org_id": self.org_id,
            "capability": self.capability,
            "controller_type": self.controller_type.value,
            "artifact_digest": self.artifact_digest,
            "arm_ids": list(self.arm_ids),
            "input_fields": list(self.input_fields),
            "preconditions": list(self.preconditions),
            "control": self.control.as_dict(),
            "termination": list(self.termination),
            "fallback_skill_id": self.fallback_skill_id,
            "verification": list(self.verification),
            "authority": self.authority.as_dict(),
            "stage": self.stage.value,
        }


@dataclass(frozen=True)
class PromotionEvidence:
    """Immutable evidence required for one adjacent promotion step."""

    receipt_id: str
    artifact_digest: str
    checks: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        _text(self.receipt_id, "promotion receipt_id")
        _text(self.artifact_digest, "promotion artifact_digest")
        if not self.checks:
            raise SkillContractError("promotion evidence requires at least one check")
        names: set[str] = set()
        for name, value in self.checks:
            name = _text(name, "promotion check")
            if name in names or not isinstance(value, bool):
                raise SkillContractError("promotion checks must have unique bool values")
            names.add(name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "artifact_digest": self.artifact_digest,
            "checks": {name: value for name, value in self.checks},
        }


@dataclass(frozen=True)
class SkillRequest:
    """A capability invocation already authorized by the governed planner."""

    request_id: str
    org_id: str
    capability: str
    operation: str
    arm_id: str
    expected_world_revision: int
    authorization: ActionAuthorization
    skill_id: str | None = None
    preferred_skill_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value, name in (
            (self.request_id, "request_id"),
            (self.org_id, "org_id"),
            (self.capability, "capability"),
            (self.operation, "operation"),
            (self.arm_id, "arm_id"),
        ):
            _text(value, name)
        if (isinstance(self.expected_world_revision, bool)
                or not isinstance(self.expected_world_revision, int)
                or self.expected_world_revision < 0):
            raise SkillContractError("expected_world_revision must be non-negative")
        if not isinstance(self.authorization, ActionAuthorization):
            raise SkillContractError("authorization must be ActionAuthorization")
        if self.skill_id is not None:
            _text(self.skill_id, "skill_id")
        object.__setattr__(self, "preferred_skill_ids",
                           _unique(tuple(self.preferred_skill_ids), "preferred_skill_ids"))

    @property
    def authorized(self) -> bool:
        return self.authorization.verdict is AuthorizationVerdict.ALLOW

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "org_id": self.org_id,
            "capability": self.capability,
            "operation": self.operation,
            "arm_id": self.arm_id,
            "expected_world_revision": self.expected_world_revision,
            "authorization": self.authorization.as_dict(),
            "skill_id": self.skill_id,
            "preferred_skill_ids": list(self.preferred_skill_ids),
        }


@dataclass(frozen=True)
class SkillObservation:
    """Read-only observation presented to a controller and supervisor."""

    org_id: str
    arm_id: str
    world_revision: int
    tcp_position_mm: tuple[float, float, float]
    contact_force_n: float = 0.0
    action_interval_ms: int = 50
    safety_stop: bool = False

    def __post_init__(self) -> None:
        _text(self.org_id, "observation org_id")
        _text(self.arm_id, "observation arm_id")
        if (isinstance(self.world_revision, bool)
                or not isinstance(self.world_revision, int)
                or self.world_revision < 0):
            raise SkillContractError("observation world_revision must be non-negative")
        if len(self.tcp_position_mm) != 3:
            raise SkillContractError("tcp_position_mm must contain x, y, z")
        for index, value in enumerate(self.tcp_position_mm):
            _finite(value, f"tcp_position_mm[{index}]")
        _finite(self.contact_force_n, "contact_force_n", minimum=0.0)
        if (isinstance(self.action_interval_ms, bool)
                or not isinstance(self.action_interval_ms, int)
                or self.action_interval_ms <= 0):
            raise SkillContractError("observation action_interval_ms must be positive")
        if not isinstance(self.safety_stop, bool):
            raise SkillContractError("observation safety_stop must be a bool")

    def as_dict(self) -> dict[str, Any]:
        return {
            "org_id": self.org_id,
            "arm_id": self.arm_id,
            "world_revision": self.world_revision,
            "tcp_position_mm": list(self.tcp_position_mm),
            "contact_force_n": self.contact_force_n,
            "action_interval_ms": self.action_interval_ms,
            "safety_stop": self.safety_stop,
        }


@dataclass(frozen=True)
class ProposedAction:
    """Controller output before the supervisor applies the skill envelope."""

    request_id: str
    skill_id: str
    values: tuple[tuple[str, float], ...]
    controller_backend: str
    residual_scale: float = 1.0

    def __post_init__(self) -> None:
        _text(self.request_id, "proposal request_id")
        _text(self.skill_id, "proposal skill_id")
        _text(self.controller_backend, "controller_backend")
        names: set[str] = set()
        for name, value in self.values:
            name = _text(name, "action name")
            if name in names:
                raise SkillContractError("action names must be unique")
            _finite(value, f"action {name}")
            names.add(name)
        _finite(self.residual_scale, "residual_scale", minimum=0.0)

    @classmethod
    def from_mapping(
        cls,
        request_id: str,
        skill_id: str,
        values: Mapping[str, float],
        controller_backend: str,
        *,
        residual_scale: float = 1.0,
    ) -> "ProposedAction":
        if not isinstance(values, Mapping):
            raise SkillContractError("action values must be a mapping")
        pairs: list[tuple[str, float]] = []
        for name, value in values.items():
            if not isinstance(name, str):
                raise SkillContractError("action names must be strings")
            pairs.append((name, value))
        return cls(
            request_id=request_id,
            skill_id=skill_id,
            values=tuple(sorted(pairs)),
            controller_backend=controller_backend,
            residual_scale=residual_scale,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "skill_id": self.skill_id,
            "values": {name: value for name, value in self.values},
            "controller_backend": self.controller_backend,
            "residual_scale": self.residual_scale,
        }


@dataclass(frozen=True)
class SupervisorReceipt:
    """Evidence of the decision made immediately before actuator access."""

    request_id: str
    skill_id: str
    org_id: str
    world_revision: int
    decision: SupervisorDecision
    reason: str
    requested_action: tuple[tuple[str, float], ...] = ()
    safe_action: tuple[tuple[str, float], ...] = ()
    controller_backend: str = ""
    fallback_skill_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "skill_id": self.skill_id,
            "org_id": self.org_id,
            "world_revision": self.world_revision,
            "decision": self.decision.value,
            "reason": self.reason,
            "requested_action": {name: value for name, value in self.requested_action},
            "safe_action": {name: value for name, value in self.safe_action},
            "controller_backend": self.controller_backend,
            "fallback_skill_id": self.fallback_skill_id,
        }
