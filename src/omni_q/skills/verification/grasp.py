"""Evidence-based grasp verification.

The verifier is intentionally separate from the controller and actuator.  A
closed gripper or a completed trajectory is not treated as a successful grasp;
the object must show contact and measured lift while remaining inside the
gripper's relative envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

__all__ = [
    "GraspEvidence",
    "GraspVerification",
    "GraspVerificationPolicy",
    "verify_grasp",
]


def _finite(value: float, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


@dataclass(frozen=True)
class GraspEvidence:
    """Read-only evidence emitted by simulation or perception after a grasp."""

    org_id: str
    arm_id: str
    object_id: str
    world_revision: int
    contact_pad_count: int
    object_lift_mm: float
    object_relative_distance_mm: float
    gripper_closed: bool
    max_contact_force_n: float = 0.0

    def __post_init__(self) -> None:
        for value, name in (
            (self.org_id, "org_id"),
            (self.arm_id, "arm_id"),
            (self.object_id, "object_id"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if (isinstance(self.world_revision, bool)
                or not isinstance(self.world_revision, int)
                or self.world_revision < 0):
            raise ValueError("world_revision must be non-negative")
        if (isinstance(self.contact_pad_count, bool)
                or not isinstance(self.contact_pad_count, int)
                or self.contact_pad_count < 0):
            raise ValueError("contact_pad_count must be non-negative")
        _finite(self.object_lift_mm, "object_lift_mm")
        _finite(self.object_relative_distance_mm,
                "object_relative_distance_mm", minimum=0.0)
        _finite(self.max_contact_force_n, "max_contact_force_n", minimum=0.0)
        if not isinstance(self.gripper_closed, bool):
            raise ValueError("gripper_closed must be a bool")


@dataclass(frozen=True)
class GraspVerificationPolicy:
    """Explicit thresholds for one environment and object family."""

    min_contact_pad_count: int = 1
    min_lift_mm: float = 20.0
    max_relative_distance_mm: float = 105.0
    max_contact_force_n: float | None = None

    def __post_init__(self) -> None:
        if (isinstance(self.min_contact_pad_count, bool)
                or not isinstance(self.min_contact_pad_count, int)
                or self.min_contact_pad_count <= 0):
            raise ValueError("min_contact_pad_count must be positive")
        _finite(self.min_lift_mm, "min_lift_mm", minimum=0.0)
        _finite(self.max_relative_distance_mm,
                "max_relative_distance_mm", minimum=0.0)
        if self.max_contact_force_n is not None:
            _finite(self.max_contact_force_n,
                    "max_contact_force_n", minimum=0.0)


@dataclass(frozen=True)
class GraspVerification:
    """Immutable verification result suitable for a receipt or event."""

    ok: bool
    reason: str
    checks: tuple[tuple[str, bool], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "checks": {name: value for name, value in self.checks},
        }


def verify_grasp(
    evidence: GraspEvidence,
    policy: GraspVerificationPolicy | None = None,
) -> GraspVerification:
    """Verify contact, lift, relative containment, and force evidence."""

    policy = policy or GraspVerificationPolicy()
    checks = (
        ("contact", evidence.contact_pad_count >= policy.min_contact_pad_count),
        ("lift", evidence.object_lift_mm >= policy.min_lift_mm),
        ("relative_position", evidence.object_relative_distance_mm
         <= policy.max_relative_distance_mm),
        ("gripper_closed", evidence.gripper_closed),
        ("force", policy.max_contact_force_n is None
         or evidence.max_contact_force_n <= policy.max_contact_force_n),
    )
    failed = tuple(name for name, passed in checks if not passed)
    if failed:
        return GraspVerification(
            ok=False,
            reason="grasp verification failed: " + ", ".join(failed),
            checks=checks,
        )
    return GraspVerification(
        ok=True,
        reason="grasp verified from contact, lift, and containment evidence",
        checks=checks,
    )
