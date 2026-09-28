"""Bounded runtime-generated expressive motion.

An expressive window is a capability grant, not a gesture name.  A policy may
choose generic motion primitives inside the granted time, arm, region, and
contact envelope.  The result is still a proposal until the normal engine
authorization path accepts it.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any

from .contracts import PlanGraph, Step, WorldState


# These are policy primitives, not human-labelled routines.  The low-level
# provider decides how a primitive is shaped for its actuator model.
EXPRESSIVE_PRIMITIVES: tuple[str, ...] = (
    "ARC", "OSCILLATE", "SWEEP", "TWIST", "PAUSE", "MIRROR", "LEAD",
    "FOLLOW", "ALTERNATE", "SYNCHRONIZE", "CHANGE_SPEED",
    "CHANGE_AMPLITUDE", "CHANGE_PHASE", "HOVER", "CIRCLE", "BOUNCE",
    "APPROACH_CONTACT", "SOFT_CONTACT", "BREAK_CONTACT",
)

_PRIMITIVE_SET = frozenset(EXPRESSIVE_PRIMITIVES)
_COORDINATION = frozenset({"MIRROR", "LEAD", "FOLLOW", "ALTERNATE", "SYNCHRONIZE"})
_CONTACT = frozenset({"APPROACH_CONTACT", "SOFT_CONTACT", "BREAK_CONTACT"})
_DEFAULT_OBJECTIVES: tuple[tuple[str, float], ...] = (
    ("novelty", 0.35),
    ("coordination", 0.25),
    ("smoothness", 0.20),
    ("visibility", 0.10),
    ("symmetry_breaking", 0.10),
)


class ExpressiveWindowRejected(ValueError):
    """A window or proposal would exceed its explicit authority envelope."""


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExpressiveWindowRejected(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_int(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExpressiveWindowRejected(f"{field_name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class ExpressiveWindow:
    """A bounded, revision-scoped capability grant for free-space motion."""

    window_id: str
    org_id: str = "local-demo"
    session_id: str = "local-session"
    base_world_revision: int = 0
    start_ms: int = 0
    deadline_ms: int = 1000
    return_deadline_ms: int | None = None
    return_margin_ms: int = 0
    allowed_arms: tuple[str, ...] = ("left", "right")
    allowed_regions: tuple[str, ...] = ("safe_free_volume",)
    allowed_primitives: tuple[str, ...] = EXPRESSIVE_PRIMITIVES
    max_contact_duration_ms: int = 0
    max_amplitude: float = 0.20
    objective_weights: tuple[tuple[str, float], ...] = _DEFAULT_OBJECTIVES

    def __post_init__(self) -> None:
        window_id = _require_text(self.window_id, "window_id")
        org_id = _require_text(self.org_id, "org_id")
        session_id = _require_text(self.session_id, "session_id")
        base_revision = _require_int(self.base_world_revision, "base_world_revision")
        start_ms = _require_int(self.start_ms, "start_ms")
        deadline_ms = _require_int(self.deadline_ms, "deadline_ms")
        if deadline_ms <= start_ms:
            raise ExpressiveWindowRejected("deadline_ms must be after start_ms")
        margin = _require_int(self.return_margin_ms, "return_margin_ms")
        return_deadline = self.return_deadline_ms
        if return_deadline is None:
            return_deadline = deadline_ms - margin
        return_deadline = _require_int(return_deadline, "return_deadline_ms")
        if not start_ms <= return_deadline <= deadline_ms:
            raise ExpressiveWindowRejected(
                "return_deadline_ms must be within the window and after start_ms"
            )

        arms = tuple(dict.fromkeys(_require_text(a, "allowed_arm").lower()
                                  for a in self.allowed_arms))
        if not arms or any(a not in {"left", "right"} for a in arms):
            raise ExpressiveWindowRejected("allowed_arms must contain left/right only")
        regions = tuple(dict.fromkeys(_require_text(r, "allowed_region")
                                     for r in self.allowed_regions))
        if not regions:
            raise ExpressiveWindowRejected("allowed_regions cannot be empty")
        primitives = tuple(dict.fromkeys(_require_text(p, "allowed_primitive").upper()
                                        for p in self.allowed_primitives))
        unknown = sorted(set(primitives) - _PRIMITIVE_SET)
        if not primitives or unknown:
            raise ExpressiveWindowRejected(f"unsupported expressive primitive(s): {unknown}")

        contact_limit = _require_int(
            self.max_contact_duration_ms, "max_contact_duration_ms")
        if not math.isfinite(float(self.max_amplitude)) or self.max_amplitude < 0.0:
            raise ExpressiveWindowRejected("max_amplitude must be finite and >= 0")

        weights = tuple((str(name).strip(), float(weight))
                        for name, weight in self.objective_weights)
        if not weights or any(not name or not math.isfinite(weight) or weight < 0.0
                              for name, weight in weights):
            raise ExpressiveWindowRejected("objective weights must be finite and non-negative")
        if sum(weight for _name, weight in weights) <= 0.0:
            raise ExpressiveWindowRejected("at least one objective weight must be positive")

        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "org_id", org_id)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "base_world_revision", base_revision)
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "deadline_ms", deadline_ms)
        object.__setattr__(self, "return_deadline_ms", return_deadline)
        object.__setattr__(self, "return_margin_ms", margin)
        object.__setattr__(self, "allowed_arms", arms)
        object.__setattr__(self, "allowed_regions", regions)
        object.__setattr__(self, "allowed_primitives", primitives)
        object.__setattr__(self, "max_contact_duration_ms", contact_limit)
        object.__setattr__(self, "objective_weights", tuple(sorted(weights)))

    @property
    def time_budget_ms(self) -> int:
        return self.deadline_ms - self.start_ms

    @property
    def return_budget_ms(self) -> int:
        return self.return_deadline_ms - self.start_ms  # type: ignore[operator]

    @property
    def objectives(self) -> dict[str, float]:
        return dict(self.objective_weights)

    def digest(self) -> str:
        payload = {
            "window_id": self.window_id,
            "org_id": self.org_id,
            "session_id": self.session_id,
            "base_world_revision": self.base_world_revision,
            "start_ms": self.start_ms,
            "deadline_ms": self.deadline_ms,
            "return_deadline_ms": self.return_deadline_ms,
            "allowed_arms": self.allowed_arms,
            "allowed_regions": self.allowed_regions,
            "allowed_primitives": self.allowed_primitives,
            "max_contact_duration_ms": self.max_contact_duration_ms,
            "max_amplitude": self.max_amplitude,
            "objective_weights": self.objective_weights,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(blob).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "org_id": self.org_id,
            "session_id": self.session_id,
            "base_world_revision": self.base_world_revision,
            "start_ms": self.start_ms,
            "deadline_ms": self.deadline_ms,
            "return_deadline_ms": self.return_deadline_ms,
            "allowed_arms": list(self.allowed_arms),
            "allowed_regions": list(self.allowed_regions),
            "allowed_primitives": list(self.allowed_primitives),
            "max_contact_duration_ms": self.max_contact_duration_ms,
            "max_amplitude": self.max_amplitude,
            "objective_weights": dict(self.objective_weights),
            "digest": self.digest(),
        }


@dataclass(frozen=True)
class ExpressionProposal:
    """One short-horizon generic motion proposal inside a window."""

    proposal_id: str
    window_id: str
    org_id: str
    session_id: str
    base_world_revision: int
    arm: str
    primitive: str
    start_ms: int
    duration_ms: int
    region_id: str = "safe_free_volume"
    axis: int = 0
    amplitude: float = 0.0
    phase_deg: float = 0.0
    cycles: float = 1.0
    coordination_group: str | None = None

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.duration_ms

    def as_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "window_id": self.window_id,
            "primitive": self.primitive,
            "start_ms": self.start_ms,
            "duration_ms": self.duration_ms,
            "region": self.region_id,
            "axis": self.axis,
            "amplitude": self.amplitude,
            "phase_deg": self.phase_deg,
            "cycles": self.cycles,
        }
        if self.coordination_group is not None:
            args["coordination_group"] = self.coordination_group
        return args

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "window_id": self.window_id,
            "org_id": self.org_id,
            "session_id": self.session_id,
            "base_world_revision": self.base_world_revision,
            "arm": self.arm,
            "primitive": self.primitive,
            "start_ms": self.start_ms,
            "duration_ms": self.duration_ms,
            "end_ms": self.end_ms,
            "region_id": self.region_id,
            "axis": self.axis,
            "amplitude": self.amplitude,
            "phase_deg": self.phase_deg,
            "cycles": self.cycles,
            "coordination_group": self.coordination_group,
        }


def validate_proposal(
    window: ExpressiveWindow,
    proposal: ExpressionProposal,
    *,
    current_world_revision: int | None = None,
) -> None:
    """Fail closed when a proposal exceeds a window or uses stale authority."""
    if proposal.window_id != window.window_id:
        raise ExpressiveWindowRejected("proposal belongs to another expressive window")
    if proposal.org_id != window.org_id or proposal.session_id != window.session_id:
        raise ExpressiveWindowRejected("proposal organization/session scope mismatch")
    if proposal.base_world_revision != window.base_world_revision:
        raise ExpressiveWindowRejected("proposal world revision does not match the window")
    if current_world_revision is not None and current_world_revision != window.base_world_revision:
        raise ExpressiveWindowRejected("expressive window is stale against current world revision")
    if proposal.arm not in window.allowed_arms:
        raise ExpressiveWindowRejected(f"arm {proposal.arm!r} is outside the window")
    if proposal.primitive not in window.allowed_primitives:
        raise ExpressiveWindowRejected(f"primitive {proposal.primitive!r} is not permitted")
    if proposal.region_id not in window.allowed_regions:
        raise ExpressiveWindowRejected(f"region {proposal.region_id!r} is outside the window")
    _require_int(proposal.start_ms, "proposal.start_ms")
    duration = _require_int(proposal.duration_ms, "proposal.duration_ms", minimum=1)
    if proposal.start_ms < window.start_ms or proposal.end_ms > window.return_deadline_ms:
        raise ExpressiveWindowRejected("proposal exceeds the expressive return deadline")
    if proposal.primitive in _CONTACT:
        if window.max_contact_duration_ms <= 0:
            raise ExpressiveWindowRejected("contact primitive requires a contact allowance")
        if duration > window.max_contact_duration_ms:
            raise ExpressiveWindowRejected("contact proposal exceeds contact duration allowance")
    if isinstance(proposal.axis, bool) or not isinstance(proposal.axis, int) \
            or not 0 <= proposal.axis <= 5:
        raise ExpressiveWindowRejected("proposal.axis must be an actuator axis from 0 through 5")
    if not math.isfinite(proposal.amplitude) or not 0.0 <= proposal.amplitude <= window.max_amplitude:
        raise ExpressiveWindowRejected("proposal amplitude exceeds the window")
    if not math.isfinite(proposal.phase_deg) or not math.isfinite(proposal.cycles):
        raise ExpressiveWindowRejected("proposal phase/cycles must be finite")
    if proposal.cycles <= 0.0:
        raise ExpressiveWindowRejected("proposal cycles must be positive")


def validate_expressive_command(args: dict[str, Any]) -> None:
    """Validate the provider-facing command after proposal materialization.

    Providers must repeat the safety boundary because a caller can construct a
    ``Step`` without going through :func:`generate_expression_plan`.
    """
    if not isinstance(args, dict):
        raise ExpressiveWindowRejected("EXPRESS arguments must be a mapping")
    if "object" in args:
        raise ExpressiveWindowRejected("EXPRESS cannot target an object")
    primitive = args.get("primitive")
    if not isinstance(primitive, str) or primitive.upper() not in _PRIMITIVE_SET:
        raise ExpressiveWindowRejected("EXPRESS requires a known generic primitive")
    if primitive.upper() in _CONTACT:
        raise ExpressiveWindowRejected(
            "contact primitives require a dedicated contact controller")
    region = args.get("region")
    if not isinstance(region, str) or not region.startswith("safe_"):
        raise ExpressiveWindowRejected("EXPRESS must stay inside a safe region")
    duration = args.get("duration_ms")
    if isinstance(duration, bool) or not isinstance(duration, int) or not 1 <= duration <= 2000:
        raise ExpressiveWindowRejected("EXPRESS duration must be between 1 and 2000 ms")
    axis = args.get("axis", 0)
    if isinstance(axis, bool) or not isinstance(axis, int) or not 0 <= axis <= 5:
        raise ExpressiveWindowRejected("EXPRESS axis must be between 0 and 5")
    amplitude = args.get("amplitude", 0.0)
    if not isinstance(amplitude, (int, float)) or isinstance(amplitude, bool) \
            or not math.isfinite(float(amplitude)) or not 0.0 <= float(amplitude) <= 0.20:
        raise ExpressiveWindowRejected("EXPRESS amplitude exceeds provider safety bound")
    phase = args.get("phase_deg", 0.0)
    cycles = args.get("cycles", 1.0)
    if not isinstance(phase, (int, float)) or not math.isfinite(float(phase)):
        raise ExpressiveWindowRejected("EXPRESS phase must be finite")
    if not isinstance(cycles, (int, float)) or not math.isfinite(float(cycles)) \
            or float(cycles) <= 0.0:
        raise ExpressiveWindowRejected("EXPRESS cycles must be finite and positive")


def validate_expressive_step(
    window: ExpressiveWindow,
    step: Step,
    *,
    current_world_revision: int | None = None,
) -> None:
    """Bind an action step to the window that authorized its proposal."""
    if step.op != "EXPRESS":
        raise ExpressiveWindowRejected("step is not an EXPRESS action")
    args = step.args
    if args.get("window_digest") != window.digest():
        raise ExpressiveWindowRejected("step window digest does not match authority")
    try:
        proposal = ExpressionProposal(
            proposal_id=step.id.removeprefix("express_"),
            window_id=args["window_id"],
            org_id=window.org_id,
            session_id=window.session_id,
            base_world_revision=window.base_world_revision,
            arm=step.arm or "",
            primitive=str(args["primitive"]).upper(),
            start_ms=args["start_ms"],
            duration_ms=args["duration_ms"],
            region_id=args["region"],
            axis=args.get("axis", 0),
            amplitude=args.get("amplitude", 0.0),
            phase_deg=args.get("phase_deg", 0.0),
            cycles=args.get("cycles", 1.0),
            coordination_group=args.get("coordination_group"),
        )
    except (KeyError, TypeError) as exc:
        raise ExpressiveWindowRejected("EXPRESS step is missing proposal fields") from exc
    validate_proposal(
        window, proposal, current_world_revision=current_world_revision)


@dataclass(frozen=True)
class ExpressionPlan:
    window_id: str
    window_digest: str
    base_world_revision: int
    seed: int
    proposals: tuple[ExpressionProposal, ...] = ()
    rejected: tuple[dict[str, str], ...] = ()

    @property
    def span_ms(self) -> int:
        if not self.proposals:
            return 0
        start = min(p.start_ms for p in self.proposals)
        end = max(p.end_ms for p in self.proposals)
        return end - start

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "window_digest": self.window_digest,
            "base_world_revision": self.base_world_revision,
            "seed": self.seed,
            "span_ms": self.span_ms,
            "proposals": [p.as_dict() for p in self.proposals],
            "rejected": [dict(r) for r in self.rejected],
        }

    def to_steps(self, *, deps: tuple[str, ...] = ()) -> tuple[Step, ...]:
        """Convert proposals to generic action steps for normal authorization."""
        return tuple(
            Step(
                id=f"express_{proposal.proposal_id}",
                contract="manipulate",
                op="EXPRESS",
                args={
                    **proposal.as_args(),
                    "window_digest": self.window_digest,
                    "policy_seed": self.seed,
                },
                deps=deps,
                arm=proposal.arm,
                rationale="runtime-generated expressive proposal inside an authorized window",
            )
            for proposal in self.proposals
        )


def generate_expression_plan(
    window: ExpressiveWindow,
    world: WorldState | None = None,
    *,
    seed: int = 0,
    max_proposals: int = 4,
) -> ExpressionPlan:
    """Generate generic proposals deterministically inside the window.

    This is deliberately a small policy reference, not a claim of learned
    physical intelligence.  A future policy can replace it while keeping the
    same window and proposal validation boundary.
    """
    if world is not None:
        if world.org_id != window.org_id:
            raise ExpressiveWindowRejected("world organization scope does not match the window")
        if world.revision != window.base_world_revision:
            raise ExpressiveWindowRejected("world revision does not match the window")
    if isinstance(max_proposals, bool) or not isinstance(max_proposals, int) or max_proposals < 1:
        raise ExpressiveWindowRejected("max_proposals must be an integer >= 1")

    rng = random.Random(seed)
    budget = window.return_budget_ms
    allowed = tuple(window.allowed_primitives)
    if budget <= 0:
        return ExpressionPlan(
            window.window_id, window.digest(), window.base_world_revision, seed,
            rejected=({"target": "window", "reason": "no return budget"},),
        )

    duration = max(1, min(450, budget))
    axis = rng.randrange(0, 6)
    amplitude = min(window.max_amplitude, 0.04 + rng.random() * 0.08)
    phase = rng.uniform(0.0, 360.0)
    cycles = 1.0 + rng.random() * 1.5
    objectives = window.objectives
    coordination = objectives.get("coordination", 0.0)
    coordination_ops = tuple(p for p in allowed if p in _COORDINATION)

    proposals: list[ExpressionProposal] = []
    if len(window.allowed_arms) >= 2 and coordination > 0.0 and coordination_ops:
        primitive = coordination_ops[rng.randrange(len(coordination_ops))]
        group = f"{window.window_id}:coord:0"
        for index, arm in enumerate(window.allowed_arms[:2]):
            arm_primitive = primitive
            if primitive == "LEAD":
                arm_primitive = "LEAD" if index == 0 else "FOLLOW"
            elif primitive == "FOLLOW":
                arm_primitive = "FOLLOW" if index == 0 else "LEAD"
            proposal = ExpressionProposal(
                proposal_id=f"{window.window_id}_p{index}",
                window_id=window.window_id,
                org_id=window.org_id,
                session_id=window.session_id,
                base_world_revision=window.base_world_revision,
                arm=arm,
                primitive=arm_primitive,
                start_ms=window.start_ms,
                duration_ms=duration,
                region_id=window.allowed_regions[0],
                axis=axis,
                amplitude=amplitude,
                phase_deg=phase + (180.0 if index else 0.0),
                cycles=cycles,
                coordination_group=group,
            )
            validate_proposal(window, proposal, current_world_revision=window.base_world_revision)
            proposals.append(proposal)
    else:
        primitive_pool = tuple(p for p in allowed if p not in _CONTACT)
        count = min(max_proposals, len(window.allowed_arms), len(primitive_pool))
        for index, arm in enumerate(window.allowed_arms[:count]):
            primitive = primitive_pool[(rng.randrange(len(primitive_pool)) + index) % len(primitive_pool)]
            proposal = ExpressionProposal(
                proposal_id=f"{window.window_id}_p{index}",
                window_id=window.window_id,
                org_id=window.org_id,
                session_id=window.session_id,
                base_world_revision=window.base_world_revision,
                arm=arm,
                primitive=primitive,
                start_ms=window.start_ms,
                duration_ms=duration,
                region_id=window.allowed_regions[0],
                axis=(axis + index) % 6,
                amplitude=amplitude,
                phase_deg=phase + index * 47.0,
                cycles=cycles,
            )
            validate_proposal(window, proposal, current_world_revision=window.base_world_revision)
            proposals.append(proposal)

    return ExpressionPlan(
        window_id=window.window_id,
        window_digest=window.digest(),
        base_world_revision=window.base_world_revision,
        seed=seed,
        proposals=tuple(proposals),
    )


def attach_expression(graph: PlanGraph, plan: ExpressionPlan) -> PlanGraph:
    """Return a graph with expression proposals as independent leaf steps.

    The caller must place the returned graph through the ordinary scheduler and
    engine.  No proposal is allowed to become a dependency of the task graph.
    """
    out = PlanGraph(goal=graph.goal, revision=graph.revision)
    insert_at = next(
        (index for index, step in enumerate(graph.steps) if step.contract == "verify"),
        len(graph.steps),
    )
    out.steps.extend(graph.steps[:insert_at])
    out.steps.extend(plan.to_steps())
    out.steps.extend(graph.steps[insert_at:])
    return out
