"""OQ-001 — frozen Omni capability contracts.

Six interfaces every provider (mock, Intel/MuJoCo, Qualcomm, Speechmatics) plugs
into. Nothing above this file calls hardware directly; it composes these.

    Observe      world  -> Observation            (perception, never raw video up)
    Plan         goal + WorldState -> PlanGraph   (compose / recompile the graph)
    Manipulate   Step -> ManipResult              (arm + bimanual primitives)
    Verify       expected vs Observation -> VerifyResult
    Device       host + route a Step               (placement is separate from function)
    Receipt      record a run -> ReceiptRecord     (evidence: inputs, graph, hashes)

Data types are frozen where they cross a contract boundary. The ``Protocol``
classes are ``runtime_checkable`` so tests can assert a provider satisfies a
contract without inheritance.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable

# ---------------------------------------------------------------------------
# Shared value types
# ---------------------------------------------------------------------------


class DataStatus(str, Enum):
    """How much a piece of world knowledge can be trusted.

    Borrowed from Open-World-Model-Harness: an inference is never promoted to an
    authoritative fact. The planner must re-``Observe`` or lower its commitment
    before acting on anything that is not ``LIVE``.
    """

    LIVE = "live"
    STALE = "stale"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class Pose:
    """World-frame pose of an object, when a source can supply real geometry
    (a MuJoCo sim, a depth camera). ``zone`` on :class:`Detection` stays the
    coarse symbolic slot; this is the metric truth underneath it. Absent
    (``pose is None``) whenever only a symbolic detector is running."""

    x: float
    y: float
    z: float
    yaw: float = 0.0            # rotation about +z, radians

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, other: "Pose") -> float:
        return (
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        ) ** 0.5

    def planar_distance_to(self, other: "Pose") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass(frozen=True)
class Detection:
    object_id: str
    cls: str
    zone: str
    target_zone: str
    conf: float = 0.95
    status: DataStatus = DataStatus.LIVE
    verified_frame: int = 0
    pose: Pose | None = None        # metric pose when a sim / depth source provides it

    @property
    def misplaced(self) -> bool:
        return self.zone != self.target_zone

    @property
    def authoritative(self) -> bool:
        return self.status is DataStatus.LIVE

    @property
    def located(self) -> bool:
        """True when a metric pose is available (not just a symbolic zone)."""
        return self.pose is not None


@dataclass(frozen=True)
class Observation:
    """Compact structured scene state. ``raw_ref`` points at a frame; it is
    never inlined into the graph."""

    frame: int
    detections: tuple[Detection, ...]
    workspace_clear: bool
    raw_ref: str | None = None

    def misplaced(self) -> tuple[Detection, ...]:
        return tuple(d for d in self.detections if d.misplaced)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "workspace_clear": self.workspace_clear,
            "raw_ref": self.raw_ref,
            "detections": [asdict(d) for d in self.detections],
        }


@dataclass(frozen=True)
class Constraint:
    """A runtime restriction on planning or placement.

    kind: ``forbid_object`` | ``keep_local`` | ``pin`` | ``prefer_arm`` | ``style``
    """

    kind: str
    value: Any = None
    source: str = "operator"
    justification: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "source": self.source,
            "justification": self.justification,
        }


class ConstraintValidationError(ValueError):
    """An external request tried to mutate a graph without adequate authority."""


_CONSTRAINT_KINDS = frozenset({"forbid_object", "keep_local", "pin", "prefer_arm", "style"})


def validate_operator_constraint(
    kind: str,
    value: Any = None,
    *,
    source: str = "operator",
    justification: str | None = None,
) -> Constraint:
    """Validate an operator-authored graph restriction before it is queued.

    AI output can suggest a constraint but cannot inject it through this path.
    The operator transcript/UI rationale is retained as evidence for later
    replay, rather than silently becoming a planner mutation.
    """
    if source != "operator":
        raise ConstraintValidationError("only an operator may submit a graph constraint")
    if kind not in _CONSTRAINT_KINDS:
        raise ConstraintValidationError(f"unsupported constraint kind: {kind}")
    if not isinstance(justification, str) or not justification.strip():
        raise ConstraintValidationError("operator constraint requires justification")
    if kind == "forbid_object" and (not isinstance(value, str) or not value.strip()):
        raise ConstraintValidationError("forbid_object requires an object id")
    if kind == "prefer_arm" and value not in {"left", "right"}:
        raise ConstraintValidationError("prefer_arm must be left or right")
    if kind == "keep_local" and value is not None:
        raise ConstraintValidationError("keep_local does not accept a value")
    return Constraint(kind=kind, value=value, source=source, justification=justification.strip())


@dataclass
class WorldState:
    frame: int
    objects: dict[str, Detection]
    goal: str | None = None
    constraints: tuple[Constraint, ...] = ()
    revision: int = 0
    ownership: dict[str, str | None] = field(default_factory=dict)
    org_id: str = "local-demo"

    def misplaced(self) -> list[Detection]:
        return [o for o in self.objects.values() if o.misplaced]

    def forbidden(self) -> set[str]:
        return {c.value for c in self.constraints if c.kind == "forbid_object"}

    def local_only(self) -> bool:
        return any(c.kind == "keep_local" for c in self.constraints)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "revision": self.revision,
            "goal": self.goal,
            "constraints": [c.as_dict() for c in self.constraints],
            "objects": {k: asdict(v) for k, v in self.objects.items()},
            "ownership": dict(self.ownership),
            "org_id": self.org_id,
        }


@dataclass
class Step:
    id: str
    contract: str                       # observe | manipulate | verify
    op: str                             # PICK, PLACE, MOVE, ROTATE, HANDOFF, ...
    args: dict[str, Any] = field(default_factory=dict)
    deps: tuple[str, ...] = ()
    arm: str | None = None              # left | right | None (planner may leave open)
    device: str | None = None          # filled by a Device provider
    state: str = "pending"             # pending | running | done | failed | denied | skipped
    result: dict[str, Any] | None = None
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "contract": self.contract, "op": self.op,
            "args": self.args, "deps": list(self.deps), "arm": self.arm,
            "device": self.device, "state": self.state, "result": self.result,
            "rationale": self.rationale,
        }


@dataclass
class PlanGraph:
    goal: str
    steps: list[Step] = field(default_factory=list)
    revision: int = 0

    def by_id(self, step_id: str) -> Step:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def ops_used(self) -> set[str]:
        return {s.op for s in self.steps}

    def topo_order(self) -> list[Step]:
        order: list[Step] = []
        done: set[str] = set()
        by_id = {s.id: s for s in self.steps}
        temp: set[str] = set()

        def visit(step: Step) -> None:
            if step.id in done:
                return
            if step.id in temp:
                raise ValueError(f"cycle at {step.id}")
            temp.add(step.id)
            for dep in step.deps:
                visit(by_id[dep])
            temp.discard(step.id)
            done.add(step.id)
            order.append(step)

        for s in self.steps:
            visit(s)
        return order

    def as_dict(self) -> dict[str, Any]:
        return {"goal": self.goal, "revision": self.revision,
                "steps": [s.as_dict() for s in self.steps]}


@dataclass(frozen=True)
class ManipResult:
    step_id: str
    ok: bool
    detail: dict[str, Any] = field(default_factory=dict)


class TransitionRejected(ValueError):
    """A world rejected a command before it could alter authoritative state."""


@dataclass(frozen=True)
class TransitionRequest:
    """A revision-bound request to alter the authoritative world.

    Providers may propose or execute a physical command, but only the world
    adapter applies its represented state change.  ``expected_revision`` keeps
    a stale plan from silently overwriting newer observations.
    """

    step_id: str
    op: str
    args: dict[str, Any]
    expected_revision: int
    actor: str | None = None
    org_id: str = "local-demo"


@dataclass(frozen=True)
class TransitionResult:
    step_id: str
    ok: bool
    state_revision: int
    detail: dict[str, Any] = field(default_factory=dict)


class AuthorizationVerdict(str, Enum):
    ALLOW = "ALLOW"
    LIMIT = "LIMIT"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    DENY = "DENY"


@dataclass(frozen=True)
class ActionAuthorization:
    """Finalized authorization evidence emitted before a step can execute."""

    run_id: str
    step_id: str
    op: str
    verdict: AuthorizationVerdict
    reason: str
    state_revision: int
    envelope_digest: str
    content_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step_id": self.step_id,
            "op": self.op,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "state_revision": self.state_revision,
            "envelope_digest": self.envelope_digest,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    expected: dict[str, Any]
    observed: dict[str, Any]
    mismatch: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    kinds: tuple[str, ...]     # step ops / contract kinds this device can host
    local: bool = True
    online: bool = True

    def can_host(self, need: str) -> bool:
        return need in self.kinds


class AutonomyMode(str, Enum):
    """Monotone-escalating operating state (SOCOM_REACT ``envelope.py``).

    Losing a capability or a placement option raises the mode; it never drops
    without an explicit reset. Higher = more constrained.
    """

    NOMINAL = "NOMINAL"
    DEGRADED = "DEGRADED"          # a capability / arm is gone
    LOCAL_ONLY = "LOCAL_ONLY"     # cloud placement withdrawn
    HOLD = "HOLD"                 # cannot make progress this cycle
    PROTECTIVE_STOP = "PROTECTIVE_STOP"


_MODE_PRECEDENCE = {
    AutonomyMode.NOMINAL: 0,
    AutonomyMode.DEGRADED: 1,
    AutonomyMode.LOCAL_ONLY: 2,
    AutonomyMode.HOLD: 3,
    AutonomyMode.PROTECTIVE_STOP: 4,
}


def merge_mode(current: AutonomyMode, incoming: AutonomyMode) -> AutonomyMode:
    """Never downgrade the safety state without an explicit reset."""
    return incoming if _MODE_PRECEDENCE[incoming] >= _MODE_PRECEDENCE[current] else current


@dataclass(frozen=True)
class MissionEnvelope:
    """Fixed operator authority (SOCOM_REACT ``SignedMissionEnvelope``).

    Transient constraints come and go on the :class:`WorldState`; this is the
    part a recompile may **not** widen. A partitioned / re-placed planner
    inherits it unchanged.
    """

    mission_id: str
    org_id: str = "local-demo"
    version: int = 1
    permitted_ops: tuple[str, ...] = ()          # empty = any op allowed
    forbidden_objects: tuple[str, ...] = ()
    local_only: bool = False
    max_revisions: int = 6
    signature: str = ""

    def digest(self) -> str:
        payload = json.dumps(
            {
                "mission_id": self.mission_id,
                "org_id": self.org_id,
                "version": self.version,
                "permitted_ops": sorted(self.permitted_ops),
                "forbidden_objects": sorted(self.forbidden_objects),
                "local_only": self.local_only,
                "max_revisions": self.max_revisions,
            },
            sort_keys=True,
        )
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def op_permitted(self, op: str) -> bool:
        return not self.permitted_ops or op in self.permitted_ops


@dataclass
class PlanDecision:
    """Reason object attached to every (re)compile (SOCOM_REACT ``planner.py``).

    Auditable without pretending the planner itself is 'explainable AI'.
    """

    goal: str
    revision: int
    selected_ops: tuple[str, ...]
    candidates_considered: int
    candidates_feasible: int
    governing_constraints: tuple[str, ...] = ()
    rejected: dict[str, str] = field(default_factory=dict)   # target -> why
    mode: AutonomyMode = AutonomyMode.NOMINAL
    state_hash: str = ""
    envelope_digest: str = ""
    latency_ms: float = 0.0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        d["selected_ops"] = list(self.selected_ops)
        d["governing_constraints"] = list(self.governing_constraints)
        return d


@dataclass(frozen=True)
class ReceiptRecord:
    run_id: str
    goal: str
    inputs: dict[str, Any]
    plan: dict[str, Any]
    actions: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    hashes: dict[str, str]
    decisions: tuple[dict[str, Any], ...] = ()
    rejected: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    parent_hash: str = "GENESIS"
    content_hash: str = ""
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "goal": self.goal, "inputs": self.inputs,
            "plan": self.plan, "actions": list(self.actions),
            "metrics": self.metrics, "hashes": self.hashes,
            "decisions": list(self.decisions), "rejected": list(self.rejected),
            "provenance": self.provenance, "parent_hash": self.parent_hash,
            "content_hash": self.content_hash, "ts": self.ts,
        }


def sha256_of(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def canonical_bytes(record_dict: dict[str, Any]) -> bytes:
    """Deterministic bytes for hashing a receipt — ``content_hash`` excluded
    (VIGIL ``audit/receipt.py``)."""
    data = {k: v for k, v in record_dict.items() if k != "content_hash"}
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()


def content_hash_of(record_dict: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(record_dict)).hexdigest()


def verify_chain(records: Sequence[ReceiptRecord]) -> None:
    """Raise ``ValueError`` on the first hash or parent-link inconsistency."""
    prev: ReceiptRecord | None = None
    for i, r in enumerate(records):
        expected = content_hash_of(r.as_dict())
        if r.content_hash != expected:
            raise ValueError(f"receipt {i} ({r.run_id}): content_hash mismatch")
        if prev is not None and r.parent_hash != prev.content_hash:
            raise ValueError(f"receipt {i} ({r.run_id}): parent_hash != prior content_hash")
        prev = r


# ---------------------------------------------------------------------------
# The six contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class Observe(Protocol):
    def observe(self, world: WorldState) -> Observation: ...


@runtime_checkable
class World(Protocol):
    """The sole authority for mutable scene state.

    This is deliberately separate from ``Manipulate``: an arm driver may
    report a command result, but it cannot mutate a planner's snapshot.
    """

    def state(self) -> WorldState: ...

    def start_mission(self, goal: str) -> None: ...

    def apply_transition(self, request: TransitionRequest) -> TransitionResult: ...

    def add_constraint(self, constraint: Constraint) -> None: ...


@runtime_checkable
class Plan(Protocol):
    def plan(self, goal: str, world: WorldState) -> PlanGraph: ...

    def replan(self, current: PlanGraph, world: WorldState, reason: str) -> PlanGraph: ...


@runtime_checkable
class Manipulate(Protocol):
    def supports(self, op: str) -> bool: ...

    def execute(self, step: Step, world: WorldState) -> ManipResult: ...


@runtime_checkable
class Verify(Protocol):
    def check(self, step: Step, observation: Observation) -> VerifyResult: ...


@runtime_checkable
class Device(Protocol):
    def route(self, step: Step, world: WorldState) -> str: ...

    def devices(self) -> list[DeviceSpec]: ...


@runtime_checkable
class Receipt(Protocol):
    def authorize(self, authorization: ActionAuthorization) -> ActionAuthorization: ...

    def record(
        self,
        run_id: str,
        goal: str,
        inputs: dict[str, Any],
        plan: PlanGraph,
        actions: list[dict[str, Any]],
        metrics: dict[str, Any],
        decisions: list[dict[str, Any]] | None = ...,
        rejected: list[dict[str, Any]] | None = ...,
    ) -> ReceiptRecord: ...


CONTRACTS: dict[str, type] = {
    "World": World,
    "Observe": Observe,
    "Plan": Plan,
    "Manipulate": Manipulate,
    "Verify": Verify,
    "Device": Device,
    "Receipt": Receipt,
}
