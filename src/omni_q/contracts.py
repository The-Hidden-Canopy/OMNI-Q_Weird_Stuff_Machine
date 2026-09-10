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
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Shared value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    object_id: str
    cls: str
    zone: str
    target_zone: str
    conf: float = 0.95

    @property
    def misplaced(self) -> bool:
        return self.zone != self.target_zone


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

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": self.value}


@dataclass
class WorldState:
    frame: int
    objects: dict[str, Detection]
    goal: str | None = None
    constraints: tuple[Constraint, ...] = ()

    def misplaced(self) -> list[Detection]:
        return [o for o in self.objects.values() if o.misplaced]

    def forbidden(self) -> set[str]:
        return {c.value for c in self.constraints if c.kind == "forbid_object"}

    def local_only(self) -> bool:
        return any(c.kind == "keep_local" for c in self.constraints)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "goal": self.goal,
            "constraints": [c.as_dict() for c in self.constraints],
            "objects": {k: asdict(v) for k, v in self.objects.items()},
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
    state: str = "pending"             # pending | running | done | failed | skipped
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


@dataclass(frozen=True)
class ReceiptRecord:
    run_id: str
    goal: str
    inputs: dict[str, Any]
    plan: dict[str, Any]
    actions: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    hashes: dict[str, str]
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "goal": self.goal, "inputs": self.inputs,
            "plan": self.plan, "actions": list(self.actions),
            "metrics": self.metrics, "hashes": self.hashes, "ts": self.ts,
        }


def sha256_of(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# The six contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class Observe(Protocol):
    def observe(self, world: WorldState) -> Observation: ...


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
    def record(
        self,
        run_id: str,
        goal: str,
        inputs: dict[str, Any],
        plan: PlanGraph,
        actions: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> ReceiptRecord: ...


CONTRACTS: dict[str, type] = {
    "Observe": Observe,
    "Plan": Plan,
    "Manipulate": Manipulate,
    "Verify": Verify,
    "Device": Device,
    "Receipt": Receipt,
}
