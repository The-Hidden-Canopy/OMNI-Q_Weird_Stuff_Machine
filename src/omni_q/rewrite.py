"""OQ-HAND-006 + OQ-025 — graph rewrites over the action vocabulary.

- :func:`optimize` — collapse a PICK + MOVE (+ PRESENT) chain into a cheaper
  contact op when the object barely needs to move: same zone -> ``NUDGE``,
  same table surface (no lift / no handoff) -> ``SLIDE``, otherwise leave the
  PICK/PLACE alone.
- :func:`apply_mutation` — turn a parsed ``nlu`` structural mutation
  (``spin`` / ``spin_on_place`` / ``nudge``) into a live graph edit, so those
  spoken commands actually execute instead of being deferred.
- :class:`RewritingPlanner` — a ``Plan`` decorator that runs ``optimize`` plus
  any registered mutations on every ``plan`` / ``replan``. Compose it under
  ``ScheduledPlanner``.

Pure transforms on ``PlanGraph`` + ``WorldState``; no engine / contracts edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from typing import Callable

from .contracts import PlanGraph, Step, WorldState
from .scheduler import DEFAULT_LAYOUT, Region

_SURFACE_SPAN = 0.35   # |x_a - x_b| under this = one arm reaches both, no handoff
_TERMINAL_MOVE = {"MOVE", "PLACE", "SLIDE", "DRAG", "CENTER_ON_MARK"}

CanRun = Callable[[str], bool]


def _ok(can_run: CanRun | None, *ops: str) -> bool:
    return can_run is None or all(can_run(o) for o in ops)


@dataclass(frozen=True)
class Rewrite:
    kind: str                 # slide | nudge | spin | spin_on_place
    object: str
    replaced: tuple[str, ...]
    added: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "object": self.object,
                "replaced": list(self.replaced), "added": list(self.added),
                "reason": self.reason}


def _region(zone: str | None, layout: dict[str, Region]) -> Region:
    if zone is None:
        return Region("center", 0.0)
    return layout.get(zone, Region(zone, 0.0))


def _same_surface(a: str | None, b: str | None, layout: dict[str, Region]) -> bool:
    ra, rb = _region(a, layout), _region(b, layout)
    if abs(ra.x_center - rb.x_center) > _SURFACE_SPAN:
        return False
    # opposite far sides need a lift/handoff
    return not (ra.x_center * rb.x_center < 0
                and min(abs(ra.x_center), abs(rb.x_center)) > 0.12)


def _object_chain(graph: PlanGraph, oid: str) -> list[Step]:
    return [s for s in graph.steps
            if s.contract == "manipulate" and s.args.get("object") == oid]


def _reparent(steps: list[Step], old: set[str], new: str) -> list[Step]:
    out = []
    for s in steps:
        if old & set(s.deps):
            deps = tuple(sorted((set(s.deps) - old) | {new}))
            out.append(replace(s, deps=deps))
        else:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# OQ-HAND-006 — cheaper op when the move is small
# ---------------------------------------------------------------------------


def optimize(graph: PlanGraph, world: WorldState, *,
             layout: dict[str, Region] | None = None,
             can_run: CanRun | None = None,
             ) -> tuple[PlanGraph, list[Rewrite]]:
    layout = layout or DEFAULT_LAYOUT
    steps = list(graph.steps)
    rewrites: list[Rewrite] = []

    # objects that have a pick+terminal-move pair
    objs = sorted({s.args["object"] for s in steps
                   if s.contract == "manipulate" and s.op == "PICK"
                   and "object" in s.args})

    for oid in objs:
        chain = _object_chain(graph, oid)
        pick = next((s for s in chain if s.op == "PICK"), None)
        move = next((s for s in chain if s.op in _TERMINAL_MOVE), None)
        if pick is None or move is None:
            continue
        if world.ownership.get(oid) is not None:
            continue                                   # held -> can't slide/nudge
        cur = world.objects[oid].zone if oid in world.objects else None
        dst = move.args.get("to")

        if cur is not None and dst == cur and _ok(can_run, "NUDGE"):
            new_op, kind = "NUDGE", "nudge"
            new_args = {"object": oid, "direction": "in_place", "amount": "a bit"}
            reason = f"{oid} already in {dst}; NUDGE not PICK+PLACE"
        elif _same_surface(cur, dst, layout) and _ok(can_run, "SLIDE"):
            new_op, kind = "SLIDE", "slide"
            new_args = {"object": oid, "to": dst}
            reason = f"{cur}->{dst} same surface; SLIDE not PICK+MOVE"
        else:
            continue                                   # keep the lift

        removed = {pick.id, move.id}
        new_id = f"{kind}_{oid}"
        new_step = Step(new_id, "manipulate", new_op, args=new_args,
                        deps=tuple(sorted(set(pick.deps) | (set(move.deps) - removed))),
                        rationale=reason)
        steps = [s for s in steps if s.id not in removed]
        steps = _reparent(steps, removed, new_id)
        steps.append(new_step)
        rewrites.append(Rewrite(kind, oid, tuple(sorted(removed)), (new_id,), reason))

    if not rewrites:
        return graph, []
    out = PlanGraph(goal=graph.goal, revision=graph.revision)
    out.steps = _topo_ish(steps)
    return out, rewrites


def _topo_ish(steps: list[Step]) -> list[Step]:
    """Stable order: dependency-respecting, else original-ish."""
    by_id = {s.id: s for s in steps}
    done: list[str] = []
    seen: set[str] = set()

    def visit(sid: str) -> None:
        if sid in seen or sid not in by_id:
            return
        seen.add(sid)
        for d in by_id[sid].deps:
            visit(d)
        done.append(sid)

    for s in steps:
        visit(s.id)
    return [by_id[i] for i in done]


# ---------------------------------------------------------------------------
# OQ-025 — spoken structural mutations become graph edits
# ---------------------------------------------------------------------------


def apply_mutation(graph: PlanGraph, world: WorldState, kind: str,
                   payload: dict[str, Any], *, can_run: CanRun | None = None
                   ) -> tuple[PlanGraph, list[Rewrite]]:
    if kind == "spin" and _ok(can_run, "SPIN"):
        return _spin(graph, [payload["object"]], payload.get("degrees", 180))
    if kind == "spin_on_place" and _ok(can_run, "SPIN"):
        cls = payload["object_class"]
        oids = [oid for oid, d in world.objects.items() if d.cls == cls]
        return _spin(graph, oids, payload.get("degrees", 180))
    if kind == "nudge" and _ok(can_run, "NUDGE"):
        return _nudge(graph, payload)
    return graph, []


def _spin(graph: PlanGraph, oids: list[str], degrees: int
          ) -> tuple[PlanGraph, list[Rewrite]]:
    steps = list(graph.steps)
    rewrites: list[Rewrite] = []
    for oid in sorted(set(oids)):
        chain = _object_chain(graph, oid)
        move = next((s for s in chain if s.op in _TERMINAL_MOVE), None)
        if move is None:
            continue
        spin_id = f"spin_{oid}"
        if any(s.id == spin_id for s in steps):
            continue
        # SPIN runs in-hand, just before the terminal move; move now deps on it
        spin_step = Step(spin_id, "manipulate", "SPIN",
                         args={"object": oid, "degrees": degrees},
                         deps=tuple(d for d in move.deps),
                         rationale=f"spoken: spin {oid} {degrees} before placing")
        steps = [replace(s, deps=(spin_id,) + tuple(x for x in s.deps if x != spin_id))
                 if s.id == move.id else s for s in steps]
        steps.append(spin_step)
        rewrites.append(Rewrite("spin", oid, (), (spin_id,),
                                f"insert SPIN({degrees}) before {move.id}"))
    if not rewrites:
        return graph, []
    out = PlanGraph(goal=graph.goal, revision=graph.revision)
    out.steps = _topo_ish(steps)
    return out, rewrites


def _nudge(graph: PlanGraph, payload: dict[str, Any]
           ) -> tuple[PlanGraph, list[Rewrite]]:
    oid = payload["object"]
    direction = payload.get("direction", "left")
    amount = payload.get("amount", "a bit")
    steps = list(graph.steps)
    chain = _object_chain(graph, oid)
    move = next((s for s in chain if s.op in _TERMINAL_MOVE), None)
    pick = next((s for s in chain if s.op == "PICK"), None)
    nid = f"nudge_{oid}"

    if move is not None and pick is not None:
        removed = {pick.id, move.id}
        new = Step(nid, "manipulate", "NUDGE",
                   args={"object": oid, "direction": direction, "amount": amount},
                   deps=tuple(sorted(set(pick.deps) | (set(move.deps) - removed))),
                   rationale=f"spoken: nudge {oid} {amount} {direction}")
        steps = _reparent([s for s in steps if s.id not in removed], removed, nid)
        steps.append(new)
        r = Rewrite("nudge", oid, tuple(sorted(removed)), (nid,),
                    f"replace PICK+MOVE with NUDGE {direction}")
    else:
        # standalone late adjustment; verify waits on it
        verify = next((s for s in steps if s.contract == "verify"), None)
        deps = tuple(verify.deps) if verify else ()
        new = Step(nid, "manipulate", "NUDGE",
                   args={"object": oid, "direction": direction, "amount": amount},
                   deps=deps, rationale=f"spoken: nudge {oid} {amount} {direction}")
        steps.append(new)
        if verify is not None:
            steps = [replace(s, deps=tuple(sorted(set(s.deps) | {nid})))
                     if s.id == verify.id else s for s in steps]
        r = Rewrite("nudge", oid, (), (nid,), f"add NUDGE {direction} before verify")

    out = PlanGraph(goal=graph.goal, revision=graph.revision)
    out.steps = _topo_ish(steps)
    return out, [r]


# ---------------------------------------------------------------------------
# live integration — a Plan decorator
# ---------------------------------------------------------------------------


class RewritingPlanner:
    """Runs :func:`optimize` and any registered mutations on every plan."""

    def __init__(self, inner: Any, *, optimize_chains: bool = True,
                 can_run: CanRun | None = None) -> None:
        self.inner = inner
        self.optimize_chains = optimize_chains
        self.can_run = can_run          # e.g. engine.manipulator.supports
        self._mutations: list[tuple[str, dict[str, Any]]] = []
        self.last_rewrites: list[Rewrite] = []
        self.last_error: str | None = None

    @property
    def last_decision(self) -> Any:
        return getattr(self.inner, "last_decision", None)

    def register_mutation(self, kind: str, payload: dict[str, Any]) -> None:
        self._mutations.append((kind, payload))

    def clear_mutations(self) -> None:
        self._mutations.clear()

    def _apply(self, graph: PlanGraph, world: WorldState) -> PlanGraph:
        self.last_rewrites = []
        try:
            if self.optimize_chains:
                graph, rw = optimize(graph, world, can_run=self.can_run)
                self.last_rewrites += rw
            for kind, payload in self._mutations:
                graph, rw = apply_mutation(graph, world, kind, payload,
                                           can_run=self.can_run)
                self.last_rewrites += rw
            self.last_error = None
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the run
            self.last_error = f"{type(exc).__name__}: {exc}"
        return graph

    def plan(self, goal: str, world: WorldState) -> PlanGraph:
        return self._apply(self.inner.plan(goal, world), world)

    def replan(self, current: PlanGraph, world: WorldState, reason: str) -> PlanGraph:
        return self._apply(self.inner.replan(current, world, reason), world)
