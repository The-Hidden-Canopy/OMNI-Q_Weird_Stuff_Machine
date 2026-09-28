"""Reactive control — the ontology's attention signal drives OMNI-Q.

The ontology raises ``workspace.conflict`` when a human enters an arm's zone.
This layer turns that into behaviour: **halt → re-observe → resume** without
touching the engine.

- :class:`ReactiveObserver` wraps any ``Observe`` provider (normally an
  :class:`~omni_q.ontology.OntologyObserver`). Each ``observe`` it checks the
  ontology's workspace state; on a *new* conflict it records a ``HALT`` event
  and (if given an engine) queues a constraint so the loop recompiles
  promptly; on clear it records ``RESUME``.
- :class:`ReactivePlanner` is a ``Plan`` decorator: while the observer is
  ``blocked`` every ``plan`` / ``replan`` returns a safe **wait graph**
  (``BRACE`` then ``VERIFY``) instead of the task plan, so the arms hold. When
  the conflict clears, the real plan resumes on the next recompile.

Compose it outermost:
``ReactivePlanner(ScheduledPlanner(RewritingPlanner(RulePlanner())), observer=ro)``
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .contracts import Observation, PlanGraph, Step, WorldState


@dataclass(frozen=True)
class ReactorEvent:
    kind: str            # HALT | RESUME
    reason: str
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "reason": self.reason, "ts": self.ts}


class ReactiveObserver:
    """``Observe`` wrapper that surfaces the ontology's workspace state."""

    def __init__(self, inner: Any, *, engine: Any = None) -> None:
        self.inner = inner
        self.engine = engine
        self.blocked = False
        self.reason = ""
        self.events: list[ReactorEvent] = []

    # -- Observe contract ---------------------------------------------
    def observe(self, world: WorldState) -> Observation:
        obs = self.inner.observe(world)
        conflict = not obs.workspace_clear
        if conflict and not self.blocked:
            self.blocked = True
            self.reason = self._describe()
            self.events.append(ReactorEvent("HALT", self.reason))
            self._nudge_recompile("style", "freeze",
                                  "human in workspace; arms holding")
        elif not conflict and self.blocked:
            self.blocked = False
            self.events.append(ReactorEvent("RESUME", "workspace clear"))
            self.reason = ""
            self._nudge_recompile("style", "minimum_time", "workspace clear; resume")
        return obs

    # -- helpers --------------------------------------------------
    def _describe(self) -> str:
        ont = getattr(self.inner, "ontology", None)
        if ont is None:
            return "workspace conflict"
        rels = [r for r in ont.relations
                if r.predicate in {"intersects", "near"} and "arm" in r.obj]
        if rels:
            r = rels[0]
            return f"{r.subject} {r.predicate} {r.obj}"
        return "workspace conflict"

    def _nudge_recompile(self, kind: str, value: Any, why: str) -> None:
        eng = self.engine
        if eng is None or not hasattr(eng, "add_constraint"):
            return
        try:
            eng.add_constraint(kind, value, justification=why)
        except Exception:  # noqa: BLE001 - reaction must never crash the run
            pass

    @property
    def wake(self) -> bool:
        return getattr(self.inner, "wake", False) or self.blocked


# ---------------------------------------------------------------------------


def wait_graph(goal: str, *, hold_op: str = "STABILIZE") -> PlanGraph:
    """A safe hold: keep the arms steady, then verify (which keeps failing
    while the goal is unmet, so the engine recompiles — and once the workspace
    clears, the real plan resumes)."""
    g = PlanGraph(goal=goal)
    g.steps = [
        Step("hold_for_human", "manipulate", hold_op,
             args={}, rationale="human in shared workspace — holding"),
        Step("verify_final", "verify", "VERIFY", deps=("hold_for_human",),
             rationale="re-check once the workspace is clear"),
    ]
    return g


class ReactivePlanner:
    def __init__(self, inner: Any, *, observer: ReactiveObserver) -> None:
        self.inner = inner
        self.observer = observer
        self.holding = False

    @property
    def last_decision(self) -> Any:
        return getattr(self.inner, "last_decision", None)

    # forward any decorator hooks used elsewhere (rewrite / schedule)
    def register_mutation(self, *a: Any, **k: Any) -> None:
        if hasattr(self.inner, "register_mutation"):
            self.inner.register_mutation(*a, **k)

    @property
    def can_run(self) -> Any:
        return getattr(self.inner, "can_run", None)

    @can_run.setter
    def can_run(self, v: Any) -> None:
        if hasattr(self.inner, "can_run"):
            self.inner.can_run = v

    def plan(self, goal: str, world: WorldState) -> PlanGraph:
        if self.observer.blocked:
            self.holding = True
            return wait_graph(goal)
        self.holding = False
        return self.inner.plan(goal, world)

    def replan(self, current: PlanGraph, world: WorldState, reason: str) -> PlanGraph:
        if self.observer.blocked:
            self.holding = True
            return wait_graph(current.goal)
        self.holding = False
        return self.inner.replan(current, world, reason)
