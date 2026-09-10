"""OmniQ engine — wires the six contracts into one loop.

    observe -> plan -> (route -> manipulate -> observe -> verify)* -> receipt

Recompiles the graph when a constraint is added, a capability disappears, a step
fails, or verification finds a mismatch. Every transition is published on the
:class:`~omni_q.events.EventBus` for the UI (OQ-005).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from .contracts import (
    AutonomyMode,
    Constraint,
    Device,
    Manipulate,
    MissionEnvelope,
    Observe,
    Plan,
    PlanGraph,
    ReceiptRecord,
    Step,
    Verify,
    WorldState,
    merge_mode,
)
from .events import EventBus


class OmniQ:
    def __init__(
        self,
        *,
        world: Any,               # MockWorld or a real world adapter
        observer: Observe,
        planner: Plan,
        manipulator: Manipulate,
        verifier: Verify,
        device: Device,
        recorder: Any,
        bus: EventBus | None = None,
        envelope: MissionEnvelope | None = None,
        max_revisions: int | None = None,
    ) -> None:
        self.world = world
        self.observer = observer
        self.planner = planner
        self.manipulator = manipulator
        self.verifier = verifier
        self.device = device
        self.recorder = recorder
        self.bus = bus or EventBus()
        self.envelope = envelope or MissionEnvelope(mission_id="mock")
        self.max_revisions = (
            max_revisions if max_revisions is not None else self.envelope.max_revisions
        )

        self.graph: PlanGraph | None = None
        self.mode: AutonomyMode = AutonomyMode.NOMINAL
        self._pending_constraints: list[Constraint] = []
        self._actions: list[dict[str, Any]] = []
        self._decisions: list[dict[str, Any]] = []

    # -- external control (Speechmatics / UI feed into these) ----------
    def add_constraint(self, kind: str, value: Any = None) -> None:
        self._pending_constraints.append(Constraint(kind, value))
        self.bus.publish("constraint.queued", kind=kind, value=value)

    def _apply_constraints(self) -> bool:
        if not self._pending_constraints:
            return False
        for c in self._pending_constraints:
            self.world.add_constraint(c)
            self.bus.publish("constraint.added", kind=c.kind, value=c.value)
            if c.kind == "keep_local":
                self._escalate(AutonomyMode.LOCAL_ONLY)
        self._pending_constraints.clear()
        return True

    def _escalate(self, incoming: AutonomyMode) -> None:
        merged = merge_mode(self.mode, incoming)
        if merged is not self.mode:
            self.bus.publish("mode.changed", frm=self.mode.value, to=merged.value)
            self.mode = merged

    def _capture_decision(self) -> None:
        dec = getattr(self.planner, "last_decision", None)
        if dec is None:
            return
        dec.mode = self.mode
        dec.envelope_digest = self.envelope.digest()
        d = dec.as_dict()
        self._decisions.append(d)
        self.bus.publish("plan.decision", **d)

    def _recompile(self, world: WorldState, reason: str) -> None:
        self.graph = self.planner.replan(self.graph, world, reason)
        self._capture_decision()
        self._emit_graph("graph.recompiled", reason=reason)

    # -- run --------------------------------------------------------
    def run(self, goal: str) -> ReceiptRecord:
        run_id = uuid.uuid4().hex[:12]
        started = time.time()
        self.world.goal = goal
        self._actions = []
        self._decisions = []
        self.bus.publish("run.started", run_id=run_id, goal=goal,
                         envelope=self.envelope.digest())

        world = self.world.state()
        obs = self.observer.observe(world)
        self.bus.publish("observed", frame=obs.frame,
                         misplaced=[d.object_id for d in obs.misplaced()])

        self.graph = self.planner.plan(goal, world)
        self._capture_decision()
        self._emit_graph("graph.compiled")

        revisions = 0
        executed: set[str] = set()

        while True:
            if self._apply_constraints():
                revisions += 1
                self._recompile(self.world.state(), "constraint change")
                executed.clear()

            step = self._next_step(executed)
            if step is None:
                break

            world = self.world.state()

            # capability lost? -> recompile onto what remains
            if step.contract == "manipulate" and not self.manipulator.supports(step.op):
                revisions += 1
                self.bus.publish("capability.lost", op=step.op, step=step.id)
                self._escalate(AutonomyMode.DEGRADED)
                if revisions > self.max_revisions:
                    self._escalate(AutonomyMode.HOLD)
                    break
                self._recompile(world, f"lost {step.op}")
                executed.clear()
                continue

            try:
                step.device = self.device.route(step, world)
            except RuntimeError as exc:
                revisions += 1
                self.bus.publish("placement.failed", step=step.id, error=str(exc))
                self._escalate(AutonomyMode.DEGRADED)
                if revisions > self.max_revisions:
                    self._escalate(AutonomyMode.HOLD)
                    break
                self._recompile(world, "placement failed")
                executed.clear()
                continue

            outcome = self._run_step(step, world)
            executed.add(step.id)

            if not outcome:
                revisions += 1
                if revisions > self.max_revisions:
                    self._escalate(AutonomyMode.HOLD)
                    break
                self._recompile(self.world.state(), f"{step.id} failed")
                executed.clear()
                continue

            if step.contract == "verify":
                fresh_obs = self.observer.observe(self.world.state())
                result = self.verifier.check(step, fresh_obs)
                self.bus.publish("verified", step=step.id, ok=result.ok,
                                 mismatch=list(result.mismatch))
                if not result.ok:
                    revisions += 1
                    if revisions > self.max_revisions:
                        self._escalate(AutonomyMode.HOLD)
                        break
                    self._recompile(self.world.state(), "verification mismatch")
                    executed.clear()
                    continue

        metrics = {
            "revisions": revisions,
            "steps_executed": len(self._actions),
            "wall_seconds": round(time.time() - started, 4),
            "resolved": not self.world.state().misplaced(),
            "mode": self.mode.value,
        }
        rejected = _merge_rejected(self._decisions)
        receipt = self.recorder.record(
            run_id=run_id,
            goal=goal,
            inputs={"goal": goal, "world0": {"frame": obs.frame},
                    "envelope": self.envelope.digest()},
            plan=self.graph,
            actions=self._actions,
            metrics=metrics,
            decisions=self._decisions,
            rejected=rejected,
        )
        self.bus.publish("run.finished", run_id=run_id, metrics=metrics,
                         content_hash=receipt.content_hash)
        return receipt

    # -- helpers --------------------------------------------------
    def _next_step(self, executed: set[str]) -> Step | None:
        for step in self.graph.topo_order():
            if step.id in executed:
                continue
            if all(dep in executed for dep in step.deps):
                return step
        return None

    def _run_step(self, step: Step, world: WorldState) -> bool:
        step.state = "running"
        self.bus.publish("step.started", **step.as_dict())
        if step.contract == "manipulate":
            res = self.manipulator.execute(step, world)
            step.result = res.detail
            step.state = "done" if res.ok else "failed"
            ok = res.ok
        else:
            step.result = {"noted": step.op}
            step.state = "done"
            ok = True
        self._actions.append({
            "step": step.id, "op": step.op, "arm": step.arm,
            "device": step.device, "state": step.state, "result": step.result,
        })
        self.bus.publish("step.finished", **step.as_dict())
        return ok

    def _emit_graph(self, kind: str, **extra: Any) -> None:
        self.bus.publish(kind, graph=self.graph.as_dict(), **extra)


def _merge_rejected(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the per-decision ``rejected`` maps into a de-duplicated list —
    kept in the receipt so the trace shows what Omni Q chose *not* to do."""
    seen: dict[tuple[str, str], None] = {}
    for d in decisions:
        for target, why in (d.get("rejected") or {}).items():
            seen.setdefault((target, why), None)
    return [{"target": t, "reason": w} for (t, w) in seen]
