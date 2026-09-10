"""Fake providers for every contract.

Enough behaviour to run the whole Observe -> Plan -> Manipulate -> Verify ->
Receipt loop end-to-end with no hardware. Real providers (Intel/MuJoCo,
Qualcomm, Speechmatics) replace these one contract at a time.
"""

from __future__ import annotations

from typing import Any

from .contracts import (
    Detection,
    ManipResult,
    Observation,
    PlanGraph,
    ReceiptRecord,
    Step,
    VerifyResult,
    WorldState,
    sha256_of,
)

# ---------------------------------------------------------------------------
# Observe
# ---------------------------------------------------------------------------


class FakeObserver:
    """Turns world objects into a compact Observation (never raw video)."""

    def observe(self, world: WorldState) -> Observation:
        dets = tuple(sorted(world.objects.values(), key=lambda d: d.object_id))
        return Observation(
            frame=world.frame,
            detections=dets,
            workspace_clear=not any(d.misplaced for d in dets),
            raw_ref=f"frame://mock/{world.frame:06d}",
        )


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class RulePlanner:
    """Rule-based planner. Pluggable stand-in for a GenieX / LLM planner.

    "correct / tidy / put away / set" goals -> for each misplaced object:
    ``PICK`` then ``MOVE`` to its target zone, then a final ``VERIFY``.
    Honours ``forbid_object`` and a ``style=show_off`` flourish (OQ-015 seed).
    """

    _ACTIONABLE = ("correct", "tidy", "put away", "clear", "set", "inspect")

    def plan(self, goal: str, world: WorldState) -> PlanGraph:
        graph = PlanGraph(goal=goal)
        low = goal.lower()
        if not any(k in low for k in self._ACTIONABLE):
            # Nothing recognised: at least look.
            graph.steps.append(Step("observe_0", "observe", "observe",
                                    rationale="unrecognised goal; observe only"))
            return graph

        forbidden = world.forbidden()
        show_off = any(c.kind == "style" and c.value == "show_off"
                       for c in world.constraints)
        prefer_arm = next((c.value for c in world.constraints
                           if c.kind == "prefer_arm"), None)

        move_ids: list[str] = []
        for det in sorted(world.misplaced(), key=lambda d: d.object_id):
            if det.object_id in forbidden:
                continue
            oid = det.object_id
            pick = Step(
                f"pick_{oid}", "manipulate", "PICK",
                args={"object": oid}, arm=prefer_arm,
                rationale=f"{oid} is in {det.zone}, belongs in {det.target_zone}",
            )
            move = Step(
                f"move_{oid}", "manipulate", "MOVE",
                args={"object": oid, "to": det.target_zone},
                deps=(pick.id,), arm=prefer_arm,
                rationale=f"carry {oid} -> {det.target_zone}",
            )
            graph.steps += [pick, move]
            move_ids.append(move.id)
            if show_off:
                graph.steps.append(Step(
                    f"present_{oid}", "manipulate", "PRESENT",
                    args={"object": oid}, deps=(move.id,),
                    rationale="style=show_off flourish; final goal unchanged",
                ))

        graph.steps.append(Step(
            "verify_final", "verify", "VERIFY",
            deps=tuple(move_ids),
            rationale="re-check workspace after action",
        ))
        return graph

    def replan(self, current: PlanGraph, world: WorldState, reason: str) -> PlanGraph:
        fresh = self.plan(current.goal, world)
        fresh.revision = current.revision + 1
        return fresh


# ---------------------------------------------------------------------------
# Manipulate
# ---------------------------------------------------------------------------


class FakeManipulator:
    _OPS = {
        "PICK", "PLACE", "MOVE", "OPEN", "CLOSE", "ROTATE", "PRESENT",
        "STABILIZE", "REGRASP", "HANDOFF", "COOPERATIVE_ROTATE", "LOCATE",
    }

    def __init__(self, world_ref: Any) -> None:
        # world_ref is the mutable MockWorld so MOVE has an effect.
        self._world = world_ref

    def supports(self, op: str) -> bool:
        return op in self._OPS

    def execute(self, step: Step, world: WorldState) -> ManipResult:
        op = step.op
        if op == "MOVE":
            obj = step.args["object"]
            zone = step.args["to"]
            self._world.move_object(obj, zone)
            return ManipResult(step.id, True, {"moved": obj, "to": zone})
        if op == "PICK":
            return ManipResult(step.id, True, {"grasped": step.args.get("object")})
        if op == "LOCATE":
            return ManipResult(step.id, True,
                               {"misplaced": [d.object_id for d in world.misplaced()]})
        # every other primitive is a no-op success in mock mode
        return ManipResult(step.id, True, {"op": op, "args": step.args})


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


class FakeVerifier:
    def check(self, step: Step, observation: Observation) -> VerifyResult:
        remaining = [d.object_id for d in observation.misplaced()]
        expected = {"workspace_clear": True}
        observed = {"workspace_clear": observation.workspace_clear,
                    "remaining": remaining}
        return VerifyResult(
            ok=not remaining,
            expected=expected,
            observed=observed,
            mismatch=tuple(remaining),
        )


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


class FakeRecorder:
    def __init__(self) -> None:
        self.records: list[ReceiptRecord] = []

    def record(
        self,
        run_id: str,
        goal: str,
        inputs: dict[str, Any],
        plan: PlanGraph,
        actions: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> ReceiptRecord:
        plan_d = plan.as_dict()
        rec = ReceiptRecord(
            run_id=run_id,
            goal=goal,
            inputs=inputs,
            plan=plan_d,
            actions=tuple(actions),
            metrics=metrics,
            hashes={
                "inputs": sha256_of(inputs),
                "plan": sha256_of(plan_d),
                "actions": sha256_of(actions),
            },
        )
        self.records.append(rec)
        return rec
