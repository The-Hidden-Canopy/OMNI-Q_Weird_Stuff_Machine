"""Fake providers for every contract.

Enough behaviour to run the whole Observe -> Plan -> Manipulate -> Verify ->
Receipt loop end-to-end with no hardware. Real providers (Intel/MuJoCo,
Qualcomm, Speechmatics) replace these one contract at a time.
"""

from __future__ import annotations

from typing import Any

from . import actions as _actions
from .contracts import (
    ActionAuthorization,
    AutonomyMode,
    Detection,
    ManipResult,
    Observation,
    PlanDecision,
    PlanGraph,
    ReceiptRecord,
    Step,
    VerifyResult,
    WorldState,
    content_hash_of,
    sha256_of,
)
from .provenance import build_provenance

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

    def __init__(self) -> None:
        self.last_decision: PlanDecision | None = None

    def plan(self, goal: str, world: WorldState) -> PlanGraph:
        graph = PlanGraph(goal=goal)
        low = goal.lower()
        if not any(k in low for k in self._ACTIONABLE):
            # Nothing recognised: at least look.
            graph.steps.append(Step("observe_0", "observe", "observe",
                                    rationale="unrecognised goal; observe only"))
            self.last_decision = PlanDecision(
                goal=goal, revision=graph.revision, selected_ops=("observe",),
                candidates_considered=0, candidates_feasible=0,
                reason="unrecognised goal; observe only",
            )
            return graph

        forbidden = world.forbidden()
        show_off = any(c.kind == "style" and c.value == "show_off"
                       for c in world.constraints)
        prefer_arm = next((c.value for c in world.constraints
                           if c.kind == "prefer_arm"), None)

        candidates = sorted(world.misplaced(), key=lambda d: d.object_id)
        rejected: dict[str, str] = {}
        move_ids: list[str] = []
        for det in candidates:
            if det.object_id in forbidden:
                rejected[det.object_id] = "forbidden by constraint"
                continue
            if not det.authoritative:
                rejected[det.object_id] = f"non-authoritative detection ({det.status.value})"
                continue
            oid = det.object_id
            # If an arm already holds this object (a replan after a PICK
            # succeeded but the following MOVE/verify failed), re-issuing the
            # PICK is a hard reject ("already held") -- carry straight from
            # the gripper instead.
            held_by = world.ownership.get(oid)
            pick = None if held_by else Step(
                f"pick_{oid}", "manipulate", "PICK",
                args={"object": oid}, arm=prefer_arm,
                rationale=f"{oid} is in {det.zone}, belongs in {det.target_zone}",
            )
            move = Step(
                f"move_{oid}", "manipulate", "MOVE",
                args={"object": oid, "to": det.target_zone},
                deps=(pick.id,) if pick else (),
                arm=prefer_arm,
                rationale=(
                    f"carry {oid} -> {det.target_zone}" if pick
                    else f"{oid} already held; place it at {det.target_zone}"
                ),
            )
            graph.steps += [pick, move] if pick else [move]
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

        gov = tuple(sorted({c.kind for c in world.constraints}))
        self.last_decision = PlanDecision(
            goal=goal,
            revision=graph.revision,
            selected_ops=tuple(s.op for s in graph.steps),
            candidates_considered=len(candidates),
            candidates_feasible=len(move_ids),
            governing_constraints=gov,
            rejected=rejected,
            state_hash=sha256_of(world.as_dict())[:16],
            reason=f"{len(move_ids)} object(s) actionable, {len(rejected)} rejected",
        )
        return graph

    def replan(self, current: PlanGraph, world: WorldState, reason: str) -> PlanGraph:
        fresh = self.plan(current.goal, world)
        fresh.revision = current.revision + 1
        if self.last_decision is not None:
            self.last_decision.revision = fresh.revision
            self.last_decision.reason = f"replan: {reason}"
        return fresh


# ---------------------------------------------------------------------------
# Manipulate
# ---------------------------------------------------------------------------


class FakeManipulator:
    _OPS = {
        "PICK", "PLACE", "MOVE", "OPEN", "CLOSE", "ROTATE", "PRESENT",
        "STABILIZE", "REGRASP", "HANDOFF", "COOPERATIVE_ROTATE", "LOCATE",
        # non-contact idle-slack flourishes (OQ-HAND-011): free-space arm
        # gestures, no object, no grasp -- executed as no-op successes here,
        # as real joint motion by intel_sim's coarse-pose path.
        *_actions.IDLE_FLOURISH_OPS, "FREEZE",
    }

    def __init__(self, world_ref: Any) -> None:
        # Kept for constructor compatibility with existing providers/tests.
        # State changes are deliberately applied by World.apply_transition(),
        # never by this executor.
        self._world = world_ref

    def supports(self, op: str) -> bool:
        return op in self._OPS

    def execute(self, step: Step, world: WorldState) -> ManipResult:
        op = step.op
        if op == "MOVE":
            obj = step.args["object"]
            zone = step.args["to"]
            return ManipResult(step.id, True, {"requested_move": obj, "to": zone})
        if op == "PICK":
            return ManipResult(step.id, True, {"requested_pick": step.args.get("object")})
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
        by_id = {d.object_id: d for d in observation.detections}
        obj_id = step.args.get("object")
        if step.op == "PICK":
            ok = obj_id in by_id
            return VerifyResult(
                ok=ok,
                expected={"object_present": obj_id},
                observed={"object_present": obj_id in by_id},
                mismatch=() if ok else (str(obj_id),),
            )
        if step.op in {"MOVE", "PLACE"}:
            target = step.args.get("to")
            observed = by_id.get(obj_id)
            ok = observed is not None and observed.zone == target
            return VerifyResult(
                ok=ok,
                expected={"object": obj_id, "zone": target},
                observed={"object": obj_id, "zone": observed.zone if observed else None},
                mismatch=() if ok else (str(obj_id),),
            )
        if step.op != "VERIFY":
            # No postcondition modelled yet for this primitive (OPEN, CLOSE,
            # ROTATE, PRESENT, HANDOFF, ...) -- trivially pass rather than
            # judging it against the "whole workspace tidy" check below,
            # which is meant for the terminal VERIFY step and is never true
            # this early in a run.
            return VerifyResult(ok=True, expected={"op": step.op}, observed={"op": step.op})
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
    """Parent-chained, self-describing receipts (FALCON ledger + VIGIL chain)."""

    def __init__(self) -> None:
        self.records: list[ReceiptRecord] = []
        self.authorizations: list[ActionAuthorization] = []

    def _last_hash(self) -> str:
        return self.records[-1].content_hash if self.records else "GENESIS"

    def authorize(self, authorization: ActionAuthorization) -> ActionAuthorization:
        """Finalize the action authorization before the engine invokes a driver."""
        base = authorization.as_dict()
        finalized = ActionAuthorization(
            run_id=authorization.run_id,
            step_id=authorization.step_id,
            op=authorization.op,
            verdict=authorization.verdict,
            reason=authorization.reason,
            state_revision=authorization.state_revision,
            envelope_digest=authorization.envelope_digest,
            content_hash=content_hash_of(base),
        )
        self.authorizations.append(finalized)
        return finalized

    def record(
        self,
        run_id: str,
        goal: str,
        inputs: dict[str, Any],
        plan: PlanGraph,
        actions: list[dict[str, Any]],
        metrics: dict[str, Any],
        decisions: list[dict[str, Any]] | None = None,
        rejected: list[dict[str, Any]] | None = None,
    ) -> ReceiptRecord:
        plan_d = plan.as_dict()
        base = ReceiptRecord(
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
            decisions=tuple(decisions or ()),
            rejected=tuple(rejected or ()),
            provenance=build_provenance("omni_q.engine"),
            parent_hash=self._last_hash(),
        )
        # freeze once more with the content hash filled in
        rec = ReceiptRecord(
            **{**base.as_dict(), "content_hash": content_hash_of(base.as_dict())}
        )
        self.records.append(rec)
        return rec
