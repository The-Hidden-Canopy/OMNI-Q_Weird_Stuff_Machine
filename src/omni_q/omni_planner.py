"""OmniPlanner — a reasoner-advising ``Plan`` contract implementation.

The reasoner (see :mod:`omni_q.omni_reasoner`) receives the NL goal plus a
JSON rendering of the structured world state — YOLO's detections as symbolic
evidence, never raw frames — and answers in a constrained grammar::

    PLAN
    STEP PICK object=<id> [arm=left|right]
    STEP MOVE object=<id> to=<zone>
    STEP ROTATE object=<id>
    STEP PRESENT object=<id>
    STEP VERIFY
    END
    RATIONALE: <one sentence, judge-facing>

Governance boundary: the model *advises*; the governed core decides. Every
model-proposed step is validated against the world before it enters the
graph — known objects, known zones, op allowlist, ``forbid_object`` and
detection authority honoured exactly as the deterministic planner honours
them. Invalid proposals land in ``PlanDecision.rejected`` with reasons, and
a model that proposes nothing usable yields to the wrapped deterministic
planner with the decision truthfully labeled as fallback. Model rationale
rides the existing ``PlanDecision.reason`` channel into receipts and the
UI's "why" display (OQ-047).

**State-tracking validation, added 2026-09-11 after composing this planner
with real FrameObserver+OpenVINO vision for the first time** (see
``docs/oq-omni-vision-integration-2026-09-11.md`` for the full writeup).
``RulePlanner`` never has to un-learn a completed step -- it generates
every candidate fresh from ``world.misplaced()``/``world.ownership`` each
call, so a done object simply never appears again. A model instead
proposes from its own belief about the goal, carried turn to turn in its
own context, which may lag the world by a step or more even when the
model is behaving reasonably (its last proposal hasn't been confirmed
successful yet when it drafts the next one). Three real, measured gaps
this exposed, all now checked here rather than trusted to the model:

1. A stale PICK on an object already held by an arm. The world layer
   already refuses this correctly (raises: "<oid> is already held"), but
   letting a stale proposal reach that refusal burns a full revision
   every time it repeats -- measured 6 of 7 revisions spent this way in
   one run. Now rejected here first, mirroring how ``RulePlanner``
   ``fakes.py`` already "carries straight from the gripper instead" of
   re-issuing a PICK it doesn't need.
2. A MOVE (or other op) on a held object routed to the wrong arm because
   the model omitted ``arm=`` or guessed. There is no ambiguity to
   resolve here -- only the arm actually holding the object can act on
   it -- so this is corrected to the true holder, not merely rejected.
3. A PICK/MOVE on an object that has already reached its target zone.
   Ownership has been released by then, so nothing else catches this;
   left unchecked it becomes a real, wasted (or disruptive) re-attempt on
   a finished object rather than a governed no-op.

None of these are hypothetical: all three were found by actually running
the composed pipeline (real render -> real OpenVINO detection -> real
camera-geometry zone mapping -> this planner -> real MuJoCo IK execution)
end to end for the first time, not by inspection. With all three fixed,
that same composition produces a genuine real PICK+MOVE success for
``cup_1`` end to end, then a clean, correctly-labeled fallback to the
deterministic scheduled planner once the (deliberately static, for this
test) mock reasoner's proposal is exhausted.
"""

from __future__ import annotations

import json
from typing import Any

from .contracts import PlanDecision, PlanGraph, Step, WorldState, sha256_of
from .fakes import RulePlanner
from .omni_reasoner import Reasoner, ReasonerResult, ReasonerUnavailable

_PICK_LIKE = {"PICK", "PLACE", "MOVE", "ROTATE", "PRESENT"}
_ALLOWED_OPS = _PICK_LIKE | {"VERIFY"}
_MANIPULATE = {"PICK", "PLACE", "MOVE", "ROTATE", "PRESENT"}


class OmniPlanner:
    """Wrap a deterministic planner with reasoner advice.

    ``fallback`` defaults to :class:`RulePlanner`; the Intel suite passes a
    ``ScheduledPlanner``-wrapped ``IntelTablePlanner`` the same way
    ``demo_intel_sim`` decorates the scheduler.
    """

    def __init__(
        self,
        reasoner: Reasoner,
        fallback: Any = None,
        *,
        max_new_tokens: int = 64,
        session_prefix: str = "omniq",
    ) -> None:
        self.reasoner = reasoner
        self.fallback = fallback if fallback is not None else RulePlanner()
        self.max_new_tokens = max_new_tokens
        self.session_prefix = session_prefix
        self.last_decision: PlanDecision | None = None
        self._turn = 0

    # -- prompt ------------------------------------------------------
    def _render_prompt(self, goal: str, world: WorldState, reason: str | None) -> str:
        known_zones = sorted({d.zone for d in world.objects.values()}
                             | {d.target_zone for d in world.objects.values()})
        objects = {
            oid: {"class": d.cls, "zone": d.zone, "target_zone": d.target_zone,
                  "misplaced": d.misplaced, "status": d.status.value}
            for oid, d in sorted(world.objects.items())
        }
        constraints = [c.as_dict() for c in world.constraints]
        lines = [
            "You advise a robot task planner. Propose steps only for objects",
            "and zones listed below. Reply in exactly this grammar:",
            "",
            "PLAN",
            "STEP PICK object=<id> [arm=left|right]",
            "STEP MOVE object=<id> to=<zone>",
            "STEP ROTATE object=<id>",
            "STEP PRESENT object=<id>",
            "STEP VERIFY",
            "END",
            "RATIONALE: <one sentence>",
            "",
            f"goal: {goal}",
            f"objects: {json.dumps(objects, sort_keys=True)}",
            f"known_zones: {known_zones}",
            f"constraints: {json.dumps(constraints, sort_keys=True, default=str)}",
        ]
        if reason:
            lines.append(f"replan_reason: {reason}")
        return "\n".join(lines)

    # -- parsing -----------------------------------------------------
    def _parse(self, text: str) -> tuple[list[dict[str, str]], str]:
        steps: list[dict[str, str]] = []
        rationale = ""
        in_plan = False
        for raw in text.splitlines():
            line = raw.strip()
            upper = line.upper()
            if upper == "PLAN":
                in_plan = True
                continue
            if upper in {"END", "END PLAN"}:
                in_plan = False
                continue
            if upper.startswith("RATIONALE:"):
                rationale = line.split(":", 1)[1].strip()
                continue
            if in_plan and upper.startswith("STEP "):
                parts = line[5:].split()
                if not parts:
                    continue
                op = parts[0].upper()
                args: dict[str, str] = {}
                for token in parts[1:]:
                    if "=" in token:
                        key, value = token.split("=", 1)
                        args[key.strip()] = value.strip()
                steps.append({"op": op, **args})
        return steps, rationale

    # -- validation --------------------------------------------------
    def _validate(self, proposals: list[dict[str, str]], world: WorldState,
                  ) -> tuple[list[Step], dict[str, str]]:
        forbidden = world.forbidden()
        zones = {d.zone for d in world.objects.values()} | {
            d.target_zone for d in world.objects.values()}
        steps: list[Step] = []
        rejected: dict[str, str] = {}
        seq = 0
        for prop in proposals:
            op = prop.get("op", "")
            label = f"omni:{op.lower()}:{prop.get('object', seq)}"
            if op not in _ALLOWED_OPS:
                rejected[label] = f"op {op or '?'} not in allowlist"
                continue
            if op == "VERIFY":
                steps.append(Step("verify_final", "verify", "VERIFY",
                                  rationale="model-proposed verification"))
                continue
            oid = prop.get("object", "")
            det = world.objects.get(oid)
            if det is None:
                rejected[label] = f"unknown object {oid!r}"
                continue
            if oid in forbidden:
                rejected[label] = "forbidden by constraint"
                continue
            if not det.authoritative:
                rejected[label] = f"non-authoritative detection ({det.status.value})"
                continue
            # A model that hasn't (yet) noticed a prior PICK already
            # succeeded will keep proposing it again on every replan --
            # the world layer correctly rejects a re-pick of something
            # already held (TransitionRejected: "<oid> is already held"),
            # but with no check here that stale proposal burns a whole
            # revision cycle every single time, exhausting max_revisions
            # on a repeatedly-rejected no-op instead of ever reaching a
            # step that would actually make progress. Measured directly:
            # composing this planner with real FrameObserver+OpenVINO
            # vision and a scripted model response reproduced exactly
            # this -- 6 of 7 revisions spent on an already-held re-pick.
            # RulePlanner already has the right answer for this
            # (fakes.py: "carry straight from the gripper instead" --
            # skip the PICK, not the whole action) -- mirror it here
            # instead of trusting the model to always track state
            # perfectly turn to turn.
            if op == "PICK" and world.ownership.get(oid):
                rejected[label] = f"{oid} is already held; PICK is redundant"
                continue
            # Third instance of the same class of gap: RulePlanner never
            # proposes PICK/MOVE for an object outside world.misplaced()
            # in the first place, because it generates candidates FROM
            # that set -- it has nothing to un-learn. A model instead
            # proposes from its own (possibly stale) belief about the
            # goal, so once an object is genuinely placed a model that
            # hasn't noticed yet will propose PICK/MOVE for it again --
            # unlike the already-held case above, ownership has been
            # released by then, so this passes every other check and
            # becomes a REAL grasp/carry attempt on an object that has
            # nothing left to do, wasting a revision on non-error work
            # (or worse, disturbing a correctly-placed object). Measured
            # directly: after a real PICK+MOVE genuinely placed cup_1
            # (real vision, real IK, both succeeded), the next replan's
            # static proposal re-picked it anyway and failed for real.
            if op in {"PICK", "MOVE"} and not det.misplaced:
                rejected[label] = f"{oid} is already at its target zone; {op} is redundant"
                continue
            arm = prop.get("arm")
            if arm is not None and arm not in {"left", "right"}:
                rejected[label] = f"invalid arm {arm!r}"
                continue
            # A second, distinct instance of the same class of gap: once an
            # object is held, only the arm actually holding it can act on
            # it -- there's no ambiguity to resolve, so a proposal for the
            # WRONG arm (the model omitted `arm=`, defaulting elsewhere, or
            # simply guessed) isn't a judgment call to reject, it's a fact
            # to correct. Left uncorrected, the world layer rejects it
            # every time ("<oid> is held by intel.right_arm, not
            # intel.left_arm") and the run burns its whole budget on a
            # step that can never succeed as proposed -- measured directly
            # composing this planner with real vision: a MOVE with no
            # `arm=` defaulted to "left" while cup_1 was held by the right
            # arm, and 6 of 7 revisions were spent on that one mismatch.
            holder = world.ownership.get(oid)
            if holder:
                holder_arm = "left" if "left" in holder else "right" if "right" in holder else None
                if holder_arm and arm != holder_arm:
                    arm = holder_arm
            args: dict[str, Any] = {"object": oid}
            if op == "MOVE":
                to = prop.get("to", "")
                if to not in zones:
                    rejected[label] = f"unknown zone {to!r}"
                    continue
                args["to"] = to
            steps.append(Step(f"omni_{op.lower()}_{oid}_{seq}",
                              "manipulate", op, args=args, arm=arm,
                              rationale="model-proposed step, core-validated"))
            seq += 1
        return steps, rejected

    @staticmethod
    def _link(steps: list[Step]) -> None:
        """Linear dependency chain; VERIFY closes the graph."""
        verifies = [s for s in steps if s.op == "VERIFY"]
        work = [s for s in steps if s.op != "VERIFY"]
        prev: str | None = None
        for step in work:
            if prev is not None:
                step.deps = (prev,)
            prev = step.id
        if verifies and work:
            verifies[0].deps = tuple(s.id for s in work)

    # -- fallback ------------------------------------------------------
    def _fallback_plan(self, goal: str, world: WorldState, note: str) -> PlanGraph:
        graph = self.fallback.plan(goal, world)
        dec = getattr(self.fallback, "last_decision", None)
        if dec is not None:
            self.last_decision = PlanDecision(
                goal=dec.goal, revision=dec.revision,
                selected_ops=dec.selected_ops,
                candidates_considered=dec.candidates_considered,
                candidates_feasible=dec.candidates_feasible,
                governing_constraints=dec.governing_constraints,
                rejected=dict(dec.rejected), mode=dec.mode,
                state_hash=dec.state_hash, envelope_digest=dec.envelope_digest,
                latency_ms=dec.latency_ms,
                reason=f"[{self.reasoner.backend}] fallback ({note}): {dec.reason}",
            )
        return graph

    # -- Plan contract -------------------------------------------------
    def plan(self, goal: str, world: WorldState) -> PlanGraph:
        self._turn += 1
        session = f"{self.session_prefix}-{self._turn}"
        prompt = self._render_prompt(goal, world, None)
        try:
            result = self.reasoner.reason(session, prompt,
                                          max_new_tokens=self.max_new_tokens)
        except ReasonerUnavailable as exc:
            return self._fallback_plan(goal, world, str(exc))
        return self._compile(goal, world, result, prompt)

    def replan(self, current: PlanGraph, world: WorldState, reason: str) -> PlanGraph:
        self._turn += 1
        session = f"{self.session_prefix}-{self._turn}"
        prompt = self._render_prompt(current.goal, world, reason)
        try:
            result = self.reasoner.reason(session, prompt,
                                          max_new_tokens=self.max_new_tokens)
        except ReasonerUnavailable as exc:
            graph = self._fallback_plan(current.goal, world, str(exc))
            graph.revision = current.revision + 1
            if self.last_decision is not None:
                self.last_decision.revision = graph.revision
                self.last_decision.reason = f"replan: {reason} — {self.last_decision.reason}"
            return graph
        graph = self._compile(current.goal, world, result, prompt)
        graph.revision = current.revision + 1
        if self.last_decision is not None:
            self.last_decision.revision = graph.revision
            self.last_decision.reason = f"replan: {reason} — {self.last_decision.reason}"
        return graph

    # -- compile -------------------------------------------------------
    def _compile(self, goal: str, world: WorldState, result: ReasonerResult,
                 prompt: str) -> PlanGraph:
        proposals, rationale = self._parse(result.text)
        steps, rejected = self._validate(proposals, world)
        manip = [s for s in steps if s.contract == "manipulate"]
        if not manip:
            note = (f"model proposed no valid steps ({len(rejected)} rejected)"
                    if proposals else "model proposed no steps")
            return self._fallback_plan(goal, world, note)

        graph = PlanGraph(goal=goal)
        graph.steps = steps
        self._link(graph.steps)
        gov = tuple(sorted({c.kind for c in world.constraints}))
        identity_note = ""
        if result.artifact_identity:
            identity_note = f" artifact={result.artifact_identity.get('checkpoint_sha256', '?')[:12]}"
        self.last_decision = PlanDecision(
            goal=goal,
            revision=graph.revision,
            selected_ops=tuple(s.op for s in graph.steps),
            candidates_considered=len(proposals),
            candidates_feasible=len(manip),
            governing_constraints=gov,
            rejected=rejected,
            state_hash=sha256_of(world.as_dict())[:16],
            reason=(f"[{result.backend}{identity_note}] "
                    f"{rationale or 'no rationale provided'}"),
        )
        return graph
