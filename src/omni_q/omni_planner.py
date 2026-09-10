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
            arm = prop.get("arm")
            if arm is not None and arm not in {"left", "right"}:
                rejected[label] = f"invalid arm {arm!r}"
                continue
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
