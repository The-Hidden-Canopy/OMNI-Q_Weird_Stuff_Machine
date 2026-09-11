"""OmniPlanner / reasoner backend acceptance tests.

Covers the Phase-0 wiring: mock backend determinism, constrained-grammar
parsing, core-side validation of model-proposed steps, truthful fallback
labeling, env-gated engine selection, and an end-to-end mock run whose
decision rationale reaches the receipt. The real IDA Omni body is exercised
only when OMNIQ_OMNI_CHECKPOINT/RECEIPT point at an artifact (skipped here).
"""

from __future__ import annotations

import os

import pytest

from omni_q import (
    OmniPlanner,
    MockReasoner,
    ReasonerUnavailable,
    build_mock_engine,
)
from omni_q.contracts import Plan
from omni_q.fakes import RulePlanner
from omni_q.omni_reasoner import ReasonerResult
from omni_q.world import MockWorld

VALID_PLAN = """\
PLAN
STEP PICK object=connector_2 arm=left
STEP MOVE object=connector_2 to=bin
STEP PICK object=plate_1
STEP MOVE object=plate_1 to=setting_1
STEP VERIFY
END
RATIONALE: connector_2 and plate_1 are misplaced; move each to its target zone.
"""

BAD_PLAN = """\
PLAN
STEP PICK object=ghost_9
STEP MOVE object=connector_2 to=void_zone
STEP PICK object=plate_1
STEP FLY object=cable_4
END
"""


class _FlakyReasoner:
    backend = "mock"

    def reason(self, session_id, prompt, *, max_new_tokens=None):
        raise ReasonerUnavailable("no weights on disk (pretrain)")


def _world():
    return MockWorld.sample().state()


def _planner(text=None, reasoner=None):
    return OmniPlanner(reasoner or MockReasoner(text))


# -- backends ------------------------------------------------------------


def test_mock_reasoner_is_deterministic_and_labels_itself():
    r = MockReasoner()
    first = r.reason("s-1", "hello world")
    second = r.reason("s-1", "hello world")
    assert first.text == second.text
    assert first.backend == "mock"
    assert r.calls == [("s-1", "hello world"), ("s-1", "hello world")]


def test_conforms_to_plan_protocol():
    assert isinstance(_planner(), Plan)
    assert isinstance(_planner().fallback, Plan)


# -- parsing + validation --------------------------------------------------


def test_parses_valid_model_plan_into_validated_graph():
    planner = _planner(VALID_PLAN)
    graph = planner.plan("tidy the workspace", _world())
    ops = [s.op for s in graph.steps]
    assert ops == ["PICK", "MOVE", "PICK", "MOVE", "VERIFY"]
    # dependency chain: each MOVE depends on its PICK; VERIFY on all work
    move = graph.by_id("omni_move_connector_2_1")
    pick = graph.by_id("omni_pick_connector_2_0")
    assert move.deps == (pick.id,)
    verify = graph.by_id("verify_final")
    assert len(verify.deps) == 4
    decision = planner.last_decision
    assert decision is not None
    assert "[mock]" in decision.reason
    assert "misplaced" in decision.reason
    assert decision.candidates_feasible == 4
    assert decision.rejected == {}


def test_invalid_proposals_are_rejected_with_reasons():
    planner = _planner(BAD_PLAN)
    graph = planner.plan("tidy the workspace", _world())
    # plate_1 is the only proposal that survives core validation
    assert [s.op for s in graph.steps] == ["PICK"]
    assert graph.steps[0].args["object"] == "plate_1"
    decision = planner.last_decision
    assert decision is not None and "fallback" not in decision.reason
    assert len(decision.rejected) == 3
    reasons = " | ".join(decision.rejected.values())
    assert "unknown object" in reasons
    assert "unknown zone" in reasons
    assert "allowlist" in reasons


def test_partial_plan_keeps_valid_steps_and_rejects_rest():
    text = ("PLAN\n"
            "STEP PICK object=connector_2\n"
            "STEP MOVE object=connector_2 to=bin\n"
            "STEP PICK object=ghost_9\n"
            "END\n"
            "RATIONALE: two good, one hallucinated.\n")
    planner = _planner(text)
    graph = planner.plan("tidy the workspace", _world())
    assert [s.op for s in graph.steps] == ["PICK", "MOVE"]
    decision = planner.last_decision
    assert decision is not None and "fallback" not in decision.reason
    assert any("ghost_9" in key for key in decision.rejected)
    assert any("unknown object" in why for why in decision.rejected.values())


def test_redundant_pick_on_an_already_held_object_is_rejected_not_reattempted():
    """A found bug, not a hypothetical: composing this planner with real
    FrameObserver+OpenVINO vision and a scripted model response measured
    a model that hadn't (yet) noticed a prior PICK already succeeded
    keep proposing it again every replan. The world layer correctly
    rejects a re-pick of something already held
    (TransitionRejected: "<oid> is already held"), but before this
    check existed, that stale proposal reached the physical layer and
    burned a whole revision on a real, futile re-attempt every single
    time -- 6 of 7 revisions in the measured case. RulePlanner already
    carries an object straight from the gripper instead of re-picking
    it (fakes.py); this mirrors that at the model-validation layer
    instead of trusting the model to always track state perfectly."""
    from dataclasses import replace

    world = replace(_world(), ownership={"connector_2": "intel.left_arm"})
    text = ("PLAN\n"
            "STEP PICK object=connector_2\n"
            "STEP MOVE object=connector_2 to=bin\n"
            "END\n")
    planner = _planner(text)

    graph = planner.plan("tidy the workspace", world)

    assert [s.op for s in graph.steps] == ["MOVE"]
    decision = planner.last_decision
    assert decision is not None
    assert any("already held" in why for why in decision.rejected.values())


def test_move_on_a_held_object_is_routed_to_the_holding_arm_not_the_models_guess():
    """A second, distinct instance of the same class of gap, found the
    same way: composing real vision + this planner, a MOVE with no
    `arm=` defaulted elsewhere while the object was actually held by the
    right arm -- rejected every time ("held by intel.right_arm, not
    intel.left_arm"), burning the whole budget on a step that could
    never succeed as proposed. Once an object is held, only the holding
    arm can act on it -- there's no ambiguity to resolve, so this is
    corrected, not merely flagged."""
    from dataclasses import replace

    world = replace(_world(), ownership={"connector_2": "intel.right_arm"})
    text = "PLAN\nSTEP MOVE object=connector_2 to=bin\nEND\n"
    planner = _planner(text)

    graph = planner.plan("tidy the workspace", world)

    assert [s.op for s in graph.steps] == ["MOVE"]
    assert graph.steps[0].arm == "right"


def test_pick_or_move_on_an_already_placed_object_is_rejected_as_redundant():
    """Third instance: RulePlanner never proposes PICK/MOVE for an object
    outside world.misplaced() in the first place, since it generates
    candidates FROM that set. A model instead proposes from its own
    (possibly stale) belief about the goal, so once an object is
    genuinely placed, a model that hasn't noticed yet may propose
    PICK/MOVE for it again -- unlike the already-held case, ownership
    has been released by then, so nothing else catches it, and it
    becomes a real, wasted (or worse, disruptive) re-attempt. Measured
    directly composing real vision + physical execution: after a real
    PICK+MOVE genuinely placed cup_1, the next replan's static proposal
    re-picked it anyway and failed for real.

    Proposes sleeve_1 (already placed) alongside connector_2 (genuinely
    misplaced) so at least one manipulate step survives and the fallback
    path doesn't swallow this planner's own rejection reasons -- a
    no-valid-steps-at-all case falls back to RulePlanner, whose *own*
    decision (with its own, unrelated rejected dict) replaces this
    planner's, which would hide the very rejection being tested."""
    world = _world()
    assert world.objects["sleeve_1"].misplaced is False  # already at target_zone
    text = ("PLAN\n"
            "STEP PICK object=sleeve_1\n"
            "STEP MOVE object=sleeve_1 to=bin\n"
            "STEP PICK object=connector_2\n"
            "END\n")
    planner = _planner(text)

    graph = planner.plan("tidy the workspace", world)

    assert [s.op for s in graph.steps] == ["PICK"]
    assert graph.steps[0].args["object"] == "connector_2"
    decision = planner.last_decision
    assert decision is not None and "fallback" not in decision.reason
    assert len(decision.rejected) == 2
    assert all("already at its target zone" in why for why in decision.rejected.values())


def test_forbidden_objects_still_blocked_from_model_steps():
    from omni_q.contracts import validate_operator_constraint

    world = MockWorld.sample()
    world.add_constraint(validate_operator_constraint(
        "forbid_object", "connector_2", source="operator",
        justification="operator holds it"))
    planner = _planner(VALID_PLAN)
    graph = planner.plan("tidy the workspace", world.state())
    assert all(s.args.get("object") != "connector_2"
               for s in graph.steps if s.op != "VERIFY")


# -- fallback truthfulness ---------------------------------------------------


def test_reasoner_unavailability_falls_back_with_label():
    planner = OmniPlanner(_FlakyReasoner())
    world = _world()
    fallback_graph = RulePlanner().plan("tidy the workspace", world)
    graph = planner.plan("tidy the workspace", world)
    assert [s.op for s in graph.steps] == [s.op for s in fallback_graph.steps]
    decision = planner.last_decision
    assert decision is not None
    assert "fallback" in decision.reason
    assert "no weights on disk" in decision.reason


def test_replan_advances_revision_and_annotates_reason():
    planner = _planner(VALID_PLAN)
    world = _world()
    graph = planner.plan("tidy the workspace", world)
    replanned = planner.replan(graph, world, "plate_1 moved mid-task")
    assert replanned.revision == graph.revision + 1
    assert planner.last_decision is not None
    assert planner.last_decision.reason.startswith("replan: plate_1 moved mid-task")
    assert planner.last_decision.revision == replanned.revision


# -- engine integration ------------------------------------------------------


def test_env_selects_omni_planner(monkeypatch):
    monkeypatch.setenv("OMNIQ_OMNI_REASONER", "mock")
    engine = build_mock_engine()
    assert isinstance(engine.planner, OmniPlanner)


def test_env_default_is_rule_planner(monkeypatch):
    monkeypatch.delenv("OMNIQ_OMNI_REASONER", raising=False)
    engine = build_mock_engine()
    assert isinstance(engine.planner, RulePlanner)


def test_env_omni_mode_requires_artifact_paths(monkeypatch):
    monkeypatch.setenv("OMNIQ_OMNI_REASONER", "omni")
    monkeypatch.delenv("OMNIQ_OMNI_CHECKPOINT", raising=False)
    monkeypatch.delenv("OMNIQ_OMNI_RECEIPT", raising=False)
    with pytest.raises(ValueError, match="OMNIQ_OMNI_CHECKPOINT"):
        build_mock_engine()


def test_end_to_end_mock_reasoner_run_records_rationale():
    engine = build_mock_engine(planner=_planner(VALID_PLAN))
    receipt = engine.run("tidy the workspace")
    assert receipt.decisions, "model decision captured"
    assert any("[mock]" in d["reason"] for d in receipt.decisions)
    assert any("misplaced" in d["reason"] for d in receipt.decisions)


def test_end_to_end_fallback_run_is_truthfully_labeled():
    engine = build_mock_engine(planner=OmniPlanner(_FlakyReasoner()))
    receipt = engine.run("tidy the workspace")
    assert any("fallback" in d["reason"] for d in receipt.decisions)
