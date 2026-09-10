"""OQ-HAND-001..005 — the manipulation action vocabulary."""

from __future__ import annotations

import pytest

from omni_q import actions as A
from omni_q.actions import ActionCategory as C


def test_registry_covers_the_structural_ops_the_code_emits():
    for op in ("PICK", "PLACE", "MOVE", "HANDOFF", "PRESENT", "VERIFY", "OBSERVE", "LOCATE"):
        assert A.known(op)


def test_priority_six_are_all_defined():
    assert all(A.known(op) for op in A.PRIORITY_SIX)
    assert set(A.PRIORITY_SIX) == {
        "SHIFT_GRIP", "SLIDE", "NUDGE", "ROTATE_IN_HAND", "HANDOFF", "CO_ROTATE"}


def test_slide_is_a_no_grasp_alternative_to_pick_place():
    slide = A.spec("SLIDE")
    assert slide.requires_contact and not slide.requires_grasp
    assert "at(object, to)" in slide.effects
    # the enabler for OQ-HAND-006: the scheduler can pick SLIDE over PICK+PLACE
    assert not slide.style_action


def test_grasp_effects_are_consistent():
    assert "grasped(object)" in A.spec("PICK").effects
    assert "-grasped(object)" in A.spec("PLACE").effects
    assert "grasped(object)" not in A.spec("PICK").preconditions
    assert A.spec("SPIN").requires_grasp
    assert "grasped(object)" in A.spec("SPIN").preconditions


def test_style_and_bimanual_predicates():
    assert A.is_style("SPIN") and A.is_style("PRESENT") and A.is_style("CO_ROTATE")
    assert not A.is_style("PICK")
    assert A.category_of("HANDOFF") is C.BIMANUAL
    assert A.is_bimanual("HANDOFF") and A.is_bimanual("CO_ROTATE")
    assert not A.is_bimanual("PICK")


def test_idle_flourish_ops_exclude_freeze():
    assert A.IDLE_FLOURISH_OPS
    assert "FREEZE" not in A.IDLE_FLOURISH_OPS
    assert all(A.category_of(op) is C.IDLE_FLOURISH for op in A.IDLE_FLOURISH_OPS)


@pytest.mark.parametrize("value, mode", [
    ("fancy", "show_off"),
    ("show_off", "show_off"),
    ("opposite", "mirrored"),
    ("together", "synchronized"),
    ("back to work", "minimum_time"),
    ("do a wave", "dance"),
    ("weird custom", "weird_custom"),
])
def test_style_mode_normalisation(value, mode):
    assert A.style_mode(value) == mode


def test_wants_idle_flourish():
    assert A.wants_idle_flourish("dance") and A.wants_idle_flourish("show_off")
    assert not A.wants_idle_flourish("minimum_time")
    assert not A.wants_idle_flourish("unset")


def test_routine_expansion_of_spin_plate():
    steps = A.expand("SPIN_PLATE", "left", "right", {"object": "plate_1", "degrees": 180})
    assert [s["op"] for s in steps] == ["PICK", "SPIN", "HANDOFF", "SPIN", "CENTER_ON_MARK"]
    assert [s["arm"] for s in steps] == ["left", "left", "left", "right", "right"]
    # linear chain
    assert steps[0]["deps"] == ()
    assert steps[1]["deps"] == (steps[0]["id"],)
    # HANDOFF names the partner
    assert steps[2]["args"]["to_actor"] == "right"


def test_unknown_op():
    assert A.spec("NOT_AN_OP") is None
    assert not A.known("NOT_AN_OP")
    assert A.category_of("NOT_AN_OP") is None
    assert A.is_blocking("NOT_AN_OP") is True   # unknown ops are conservatively blocking


def test_spec_as_dict_has_the_review_schema():
    d = A.spec("SPIN").as_dict()
    for key in ("action", "category", "requires_contact", "requires_grasp",
                "supports_bimanual", "precision_level", "style_action", "blocking"):
        assert key in d
