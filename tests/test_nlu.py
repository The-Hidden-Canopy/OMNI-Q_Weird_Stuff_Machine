"""OQ-023 — natural-language goal parser."""

from __future__ import annotations

import pytest

from omni_q import build_mock_engine
from omni_q.nlu import ParsedInstruction, parse, resolve


# -- goal classification ---------------------------------------------------

@pytest.mark.parametrize("text, goal", [
    ("set the table", "set the table"),
    ("please lay the table for two", "set the table"),
    ("tidy up the workspace", "inspect and correct the workspace"),
    ("put the objects away", "inspect and correct the workspace"),
    ("clean up and reset the bench", "inspect and correct the workspace"),
    ("inspect the workspace", "inspect the workspace"),
    ("check everything looks right", "inspect the workspace"),
])
def test_goal_is_mapped_to_a_planner_keyword(text, goal):
    assert parse(text).goal == goal


def test_unrecognised_goal_passes_through_with_a_note():
    p = parse("make me a coffee")
    assert p.goal == "make me a coffee"
    assert p.constraints == []
    assert any("not recognised" in n for n in p.notes)


# -- constraint extraction ----------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("tidy up but don't touch the red connector", ("forbid_object", "connector_2")),
    ("clear the bench and leave the cable alone", ("forbid_object", "cable_4")),
    ("set the table, keep everything local", ("keep_local", None)),
    ("tidy up and keep inference on-device", ("keep_local", None)),
    ("set the table and show off", ("style", "show_off")),
    ("put things away with a flourish", ("style", "show_off")),
    ("tidy up using the left arm", ("prefer_arm", "left")),
    ("set the table, right-arm only", ("prefer_arm", "right")),
])
def test_single_constraint_is_extracted(text, expected):
    assert expected in parse(text).constraints


def test_dont_use_the_left_arm_prefers_the_other_arm():
    # the DEMO.md constraint-change moment
    assert ("prefer_arm", "right") in parse("Omni, don't use the left arm anymore").constraints


def test_compound_instruction_yields_ordered_constraints():
    p = parse("tidy the workspace, show off, but don't touch the red connector "
              "and keep it local")
    assert p.goal == "inspect and correct the workspace"
    assert p.constraints == [
        ("style", "show_off"),
        ("forbid_object", "connector_2"),
        ("keep_local", None),
    ]


def test_singleton_kinds_are_deduplicated():
    p = parse("keep it local and keep everything on the device, stay local")
    assert p.constraints.count(("keep_local", None)) == 1


def test_layout_hints_go_to_notes_not_constraints():
    p = parse("set the table, forks on the left, plates centered, keep the middle clear")
    assert p.constraints == []
    assert len(p.notes) >= 2
    assert all(k != "forbid_object" for k, _ in p.constraints)


# -- object resolution -------------------------------------------------

def test_resolve_maps_phrases_and_falls_back_to_last_word():
    assert resolve("the red connector") == "connector_2"
    assert resolve("green connector") == "connector_2"        # unknown adj -> last word
    assert resolve("gremlin") == "gremlin"                    # unknown -> cleaned phrase


def test_resolve_accepts_a_custom_vocab():
    vocab = {"widget": "widget_9"}
    assert resolve("the widget", vocab) == "widget_9"


# -- integration with the mock engine --------------------------------

def test_apply_to_engine_runs_and_honours_a_forbidden_object():
    p = parse("put the objects away and don't touch the red connector")
    assert isinstance(p, ParsedInstruction)
    assert ("forbid_object", "connector_2") in p.constraints

    engine = build_mock_engine()
    goal = p.apply_to(engine)
    receipt = engine.run(goal)

    moved = [a["result"].get("moved") for a in engine._actions if a["op"] == "MOVE"]
    assert "connector_2" not in moved            # the forbidden object was left alone
    assert "plate_1" in moved                    # the other misplaced object was handled
    still_misplaced = [d.object_id for d in engine.world.state().misplaced()]
    assert still_misplaced == ["connector_2"]
    assert any("connector_2" in str(r) for r in receipt.rejected)


def test_apply_to_plain_instruction_resolves_the_workspace():
    p = parse("tidy up the workspace")
    engine = build_mock_engine()
    receipt = engine.run(p.apply_to(engine))
    assert receipt.metrics["resolved"] is True
