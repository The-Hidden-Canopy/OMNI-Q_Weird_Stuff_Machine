"""Grammar-constrained decoding over the plan language.

No model and no GPU: the constraint is driven by hand with a real tokenizer
substitute, so the fence itself is testable independently of whatever is
generating.
"""

from __future__ import annotations

import pytest

from omni_q.contracts import Detection, WorldState
from omni_q.plan_grammar import (
    PlanGrammarConstraint,
    SlotVocabulary,
    vocabulary_from_world,
)

# A toy tokenizer with the same shape as the real one: identifiers split into
# [word, "_", digit], which is what the 256k BPE actually does.
_VOCAB = {"object=": [1, 2], "to=": [3, 2], "cup": [10], "plate": [11],
          "setting": [12], "table": [13], "_": [20], "1": [21], "4": [24],
          "STEP PICK ": [30], "\nSTEP MOVE ": [31], " ": [32]}
_PIECES = {i: p for p, ids in _VOCAB.items() for i in ids if len(ids) == 1}
_PIECES.update({1: "object", 2: "=", 3: "to", 20: "_", 21: "1", 24: "4"})


def _encode(text: str) -> list[int]:
    out: list[int] = []
    for part in (text.replace("_", " _ ").split(" ")):
        if not part:
            continue
        if part in _VOCAB:
            out.extend(_VOCAB[part])
        else:
            out.extend(_VOCAB[part[:-1]] + _VOCAB[part[-1]]
                       if part[:-1] in _VOCAB else [99])
    return out


def _world(objects: dict[str, tuple[str, str]]) -> WorldState:
    return WorldState(frame=0, objects={
        oid: Detection(oid, oid.split("_")[0], zone, target)
        for oid, (zone, target) in objects.items()})


def test_vocabulary_comes_from_the_world_not_a_fixed_list():
    world = _world({"cup_1": ("cart", "setting_4"), "plate_1": ("tray", "setting_1")})
    vocab = vocabulary_from_world(world)
    assert vocab.objects == ("cup_1", "plate_1")
    assert vocab.zones == ("cart", "setting_1", "setting_4", "tray")


def test_the_slot_opens_only_after_its_cue():
    vocab = SlotVocabulary(objects=("cup_1",), zones=("setting_4",))
    c = PlanGrammarConstraint(vocab, _encode)
    assert c.allowed_tokens() is None
    c.accept(30, "STEP PICK ")
    assert c.allowed_tokens() is None, "no cue yet, nothing is constrained"
    c.accept(1, "object")
    assert c.allowed_tokens() is None
    c.accept(2, "=")
    assert c.in_slot == "object"


def test_every_token_of_a_multi_token_identifier_is_fenced():
    """Constraining only the first token would still permit cup_9."""
    vocab = SlotVocabulary(objects=("cup_1",), zones=("setting_4",))
    c = PlanGrammarConstraint(vocab, _encode)
    for token, piece in ((1, "object"), (2, "=")):
        c.accept(token, piece)
    assert c.allowed_tokens() == {10}          # ' cup' only
    c.accept(10, "cup")
    assert c.allowed_tokens() == {20}          # '_'
    c.accept(20, "_")
    assert c.allowed_tokens() == {21}          # '1', not '4'
    c.accept(21, "1")
    assert c.in_slot is None, "slot closes when the identifier is complete"
    assert c.allowed_tokens() is None


def test_the_zone_slot_uses_zones_not_objects():
    """The r1 failure in miniature: a zone where an object belongs."""
    vocab = SlotVocabulary(objects=("cup_1",), zones=("setting_4",))
    c = PlanGrammarConstraint(vocab, _encode)
    for token, piece in ((3, "to"), (2, "=")):
        c.accept(token, piece)
    assert c.in_slot == "zone"
    assert c.allowed_tokens() == {12}          # ' setting', never ' cup'


def test_an_object_slot_can_never_emit_a_zone_name():
    vocab = SlotVocabulary(objects=("cup_1", "plate_1"), zones=("setting_1",))
    c = PlanGrammarConstraint(vocab, _encode)
    for token, piece in ((1, "object"), (2, "=")):
        c.accept(token, piece)
    allowed = c.allowed_tokens()
    assert 12 not in allowed, "'setting' must be unreachable in an object slot"
    assert allowed == {10, 11}


def test_ambiguous_prefixes_stay_open_until_they_disambiguate():
    vocab = SlotVocabulary(objects=("cup_1", "cup_4"), zones=())
    c = PlanGrammarConstraint(vocab, _encode)
    for token, piece in ((1, "object"), (2, "=")):
        c.accept(token, piece)
    c.accept(10, "cup")
    c.accept(20, "_")
    assert c.allowed_tokens() == {21, 24}, "both ids still reachable"
    c.accept(24, "4")
    assert c.in_slot is None


def test_an_empty_vocabulary_never_opens_a_slot():
    """A world with no objects must not deadlock generation."""
    c = PlanGrammarConstraint(SlotVocabulary(objects=(), zones=()), _encode)
    c.accept(1, "object")
    c.accept(2, "=")
    assert c.in_slot is None and c.allowed_tokens() is None


def test_nothing_outside_identifier_slots_is_constrained():
    """The fence covers identifiers only -- not actions, not plan length."""
    vocab = SlotVocabulary(objects=("cup_1",), zones=("setting_4",))
    c = PlanGrammarConstraint(vocab, _encode)
    c.accept(30, "STEP PICK ")
    assert c.allowed_tokens() is None
    for token, piece in ((1, "object"), (2, "="), (10, "cup"), (20, "_"), (21, "1")):
        c.accept(token, piece)
    c.accept(31, "\nSTEP MOVE ")
    assert c.allowed_tokens() is None, "free again between slots"
