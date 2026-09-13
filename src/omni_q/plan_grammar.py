"""Grammar-constrained decoding for the OmniPlanner plan language.

``OmniPlanner._validate`` already knows which object ids and zones are legal in
the current world -- it checks them *after* generation and rejects whatever
does not match. This module applies the same knowledge *during* generation, so
the illegal step is never produced in the first place.

The motivating measurement (2026-09-13, training run r1). The model learned the
grammar but not which identifier belongs in which slot: it emitted
``STEP PICK object=setting_1`` where ``setting_1`` is a *zone* name that appears
as an ``object=`` value **zero times in 14,583 training object slots**. Every
step was consequently rejected and acceptance sat at 0.00 while loss fell from
12.55 to 2.43. Fed to the real planner:

===========================================  ========  ========
text                                          proposed  accepted
===========================================  ========  ========
raw model output                                     2         0
same text, repetition perfectly removed              3         0
identical shape with a *real* object id              3         2
===========================================  ========  ========

Fixing the decoder's repetition changes nothing; fixing the slot vocabulary is
the whole difference. Constraining is therefore not a cosmetic improvement, it
is the difference between a plan and a rejection.

This constrains **identifiers only** -- what fills an ``object=`` or ``to=``
slot. It does not force the model to emit steps, choose actions, or decide how
many steps a plan has; a model that wants to stop, or to propose a PICK where a
MOVE was wanted, still does so. The grammar is a fence, not a script, and the
planner's own validation still runs afterwards: a constrained identifier can
still be the *wrong* legal object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = ["SlotVocabulary", "PlanGrammarConstraint", "vocabulary_from_world"]

#: Text that, once emitted, means the next tokens fill that slot.
OBJECT_CUE = "object="
ZONE_CUE = "to="


@dataclass(frozen=True)
class SlotVocabulary:
    """Legal identifiers for each slot of the plan language."""

    objects: tuple[str, ...]
    zones: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {"objects": list(self.objects), "zones": list(self.zones)}


def vocabulary_from_world(world: Any) -> SlotVocabulary:
    """Legal object ids and zones for ``world``.

    Zones are taken from what objects actually occupy or target, rather than a
    fixed list, so a scene with new zones needs no change here.
    """
    objects = tuple(sorted(getattr(world, "objects", {}) or {}))
    zones: set[str] = set()
    for detection in (getattr(world, "objects", {}) or {}).values():
        for attr in ("zone", "target_zone"):
            value = getattr(detection, attr, None)
            if isinstance(value, str) and value:
                zones.add(value)
    return SlotVocabulary(objects=objects, zones=tuple(sorted(zones)))


@dataclass
class PlanGrammarConstraint:
    """Token-level fence over identifier slots.

    Drive it one generated token at a time: ``allowed_tokens()`` returns the
    token ids permitted next (or ``None`` when unconstrained), and ``accept()``
    records what was actually emitted.

    Implemented as a trie over the *token sequences* of the legal identifiers,
    so a multi-token id like ``cup_1`` -> ``[' cup', '_', '1']`` is constrained
    at every one of its tokens, not just the first. Constraining only the first
    token would still allow ``cup_9`` in a world that has no ``cup_9``.
    """

    vocabulary: SlotVocabulary
    encode: Callable[[str], Sequence[int]]
    #: Tokens after which a slot opens, e.g. the ``=`` of ``object=``.
    _object_seqs: tuple[tuple[int, ...], ...] = field(default_factory=tuple, init=False)
    _zone_seqs: tuple[tuple[int, ...], ...] = field(default_factory=tuple, init=False)
    _prefix: tuple[int, ...] = field(default_factory=tuple, init=False)
    _slot: str | None = field(default=None, init=False)
    text: str = field(default="", init=False)
    forced: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._object_seqs = self._encode_all(self.vocabulary.objects)
        self._zone_seqs = self._encode_all(self.vocabulary.zones)

    def _encode_all(self, names: Iterable[str]) -> tuple[tuple[int, ...], ...]:
        # Identifiers follow a space in the emitted text ("object= cup_1" is not
        # how it reads -- the model writes "object=cup_1"), so encode bare.
        return tuple(tuple(int(t) for t in self.encode(name)) for name in names)

    # -- driving ---------------------------------------------------------

    def _active_sequences(self) -> tuple[tuple[int, ...], ...]:
        if self._slot == "object":
            return self._object_seqs
        if self._slot == "zone":
            return self._zone_seqs
        return ()

    def allowed_tokens(self) -> set[int] | None:
        """Token ids permitted next, or ``None`` when not inside a slot."""
        if self._slot is None:
            return None
        candidates = [s for s in self._active_sequences()
                      if len(s) > len(self._prefix)
                      and s[:len(self._prefix)] == self._prefix]
        if not candidates:
            return None
        return {s[len(self._prefix)] for s in candidates}

    def accept(self, token: int, piece: str) -> None:
        """Record an emitted token and its decoded text."""
        self.text += piece
        if self._slot is not None:
            self._prefix = self._prefix + (token,)
            self.forced += 1
            complete = any(s == self._prefix for s in self._active_sequences())
            extendable = any(len(s) > len(self._prefix)
                             and s[:len(self._prefix)] == self._prefix
                             for s in self._active_sequences())
            if complete and not extendable:
                self._close_slot()
            elif not extendable and not complete:
                # Model escaped the fence (only possible when a slot had no
                # candidates); stop pretending to constrain it.
                self._close_slot()
            return
        self._maybe_open_slot()

    def _close_slot(self) -> None:
        self._slot = None
        self._prefix = ()

    def _maybe_open_slot(self) -> None:
        tail = self.text
        if tail.endswith(OBJECT_CUE) and self._object_seqs:
            self._slot = "object"
        elif tail.endswith(ZONE_CUE) and self._zone_seqs:
            self._slot = "zone"

    @property
    def in_slot(self) -> str | None:
        return self._slot
