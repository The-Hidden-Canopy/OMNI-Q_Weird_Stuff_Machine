# Portions derived from *The Hidden Canopy LLC* — [`open_world_model_harness`](https://github.com/The-Hidden-Canopy/open_world_model_harness). Used with permission.
"""Evaluator-only secrets, deterministic seeding, and unreliable cues.

Three patterns for keeping evaluation honest:

* ``stable_unit_float`` derives a hidden, reproducible per-entity scalar from
  a domain string and entity keys — the same inputs always yield the same
  value, with no shared global RNG state to leak or drift.
* :class:`SealedSnapshot` persists evaluator-only ground truth to its own
  file (the ``final_evaluator_snapshot.json`` role) and hands out only a
  caller-declared public projection. The sealed payload and the public view
  are *different dataclasses*, so serializing the secret through the public
  path is structurally impossible — not flag-gated.
* :class:`SensoryCue` marks cues from hallucination-prone channels with
  ``reliable=False`` so consumers can weight them accordingly.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .evidence_bundle import _write_json


def stable_unit_float(domain: str, *parts: str) -> float:
    """Deterministic hidden scalar in ``[0.65, 1.35)`` for one entity.

    SHA-256 over ``"domain:part1:part2:..."``; the first digest byte maps
    linearly onto ``0.65 + (byte / 255) * 0.7`` (rounded to 3 decimals) —
    the aptitude-topology seeding scheme of the source engine, generalized
    from ``session_id:player_id:skill:node`` to any domain/entity keys.
    """
    material = ":".join((domain, *parts))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return round(0.65 + (digest[0] / 255.0) * 0.7, 3)


@dataclass(frozen=True)
class PublicSnapshot:
    """The evaluator-visible projection of a sealed snapshot.

    Holds *only* the caller-declared non-secret fields. There is no reference
    to the sealed payload anywhere in this type, so it cannot be serialized
    out by accident.
    """

    fields: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.fields)


@dataclass(frozen=True)
class SealedSnapshot:
    """Evaluator-only ground truth, sealed to its own file at construction.

    ``payload`` is the secret (hidden state, answer keys, latent values);
    ``public_fields`` names the payload keys that are safe to expose. The
    payload is written to ``destination`` immediately and is never part of
    any observation or public projection.
    """

    payload: Mapping[str, Any]
    destination: Path
    public_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "destination", Path(self.destination))
        self.write()

    @classmethod
    def load(
        cls,
        source: str | Path,
        public_fields: tuple[str, ...] = (),
    ) -> "SealedSnapshot":
        """Re-read a sealed snapshot without rewriting (and thus repairing)
        its file — tamper evidence must survive a read."""
        source = Path(source)
        payload = json.loads(source.read_text(encoding="utf-8"))
        self = object.__new__(cls)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "destination", source)
        object.__setattr__(self, "public_fields", tuple(public_fields))
        return self

    def write(self) -> Path:
        _write_json(self.destination, self.payload)
        return self.destination

    def public_view(self) -> PublicSnapshot:
        """Detach the caller-declared non-secret fields.

        Values are deep-copied, so mutating the returned view can neither
        change the sealed payload nor leak into later views.
        """
        fields = {
            key: deepcopy(self.payload[key])
            for key in self.public_fields
            if key in self.payload
        }
        return PublicSnapshot(fields=fields)


@dataclass(frozen=True)
class SensoryCue:
    """An observation channel cue.

    ``reliable=False`` means this channel is hallucination-prone; consumers
    must weight the cue accordingly instead of treating it as ground truth.
    """

    cue_id: str
    text: str
    reliable: bool = True
    source: str = "game_sensory"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue_id": self.cue_id,
            "text": self.text,
            "reliable": self.reliable,
            "source": self.source,
        }
