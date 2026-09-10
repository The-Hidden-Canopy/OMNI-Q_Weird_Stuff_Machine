"""OQ-025 — runtime constraint mutation.

Turn a spoken instruction into a **safe** change to a running (or about-to-run)
:class:`~omni_q.engine.OmniQ`.

- Constraint-shaped mutations (``forbid_object`` / ``keep_local`` /
  ``prefer_arm`` / ``style``) are queued through ``engine.add_constraint``,
  which validates them; the engine then recompiles atomically on its next loop
  iteration with the fixed ``MissionEnvelope`` re-applied on top — a live change
  can only *narrow* authority, never widen it.
- Structural mutations (``spin`` / ``nudge``) are parsed and reported but
  **deferred** until the planner grows the matching primitives
  (OQ-011 / OQ-014) and real poses (OQ-009 remainder).

Standalone: depends on :mod:`omni_q.nlu` and a duck-typed engine only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import nlu

_DEFER_REASON: dict[str, str] = {
    "spin": ("needs the rotate -> handoff -> regrasp primitive (OQ-011 / OQ-014); "
             "recorded as a planner hint until that lands"),
    "nudge": ("needs a within-zone pose adjustment; the planner works in discrete "
              "zones until real poses arrive (OQ-009 remainder)"),
    "spin_on_place": ("standing rule: the planner should swap PLACE -> SPIN_AND_PLACE "
                      "for this object class on the remaining ops (OQ-HAND-007)"),
}


@dataclass
class MutationResult:
    text: str
    applied: list[tuple[str, Any]] = field(default_factory=list)
    rejected: list[tuple[str, Any, str]] = field(default_factory=list)
    deferred: list[tuple[str, dict[str, Any], str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.applied) and not self.rejected

    def summary(self) -> str:
        bits: list[str] = []
        if self.applied:
            bits.append("applied " + ", ".join(
                f"{k}={v}" if v is not None else k for k, v in self.applied))
        if self.rejected:
            bits.append("rejected " + ", ".join(f"{k} ({r})" for k, _v, r in self.rejected))
        if self.deferred:
            bits.append("deferred " + ", ".join(
                f"{k}({p.get('object', '?')})" for k, p, _r in self.deferred))
        return "; ".join(bits) or "no change"

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "applied": [list(a) for a in self.applied],
            "rejected": [list(r) for r in self.rejected],
            "deferred": [[k, p, r] for k, p, r in self.deferred],
            "ok": self.ok,
        }


class RuntimeMutator:
    """Applies spoken changes to a live ``OmniQ`` engine, safely."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.history: list[MutationResult] = []

    def apply(self, text: str, *, vocab: dict[str, str] | None = None) -> MutationResult:
        parsed = nlu.parse(text, vocab=vocab)
        res = MutationResult(text=text)
        why = f"live operator instruction: {text!r}"

        for kind, value in parsed.constraints:
            try:
                self.engine.add_constraint(kind, value, justification=why)
            except TypeError:
                try:
                    self.engine.add_constraint(kind, value)
                except Exception as exc:  # noqa: BLE001 - report, don't crash the run
                    res.rejected.append((kind, value, f"{type(exc).__name__}: {exc}"))
                    continue
            except Exception as exc:  # noqa: BLE001
                res.rejected.append((kind, value, f"{type(exc).__name__}: {exc}"))
                continue
            res.applied.append((kind, value))

        for kind, payload in parsed.mutations:
            res.deferred.append((kind, payload, _DEFER_REASON.get(kind, "unsupported mutation")))

        self.history.append(res)
        return res

    def apply_all(self, texts: list[str]) -> list[MutationResult]:
        return [self.apply(t) for t in texts]


def _main() -> None:  # pragma: no cover - manual smoke
    from . import build_mock_engine

    engine = build_mock_engine()
    m = RuntimeMutator(engine)
    for line in ["don't touch the red connector",
                 "keep everything local",
                 "spin that plate",
                 "move the fork farther left"]:
        print(f"{line!r:45} -> {m.apply(line).summary()}")
    receipt = engine.run("inspect and correct the workspace")
    print("resolved:", receipt.metrics["resolved"], "mode:", receipt.metrics["mode"])
    print("rejected:", list(receipt.rejected))


if __name__ == "__main__":
    _main()
