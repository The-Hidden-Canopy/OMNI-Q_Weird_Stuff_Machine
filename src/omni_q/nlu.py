"""OQ-023 — natural-language goal parser.

Turns a free-text instruction into two explicit things Omni Q already
understands:

- a **canonical goal** string the planner recognises
  (``RulePlanner._ACTIONABLE``), and
- a list of **constraints** as ``(kind, value)`` pairs
  (``forbid_object`` / ``keep_local`` / ``style`` / ``prefer_arm``).

Deliberately standalone: it imports nothing from the engine/contracts core, so
it stays decoupled from in-flight changes there. The only coupling point is
``ParsedInstruction.apply_to(engine)``, a one-liner over ``engine.add_constraint``.

    >>> p = parse("put the objects away, show off, but don't touch the red connector")
    >>> p.goal
    'inspect and correct the workspace'
    >>> p.constraints
    [('style', 'show_off'), ('forbid_object', 'connector_2')]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["ParsedInstruction", "parse", "resolve"]


# ---------------------------------------------------------------------------
# object-name vocabulary (mirrors MockWorld.sample(); override per real world)
# ---------------------------------------------------------------------------

_DEFAULT_VOCAB: dict[str, str] = {
    "connector": "connector_2",
    "red connector": "connector_2",
    "sleeve": "sleeve_1",
    "black sleeve": "sleeve_1",
    "cable": "cable_4",
    "loose cable": "cable_4",
    "plate": "plate_1",
    "dish": "plate_1",
}

_ARTICLES = ("the ", "that ", "this ", "my ", "a ", "an ", "some ")
_ARM_WORDS = {"arm", "hand", "left arm", "right arm", "left hand", "right hand"}


def resolve(phrase: str, vocab: dict[str, str] | None = None) -> str:
    """Best-effort map a noun phrase to a world object id.

    Falls back to the last word, then to the cleaned phrase itself. ``None`` is
    never returned — an unresolved phrase is simply inert against a world that
    keys ``forbid_object`` on exact ids (a world-aware resolver is OQ-025).
    """
    table = {**_DEFAULT_VOCAB, **(vocab or {})}
    p = phrase.strip().lower().strip(".,;:!?")
    for art in _ARTICLES:
        if p.startswith(art):
            p = p[len(art):]
    p = re.sub(r"\s+", " ", p).strip()
    if p in table:
        return table[p]
    words = p.split()
    if len(words) > 1 and words[-1] in table:
        return table[words[-1]]
    return p


# ---------------------------------------------------------------------------
# goal intents  (first match wins; result must contain a RulePlanner keyword)
# ---------------------------------------------------------------------------

_GOAL_INTENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(set|lay|arrange)\s+(up\s+)?(the\s+)?table\b|\bplace settings?\b"
                r"|\btable setting\b"), "set the table"),
    (re.compile(r"\b(tidy|clean[ -]?up|cleanup|clear|declutter|straighten up"
                r"|put\s+[\w ]+?\s+away|put away|correct|fix|reset|sort out"
                r"|organi[sz]e)\b"), "inspect and correct the workspace"),
    (re.compile(r"\b(inspect|check|examine|look at|review|assess)\b"),
     "inspect the workspace"),
]


def _classify_goal(text: str) -> tuple[str, list[str]]:
    for pattern, canonical in _GOAL_INTENTS:
        if pattern.search(text):
            return canonical, []
    return text.strip(), ["goal not recognised; planner will observe only"]


# ---------------------------------------------------------------------------
# constraint rules  (kind, value) — order defines output order
# ---------------------------------------------------------------------------

_CLAUSE_END = r"(?=,|\.|;|$|\s+and\b|\s+but\b|\s+then\b|\s+while\b|\s+alone\b)"

Rule = Callable[[str, "dict[str, str] | None"], "list[tuple[str, Any]]"]


def _rule_stop_using_arm(text: str, vocab: dict[str, str] | None) -> list[tuple[str, Any]]:
    # "don't use the left arm" / "stop using the right hand" -> prefer the other
    out: list[tuple[str, Any]] = []
    pat = re.compile(
        r"\b(do ?n['’]?t|do not|stop|never|quit)\s+(use|using)\s+"
        r"(the\s+)?(?P<side>left|right)\s*(arm|hand)\b"
    )
    for m in pat.finditer(text):
        other = "right" if m.group("side") == "left" else "left"
        out.append(("prefer_arm", other))
    return out


def _rule_style(text: str, vocab: dict[str, str] | None) -> list[tuple[str, Any]]:
    if re.search(r"\b(show(ing)? off|showboat|flourish|be fancy|with flair|fancy"
                 r"|razzle|make it look good)\b", text):
        return [("style", "show_off")]
    return []


def _rule_forbid(text: str, vocab: dict[str, str] | None) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    verbs = r"(?:do ?n['’]?t|do not|never|no)\s+(?:touch|move|disturb|bump|take|grab)"
    leave = r"(?:leave|avoid|keep away from|stay away from|don['’]?t go near)"
    for pat in (
        re.compile(rf"\b{verbs}\s+(the\s+|that\s+|my\s+)?(?P<obj>[a-z][a-z ]*?)\s*{_CLAUSE_END}"),
        re.compile(rf"\b{leave}\s+(the\s+)?(?P<obj>[a-z][a-z ]*?)\s*{_CLAUSE_END}"),
    ):
        for m in pat.finditer(text):
            obj = m.group("obj").strip()
            if not obj or obj in _ARM_WORDS:
                continue
            out.append(("forbid_object", resolve(obj, vocab)))
    return out


def _rule_keep_local(text: str, vocab: dict[str, str] | None) -> list[tuple[str, Any]]:
    if re.search(r"\b(keep (everything|it|inference|things|the model)?\s*"
                 r"(local|on[- ]device)|on[- ]device|no cloud|stay local"
                 r"|stay offline|fully local|all local|locally only"
                 r"|keep everything on the device)\b", text):
        return [("keep_local", None)]
    return []


def _rule_prefer_arm(text: str, vocab: dict[str, str] | None) -> list[tuple[str, Any]]:
    m = re.search(r"\b(use|using|prefer|preferring|with|favou?r(?:ing)?|just)\s+"
                  r"(the\s+)?(?P<side>left|right)(\s+(arm|hand))?\b", text)
    if not m:
        m = re.search(r"\b(?P<side>left|right)[- ](arm|hand)\s+only\b", text)
    return [("prefer_arm", m.group("side"))] if m else []


_CONSTRAINT_RULES: list[Rule] = [
    _rule_stop_using_arm,
    _rule_style,
    _rule_forbid,
    _rule_keep_local,
    _rule_prefer_arm,
]

# understood, but no consumer yet -> notes only
_NOTE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bkeep [\w ]+? clear\b"),
    re.compile(r"\b(forks?|knives?|spoons?|cups?|plates?)\s+(on the\s+)?(left|right|centre|center|middle|upper|lower)\b"),
    re.compile(r"\b[\w ]+? cent(er|re)ed\b"),
]

_SINGLETON_KINDS = {"keep_local", "style", "prefer_arm"}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def _rule_spin(text: str, vocab: dict[str, str] | None) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    pat = re.compile(r"\b(spin|rotate|turn|twirl|flip)\s+(the\s+|that\s+|this\s+)?"
                     r"(?P<obj>[a-z][a-z ]*?)\s*(?P<deg>\d{2,3})?\s*(deg|degrees|°)?"
                     rf"\s*{_CLAUSE_END}")
    for m in pat.finditer(text):
        obj = m.group("obj").strip()
        if not obj or obj in _ARM_WORDS:
            continue
        d: dict[str, Any] = {"object": resolve(obj, vocab)}
        if m.group("deg"):
            d["degrees"] = int(m.group("deg"))
        out.append(("spin", d))
    return out


def _rule_nudge(text: str, vocab: dict[str, str] | None) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    pat = re.compile(
        r"\b(move|nudge|shift|scoot|slide)\s+(the\s+|that\s+)?(?P<obj>[a-z][a-z ]*?)\s+"
        r"(?P<amount>a bit|a little|slightly|farther|further|way|much)?\s*"
        r"(to the\s+|towards the\s+|toward\s+)?(?P<dir>left|right|back|forward|up|down|centre|center)"
        rf"\s*{_CLAUSE_END}")
    for m in pat.finditer(text):
        obj = m.group("obj").strip()
        if not obj or obj in _ARM_WORDS:
            continue
        out.append(("nudge", {
            "object": resolve(obj, vocab),
            "direction": m.group("dir"),
            "amount": (m.group("amount") or "a bit").replace("further", "farther"),
        }))
    return out


_MUTATION_RULES = [_rule_spin, _rule_nudge]


@dataclass
class ParsedInstruction:
    raw: str
    goal: str
    constraints: list[tuple[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    mutations: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def apply_to(self, engine: Any) -> str:
        """Push the constraints onto a live engine; return the goal to run.

        Passes a ``justification`` (the raw instruction) when the engine's
        ``add_constraint`` accepts one — the operator-provenance path — and
        degrades to the positional call otherwise.
        """
        why = f"parsed from operator instruction: {self.raw!r}"
        for kind, value in self.constraints:
            try:
                engine.add_constraint(kind, value, justification=why)
            except TypeError:
                engine.add_constraint(kind, value)
        return self.goal

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "goal": self.goal,
            "constraints": [list(c) for c in self.constraints],
            "notes": list(self.notes),
            "mutations": [[k, v] for k, v in self.mutations],
        }


def parse(text: str, *, vocab: dict[str, str] | None = None) -> ParsedInstruction:
    low = " " + re.sub(r"\s+", " ", text.strip().lower()) + " "
    goal, notes = _classify_goal(low)

    constraints: list[tuple[str, Any]] = []
    seen: set[tuple[str, Any]] = set()
    for rule in _CONSTRAINT_RULES:
        for kind, value in rule(low, vocab):
            if kind in _SINGLETON_KINDS and any(k == kind for k, _ in constraints):
                continue
            if (kind, value) in seen:
                continue
            seen.add((kind, value))
            constraints.append((kind, value))

    mutations: list[tuple[str, dict[str, Any]]] = []
    for rule in _MUTATION_RULES:
        for kind, payload in rule(low, vocab):
            if (kind, tuple(sorted(payload.items()))) not in {
                (k, tuple(sorted(p.items()))) for k, p in mutations
            }:
                mutations.append((kind, payload))

    for pat in _NOTE_PATTERNS:
        for m in pat.finditer(low):
            notes.append(f"unhandled layout hint: {m.group(0).strip()}")

    return ParsedInstruction(raw=text, goal=goal, constraints=constraints,
                             notes=notes, mutations=mutations)


def _main() -> None:  # pragma: no cover - manual smoke
    from omni_q import build_mock_engine

    sentence = ("Put the objects away and show off, but don't touch the red "
                "connector and keep everything local.")
    p = parse(sentence)
    print("instruction :", sentence)
    print("goal        :", p.goal)
    print("constraints :", p.constraints)
    print("notes       :", p.notes)

    engine = build_mock_engine()
    goal = p.apply_to(engine)
    receipt = engine.run(goal)
    print("resolved    :", receipt.metrics.get("resolved"))
    print("rejected    :", list(receipt.rejected))


if __name__ == "__main__":
    _main()
