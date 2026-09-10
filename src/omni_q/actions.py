"""OQ-HAND-001..005 — the manipulation action vocabulary.

Splits "get the limb there" from "do something once you're in contact", and
attaches the metadata the scheduler and planner need to reason about each op:

    category · requires_contact · requires_grasp · supports_bimanual
    precision · style_action · blocking · preconditions · effects · args

Predicate strings in ``preconditions`` / ``effects`` are deliberately simple so
a planner can chain them later (OQ-HAND-006):

    reachable(object)  clear(object)  contact(object)  grasped(object)
    held_by(object, arm)  at(object, zone)  aligned(object)  centered(object)

A leading ``-`` in an effect deletes the predicate.

Composite *routines* (``composes``) expand to a primitive ``(op, arm)`` sequence
via :func:`expand` — ``"self"`` means the arm the routine was assigned to,
``"other"`` its partner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionCategory(str, Enum):
    ARM_MOTION = "arm_motion"
    GRIP_STATE = "grip_state"
    CONTACT_MANIP = "contact_manipulation"
    ORIENTATION = "orientation"
    PLACEMENT = "placement"
    BIMANUAL = "bimanual"
    PERCEPTION = "perception"
    ROUTINE = "routine"
    IDLE_FLOURISH = "idle_flourish"   # dance-while-working: fills scheduler slack


class Precision(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ActionSpec:
    name: str
    category: ActionCategory
    requires_contact: bool = False
    requires_grasp: bool = False
    supports_bimanual: bool = False
    precision: Precision = Precision.MEDIUM
    style_action: bool = False
    blocking: bool = True
    preconditions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    args: tuple[str, ...] = ("object",)
    composes: tuple[tuple[str, str], ...] = ()   # (op, "self"|"other") — () = primitive

    @property
    def is_bimanual(self) -> bool:
        return self.supports_bimanual or self.category is ActionCategory.BIMANUAL

    @property
    def is_routine(self) -> bool:
        return bool(self.composes)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "action": self.name,
            "category": self.category.value,
            "requires_contact": self.requires_contact,
            "requires_grasp": self.requires_grasp,
            "supports_bimanual": self.supports_bimanual,
            "precision_level": self.precision.value,
            "style_action": self.style_action,
            "blocking": self.blocking,
            "preconditions": list(self.preconditions),
            "effects": list(self.effects),
            "args": list(self.args),
        }
        if self.composes:
            d["composes"] = [list(c) for c in self.composes]
        return d


def _a(name, category, **kw) -> ActionSpec:
    return ActionSpec(name=name, category=category, **kw)


C = ActionCategory
P = Precision
_G = ("grasped(object)",)          # common precondition: something is in hand
_C = ("contact(object)",)

_SPECS: tuple[ActionSpec, ...] = (
    # -- arm motion -------------------------------------------------
    _a("HOVER", C.ARM_MOTION, blocking=False, precision=P.LOW, effects=("near(object)",)),
    _a("APPROACH", C.ARM_MOTION, preconditions=("reachable(object)",), effects=("near(object)",)),
    _a("MOVE", C.ARM_MOTION, requires_grasp=True, preconditions=_G,
       effects=("at(object, to)",), args=("object", "to")),
    _a("RETRACT", C.ARM_MOTION, blocking=False, precision=P.LOW),

    # -- grip state ------------------------------------------------
    _a("PRE_SHAPE", C.GRIP_STATE, blocking=False, precision=P.HIGH, args=("shape",)),
    _a("PINCH_GRIP", C.GRIP_STATE, requires_contact=True, precision=P.HIGH,
       preconditions=("near(object)",), effects=("grasped(object)",)),
    _a("WIDE_GRIP", C.GRIP_STATE, requires_contact=True,
       preconditions=("near(object)",), effects=("grasped(object)",)),
    _a("EDGE_GRIP", C.GRIP_STATE, requires_contact=True, precision=P.HIGH,
       preconditions=("near(object)",), effects=("grasped(object)",)),
    _a("SOFT_HOLD", C.GRIP_STATE, requires_grasp=True, precision=P.HIGH,
       preconditions=_G, effects=("grasped(object)", "gentle(object)")),
    _a("FIRM_HOLD", C.GRIP_STATE, requires_grasp=True, preconditions=_G),
    _a("RELEASE", C.GRIP_STATE, requires_grasp=True,
       preconditions=_G, effects=("-grasped(object)",)),
    _a("MICRO_RELEASE", C.GRIP_STATE, requires_grasp=True, precision=P.HIGH,
       blocking=False, preconditions=_G),
    _a("REGRIP", C.GRIP_STATE, requires_grasp=True, precision=P.HIGH,
       preconditions=_G, effects=("grasped(object)",)),
    _a("SHIFT_GRIP", C.GRIP_STATE, requires_grasp=True, precision=P.HIGH,
       blocking=False, preconditions=_G, effects=("grasped(object)",),
       args=("object", "direction")),

    # -- contact manipulation (no full grasp needed) ---------------
    _a("NUDGE", C.CONTACT_MANIP, requires_contact=True, precision=P.LOW, blocking=False,
       preconditions=("reachable(object)",), effects=("at(object, near_target)",),
       args=("object", "direction", "amount")),
    _a("PUSH", C.CONTACT_MANIP, requires_contact=True, precision=P.LOW,
       args=("object", "direction", "amount")),
    _a("PULL", C.CONTACT_MANIP, requires_contact=True, precision=P.LOW,
       args=("object", "direction", "amount")),
    _a("SLIDE", C.CONTACT_MANIP, requires_contact=True, precision=P.LOW,
       preconditions=("reachable(object)", "clear(object)"),
       effects=("at(object, to)",), args=("object", "to")),
    _a("DRAG", C.CONTACT_MANIP, requires_contact=True, requires_grasp=True,
       preconditions=_G, effects=("at(object, to)",), args=("object", "to")),
    _a("SWEEP", C.CONTACT_MANIP, requires_contact=True, precision=P.LOW,
       args=("region", "direction")),
    _a("PRESS", C.CONTACT_MANIP, requires_contact=True, precision=P.HIGH),
    _a("PIN", C.CONTACT_MANIP, requires_contact=True),
    _a("BRACE", C.CONTACT_MANIP, requires_contact=True, blocking=False,
       effects=("stable(object)",)),
    _a("STABILIZE", C.CONTACT_MANIP, requires_contact=True, blocking=False,
       effects=("stable(object)",)),
    _a("TAP", C.CONTACT_MANIP, requires_contact=True, precision=P.LOW, blocking=False),
    _a("BUMP_ALIGN", C.CONTACT_MANIP, requires_contact=True, precision=P.LOW, blocking=False,
       effects=("aligned(object)",)),

    # -- orientation / flourish (object is in hand) ----------------
    _a("ROTATE", C.ORIENTATION, requires_grasp=True, preconditions=_G, args=("object", "degrees")),
    _a("ROTATE_IN_HAND", C.ORIENTATION, requires_grasp=True, precision=P.HIGH, blocking=False,
       preconditions=_G, effects=("reoriented(object)",), args=("object", "degrees")),
    _a("SPIN", C.ORIENTATION, requires_grasp=True, style_action=True, supports_bimanual=True,
       preconditions=_G, effects=("reoriented(object)",), args=("object", "degrees")),
    _a("TWIRL", C.ORIENTATION, requires_grasp=True, style_action=True, precision=P.HIGH,
       preconditions=_G, args=("object", "degrees")),
    _a("ROLL_OBJECT", C.ORIENTATION, requires_grasp=True, style_action=True, preconditions=_G),
    _a("FLIP", C.ORIENTATION, requires_grasp=True, precision=P.HIGH, preconditions=_G,
       effects=("face_up(object)",)),
    _a("TURN_HANDLE_TO", C.ORIENTATION, requires_grasp=True, precision=P.HIGH,
       preconditions=_G, effects=("handle_at(object, angle)",), args=("object", "angle")),
    _a("PRESENT", C.ORIENTATION, style_action=True, blocking=False, precision=P.LOW,
       effects=("presented(object)",)),
    _a("SHOWCASE", C.ORIENTATION, style_action=True, blocking=False, precision=P.LOW),
    _a("ORIENT_FACE_UP", C.ORIENTATION, requires_grasp=True, preconditions=_G,
       effects=("face_up(object)",)),
    _a("ALIGN_EDGE", C.ORIENTATION, requires_contact=True, precision=P.HIGH,
       effects=("aligned(object)",)),
    _a("NAPKIN_FLICK", C.ORIENTATION, requires_grasp=True, style_action=True, preconditions=_G,
       effects=("spread(object)",)),

    # -- placement ------------------------------------------------
    _a("PICK", C.PLACEMENT, requires_contact=True,
       preconditions=("reachable(object)", "clear(object)"),
       effects=("grasped(object)",)),
    _a("PLACE", C.PLACEMENT, requires_grasp=True,
       preconditions=_G, effects=("-grasped(object)", "at(object, to)"), args=("object", "to")),
    _a("CENTER_ON_MARK", C.PLACEMENT, requires_grasp=True, precision=P.HIGH,
       preconditions=_G, effects=("-grasped(object)", "centered(object)"), args=("object", "mark")),
    _a("PLACE_LEFT_OF", C.PLACEMENT, requires_grasp=True, preconditions=_G,
       effects=("-grasped(object)", "left_of(object, ref)"), args=("object", "ref")),
    _a("PLACE_RIGHT_OF", C.PLACEMENT, requires_grasp=True, preconditions=_G,
       effects=("-grasped(object)", "right_of(object, ref)"), args=("object", "ref")),
    _a("PLACE_ABOVE_RIGHT_OF", C.PLACEMENT, requires_grasp=True, precision=P.HIGH,
       preconditions=_G, effects=("-grasped(object)",), args=("object", "ref")),
    _a("ALIGN_PARALLEL", C.PLACEMENT, requires_contact=True, precision=P.HIGH,
       effects=("aligned(object)",), args=("object", "ref")),
    _a("ALIGN_PERPENDICULAR", C.PLACEMENT, requires_contact=True, precision=P.HIGH,
       effects=("aligned(object)",), args=("object", "ref")),
    _a("SET_DISTANCE", C.PLACEMENT, requires_contact=True, args=("object", "ref", "distance")),
    _a("SQUARE_TO_TABLE", C.PLACEMENT, requires_contact=True, effects=("aligned(object)",)),
    _a("STRAIGHTEN", C.PLACEMENT, requires_contact=True, precision=P.LOW, blocking=False,
       effects=("aligned(object)",)),
    _a("NEST", C.PLACEMENT, requires_grasp=True, preconditions=_G, args=("object", "into")),
    _a("STACK", C.PLACEMENT, requires_grasp=True, preconditions=_G, args=("object", "onto")),
    _a("SPREAD", C.PLACEMENT, requires_grasp=True, preconditions=_G, effects=("spread(object)",)),
    _a("FAN", C.PLACEMENT, requires_contact=True, precision=P.LOW, args=("objects",)),

    # -- bimanual -----------------------------------------------
    _a("HANDOFF", C.BIMANUAL, requires_grasp=True, supports_bimanual=True,
       preconditions=_G, effects=("held_by(object, other)", "-held_by(object, self)"),
       args=("object", "to_actor")),
    _a("RECEIVE", C.BIMANUAL, supports_bimanual=True, effects=("grasped(object)",)),
    _a("PASS_THROUGH", C.BIMANUAL, requires_grasp=True, supports_bimanual=True, preconditions=_G),
    _a("CO_HOLD", C.BIMANUAL, requires_contact=True, supports_bimanual=True, blocking=False,
       effects=("stable(object)",)),
    _a("CO_ROTATE", C.BIMANUAL, requires_grasp=True, supports_bimanual=True, precision=P.HIGH,
       style_action=True, preconditions=_G, effects=("reoriented(object)",),
       args=("object", "degrees")),
    _a("CO_ALIGN", C.BIMANUAL, requires_contact=True, supports_bimanual=True,
       effects=("aligned(object)",)),
    _a("CO_STABILIZE", C.BIMANUAL, requires_contact=True, supports_bimanual=True, blocking=False,
       effects=("stable(object)",)),
    _a("OPEN_SPACE_FOR", C.BIMANUAL, supports_bimanual=True, blocking=False, args=("region",)),
    _a("ASSIST_GRASP", C.BIMANUAL, requires_contact=True, supports_bimanual=True,
       effects=("grasped(object)",)),
    _a("TRANSFER_LOAD", C.BIMANUAL, requires_grasp=True, supports_bimanual=True, preconditions=_G,
       effects=("held_by(object, other)",)),
    _a("REGRASP_WITH_PARTNER", C.BIMANUAL, requires_grasp=True, supports_bimanual=True,
       precision=P.HIGH, preconditions=_G, effects=("grasped(object)",)),
    _a("HOLD_WHILE_OTHER_ACTS", C.BIMANUAL, requires_contact=True, supports_bimanual=True,
       blocking=False, effects=("stable(object)",)),

    # -- idle flourish (no object, fills scheduler slack) ----------
    _a("SWAY", C.IDLE_FLOURISH, style_action=True, blocking=False, precision=P.LOW, args=()),
    _a("BOUNCE", C.IDLE_FLOURISH, style_action=True, blocking=False, precision=P.LOW, args=()),
    _a("WAVE", C.IDLE_FLOURISH, style_action=True, blocking=False, precision=P.LOW, args=()),
    _a("SPIN_WRISTS", C.IDLE_FLOURISH, style_action=True, blocking=False, precision=P.LOW, args=()),
    _a("CROSS", C.IDLE_FLOURISH, style_action=True, blocking=False, precision=P.LOW,
       supports_bimanual=True, args=()),
    _a("MIRROR", C.IDLE_FLOURISH, style_action=True, blocking=False, precision=P.LOW,
       supports_bimanual=True, args=()),
    _a("BOW", C.IDLE_FLOURISH, style_action=True, blocking=False, precision=P.LOW, args=()),
    _a("HIGH_FIVE", C.IDLE_FLOURISH, style_action=True, blocking=False, precision=P.LOW,
       supports_bimanual=True, args=()),
    _a("CALL_AND_RESPONSE", C.IDLE_FLOURISH, style_action=True, blocking=False,
       supports_bimanual=True, args=()),
    _a("FREEZE", C.IDLE_FLOURISH, blocking=True, precision=P.LOW, args=()),

    # -- perception ---------------------------------------------
    _a("OBSERVE", C.PERCEPTION, blocking=True, args=()),
    _a("VERIFY", C.PERCEPTION, blocking=True, args=()),
    _a("LOCATE", C.PERCEPTION, blocking=False, args=("target",)),

    # -- routines (expand to primitives) -----------------------
    _a("SPIN_PLATE", C.ROUTINE, style_action=True, args=("object", "degrees"),
       composes=(("PICK", "self"), ("SPIN", "self"), ("HANDOFF", "self"),
                 ("SPIN", "other"), ("CENTER_ON_MARK", "other"))),
    _a("TWIRL_UTENSIL", C.ROUTINE, style_action=True, args=("object",),
       composes=(("EDGE_GRIP", "self"), ("TWIRL", "self"), ("ALIGN_PARALLEL", "self"),
                 ("PLACE", "self"))),
    _a("ROTATE_CUP_HANDLE", C.ROUTINE, args=("object", "angle"),
       composes=(("PICK", "self"), ("TURN_HANDLE_TO", "self"),
                 ("PLACE_ABOVE_RIGHT_OF", "self"))),
    _a("NAPKIN_ROUTINE", C.ROUTINE, style_action=True, args=("object",),
       composes=(("PINCH_GRIP", "self"), ("SPREAD", "self"), ("PRESENT", "self"),
                 ("PLACE", "self"))),
    _a("PRESENT_AND_PLACE", C.ROUTINE, style_action=True, args=("object", "to"),
       composes=(("PRESENT", "self"), ("PLACE", "self"))),
    _a("SPIN_AND_PLACE", C.ROUTINE, style_action=True, args=("object", "to"),
       composes=(("SPIN", "self"), ("PLACE", "self"))),
)

ACTIONS: dict[str, ActionSpec] = {s.name: s for s in _SPECS}

# the six the review said give the most immediate payoff
PRIORITY_SIX: tuple[str, ...] = (
    "SHIFT_GRIP", "SLIDE", "NUDGE", "ROTATE_IN_HAND", "HANDOFF", "CO_ROTATE",
)


# ---------------------------------------------------------------------------
# style / choreography modes  (last STYLE constraint on the world wins)
# ---------------------------------------------------------------------------

# canonical mode -> its aliases as they arrive from nlu / speech
STYLE_MODES: dict[str, tuple[str, ...]] = {
    "minimum_time": ("minimum_time", "none", "efficient", "back_to_work", "stop"),
    "show_off": ("show_off", "fancy", "excited"),
    "dance": ("dance", "do_a_wave"),
    "synchronized": ("synchronized", "together"),
    "mirrored": ("mirrored", "opposite", "mirror_me"),
    "take_turns": ("take_turns",),
    "freeze": ("freeze",),
    "fast": ("fast", "faster"),
    "slow": ("slow", "slow_down"),
}
_ALIAS_TO_MODE = {a: m for m, aliases in STYLE_MODES.items() for a in aliases}

# modes that make the scheduler fill idle-arm slack with IDLE_FLOURISH steps
_DANCE_MODES = {"show_off", "dance", "synchronized", "mirrored", "take_turns"}


def style_mode(value: Any) -> str:
    """Normalise a STYLE constraint value to a canonical mode."""
    key = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return _ALIAS_TO_MODE.get(key, key)


def wants_idle_flourish(mode: str) -> bool:
    return mode in _DANCE_MODES


IDLE_FLOURISH_OPS: tuple[str, ...] = tuple(
    n for n, s in ACTIONS.items() if s.category is ActionCategory.IDLE_FLOURISH
    and n != "FREEZE")


# ---------------------------------------------------------------------------
# predicates the scheduler / planner call
# ---------------------------------------------------------------------------


def spec(op: str) -> ActionSpec | None:
    return ACTIONS.get(op)


def known(op: str) -> bool:
    return op in ACTIONS


def category_of(op: str) -> ActionCategory | None:
    s = ACTIONS.get(op)
    return s.category if s else None


def is_style(op: str) -> bool:
    s = ACTIONS.get(op)
    return bool(s and s.style_action)


def is_bimanual(op: str) -> bool:
    s = ACTIONS.get(op)
    return bool(s and s.is_bimanual)


def is_blocking(op: str) -> bool:
    s = ACTIONS.get(op)
    return s.blocking if s else True


def needs_grasp(op: str) -> bool:
    s = ACTIONS.get(op)
    return bool(s and s.requires_grasp)


def needs_contact(op: str) -> bool:
    s = ACTIONS.get(op)
    return bool(s and s.requires_contact)


def is_routine(op: str) -> bool:
    s = ACTIONS.get(op)
    return bool(s and s.composes)


def expand(op: str, arm: str, other: str, args: dict[str, Any] | None = None,
           prefix: str | None = None) -> list[dict[str, Any]]:
    """Expand a routine into ordered primitive step descriptors.

    Returns ``[{"id","op","arm","args","deps"}]`` — a linear chain (each step
    depends on the one before). ``arm``/``other`` fill the ``"self"``/``"other"``
    slots in :attr:`ActionSpec.composes`.
    """
    s = ACTIONS.get(op)
    if not s or not s.composes:
        raise ValueError(f"{op} is not a routine")
    args = args or {}
    base = prefix or f"{op.lower()}_{args.get('object', 'obj')}"
    out: list[dict[str, Any]] = []
    prev: str | None = None
    for i, (prim, who) in enumerate(s.composes):
        sid = f"{base}_{i}_{prim.lower()}"
        step_arm = arm if who == "self" else other
        step_args = dict(args)
        if prim == "HANDOFF":
            step_args["to_actor"] = other
        out.append({
            "id": sid, "op": prim, "arm": step_arm,
            "args": step_args, "deps": (prev,) if prev else (),
        })
        prev = sid
    return out
