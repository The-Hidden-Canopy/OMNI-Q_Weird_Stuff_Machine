# Manipulation action vocabulary (`omni_q.actions`)

> OQ-HAND-001..005. Splits "get the limb there" from "do something once you're
> in contact", and attaches the metadata the scheduler and planner reason over.

`ACTIONS: dict[str, ActionSpec]` — 84 ops across:

| category | examples |
|----------|----------|
| `ARM_MOTION` | HOVER, APPROACH, MOVE, RETRACT |
| `GRIP_STATE` | PRE_SHAPE, PINCH/WIDE/EDGE_GRIP, SOFT/FIRM_HOLD, RELEASE, MICRO_RELEASE, REGRIP, SHIFT_GRIP |
| `CONTACT_MANIP` | NUDGE, PUSH, PULL, SLIDE, DRAG, SWEEP, PRESS, PIN, BRACE, STABILIZE, TAP, BUMP_ALIGN |
| `ORIENTATION` | ROTATE_IN_HAND, TWIRL, SPIN, ROLL_OBJECT, FLIP, TURN_HANDLE_TO, PRESENT, ORIENT_FACE_UP, ALIGN_EDGE, NAPKIN_FLICK |
| `PLACEMENT` | PICK, PLACE, CENTER_ON_MARK, PLACE_LEFT/RIGHT/ABOVE_RIGHT_OF, ALIGN_PARALLEL/PERPENDICULAR, SET_DISTANCE, SQUARE_TO_TABLE, STRAIGHTEN, NEST, STACK, SPREAD, FAN |
| `BIMANUAL` | HANDOFF, RECEIVE, PASS_THROUGH, CO_HOLD, CO_ROTATE, CO_ALIGN, CO_STABILIZE, OPEN_SPACE_FOR, ASSIST_GRASP, TRANSFER_LOAD, REGRASP_WITH_PARTNER, HOLD_WHILE_OTHER_ACTS |
| `PERCEPTION` | OBSERVE, VERIFY, LOCATE |
| `IDLE_FLOURISH` | SWAY, BOUNCE, WAVE, SPIN_WRISTS, CROSS, MIRROR, BOW, HIGH_FIVE, CALL_AND_RESPONSE, FREEZE |
| `ROUTINE` | SPIN_PLATE, TWIRL_UTENSIL, ROTATE_CUP_HANDLE, NAPKIN_ROUTINE, PRESENT_AND_PLACE, SPIN_AND_PLACE |

## `ActionSpec`

```python
name · category · requires_contact · requires_grasp · supports_bimanual
precision (LOW/MEDIUM/HIGH) · style_action · blocking
preconditions · effects · args · composes
```

Predicate strings are simple so a planner can chain them (OQ-HAND-006):
`reachable(object)`, `clear(object)`, `contact(object)`, `grasped(object)`,
`held_by(object, arm)`, `at(object, zone)`, `aligned(object)`, `centered(object)`;
a leading `-` in an effect deletes the predicate.

### Predicates the scheduler calls

`spec` · `known` · `category_of` · `is_style` · `is_bimanual` · `is_blocking`
· `needs_grasp` · `needs_contact` · `is_routine`

The scheduler (`omni_q.scheduler`) now reads these instead of hard-coded op
sets: `is_style` → flourish handling (OQ-015), `category_of == BIMANUAL` → the
step takes both arms in its wave, `needs_grasp` → an arm can't manipulate an
object another arm is holding.

## Routines

```python
from omni_q.actions import expand
expand("SPIN_PLATE", "left", "right", {"object": "plate_1", "degrees": 180})
# PICK(left) -> SPIN(left) -> HANDOFF(left→right) -> SPIN(right) -> CENTER_ON_MARK(right)
```

`"self"` / `"other"` in `ActionSpec.composes` fill from the two arm args; the
result is a linear dependency chain of primitive step descriptors.

## Style / choreography modes (OQ-HAND-011)

`STYLE_MODES` maps spoken aliases to a canonical mode; `style_mode(value)`
normalises (spaces/hyphens too). The **last** `STYLE` constraint on the world
wins ("make it fancy" … "back to work").

| mode | aliases | scheduler effect |
|------|---------|------------------|
| `minimum_time` | none, efficient, back to work, stop screwing around | strips every flourish |
| `show_off` | fancy, excited, with a flourish | keeps `PRESENT`/flourish steps; fills idle-arm slack |
| `dance` | do a wave, boogie | fills idle-arm slack, rotating through IDLE_FLOURISH ops |
| `synchronized` | together, in sync | idle arms get the **same** primitive |
| `mirrored` | opposite, mirror me | idle arms get a mirror pair |
| `take_turns` | one at a time | fills idle-arm slack |
| `freeze` | hold still, stop moving | no motion; strips flourishes |
| `fast` / `slow` | faster / slow down | tempo hint (no scheduling change yet) |

**Invariant:** dance fills already-idle arms in already-existing non-final
waves — it never adds a wave, reorders work, or becomes a dependency of `verify`.
`Schedule.metrics["idle_flourishes"]` counts what was injected;
`Schedule.style_mode` reports the active mode.

```
PYTHONPATH=src python -m omni_q.scheduler
PYTHONPATH=src python -m pytest -q tests/test_actions.py tests/test_scheduler.py
```
