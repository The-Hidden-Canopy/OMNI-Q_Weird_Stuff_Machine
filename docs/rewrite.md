# Graph rewrites (`omni_q.rewrite`)

> OQ-HAND-006 (planner chooses SLIDE/NUDGE vs PICK/PLACE) and the executable
> half of OQ-025 (spoken `spin` / `nudge` become graph edits, not deferrals).
> Pure transforms on `PlanGraph` + `WorldState`; no engine / contracts edits.

## `optimize(graph, world, *, can_run=None)`

Collapses a `PICK + MOVE (+ PRESENT)` chain to a cheaper contact op when the
object barely needs to move:

| condition | rewrite | why |
|-----------|---------|-----|
| move target == current zone | `NUDGE` | it's already there — a small adjust, not a lift |
| source & target on the **same table surface** (one arm reaches both, no handoff) | `SLIDE` | slide it, no grasp/lift cycle |
| cross-table move | keep `PICK` / `PLACE` | needs a real lift (or a handoff) |

- "same surface" = `|x_center(a) − x_center(b)| ≤ 0.35` in the scheduler's
  region model, and not opposite far sides.
- Held objects (`world.ownership[oid]` set) are left alone.
- `can_run(op)` gates every rewrite: it only produces an op the manipulator can
  actually execute (wire it to `engine.manipulator.supports`); unsupported →
  the chain is left as PICK/MOVE.
- Returns `(PlanGraph, list[Rewrite])`; each `Rewrite` records `kind`,
  `object`, the step ids `replaced`, the id `added`, and a reason — kept in the
  receipt so the trace shows *why* SLIDE beat PICK+PLACE.

## `apply_mutation(graph, world, kind, payload, *, can_run=None)`

Turns a parsed `nlu` structural mutation into a graph edit:

| kind | effect |
|------|--------|
| `spin` | insert a `SPIN` step before the object's terminal move; the move now depends on it |
| `spin_on_place` | `SPIN` **every** chain whose object class matches (`{"object_class": "plate"}`) — the standing "spin the plates when you put them down" rule |
| `nudge` | collapse the object's chain to a single `NUDGE`, or (no chain) add a standalone `NUDGE` before `verify_final` |

## `RewritingPlanner(inner, *, optimize_chains=True, can_run=None)`

A `Plan` decorator. Every `plan` / `replan`: run `optimize`, then apply each
mutation registered via `register_mutation(kind, payload)`. Keeps
`last_rewrites` (for the receipt / UI) and `last_error`; on any exception it
degrades to the inner graph unchanged — a rewrite bug can't break a run.
Forwards `last_decision`. Compose it under `ScheduledPlanner`.

`RuntimeMutator(engine, planner=…)` routes `spin` / `nudge` / `spin_on_place`
to `planner.register_mutation(...)` when the manipulator can run the resulting
op; otherwise it defers with a clear reason
(`"manipulator can't run SPIN yet (OQ-HAND-007+)"`).

## End to end

```
"spin the plates when you put them down" + "make it fancy"
  → final plan: SLIDE  SPIN  SLIDE  VERIFY        rewrites: [slide, slide, spin]
```

```
PYTHONPATH=src python -m pytest -q tests/test_rewrite.py
```
