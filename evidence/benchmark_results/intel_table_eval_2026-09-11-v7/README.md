# Legacy Intel table-setting randomized report (v7)

Same protocol as v2/v3/v5/v6: `run_intel_table_evaluation_report(...,
trials=10, seed=701)`, `simulation-scripted-manipulation` route, one
canonical hash-checked receipt per trial. Generated after a scheduler fix
(`IntelTablePlanner`, `src/omni_q/intel_sim.py`): a persistently-failing
object no longer consumes the entire run's revision budget by itself.

| Outcome | Count |
| --- | ---: |
| Success | 0 |
| Grasp failure | 10 |
| Placement failure | 0 |
| Timeout | 0 |
| Collision | 0 |
| Transition failure | 0 |
| Unresolved | 0 |

**Aggregate unchanged from v6, exactly as expected and checked, not
assumed** -- `_classify_intel_table_receipt` scans the whole receipt for
any failed grasp anywhere in it, so this label was never going to move
just from trying more objects; every object still has to actually
succeed for `resolved: True`. What this fix actually changes is visible
in the raw receipts: before it (v6), a trial's `pick_napkin_1` (the first
object after `cup_1` in priority order) was measured failing 6 times in a
row, consuming the whole revision budget alone -- `plate_1`/`fork_1`/
`spoon_1` never got a single real attempt. After it (this directory),
`trial-00-seed-701.json` shows the same budget spent cycling through
`napkin_1`, `plate_1`, `fork_1`, `spoon_1` in turn instead of hammering
one. Real value: richer, more representative evidence per trial, and a
live demo that visibly tries different objects instead of appearing
stuck repeating the same failed motion -- not a change to the coarse
success/failure label, which was never the point of this specific fix.

Root cause (see `IntelTablePlanner.replan`'s docstring in
`intel_sim.py`): the engine replans on any failure and always regenerates
every misplaced object into the graph in the same priority order; since
only the first ready step ever executes before the next replan, the first
still-failing object in that order blocked everything behind it from ever
being tried, regardless of how many revisions remained.

Kept alongside v2/v3/v5/v6, not replacing them, so this progression stays
auditable.
