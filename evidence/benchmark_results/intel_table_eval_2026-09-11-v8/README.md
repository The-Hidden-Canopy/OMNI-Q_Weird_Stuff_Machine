# Legacy Intel table-setting randomized report (v8)

Same protocol as v2/v3/v5/v6/v7: `run_intel_table_evaluation_report(...,
trials=10, seed=701)`, `simulation-scripted-manipulation` route, one
canonical hash-checked receipt per trial. First report generated with the
new `schema_version: 2` per-object tally (`per_object_summary` +
per-receipt `per_object` field) -- see `_per_object_pick_place_outcomes`
in `src/omni_q/intel_sim.py`.

| Outcome | Count |
| --- | ---: |
| Success | 0 |
| Grasp failure | 10 |
| Placement failure | 0 |
| Timeout | 0 |
| Collision | 0 |
| Transition failure | 0 |
| Unresolved | 0 |

**The real, previously-invisible signal, now surfaced directly in the
report rather than requiring someone to read raw receipts by hand:**

| Object | Held (of 10) | Placed (of 10) |
| --- | ---: | ---: |
| `cup_1` | **10/10** | 9/10 |
| `plate_1` | 1/10 | 0/10 |
| `napkin_1` / `fork_1` / `spoon_1` | 0/10 | 0/10 |

`cup_1`'s grasp is genuinely robust -- held successfully in **every
single trial** across ±3mm/±0.08rad randomized scene jitter, not a
one-off. Placement landed in 9/10 (one trial's carry-and-release didn't
settle within the placement tolerance -- a real, if minor, remaining gap
worth a look, not investigated further this pass). `plate_1` held once
out of ten, consistent with the earlier finding that its grasp has a
narrow, fragile margin (see `intel_sim.py`'s module docstring, "Fifth
update") -- not reliable, but not zero either, which the coarse
`grasp_failure` label alone could never show.

The coarse `outcomes` table is unchanged from v6/v7 and was never
expected to move from this specific change -- `_classify_intel_table_
receipt` scans the whole receipt for the first failure found anywhere in
it, so a trial where `cup_1` completes a full real pick-and-place still
classifies as `grasp_failure` the moment any other object fails later in
the same run. That was always a real limitation of the aggregate label,
not something this schema change was meant to fix directly; what it does
fix is making the real per-object signal underneath it visible without
reading 10 raw receipts by hand.

Kept alongside v2/v3/v5/v6/v7, not replacing them, so this progression
stays auditable.
