# Legacy Intel table-setting randomized report (v6)

Same protocol as v2/v3/v5: `run_intel_table_evaluation_report(..., trials=10,
seed=701)`, `simulation-scripted-manipulation` route, one canonical
hash-checked receipt per trial. Generated after two real, independent
fixes landed the same day: a teammate's orientation-aware IK
(`_grasp_frame`/`_ik_reach_pad_pose`) plus a `cup_1` resized to fit the
gripper's real measured envelope, and a no-simulation-cheating fix
(reverted a `contype`/`conaffinity` no-clip exemption the same merge had
introduced on the arm mesh — see `src/omni_q/intel_sim.py`'s module
docstring for the full derivation of both).

| Outcome | Count |
| --- | ---: |
| Success | 0 |
| Grasp failure | 10 |
| Placement failure | 0 |
| Timeout | 0 |
| Collision | 0 |
| Transition failure | 0 |
| Unresolved | 0 |

**Aggregate unchanged from v2/v3/v5, but the per-trial receipts tell a
real, different, better story this time.** Inspecting `trial-00-seed-701
.json` directly: `pick_cup_1` succeeds (`held: true`, real lift), and
`move_cup_1` succeeds too (`placed: true`) — a full real pick-and-place,
genuinely completed, not a near-miss. The trial only ends in
`grasp_failure` because the plan then reaches `fork_1`, which still
doesn't converge (position error ~0.06m, orientation not satisfied) and
exhausts its bounded retries before the plan can move on to `napkin_1`/
`plate_1`/`spoon_1`.

**This is a real limitation of the outcome classification, not a
fabricated result**: `run_intel_table_evaluation_report`'s per-trial
`outcome` field is binary (first blocking failure wins), so a trial that
genuinely completes one full object's pick-and-place and then gets stuck
on the next reports identically to a trial that never succeeds at
anything. The receipts underneath (`trial-*.json`) are the actual,
honest, unabridged record and should be read directly rather than trusting
the summary table alone when auditing partial progress — which is exactly
what surfaced this. Worth a follow-up: either a finer-grained per-object
outcome tally in the report generator, or a scheduler change so one
persistently-failing object doesn't block attempts on the rest of the
plan (both already flagged in `agents/damion-claude/TASK.md`).

Kept alongside v2/v3/v5, not replacing them, so this progression stays
auditable.
