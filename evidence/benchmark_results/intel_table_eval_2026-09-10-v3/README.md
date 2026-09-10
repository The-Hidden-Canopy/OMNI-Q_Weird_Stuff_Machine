# Legacy Intel table-setting randomized report (v3)

Same protocol as `intel_table_eval_2026-09-10-v2`:
`run_intel_table_evaluation_report(..., trials=10, seed=701)` for the
`simulation-scripted-manipulation` route, bounded build-time tableware
position/yaw perturbation (±3mm / ±0.08rad) for seeds 701-710, one
canonical hash-checked Omni Q receipt per trial.

**Re-run, not a re-tune, after fixing a real scene bug:** `fork_1`/
`spoon_1` previously sat ~0.64m from each arm's base -- ~65% past the
arm's own independently measured max reach (~0.386m, `so101_capability_map.md`,
OQ-003), confirmed by two separate methods (this session's IK convergence
sweep landed on the same ~0.39-0.40m boundary independently). That's a
physical impossibility regardless of grasp technique, not a physics
realism question, so it's a legitimate correction under the
no-simulation-cheating rule (same category as fixing any other
factually-wrong parameter) -- see `src/omni_q/intel_sim.py`'s module
docstring and the `tableware_pose(...)` call site for fork_1/spoon_1 for
the full writeup, including a second bug this surfaced (the drawer's
`OPEN` op was never kinematically linked to these bodies at all, so
"opening" it never actually revealed them either way).

Observed outcomes, same as v2:

| Outcome | Count |
| --- | ---: |
| Success | 0 |
| Grasp failure | 10 |
| Placement failure | 0 |
| Timeout | 0 |
| Collision | 0 |
| Transition failure | 0 |
| Unresolved | 0 |

**Unchanged, and that's the honest, expected result, not a null result.**
Fixing reachability was necessary but was never expected to be sufficient:
a follow-up probe (free-roll 5-DOF IK on the repositioned targets)
measured the pad-tracking controller's own position error only drops to
~0.05m even under best-case conditions -- looser than the ~0.01-0.02m a
reliable pinch on cutlery actually needs. That same ceiling already
applied to `plate_1`/`cup_1`, which were never a reachability problem.  The
real remaining blocker is controller precision (needs real 6-DOF
pose-aware IK, not just position tracking with pinned/freed roll), not
reach and not orientation search range. Kept as failure evidence, not
converted into a completion claim -- retained alongside v2 (not replacing
it) so the scene-geometry change this run reflects stays auditable against
the prior state.
