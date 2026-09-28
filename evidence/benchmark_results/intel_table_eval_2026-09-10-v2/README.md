# Legacy Intel table-setting randomized report

This directory contains the retained output of
`run_intel_table_evaluation_report(..., trials=10, seed=701)` for the separate
`simulation-scripted-manipulation` route.

The harness applies bounded **build-time** tableware position/yaw perturbations
(default ±3 mm and ±0.08 rad; hard limits ±10 mm and ±0.25 rad) for seeds 701
through 710, then records one canonical Omni Q receipt per trial. Randomized
scene identity is included in the governed envelope, so every receipt has a
distinct run ID. Each receipt is hash-checked before persistence. The report is
exploratory controller evidence, not a promotion gate and not contact-handoff
evidence.

Observed outcomes:

| Outcome | Count |
| --- | ---: |
| Success | 0 |
| Grasp failure | 10 |
| Placement failure | 0 |
| Timeout | 0 |
| Collision | 0 |
| Transition failure | 0 |
| Unresolved | 0 |

The all-failure result is consistent with the documented limitation of the
general position-only 3-DOF grasp controller. It is retained as failure
evidence, not converted into a completion claim.
