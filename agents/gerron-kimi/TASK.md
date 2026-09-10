# Gerron/Kimi — TASK

Focus: measurement and characterization — the arms, the layout evaluator, the
flourish envelope, Qualcomm profiling, receipts, comparative trials.

> **Standing note:** do NOT spend the morning on the 400M corpus. It keeps
> moving in the background but must not block the demo. Your immediate
> highest-value work is OQ-003, OQ-019, OQ-029/037.

## Queue

| ✓ | ID | Pri | Task | Depends on |
|---|----|-----|------|------------|
| ~ | OQ-003 | P0 | Inspect SO-101/MuJoCo, write arm capability map: joints, limits, gripper range, workspace, wrist-roll limits, control API, cameras | — |
|   | OQ-019 | P0 | Table-layout evaluator: positional/orientation errors + PASS/FAIL for final setting | OQ-007 |
|   | OQ-026 | P1 | Characterize flourish envelope: safe rotation amounts, handoff poses, velocity limits, failure rates | OQ-003, OQ-014 |
|   | OQ-029 | P1 | Qualcomm quantization/perf experiment: latency, memory, accuracy per deployment variant | OQ-028 |
|   | OQ-037 | P1 | Evidence/receipt collector: inputs, model/version, task graph, actions, metrics, hashes per run | OQ-018, OQ-028 |
|   | OQ-046 | P2 | Comparative trials: sequential vs bimanual vs bimanual+flourish (completion, errors, collisions, time) | OQ-045 |

## Start here

OQ-003 is the head of the Intel critical path — nobody can build arm primitives
(OQ-010) safely until the capability map exists.
