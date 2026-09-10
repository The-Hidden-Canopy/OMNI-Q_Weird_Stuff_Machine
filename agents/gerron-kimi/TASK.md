# Gerron/Kimi — TASK

Focus: measurement and characterization — the arms, the layout evaluator, the
flourish envelope, Qualcomm profiling, receipts, comparative trials.

> **Standing note:** do NOT spend the morning on the 400M corpus. It keeps
> moving in the background but must not block the demo. Your immediate
> highest-value work is OQ-003, OQ-019, OQ-029/037.

## Queue

| ✓ | ID | Pri | Task | Depends on |
|---|----|-----|------|------------|
| x | OQ-003 | P0 | Inspect SO-101/MuJoCo, write arm capability map: joints, limits, gripper range, workspace, wrist-roll limits, control API, cameras | — |
| x | OQ-019 | P0 | Table-layout evaluator: positional/orientation errors + PASS/FAIL for final setting | OQ-007 (stub behind neutral PlacedObject input — mock fixtures today) |
| ~ | OQ-026 | P1 | Characterize flourish envelope: safe rotation amounts, handoff poses, velocity limits, failure rates | OQ-003, OQ-014 — **measured bounds done** (`integrations/intel/flourish_envelope.md` + `evidence/benchmark_results/flourish_envelope_2026-09-10/`): rotation limits, joint/TCP velocities, grip-force envelope, shared-workspace handoff zone; failure-rate trials await OQ-014, protocol ready in doc §5 |
| ~ | OQ-029 | P1 | Qualcomm quantization/perf experiment: latency, memory, accuracy per deployment variant | OQ-028 — **host baseline measured** (`evidence/benchmark_results/yolo_host_baseline_2026-09-10/`: ONNX Runtime CPU + torch CUDA, shared protocol script `integrations/qualcomm/scripts/profile_yolo_host.py`); X Elite variants remain, blocked on OQ-028 |
| x | OQ-037 | P1 | Evidence/receipt collector: inputs, model/version, task graph, actions, metrics, hashes per run | OQ-018, OQ-028 (stub behind frozen Receipt contract — mock runs today) |
|   | OQ-046 | P2 | Comparative trials: sequential vs bimanual vs bimanual+flourish (completion, errors, collisions, time) | OQ-045 |

## Start here

OQ-003 is the head of the Intel critical path — nobody can build arm primitives
(OQ-010) safely until the capability map exists.
