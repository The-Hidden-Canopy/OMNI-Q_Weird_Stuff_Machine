# Damion/Claude — TASK

Focus: independent verification. You do not build the demo — you audit it,
break it, and score it against the rubric. Stay independent of the design
discussions where you can.

## Queue

| ✓ | ID | Pri | Task | Depends on |
|---|----|-----|------|------------|
| x | OQ-004 | P0 | Audit Intel challenge requirements vs intended design → PASS/GAP matrix (natural language, camera reasoning, two-arm coordination, multi-step table setting, Intel execution) | — |
|   | OQ-021 | P0 | Break the Intel demo deliberately: moved objects, failed grasp, unreachable object, collision risk, missing detection, bad instruction — record behavior | OQ-018 |
|   | OQ-033 | P1 | Qualcomm rubric audit: model use, on-device AI, hardware integration, device-to-device collaboration each demonstrable | OQ-031 |
|   | OQ-038 | P1 | End-to-end acceptance suite: clean-environment reproduction passes documented demo cases | OQ-035, OQ-037 |
|   | OQ-042 | P1 | Judge-rubric scorecard: every rubric item → a concrete demo behavior / evidence artifact | OQ-039–041 |
|   | OQ-048 | P2 | Final red-team pass: flag anything scripted, unsupported, unverifiable, flaky, or confusing before submission | all demo-critical tasks |

## Done: OQ-004

[`docs/oq-004-requirements-audit.md`](../../docs/oq-004-requirements-audit.md).
Headline: task completion (30 pts) and OpenVINO optimization (20 pts) are the
two biggest point risks, both blocked on the same grasp-reliability gap
(10/10 failure on `evidence/benchmark_results/intel_table_eval_2026-09-10-v2/`,
which I produced this session doing the IK work myself). Camera reasoning is
a flat gap (observer reads ground-truth state, not rendered frames). Note:
I wasn't fully independent of the design work this session — I did hands-on
IK/grasp engineering earlier, which is exactly the area OQ-021 (break the
demo) and OQ-048 (red-team) below need someone who *didn't* build it. Worth
a second, colder pass before submission, or handing OQ-021/048 to someone
else if that independence matters more than my having full context.

## Next up: OQ-021

Break the Intel demo deliberately, once OQ-018's closed-loop verify/replan
has something real to break against — currently blocked on the grasp fix
landing (see the audit), since a 0%-success path doesn't have a "working"
state to break yet. Revisit once task completion moves off 0/10.
