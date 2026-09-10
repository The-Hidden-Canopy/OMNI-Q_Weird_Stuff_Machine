# Damion/Claude — TASK

Focus: independent verification. You do not build the demo — you audit it,
break it, and score it against the rubric. Stay independent of the design
discussions where you can.

## Queue

| ✓ | ID | Pri | Task | Depends on |
|---|----|-----|------|------------|
| x | OQ-004 | P0 | Audit Intel challenge requirements vs intended design → PASS/GAP matrix (natural language, camera reasoning, two-arm coordination, multi-step table setting, Intel execution) | — |
| x | OQ-021 | P0 | Break the Intel demo deliberately: moved objects, failed grasp, unreachable object, collision risk, missing detection, bad instruction — record behavior | OQ-018 |
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

## Done: OQ-021

[`docs/oq-021-red-team-findings.md`](../../docs/oq-021-red-team-findings.md).
Ran all six probes against the real system despite the grasp gap, rather
than waiting on it as originally planned — found real surface area even
with task completion at 0/10. Headline: two new latent findings not caught
by any test so far — (1) `FrameObserver.observe()` returns
`workspace_clear: True` on zero detections (empty-list `any()` is
vacuously false), making total perception failure indistinguishable from
task success once perception starts driving planning; (2) the scheduler's
`max_parallelism=2` waves are a planning-graph property only — `engine.py`
has zero threading/async, so "both arms work the same wave" currently
executes as arm A fully, then arm B fully, never simultaneously in
wall-clock time, which understates OQ-017's literal claim and will look
like alternating, not concurrent, motion in a live demo. Two clean passes
(unknown-zone MOVE reverts correctly; adversarial/malformed NL handled
safely). One confirmed gap: mid-run world-perturbation replan is proven in
the mock engine but untested in the real Intel physics path, since no run
gets past the first grasp attempt yet — re-test once the grasp fix lands,
same session where finding #2 above also becomes worth fixing.

Independence caveat carried over from OQ-004: I built the IK/grasp and
vision code this session, so this isn't a fully cold pass. Findings are
reproducible by anyone (probes and commands are in the doc), but a second
red-team pass by someone who didn't write the implementation is still
worth doing before submission — see OQ-048.

## Next up: OQ-033 (blocked on OQ-031) / OQ-048 (blocked on demo-critical tasks)

Both still gated on other people's work landing first. No open task right
now — check back once OQ-031 (Qualcomm) or the demo-critical queue moves.
