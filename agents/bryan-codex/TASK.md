# Bryan/Codex — TASK

Focus: everything the judges see — the live graph UI, bimanual visualization,
the Qualcomm device graph, judge demo mode, README path, presentation.

## Queue

| ✓ | ID | Pri | Task | Depends on |
|---|----|-----|------|------------|
|   | OQ-005 | P0 | Minimal live Omni graph UI shell: goal, observations, both arms, graph nodes, current action, verification state | OQ-001 |
|   | OQ-020 | P0 | Visualize bimanual execution: ARM-A/ARM-B actions, parallel intervals, handoffs, barriers, object ownership — live | OQ-012, OQ-005 |
|   | OQ-032 | P1 | Qualcomm device graph UI: which node runs on Snapdragon vs Arduino, data moving between them | OQ-031 |
|   | OQ-036 | P1 | Judge-facing demo mode: one screen = goal → perception → plan → execution → verification, no narration needed | OQ-020, OQ-032 |
|   | OQ-039 | P1 | GitHub judge path: README "run this", architecture image, demo GIF/video link, sponsor-tech mapping | OQ-035 |
|   | OQ-040 | P1 | Product description (~1 paragraph, no jargon) — with Gerron | OQ-036 |
|   | OQ-041 | P1 | Five-minute presentation structure with failure margin | OQ-036 |
|   | OQ-047 | P2 | "Why Omni did that" display: one-sentence reasoning per graph transition | OQ-018 |

## Start here

OQ-005 only depends on OQ-001 (contracts). Consume the engine's `EventBus`
stream — `graph.compiled`, `graph.recompiled`, `step.started/finished`,
`observed`, `verified`, `constraint.added`. Stack: Python core + thin TS UI.

Contract shapes to render are in `src/omni_q/contracts.py`
(`PlanGraph.as_dict()`, `Step.as_dict()`, `Observation.as_dict()`).
