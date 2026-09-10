# Gerron/Claude — TASK

Focus: the reasoning spine — contracts, world state, bimanual scheduling,
closed-loop verification, natural language + runtime constraint mutation.

## Queue

| ✓ | ID | Pri | Task | Depends on |
|---|----|-----|------|------------|
| x | OQ-001 | P0 | Freeze capability contracts: Observe, Plan, Manipulate, Verify, Device, Receipt | — |
|   | OQ-009 | P0 | World-state representation: objects, poses, ownership, goals, constraints across frames | OQ-001, OQ-008 |
|   | OQ-012 | P0 | Bimanual task scheduler: reachability, occupied grippers, deps, workspace conflicts, parallelism | OQ-009, OQ-010 |
|   | OQ-013 | P0 | Collision / resource barriers between arms | OQ-012 |
|   | OQ-015 | P0 | Style constraints in planning ("show off" adds flourishes, final goal unchanged) | OQ-012, OQ-014 |
|   | OQ-018 | P0 | Closed-loop verification / replanning after every manipulation | OQ-008, OQ-009, OQ-016 |
|   | OQ-023 | P1 | Natural-language goal parser → explicit graph constraints | OQ-001 |
|   | OQ-025 | P1 | Runtime constraint mutation ("don't touch the red cup", "spin that plate") | OQ-023, OQ-018 |
|   | OQ-031 | P1 | Device-to-device task routing: complementary work across X Elite + Arduino | OQ-028, OQ-030 |
|   | OQ-034 | P1 | Unified Omni provider abstraction: Intel + Qualcomm as one graph | OQ-022, OQ-031 |
|   | OQ-044 | P2 | Dynamic arm-role reassignment by reach/state, not fixed left/right | OQ-012 |

## Done

- **OQ-001** — six `runtime_checkable` Protocols in `src/omni_q/contracts.py`;
  fake providers (`fakes.py`, `devices.py`); engine loop (`engine.py`); SSE
  bridge (`server.py`); `python -m omni_q.demo` runs all three observable
  states; `pytest` = 6 passing (import surface, contract conformance, normal /
  constraint-change / world-change end-to-end, forbidden-object).

## Next up

- **OQ-023** (no new deps) — natural-language goal parser feeding
  `RulePlanner` explicit constraints. Then OQ-009 once OQ-008 lands.

## Notes

- Keep every stage swappable behind its Protocol; no stage calls hardware.
- Scheduler (OQ-012) and verification (OQ-018) are the two things the demo's
  "recover when the world changes" moment depends on — build them defensively.
