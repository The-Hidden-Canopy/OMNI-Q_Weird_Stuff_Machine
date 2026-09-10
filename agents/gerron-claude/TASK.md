# Gerron/Claude — TASK

Focus: the reasoning spine — contracts, world state, bimanual scheduling,
closed-loop verification, natural language + runtime constraint mutation.

## Queue

| ✓ | ID | Pri | Task | Depends on |
|---|----|-----|------|------------|
| x | OQ-001 | P0 | Freeze capability contracts: Observe, Plan, Manipulate, Verify, Device, Receipt | — |
| ~ | OQ-009 | P0 | World-state representation: objects, poses, ownership, goals, constraints across frames | OQ-001, OQ-008 |
| x | OQ-012 | P0 | Bimanual task scheduler: reachability, occupied grippers, deps, workspace conflicts, parallelism | OQ-009, OQ-010 |
| x | OQ-013 | P0 | Collision / resource barriers between arms | OQ-012 |
| x | OQ-015 | P0 | Style constraints in planning ("show off" adds flourishes, final goal unchanged) | OQ-012, OQ-014 |
| x | OQ-018 | P0 | Closed-loop verification / replanning after every manipulation | OQ-008, OQ-009, OQ-016 |
| x | OQ-023 | P1 | Natural-language goal parser → explicit graph constraints | OQ-001 |
| x | OQ-025 | P1 | Runtime constraint mutation ("don't touch the red cup", "spin that plate") | OQ-023, OQ-018 |
| ~ | OQ-031 | P1 | Device-to-device task routing: complementary work across X Elite + Arduino | OQ-028, OQ-030 |
| ~ | OQ-034 | P1 | Unified Omni provider abstraction: Intel + Qualcomm as one graph | OQ-022, OQ-031 |
| x | OQ-044 | P2 | Dynamic arm-role reassignment by reach/state, not fixed left/right | OQ-012 |

## Done

- **OQ-001** — 7 `runtime_checkable` Protocols in `src/omni_q/contracts.py`
  (`World` added in the governed-execution refactor); fakes; `python -m omni_q.demo`.
- **OQ-023** — `src/omni_q/nlu.py`: instruction → canonical goal + `(kind,value)`
  constraints + `spin`/`nudge` `mutations`; `docs`… none, 24 tests.
- **OQ-012 / OQ-013 / OQ-044** — `src/omni_q/scheduler.py` + `ScheduledPlanner`
  (plug-in `Plan` decorator), `docs/scheduler.md`, 20 tests. Wave layering,
  gripper occupancy, workspace/verify/reach barriers, reach+load arm choice,
  HANDOFF chain split. `annotate()` runs on the current engine unchanged.
- **OQ-015** — flourishes (`PRESENT`/…) slot into slack or one pre-verify wave
  or drop; never block a step, never a dep of verify.
- **OQ-025** — `src/omni_q/mutation.py` `RuntimeMutator`, `docs/runtime-mutation.md`,
  10 tests. Constraint-shaped changes queued via `add_constraint` (validated,
  recompiled under the `MissionEnvelope`); `spin`/`nudge` deferred to OQ-011/OQ-014.
- **OQ-031 / OQ-034 (Intel track)** — `src/omni_q/providers.py`
  `Provider` + `ProviderRouter`, `docs/providers.md`, 9 tests. `INTEL_PROVIDER`
  live; `QUALCOMM_PROVIDER` `available=False` and re-routes the same graph once
  OQ-028/OQ-030 land. Complementary routing: perception ≠ manipulation device.

## Blocked / remaining

- **OQ-009** — poses need real sim data (OQ-006/OQ-008). Everything else done.
- **OQ-031 / OQ-034** — Qualcomm track follows OQ-028 (X Elite) / OQ-030 (Arduino).

## Prior-art pass (folded in — see `docs/prior-art.md`)

Lifted from sibling THC repos, additive to OQ-001:
- `AutonomyMode` + `merge_mode` (SOCOM_REACT) — monotone degradation; engine
  escalates on lost capability / `keep_local` / revision-cap.
- `PlanDecision` reason object (SOCOM_REACT) — per-compile audit; forbidden and
  non-authoritative detections are now *recorded as rejected*, not silently
  dropped. Feeds OQ-047.
- `MissionEnvelope` (SOCOM_REACT `SignedMissionEnvelope`) — signed authority,
  `.digest()`, `max_revisions`. Feeds OQ-025/031/034.
- Parent-chained `ReceiptRecord` + `verify_chain` + provenance block
  (FALCON `artifacts.py` + VIGIL `audit/receipt.py`). Feeds OQ-037/038.
- `DataStatus` on `Detection` (Open-World-Model-Harness) — planner won't act on
  non-`LIVE` knowledge. Feeds OQ-009/018.

Deferred (documented): executive/planner split, world transition-request seam,
event causal lineage.

## Next up

- Nothing unblocked in this queue. When OQ-006/OQ-008 land: add poses to
  `Detection` (OQ-009) and swap `FakeObserver` for the real detector.
- When OQ-028/OQ-030 land: fill `QUALCOMM_PROVIDER` device specs, flip
  `available`, verify the same graph routes (already tested).

## Notes

- Keep every stage swappable behind its Protocol; no stage calls hardware.
- The scheduler is live via `ScheduledPlanner` (composition, not a `RulePlanner`
  edit) — wire it into `build_mock_engine` / the demo once the core settles.
- Everything I added is a standalone module + `annotate()` / decorator seam, so
  it survives the concurrent core refactor without conflicts.
