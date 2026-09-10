# Contact-handoff debug notes — picking up Gerron/GPT's lane (2026-09-10)

Status: GPT's OQ-010/OQ-011 slice (871 lines, **uncommitted** in
`src/omni_q/intel_sim.py`) loads and runs but does **not yet succeed**.
No source files were modified during this debug session.

## Failure modes observed (all deterministic MuJoCo, same config)

| Path | Failure |
|------|---------|
| `run_contact_handoff()` via engine | final pending receipt: `left inverse-kinematics target not reached`, phases=1, cup knocked to (0.478, -0.019, -0.052) — below table plane, scene has **no floor geom** so knocked objects fall forever |
| `_ContactHandoffController.run()` direct | `cup dropped during left_lift_transfer` — grasp + `left_grasp` phase complete, cup lost on lift |
| Manual phase drive (settle → grasp_target → approach) | approach converges (IK 7 iters), but pads **nudge the cup** off its spawn: (-0.260, -0.150, 0.050) → (-0.265, -0.168, 0.051) |

## Diagnosis so far

1. **Root cause candidate**: descending open pads push the cup before the
   grasp closes; grip then closes on the wrong geometry and the cup slides out
   during `left_lift_transfer` (guard trips at cup z < 0.020 m). `_grasp_target`
   computes the pad-bracket target from the pre-approach cup pose and is never
   re-verified against the actual (nudged) cup pose before `left_grasp`.
2. **Engine path amplifies it**: `OmniQ.run()` retries failed steps up to
   `max_revisions`; retries re-enter the controller on a dirty scene (arm
   mid-pose, cup possibly punted), so later attempts fail at `left_approach`
   IK from an awkward start pose. The controller is not reentrant and the
   engine doesn't know that.
3. **No floor** in `contact_handoff_xml` — any punted object drifts below
   z=0 unchecked, corrupting every later measurement. Cheap fix: add floor
   plane with contype/conaffinity matching the contact mask scheme.
4. IK itself is fine: converges in ~7 iterations from HOME to both the
   approach and grasp targets; workspace/range not the issue (capability-map
   reach 0.386 m vs needed ~0.35 m).

## Next steps (priority order)

1. Add a floor geom to `contact_handoff_xml` (contype=0? no — make it
   contype 2/conaffinity 16 like the table so cup:floor contacts are live and
   a punted cup lands instead of falling through).
2. Descend-phase contact handling: stop descent when pads touch the cup
   (`_contact_snapshot` already reports cup:pad labels), re-center the target
   on the live cup pose, then close.
3. Re-verify grasp ownership *before* lifting (`_require_owner("left", ...)`
   exists but runs right after closing; also check pad contact count ≥ 2 per
   arm and cup height not dropping) — then lift.
4. Engine retry interaction: either make `IntelContactHandoffManipulator`
   one-shot (world rejects a second HANDOFF once a receipt is pending) or
   have the controller restore a saved qpos/ctrl snapshot on failure so
   retries start clean. Decide with Gerron/Claude — touches the engine loop.
5. Only after a green deterministic trial: add `tests/test_intel_contact_handoff.py`
   (scene loads; deterministic receipt verifies via `verify_contact_handoff_receipt`;
   world admits only receipt-backed HANDOFF; randomized report smoke with
   small trial count), then mark OQ-010/OQ-011 progress in BACKLOG.
6. Then resume GPT's queue: OQ-007 completion (target place-setting markers
   in `dual_so101_xml` + layout fixture shared with `omni_q.evaluator`),
   OQ-010 general primitives, OQ-016.

## Verified-good facts (rely on these)

- `load_contact_handoff_model()` loads, nu=12, settle 140 steps: cup stable
  at (-0.260, -0.150, ~0.049); parked pads clear the cup spawn.
- `yaw_error`/evaluator/evidence modules: 101+ tests green as of 2026-09-10
  evening run (75→ grew as teammates committed).
- Deterministic run-id collision behavior documented in OQ-037 notes/demo
  wiring (per-scenario evidence bundles).
