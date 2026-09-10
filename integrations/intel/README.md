# Intel integration

Two separate challenges. **Intel online is the entry track — the only one we
qualify for** (see the top-level [`README.md`](../../README.md)). Intel
onsite below is exploratory/bonus, not part of what we're submitting.

## Intel online (entry track)

Official title: **"Bimanual VLA Manipulation with Multi-Modal Reasoning"**, challenge
option **"Setting Up a Dinner Table"**. Full brief:
[`docs/challenge-briefs/intel-online-physical-ai-challenge.md`](../../docs/challenge-briefs/intel-online-physical-ai-challenge.md)
([source PDF](../../docs/challenge-briefs/intel-online-physical-ai-challenge.pdf)).
Host "Getting Started" reference links (MuJoCo, SO-101 assets, LeRobot,
Physical AI Studio, OpenVINO) mapped to what's done here:
[`docs/challenge-briefs/intel-online-getting-started.md`](../../docs/challenge-briefs/intel-online-getting-started.md).
Per that guidance, participants build the MuJoCo scene from scratch — no stock
scene is provided; **SO-101 assets are exactly what's already vendored**
(`assets/menagerie_so_arm100/`, Apache-2.0, sourced from the same
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) lineage
via MuJoCo Menagerie's mirror) — no licensing action needed there.

Two simulated **SO-101** arms in **MuJoCo**, natural-language instructions, camera
reasoning, coordinated manipulation, multi-step table-setting (open a drawer,
retrieve spoons/forks, pick up plate + cup, pour). Runs on Intel Core Ultra Series
2/3. Judged across 10 randomized environment seeds; 100-point rubric weighted
30/20/15/20/10/5 across task completion, VLA reasoning, robustness, OpenVINO
optimization, reproducibility, and innovation — see the brief for the full table.

Omni Q role: **bimanual planner**. Expose `VISION`, `LEFT_ARM`, `RIGHT_ARM`,
`GRASP`, `MOVE`, `VERIFY` as capability nodes; give a goal (*"put the objects
away"*); Omni builds the execution topology. Signature demo move: disable
`LEFT_ARM` mid-run and watch the task recompile onto `RIGHT_ARM → GRASP → MOVE →
VERIFY`.

### Confirmed stack (per the official brief)

| Stage | Tooling |
| --- | --- |
| Simulation engine | **MuJoCo** (or a compatible LeRobot Gym env) — hosts the dual SO-101 scene |
| Policy | **Hugging Face LeRobot** training/fine-tuning; candidate policies **SmolVLA, Pi0.5, ACT** or another VLA/imitation-learning policy |
| Model training | local machine or cloud — Intel provides no training infrastructure |
| Model inference | **Intel OpenVINO** (+ **OpenVINO Physical AI**) — quantize/compile to OpenVINO IR, target CPU/iGPU/NPU on Core Ultra Series 2/3 |

This resolves the ARCHITECTURE.md open question on the OpenVINO export path for
the online track too: the trained policy gets exported to OpenVINO IR for
on-device inference, same shape as the onsite track. **OMPL is not named in the
official brief** — the brief only requires MuJoCo/LeRobot-Gym + LeRobot +
OpenVINO. Treat classical motion planning (OMPL or otherwise) as an optional
internal implementation detail behind `MOVE`/`GRASP`, not a scored/required
component.

## Intel onsite (bonus, not an entry)

Not a track we qualify for — kept only as exploratory notes. One **SO-101**
arm, autonomous defect detection + physical response. Uses Intel Physical AI
Studio, **Anomalib**, and **OpenVINO** on Core Ultra Series 3.

Omni Q role: **inspect-and-remediate** — detect defect, reason about it,
physically correct it with one arm. Anomalib should appear in the workflow
alongside the YOLO perception node.

## TODO

- [x] SO-101 capability map + measurement probe — [`so101_capability_map.md`](so101_capability_map.md),
  [`scripts/probe_so101.py`](scripts/probe_so101.py), vendored MJCF in [`assets/menagerie_so_arm100/`](assets/menagerie_so_arm100/SOURCE.md) (OQ-003)
- [x] MuJoCo dual SO-101 scene + controller smoke — `src/omni_q/intel_sim.py` builds a pinned two-arm proxy (`nu=12`) with tableware and two cameras; the legacy table-setting route remains explicitly scripted
- [x] Bimanual scheduler wired into the Intel sim path (zero-touch decorator, no
  edits to `intel_sim.py`/`scheduler.py`) — `src/omni_q/demo_intel_sim.py`
  (`PYTHONPATH=src python -m omni_q.demo_intel_sim` or `omni-q-intel-demo` once
  installed), `tests/test_intel_sim_scheduled.py`. Confirms real MuJoCo physics
  step under a scheduled graph.
- [x] Table-setting object pack + drawer (OQ-007 first pass) —
  `dual_so101_xml()` now adds a passive slide-jointed `drawer` (opens toward
  the arms, no actuator so `nu` stays 12) holding `fork_1`/`spoon_1`, matching
  the brief's "open the top drawer, retrieve spoons and forks" scenario; every
  tableware geom has an explicit per-material `friction` (ceramic plate/cup,
  metal cutlery, cloth napkin). Fixed the `max_parallelism=1` gap noted above:
  `IntelTableWorld` now gives each object a distinct start zone instead of one
  shared `"staging"` string, and `scheduler.DEFAULT_LAYOUT` got real x-centers
  for those zones (they were previously falling back to `x_center=0.0` for
  *every* unmapped zone, which re-collapsed concurrency even with distinct
  names). Net effect on `set the table`: 11 waves → 7, `max_parallelism` 1 → 2,
  `serialized_conflicts` 12 → 0. Tests: `tests/test_intel_sim_scene.py`,
  extended `tests/test_intel_sim_scheduled.py`. Object placement is still
  explicitly scripted; pose-based (rather than zone-label) region inference
  is still open.
- [x] OQ-010 first pass — single-arm primitives execute individually with
  distinct controller targets instead of PICK/MOVE/PLACE sharing one pose and
  everything else silently falling through to HOME: PICK/MOVE close the
  gripper, PLACE opens it (the gripper joint was never actuated before this),
  OPEN/CLOSE on `object="drawer"` kinematically drive the (unactuated)
  `drawer_slide` joint, ROTATE/PRESENT get their own wrist/pitch targets.
  `IntelTablePlanner` now prepends an `OPEN(drawer)` step before
  `fork_1`/`spoon_1` PICKs, matching the brief's literal scenario. Found and
  fixed two bugs surfaced by actually running this: (1) `MockWorld.apply_transition`
  rejected any step targeting `"drawer"` since it wasn't a registered
  object — fixed by registering it as a `zone == target_zone` fixture so
  `RulePlanner` never treats it as tableware to tidy; (2) `FakeVerifier`'s
  fallback branch judged *any* unrecognized op (including the new `OPEN`)
  against "is the whole workspace tidy," which is never true early in a run —
  it was written assuming only the terminal `VERIFY` step reached that
  branch. Fixed in `src/omni_q/fakes.py` (shared, not Intel-specific) to only
  apply that check to `op == "VERIFY"`. Tests: `tests/test_intel_sim_primitives.py`.
  Still open: bimanual primitives (OQ-011: HANDOFF has a minimal branch,
  STABILIZE/REGRASP/COOPERATIVE_ROTATE don't yet). `OPEN`/`CLOSE` on the
  drawer now have a MuJoCo-backed postcondition check in
  `IntelTableVerifier`; the other legacy primitives retain their existing
  bounded verifier behavior.
- [x] Visual proof + kinematic object placement — `demo_intel_sim.py` gained
  `--viewer` (live interactive MuJoCo window, synced + paced per step) and
  `--render-dir` (PNG per step + assembled `trace.gif`, a first step toward
  the brief's required demonstration video). Rendering the actual physics
  surfaced that PICK/MOVE/PLACE only changed the WorldState zone label —
  the MuJoCo body never moved, so a render showed tableware floating near
  the grippers disconnected from what the receipt claimed happened.
  `IntelTableWorld.apply_transition` now kinematically teleports the
  object (lift-in-place on PICK, snap to `ZONE_POSITIONS[to]` on
  MOVE/PLACE) so what's rendered matches the scripted plan. Still not IK
  or a contact-driven grasp — nothing actually grips anything, it's
  scripted all the way down to the render now instead of stopping at the
  WorldState label.
- [x] OQ-010 real IK + contact grasp for the general table-setting path —
  replaced the kinematic teleport with a real weighted damped-least-squares
  differential IK controller (`IntelTableWorld._ik_reach`/`_ik_track_line`/
  `_move_to`) driving PICK/MOVE/PLACE for all five tracked objects, plus a
  genuine contact grasp attempt (open/close the gripper, measure lift
  height / placement error — no weld, no velocity override). Failure is
  grounded in that measurement, not assumed: a real grasp/placement failure
  reverts the WorldState change and reports the step failed, so the
  engine's replan loop actually retries. Found and fixed two real,
  previously-invisible bugs while getting this running:
  (1) `dual_so101_xml()` silently dropped the source MJCF's
  `<contact><exclude body1="Base" body2="Rotation_Pitch"/></contact>` when
  assembling the dual-arm scene, so the shoulder joint self-collided and
  jammed (actuator saturated, zero net motion) the instant anything tried
  to rotate it off its resting angle on *either* arm — invisible until this
  work was the first thing to ever command that joint. Re-declared per arm
  with prefixed body names. (2) `FakeVerifier`'s fallback branch judged
  *any* unrecognized op against "is the whole workspace tidy" (meant only
  for the terminal `VERIFY` step); every new primitive fell through it and
  was judged against global tidiness at the start of a run. Fixed in
  `src/omni_q/fakes.py` (shared) to only apply to `op == "VERIFY"`.
  **Honest current fidelity**: IK position convergence is reliable (~1cm)
  once aimed at a target clearing the object's own volume, and a
  shoulder-sweep hazard (unweighted redundant IK swinging the forearm
  through tableware even for small vertical motions with no reach need)
  is mitigated via joint-weighted IK + safe-transit-height waypointing.
  The pinch itself has a low success rate: this is 3-DOF position-only
  IK, so wrist orientation is whatever the redundant null-space settles
  into, not controlled to face the jaws at the object. See below —
  the parallel contact-handoff work independently hit and diagnosed the
  same root cause, and has a proven fix pattern (fixed wrist-roll per
  arm) this general path doesn't apply yet. Tests:
  `tests/test_intel_sim_primitives.py` (IK/gripper/revert behavior;
  updated to assert the *honest* outcome, not a success rate not yet
  achieved), full suite green (180/180 at this checkout after merging with the
  contact-handoff/perception work below).
- [ ] LeRobot dataset/demonstration capture from the MuJoCo scene
- [ ] Train/fine-tune a VLA or imitation-learning policy (SmolVLA, Pi0.5, ACT, or other)
- [ ] Capability-node wrappers for arm primitives
- [ ] Policy/perception export to OpenVINO IR, run on Core Ultra Series 2/3
- [x] Bounded environment randomization + 10-seed evaluation harness —
  `IntelSceneConfig` applies explicit build-time tableware position/yaw jitter;
  `run_intel_table_evaluation_report(...)` retains one hashed receipt per seed
  and classifies observed outcomes. This is exploratory controller evidence,
  not a promotion claim.
- [x] Intel inference benchmark script (latency, throughput, device, precision) —
  first real model actually exported to OpenVINO IR and run through OpenVINO
  inference anywhere in this repo. `integrations/intel/scripts/profile_yolo_openvino.py`
  (same fixed-seed/warmup/iteration protocol as
  `integrations/qualcomm/scripts/profile_yolo_host.py`, for direct
  comparability) benchmarked on this machine's real Intel Core 5 210H CPU +
  Intel iGPU: `evidence/benchmark_results/openvino_inference_2026-09-10/`.
  FP32 CPU 42.2ms mean → iGPU 12.0ms (3.5× — device utilization) → NNCF
  INT8 CPU 10.1ms / iGPU 8.5ms (4.2-5× over CPU FP32 — precision/quantization).
  **Scope**: benchmarks the already-published thermal YOLO
  (`KissTheHabit/yolov8n-hituav-thermal-finetune`), not yet the fine-tuned
  7-class table-setting detector (OQ-008's `perception/` fine-tune hasn't
  run — needs a 150-300k-image pull that doesn't fit a short session). This
  proves the export→optimize→benchmark pipeline works end-to-end on real
  Intel hardware; swap the `.pt` when the real fine-tune lands, same
  commands. Also not yet run on actual Core Ultra Series 2/3 hardware
  (this dev machine has a Core 5 210H) — re-run on target hardware before
  submission per the brief's requirement.
- [x] Camera reasoning — closes the OQ-004 audit's other flat gap ("observer
  reads ground-truth state, not rendered frames"). Two independently-built
  pieces landed the same session and compose cleanly: `src/omni_q/vision.py`
  (this work) — a swappable `CameraSource` (`MuJoCoCameraSource`, real
  rendering today; `OpenCVCameraSource` for a real UVC/USB camera on the
  onsite track, **not exercised against physical hardware here** — no rig
  available) feeding a real `OpenVINODetector` (proper YOLOv8 decode: box
  regression + class sigmoid already baked into the exported graph, this
  class does the NMS) and `project_to_table`, a real pinhole back-projection
  from the camera's actual MuJoCo fovy/position/rotation onto the table
  plane — validated by hand against a known object's position before
  trusting it (round-trip + visual cross-check against the rendered frame,
  not just internal self-consistency). `src/omni_q/frame_observer.py`
  (built independently, same session, by someone else acting on the same
  audit finding) — `FrameObserver` implements the full `Observe` contract
  on top of an injectable detector + zone map, with IoU-based stable object
  ids across frames (the brief's "stable object IDs from simulated camera
  frames"), deliberately built with `StubDetector`/`grid_zone_map`
  placeholders precisely so a real detector/zone-map could swap in without
  touching `FrameObserver` itself. `vision.as_frame_detector` /
  `vision.make_camera_zone_map` are that swap-in: real inference in place
  of the stub, and a zone map grounded in actual camera geometry
  (`project_to_table` + nearest-zone lookup) in place of the coarse 3×3
  image-grid guess. End-to-end: a real "Car" detection (thermal-model label,
  honestly meaningless for tableware) lands in zone `tray_cup` — cup_1's
  actual start position. `src/omni_q/demo_vision.py` for a standalone
  run. Same class-label caveat as the OpenVINO benchmark above: real
  pipeline, placeholder classes until OQ-008's fine-tune lands. Tests:
  `tests/test_vision.py`, `tests/test_frame_observer.py`.
- [ ] Anomalib + OpenVINO defect path (onsite)
- [ ] Natural-language instruction → capability graph binding
- [x] Attempted: apply the contact-handoff grasp fix (fixed wrist-roll +
  pad-geom IK target) to the general `_do_pick`/`_do_place` path —
  `IntelTableWorld._ik_reach_pad`, ported from `_ContactHandoffController`.
  **Real, measured improvement, still not a working grasp.** Position
  accuracy tightened significantly (reach error ~0.04-0.05m → as low as
  ~0.01m for some targets), but a swept roll/height search around the
  best-converging configuration never produced a positive lift for
  `cup_1`. More telling: the roll value that converges best for `cup_1`'s
  position (0.8 rad) converges *worst* for `plate_1`'s (1.65 rad — the
  other end of the sweep — is best there). **One fixed scalar per arm does
  not generalize across object geometries/positions** — this needs either
  real per-target orientation solving (not just one pinned roll angle) or
  adopting the contact-handoff scene's more forgiving physics (much higher
  friction, compliant `solref`/`solimp`, contact-group masking, a
  lighter/smaller test object all at once, not individually). Tests still
  assert the honest outcome (195/195 green — none of this regressed
  anything, it just didn't yet close the gap).
- [x] Follow-up investigation, one level deeper on the finding above — two
  separate results, both real:
  1. **Falsified a real hypothesis.** Tested whether the best roll per
     object correlates with the arm's own shoulder bearing to the target
     (a genuine kinematic-compensation idea — rotating the shoulder yaw
     rotates the wrist's reference frame with it, so a fixed world-frame
     roll would need to counter-rotate). Linear fit against measured data
     across all 5 tracked objects: large residuals (up to ~1.9 rad).
     Roll alone doesn't explain the variance — ruled out, not just
     untried.
  2. **Found and fixed a real scene bug.** With wrist-roll fully freed
     (5-DOF, best case), `fork_1`/`spoon_1`'s pad IK still failed to
     converge (~0.20m error) *regardless of roll* — because their scene
     position was ~0.64m from each arm's base, ~65% past the arm's own
     independently measured max reach (~0.386m radius,
     `so101_capability_map.md`, OQ-003 — cross-validated by this
     session's own IK sweep landing on the same ~0.39-0.40m boundary via
     a completely different method). Physically unreachable, full stop,
     independent of grasp technique — a legitimate factually-wrong-
     parameter fix under the no-simulation-cheating rule, not a realism
     compromise. Repositioned both to an in-reach, collision-checked spot
     (`src/omni_q/intel_sim.py`, `tableware_pose(...)` call for
     fork_1/spoon_1). Also surfaced, while investigating: the drawer's
     `OPEN` op was never kinematically linked to these bodies at all —
     "opening" it never moved them regardless of scene geometry, so that
     part of the brief's scenario has been symbolic since OQ-007. Not
     fixed this pass (flagged, scope was the reach bug).
  3. **Reposition alone doesn't close the gap** (expected, checked, not
     assumed): re-ran the 10-seed harness after the fix —
     `evidence/benchmark_results/intel_table_eval_2026-09-10-v3/`, still
     10/10 grasp failure, unchanged from v2. Reachability was necessary
     but never sufficient: a direct probe shows the pad-tracking
     controller's own position error only drops to ~0.05m even at best
     case, looser than the ~0.01-0.02m a reliable cutlery pinch needs —
     the same ceiling `plate_1`/`cup_1` already hit despite never having a
     reachability problem. **The real remaining blocker is controller
     precision** (real 6-DOF pose-aware IK, jointly solving position and
     full orientation with tighter convergence), not reach, not orientation
     search range. 207/207 tests still green.
- [x] **Real orientation-aware IK landed (a teammate's independent work,
  merged same day) -- the first genuine held grasp this whole
  investigation has produced -- plus a real bug caught in the same merge.**
  `_grasp_frame`/`_ik_reach_pad_pose` solve position and orientation
  jointly (using `mj_jacGeom`'s rotational Jacobian, not just position),
  with a bounded roll-candidate retry verified against actual close+lift
  outcome, not just reach error -- exactly the "real 6-DOF pose-aware IK"
  called for above. `cup_1` was also resized to fit the gripper's actually
  -measured envelope (the previous 64mm cup left no real margin) and its
  friction/contact-softness tuned to real ceramic/rubber-pad values.
  **Result, honestly measured**: `world._do_pick(6,"cup_1")` now returns a
  genuine held grasp. `plate_1`/`fork_1`/`spoon_1`/`napkin_1` still don't
  hold. **The same commit also introduced a real problem**: `contype`/
  `conaffinity` set to 0/0 on every non-fingertip-pad arm geom. Under
  MuJoCo's collision rule (`(contype1 & conaffinity2) | (contype2 &
  conaffinity1)` must be nonzero to collide), a geom with both at zero can
  never collide with anything -- the entire arm mesh except the two pads
  could pass through the table, the drawer, and every object with zero
  contact resistance. Not a control improvement; the simulation no longer
  simulating the arm's own body, exactly the category of change the
  no-simulation-cheating rule exists to rule out. Flagged to the user
  before touching it (this repo's collaboration norm for exactly this kind
  of judgment call), given explicit direction to fix it while keeping real-
  world realism, reverted entirely -- no contype/conaffinity anywhere in
  the main scene now, plain MuJoCo defaults, everything collides with
  everything. A second commit landed in parallel re-added a similar
  arm/table exemption comment plus a bitmask-based tableware-vs-tableware
  exclusion (to stop one placement shoving a later object before its own
  governed PICK -- a real, separate, legitimate fix); merged that intent
  forward as explicit named `<exclude>` pairs between the 5 tableware
  bodies instead of a bitmask, so it doesn't reintroduce any arm exemption
  alongside it. **Verified the fix doesn't regress the real grasp**:
  `cup_1` still holds with full collision restored, because the
  orientation-aware control + correctly-sized geometry was doing the real
  work, not the no-clip exemption. Real cost, accepted not hidden: full
  suite runtime ~95s → ~6min once physics is honest (more contact
  resolution, more retry attempts on the objects that still fail).
  Reconciled tests whose premises no longer held (an adversarial pad-
  disable fixture, a separately-landed pattern, replaced two tests that
  had assumed `cup_1` always fails, and a `contype`/`conaffinity`-asserting
  test was rewritten to check for plain MuJoCo defaults + the exclude pairs
  instead), plus one asserting a uniform tableware friction value that had
  lost real material differentiation (napkin/cloth restored to grip more
  than metal cutlery), and one asserting zero workspace scheduling
  conflicts are always achievable (not a real guarantee once physics is
  honest -- loosened to guard the actual regression it was meant to
  catch). 294+/294+ tests green. Not addressed this pass: the separate,
  pre-existing `_ContactHandoffController` scene below has used a similar
  contype/conaffinity scheme since before this session -- predates this
  merge, stays its own deliberately bounded evidence track, worth the
  team's attention on its own terms but out of scope here.
- [x] **The win is bigger than the 10-seed aggregate shows.** Re-ran the
  evaluation harness after both fixes above
  (`evidence/benchmark_results/intel_table_eval_2026-09-10-v6/`): still
  10/10 `grasp_failure` in the summary table, but a trial's raw receipt
  shows `pick_cup_1` *and* `move_cup_1` both genuinely succeeding -- a full
  real pick-and-place -- before the plan reaches `fork_1`, which still
  doesn't converge and exhausts its retries. The report's per-trial
  outcome is a single first-blocking-failure label, so a trial that
  completes one whole object looks identical in the summary to one that
  never succeeds at anything. Two concrete follow-ups this surfaces: (1) a
  finer-grained per-object outcome tally in the report generator, so real
  progress like this doesn't hide behind a single failure label; (2) the
  scheduler could let a persistently-failing object stop blocking attempts
  on the rest of the plan, since that's the only reason `cup_1`'s real
  success doesn't already show up here.

### Legacy table-setting follow-up (2026-09-10)

The legacy `simulation-scripted-manipulation` route now has a bounded,
object-aware contact primitive in `IntelTableWorld`: it derives a live grasp
frame from each object's observed yaw, tracks a named fingertip pad with a
6D damped-least-squares solve, preserves joint-limit margins, and retries only
from a restored MuJoCo snapshot. The scene uses real fingertip-pad friction
tuning and a calibrated `cup_1` envelope (not, as an earlier version of this
note said, pad-only contact geometries or arm/table collision groups --
that turned out to be a no-clip exemption, reverted above); tableware not
shoving other tableware before its own governed PICK is now explicit named
`<exclude>` pairs between the 5 tableware bodies.

This is a capability improvement, not a completed table-setting claim.  The
latest local probes show physically grounded isolated grasps for the calibrated
cup and some of the other objects, while the full legacy sequence still has
shared-workspace/order sensitivity and unresolved cutlery/placement failures.
The engine therefore continues to report failed transitions and replan/hold;
it does not convert those failures into a success score.  The separate
`simulation-contact-handoff` path remains the only OQ-010/OQ-011 promotion
evidence, with its own receipts and 10/10 deterministic gate.
An earlier [10-seed legacy report](../../evidence/benchmark_results/intel_table_eval_2026-09-10-v4/report.json)
(before the no-clip fix) recorded 0/10 complete runs (2 grasp failures, 8
placement failures); the [v6 report](../../evidence/benchmark_results/intel_table_eval_2026-09-10-v6/README.md)
above is the current state. Both are exploratory diagnostic evidence, not a
promotion score.

### Contact-handoff evidence boundary

The OQ-010/OQ-011 contact tranche is available through
`src/omni_q/intel_sim.py` as the separate `simulation-contact-handoff` mode.
It uses the pinned SO-ARM100 proxy, real joint interpolation and named MuJoCo
pad contacts for one `cup_1` transfer. The deterministic acceptance gate is
the pinned controller seed `19` (10/10 in the test suite). The
`run_randomized_contact_handoff_report(root, trials=20)` helper retains every
seeded receipt and labels the summary exploratory, not a promotion claim.
The current retained artifacts are the [deterministic gate](../../evidence/benchmark_results/contact_handoff_deterministic_2026-09-10/report.json)
and the [20-trial randomized report](../../evidence/benchmark_results/contact_handoff_2026-09-10/report.json).

This path never writes the cup free-joint pose, uses weld/equality attachment,
or changes the existing `simulation-scripted-manipulation` table-setting
route. It is MuJoCo-only evidence and does not claim camera perception, VLA
control, hardware, complete table setting, or concurrent execution.
