# Intel integration

Two separate challenges — pick per submission.

## Intel online (priority)

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

## Intel onsite

One **SO-101** arm, autonomous defect detection + physical response. Uses Intel
Physical AI Studio, **Anomalib**, and **OpenVINO** on Core Ultra Series 3.

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
  STABILIZE/REGRASP/COOPERATIVE_ROTATE don't yet), and OPEN/CLOSE postcondition
  verification (currently a trivial pass, see `FakeVerifier`).
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
  achieved), full suite green (173/173 after merging with the
  contact-handoff/perception work below).
- [ ] LeRobot dataset/demonstration capture from the MuJoCo scene
- [ ] Train/fine-tune a VLA or imitation-learning policy (SmolVLA, Pi0.5, ACT, or other)
- [ ] Capability-node wrappers for arm primitives
- [ ] Policy/perception export to OpenVINO IR, run on Core Ultra Series 2/3
- [ ] Environment randomization + 10-seed evaluation harness
- [ ] Intel inference benchmark script (latency, throughput, device, precision)
- [ ] Anomalib + OpenVINO defect path (onsite)
- [ ] Natural-language instruction → capability graph binding
- [ ] Apply the contact-handoff grasp fix (fixed wrist-roll per arm +
  pad-geom IK target + iterative pad-bracket refinement, all proven 10/10
  in `_ContactHandoffController`) to the general `_do_pick`/`_do_place`
  path above, generalized across object geometries instead of one
  hand-tuned cup sequence.
### Contact-handoff evidence boundary

The OQ-010/OQ-011 contact tranche is available through
`src/omni_q/intel_sim.py` as the separate `simulation-contact-handoff` mode.
It uses the pinned SO-ARM100 proxy, real joint interpolation and named MuJoCo
pad contacts for one `cup_1` transfer. The deterministic acceptance gate is
the pinned controller seed `19` (10/10 in the test suite). The
`run_randomized_contact_handoff_report(root, trials=20)` helper retains every
seeded receipt and labels the summary exploratory, not a promotion claim.

This path never writes the cup free-joint pose, uses weld/equality attachment,
or changes the existing `simulation-scripted-manipulation` table-setting
route. It is MuJoCo-only evidence and does not claim camera perception, VLA
control, hardware, complete table setting, or concurrent execution.
