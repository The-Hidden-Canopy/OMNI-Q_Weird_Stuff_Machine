# Contact honesty, handoff variation, and the pickup point — 2026-09-13 (end of day)

Status note written at the stopping point so anyone can pick this up. Every
number below is from a run whose artifacts are in the repo; nothing is
quoted from memory.

## 1. What was found in the physics (and what shipped)

Operator observation in the live viewer: "the arm and props are clipping
through each other." Measured with a contact-buffer spy over full seed-900
trials (`evidence/benchmark_results/contact_honesty_2026-09-13/*.json`,
max penetration per contact-pair class):

| configuration | placed (3 seeds, 5 objects) | pad→cup | pad→napkin | plate→table |
|---|---|---|---|---|
| **soft** (shipped before today: pads+cup `solref .050 1`, `solimp .80 .95 .010`) | 9/15 | 16.5 mm | 10.2 mm | 22.5 mm |
| **firm** time-constant (`0.02 1`, `.90 .99 .0015`) — **now the default** | 10/15 | 7.7 mm | 8.5 mm | 23.8 mm |
| **rigid** direct-stiffness (`-50000 -200`, priority 1) — opt-in `OMNIQ_RIGID_CONTACT=1` | 8/15 | 3.8 mm | 8.1 mm | plate launched |

Two mechanisms, both real:

1. The soft pad contact's `solimp` width of 10 mm meant impedance only
   saturated after a centimetre of penetration. Part of every "grip" was
   interpenetration. Halved by the firm default with no regression.
2. MuJoCo's default (time-constant) contact stiffness scales with the mass
   of the body in contact, so a 0.18 kg plate at 20 ms is ≈450 N/m: 12 N of
   arm push sinks it 27 mm. Direct stiffness (`solref="-k -b"`) is
   mass-independent (measured in isolation: 80 N → 3.3 mm). But under rigid
   contact the current **32 mm puck plate jams its rim in the V between the
   fixed pad and the jaw body** while the open gripper descends (jaw stays at
   q=1.50 the whole time); the wedge turns ~1 N·m of wrist servo into
   1.2–4.7 kN and pops the plate off the table at 2.9 m/s. And the solid cup
   goes 3/3 → 0/3 because its grasp was resting on the penetration.

So rigid contact is the physics the new models must be built against, not
something to switch on with the current primitives. A force-guarded descent
(`OMNIQ_DESCEND_CONTACT_STOP_N`, `_ik_reach_pad(stop_on_contact=…)`,
`_jaw_object_force`) was added and measured: at 10 N it did **not** prevent
the wedge jam and it aborted good descents (10/15 → 7/15), so it is off by
default and kept as the primitive for the edge-pinch plate grasp.

The isolated contact-handoff scene keeps its original compliant pads
(`HANDOFF_PAD_SOLREF/SOLIMP`); its 10/10 deterministic gate failed on first
contact with the firm default and was not re-tuned tonight.

Also fixed: a failed PICK rewinds physics to its pre-attempt snapshot
(`_restore_physics`) — correct as a governed rollback, but it hides the
violence of the failed attempt from the receipt. The plate launch above was
only visible through the step spy.

## 2. Handoff success rate under real variation

`run_randomized_contact_handoff_report(..., variation=...)` now varies three
axes from the trial seed: cup start pose (`position_jitter_m`,
`orientation_jitter_rad`), **handoff location** (`shared_point_jitter_m`,
recorded per receipt as `controller.handoff_point`), and the **receiving
arm's start posture** (`receiver_pose_jitter_rad`, joints 0–3 of the right
arm, commanded through the servos).

`evidence/benchmark_results/handoff_variation_2026-09-13/` (20 trials,
seeds 700–719, variation `{position 0.02 m, orientation 0.15 rad,
shared point 0.03 m, receiver pose 0.15 rad}`):

| outcome | n | where |
|---|---|---|
| success | **6/20** | — |
| right IK target not reached (timeout) | 8 | after `left_lift_transfer` — shifted handoff point outside the right arm's comfortable reach |
| joint limit approached at start | 3 | `initial_settle` — the receiver-pose jitter hit the controller's limit margin |
| cup not stable after release | 3 | `final_settle` |
| drop / collision / ambiguity | 0 | — |

Compare 10/10 with the previous 2 mm jitter: that was repeatability, this is
robustness. Recovery behaviour: none yet in the handoff controller — a
failed phase raises and the trial ends (the table-setting pick has
evidence-driven retries; the handoff does not).

## 3. The Omni checkpoint and the vocabulary question

`KissTheHabit/IDA_OMNI_Q` → `students/PRISM/omni_state_transition_v3/omni_strict_checkpoint.bin`
(977 MB, sha256 `c2ce8259…`, format `ida_omni_strict_checkpoint_v1`,
architecture `ida_omni_state_coupled_v1`, hash `e3ce7085…`, `spec_hash
590584bf1c90677c`, optimizer_step 20). Reader:
`IDA-TRAIN-V2/scripts/read_omni_strict_checkpoint.py`. The download
directory `models/hf_ida_omni_q/` is git-ignored (over GitHub's limit);
re-download with `hf download KissTheHabit/IDA_OMNI_Q`.

Is it a 32k-vocab model? Checked in the weights, not the labels:

- Header geometry (written by the trainer): `hidden 1024, vocab 128000,
  layers 8, experts 16, inner 3072`. Corroborated by tensor sizes
  (`input_projection` = 1024×3072, `position_embeddings` = 2048×1024).
  The embedding's 131,072,000 elements are *also* 32000×4096, so element
  count alone cannot decide — geometry does.
- Lion momentum of `boundary.embedding` (BF16, 128000×1024): the hottest
  rows (norm > p99) are spread across the whole id range — top ids 65408,
  127341, 114757, … A 32k tokenizer can only emit ids < 32000, so those rows
  would be cold. **`ida_lattice_bpe_32k` did not train this checkpoint.**
- HF under the author has `ida_lattice_bpe_32k` and `omni_prism_bpe_256k`;
  the 128k tokenizer the manifest names (`omni_prism_bpe_128k`) is still not
  published. Tensor mapping to the Python reference model is not built.

## 4. What the end-to-end and interruption items look like today

- Perception → OMNI → controller → verification for the **handoff** exists
  as `build_intel_contact_handoff_engine(...).run("transfer cup_1 from left
  arm to right arm")` (receipt-verified, contact-owned). Perception there is
  world-state, not a camera; the camera path (YOLO/OpenVINO) exists for the
  table scene. Wiring the two is the e2e task.
- "Trained model executes the handoff": per the skill-runtime architecture
  (`docs/skill-runtime-architecture-2026-09-13.md`) **OMNI owns intent and
  authority; controllers own motion.** The trained checkpoint is the
  reasoner, not a motor policy. The demo claim should be phrased that way.
- Mid-run "don't use the left arm anymore": the voice side resolves it
  (`InterruptionGate`, NLU `_rule_prefer_arm`/negated side, fixture
  `live_stop_using`); the engine-side re-authorization mid-step is the
  OQ-012 gap and is not done.

## 5. Tomorrow, in order

1. **Realistic models, built from primitives, gated on the harness** (the
   operator's call, and the physics above agrees): plate as a thin disc on a
   foot ring with a graspable lip (edge pinch needs the force-stop
   primitive); hollow cup (prototype in the session scratch: 16 wall boxes +
   bottom, loads and settles, pick untested under firm contact); fork/spoon
   with a bent handle that lifts the grip point off the table, tines and a
   bowl. Then re-run the contact probe with `OMNIQ_RIGID_CONTACT=1` — the
   target is single-digit-mm penetration everywhere with no launches.
2. Handoff recovery (retry the right approach from a re-observed cup pose;
   clamp the receiver-pose jitter inside the limit margin) and re-run the
   variation sweep; re-tune the handoff pads to the firm default.
3. Live viewer: `watch_sim.py --live --trial --seed N` (both arms, all
   objects) and `--live --handoff --seed N` (two-arm pass) both work now —
   the viewer is attached to the world the harness actually steps.
4. OQ-012 mid-run re-authorization, then the e2e with a camera observer.


---

## Update 2026-09-14 — realistic tableware, sideways pinch, two-arm plate

Everything in section 5 item 1 is done and pushed (`58ca3a6`). What was
built, what was measured, and where the numbers are.

### Models (all MuJoCo primitives; `OMNIQ_LEGACY_MODELS=1` restores the old proxies)

| object | model | why this shape |
|---|---|---|
| plate_1 | rimmed soup plate: 5 mm disc on a 10 mm foot ring, stepped wall, 5 mm lip at +25 mm, 200 mm, 200 g | The gripper's moving-jaw tip is 12.5 mm thick and the jaw body behind the fingertip 42 mm; nothing fits under a flat plate's ~8 mm lip. A rimmed soup bowl (real ones are 230×40 mm) carries its lip high enough for the thin fixed fingertip to slide under and the jaw body to clear the table. |
| cup_1 | hollow: bottom disc + 16 wall boxes, 60×90 mm, 3 mm wall, 100 g | A solid cylinder was gripped by interpenetration; the diameter pinch now closes on two thin walls (2.1 mm max penetration under rigid contact). |
| fork_1 / spoon_1 | bent handle (tip and neck slope to the table, grip section arches 7 mm up), 10×7 mm grip section, four tines / ellipsoid bowl, 150 mm, 40 g | A flat slab had nothing to straddle. The arch puts a 7 mm section 8 mm-tall pads can close on; 7 mm because the jaw's hard stop is a 5.6 mm tip gap. Gripped at the centre of mass (`OBJECT_GRASP_ALONG`), not the handle middle — at the middle the bowl-heavy spoon rolled 61° in the pinch and slipped in the first 0.2 s of the carry. |

Render: `evidence/benchmark_results/realistic_models_2026-09-14/models.png`.

### Sideways rim pinch (`_do_pick_edge`, `EDGE_PINCH`)

Horizontal finger, roll 0, fixed fingertip under the lip, moving jaw closes
down on top. Found by scanning joint space, not by the solver: from HOME the
6D damped-least-squares solve converged 0.88 rad off every time; the
forward-kinematic scan (`_edge_seed_joints`) shows the posture exists only
with the fingertip **0.30–0.48 m from the base** at lip height, so the plate
lives at the far centre of the table (near rims at 0.385 m). Waypoints are
scan-seeded then refined by a 4-DOF tip track (`_edge_waypoint`).

### One arm cannot lift the plate; two can

`plate_single_arm_edge_pinch_tilts.png`: one arm pinches the rim, lifts the
near edge 21 mm and the plate hangs at 19° with the far rim on the table —
0.2 N·m of plate torque against ~0.1 N·m of pinch. `_do_pick_bimanual` has
both arms pinch opposite rims and drive to their lifted postures in
lockstep; `_do_place_bimanual` tracks both fingertips along a straight line,
lowers, releases both, homes both. Isolated: **5/5 held and placed** (seeds
900–904, 67 mm lift, ≤1.4 mm penetration). GIF: `tmp/bimanual_plate_900.gif`
(regenerate with `watch_sim.py --record ... --pick plate_1 --move`).

### Through the engine (scheduler-driven, all five objects)

Seeds 900–903, firm contact default: **901, 902, 903 set the whole table
(plate, cup, fork, spoon, napkin all placed); 900 placed 4/5** (plate set
down >60 mm off centre). The 10-trial harness is in
`evidence/benchmark_results/realistic_models_2026-09-14/harness_seed900_x10/`.

Three engine-only failures were found and fixed on the way, all real:

1. A failed MOVE rewound physics into a phantom "still holding" state and the
   scheduler sent that arm, jaw closed on the napkin, to pick the plate. A
   failed placement now leaves the object where it landed, ownership cleared.
2. `OPEN` teleports the drawer by its travel; at 120 mm the open front reached
   the plate's rim at max jitter and slammed it. Drawer at the table edge,
   50 mm travel.
3. Both arms' sideways approaches run down the middle of the table, so the
   plate is set **first** and every zone sits outside those corridors. After
   the two-arm place both arms go HOME — the extended sideways posture is
   near a wrist limit and every later safety check on that arm failed.

### Still open

- Seed 900's plate placement offset (carry drift in the lockstep track).
- Rigid contact (`OMNIQ_RIGID_CONTACT=1`) with the new models: cup/spoon/fork
  hold with 2–5 mm penetration in isolation; the full-engine run under rigid
  contact has not been re-gated since the layout changes.
- Handoff recovery, OQ-012 mid-run re-authorization, camera-observer e2e
  (section 4 unchanged).

### Mid-run change of authority — "don't use the left arm anymore"

`integrations/intel/scripts/demo_authority_change.py` runs the table-setting
goal and, the moment the two-arm plate is set, injects the operator phrase
through the voice path's own NLU (`omni_q.nlu.parse` → `prefer_arm=right` →
`engine.add_constraint`). The engine applies it at its next control boundary
and recompiles (`graph.recompiled`, 7 revisions in the run). The Intel
planner now honours it (it used to set `step.arm` explicitly, which the
scheduler takes before consulting `prefer_arm`, so the instruction was
accepted and ignored): steps the right arm can reach are reassigned, steps
it cannot are **pruned and listed** (`last_authority_report`), the two-arm
plate would be pruned too. Receipt-backed result
(`evidence/benchmark_results/authority_change_2026-09-14/seed-901.json`):
arms used before the instruction `['left']`, after it `['right']`; fork and
napkin pruned "out of the right arm's reach"; objective preserved for the
spoon and the cup (both placed by the right arm after the instruction);
`resolved: false` is the honest verdict — the objective cannot complete
without the left arm, and the receipt says exactly which steps were dropped
and why. (Two engine fixes on the way: a constraint-triggered replan no longer
counts the next, never-attempted step as an attempt — that had pushed the cup
behind the spoon; and the drawer step is dropped once no cutlery remains
planned. `_go_home` now waits for the servos to arrive rather than assuming
a fixed schedule.)

Second harness range, seeds 910–919
(`evidence/benchmark_results/realistic_models_2026-09-14/harness_seed910_x10/`):
**10/10 trials fully successful, 50/50 placements.** Rigid (honest) contact
through the engine, seeds 900–902
(`rigid_contact_engine_3seeds.json`): held 15/15, placed 13/15, max
penetration 7.7 mm (the cloth napkin), 2–3 mm on the cutlery.

Run it live: `python integrations/intel/scripts/demo_authority_change.py --seed 901 --live`

### Handoff: recoveries, and an honest success criterion (2026-09-14)

Two recoveries in `_ContactHandoffController` (recorded per receipt under
`controller.recoveries`): if the receiver's IK times out, the giver — still
holding the cup — re-presents it at the default shared point and the
receiver tries once more; and the receiver's jittered start posture is
clamped inside the controller's own joint-limit margin. Re-running
yesterday's 20-trial wide-variation sweep: 6/20 → 15/20 by the old criterion,
every IK timeout recovered.

Then the criterion was found to be wrong: `_cup_is_stable_on_table` accepted
a cup **lying on its side** (22 mm radius → z = 0.021, and it passes once it
stops rolling). Six of those 15 ended at 90°. The criterion now requires
upright (R[2,2] > 0.985, ~10°). Honest result,
`evidence/benchmark_results/handoff_variation_2026-09-14_recovery/`:
**8/20**. The single remaining failure class: the cup rotates 30–50° in the
receiver's clamp during the carry and tips on release. Tried and measured
worse, all with the deterministic gate failing: firm pads (1/20), closure
0.25 rad (8/20 with 4 drops), 0.15 rad (3/20). The handoff scene keeps its
soft pads and 0.35 rad closure; the real fix is a receiver grasp that
constrains rotation (two contact rows, or a lower grasp on the cup), not a
harder squeeze. The placement now lowers by the cup's own height rather than
a fixed pad height.

### Perception end to end (2026-09-14, afternoon)

The published `table_yolo_v2` detector saw nothing on the current overhead
render (legacy or realistic scene, conf 0.05). The operator pointed at
`KissTheHabit/yolov8n-table-yolo` (real-photo lineage, ftv3-30ep at mAP50
0.39 on 3,387 real images): on the sim render it finds the plate (0.68) and
cup (0.85) but not the cutlery or napkin. So the detector is regenerated from
the scene: `make_table_yolo_dataset.py` renders every scene camera under seed
variation (positions, yaw, colour, light, random subsets already set, an arm
over the table) and labels every object by projecting its own geometry
through that camera; `train_table_yolo.py` fine-tunes and exports OpenVINO IR
with a receipt (val mAP, dataset size, IR sha256).

| detector | base | data | val mAP50 |
|---|---|---|---|
| `table_yolo_v3_2026-09-14` | stock yolov8n | 800 imgs, overhead + third-person | 0.994 |
| `table_yolo_v3_hfbase_2026-09-14` | HF ftv3 (real photos) | same, 7-class map | 0.990 |
| `table_yolo_v4_hfbase_2026-09-14` | HF ftv3 | 1,600 imgs, four scene cameras | **0.980** (fork 0.94, spoon 0.96, rest 0.99) — the demo's detector |

**Multi-camera fusion** (`vision.MultiCameraFusion`): one view misses things
— the overhead camera never saw the spoon parked at x = 0.34, so the
camera-driven plan was built without it. Each camera's detections are
back-projected through that camera's real geometry onto a per-class plane
height (the 90 mm cup's centre landed 7 cm off when the table plane was
assumed from an oblique view), merged by class and world distance, and a
detection needs two cameras or conf ≥ 0.9. Measured on seed 903 with three
cameras: plate, fork, napkin, cup all within 1–2 cm of truth, false
positives (cup-as-napkin, spoon-as-fork) gone. The table-height grazing
camera is replaced by two **flank cameras** looking in at the cutlery — six
cameras with the two wrist cams, the challenge's maximum.

**End to end** (`demo_camera_e2e.py`, `evidence/benchmark_results/camera_e2e_2026-09-14/`):
fused cameras → FrameObserver → OMNI plan → controllers → fused cameras again,
with the camera's per-object "in target zone" verdict compared to the
controllers' own receipts. Two real bugs found by running it:

1. With the camera observer the observation carries no ownership, and the
   scheduler dispatched `PICK cup` to the right arm while it still held the
   spoon; both MOVEs then failed "unsafe carry separation". The world now
   refuses a PICK on an arm that is holding something (one object per
   gripper is a physical fact the world knows regardless of perception).
2. The cup's top-down pick was a tilted local minimum (fingertips 30–80 mm
   apart in height at closure) that lifted by leaning on the wall and broke
   whenever the arm arrived from the spoon's place posture. It now starts
   from a scanned vertical-finger posture (`_topdown_seed_joints`); fresh and
   after-spoon picks behave identically (3/3 placed both ways).

Where camera and controller disagree, the camera is the stricter judge: the
plate the controller counts as placed (within 60 mm) the camera puts nearest
its spawn zone, because the two-arm carry ends ~3 cm short and spawn and
target are only 6 cm apart. That is a true statement about the placement.

Camera end-to-end with the v4 detector and four-camera fusion, seeds 903/905/911
(`evidence/benchmark_results/camera_e2e_2026-09-14/`, with the four post-run
camera frames per seed). Two more fixes came out of the first pass: the fused
position is now the confidence-weighted mean over agreeing cameras (one
oblique view was 3–4 cm off), the camera's verdict is "within 50 mm of the
target zone" rather than nearest-zone argmin (a cup set down 3 cm short of a
zone 6 cm from its spawn zone read as "not moved"), and — the one that
mattered — **the arm now withdraws to HOME after every single-arm place**: it
used to stay parked right above the object it had just set down, and the
overhead camera could not see the napkin under the left gripper.

| seed | camera says in target | controller says placed | agree |
|---|---|---|---|
| 903 | **5/5** | 5/5 | yes — `resolved: true`, fully camera-verified |
| 905 | 4/5 | 4/5 | full agreement; napkin set down >60 mm off (controller) |
| 911 | 4/5 | 5/5 — `resolved: true` | napkin: camera has it ~6 cm off, just outside its 50 mm bar |

(905/911 re-run after one more fix: `_go_home`'s arrival check was position-only,
so a 4.3 rad wrist-roll swing was declared "arrived" as it crossed HOME at speed
and momentum carried it 1.25 rad past into its hard limit — the next place then
failed "joint-limit proximity" on an arm commanded to HOME. Arrival now also
requires low joint velocity.)

Where they disagree the receipt shows both sides. Regression suites after all
of today's changes: 63 passed, 0 failed.

Relationship to `src/omni_q/perception_broker.py` (teammate PR #3, merged
2026-09-14): the broker keeps every detection attached to its camera, link
pose and robot-state sample and deliberately does not fuse; `MultiCameraFusion`
is the resolver above it that associates across cameras into the compact
observation. They compose; nothing in one replaces the other.

### Real drawers (opt-in, `OMNIQ_REAL_DRAWERS=1`), 2026-09-14 late afternoon

Operator: "the arms can be rearranged so they can reach everything including
the drawer." Built without moving the arm bases (everything is tuned to
them): a shallow pull-out cutlery tray on each flank, inside its arm's
reach, the fork lying in the left one and the spoon in the right, a handle
bar standing off the front wall. `OPEN` is a real pinch on the bar and a
50 mm pull along the slide (`_do_open_drawer`), verified by the slide joint
like before. Measured: **pull 6/6** (3 seeds × 2 arms, 50 mm, three-pad
pinch). Three things had to be true first — the tray rides 3 mm above the
tabletop (on the table, finger load became sliding friction and it stalled
at 30 mm), the handle stands clear of the wall top (level with it, the inner
pad landed on the wall), and the walls are 6 mm (12 mm blocked the open
jaw, 121 mm at the tips, from straddling a handle in a 90 mm tray).

Picking the cutlery *out* of the trays is where it stands: ~50% (fork 2/3
held, spoon 1/3 placed). Cutlery at 90° in a tray put the wrist-roll hint
near ±π, which the clip turned into a jaw closing along the handle; the hint
is now chosen among its half-turn twins (`_grasp_frame`), which fixed the
direction but the remaining picks are marginal. So the default layout stays
the symbolic-drawer one (47/50, 50/50 measured), and the real drawers are
opt-in until the tray picks are at that level. The wrist cameras were also
re-aimed (they looked at their own jaw) and the six-camera recording layout
added.

### One arm fails, the other finishes (2026-09-14 evening)

`demo_arm_failure.py`: after the left arm sets the fork, `IntelTableWorld.fail_arm`
freezes its six actuator commands on every physics step (a servo bus with no
host holds its last target) -- a physical failure, the arm stays in frame not
moving. The fault handler withdraws it through the same constraint path an
operator's "don't use the left arm" takes (the constraint contract only lets an
operator change graph authority, so the handler submits under the operator's
standing authority with the fault as justification -- both are in the receipt);
the engine recompiles; the planner re-routes what the right arm can reach. To
make that a real re-route, the napkin now spawns in the **shared band** between
the arms (0.24 m from the left base, 0.32 m from the right) with its zone above
the plate at (0, 0.04), reachable by both (0.30 m); both arms pick and place it
in isolation. Result, seed 903 (`evidence/benchmark_results/arm_failure_2026-09-14/`):
plate (both arms), fork (left) before the failure; spoon, cup **and the napkin**
by the right arm after it; all five placed, `resolved: true`.

What the brief scores here: "adapts plan as scene state changes" (reasoning,
20 pts) and robustness. What it is not: the `ScheduledPlanner`/`ManipulationFleet`
lease path (`engine.report_fleet_fault`) -- the Intel engine uses
`IntelTablePlanner` directly, so that path would HOLD; wiring the fleet into the
Intel engine is the proper follow-up.

### Both arms moving at once (built 2026-09-14 evening — the scope below is what was implemented)

Today only the plate moves both arms simultaneously; every other object is one
arm at a time, because `OmniQ.run` executes one revision-bound transition per
control boundary (`_next_step` → `_run_step` → `world.apply_transition`, and
`TransitionRequest.expected_revision` refuses a second transition compiled
against the same revision). To run e.g. `PICK cup (right)` and `PICK fork
(left)` at the same time honestly:

1. Engine: `_next_steps_parallel` returns up to one ready step per arm whose
   regions do not conflict (the scheduler's barrier/region data already exists),
   and a composite `PARALLEL` transition carries both under one revision.
2. World: a cooperative stepper — each primitive runs in its own thread, its
   `mj_step` calls block on a barrier, and one real physics step advances when
   every active primitive has asked for one; a primitive that finishes
   deregisters. Each thread writes only its own arm's six `ctrl` entries.
3. Receipts: the composite step records both children's results; the
   evidence-driven retry stays per arm.

Re-gate the 10-trial harness afterwards. Until then the claim on camera is:
plate = coordinated two-arm lift; hand-off = two-arm pass; the rest of the
table alternates arms, and a failed arm's work is re-routed.

**Built and measured** (`86d4a98` +): the engine pairs the next ready step
with a ready step on the other arm (different object, independent, and only
once every two-arm step is done — pairing before the plate let one arm pick
the cup and the plate's pinch then dropped it); the pair is one
revision-checked composite (`IntelTableWorld.apply_transitions_parallel`),
both primitives run in threads under a cooperative stepper (one real
`mj_step` per barrier round, each thread writing only its own actuators).
`OMNIQ_PARALLEL_ARMS=0` restores strict one-at-a-time.

What it exposed, fixed on the way: the retry rollback rewound the *whole*
world (per-arm/object now); the cup's diameter pinch only ever held because
the MOVE followed the PICK immediately — a held cup slid out within ~10 s —
so the cup is now a **wall pinch** (jaw part-open on descent, moving
fingertip inside the rim, 3 mm wall clamped; release opens only to the
descent angle until clear of the rim, because a full open from inside shoved
the cup 60–120 mm). Cup placement error is now 5–9 mm.

**Harness, both arms simultaneous, seeds 900–909**
(`evidence/benchmark_results/parallel_arms_2026-09-14/harness_seed900_x10/`):
9/10 trials fully successful, **49/50 placements**; a trial now takes
12–45 s of wall time. Sequence on camera: plate (both arms together) →
cup + fork at the same time → spoon + napkin at the same time.
