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
spoon (placed by the right arm). Known: the cup's top-down pick fails when it
follows the spoon in this ordering (grasp slips at 9 mm; the cup has not
moved) — not yet understood.

Run it live: `python integrations/intel/scripts/demo_authority_change.py --seed 901 --live`
