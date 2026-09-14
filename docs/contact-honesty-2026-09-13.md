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
