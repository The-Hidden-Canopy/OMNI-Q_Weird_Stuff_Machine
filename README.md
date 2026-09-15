# OMNI-Q Weird Stuff Machine

**Omni Q** takes a natural-language objective, observes the world through a vision
stack, decomposes the objective into executable capabilities, routes those
capabilities across available hardware and models, and verifies the result.

Most robotic demos bind a model to a workflow. Omni Q binds an objective to
available capabilities and constructs the workflow at runtime — and rebuilds it
when a constraint changes, a capability disappears, or an attempt fails.

## Public Hub

OMNI-Q is a LabLab hackathon project from The Hidden Canopy.

- [OMNI-Q public software directory](https://thehiddencanopy.com/consumer-research-software.html)
- [OMNI-Q release notes](https://thehiddencanopy.com/updates.html#release-notes)
- [Support OMNI-Q Weird Stuff Machine](https://thehiddencanopy.com/flight-deck.html#support)

## Why

Fixed AI workflows break the moment the world moves. Omni Q treats each device,
sensor, and model as a capability node and composes an execution graph on demand:

```
CAMERA → perception → structured scene state → OMNI Q → execution graph
       → device / capability assignment → physical action → verification → success / replan
```

Tell it *"don't use the left arm anymore"* and it recompiles onto the right arm.
Tell it *"keep inference on-device"* and it re-places the graph. Move the object
mid-task and it notices and replans.

## Product framing

The primary product story is **OMNI HOME / OMNI CHEF**: a physical host that
works with a homeowner, entertainer, or personal chef as another set of hands.
It can prepare a configured setting, respond to multiple speakers, ground
references such as “that glass” or “his place,” preserve explicit preferences,
and use bounded free time for useful or expressive behavior.

**OMNI-Q TableOps** is the enterprise expansion: adaptive bimanual setup, reset,
inspection, and reconfiguration of hospitality spaces. Table setting remains
the visible benchmark because it makes the physical intelligence legible. The
repo does not claim universal ROI, room-scale inventory, or live household
capability yet. See [`docs/tableops.md`](docs/tableops.md) for the product
ladder, demo story, pilot metrics, and evidence boundary.

The product surface is now profile-first:

```text
venue / event profile → current-state observation → residual diff
                    → OMNI-Q graph → authorized execution → verification
                    → exception / recovery → hash-chained report
```

The checked-in [`formal_dinner_v3`](profiles/formal_dinner_v3.yaml) profile is
selectable from the UI or through `POST /sessions` with a `profile` field. It
defines six enabled item classes for four seats (24 required placements). The
UI exposes profile status, first-pass placements, self-corrections, replans,
unresolved residuals, final compliance, and the underlying graph/audit/receipt
evidence. This is an executable symbolic-zone workflow, not a claim that the
mock observer has metric camera pose or has physically met a millimetre
tolerance.

## Hackathon context

Built for a LabLab hackathon. **We qualify to enter only the Intel Online
Physical AI Challenge** — that's the actual submission goal. Everything else
below is optional bonus work explored during planning, not part of what we're
entering:

| Track | Status | Omni Q manifestation |
| --- | --- | --- |
| **Intel online** | **Entry track — the only one we qualify for** | Bimanual planner — two simulated SO-101 arms in MuJoCo, camera reasoning, cooperative manipulation, per the official [Intel Physical AI Online Challenge brief](docs/challenge-briefs/intel-online-physical-ai-challenge.md) |
| **Speechmatics** | Bonus, stacks on the Intel online entry | Voice layer — spoken constraints and commands mutate the graph |
| **Qualcomm** | Not an entry — exploratory/bonus only | Distributed edge brain — model on Snapdragon X Elite, hardware endpoint on Arduino UNO Q, device-to-device capability routing |
| **Intel onsite** | Not an entry — exploratory/bonus only | Inspect-and-remediate — detect defect, reason, physically correct with one arm |

Full reasoning in [`docs/strategy-notes.md`](docs/strategy-notes.md).

## Models

Perception starts from an already-published artifact:
[`KissTheHabit/yolov8n-hituav-thermal-finetune`](https://huggingface.co/KissTheHabit/yolov8n-hituav-thermal-finetune)
— YOLOv8 thermal detector, HIT-UAV, 0.825 mAP50 / 0.538 mAP50-95. See
[`models/`](models/).

### Public multimodal OMNI path

The published multimodal OMNI artifact is
[`KissTheHabit/IDA_OMNI_Q`](https://huggingface.co/KissTheHabit/IDA_OMNI_Q).
The strict checkpoint is 977 MB and is intentionally not committed to GitHub;
the repository's `.gitignore` reserves `models/hf_ida_omni_q/` for a local
download:

```bash
hf download KissTheHabit/IDA_OMNI_Q --local-dir models/hf_ida_omni_q
```

The public runtime path is explicit:

```text
HF: KissTheHabit/IDA_OMNI_Q
  → artifact-matched checkpoint + receipt
  → OmniReferenceReasoner
      (src/omni_q/omni_reasoner.py)
  → identity-gated load_omni_reference
      (integrations/intel/vendor/omni_reference/omni_inference.py)
  → OmniPlanner constrained advice
  → governed Intel MuJoCo engine
  → validated controller actions, receipt, and optional render trace
```

Run it with `OMNIQ_OMNI_REASONER=omni`, an artifact-matched
`OMNIQ_OMNI_CHECKPOINT`, and `OMNIQ_OMNI_RECEIPT`, then use
`python -m omni_q.demo_intel_reasoner --render-dir <dir>`. The loader checks
the checkpoint digest, architecture, execution revision, precision, and
completed-example boundary before generation; a missing or mismatched receipt
is refused. The checkpoint is the trained OMNI reasoner, not a monolithic
motor policy: OMNI advises intent and authority while governed providers own
bounded motion.

For public provenance, `demo_intel_reasoner.py` is the model-composed rollout
entry point. `integrations/intel/scripts/watch_sim.py` is a controller/physics
viewer and recorder; its GIFs are useful physical evidence but are not by
themselves proof that the HF reasoner selected the actions. The rollout receipt
must retain the planner decision backend and any `fallback` label alongside the
GIF.

### Real eyes (wired)

The fine-tuned 7-class tabletop detector (OQ-008,
[`KissTheHabit/yolov8n-table-yolo`](https://huggingface.co/KissTheHabit/yolov8n-table-yolo),
v2 fine-tune, **val mAP50 0.324**) is wired into the perception seam behind an
env gate — the weights are a local artifact (`*.pt` is gitignored), so the
real path is opt-in and everything stays green without it:

```
OMNIQ_PERCEPTION=stub        StubDetector (default; no model, no renderer)
OMNIQ_PERCEPTION=yolo        YoloDetector(models/table_yolo_v2_ft_2026-09-11.pt)
                             (+ OMNIQ_YOLO_WEIGHTS to override the .pt path;
                             missing weights -> clear error naming both vars,
                             never a silent fallback to the stub)
```

`src/omni_q/yolo_perception.py` loads the weights through ultralytics and
emits the same `{class, bbox, center, conf}` pixel-space detection as the
OpenVINO path (`vision.as_frame_detector`); `frame_observer.Tracker` then
assigns the **stable per-object id** across frames, so a detection is
`{id, class, bbox, center, conf, zone}` end to end. Class labels come
straight from the fine-tune's own metadata: `plate, cup, fork, spoon, knife,
napkin, drawer`. The same weights are exported for both runtimes:
`models/table_yolo_v2_ft_2026-09-11.onnx` and
`models/table_yolo_v2_ft_2026-09-11_openvino_model/` (IR + `metadata.yaml`).
Try it:

```
PYTHONPATH=src python -m omni_q.demo_yolo_wired   # synthetic tabletop frames
```

### Training from packed masters

`perception/finetune.py` accepts a packaged low-bit master as `--base` —
produce one with `perception/package_quant_weights.py`:

```
python perception/finetune.py \
    --base masters/table_yolo_v2_ft_2026-09-11.mxfp8.npz \
    --w-master mxfp8 --epochs 4
```

With `--w-master mxfp8` the trainer keeps the authoritative weight state as
MXFP8 E4M3 payload plus UE8M0 K32 scales, and Lion momentum as BF16. Repacking
happens inside every successful optimizer step, matching the
IDA-TRAIN-V2 `k_lion_mxfp8` decode→update→RNE-repack equation. The stock YOLO
Conv2d/Linear graph still retains a floating-point compute view; this mode
removes FP32 Adam moments but does not claim zero floating-point model
residency. Removing that compute view requires native packed operators.

The design is evidence-backed by the IDA-TRAIN-V2 MXFP4-master simulation
ablation (`docs/mxfp4-master-sim-ablation-2026-09-09.md` in IDA-TRAIN-V2):
per-step RNE re-encode of the master is stable, while stochastic rounding
*on the master* is a refuted unbounded random walk (SR belongs on gradients
only). Consequently `--w-master mxfp4` / `mxfp2` are **refused at admission**
with a clear error — those tiers are sim-only residency states, not
trainable masters. Loading a master whose sibling manifest disagrees with the
requested `--w-master` is likewise refused (precision contract at the
checkpoint boundary, per IDA-TRAIN-V2 `omni_precision.hpp`).

## Layout

```
docs/            strategy + design notes
  ui.md           judge-facing UI runbook and event-stream scope
src/             Omni Q core: objective → capability graph → routing → verification
models/          model references and export/deploy paths
integrations/    sponsor tech, each doing a necessary job in the loop
  qualcomm/        AI Hub / QAIRT perception acceleration + GenieX local reasoning
  intel/           SO-101 arm control (MuJoCo online / hardware onsite)
  speechmatics/    speech → graph mutation / placement constraints
evidence/        screenshots, receipts, benchmark results
demo/            run_demo entry point
```

## Getting started

Mock mode runs the full Observe → Plan → Manipulate → Verify → Receipt loop with
no hardware and no third-party dependencies (Python 3.10+):

```
git clone https://github.com/The-Hidden-Canopy/OMNI-Q_Weird_Stuff_Machine.git
cd OMNI-Q_Weird_Stuff_Machine
./demo/run_demo.sh                      # PYTHONPATH=src python -m omni_q.demo
pip install -e ".[dev]" && pytest       # governed core + API/UI tests
```

For the current Intel online simulation adapter, install only its measured
MuJoCo dependency:

```
py -3 -m venv .venv
.venv/Scripts/python -m pip install -r integrations/intel/requirements.txt
PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_intel_sim.py -q
```

The current Intel slice loads a real dual-arm MuJoCo scene from the pinned
SO-ARM100 mechanical proxy, moves both controller stacks, and records
contact- and verification-backed state transitions. The table-setting route
still contains a bounded deterministic controller, while the public trained
OMNI path is the reasoner/controller composition described above; those are
separate claims and are labelled separately in receipts. Governed command-level
demonstrations can now be captured with optional MuJoCo telemetry through
[`integrations/intel/demonstrations.py`](integrations/intel/demonstrations.py),
but the current trained artifact is not an end-to-end motor policy and there is
no hardware or OpenVINO policy-deployment claim here.

### VLA-first table setting (2026-09-15, measured)

`integrations/intel/vla/` turns the seam above into a running policy:

```text
overhead + front + wrist cameras, arm state, the step's language
    → SmolVLA (lerobot/smolvla_base, action expert fine-tuned on 40 demonstration episodes recorded from this stack)
    → 7 Cartesian deltas / 10 Hz tick  (SmolVLAController, the governed seam)
    → unweighted 4-joint IK realises each delta; wrist roll from the yaw delta; jaw command direct
    → the world's own grasp / placement verifiers; grasp-integrity guard on carries
    → if the policy stalls: the governed contact primitive continues from where the policy left the arm
```

Measured on the 10-seed randomized harness in VLA mode
([`evidence/benchmark_results/vla_smolvla_2026-09-15/harness_seed900_x10/summary.json`](evidence/benchmark_results/vla_smolvla_2026-09-15/harness_seed900_x10/summary.json)):
8/10 resolved, 48/50 placements, 8/10 fully set at the end. Every single-arm
PICK/MOVE (124) was VLA-led for its budget and completed by the governed
primitive; 0 were completed by the policy alone (it reaches and closes on the
cup, carried the napkin once, and cannot yet localise thin cutlery). The plate
is the governed two-arm primitive. Recordings made in this mode tag every step
on the HUD. Reproduce: `integrations/intel/vla/README.md`.

### Governed learned motor seam: SmolVLA → supervisor → IK

The repository now contains the executable boundary for a future motor-level
VLA without pretending that a trained motor checkpoint already exists:

```text
RGB cameras + arm state + language instruction
                    ↓
              LeRobot SmolVLA
                    ↓  proposal only
             seven Cartesian deltas
                    ↓
             SkillSupervisor
       org / ACTIVE stage / revision / force /
       workspace / speed / safety-stop checks
                    ↓
          MuJoCo Cartesian differential IK
                    ↓
             named position actuators
                    ↓
             physics + verification
```

`ControllerType.VLA` is promotion-gated like the existing learned
controllers: a manifest must carry a `sha256:` artifact digest and at least
one verification channel, and only an `ACTIVE` manifest can reach
`SkillRuntime`. The proposal-only controller is
[`src/omni_q/skills/controllers/smolvla.py`](src/omni_q/skills/controllers/smolvla.py);
the actuator boundary is
[`src/omni_q/skills/actuators/mujoco_cartesian.py`](src/omni_q/skills/actuators/mujoco_cartesian.py).
It requires explicit named joints, actuators, and an end-effector site, so a
missing or mismatched robot binding fails closed rather than silently mapping
the VLA output to the wrong arm.

Install the optional runtime/data-plane dependency with:

```text
.venv\Scripts\python.exe -m pip install -e ".[smolvla]"
```

The adapter follows the current official
[LeRobot SmolVLA documentation](https://huggingface.co/docs/lerobot/smolvla)
and [inference example](https://github.com/huggingface/lerobot/blob/main/examples/tutorial/smolvla/using_smolvla_example.py).
Use [`scripts/record_smolvla_dataset.py`](scripts/record_smolvla_dataset.py)
to write camera/state/action demonstrations from an already governed expert;
it finalizes before any Hub push. A dataset, fine-tuned checkpoint, digest,
promotion evidence, and hardware result are still separate follow-up work and
are intentionally not implied by this code seam.

### Bimanual handoff

One SO-101 acquires an object, transfers it within the shared workspace to the
second arm, and the receiving arm continues the task without resetting the
objective:

```text
SO-101 A: acquire object
        ↓
shared workspace: contact-verified transfer
        ↓
SO-101 B: receive object and continue the plan
```

The current retained evidence is the separate
[`simulation-contact-handoff`](integrations/intel/README.md#contact-handoff-evidence-boundary)
MuJoCo path: a contact-grounded `cup_1` transfer with a deterministic 10/10
acceptance gate and retained randomized receipts. The wider variation sweep is
now a robustness result: **6/20** successful handoffs under position,
orientation, handoff-point, and receiving-posture variation. That is an
engineering envelope for recovery and receiver targeting, not an absence of
bimanual capability. The remaining ugly physical corner is flat/deformable
napkin handling; it is an object-manipulation problem, not a missing
architecture. See the [contact-honesty note](docs/contact-honesty-2026-09-13.md)
and [handoff variation evidence](evidence/benchmark_results/handoff_variation_2026-09-13/README.md).

### Unitree G1 locomotion provider

The repository also contains a separate OMNI-facing provider for Unitree's
actual pretrained G1 12-DOF MuJoCo locomotion policy:
[`src/omni_q/humanoid_g1.py`](src/omni_q/humanoid_g1.py). OMNI supplies only
bounded forward/lateral/yaw velocity intent; the Unitree TorchScript policy
produces leg targets inside its validated 50 Hz controller loop. It is not
merged into the arm fleet backend and it does not receive authority over
objectives, world state, or replanning.

The vendor checkout and weights stay local and are not committed to GitHub:

```powershell
git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_gym.git external/unitree_rl_gym
$env:UNITREE_RL_GYM = (Resolve-Path .\external\unitree_rl_gym).Path
.venv\Scripts\python -m pip install -e ".[g1]"
$env:PYTHONPATH = "src"
.venv\Scripts\python -m omni_q.demo_g1
```

The adapter validates the official `g1.yaml`, `scene.xml`, policy dimensions,
joint names, actuator bindings, and finite policy output before stepping. The
headless test proves that the real local `motion.pt` loads, binds to the
12-DOF model, and advances physics; it is not a claim of walking quality or
hardware deployment. Interactive rollout evidence still requires the viewer
run and a retained receipt.

### Closed-loop demo acceptance

The handoff is only the beginning of the physical-intelligence demo. The
trained controller must remain subordinate to the observed world and to live
authority changes:

```text
camera → updated object state → OMNI objective/authority
       → provider proposal → supervisor → arms
       → contact/vision verification → replan or continue
```

The next proof gates are:

- **Camera reacquisition:** move an object somewhere unexpected; perception
  updates `WorldState`, and the learned policy reacquires the object instead of
  replaying a stale trajectory.
- **Dynamic authority:** while a learned bimanual skill is moving, say
  “don't use the left arm anymore.” OMNI mutates the graph, stops left-arm
  proposals, and replans around the surviving capability.
- **Interrupt and resume:** say “OMNI, stop” mid-handoff, then “continue.” The
  system preserves the objective, re-observes the world, reacquires the object
  state, and resumes from verified reality rather than restarting blindly.
- **Provider fallback:** let deterministic IK attempt a grasp and fail
  verification, then select the learned grasp provider; also exercise the
  reverse path when learned-controller confidence drops. This demonstrates
  interchangeable bounded providers rather than a monolithic robot policy.
- **Object and assignment variation:** repeat across the cup, plate,
  fork/spoon/knife, and napkin with changed positions and arm assignments.
  Partial success is useful evidence only when the receipts retain the failed
  cases and show that the controller did not memorize one prop and one
  choreography.
- **Operational speech:** keep responses short and state-grounded, for
  example, “Left arm unavailable. Reassigning grasp to right arm.” The voice
  output should expose the real authority/replan result, not simulate a
  chatbot layer.
- **Speechmatics turn control:** use Smart Turn, speaker identification/focus,
  and real interruption handling in the live path; raw transcript transport
  alone is not sufficient evidence.
- **OMNI as the brain:** the multimodal OMNI model must provide the actual
  objective/state reasoning used by the demo, with the governed core retaining
  authority. It must not sit beside the demo as an unconnected artifact.

These are acceptance criteria for the next closed-loop demo, not claims that
every gate is complete today. Each gate should retain the camera/world
revision, selected provider, authority decision, interruption or fallback
event, verification result, and final receipt so capability progress can be
distinguished from model, dataset, and promotion progress.

The `intel` project extra tracks the later LeRobot/OpenVINO policy stack; it is
not required for, nor proof of, the current MuJoCo controller smoke.

Set `OMNIQ_RECEIPTS_DIR` to keep per-run evidence bundles (inputs, graph,
actions, metrics, hashes — verifiable via `verify_ledger`) instead of the
default temp location.

To run the judge-facing UI and its default mock session event stream:

```bash
PYTHONPATH=src python -m omni_q.server
# open http://127.0.0.1:8770
```

The default objective-only UI is deliberately labelled **MOCK / NO HARDWARE**.
Selecting a profile shows **TABLEOPS PROFILE / SIMULATED** and keeps the same
no-hardware boundary while adding desired-state diff and profile-report
evidence. To wire the same front end to the Intel MuJoCo session factory, set
`OMNIQ_UI_RUNTIME=intel`; add `OMNIQ_OMNI_REASONER=omni` plus the matched
checkpoint and receipt to use the identity-gated OMNI reasoner. This remains
simulation-only and does not activate hardware. The UI renders runtime
metadata, profile/current-state residuals, table-setting observations, planner
decisions, dependency-aware graph steps, reconnectable event streams, and the
full receipt summary. See [`docs/ui.md`](docs/ui.md) for the profile API,
screen map, live constraint behavior, and the boundary between the frontend
event surface and real provider work.

## Working the backlog

- [`BACKLOG.md`](BACKLOG.md) — all 48 tasks (OQ-001…OQ-048), owners, dependencies, critical path — 17 marked done, 10 in progress (core + hand-action specs) as of 2026-09-10
- [`agents/`](agents/) — per-owner filtered task lists; bootstrap here without chat history

## Measured evidence

- [`integrations/intel/so101_capability_map.md`](integrations/intel/so101_capability_map.md) —
  measured SO-101 joint limits, gripper range, wrist-roll travel, and reach from the pinned MuJoCo model (OQ-003)
- [`integrations/intel/flourish_envelope.md`](integrations/intel/flourish_envelope.md) —
  measured joint-velocity, grip-force, and bimanual shared-workspace bounds, with the OQ-014 spin trial protocol (OQ-026 pre-work)
- [`docs/contact-honesty-2026-09-13.md`](docs/contact-honesty-2026-09-13.md) —
  current contact defaults, the 6/20 handoff robustness envelope, the HF OMNI
  checkpoint provenance, and the remaining camera/re-authorization boundary
- [`evidence/benchmark_results/handoff_variation_2026-09-13/README.md`](evidence/benchmark_results/handoff_variation_2026-09-13/README.md) —
  20 retained handoff receipts showing the wider-variation success/failure
  envelope rather than only the tuned 10/10 case
- [`evidence/benchmark_results/flat_object_grasp_2026-09-13/README.md`](evidence/benchmark_results/flat_object_grasp_2026-09-13/README.md) —
  evidence for the thin, flat, and deformable-object manipulation corner,
  including the remaining napkin gap
- [`evidence/benchmark_results/intel_table_eval_2026-09-10-v2/README.md`](evidence/benchmark_results/intel_table_eval_2026-09-10-v2/README.md) —
  ten seeded legacy table-setting trials with retained, hash-checked receipts; all grasp failures are preserved as exploratory evidence
- [`evidence/benchmark_results/yolo_host_baseline_2026-09-10/README.md`](evidence/benchmark_results/yolo_host_baseline_2026-09-10/README.md) —
  host latency/memory baseline for the perception model (ONNX CPU vs PyTorch CUDA) anchoring the Qualcomm variants (OQ-029)
- [`evidence/benchmark_results/openvino_inference_2026-09-10/README.md`](evidence/benchmark_results/openvino_inference_2026-09-10/README.md) —
  first model actually run through OpenVINO on real Intel CPU + iGPU: FP32 device comparison + NNCF INT8 quantization, same protocol as the ONNX baseline above
- [`evidence/benchmark_results/yolo_2bit_cpu_20260910/README.md`](evidence/benchmark_results/yolo_2bit_cpu_20260910/README.md) —
  2-bit weight-format evaluation on CPU (fp32/mxfp4/nvint2/mxfp2 arms; ~14× footprint cut; output agreement measured; software-dequantize only) — vendored codec at [`integrations/qualcomm/lowbit/`](integrations/qualcomm/lowbit/) (OQ-029)
- [`evidence/datasets/table_yolo_v1_20260910/README.md`](evidence/datasets/table_yolo_v1_20260910/README.md) —
  first multi-source tabletop dataset haul (2,723 unique images, measured dedup; feeds `perception/finetune.py`)

## IDA Omni reasoner (brain)

The planning seam is built for a local reasoner: `src/omni_q/omni_planner.py`
(OmniPlanner — the model advises in a constrained grammar, the governed core
validates every step, deterministic fallback stays truthfully labeled on every
decision) and `src/omni_q/omni_reasoner.py` (backends). YOLO sees; Omni Q
decides; every rationale rides the receipt. Intel-suite hook:
`python -m omni_q.demo_intel_reasoner` (real MuJoCo + reasoner advice, zero
edits to `intel_sim.py`). The IDA Omni body loads through the vendored,
identity-gated harness at
[`integrations/intel/vendor/omni_reference/`](integrations/intel/vendor/omni_reference/)
— select with `OMNIQ_OMNI_REASONER=mock|omni` (+ checkpoint/receipt paths).

## Governed skill runtime

OMNI owns intent and authority; a controller owns only bounded motion
proposals. `src/omni_q/skills/` is the seam for deterministic, scripted, RL,
and residual-RL skills:

```text
PlanGraph / ActionAuthorization
              ↓
        SkillRegistry (ACTIVE only)
              ↓
        controller.propose()
              ↓
        SkillSupervisor (ALLOW / CLAMP / DENY / STOP)
              ↓
        actuator.apply(safe_action)
              ↓
        caller verifies reality and applies the world transition
```

Learned artifacts cannot enter as `ACTIVE`: they move through adjacent,
artifact-matched promotion receipts (`TRAINED → SIM_EVAL →
DOMAIN_RANDOMIZATION → SHADOW → SUPERVISED_HARDWARE → VALIDATED → ACTIVE`). A
residual policy must declare its maximum scale, workspace, velocity, force, and
action bounds. A supervisor denial or stop does not automatically invoke a
fallback skill; the governed planner must issue a fresh, revision-bound
request. This package does not claim hardware validation or a live RL policy—
the current proof is the proposal/supervisor/actuator boundary and its
adversarial tests.

The first concrete learned-controller seam is
[`skills/controllers/rl_grasp.py`](src/omni_q/skills/controllers/rl_grasp.py):
it adapts inference-only grasp policies into proposals without exposing an
actuator. Physical success is a separate evidence decision in
[`skills/verification/grasp.py`](src/omni_q/skills/verification/grasp.py),
which requires measured contact, lift, and relative containment rather than
accepting a completed motion as proof of grasp.

## Speechmatics voice path

Speechmatics is an optional voice layer over the same governed runtime, not a
second command architecture:

```text
Speechmatics partials/finals or Voice API events
    → utterance aggregation and semantic turn handling
    → VoiceRuntime authority and reference checks
    → RuntimeMutator / read-only dialogue handler
    → existing Speechmatics TTS response sink
```

The raw Realtime v2 transport remains available for replay, file, and
microphone runs. The optional `speechmatics-voice` path adds provider-side
Smart Turn, explicit speaker focus, and known-speaker bindings. Speaker focus
is provider filtering only; it never grants command authority. Voice authority
is still granted explicitly with `--operator`, and live post-fix latency plus
multi-speaker acceptance remain unmeasured. See
[`integrations/speechmatics/README.md`](integrations/speechmatics/README.md).

## Residency slider (born-compressed OMNI)

OMNI is **born compressed**: the model never lives in FP32/BF16 as a residency
state — BF16 appears only as plumbing (accumulators, norms) inside kernels.
One slider, selected by available memory:

```
MXFP8 -- MXFP4 -- MXFP2 -- CORE_ONLY
```

MXFP8 is the high-fidelity compressed master (~1 byte/weight), MXFP4 the same
body recompiled lower, MXFP2 extreme residency — through all three MX tiers the
*same* body/capability stays resident; only the format changes. Below the MXFP2
envelope the neural reasoner is evicted: OMNI-Q core remains, autonomy is
DEGRADED, and a novel task produces **HOLD** (queued, never degraded-executed).
`src/omni_q/residency.py` (envelope math, hysteresis, graph-revision lineage,
evidence-bundle receipts) · `demo/residency_slider.py` (walks 8 → 6 → 3 →
1.8 → 1.2 GB and back, writes a validated receipt bundle).

## Documents

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the layered stack: perception swarm → ontology → OMNI-Q → execution; contracts, engine loop, the planner/observer decorator stacks
- [`DEMO.md`](DEMO.md) — the one reproducible behavior and how to run it
- [`docs/ui.md`](docs/ui.md) — judge-facing UI runbook, controls, and scope boundary
- **Reasoning spine** —
  [`docs/scheduler.md`](docs/scheduler.md) (bimanual scheduling, barriers, flourishes, dance-in-slack) ·
  [`docs/rewrite.md`](docs/rewrite.md) (SLIDE/NUDGE optimization, spoken spin/nudge as graph edits) ·
  [`docs/actions.md`](docs/actions.md) (84-op vocabulary + metadata + routines) ·
  [`docs/runtime-mutation.md`](docs/runtime-mutation.md) (`RuntimeMutator` — live spoken changes) ·
  [`docs/ontology.md`](docs/ontology.md) (Claims → fusion authority → deltas → reactive halt/resume)
- **Deployment** —
  [`docs/providers.md`](docs/providers.md) (Intel + Qualcomm as one Omni graph) ·
  [`docs/datasets.md`](docs/datasets.md) (YOLO real-image haul + synthetic depth; Omni policy/planner data)
- [`docs/strategy-notes.md`](docs/strategy-notes.md) — track analysis and rubric strategy
- [`docs/skill-runtime-architecture-2026-09-13.md`](docs/skill-runtime-architecture-2026-09-13.md) —
  current skill contracts, supervisor, promotion gates, and remaining Intel-arm wiring
- [`docs/prior-art.md`](docs/prior-art.md) — patterns borrowed from sibling THC repos (SOCOM_REACT, Open-World-Model-Harness, FALCON-DARPA, VIGIL, IDA-TRAIN-V2)
- [`docs/challenge-briefs/intel-online-physical-ai-challenge.md`](docs/challenge-briefs/intel-online-physical-ai-challenge.md) —
  official Intel Online Challenge brief (transcription + [source PDF](docs/challenge-briefs/intel-online-physical-ai-challenge.pdf))
- [`docs/challenge-briefs/intel-online-getting-started.md`](docs/challenge-briefs/intel-online-getting-started.md) —
  host reference links (MuJoCo, SO-101 assets, LeRobot, Physical AI Studio, OpenVINO) mapped to what's done/pending in this repo
- [`docs/oq-004-requirements-audit.md`](docs/oq-004-requirements-audit.md) —
  independent PASS/GAP audit against the brief + 100-pt rubric: task completion and OpenVINO optimization are the two biggest current point risks
- [`docs/oq-021-red-team-findings.md`](docs/oq-021-red-team-findings.md) —
  deliberate-breakage findings: latent perception false-positive, graph-level vs real-time concurrency, clean passes on unknown-zone / adversarial NL

## License

Released under the [MIT License](LICENSE).

Third-party source and asset provenance is recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Vendored material retains
its own license and is not relicensed by this repository's MIT license.
