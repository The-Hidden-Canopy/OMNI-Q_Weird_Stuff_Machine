# Omni Q — Hackathon Strategy Notes

> Archived planning transcript (LabLab hackathon, sponsor tracks: Intel, SiMa.ai,
> Qualcomm, Speechmatics). Source: Google Doc, imported 2026-09-10. Lightly
> cleaned from an export (escape artifacts removed); two chat turns were mashed
> together in the original around the "spreading attention evenly" line and are
> left as-is.
>
> Superseded for the Intel online track by the official host brief:
> [`docs/challenge-briefs/intel-online-physical-ai-challenge.md`](challenge-briefs/intel-online-physical-ai-challenge.md).
> Where this document speculates ("after the screenshots" section) and the brief
> states something different, the brief is authoritative.
>
> **Track priority is settled: Intel online is the entry track — the only one
> we qualify for.** Qualcomm and Intel onsite are exploratory/bonus, not part
> of the entry; Speechmatics is a bonus that stacks on the Intel online entry.
> Earlier sections below ("Ranking," "Priority stack," "Prioritize: Qualcomm
> first, Intel online second") reflect the pre-brief speculation of Sept 9,
> when Qualcomm's own brief wasn't public yet, briefs hadn't landed, and
> eligibility wasn't yet settled. They're kept as-is for the historical
> record, not as current guidance — see `README.md` and `BACKLOG.md` for the
> live priority.

## Track situation (as of Sept 9)

LabLab's live dashboard still says "Tracks: TBA — Announced soon." What is
published separately:

| Sponsor | What's actually public |
| --- | --- |
| **Intel** | Physical AI / robotics. Both a virtual and onsite challenge using Intel Physical AI tech to build, optimize, and deploy real robotics systems. LabLab previewed an onsite dual-arm object-sorting task using voice commands and Panther Lake. |
| **SiMa.ai** | Named challenge: "Build Physical AI That Sees, Understands, and Acts." Real-world input → AI understanding → meaningful insight/decision/action, using Modalix + Palette. |
| **Qualcomm** | On-device AI across Snapdragon, with Qualcomm AI Hub, GenieX, and Device Cloud. Final specific challenge brief not public yet. |
| **Speechmatics** | Not a primary track — a stackable bonus award. Add Speechmatics to whatever primary track you enter and compete for Best Use of Speechmatics too. |

## SiMa is the obvious Omni playground

Requirement is basically: physical input → understand → do something meaningful.
Don't make a generic vision app — give Omni Q access to the Modalix as one of its
sensory/execution resources.

```
CAMERA → Modalix / YOLO → objects + motion → OMNI Q → dynamically generated task graph → output/action
```

The interesting contribution: what happens after perception isn't hardcoded.
Omni receives capabilities (`detect_person`, `detect_object`, `segment`, `VLM`,
`camera_1`) and composes them.

## Intel gets even stranger

Intel is explicitly looking at models → machines and robotics, describing Physical
AI as heterogeneous compute spanning reasoning, edge inference and control. Omni Q
becomes the layer above the robot primitives.

Instead of programming `recognize red cube → arm.pick(red_cube)`, you expose
`VISION`, `LEFT_ARM`, `RIGHT_ARM`, `GRASP`, `MOVE`, `VOICE`, `VERIFY` to Omni, then
tell it: *"Put the objects away."* Omni creates the execution topology.

The strong demo isn't sorting blocks — everyone sorts blocks. It's changing the
available topology while operating:

```
Disable LEFT_ARM.
Omni:  LEFT_ARM ❌
recompiles onto:  RIGHT_ARM → GRASP → MOVE → VERIFY
```

That demonstrates cognition over physical capabilities.

## Qualcomm may be the most Omni-Q-native track

Wait for the full brief before locking this one. Publicly advertised: on-device
Snapdragon AI, not distributed infrastructure generally. If rules stay broad,
expose device execution options to Omni:

```
SNAPDRAGON            HOST
 ├ NPU                 ├ GPU
 ├ GPU                 └ CPU
 └ CPU
```

Graph becomes function separate from placement:

```
camera → vision → reason → speech
vision → NPU     reason → GPU     speech → NPU
```

Then: drag vision NPU→GPU, or kill the cloud connection, or impose *"Keep
everything local."* Omni recompiles the graph under the new constraint. Much
closer to the Omni Q idea than "deploy our model to Snapdragon."

## Speechmatics rides shotgun

Don't treat it as one of four equal Omni nodes. Use it to make whichever primary
demo you choose controllable by speech — the prize stacks. One project competing
twice.

- *"Omni, don't use the left arm anymore."* → graph mutation
- *"Keep inference on-device."* → placement constraint → recompilation
- *"Watch the second camera too."* → new sensory node → topology extension

## Ranking (based on currently published tracks)

1. **Intel** — Omni controlling/recomposing embodied capabilities
2. **SiMa** — Omni composing perception into arbitrary physical-AI behaviors
3. **Qualcomm** — potentially the best infrastructure experiment, but wait for the wording
4. **+ Speechmatics** on whichever one is chosen

Timing: complete sponsor briefs may appear at kickoff — Sept 10, 15:00 UTC
(8:00 AM Pacific). Don't commit Omni Q's architecture to Qualcomm/Intel
requirements tonight beyond the execution-node abstraction (which works for all
three).

Clean story once briefs land:

```
CAMERA → YOUR YOLO → QUALCOMM / HEXAGON NPU → structured scene state
       → OMNI Q → Intel dual-arm execution → camera verification loop
```

Qualcomm AI Hub already has YOLOv8 deployment paths across Snapdragon/Dragonwing
(incl. Snapdragon X Elite / X2 Elite), tagged for factory automation, robotic
navigation, camera workloads. Stronger angle than "we ran YOLO on Qualcomm":

> Qualcomm becomes Omni Q's local perception substrate.

Omni doesn't ingest raw video unless necessary — Qualcomm produces something
compact:

```json
{
  "frame": 8814,
  "objects": [
    {"id": "part_7", "class": "connector", "conf": 0.96, "bbox": [/* ... */]},
    {"id": "part_8", "class": "sleeve",    "conf": 0.92, "bbox": [/* ... */]}
  ],
  "workspace_clear": true
}
```

YOLO-WORLD support (also on AI Hub across Snapdragon/Dragonwing) means the eyes
can be open-vocabulary: Omni supplies a dynamic object vocabulary
(`black sleeve`, `loose cable`, `red connector`, `screwdriver`) instead of
retraining classes.

Priority stack:

1. Qualcomm — your YOLO / perception
2. Intel — useful bimanual action
3. Omni Q — planning + dynamic task graph
4. Speechmatics — spoken constraints/commands
5. SiMa — only if it earns points or provides something Qualcomm doesn't

Tighter demo: *"Find the damaged connector, hold the assembly steady, reseat it,
and verify the repair."* Qualcomm sees. Omni decides. Intel acts. Qualcomm checks
again.

## After the screenshots — actual challenge wording

- **Intel online** — dual-arm: two simulated SO-101 arms in MuJoCo, natural-language
  instructions, camera reasoning, coordinated manipulation, multi-step
  table-setting, then run on Intel Core Ultra Series 2/3. The "two useful arms"
  idea belongs here, not onsite.
- **Intel onsite** — one SO-101 arm doing autonomous defect detection + physical
  response, using Intel Physical AI Studio, Anomalib, and OpenVINO on Core Ultra
  Series 3. Tailor-made for the YOLO-as-eyes concept (they want Anomalib in the
  workflow too).
- **Qualcomm onsite** — the interesting one:

  > Use GenieX, select and deploy a model from Hugging Face or Qualcomm AI Hub,
  > then bring it to life through on-device AI, hardware integration, and seamless
  > device-to-device collaboration.

  Teams get a Snapdragon X Elite platform + Arduino UNO Q.

  ```
  MODEL → GENIEX → SNAPDRAGON X ELITE → ON-DEVICE AI
        → ARDUINO UNO Q / HARDWARE → DEVICE ↔ DEVICE COLLABORATION
  ```

### One core project, different sponsor manifestations

| Track | Omni Q manifestation |
| --- | --- |
| Qualcomm | Distributed edge brain — model on Snapdragon, hardware endpoint on Arduino, device-to-device capability routing |
| Intel online | Bimanual planner — two simulated SO-101 arms, camera reasoning, cooperative manipulation |
| Intel onsite | Inspect-and-remediate — detect defect, reason, physically correct with one arm |
| Speechmatics | Voice layer on whichever track: "inspect this," "don't move that," "use the other device" |
| SiMa | Optional physical-perception implementation; no longer central |

Qualcomm device split as distinct Omni capabilities:

```
SNAPDRAGON X ELITE          ARDUINO UNO Q
├── YOLO / vision           ├── sensors
├── Omni Q reasoning        ├── actuators
├── local model            ├── physical I/O
└── task routing           └── realtime device state
```

Example issued goal: *"Watch this workspace. When the blue object enters region B,
inspect it. If it matches the target, trigger the actuator. If the connection
disappears, continue locally."* Device-to-device collaboration is intrinsic, not
bolted on.

The challenge says model from HF **or** Qualcomm AI Hub — so the Qualcomm
submission needs a deployable HF/AI Hub artifact, not a 400M-sample model trained
during the competition. The 400M+ corpus is for building Omni Q; the hackathon
deployment artifact can be a much smaller trained/exported model GenieX can carry.

Prioritize: **Qualcomm first, Intel online second, Speechmatics bonus
automatically.**

## Rubric

| Criterion | What Omni Q needs to prove |
| --- | --- |
| Application of Technology | Qualcomm/Intel/Speechmatics are not decorative SDK calls — each performs a necessary function in the loop. |
| Business Value | One system coordinates heterogeneous edge AI + hardware for inspection, manipulation, automation, robotics, field systems. |
| Originality | Omni doesn't just "run an agent" — it dynamically composes capabilities across devices and models into an execution graph. |
| Presentation | The audience can watch the graph form, watch nodes execute, and understand the whole system in under 30 seconds. |

> Do not try to win by showing the most components. Win by showing one behavior
> where every component has a reason to exist.

Strong demo: *"Inspect this workspace, identify the misplaced component, fix it,
and verify the result."*

```
Speechmatics → natural-language goal
Qualcomm Snapdragon → YOLO / local perception → structured world state
OMNI Q → execution graph → task decomposition → device / capability assignment
Intel robotics → physical action
Qualcomm vision → verification → success / replan
```

Business value pitch: **a physical host that adapts without hardcoded workflows**
(set this; move that glass; keep this area clear; hold this while plating; make
space; verify after action). The primary demo framing is OMNI HOME / OMNI CHEF:
another set of hands for a luxury household or personal chef. OMNI-Q TableOps is
the enterprise expansion into banquet reset, hospitality changeover, inspection,
and room operations. Maps later to manufacturing, warehouse automation,
inspection, repair, logistics, lab automation, and field robotics.

Originality sentence:

> Most robotic demos bind a model to a workflow. Omni Q binds an objective to
> available capabilities and constructs the workflow at runtime.

Demo needs ≥3 observable states:

1. **NORMAL** — Omni builds graph and executes.
2. **CONSTRAINT CHANGE** — "Don't touch the red object." Omni modifies graph.
3. **FAILURE / WORLD CHANGE** — object moves / capability disappears / attempt
   fails. Omni replans.

Presentation UI (one screen):

```
┌──────────────┬─────────────────────┬────────────────┐
│ CAMERA       │ OMNI EXECUTION      │ DEVICE STATE   │
│ live scene   │ GRAPH               │ Snapdragon ✓   │
│              │ SEE → PLAN → ACT    │ Intel arm ✓    │
│              │      → VERIFY       │ Speech ✓       │
└──────────────┴─────────────────────┴────────────────┘

GOAL:           "Inspect and correct the workspace."
CURRENT ACTION: STABILIZE(object_4)
WHY:            Required before INSERT(connector_2)
EXECUTING ON:   Intel SO-101
```

Development priority:

1. Visible end-to-end behavior first
2. Qualcomm model actually doing useful work
3. Dynamic Omni graph
4. Physical/simulated action
5. Closed-loop verification
6. Speechmatics as natural-language control
7. Extra models/datasets only after that works

The judges cannot see 400M rows. They can see: *"I moved the object while it was
working, Omni noticed, changed its plan, and the robot finished anyway."*

## GenieX / YOLO deployment path caveat

Hosted on Hugging Face ≠ GenieX can run that exact YOLO artifact directly. GenieX
is primarily an on-device GenAI runtime for LLM/VLM models; object-detection
models live more in AI Hub / QAIRT / ONNX / TFLite paths.

```
YOUR HF YOLO → Qualcomm AI Hub / QAIRT → Snapdragon X Elite → live object detection → OMNI Q
```

Use GenieX for the Omni reasoning/VLM node where visible GenieX usage is expected:

```
camera → YOUR YOLO on Qualcomm → structured scene state → GenieX local model → OMNI Q task decision → Arduino / device collaboration
```

Two real Qualcomm technologies doing two different jobs: AI Hub / QAIRT =
perception acceleration; GenieX = local reasoning/model execution. Claim:

> We brought our own vision model from Hugging Face onto Qualcomm hardware, then
> coupled it to an on-device reasoning layer and physical device control.

## Existing reusable assets

HF account **KissTheHabit**, org **The-Hidden-Canopy**:

- **`KissTheHabit/yolov8n-hituav-thermal-finetune`** — real YOLOv8 thermal
  detector, trained on HIT-UAV, held-out test 0.825 mAP50 / 0.538 mAP50-95.
- **IDA_AI** — ~966M native IDA Lattice body: recurrent selective state,
  local-attention workspaces, cognitive-pressure routing, governed memory.
- **IDA_MoE, IDA_Swift, IDA_Swift_Native** — multiple published model bodies.
- **IDA-TRAIN-V2-nemotron-4b-mxfp8** — 4.19B Nemotron/Minitron via native
  C++/CUDA engine on 2× RTX 5070 Ti, NVFP4 compute, MXFP8 persisted masters;
  preserves failed SFT experiments.
- **MLPerf/SQuAD native-training artifacts** for GPT-2, SmolLM2-135M,
  Qwen2.5-0.5B (records HF checkpoint → native conversion → packed input →
  training-output chain).
- **IDA-family-data** — gated family training/eval/convergence/memory substrate.
- **IDA-TRAIN-V2-mlperf-squad-gb10-results** — metrics/evidence repo for GB10 runs.
- **Neural Foundry / Canopy Foundry** — local native C++20/CUDA training, local
  custody of datasets/weights/checkpoints, bounded coordination.

Git lineage: `GB-THC/IDA-TRAIN-V2`, `ShaggyBeats/Open-World-Model-Harness`,
`RegOS`, `ShaggyBeats/FALCON-DARPA`, `The-Hidden-Canopy/Canopy-Foundry`.
Persistent world/evaluator boundary, immutable run artifacts, hidden truth,
events/audits/state hashes; RegOS execution governance; FALCON's hundreds of
reproducible chained model runs; IDA-TRAIN-V2 native CUDA training, checkpoint
conversion, multi-GPU, FP8/FP4/MXFP8, promotion gates, failure-preserving
evidence. Blackwell write-up documents 5070 Ti experimentation, two-5090
execution, RoPE work, SmolLM2-1.7B forward/backward crossing into the native
engine.

### Primitives already owned

```
PERCEPTION          your YOLO, real EO/thermal pipeline
MODEL EXECUTION      IDA bodies, HF models, native checkpoint conversion
LOW-PRECISION        FP8, NVFP4, MXFP4, MXFP8 masters
TRAINING             IDA-TRAIN-V2, Neural Foundry
WORLD / SIMULATION   Open World Model Harness, persistent truth, scenario/event/eval
GOVERNANCE           RegOS — authority / receipts / traceability
EVIDENCE             hashes, receipts, run artifacts, benchmarks
MULTI-MODEL          IDA family / MoE / differentiated seats
```

Note on the YOLO: it's thermal UAV vision (5-class HIT-UAV), **not** a
table-setting detector. The asset that matters is a *proven YOLO
training/export/deployment path* + a real HF artifact. For workcell/tabletop
vocab, either adapt that pipeline to competition objects or use it as proof the
Qualcomm perception node is your own stack, not a stock demo model.

> Omni Q doesn't need another pile of features. It needs to make existing builds
> interoperable and visibly movable across Qualcomm/Intel hardware.

## Deliverables (confirmed)

1. **Product description** — one paragraph, plain: Omni Q takes a natural-language
   objective, observes the world through the vision stack, decomposes the
   objective into executable capabilities, routes those across available
   hardware/models, and verifies the result.
2. **Presentation** — ≤ 5 minutes, almost entirely behavior:
   - 0:00–0:30  What Omni Q is
   - 0:30–1:00  Why fixed AI workflows suck
   - 1:00–3:30  Live demo behavior
   - 3:30–4:20  Qualcomm / Intel / Speechmatics integration
   - 4:20–5:00  Business value + why this is different
3. **GitHub link with the demo** — the proof surface, not the pitch deck.

### Repo structure (judges inspect fast)

```
README.md
DEMO.md
ARCHITECTURE.md
requirements / setup
src/
models/
integrations/
  qualcomm/
  intel/
  speechmatics/
evidence/
  screenshots/
  receipts/
  benchmark_results/
demo/
  run_demo.*
```

README path: `clone → install → run demo`. Link the existing HF YOLO directly
from the README as an already-published artifact.

> You need one nasty, visible, reproducible behavior and a repo that proves it
> wasn't smoke and mirrors.
