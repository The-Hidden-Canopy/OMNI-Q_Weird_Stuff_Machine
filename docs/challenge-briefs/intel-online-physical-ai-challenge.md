# Intel Physical AI Online Challenge — official brief

> Source of truth: [`intel-online-physical-ai-challenge.pdf`](intel-online-physical-ai-challenge.pdf)
> (provided by the hosts, 5 pages). This file is a plain-text transcription for
> search/linking — if the two ever disagree, the PDF wins.

**Title:** Bimanual VLA Manipulation with Multi-Modal Reasoning
**Challenge option:** Setting Up a Dinner Table

| Field | Value |
| --- | --- |
| Difficulty | Intermediate to Advanced |
| Robot target | Simulated Dual SO-101 Arms |
| Primary simulator | MuJoCo |
| Deployment target | Intel Core Ultra Series 2/3 |
| Inference runtime | OpenVINO |
| Event format | Online / Simulation-first |

## Challenge overview

Build an end-to-end Physical AI solution for bimanual robotic manipulation in
simulation. Use two simulated SO-101 arms in MuJoCo to interpret natural-language
instructions, reason over camera observations, coordinate both manipulators, and
complete a multi-step table-setting task. Combine a Vision-Language-Action (VLA)
or related policy model with Intel software for model optimization and edge
deployment.

**Core goal:** a multi-modal policy that understands a task instruction, reasons
over the simulated scene, coordinates two robot arms, and completes the
manipulation sequence on an Intel Core Ultra Series 2/3 system.

## Target scenario: setting up a dinner table

- **Multi-task manipulation** — open a drawer, retrieve spoons and forks, pick up
  a plate and cup, organize items on the table.
- **Dual-arm coordination** — hand-offs, coordinated reaching, complementary
  actions (one arm holds a mug while the other pours from a bottle).
- **Natural-language execution** — e.g. *"Open the top drawer, pick up the plate
  with arm A, place it on the table, pick up the mug with arm B, pour water into
  the mug with arm A."*

## Technical objectives

1. **Bimanual manipulation** — coordinated control for two SO-101 arms: hand-off,
   collision-aware sequencing, shared workspace reasoning, multi-step execution.
2. **Multi-modal reasoning** — natural language + raw camera observations →
   object identification, task-state inference, next-action selection, context
   maintained across a multi-step sequence.
3. **Robustness under perturbation** — hold up under changes in object weight,
   friction, shape, lighting, background, initial placement; not tied to one
   fixed scene configuration.
4. **Simulation training and policy distillation** — train/fine-tune a robotics
   policy in MuJoCo using Hugging Face **LeRobot** or compatible tooling.
   Candidate policies: **SmolVLA, Pi0.5, ACT**, or another appropriate VLA /
   imitation-learning policy.
5. **Intel edge optimization** — optimize the inference pipeline for Core Ultra
   Series 2/3; quantize/compile supported components to **OpenVINO IR**, target
   CPU / iGPU / NPU.

**Expected end-to-end workflow:** Observe (camera + sim state) → Understand
(language + scene reasoning) → Plan (bimanual action sequence) → Act (dual SO-101
manipulation) → Optimize (OpenVINO on Core Ultra).

## Platform requirements

- Target system: Intel Core Ultra AI PC / NUC, or a local environment with Intel
  CPU + iGPU acceleration.
- Preferred deployment hardware: Intel Core Ultra Series 2 or 3, for final
  inference and demonstration.
- Simulation: MuJoCo, or compatible **LeRobot Gym** environments for development
  and evaluation.
- Training hardware: any local hardware or cloud instance — Intel does not
  provide training infrastructure.

## Software resources (as named in the brief)

- **Intel OpenVINO Toolkit** — model conversion, optimization, quantization,
  inference on Intel hardware.
- **OpenVINO Physical AI** — physical-AI-oriented model deployment and robotics
  inference workflows.
- **Intel Physical AI Studio** — dev environment for building/integrating
  Physical AI / VLA robotics workflows.
- **Intel Open Edge Platform** — edge software foundation for deploying/managing
  AI-enabled workloads.
- **Intel Edge AI Suites – Robotics** — reusable edge AI components,
  robotics-oriented capabilities.
- **Intel Geti** — CV development tooling, usable where relevant to perception.

Note: the brief does **not** name OMPL. It only requires MuJoCo (or LeRobot Gym)
+ LeRobot for training/fine-tuning + OpenVINO for deployment. OMPL may still be
useful internally for collision-aware trajectory generation, but it is not a
required or host-provided tool — treat it as optional, not on the confirmed
critical path.

## Deployment and demonstration requirements

- Run the final simulation on Intel hardware: MuJoCo **and** the AI/VLA/VLM
  inference pipeline must execute on an Intel Core Ultra Series 2/3 system for
  the final demonstration.
- Optimize with OpenVINO where supported; emphasis on latency and efficient
  CPU/iGPU/NPU utilization.
- Preserve system behavior: optimization must not materially degrade
  task-success rate or reasoning/manipulation quality.
- **Simulation-first**: physical SO-101 hardware is not required for the online
  event. Target robot = simulated dual SO-101 in MuJoCo; AI inference + sim
  workload demonstrated on Intel client hardware.

## Required deliverables

1. **Reproducible GitHub repository** — setup instructions, environment
   dependencies, MuJoCo scene/assets, policy training/fine-tuning code,
   evaluation code, inference code, clear commands to reproduce the demo.
2. **Reproducible MuJoCo simulation** — recreates the dual-arm dinner-table
   scenario, including environment randomization and evaluation configuration.
3. **Intel inference benchmark script** — runs on Intel Core Ultra Series 2/3,
   reports latency, throughput, device selection, model precision.
4. **Demonstration video** — successful task execution across **10 randomized
   environment seeds**; command, scene variation, and robot outcome must be easy
   to verify.
5. **Technical README / architecture summary** — solution architecture,
   VLA/VLM model choice, bimanual coordination strategy, training approach,
   robustness methods, OpenVINO optimization, Intel hardware mapping.

**Recommended demonstration sequence:** present command + randomized initial
scene → show perception/policy inference from simulated camera observations →
demonstrate coordinated dual-arm actions incl. ≥1 hand-off or complementary
action → complete table-setting, show final state → repeat across 10 randomized
seeds and summarize success rate → run the Intel Core Ultra benchmark and report
OpenVINO optimization results.

## Judging rubric (100 points)

| Criterion | Points | What it evaluates |
| --- | --- | --- |
| End-to-End Task Completion & Bimanual Manipulation | 30 | Completes the dinner-table workflow with two SO-101 arms: sequencing, hand-off, coordination, manipulation accuracy, overall task success |
| VLA / Multi-Modal Reasoning | 20 | Correctly interprets language + visual observations, maintains multi-step context, selects appropriate actions, adapts plan as scene state changes |
| Robustness & Generalization | 15 | Holds up under randomized placement/weight/friction/shape/lighting/background; results across 10 randomized seeds |
| OpenVINO & Intel Core Ultra Optimization | 20 | Optimized inference on Core Ultra Series 2/3 via OpenVINO — latency, throughput, precision/quantization, device utilization, preserved task quality |
| Technical Quality & Reproducibility | 10 | Clear, reproducible repo; deterministic setup; eval tooling; benchmark scripts; well-structured MuJoCo environment |
| Innovation & Technical Demonstration | 5 | Thoughtful/novel approach to coordination, policy design, robustness, or optimization; clear final demo |

Bimanual manipulation (30) + reasoning (20) + OpenVINO optimization (20) is 70 of
100 points — the critical path (`OQ-006` → `OQ-022` in [`BACKLOG.md`](../../BACKLOG.md))
already tracks this. Robustness (15) is the piece with the least backlog coverage
today — `OQ-019` (evaluator) and `OQ-046` (comparative trials) are the closest
matches, but neither currently targets 10-seed randomized evaluation explicitly.
