# OMNI-Q Weird Stuff Machine

**Omni Q** takes a natural-language objective, observes the world through a vision
stack, decomposes the objective into executable capabilities, routes those
capabilities across available hardware and models, and verifies the result.

Most robotic demos bind a model to a workflow. Omni Q binds an objective to
available capabilities and constructs the workflow at runtime — and rebuilds it
when a constraint changes, a capability disappears, or an attempt fails.

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

## Hackathon context

Built for a LabLab hackathon across sponsor tracks:

| Track | Omni Q manifestation |
| --- | --- |
| **Qualcomm** (primary) | Distributed edge brain — model on Snapdragon X Elite, hardware endpoint on Arduino UNO Q, device-to-device capability routing |
| **Intel online** | Bimanual planner — two simulated SO-101 arms in MuJoCo, camera reasoning, cooperative manipulation |
| **Intel onsite** | Inspect-and-remediate — detect defect, reason, physically correct with one arm |
| **Speechmatics** (bonus) | Voice layer on top of the chosen track — spoken constraints and commands mutate the graph |

Full reasoning in [`docs/strategy-notes.md`](docs/strategy-notes.md).

## Models

Perception starts from an already-published artifact:
[`KissTheHabit/yolov8n-hituav-thermal-finetune`](https://huggingface.co/KissTheHabit/yolov8n-hituav-thermal-finetune)
— YOLOv8 thermal detector, HIT-UAV, 0.825 mAP50 / 0.538 mAP50-95. See
[`models/`](models/).

## Layout

```
docs/            strategy + design notes
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

```
git clone https://github.com/The-Hidden-Canopy/OMNI-Q_Weird_Stuff_Machine.git
cd OMNI-Q_Weird_Stuff_Machine
# install — see requirements once the stack lands
./demo/run_demo.sh
```

## Working the backlog

- [`BACKLOG.md`](BACKLOG.md) — all 48 tasks (OQ-001…OQ-048), owners, dependencies, critical path
- [`agents/`](agents/) — per-owner filtered task lists; bootstrap here without chat history

## Documents

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — capability nodes, graph compilation, placement
- [`DEMO.md`](DEMO.md) — the one reproducible behavior and how to run it
- [`docs/strategy-notes.md`](docs/strategy-notes.md) — track analysis and rubric strategy

## License

Released under the [MIT License](LICENSE).
