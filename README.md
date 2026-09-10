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
explicitly labelled `simulation-scripted-manipulation` state transitions. It is
not camera perception, contact-rich grasp control, a trained VLA, OpenVINO, or
hardware evidence yet.

The `intel` project extra tracks the later LeRobot/OpenVINO policy stack; it is
not required for, nor proof of, the current MuJoCo controller smoke.

Set `OMNIQ_RECEIPTS_DIR` to keep per-run evidence bundles (inputs, graph,
actions, metrics, hashes — verifiable via `verify_ledger`) instead of the
default temp location.

To run the judge-facing UI and its mock session event stream:

```bash
PYTHONPATH=src python -m omni_q.server
# open http://127.0.0.1:8770
```

The UI is deliberately labelled **MOCK MODE — NOT HARDWARE**. See
[`docs/ui.md`](docs/ui.md) for the screen map, live constraint behavior, and
the boundary between the frontend event surface and real provider work.

## Working the backlog

- [`BACKLOG.md`](BACKLOG.md) — all 48 tasks (OQ-001…OQ-048), owners, dependencies, critical path — 17 marked done, 10 in progress (core + hand-action specs) as of 2026-09-10
- [`agents/`](agents/) — per-owner filtered task lists; bootstrap here without chat history

## Measured evidence

- [`integrations/intel/so101_capability_map.md`](integrations/intel/so101_capability_map.md) —
  measured SO-101 joint limits, gripper range, wrist-roll travel, and reach from the pinned MuJoCo model (OQ-003)
- [`integrations/intel/flourish_envelope.md`](integrations/intel/flourish_envelope.md) —
  measured joint-velocity, grip-force, and bimanual shared-workspace bounds, with the OQ-014 spin trial protocol (OQ-026 pre-work)
- [`evidence/benchmark_results/yolo_host_baseline_2026-09-10/README.md`](evidence/benchmark_results/yolo_host_baseline_2026-09-10/README.md) —
  host latency/memory baseline for the perception model (ONNX CPU vs PyTorch CUDA) anchoring the Qualcomm variants (OQ-029)

## Documents

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — capability nodes, graph compilation, placement
- [`DEMO.md`](DEMO.md) — the one reproducible behavior and how to run it
- [`docs/ui.md`](docs/ui.md) — judge-facing UI runbook, controls, and scope boundary
- [`docs/strategy-notes.md`](docs/strategy-notes.md) — track analysis and rubric strategy
- [`docs/prior-art.md`](docs/prior-art.md) — patterns borrowed from sibling THC repos (SOCOM_REACT, Open-World-Model-Harness, FALCON-DARPA, VIGIL)
- [`docs/challenge-briefs/intel-online-physical-ai-challenge.md`](docs/challenge-briefs/intel-online-physical-ai-challenge.md) —
  official Intel Online Challenge brief (transcription + [source PDF](docs/challenge-briefs/intel-online-physical-ai-challenge.pdf))
- [`docs/challenge-briefs/intel-online-getting-started.md`](docs/challenge-briefs/intel-online-getting-started.md) —
  host reference links (MuJoCo, SO-101 assets, LeRobot, Physical AI Studio, OpenVINO) mapped to what's done/pending in this repo

## License

Released under the [MIT License](LICENSE).
