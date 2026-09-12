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

The UI is deliberately labelled **MOCK / NO HARDWARE**. It renders runtime
metadata, current table-setting observations, planner decisions, dependency-
aware graph steps, reconnectable event streams, and the full receipt summary.
See
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
- [`docs/prior-art.md`](docs/prior-art.md) — patterns borrowed from sibling THC repos (SOCOM_REACT, Open-World-Model-Harness, FALCON-DARPA, VIGIL)
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
