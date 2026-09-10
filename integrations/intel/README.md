# Intel integration

Two separate challenges — pick per submission.

## Intel online (priority)

Official title: **"Bimanual VLA Manipulation with Multi-Modal Reasoning"**, challenge
option **"Setting Up a Dinner Table"**. Full brief:
[`docs/challenge-briefs/intel-online-physical-ai-challenge.md`](../../docs/challenge-briefs/intel-online-physical-ai-challenge.md)
([source PDF](../../docs/challenge-briefs/intel-online-physical-ai-challenge.pdf)).

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
- [x] MuJoCo dual SO-101 scene + controller smoke — `src/omni_q/intel_sim.py` builds a pinned two-arm proxy (`nu=12`) with tableware and two cameras; controller steps are real MuJoCo, while object placement is explicitly scripted pending OQ-010
- [x] Bimanual scheduler wired into the Intel sim path (zero-touch decorator, no
  edits to `intel_sim.py`/`scheduler.py`) — `src/omni_q/demo_intel_sim.py`
  (`PYTHONPATH=src python -m omni_q.demo_intel_sim` or `omni-q-intel-demo` once
  installed), `tests/test_intel_sim_scheduled.py`. Confirms real MuJoCo physics
  step under a scheduled graph; surfaced a real gap for OQ-007/OQ-017: every
  tableware item shares the literal zone `"staging"`, so the scheduler's
  workspace-conflict check correctly serializes every pickup
  (`max_parallelism` stays 1) until objects get distinct staging positions.
- [ ] LeRobot dataset/demonstration capture from the MuJoCo scene
- [ ] Train/fine-tune a VLA or imitation-learning policy (SmolVLA, Pi0.5, ACT, or other)
- [ ] Capability-node wrappers for arm primitives
- [ ] Policy/perception export to OpenVINO IR, run on Core Ultra Series 2/3
- [ ] Environment randomization + 10-seed evaluation harness
- [ ] Intel inference benchmark script (latency, throughput, device, precision)
- [ ] Anomalib + OpenVINO defect path (onsite)
- [ ] Natural-language instruction → capability graph binding
