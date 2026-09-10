# Intel integration

Two separate challenges — pick per submission.

## Intel online (priority)

Two simulated **SO-101** arms in **MuJoCo**, natural-language instructions, camera
reasoning, coordinated manipulation, multi-step table-setting. Runs on Intel Core
Ultra Series 2/3.

Omni Q role: **bimanual planner**. Expose `VISION`, `LEFT_ARM`, `RIGHT_ARM`,
`GRASP`, `MOVE`, `VERIFY` as capability nodes; give a goal (*"put the objects
away"*); Omni builds the execution topology. Signature demo move: disable
`LEFT_ARM` mid-run and watch the task recompile onto `RIGHT_ARM → GRASP → MOVE →
VERIFY`.

### Confirmed stack (per Intel's "Recommended Software Resources" brief)

| Stage | Tooling |
| --- | --- |
| Simulation engine | **MuJoCo** — hosts the dual SO-101 scene |
| Data collection | **LeRobot** (teleop/demonstration capture) + **OMPL** (motion planning) |
| Model training | local machine or cloud — no mandated trainer |
| Model inference | **Intel OpenVINO** + **OpenVINO Physical AI** — deployed target is Core Ultra Series 2/3 |

This resolves the ARCHITECTURE.md open question on the onsite OpenVINO export path
for the online track too: perception/policy models get exported to OpenVINO IR
for on-device inference, same as onsite. OMPL supplies collision-free arm
trajectories that Omni Q's `MOVE`/`GRASP` nodes call into; LeRobot's dataset
format is the bridge between MuJoCo demonstrations and whatever policy gets
trained.

## Intel onsite

One **SO-101** arm, autonomous defect detection + physical response. Uses Intel
Physical AI Studio, **Anomalib**, and **OpenVINO** on Core Ultra Series 3.

Omni Q role: **inspect-and-remediate** — detect defect, reason about it,
physically correct it with one arm. Anomalib should appear in the workflow
alongside the YOLO perception node.

## TODO

- [ ] MuJoCo dual SO-101 scene + control interface (online)
- [ ] LeRobot dataset/demonstration capture from the MuJoCo scene
- [ ] OMPL motion planning wired into `MOVE`/`GRASP` capability nodes
- [ ] Capability-node wrappers for arm primitives
- [ ] Policy/perception export to OpenVINO IR, run on Core Ultra Series 2/3
- [ ] Anomalib + OpenVINO defect path (onsite)
- [ ] Natural-language instruction → capability graph binding
