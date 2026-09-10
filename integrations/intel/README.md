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

## Intel onsite

One **SO-101** arm, autonomous defect detection + physical response. Uses Intel
Physical AI Studio, **Anomalib**, and **OpenVINO** on Core Ultra Series 3.

Omni Q role: **inspect-and-remediate** — detect defect, reason about it,
physically correct it with one arm. Anomalib should appear in the workflow
alongside the YOLO perception node.

## TODO

- [ ] MuJoCo dual SO-101 scene + control interface (online)
- [ ] Capability-node wrappers for arm primitives
- [ ] Anomalib + OpenVINO defect path (onsite)
- [ ] Natural-language instruction → capability graph binding
