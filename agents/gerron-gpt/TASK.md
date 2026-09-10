# Gerron/GPT — TASK

Focus: everything that touches the simulator, the arms, the sponsor runtimes,
and the launchers. You own the Intel critical path's execution half and OQ-002.

## Queue

| ✓ | ID | Pri | Task | Depends on |
|---|----|-----|------|------------|
|   | OQ-002 | P0 | Agent-control repo scaffold: five agent dirs, task files, repo maps, coding pointers, active scopes | — |
|   | OQ-006 | P0 | Intel MuJoCo dual-SO-101 environment boots + accepts joint / end-effector actions | OQ-003 |
|   | OQ-007 | P0 | Table-setting scene/object pack (plates, cups, forks, spoons, napkins + target settings) | OQ-006 |
|   | OQ-008 | P0 | Adapt YOLO pipeline to tabletop objects: class, bbox/center, conf, stable IDs | OQ-007 |
|   | OQ-010 | P0 | Basic arm primitives: PICK, PLACE, MOVE, OPEN, CLOSE, ROTATE, PRESENT | OQ-006 |
|   | OQ-011 | P0 | Bimanual primitives: HANDOFF, STABILIZE, REGRASP, COOPERATIVE_ROTATE | OQ-010 |
|   | OQ-014 | P0 | Spin primitive: rotate → handoff → regrasp → rotate past one wrist's range | OQ-011, OQ-013 |
|   | OQ-016 | P0 | Execute basic table setting from initial scene | OQ-007, OQ-012 |
|   | OQ-017 | P0 | Execute concurrent table setting (independent simultaneous actions) | OQ-016 |
|   | OQ-022 | P0 | Intel hardware/runtime packaging — run through the required Intel path | OQ-016 |
|   | OQ-024 | P1 | Speechmatics realtime adapter into the goal parser | OQ-023 |
|   | OQ-027 | P1 | Qualcomm HF model intake → supported export path | — |
|   | OQ-028 | P1 | Qualcomm X Elite inference node returns structured detections | OQ-027 |
|   | OQ-030 | P1 | Arduino UNO Q capability node: ≥1 input read, ≥1 output/action | — |
|   | OQ-035 | P1 | Unified demo launcher: one command → Intel sim / Qualcomm hw / mock | OQ-034 |
|   | OQ-040 | P1 | Product description (~1 paragraph, no jargon) — with Bryan | OQ-036 |
|   | OQ-043 | P2 | Failure-injection demo control (move object / disable capability) | OQ-018 |
|   | OQ-045 | P2 | Fancy synchronized choreography without hurting completion time | OQ-017, OQ-026 |

## Start here

OQ-002 unblocks everyone's bootstrap. Then OQ-006 is the head of the critical
path — get two arms moving in MuJoCo before anything else.
