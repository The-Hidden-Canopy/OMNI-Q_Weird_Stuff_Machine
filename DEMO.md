# Demo

> One nasty, visible, reproducible behavior — and a repo that proves it wasn't
> smoke and mirrors.

## The behavior

> "Inspect this workspace, identify the misplaced component, fix it, and verify
> the result."

The system visibly runs:

```
Speechmatics        → natural-language goal
Qualcomm Snapdragon → YOLO / local perception → structured world state
OMNI Q              → execution graph → task decomposition → device / capability assignment
Intel robotics      → physical action
Qualcomm vision     → verification → success / replan
```

Then, on camera, one of these happens and the audience watches Omni Q react:

| State | Trigger | What Omni Q does |
| --- | --- | --- |
| NORMAL | goal given | builds graph, executes |
| CONSTRAINT CHANGE | *"Don't touch the red object."* / *"Keep inference on-device."* | modifies / re-places the graph |
| FAILURE / WORLD CHANGE | object moved, capability disabled, first attempt fails | replans and finishes anyway |

## Run

```
./demo/run_demo.sh
```

_TODO: prerequisites, hardware/sim setup, expected output, screen recording._

## Presentation timing (≤ 5 min)

- 0:00–0:30  What Omni Q is
- 0:30–1:00  Why fixed AI workflows suck
- 1:00–3:30  Live demo behavior
- 3:30–4:20  Qualcomm / Intel / Speechmatics integration
- 4:20–5:00  Business value + why this is different

## Screen layout

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
