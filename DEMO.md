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

Mock mode works today — no hardware, no third-party deps (Python 3.10+):

```
./demo/run_demo.sh                 # or:  PYTHONPATH=src python -m omni_q.demo
```

Prints all three states end-to-end: NORMAL resolves in one graph;
CONSTRAINT CHANGE recompiles onto the right arm when the left goes offline;
WORLD CHANGE replans after an object is knocked away mid-run. Each run emits a
receipt with metrics and SHA-256 hashes of inputs / plan / actions.

Event stream for the UI (OQ-005):

```
PYTHONPATH=src python -m omni_q.server      # GET /events (SSE), POST /run
```

_TODO: hardware/sim setup (OQ-006), real perception (OQ-008), screen recording._

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
