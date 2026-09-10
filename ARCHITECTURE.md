# Architecture

> Draft. Fills in as the stack lands. Rationale: [`docs/strategy-notes.md`](docs/strategy-notes.md).

## Core idea

Function is separate from placement. A capability describes *what* can be done; a
placement decision says *where* it runs. Omni Q holds both and can change either
without rewriting the other.

```
objective (natural language)
        ↓
world state  ← perception nodes
        ↓
capability graph        ← decompose objective into capability calls
        ↓
placement                ← assign each node to a device/runtime under constraints
        ↓
execution                ← run nodes, stream state
        ↓
verification             ← check result → success or replan
```

## Capability nodes

Each device, sensor, and model is registered as a node with a typed interface,
not called directly by hardcoded logic. Examples:

```
detect_object   segment   VLM   camera_1
VISION   LEFT_ARM   RIGHT_ARM   GRASP   MOVE   VOICE   VERIFY
```

Removing a node (`LEFT_ARM ❌`) forces recompilation of any graph that depended
on it onto the remaining nodes.

## Placement targets

```
SNAPDRAGON X ELITE          ARDUINO UNO Q            HOST
├── NPU  (perception)       ├── sensors              ├── GPU
├── GPU  (reasoning)        ├── actuators            └── CPU
├── CPU                     ├── physical I/O
└── task routing            └── realtime device state
```

Constraints that trigger re-placement: *"keep everything local"*, *"move vision
to GPU"*, *"kill the cloud connection"*.

## Perception substrate (Qualcomm)

Omni Q does not ingest raw video unless necessary. The perception node emits
compact structured state:

```json
{
  "frame": 8814,
  "objects": [
    {"id": "part_7", "class": "connector", "conf": 0.96, "bbox": [/* ... */]}
  ],
  "workspace_clear": true
}
```

Deploy path (object detection does not go through GenieX directly):

```
HF YOLO → Qualcomm AI Hub / QAIRT → Snapdragon X Elite → live detection → OMNI Q
```

GenieX carries the reasoning/VLM node:

```
camera → YOLO on Qualcomm → scene state → GenieX local model → OMNI Q decision → Arduino / device collaboration
```

## Observable states (for the demo)

1. **NORMAL** — graph builds and executes.
2. **CONSTRAINT CHANGE** — spoken constraint mutates the graph.
3. **FAILURE / WORLD CHANGE** — capability lost or world moved → replan.

## Open questions

- Graph representation and scheduler (library vs. hand-rolled).
- Node interface schema and registration format.
- Verification signals per capability type.
- Export path specifics for the thermal YOLO on X Elite (ONNX / QNN / QAIRT).

## Intel online stack (confirmed)

Per Intel's published brief, the dual-arm MuJoCo track is pinned to a specific
toolchain rather than left open:

```
Simulation Engine   MuJoCo
Data Collection      LeRobot + OMPL (motion planning)
Model Training        local or cloud, unconstrained
Model Inference        Intel OpenVINO + OpenVINO Physical AI (Core Ultra Series 2/3)
```

See [`integrations/intel/README.md`](integrations/intel/README.md) for how this
maps onto the `LEFT_ARM`/`RIGHT_ARM`/`GRASP`/`MOVE`/`VERIFY` capability nodes.
