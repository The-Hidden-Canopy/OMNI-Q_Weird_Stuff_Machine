# Qualcomm integration (bonus, not an entry)

We only qualify to enter the **Intel online** track — see
[`integrations/intel/README.md`](../intel/README.md) and the top-level
[`README.md`](../../README.md). This Qualcomm work is exploratory/bonus, not
part of what we're actually submitting.

**Challenge:** use GenieX, select and deploy a model from Hugging Face or
Qualcomm AI Hub, then bring it to life through on-device AI, hardware
integration, and seamless device-to-device collaboration.

**Hardware provided:** Snapdragon X Elite platform + Arduino UNO Q.

## Role in the loop

Two Qualcomm technologies, two distinct jobs:

| Tech | Job |
| --- | --- |
| AI Hub / QAIRT | perception acceleration — runs the thermal YOLO on the Hexagon NPU |
| GenieX | on-device reasoning / VLM node |

```
camera → YOLO on Qualcomm (AI Hub/QAIRT) → structured scene state
       → GenieX local model → OMNI Q task decision → Arduino UNO Q (device collaboration)
```

## Device split as capability nodes

```
SNAPDRAGON X ELITE          ARDUINO UNO Q
├── YOLO / vision           ├── sensors
├── Omni Q reasoning        ├── actuators
├── local model            ├── physical I/O
└── task routing           └── realtime device state
```

Device-to-device collaboration is intrinsic: *"Watch this workspace. When the
blue object enters region B, inspect it. If it matches the target, trigger the
actuator. If the connection disappears, continue locally."*

## TODO

- [ ] Export thermal YOLO → QAIRT / QNN, run on X Elite NPU
- [ ] Pick + deploy GenieX model (HF or AI Hub)
- [ ] Snapdragon ↔ Arduino transport for capability calls + device state
- [ ] Structured scene-state schema emitted by the perception node

## Low-bit CPU formats

The public low-bit surface is deliberately small and independently
releasable. It provides MXFP8 and MXFP4 tensor packing through
`integrations.qualcomm.lowbit`, with NumPy reference encode/decode behavior
and explicit cosine/error statistics.

Experimental or deployment-specific 2-bit formats are not shipped here. They
are rejected by the public codec rather than represented by incomplete vendor
payloads or copied runtime code. A future format must arrive with its own
license, implementation, reference tests, and hardware evidence before it is
added to this public surface.

Run the public codec tests with:

```bash
PYTHONPATH=src python -m pytest tests/test_lowbit.py -q
```
