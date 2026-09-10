# Qualcomm integration (primary track)

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
