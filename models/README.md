# Models

## Perception — thermal YOLO

[`KissTheHabit/yolov8n-hituav-thermal-finetune`](https://huggingface.co/KissTheHabit/yolov8n-hituav-thermal-finetune)

- YOLOv8n fine-tuned on HIT-UAV (thermal UAV imagery, 5 classes).
- Held-out test: **0.825 mAP50**, **0.538 mAP50-95**.
- Already published — the Qualcomm perception node is our own vision stack, not a
  stock demo model.

**Caveat:** these weights know thermal UAV classes, not tabletop/workcell objects.
For competition objects, either adapt the training/export pipeline to the target
vocabulary or swap in an open-vocabulary detector (YOLO-WORLD, also on AI Hub
across Snapdragon / Dragonwing) fed a dynamic object list from Omni Q.

### Export / deploy path

Object detection does **not** go through GenieX directly:

```
HF YOLO → Qualcomm AI Hub / QAIRT → Snapdragon X Elite → live detection
```

_TODO: confirm ONNX / QNN / QAIRT export path and any X Elite architecture
friction._

## Reasoning — GenieX local model

GenieX carries the on-device reasoning / VLM node (LLM/VLM runtime). Model TBD
from Hugging Face or Qualcomm AI Hub — must be small enough for GenieX to carry.

## Datasets

Sourcing plan for both the YOLO detector (OQ-008) and the Omni policy /
planner — scoped to the one use case (bimanual dual-SO-101 table setting):
[`../docs/datasets.md`](../docs/datasets.md).

TL;DR (2 h to deadline): the detector is a **head-swap + fine-tune of our own
`KissTheHabit/yolov8n-hituav-thermal-finetune`** on labelled frames rendered
straight from the OQ-006/007 MuJoCo scene, exported to OpenVINO IR + QAIRT.
Omni trains nothing now — it runs on `RulePlanner` + `ScheduledPlanner` +
`RuntimeMutator` + the real MuJoCo path. Everything else is a post-deadline
appendix.

## Related owned assets

IDA model family (IDA_AI ~966M, IDA_MoE, IDA_Swift/_Native), IDA-TRAIN-V2
Nemotron 4.19B, MLPerf/SQuAD native-training artifacts. See
[`../docs/strategy-notes.md`](../docs/strategy-notes.md) for the full inventory —
most are for building Omni Q, not for the on-device hackathon artifact.
