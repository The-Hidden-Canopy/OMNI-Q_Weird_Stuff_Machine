# YOLO 2-bit mAP50 — weight-only RNE arms on table_yolo val (2026-09-10)

Accuracy complement to `../yolo_2bit_cpu_20260910/` (the footprint/latency
bundle, cited). Same arms, same codec, same honest-labeling rules; **latency
is NOT re-measured here** and the latency bundle's **mAP50 row there stays
"blocked" for the HIT-UAV model** — this bundle measures accuracy on the
model that has its val split on disk.

## Question

What does the weight-only RNE MXFP4 / NVINT2 / MXFP2 round-trip (software
dequantize, fp32 torch CPU kernels) do to the table_yolo YOLOv8n's val mAP50
vs the fp32 baseline?

## Model + data

- Weights: `E:\HiddenCanopy\OMNI-Q_Weird_Stuff_Machine\runs\detect\runs\table_yolo\ft\weights\best.pt` (ultralytics YOLOv8n, 7-class
  fine-tune, 4 CPU epochs — the baseline is weak, ~0.058 mAP50, and that is
  the honest fp32 arm below; the data decides).
- Data: `E:\HiddenCanopy\OMNI-Q_Weird_Stuff_Machine\data\table_yolo\data.yaml` — val split,
  276 images / 1062
  instances, classes plate, cup, fork, spoon, knife, napkin, drawer.
- Protocol: per arm, fresh `ultralytics.YOLO` load; fmt arms quantize every
  Conv2d/Linear weight with `integrations/qualcomm.lowbit` and load the
  dequantized weights back; `yolo.val(split="val", imgsz=640, device="cpu")`
  on the restored model. Execution = software dequantize; no native 2-bit
  hardware is exercised.

## mAP50 per arm (table_yolo val)

| Arm | Packed bytes | Ratio vs fp32 | Weight cosine (mean) | mAP50 | mAP50-95 | mAP50 delta vs fp32 |
|---|---|---|---|---|---|---|
| fp32 | 12,006,400 | 1.0000 | - | 0.0576 | 0.0320 | — |
| mxfp4 | 1,594,601 | 0.1328 | 0.9921 | 0.0062 | 0.0020 | -0.0514 |
| nvint2 | 938,256 | 0.0781 | 0.9119 | 0.0000 | 0.0000 | -0.0576 |
| mxfp2 | 844,201 | 0.0703 | 0.6775 | 0.0000 | 0.0000 | -0.0576 |

Per-class AP50 (val):

| Class | fp32 | mxfp4 | nvint2 | mxfp2 |
|---|---|---|---|---|
| plate | 0.0538 | 0.0124 | 0.0000 | 0.0000 |
| cup | 0.0837 | 0.0117 | 0.0000 | 0.0000 |
| fork | 0.0067 | 0.0017 | 0.0000 | 0.0000 |
| spoon | 0.0062 | 0.0001 | 0.0000 | 0.0000 |
| knife | 0.0014 | 0.0006 | 0.0000 | 0.0000 |
| napkin | 0.0040 | 0.0003 | 0.0000 | 0.0000 |
| drawer | 0.2477 | 0.0167 | 0.0000 | 0.0000 |

## Honest labels

- **Software dequantize; no native 2-bit hardware.** Every quantized arm
  executes the dequantized weights with ordinary fp32 torch CPU kernels; the
  mAP deltas below are pure weight-quantization accuracy deltas.
- **Latency NOT re-measured here** — see `evidence/benchmark_results/yolo_2bit_cpu_20260910/` for footprint,
  latency, output cosine, and detection-count parity.
- The fp32 baseline itself is weak (~0.058 mAP50 after 4 CPU epochs); read
  quantized-arm numbers against that baseline, not against an external
  reference.
- **Standing note**: a format is a format — all quantized arms are reported
  symmetrically with measurements only.

## Files + re-run

- `results.json` — full machine-readable receipt (stats, environment, labels).
- Harness: `integrations/qualcomm/scripts/eval_yolo_2bit_map.py`
- Codec + provenance: `integrations/qualcomm/lowbit/`
- Bundle format: `src/omni_q/evidence_bundle.py` (vendored from
  open_world_model_harness); validated with `validate_evidence_bundle`.

```bash
YOLO_DATASETS_DIR=E:/HiddenCanopy/OMNI-Q_Weird_Stuff_Machine/data \
    ./.venv/Scripts/python.exe \
    integrations/qualcomm/scripts/eval_yolo_2bit_map.py
```
