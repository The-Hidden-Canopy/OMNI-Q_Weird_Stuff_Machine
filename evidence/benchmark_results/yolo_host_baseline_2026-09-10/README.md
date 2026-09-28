# YOLO host baseline — OQ-029 comparison point (2026-09-10)

Measured by Gerron/Kimi. The Qualcomm X Elite variants (AI Hub / QAIRT) and
Intel OpenVINO IR get profiled on their own nodes with the same protocol
(`integrations/qualcomm/scripts/profile_yolo_host.py`); this bundle is the
**host baseline** they compare against.

## Model

`hituav_yolov8n_finetuned_2026-08-26.pt` → ONNX FP32 640×640
(`models/`, sha256 alongside). Published reference accuracy:
**0.825 mAP50 / 0.538 mAP50-95** (HIT-UAV held-out test, HF
`KissTheHabit/yolov8n-hituav-thermal-finetune`).

## Protocol

Batch 1, fixed-seed synthetic 640×640 input, 50 warmup + 200 measured
iterations, wall-clock per inference. **Latency only** — the HIT-UAV val split
is not local, so accuracy is NOT re-measured here; it is re-measured where the
data lives. Do not quote these numbers as accuracy evidence.

## Results (dev box: 6C/12T Intel, Quadro P5200 16 GB, Windows)

| Variant | mean | p50 | p95 | Memory |
|---|---|---|---|---|
| ONNX Runtime CPU FP32 | 82.9 ms | see JSON | 117.3 ms | +49.5 MB RSS |
| torch CUDA FP32 (P5200) | 20.4 ms | see JSON | 27.7 ms | 35.3 MB GPU peak |

Files: `onnxruntime_cpu_fp32.json`, `torch_cuda_fp32.json` (full stats),
`models/` (ONNX artifact + sha256).

## Reading for the demo story

- ~5× latency gap between this dev CPU and a 2018-era discrete GPU shows why
  the Snapdragon NPU placement matters; quote X Elite numbers beside these.
- The 35 MB GPU footprint of the FP32 model previews what the quantized
  variants must beat to earn their place.
