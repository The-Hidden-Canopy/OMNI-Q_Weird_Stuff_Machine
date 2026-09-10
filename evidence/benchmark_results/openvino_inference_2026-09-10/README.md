# OpenVINO inference benchmark — first real Intel Core Ultra-adjacent run

First model in this repo actually exported to OpenVINO IR and run through
OpenVINO inference on real Intel hardware. Closes the gap flagged in
[`docs/oq-004-requirements-audit.md`](../../../docs/oq-004-requirements-audit.md)
("no model — perception or policy — has actually been exported to OpenVINO
IR or run through OpenVINO inference anywhere in the codebase yet") and is a
first pass at the brief's required "Intel Inference Benchmark Script"
deliverable.

## Scope — what this is and isn't

This benchmarks the **already-published thermal YOLO**
(`KissTheHabit/yolov8n-hituav-thermal-finetune`, HIT-UAV 5-class, 0.825 mAP50
held-out), **not** the fine-tuned 7-class table-setting detector — that
fine-tune (OQ-008, `perception/`) hasn't been run yet; it needs a large
real-image pull (Open Images/COCO/LVIS, 150-300k images) that doesn't fit a
short session. Using the already-trained model instead proves the actual
**export → optimize → benchmark pipeline** works end-to-end on real Intel
hardware, using real numbers. Swap the `.pt` when the table-setting
fine-tune lands; the export/benchmark commands below don't change.

Machine: this dev laptop's own **Intel Core 5 210H** (CPU + integrated
`Intel(R) Graphics` iGPU) — confirmed via `Core().available_devices` earlier
this session. Not a Core Ultra Series 2/3 part specifically (this machine
doesn't have one); same OpenVINO/Intel architecture family, but re-run on
the actual target hardware before submission per the brief's requirement.

**Latency only** — same caveat as the retained ONNX Runtime baseline this
compares against
([`../yolo_host_baseline_2026-09-10/README.md`](../yolo_host_baseline_2026-09-10/README.md)):
the HIT-UAV val split isn't local, so accuracy is not re-measured here.

## Protocol

Same as `integrations/qualcomm/scripts/profile_yolo_host.py` (the retained
ONNX Runtime CPU baseline), implemented for OpenVINO in
[`../../../integrations/intel/scripts/profile_yolo_openvino.py`](../../../integrations/intel/scripts/profile_yolo_openvino.py):
fixed-seed synthetic 640×640 input, batch 1, 50 warmup + 200 measured
iterations, wall-clock per-inference, process RSS delta.

## Results

| Variant | Device | Mean | p50 | p95 | Memory Δ |
| --- | --- | ---: | ---: | ---: | ---: |
| OpenVINO FP32 | Intel Core 5 210H (CPU) | 42.2 ms | 42.7 ms | 59.8 ms | +12.4 MB |
| OpenVINO FP32 | Intel iGPU (`GPU.0`) | 12.0 ms | 12.0 ms | 14.1 ms | +9.0 MB |
| OpenVINO INT8 (NNCF PTQ) | Intel Core 5 210H (CPU) | 10.1 ms | — | 14.3 ms | see JSON |
| OpenVINO INT8 (NNCF PTQ) | Intel iGPU (`GPU.0`) | 8.5 ms | — | 10.8 ms | see JSON |

Full stats: `openvino_cpu_fp32.json`, `openvino_igpu_fp32.json`,
`openvino_cpu_int8.json`, `openvino_igpu_int8.json`. Model hashes:
`models.sha256` (binaries not committed — regenerate below; same convention
as the ONNX baseline directory).

## Reading for the demo story

- **Device utilization**: the same FP32 model is **~3.5× faster on the
  Intel iGPU than the Intel CPU** on this one machine (12.0ms vs 42.2ms) —
  a real, measured argument for targeting `GPU.0` on Core Ultra hardware,
  which ships a materially stronger iGPU than this dev laptop's.
- **Precision/quantization**: NNCF post-training INT8 quantization (60
  synthetic calibration samples — a real quantization pass, not a
  configuration no-op) cuts CPU latency **~4.2×** versus CPU FP32 (10.1ms vs
  42.2ms) and iGPU latency further to 8.5ms — the fastest variant measured.
  Calibration was synthetic (matching the latency-only protocol above), so
  this is not an accuracy-preservation claim; re-quantize against real
  images before citing accuracy alongside these latency numbers.
- Compare directly against the retained ONNX Runtime CPU baseline (82.9ms
  mean, different machine — 6C/12T Intel + Quadro P5200) for the
  "OpenVINO vs. generic ONNX Runtime" half of the optimization story; this
  session's OpenVINO CPU FP32 number (42.2ms) already halves it, before INT8.

## Reproduce

```bash
pip install -e ".[dev,intel]"   # openvino, optimum-intel, nncf
pip install ultralytics huggingface-hub

# 1. pull the published weights and export
python -c "
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
pt = hf_hub_download('KissTheHabit/yolov8n-hituav-thermal-finetune', 'yolov8n_hituav_thermal.pt')
YOLO(pt).export(format='openvino', imgsz=640)
"

# 2. FP32 benchmark (repeat --device for CPU / GPU.0 / GPU.1 / NPU as available)
python integrations/intel/scripts/profile_yolo_openvino.py \
    --model <exported>/yolov8n_hituav_thermal_openvino_model/yolov8n_hituav_thermal.xml \
    --device CPU

# 3. INT8 (NNCF post-training quantization, synthetic calibration)
python -c "
import numpy as np, nncf, openvino as ov
core = ov.Core()
model = core.read_model('<exported>/yolov8n_hituav_thermal_openvino_model/yolov8n_hituav_thermal.xml')
rng = np.random.default_rng(7)
calib = nncf.Dataset([rng.standard_normal((1,3,640,640), dtype=np.float32) for _ in range(60)])
ov.save_model(nncf.quantize(model, calib, subset_size=60), 'int8/yolov8n_hituav_thermal_int8.xml')
"
python integrations/intel/scripts/profile_yolo_openvino.py --model int8/yolov8n_hituav_thermal_int8.xml --device CPU
```
