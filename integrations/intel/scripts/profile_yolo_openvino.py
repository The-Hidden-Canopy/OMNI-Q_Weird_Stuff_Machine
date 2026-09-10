"""OQ-029/OpenVINO host benchmark -- latency/memory for the HIT-UAV YOLOv8n
detector running through OpenVINO on real Intel hardware (CPU / iGPU / NPU,
whichever ``mujoco``-style ``Core().available_devices`` reports on this box).

Same measurement protocol as
``integrations/qualcomm/scripts/profile_yolo_host.py`` (the retained
ONNX Runtime CPU baseline this compares against), so results are directly
comparable, not just parallel numbers:

- fixed-seed synthetic 640x640 input (LATENCY ONLY -- not an accuracy claim;
  the HIT-UAV val split is not local, accuracy is re-measured where the data
  lives, per the published artifact's 0.825 mAP50 reference)
- batch 1, warmup 50, measured 200 iterations
- wall-clock per-iteration mean/p50/p95, process RSS delta

Requires an OpenVINO IR export of the published thermal YOLO
(``KissTheHabit/yolov8n-hituav-thermal-finetune``); produce one with:

    .venv/Scripts/python -c "
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    pt = hf_hub_download('KissTheHabit/yolov8n-hituav-thermal-finetune', 'yolov8n_hituav_thermal.pt')
    YOLO(pt).export(format='openvino', imgsz=640)
    "

Usage:

    .venv/Scripts/python integrations/intel/scripts/profile_yolo_openvino.py \\
        --model path/to/yolov8n_hituav_thermal_openvino_model/yolov8n_hituav_thermal.xml \\
        --device CPU
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path


def benchmark(model_path: Path, device: str, warmup: int, iters: int, seed: int) -> dict:
    import numpy as np
    import openvino as ov
    import psutil

    core = ov.Core()
    available = core.available_devices
    device_names = {d: core.get_property(d, "FULL_DEVICE_NAME") for d in available}
    if device not in available:
        raise SystemExit(f"device {device!r} not in available devices {available}")

    model = core.read_model(model_path)
    compiled = core.compile_model(model, device)
    input_port = compiled.input(0)
    shape = list(input_port.shape)
    shape = [1 if isinstance(d, str) or d < 0 else d for d in shape]

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape, dtype=np.float32)

    request = compiled.create_infer_request()
    process = psutil.Process()
    rss_before_mb = process.memory_info().rss / (1024 ** 2)

    for _ in range(warmup):
        request.infer({input_port: x})
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        request.infer({input_port: x})
        samples.append((time.perf_counter() - t0) * 1000.0)
    rss_peak_mb = process.memory_info().rss / (1024 ** 2)

    samples.sort()
    return {
        "variant": f"openvino-{device.lower()}-fp32",
        "model": model_path.name,
        "device": device,
        "device_full_name": device_names[device],
        "available_devices": {d: device_names[d] for d in available},
        "input_shape": shape,
        "warmup": warmup,
        "iterations": iters,
        "seed": seed,
        "latency_ms": {
            "mean": round(statistics.fmean(samples), 3),
            "p50": round(samples[len(samples) // 2], 3),
            "p95": round(samples[int(len(samples) * 0.95)], 3),
            "min": round(samples[0], 3),
            "max": round(samples[-1], 3),
        },
        "memory_rss_mb": {
            "before": round(rss_before_mb, 1),
            "after_peak": round(rss_peak_mb, 1),
            "delta": round(rss_peak_mb - rss_before_mb, 1),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, type=Path, help="OpenVINO IR .xml path")
    p.add_argument("--device", default="CPU", help="CPU, GPU.0, GPU.1, NPU, ... (see Core().available_devices)")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=None,
                   help="write the result JSON here (default: stdout)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = benchmark(args.model, args.device, args.warmup, args.iters, args.seed)
    result["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["python"] = sys.version.split()[0]
    result["platform"] = platform.platform()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"[profile] wrote {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
