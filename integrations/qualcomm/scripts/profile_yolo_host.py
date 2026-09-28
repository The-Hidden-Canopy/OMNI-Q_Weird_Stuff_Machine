"""OQ-029 host baseline — latency/memory for the HIT-UAV YOLOv8n deployment variants.

Benchmarks the exported ONNX artifact (see
``evidence/benchmark_results/yolo_host_baseline_*/``) with ONNX Runtime CPU.
The Qualcomm X Elite variants (AI Hub / QAIRT) and Intel OpenVINO IR land on
their own nodes; this host baseline is the comparison point they get measured
against, so all variants share this script's measurement protocol:

- fixed-seed synthetic 640x640 input (LATENCY ONLY -- not an accuracy claim;
  the HIT-UAV val split is not local, accuracy is re-measured where the data
  lives, per the published artifact's 0.825 mAP50 reference)
- batch 1, warmup 50, measured 200 iterations
- wall-clock per-iteration mean/p50/p95, process RSS delta

Measurement pattern derived from *The Hidden Canopy LLC* —
[`Semantically-Aware_ISR`](https://github.com/The-Hidden-Canopy/Semantically-Aware_ISR)
(`scripts/benchmark_edge.py`). Used with permission.

    .venv/Scripts/python integrations/qualcomm/scripts/profile_yolo_host.py \
        --onnx evidence/benchmark_results/yolo_host_baseline_2026-09-10/models/hituav_yolov8n_finetuned_2026-08-26.onnx
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path


def benchmark(onnx_path: Path, warmup: int, iters: int, seed: int,
              threads: int | None) -> dict:
    import numpy as np
    import onnxruntime as ort
    import psutil

    providers = ort.get_available_providers()
    sess_options = ort.SessionOptions()
    if threads is not None:
        sess_options.intra_op_num_threads = threads
        sess_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options,
                                   providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    if isinstance(shape[0], str) or any(isinstance(d, str) for d in shape):
        shape = [1 if isinstance(d, str) else d for d in shape]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape, dtype=np.float32)

    rss_before_mb = psutil.Process().memory_info().rss / (1024 ** 2)
    for _ in range(warmup):
        session.run(None, {session.get_inputs()[0].name: x})
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, {session.get_inputs()[0].name: x})
        samples.append((time.perf_counter() - t0) * 1000.0)
    rss_peak_mb = psutil.Process().memory_info().rss / (1024 ** 2)

    samples.sort()
    return {
        "variant": "onnxruntime-cpu-fp32",
        "onnx": onnx_path.name,
        "input_shape": list(shape),
        "providers": providers,
        "warmup": warmup,
        "iterations": iters,
        "seed": seed,
        "threads": threads or "default",
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
    p.add_argument("--onnx", required=True, type=Path)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--output", type=Path, default=None,
                   help="write the result JSON here (default: stdout)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = benchmark(args.onnx, args.warmup, args.iters, args.seed, args.threads)
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
