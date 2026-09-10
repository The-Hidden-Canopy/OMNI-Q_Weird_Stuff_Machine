"""OQ-029 MXFP2/2-bit CPU evaluation — YOLOv8n thermal model, weight-only RNE.

Four arms — fp32 (baseline), mxfp4, nvint2, mxfp2 — over the HIT-UAV
finetuned YOLOv8n, all executed as torch FP32 on CPU after a weight-only RNE
round-trip through ``integrations/qualcomm/lowbit`` (software dequantize; no
native 2-bit hardware is exercised here).

Tiers:
  1. footprint + latency (per arm: packed payload bytes vs fp32; forward
     latency at 1 thread and default threads; peak RSS)
  2. bandwidth (weights re-dequantized from PackedTensor right before each
     forward; dequant timed separately, latency reported with and without it)
  3. ternary multiply-free Linear prototype (mxfp2 only, nn.Linear,
     [512,1024] x [1024,1024]; prototype, not production)

Output agreement is reported as output-tensor cosine vs the fp32 arm on
fixed-seed synthetic thermal-like inputs, plus detection-count parity via the
ultralytics predict path at a fixed confidence threshold.

Accuracy (mAP50) is NOT re-measured here: the HIT-UAV val split is not on
this box; 0.825 mAP50 is cited as the published reference (see
evidence/benchmark_results/yolo_host_baseline_2026-09-10/).

Labels and protocol follow the host baseline bundle. Measurement pattern
derived from *The Hidden Canopy LLC* —
[`Semantically-Aware_ISR`](https://github.com/The-Hidden-Canopy/Semantically-Aware_ISR)
(`scripts/benchmark_edge.py`). Used with permission.

Portions derived from *The Hidden Canopy LLC* —
[`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2).
Used with permission.

Run (the Semantically-Aware_ISR venv has torch + ultralytics):

    E:/HiddenCanopy/Semantically-Aware_ISR/.venv/Scripts/python.exe \\
        integrations/qualcomm/scripts/eval_yolo_2bit_cpu.py
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

OMNIQ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OMNIQ_ROOT))

MODEL_DEFAULT = (
    "E:/HiddenCanopy/Semantically-Aware_ISR/evidence/models/"
    "hituav_yolov8n_finetuned_2026-08-26.pt"
)

ARMS = ("fp32", "mxfp4", "nvint2", "mxfp2")
FMTS = ("mxfp4", "nvint2", "mxfp2")


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except ImportError:  # pragma: no cover - psutil is present on this box
        return 0.0


def _quantiles(samples: list[float]) -> dict:
    s = sorted(samples)
    return {
        "mean": round(statistics.fmean(s), 3),
        "p50": round(s[len(s) // 2], 3),
        "p95": round(s[int(len(s) * 0.95)], 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
    }


# --------------------------------------------------------------------------- #
# Model loading / arm construction                                            #
# --------------------------------------------------------------------------- #
def load_model(model_path: str):
    import torch
    from ultralytics import YOLO

    yolo = YOLO(model_path)
    net = yolo.model
    net.eval()
    return yolo, net


def _weight_modules(net):
    import torch

    return [(name, mod) for name, mod in net.named_modules()
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear))]


def quantize_arm(net, fmt: str):
    """W-only RNE: quantize every Conv2d/Linear weight, return packed tensors
    keyed by parameter, and load the dequantized weights back into the model."""
    import numpy as np
    import torch

    from integrations.qualcomm.lowbit import dequantize_tensor, quantize_tensor

    packed = {}
    fp32_bytes = 0
    packed_bytes = 0
    with torch.no_grad():
        for name, mod in _weight_modules(net):
            w = mod.weight.detach().clone()
            fp32_bytes += w.numel() * 4
            pt = quantize_tensor(w.cpu().numpy(), fmt)
            packed[name] = (mod, pt)
            packed_bytes += pt.bytes_total
            mod.weight.copy_(torch.from_numpy(
                np.ascontiguousarray(dequantize_tensor(pt))).reshape(w.shape))
    return packed, fp32_bytes, packed_bytes


def restore_arm(packed) -> float:
    """Re-dequantize every packed weight and load it back; returns ms."""
    import time as _time

    import numpy as np
    import torch

    from integrations.qualcomm.lowbit import dequantize_tensor

    t0 = _time.perf_counter()
    with torch.no_grad():
        for mod, pt in packed.values():
            mod.weight.copy_(torch.from_numpy(
                np.ascontiguousarray(dequantize_tensor(pt))).reshape(pt.shape))
    return (_time.perf_counter() - t0) * 1000.0


# --------------------------------------------------------------------------- #
# Measurements                                                                #
# --------------------------------------------------------------------------- #
def latency(net, x, warmup: int, timed: int) -> dict:
    import torch

    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        samples = []
        for _ in range(timed):
            t0 = time.perf_counter()
            net(x)
            samples.append((time.perf_counter() - t0) * 1000.0)
    return _quantiles(samples)


def forward_output(net, x):
    import torch

    with torch.no_grad():
        out = net(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.detach().float().reshape(-1)


def ternary_prototype() -> dict:
    """Tier 3: multiply-free Linear via ternary codes {+1, 0, -1} (mxfp2).

    y[b, o] = sum_k scale[k] * (sum_{i in block k} x[b, i] * sign(W[o, i]))
    computed as an einsum against the 0/+-1 sign payload with the UE8M0 block
    scale applied once per 32-input block -- the shape a ternary accelerator
    would execute (adds for the payload, one multiply per block). Compared
    against the fp32 matmul with the same dequantized ternary weights and the
    pre-quantization fp32 weight.
    """
    import numpy as np
    import torch

    from integrations.qualcomm.lowbit import dequantize_tensor, quantize_tensor
    from integrations.qualcomm.lowbit.vendor.mxfp_scales import safe_scale

    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    x_np = rng.standard_normal((512, 1024), dtype=np.float32)
    w_np = rng.standard_normal((1024, 1024), dtype=np.float32)

    pt = quantize_tensor(w_np, "mxfp2")
    wq = torch.from_numpy(np.ascontiguousarray(dequantize_tensor(pt)))

    block = 32
    nblocks = 1024 // block
    sign_np = np.sign(wq.numpy()).astype(np.float32)  # +-1/0 ternary payload
    sign = torch.from_numpy(sign_np.reshape(1024, nblocks, block))
    scales = torch.from_numpy(
        safe_scale(np.frombuffer(pt.scales, dtype=np.uint8)).astype(np.float32))

    x = torch.from_numpy(x_np)
    with torch.no_grad():
        t0 = time.perf_counter()
        partial = torch.einsum("bk,okn->bok", x.reshape(512, nblocks, block), sign)
        y_proto = (partial * scales).sum(-1)
        t_proto = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        y_fp32wq = x @ wq.T
        t_fp32wq = (time.perf_counter() - t0) * 1000.0

        y_ref = x @ torch.from_numpy(w_np).T

    def _cos(a, b):
        a = a.reshape(-1).double()
        b = b.reshape(-1).double()
        return float(torch.dot(a, b) / (a.norm() * b.norm()))

    return {
        "label": "prototype, not production",
        "case": "[512,1024] x [1024,1024] torch.nn.Linear, mxfp2 ternary payload",
        "max_abs_diff_vs_fp32_matmul_same_weights": round(
            float((y_proto - y_fp32wq).abs().max()), 6),
        "cosine_vs_fp32_matmul_same_weights": _cos(y_proto, y_fp32wq),
        "max_abs_diff_vs_prequant_fp32": round(
            float((y_proto - y_ref).abs().max()), 6),
        "cosine_vs_prequant_fp32": _cos(y_proto, y_ref),
        "ms_prototype_binary_mask": round(t_proto, 3),
        "ms_fp32_matmul_same_weights": round(t_fp32wq, 3),
        "weight_cosine": round(pt.stats["weight_cosine"], 6),
    }


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--n-inputs", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence threshold for the detection-count parity row")
    ap.add_argument("--out-root", type=Path,
                    default=OMNIQ_ROOT / "evidence" / "benchmark_results")
    args = ap.parse_args()

    import numpy as np
    import torch

    import integrations.qualcomm.lowbit as lowbit
    from integrations.qualcomm.lowbit import lowbit_formats

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stamp = time.strftime("%Y%m%d")
    out_dir = args.out_root / f"yolo_2bit_cpu_{stamp}"
    if out_dir.exists():
        out_dir = args.out_root / f"yolo_2bit_cpu_{stamp}_{time.strftime('%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    default_threads = torch.get_num_threads()
    results = {
        "harness": "integrations/qualcomm/scripts/eval_yolo_2bit_cpu.py",
        "question": ("What do weight-only RNE MXFP4 / NVINT2 / MXFP2 masters do to "
                     "the HIT-UAV YOLOv8n's CPU forward latency, memory footprint and "
                     "output agreement vs the fp32 baseline?"),
        "created_utc": started,
        "attribution": ("Portions derived from *The Hidden Canopy LLC* - "
                        "[IDA-TRAIN-V2](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). "
                        "Used with permission."),
        "labels": {
            "execution": ("software dequantize; no native 2-bit hardware — weights are "
                          "restored to fp32 and executed by ordinary torch CPU kernels"),
            "inputs": ("fixed-seed synthetic thermal-like inputs (rng.normal 640x640 "
                       "grayscale broadcast to 3 channels); LATENCY + agreement probe "
                       "only, not an accuracy claim"),
            "accuracy": ("mAP50 blocked: HIT-UAV val set not on disk; 0.825 cited as "
                         "published reference (yolo_host_baseline_2026-09-10)"),
            "standing_note": ("a format is a format - all quantized arms are reported "
                              "symmetrically with measurements only"),
            "tier3": "ternary multiply-free Linear is a prototype, not production",
        },
        "environment": {
            "interpreter": sys.executable,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or platform.uname().processor,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_default_threads": default_threads,
            "cpu_count": __import__("os").cpu_count(),
        },
        "model": {
            "path": args.model,
            "loader": "ultralytics YOLO",
            "protocol": {
                "n_inputs": args.n_inputs,
                "warmup": args.warmup,
                "timed": args.timed,
                "seed": args.seed,
                "conf_for_count_parity": args.conf,
                "input": "fixed-seed rng.normal, 640x640 grayscale -> 3ch",
            },
        },
        "codec": {
            "module": "integrations/qualcomm.lowbit",
            "formats": {f: {"block_size": lowbit.BLOCK_SIZES[f],
                            "code_bits": lowbit_formats.CODE_BITS[f]}
                        for f in lowbit.FORMATS},
        },
        "tiers": {},
    }

    print(f"[eval] loading {args.model}", flush=True)
    yolo, net = load_model(args.model)
    net.to("cpu")

    rng = np.random.default_rng(args.seed)
    base = rng.standard_normal((args.n_inputs, 640, 640), dtype=np.float32)
    inputs = np.repeat(base[:, None, :, :], 3, axis=1)  # grayscale -> 3ch
    x_lat = torch.from_numpy(inputs[:1].copy())

    # ---- fp32 baseline reference outputs + weights snapshot ----------------
    ref_weights = {name: mod.weight.detach().clone()
                   for name, mod in _weight_modules(net)}
    print("[eval] fp32 reference outputs ...", flush=True)
    ref_outputs = [forward_output(net, torch.from_numpy(inputs[i:i + 1].copy()))
                   for i in range(args.n_inputs)]

    arms = {}
    packed_by_arm = {}
    for arm in ARMS:
        print(f"[eval] arm {arm} ...", flush=True)
        if arm == "fp32":
            with torch.no_grad():
                for name, mod in _weight_modules(net):
                    mod.weight.copy_(ref_weights[name])
            packed, fp32_bytes, packed_bytes = {}, 0, 0
            for name, mod in _weight_modules(net):
                fp32_bytes += mod.weight.numel() * 4
        else:
            packed, fp32_bytes, packed_bytes = quantize_arm(net, arm)
        packed_by_arm[arm] = packed

        cosines = []
        for i in range(args.n_inputs):
            out = forward_output(net, torch.from_numpy(inputs[i:i + 1].copy()))
            a = out.double()
            b = ref_outputs[i].double()
            cosines.append(float(torch.dot(a, b) / (a.norm() * b.norm())))
        cos = np.asarray(cosines)

        arm_entry = {
            "fp32_weight_bytes": fp32_bytes,
            "packed_bytes": packed_bytes,
            "packed_ratio": round(packed_bytes / fp32_bytes, 4) if fp32_bytes else None,
            "output_cosine_vs_fp32": {
                "mean": round(float(cos.mean()), 6),
                "std": round(float(cos.std()), 6),
                "min": round(float(cos.min()), 6),
            },
            "rss_peak_mb": round(_rss_mb(), 1),
        }
        if arm != "fp32":
            wcos = [pt.stats["weight_cosine"] for _, pt in packed.values()]
            arm_entry["weight_cosine_mean"] = round(float(np.mean(wcos)), 6)
            arm_entry["weight_cosine_min"] = round(float(np.min(wcos)), 6)

        # ---- Tier 1 latency: 1 thread and default threads -------------------
        tier1 = {}
        for label, threads in (("threads_1", 1), ("threads_default", default_threads)):
            torch.set_num_threads(threads)
            tier1[label] = latency(net, x_lat, args.warmup, args.timed)
        arm_entry["tier1_latency_ms"] = tier1
        arms[arm] = arm_entry

        # ---- Tier 2 bandwidth: re-dequantize right before each forward ------
        if arm != "fp32":
            tier2 = {}
            for label, threads in (("threads_1", 1), ("threads_default", default_threads)):
                torch.set_num_threads(threads)
                with torch.no_grad():
                    for _ in range(args.warmup):
                        restore_arm(packed)
                        net(x_lat)
                    combined, deq_only = [], []
                    for _ in range(args.timed):
                        t0 = time.perf_counter()
                        deq_ms = restore_arm(packed)
                        net(x_lat)
                        combined.append((time.perf_counter() - t0) * 1000.0)
                        deq_only.append(deq_ms)
                fwd_only = tier1[label]["mean"]
                tier2[label] = {
                    "restore_plus_forward_ms": _quantiles(combined),
                    "forward_only_ms": tier1[label],
                    "dequant_ms_mean": round(statistics.fmean(deq_only), 3),
                    "inferred_included_ms_mean": round(
                        statistics.fmean(combined), 3),
                    "excluded_forward_ms_mean": fwd_only,
                }
            arm_entry["tier2_bandwidth"] = tier2

    torch.set_num_threads(default_threads)

    # ---- decode sanity: detection-count parity via predict -----------------
    print("[eval] detection-count parity via predict ...", flush=True)
    detect_counts = {}
    try:
        for arm in ARMS:
            if arm == "fp32":
                with torch.no_grad():
                    for name, mod in _weight_modules(net):
                        mod.weight.copy_(ref_weights[name])
            else:
                restore_arm(packed_by_arm[arm])
            counts = []
            for i in range(args.n_inputs):
                r = yolo.predict(inputs[i].transpose(1, 2, 0), verbose=False,
                                 conf=args.conf, device="cpu", imgsz=640)
                counts.append(int(len(r[0].boxes)))
            detect_counts[arm] = counts
        parity = {}
        for arm in FMTS:
            same = sum(1 for a, b in zip(detect_counts[arm], detect_counts["fp32"]) if a == b)
            parity[arm] = {
                "count_agreement_rate": round(same / args.n_inputs, 4),
                "mean_count_fp32": round(statistics.fmean(detect_counts["fp32"]), 3),
                "mean_count_arm": round(statistics.fmean(detect_counts[arm]), 3),
            }
        results["decode_sanity"] = {
            "method": f"ultralytics predict path, conf={args.conf}, detection count per input",
            "parity_vs_fp32": parity,
        }
    except Exception as exc:  # keep receipt honest rather than silent
        results["decode_sanity"] = {
            "method": "ultralytics predict path",
            "status": f"failed: {type(exc).__name__}: {exc}; output cosine above stands",
        }

    # ---- Tier 3 ternary prototype ------------------------------------------
    print("[eval] tier 3 ternary Linear prototype ...", flush=True)
    results["tiers"]["tier3_ternary_linear_prototype"] = ternary_prototype()

    results["tiers"]["tier1_footprint_latency"] = arms
    results["map50"] = ("blocked: HIT-UAV val set not on disk; 0.825 cited as "
                        "reference (evidence/benchmark_results/"
                        "yolo_host_baseline_2026-09-10)")
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    results_path = out_dir / "results.json"
    results_path.write_text(payload, encoding="utf-8")
    print(f"[eval] wrote {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
