"""Render the OpenVINO benchmark receipt as one figure (latency, throughput, mAP).

    .venv/Scripts/python integrations/intel/scripts/plot_openvino_benchmark.py \
        evidence/benchmark_results/openvino_table_detector_2026-09-15/receipt.json --out openvino_benchmark.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = json.loads(args.receipt.read_text())
    precs = [p for p in ("fp32", "fp16", "int8") if p in r["quality"] or any(x["precision"] == p for x in r["rows"])]
    devices = [d for d in r["devices"] if any(x.get("device") == d for x in r["rows"])]
    lat = {(x["precision"], x["device"]): x.get("latency_ms_mean") for x in r["rows"] if x["hint"] == "LATENCY" and "error" not in x}
    thr = {(x["precision"], x["device"]): x.get("throughput_fps") for x in r["rows"] if x["hint"] == "THROUGHPUT" and "error" not in x}
    ir = {x["precision"]: x["ir_size_mb"] for x in r["rows"]}
    host = r.get("host", "")
    colors = {"fp32": "#9aa5b1", "fp16": "#5aaaf0", "int8": "#f6b352"}
    hatch = ["", "//", "..", "xx"]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle(f"OpenVINO {r['openvino']} on {host}  --  scene-trained YOLOv8n table detector, real 640x640 frames\n"
                 f"INT8 = NNCF post-training quantisation calibrated on this scene; IR {ir.get('fp32', '?')} MB -> {ir.get('int8', '?')} MB", fontsize=11)
    w = 0.8 / max(1, len(devices))
    for ax, table, title in ((axes[0], lat, "latency, ms (batch 1, lower is better)"), (axes[1], thr, "throughput, FPS (async queue, higher is better)")):
        for i, p in enumerate(precs):
            for j, d in enumerate(devices):
                v = table.get((p, d))
                if v is None:
                    continue
                b = ax.bar(i + (j - (len(devices) - 1) / 2) * w, v, w * 0.95, color=colors[p], hatch=hatch[j % len(hatch)],
                           edgecolor="black", alpha=0.9 if j == 0 else 0.7)
                if j == 0:
                    ax.bar_label(b, fmt="%.1f" if table is lat else "%.0f", fontsize=9)
        ax.set_xticks(range(len(precs)), [p.upper() for p in precs]); ax.set_title(title)
    ax = axes[2]
    for i, p in enumerate(precs):
        q = r["quality"].get(p)
        if q:
            b = ax.bar(i, q["map50"], 0.6, color=colors[p]); ax.bar_label(b, fmt="%.3f", fontsize=9)
    ax.set_ylim(0.9, 1.0); ax.set_xticks(range(len(precs)), [p.upper() for p in precs])
    ax.set_title(f"mAP50 on the {r.get('val_images', '?')}-image val split (quality preserved)")
    dev_names = "; ".join(f"{d} = {r['devices'][d]}" for d in devices)
    fig.text(0.5, 0.005, f"devices: {dev_names}  (first device solid, others hatched)", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    out = args.out or args.receipt.with_name("benchmark.png")
    fig.savefig(out, dpi=120)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
