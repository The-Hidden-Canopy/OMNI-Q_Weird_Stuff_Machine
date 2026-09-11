"""OQ-029 MXFP2/2-bit CPU evaluation — accuracy complement (mAP50 on table_yolo val).

Real mAP50 measurement of the four weight-only RNE arms from
``eval_yolo_2bit_cpu.py`` — fp32 (baseline), mxfp4, nvint2, mxfp2 — on the
table_yolo 7-class tabletop val split (plate/cup/fork/spoon/knife/napkin/
drawer). Each arm is built exactly like the latency harness: fresh ultralytics
YOLO load per arm; fmt arms run a weights-only RNE round-trip through
``integrations/qualcomm/lowbit`` (quantize -> dequantize -> load back) so the
val pass executes the dequantized weights with ordinary fp32 torch CPU
kernels (software dequantize; no native 2-bit hardware is exercised).

This is the accuracy complement to the footprint/latency bundle
``evidence/benchmark_results/yolo_2bit_cpu_20260910/``: latency is NOT
re-measured here; output agreement (cosine) is NOT re-measured here. The two
bundles are meant to be read together.

Labels and protocol follow the host baseline and 2-bit CPU bundles.
Measurement pattern derived from *The Hidden Canopy LLC* —
[`Semantically-Aware_ISR`](https://github.com/The-Hidden-Canopy/Semantically-Aware_ISR)
(`scripts/benchmark_edge.py`). Used with permission.

Portions derived from *The Hidden Canopy LLC* —
[`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2).
Used with permission.

Portions derived from *The Hidden Canopy LLC* —
[`open_world_model_harness`](https://github.com/The-Hidden-Canopy/open_world_model_harness).
Used with permission. (Evidence bundle writer is vendored from there.)

Run (OMNI-Q venv: torch 2.14.0+cpu + ultralytics 8.4.146; set
YOLO_DATASETS_DIR so ultralytics resolves datasets inside this repo):

    YOLO_DATASETS_DIR=E:/HiddenCanopy/OMNI-Q_Weird_Stuff_Machine/data \\
        ./.venv/Scripts/python.exe \\
        integrations/qualcomm/scripts/eval_yolo_2bit_map.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

OMNIQ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OMNIQ_ROOT))
sys.path.insert(0, str(OMNIQ_ROOT / "src"))  # omni_q.evidence_bundle

MODEL_DEFAULT = str(
    OMNIQ_ROOT / "runs" / "detect" / "runs" / "table_yolo" / "ft" / "weights" / "best.pt"
)
DATA_DEFAULT = str(OMNIQ_ROOT / "data" / "table_yolo" / "data.yaml")

ARMS = ("fp32", "mxfp4", "nvint2", "mxfp2")
FMTS = ("mxfp4", "nvint2", "mxfp2")

CPU_BUNDLE_CITE = "evidence/benchmark_results/yolo_2bit_cpu_20260910/"


# --------------------------------------------------------------------------- #
# Model loading / arm construction (same pattern as eval_yolo_2bit_cpu.py)    #
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


# --------------------------------------------------------------------------- #
# Dataset bookkeeping                                                          #
# --------------------------------------------------------------------------- #
def count_val_images(data_yaml: str) -> int:
    import yaml

    with open(data_yaml, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    val_dir = Path(cfg["path"]) / cfg["val"]
    return sum(1 for p in val_dir.iterdir()
               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def count_val_instances(data_yaml: str) -> int:
    import yaml

    with open(data_yaml, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    labels_dir = Path(cfg["path"]) / "labels" / Path(cfg["val"]).name
    total = 0
    for p in labels_dir.glob("*.txt"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += 1
    return total


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def run_val(yolo, data_yaml: str, imgsz: int) -> dict:
    """ultralytics validate on the val split; returns mAP + per-class AP50."""
    metrics = yolo.val(data=data_yaml, split="val", imgsz=imgsz, device="cpu",
                       plots=False, verbose=False)
    results = metrics.results_dict
    per_class = {}
    for row in metrics.summary():
        per_class[str(row["Class"])] = {
            "mAP50": round(float(row["mAP50"]), 6),
            "mAP50-95": round(float(row["mAP50-95"]), 6),
            "instances": int(row["Instances"]),
        }
    return {
        "map50": round(float(results["metrics/mAP50(B)"]), 6),
        "map50_95": round(float(results["metrics/mAP50-95(B)"]), 6),
        "precision": round(float(results["metrics/precision(B)"]), 6),
        "recall": round(float(results["metrics/recall(B)"]), 6),
        "per_class": per_class,
    }


# --------------------------------------------------------------------------- #
# README                                                                       #
# --------------------------------------------------------------------------- #
def render_readme(results: dict, arms: dict, deltas: dict) -> str:
    fmt_rows = []
    for arm in ARMS:
        a = arms[arm]
        cos = "-" if arm == "fp32" else f"{a['weight_cosine_mean']:.4f}"
        ratio = f"{a['packed_ratio']:.4f}"
        if arm == "fp32":
            delta = "—"
        else:
            d = deltas[arm]["map50_delta"]
            delta = f"{d:+.4f}"
        fmt_rows.append(
            f"| {arm} | {a['packed_bytes']:,} | {ratio} | {cos} "
            f"| {a['map50']:.4f} | {a['map50_95']:.4f} | {delta} |"
        )
    table = "\n".join(fmt_rows)

    per_class_lines = []
    for cls in results["data"]["classes"]:
        cells = [cls]
        for arm in ARMS:
            row = arms[arm]["per_class"].get(cls)
            cells.append("-" if row is None else f"{row['mAP50']:.4f}")
        per_class_lines.append("| " + " | ".join(cells) + " |")
    per_class_table = "\n".join(per_class_lines)

    return f"""# YOLO 2-bit mAP50 — weight-only RNE arms on table_yolo val ({results['created_utc'][:10]})

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

- Weights: `{results['model']['path']}` (ultralytics YOLOv8n, 7-class
  fine-tune, 4 CPU epochs — the baseline is weak, ~0.058 mAP50, and that is
  the honest fp32 arm below; the data decides).
- Data: `{results['data']['yaml']}` — val split,
  {results['data']['val_images']} images / {results['data']['val_instances']}
  instances, classes {", ".join(results['data']['classes'])}.
- Protocol: per arm, fresh `ultralytics.YOLO` load; fmt arms quantize every
  Conv2d/Linear weight with `integrations/qualcomm.lowbit` and load the
  dequantized weights back; `yolo.val(split="val", imgsz=640, device="cpu")`
  on the restored model. Execution = software dequantize; no native 2-bit
  hardware is exercised.

## mAP50 per arm (table_yolo val)

| Arm | Packed bytes | Ratio vs fp32 | Weight cosine (mean) | mAP50 | mAP50-95 | mAP50 delta vs fp32 |
|---|---|---|---|---|---|---|
{table}

Per-class AP50 (val):

| Class | fp32 | mxfp4 | nvint2 | mxfp2 |
|---|---|---|---|---|
{per_class_table}

## Honest labels

- **Software dequantize; no native 2-bit hardware.** Every quantized arm
  executes the dequantized weights with ordinary fp32 torch CPU kernels; the
  mAP deltas below are pure weight-quantization accuracy deltas.
- **Latency NOT re-measured here** — see `{CPU_BUNDLE_CITE}` for footprint,
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
YOLO_DATASETS_DIR=E:/HiddenCanopy/OMNI-Q_Weird_Stuff_Machine/data \\
    ./.venv/Scripts/python.exe \\
    integrations/qualcomm/scripts/eval_yolo_2bit_map.py
```
"""


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--out-root", type=Path,
                    default=OMNIQ_ROOT / "evidence" / "benchmark_results")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    import numpy as np
    import torch

    import integrations.qualcomm.lowbit as lowbit
    from integrations.qualcomm.lowbit import lowbit_formats
    from omni_q.evidence_bundle import (
        EvidenceBundleWriter,
        validate_evidence_bundle,
    )

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stamp = time.strftime("%Y%m%d")

    import yaml as _yaml
    with open(args.data, encoding="utf-8") as fh:
        data_cfg = _yaml.safe_load(fh)
    classes = [str(v) for _, v in sorted(data_cfg["names"].items(), key=lambda kv: kv[0])]
    val_images = count_val_images(args.data)
    val_instances = count_val_instances(args.data)

    results = {
        "harness": "integrations/qualcomm/scripts/eval_yolo_2bit_map.py",
        "question": ("What does the weight-only RNE MXFP4 / NVINT2 / MXFP2 "
                     "round-trip do to the table_yolo YOLOv8n's val mAP50 vs "
                     "the fp32 baseline?"),
        "created_utc": started,
        "attribution": ("Portions derived from *The Hidden Canopy LLC* - "
                        "[IDA-TRAIN-V2](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2), "
                        "[Semantically-Aware_ISR](https://github.com/The-Hidden-Canopy/Semantically-Aware_ISR), "
                        "and [open_world_model_harness](https://github.com/The-Hidden-Canopy/open_world_model_harness). "
                        "Used with permission."),
        "labels": {
            "execution": ("software dequantize; no native 2-bit hardware — "
                          "quantized arms restore weights to fp32 via "
                          "integrations/qualcomm/lowbit and execute ordinary "
                          "torch CPU kernels"),
            "accuracy_complement": ("latency and output agreement are NOT "
                                    "re-measured here; see "
                                    f"{CPU_BUNDLE_CITE} (eval_yolo_2bit_cpu.py) "
                                    "for footprint/latency/cosine; the mAP50 "
                                    "row there refers to the HIT-UAV model "
                                    "and stays 'blocked'"),
            "standing_note": ("a format is a format - all quantized arms are "
                              "reported symmetrically with measurements only"),
            "baseline": ("fp32 arm is the same 4-CPU-epoch fine-tune measured "
                         "in training (val mAP50 ~0.058); the baseline is weak "
                         "and that is reported as-is"),
        },
        "environment": {
            "interpreter": sys.executable,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or platform.uname().processor,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cpu_count": __import__("os").cpu_count(),
        },
        "model": {"path": args.model, "loader": "ultralytics YOLO"},
        "data": {
            "yaml": args.data,
            "val_images": val_images,
            "val_instances": val_instances,
            "classes": classes,
        },
        "codec": {
            "module": "integrations.qualcomm.lowbit",
            "formats": {f: {"block_size": lowbit.BLOCK_SIZES[f],
                            "code_bits": lowbit_formats.CODE_BITS[f]}
                        for f in lowbit.FORMATS},
        },
        "protocol": {
            "imgsz": args.imgsz,
            "device": "cpu",
            "split": "val",
            "arm_construction": ("fresh YOLO load per arm; fmt arms run "
                                 "quantize_arm (W-only RNE) then execute the "
                                 "dequantized weights"),
        },
        "arms": {},
    }

    arms = {}
    for arm in ARMS:
        print(f"[eval] arm {arm}: fresh load + build ...", flush=True)
        yolo, net = load_model(args.model)
        net.to("cpu")

        if arm == "fp32":
            fp32_bytes = sum(m.weight.numel() * 4 for _, m in _weight_modules(net))
            packed_bytes = fp32_bytes
            packed_ratio = 1.0
        else:
            packed, fp32_bytes, packed_bytes = quantize_arm(net, arm)
            packed_ratio = round(packed_bytes / fp32_bytes, 4)

        print(f"[eval] arm {arm}: yolo.val on {args.data} ...", flush=True)
        val_results = run_val(yolo, args.data, args.imgsz)

        entry = {
            "fp32_weight_bytes": fp32_bytes,
            "packed_bytes": packed_bytes,
            "packed_ratio": packed_ratio,
            **val_results,
        }
        if arm != "fp32":
            wcos = [pt.stats["weight_cosine"] for _, pt in packed.values()]
            entry["weight_cosine_mean"] = round(float(np.mean(wcos)), 6)
            entry["weight_cosine_min"] = round(float(np.min(wcos)), 6)
        arms[arm] = entry
        results["arms"][arm] = entry
        print(f"[eval] arm {arm}: mAP50={entry['map50']:.4f} "
              f"mAP50-95={entry['map50_95']:.4f}", flush=True)

        del yolo, net  # fresh load per arm; no cross-arm weight leakage

    fp32_map50 = arms["fp32"]["map50"]
    deltas = {}
    for arm in FMTS:
        deltas[arm] = {
            "map50_delta": round(arms[arm]["map50"] - fp32_map50, 6),
            "map50_95_delta": round(arms[arm]["map50_95"] - arms["fp32"]["map50_95"], 6),
        }
    results["deltas_vs_fp32"] = deltas
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    readme = render_readme(results, arms, deltas)

    run_id = f"yolo_2bit_map_{stamp}"
    writer = EvidenceBundleWriter(args.out_root)
    run_dir = writer.write_run(
        run_id=run_id,
        metadata={
            "question": results["question"],
            "model": args.model,
            "data": args.data,
            "arms": list(ARMS),
        },
        counts={},
        artifacts={
            "results.json": payload.encode("utf-8"),
            "README.md": readme.encode("utf-8"),
        },
    )
    print(f"[eval] bundle written to {run_dir}", flush=True)

    ok, errors = validate_evidence_bundle(run_dir)
    if not ok:
        raise SystemExit(f"[eval] FATAL: evidence bundle invalid: {errors}")
    print(f"[eval] bundle validated OK: {run_dir}", flush=True)

    print("\n== mAP50 summary (table_yolo val) ==")
    for arm in ARMS:
        d = "" if arm == "fp32" else f" (delta {deltas[arm]['map50_delta']:+.4f})"
        print(f"  {arm:7s} mAP50={arms[arm]['map50']:.4f} "
              f"mAP50-95={arms[arm]['map50_95']:.4f}{d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
