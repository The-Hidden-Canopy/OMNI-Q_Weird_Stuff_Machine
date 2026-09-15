"""OpenVINO optimisation of the table detector for THIS Intel chip (2026-09-15).

The brief asks for an Intel inference benchmark reporting latency,
throughput, device selection and precision, and for optimisation that does
not degrade task quality. This script does all four on the machine it runs
on, with the real scene-trained detector and real scene images:

  precision:  FP32 IR (ultralytics export) -> FP16 IR -> INT8 IR (NNCF
              post-training quantisation, calibrated on this scene's own
              training renders)
  devices:    every OpenVINO device the box reports (CPU, Intel iGPU, ...)
  hints:      LATENCY (batch 1, synchronous) and THROUGHPUT (async queue)
  quality:    mAP50 / mAP50-95 of every precision on the 268-image val
              split the detector was trained against

Inputs are real 640x640 letterboxed val images (not random tensors), 50
warm-up + 200 timed inferences per configuration. Output: a JSON receipt
plus a Markdown table under evidence/benchmark_results/.

    .venv/Scripts/python integrations/intel/scripts/benchmark_openvino_table.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

PT = ROOT / "runs/detect/tmp/yolo_runs/table_yolo_v4_hfbase_2026-09-14/weights/best.pt"
DATA = ROOT / "data/table_yolo_v4/data.yaml"


# IRs committed to the repo (the .pt weights are not): a fresh clone can still
# benchmark FP32 and INT8 without re-exporting; FP16 needs the weights.
TRACKED_IRS = {"fp32": ROOT / "models/table_yolo_v4_hfbase_2026-09-14_openvino_model",
               "int8": ROOT / "models/table_yolo_v4_int8_openvino_model"}


def export_precisions(out: Path) -> dict[str, Path]:
    """FP32, FP16 and NNCF-INT8 OpenVINO IRs of the same weights."""
    irs = {}
    for name, kw in (("fp32", {"half": False}), ("fp16", {"half": True}), ("int8", {"int8": True, "data": str(DATA)})):
        target = out / f"table_yolo_v4_{name}_openvino_model"
        if not (target / "best.xml").exists():
            if PT.exists() and (name != "int8" or DATA.exists()):
                from ultralytics import YOLO
                exported = Path(YOLO(str(PT)).export(format="openvino", imgsz=640, **kw))
                exported.rename(target)
            elif name in TRACKED_IRS and (TRACKED_IRS[name] / "best.xml").exists():
                import shutil
                shutil.copytree(TRACKED_IRS[name], target)
                print(f"  {name}: using the committed IR {TRACKED_IRS[name].relative_to(ROOT)} (weights not present)", flush=True)
            else:
                print(f"  {name}: skipped (needs {PT.relative_to(ROOT)})", flush=True)
                continue
        irs[name] = target
    return irs


def val_quality(irs: dict[str, Path]) -> dict[str, dict[str, float]]:
    from ultralytics import YOLO
    quality = {}
    for name, d in irs.items():
        r = YOLO(str(d)).val(data=str(DATA), imgsz=640, batch=8, device="cpu", verbose=False, plots=False)
        quality[name] = {"map50": float(r.box.map50), "map50_95": float(r.box.map)}
        print(f"  {name}: mAP50 {quality[name]['map50']:.4f}  mAP50-95 {quality[name]['map50_95']:.4f}", flush=True)
    return quality


def sample_inputs(n: int = 16):
    """Real val images, letterboxed to 640x640, NCHW float32 in [0, 1]."""
    import cv2
    import numpy as np
    imgs = sorted((ROOT / "data/table_yolo_v4/images/val").glob("*.png"))[:n] or \
        sorted((ROOT / "data/table_yolo_v4/images/val").glob("*.jpg"))[:n]
    frames = [cv2.imread(str(p)) for p in imgs]
    if not frames:
        # fresh clone without the rendered dataset: render the real scene now
        # (same cameras the perception loop uses) -- still real inputs, not noise
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "src"))
        import mujoco
        from omni_q.intel_sim import IntelSceneConfig, load_dual_so101_model
        for seed in range(900, 900 + max(1, n // 2)):
            model = load_dual_so101_model(IntelSceneConfig(seed=seed, randomized=True))
            data = mujoco.MjData(model); mujoco.mj_forward(model, data)
            r = mujoco.Renderer(model, height=480, width=640)
            for cam in ("table_overhead", "third_person"):
                r.update_scene(data, camera=cam); frames.append(cv2.cvtColor(r.render().copy(), cv2.COLOR_RGB2BGR))
            r.close()
        print(f"  inputs: {len(frames)} scene renders (data/table_yolo_v4 not present)", flush=True)
    batch = []
    for im in frames:
        h, w = im.shape[:2]
        s = 640 / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[: im.shape[0], : im.shape[1]] = im
        x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        batch.append(np.ascontiguousarray(x[None]))
    return batch


def bench(core, xml: Path, device: str, hint: str, inputs, warmup: int, iters: int) -> dict:
    import openvino as ov
    model = core.read_model(str(xml))
    compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": hint})
    n = len(inputs)
    if hint == "LATENCY":
        req = compiled.create_infer_request()
        for i in range(warmup):
            req.infer({0: inputs[i % n]})
        times = []
        for i in range(iters):
            t0 = time.perf_counter(); req.infer({0: inputs[i % n]}); times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        return {"device": device, "hint": hint, "iters": iters,
                "latency_ms_mean": round(statistics.fmean(times), 3), "latency_ms_p50": round(times[len(times) // 2], 3),
                "latency_ms_p95": round(times[int(len(times) * 0.95) - 1], 3), "fps": round(1000.0 / statistics.fmean(times), 1),
                "optimal_requests": int(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))}
    queue = ov.AsyncInferQueue(compiled)
    for i in range(warmup):
        queue.start_async({0: inputs[i % n]})
    queue.wait_all()
    t0 = time.perf_counter()
    for i in range(iters):
        queue.start_async({0: inputs[i % n]})
    queue.wait_all()
    wall = time.perf_counter() - t0
    return {"device": device, "hint": hint, "iters": iters, "throughput_fps": round(iters / wall, 1),
            "wall_s": round(wall, 3), "infer_requests": len(queue)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "evidence/benchmark_results" / f"openvino_table_detector_{datetime.date.today().isoformat()}")
    ap.add_argument("--devices", nargs="*", default=None, help="default: every non-NVIDIA OpenVINO device")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--skip-val", action="store_true")
    args = ap.parse_args()
    import openvino as ov
    args.out.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    names = {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}
    devices = args.devices or [d for d in core.available_devices if "NVIDIA" not in names[d]]
    print("devices:", {d: names[d] for d in devices}, flush=True)
    print("exporting precisions ...", flush=True)
    irs = export_precisions(args.out)
    quality = {} if (args.skip_val or not DATA.exists()) else val_quality(irs)
    if not DATA.exists() and not args.skip_val:
        print(f"  mAP validation skipped: {DATA.relative_to(ROOT)} not present (make_table_yolo_dataset.py regenerates it)", flush=True)
    inputs = sample_inputs()
    rows = []
    for prec, d in irs.items():
        for device in devices:
            for hint in ("LATENCY", "THROUGHPUT"):
                try:
                    r = bench(core, d / "best.xml", device, hint, inputs, args.warmup, args.iters)
                except Exception as exc:  # noqa: BLE001 - report, don't abort the sweep
                    r = {"device": device, "hint": hint, "error": str(exc)[:200]}
                r["precision"] = prec
                r["ir_size_mb"] = round(sum(p.stat().st_size for p in d.glob("best.*")) / 1e6, 2)
                rows.append(r)
                print("  ", {k: v for k, v in r.items() if k not in ("iters",)}, flush=True)
    receipt = {"date": datetime.date.today().isoformat(), "host": platform.processor() or platform.machine(),
               "openvino": ov.__version__, "devices": names, "detector": str(PT.relative_to(ROOT)), "val_images": 268,
               "quality": quality, "rows": rows,
               "protocol": {"input": "real val images, letterboxed 640x640, batch 1", "warmup": args.warmup, "iters": args.iters}}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=1))
    md = ["# OpenVINO table detector on this chip", "", f"{names}", "", f"OpenVINO {ov.__version__}; detector `{PT.name}` (scene-trained YOLOv8n, 7 classes)", "",
          "| precision | mAP50 | mAP50-95 | IR MB | device | hint | latency mean / p50 / p95 ms | FPS |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        q = quality.get(r["precision"], {})
        if "error" in r:
            md.append(f"| {r['precision']} | | | {r['ir_size_mb']} | {r['device']} | {r['hint']} | error: {r['error'][:60]} | |")
        elif r["hint"] == "LATENCY":
            md.append(f"| {r['precision']} | {q.get('map50', float('nan')):.3f} | {q.get('map50_95', float('nan')):.3f} | {r['ir_size_mb']} | {r['device']} | LATENCY | "
                      f"{r['latency_ms_mean']} / {r['latency_ms_p50']} / {r['latency_ms_p95']} | {r['fps']} |")
        else:
            md.append(f"| {r['precision']} | {q.get('map50', float('nan')):.3f} | {q.get('map50_95', float('nan')):.3f} | {r['ir_size_mb']} | {r['device']} | THROUGHPUT ({r['infer_requests']} req) | | {r['throughput_fps']} |")
    (args.out / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
