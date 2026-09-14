"""Fine-tune YOLOv8n on the auto-labelled MuJoCo table dataset and export to
OpenVINO IR for `omni_q.vision.OpenVINODetector`.

    python integrations/intel/scripts/make_table_yolo_dataset.py --out data/table_yolo_v3 --n 400
    python integrations/intel/scripts/train_table_yolo.py --data data/table_yolo_v3/data.yaml --epochs 30

Writes `models/table_yolo_v3_<date>_openvino_model/` (xml + bin + metadata.yaml)
and a receipt JSON next to it with the validation metrics, dataset size and
sha256 of the IR, so the detector the demo runs is traceable to the data it
was trained on. Weights (.pt) and the dataset are git-ignored; the IR is not.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/table_yolo_v3/data.yaml"))
    ap.add_argument("--base", default="yolov8n.pt",
                    help="starting weights; 'hf:KissTheHabit/yolov8n-table-yolo/yolov8n-table-yolo-ftv3-30ep.pt' "
                         "fine-tunes from the real-photo tableware detector on HF")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default=f"table_yolo_v3_{dt.date.today().isoformat()}")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    args = ap.parse_args()

    from ultralytics import YOLO  # noqa: PLC0415

    base = args.base
    if base.startswith("hf:"):
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        repo, _, filename = base[3:].rpartition("/")
        base = hf_hub_download(repo, filename)
    model = YOLO(base)
    results = model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=0, project="tmp/yolo_runs", name=args.name, exist_ok=True,
        verbose=False, plots=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    trained = YOLO(str(best))
    metrics = trained.val(data=str(args.data), imgsz=args.imgsz, device=args.device, workers=0, plots=False, verbose=False)
    export_dir = Path(trained.export(format="openvino", imgsz=args.imgsz, half=False))

    target = args.models_dir / f"{args.name}_openvino_model"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(export_dir, target)
    xml = next(target.glob("*.xml"))
    binf = next(target.glob("*.bin"))
    n_train = len(list((args.data.parent / "images" / "train").glob("*.png")))
    n_val = len(list((args.data.parent / "images" / "val").glob("*.png")))
    receipt = {
        "name": args.name, "base": args.base, "epochs": args.epochs, "imgsz": args.imgsz,
        "dataset": {"yaml": str(args.data), "train_images": n_train, "val_images": n_val,
                    "generator": "integrations/intel/scripts/make_table_yolo_dataset.py"},
        "val": {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
                "per_class_map50": {trained.names[i]: float(v) for i, v in zip(metrics.box.ap_class_index, metrics.box.ap50)}},
        "openvino": {"xml": xml.name, "bin": binf.name, "bin_sha256": sha256(binf), "xml_sha256": sha256(xml)},
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (target / "receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))
    print(f"\nexported to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
