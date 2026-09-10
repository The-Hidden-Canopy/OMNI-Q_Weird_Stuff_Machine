"""OQ-008 step 3 — fine-tune the table detector from OUR thermal YOLOv8n.

Base weights: ``KissTheHabit/yolov8n-hituav-thermal-finetune`` (a trained
YOLOv8n we own). The head is re-initialised for 7 classes automatically by
Ultralytics when ``data.yaml`` has 7 names.

    python perception/finetune.py --epochs 4 --freeze 10 --imgsz 640
    python perception/finetune.py --base runs/last.pt --epochs 2 --freeze 0

Breadth over repetition: the variety is in the real+synth data, so keep passes
few and early-stop on val mAP50.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HF_BASE_REPO = "KissTheHabit/yolov8n-hituav-thermal-finetune"
_BASE_CANDIDATES = ("best.pt", "weights/best.pt", "yolov8n-hituav.pt", "model.pt")


def resolve_base(base: str | None) -> str:
    if base:
        return base
    from huggingface_hub import HfApi, hf_hub_download

    files = {f for f in HfApi().list_repo_files(HF_BASE_REPO)}
    for cand in _BASE_CANDIDATES:
        if cand in files:
            return hf_hub_download(HF_BASE_REPO, cand)
    pt = next((f for f in files if f.endswith(".pt")), None)
    if pt is None:
        raise SystemExit(f"no .pt in {HF_BASE_REPO}; pass --base explicitly")
    return hf_hub_download(HF_BASE_REPO, pt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).with_name("data.yaml")))
    ap.add_argument("--base", default=None, help="local .pt; default = pull from HF")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--freeze", type=int, default=10, help="freeze first N layers")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=2, help="early-stop patience")
    ap.add_argument("--project", default="runs/table_yolo")
    ap.add_argument("--name", default="ft")
    args = ap.parse_args()

    from ultralytics import YOLO

    base = resolve_base(args.base)
    print(f"base weights: {base}")
    model = YOLO(base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        patience=args.patience,
        project=args.project,
        name=args.name,
        pretrained=True,
        exist_ok=True,
    )
    print(f"best: {args.project}/{args.name}/weights/best.pt")
    print("next: python perception/export.py --weights "
          f"{args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
