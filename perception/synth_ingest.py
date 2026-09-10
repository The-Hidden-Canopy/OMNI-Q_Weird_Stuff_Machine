"""OQ-008 step 2 — fold synthetic / extra YOLO-txt frames into the dataset.

Use for the MuJoCo-rendered depth top-up (`drawer`, deploy camera, hard
occlusion/clutter) and for Objects365 or Roboflow exports that already ship
YOLO-txt.

    python perception/synth_ingest.py --src renders/mujoco --out data/table_yolo \
        --val-frac 0.10 --tag synth

The ``--src`` tree must be YOLO layout:  images/*.{jpg,png}  +  labels/*.txt
with class ids already in the 7-class order of perception/data.yaml. If a
``classes.txt`` sits next to it, pass ``--src-names classes.txt`` to remap.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from classmap import TARGET_CLASSES, TARGET_INDEX

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_names(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _remap_line(line: str, src_names: list[str] | None) -> str | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    cid = int(float(parts[0]))
    if src_names is not None:
        if cid >= len(src_names):
            return None
        name = src_names[cid]
        if name not in TARGET_INDEX:
            return None
        cid = TARGET_INDEX[name]
    if cid < 0 or cid >= len(TARGET_CLASSES):
        return None
    return " ".join([str(cid), *parts[1:5]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/table_yolo"))
    ap.add_argument("--src-names", type=Path, default=None)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--tag", default="synth")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    src_names = _load_names(args.src_names)
    rng = random.Random(args.seed)
    img_root = args.src / "images"
    lbl_root = args.src / "labels"
    imgs = sorted(p for p in img_root.rglob("*") if p.suffix.lower() in _IMG_EXT)
    if not imgs:
        raise SystemExit(f"no images under {img_root}")

    kept = dropped = 0
    counts = {c: 0 for c in TARGET_CLASSES}
    for i, img in enumerate(imgs):
        lbl = lbl_root / f"{img.stem}.txt"
        if not lbl.exists():
            dropped += 1
            continue
        rows = [r for r in (_remap_line(ln, src_names)
                            for ln in lbl.read_text().splitlines() if ln.strip())
                if r]
        if not rows:
            dropped += 1
            continue
        split = "val" if rng.random() < args.val_frac else "train"
        stem = f"{args.tag}_{split}_{i:07d}"
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        dst = args.out / "images" / split / f"{stem}{img.suffix.lower()}"
        if not dst.exists():
            try:
                dst.hardlink_to(img)
            except (OSError, AttributeError):
                shutil.copy2(img, dst)
        (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(rows))
        kept += 1
        for r in rows:
            counts[TARGET_CLASSES[int(r.split()[0])]] += 1

    print(f"ingested {kept} frames ({dropped} skipped) from {args.src}")
    print("per-class boxes added:", counts)


if __name__ == "__main__":
    main()
