"""OQ-008 step 1 — assemble the REAL-image training set.

Pull tableware-class detections from public datasets via FiftyOne, remap every
source label to the 7-class table vocabulary (``classmap.remap``), merge, and
export a YOLOv5-format dataset that ``finetune.py`` trains on.

    python perception/build_dataset.py \
        --sources open-images-v7 coco-2017 lvis \
        --max-per-source 60000 --val-frac 0.10 --out data/table_yolo

    # verify the plumbing without a big download:
    python perception/build_dataset.py --sources coco-2017 --smoke

Objects365 has no FiftyOne zoo loader — fetch it from OpenDataLab / the
Ultralytics ``Objects365.yaml`` and point ``--extra-yolo`` at its labels, or use
``synth_ingest.py`` which takes any YOLO-txt tree.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from classmap import TARGET_CLASSES, TARGET_INDEX, remap, request_labels

_ZOO = {  # source -> (zoo name, split, kwargs)
    "open-images-v7": ("open-images-v7", "train", {"label_types": ["detections"]}),
    "coco-2017": ("coco-2017", "train", {}),
    "lvis": ("coco-2017", "train", {"label_field": "ground_truth", "dataset_name": None}),
}


def _load(source: str, max_samples: int | None):
    import fiftyone as fo  # noqa: F401
    import fiftyone.zoo as foz

    if source == "lvis":
        # LVIS shares COCO images; FiftyOne exposes it as its own zoo dataset.
        return foz.load_zoo_dataset(
            "lvis", split="train",
            classes=request_labels("lvis"),
            max_samples=max_samples,
        ), "detections"

    name, split, kw = _ZOO[source]
    ds = foz.load_zoo_dataset(
        name, split=split,
        classes=request_labels(source),
        max_samples=max_samples,
        **kw,
    )
    # the detections field differs by dataset
    field = "detections" if source == "open-images-v7" else "ground_truth"
    return ds, field


def _iter_samples(ds, field, source):
    """Yield (image_path, [(cls_idx, cx, cy, w, h), ...]) in YOLO-norm coords."""
    for smp in ds:
        dets = smp[field]
        if dets is None:
            continue
        rows = []
        for d in dets.detections:
            tgt = remap(source, d.label)
            if tgt is None:
                continue
            x, y, w, h = d.bounding_box  # FiftyOne: top-left x,y + w,h, normalised
            rows.append((TARGET_INDEX[tgt], x + w / 2, y + h / 2, w, h))
        if rows:
            yield smp.filepath, rows


def _write(out: Path, split: str, image_path: str, rows, idx: int) -> None:
    stem = f"{split}_{idx:07d}"
    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(image_path).suffix or ".jpg"
    dst = img_dir / f"{stem}{ext}"
    if not dst.exists():
        try:
            dst.hardlink_to(image_path)
        except (OSError, AttributeError):
            import shutil
            shutil.copy2(image_path, dst)
    (lbl_dir / f"{stem}.txt").write_text(
        "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in rows)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["open-images-v7", "coco-2017", "lvis"])
    ap.add_argument("--out", type=Path, default=Path("data/table_yolo"))
    ap.add_argument("--max-per-source", type=int, default=60000)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--smoke", action="store_true", help="tiny pull to test plumbing")
    args = ap.parse_args()

    if args.smoke:
        args.max_per_source = 50

    rng = random.Random(args.seed)
    counts = {c: 0 for c in TARGET_CLASSES}
    n_train = n_val = idx = 0

    for source in args.sources:
        ds, field = _load(source, args.max_per_source)
        for image_path, rows in _iter_samples(ds, field, source):
            split = "val" if rng.random() < args.val_frac else "train"
            _write(args.out, split, image_path, rows, idx)
            idx += 1
            n_train += split == "train"
            n_val += split == "val"
            for c, *_ in rows:
                counts[TARGET_CLASSES[c]] += 1
        print(f"[{source}] done  (running totals: train={n_train} val={n_val})")

    print("\nper-class boxes:", {k: v for k, v in counts.items()})
    thin = [k for k, v in counts.items() if v < 500]
    if thin:
        print(f"THIN classes {thin} — top up with synth_ingest.py (esp. 'drawer')")
    print(f"dataset at {args.out}  (point perception/data.yaml 'path' here)")


if __name__ == "__main__":
    main()
