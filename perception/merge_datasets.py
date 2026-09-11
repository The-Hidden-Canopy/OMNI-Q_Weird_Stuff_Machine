"""OQ-008 v3 — merge YOLO dataset dirs into a single bigger set.

Merges e.g. table_yolo_v2 (COCO+LVIS train) + table_yolo_v3_oi (Open Images
train) into data/table_yolo_v3 with a FRESH train/val split:

  * images are HARDLINKED (copy fallback on cross-device OSError) — sources
    are never touched, let alone destroyed
  * sha256 + dHash dedup ACROSS inputs (belt and braces on top of the
    haul-time cross-set dedup): an image already seen from an earlier input
    is dropped and counted
  * output stems are prefixed with the source dir name
    (`table_yolo_v2__train_0000123.jpg`) so every image stays traceable
  * fresh seeded split, absolute-path data.yaml (repo convention), manifest,
    evidence receipt under evidence/datasets/

    python perception/merge_datasets.py \
        --input data/table_yolo_v2 --input data/table_yolo_v3_oi \
        --out data/table_yolo_v3 --val-frac 0.10 --seed 13

Pure helpers at the top; I/O in merge_datasets() / main().
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import multisource_haul as mh  # noqa: E402

TARGET_CLASSES = mh.TARGET_CLASSES
LICENSE_NOTE = mh.LICENSE_NOTE


# ---------------------------------------------------------------------------
# Pure helpers (testable: no network)
# ---------------------------------------------------------------------------

@dataclass
class MergeItem:
    source: str            # input dir name (stem prefix)
    split: str             # ORIGINAL split inside its input dataset
    name: str              # original image file name
    img: Path
    label: Path
    rows: list[mh.YoloRow]


def iter_dataset_items(data_dir: Path) -> list[MergeItem]:
    """(image, label) pairs of a YOLO tree, sorted deterministically.

    Images without a matching label file are skipped (YOLO convention:
    unlabeled images carry no training signal here) and the caller counts
    them via the return difference if it cares.
    """
    out: list[MergeItem] = []
    for split in ("train", "val"):
        for img in sorted((data_dir / "images" / split).glob("*.jpg")):
            label = data_dir / "labels" / split / f"{img.stem}.txt"
            if not label.exists():
                continue
            out.append(MergeItem(source=data_dir.name, split=split, name=img.name,
                                 img=img, label=label,
                                 rows=mh.parse_yolo_label(label)))
    return out


def plan_merge(items: list[MergeItem], tolerance: int = mh.NEAR_DUP_TOLERANCE
               ) -> tuple[list[MergeItem], list[tuple[str, str]]]:
    """Cross-input sha256 + dHash dedup (belt and braces on top of the
    haul-time dedup).  Returns (kept, dropped); dropped pairs are
    (source:name, matched source:name) — a dropped image duplicating its
    OWN input still carries that input's name as the match prefix.
    """
    deduper = mh.Deduper(tolerance=tolerance)
    kept: list[MergeItem] = []
    dropped: list[tuple[str, str]] = []
    for it in items:
        sha = mh.sha256_of(it.img)
        with Image.open(it.img) as im:
            dh = mh.dhash64(im)
        key = f"{it.source}:{it.name}"
        dec = deduper.add(key, sha, dh)
        if dec.status == "unique":
            kept.append(it)
        else:
            dropped.append((key, dec.matched or ""))
    return kept, dropped


def merge_datasets(inputs: list[Path], out: Path, val_frac: float = 0.10,
                   seed: int = 13, link: bool = True) -> dict:
    """Merge inputs -> a fresh YOLO tree at ``out`` (never touches inputs).

    Returns the manifest dict (also written to <out>/manifest.json).
    """
    t_start = time.time()
    inputs = [Path(i) for i in inputs]
    for d in inputs:
        if not (d / "images").is_dir():
            raise SystemExit(f"{d}: no images/ dir — not a YOLO dataset tree")
    out = mh._fresh_dir(Path(out))
    out.mkdir(parents=True)

    # gather in a FIXED order (input order, then name) so the split is stable
    items: list[MergeItem] = []
    source_counts: dict[str, int] = {}
    for d in inputs:
        got = iter_dataset_items(d)
        items.extend(got)
        source_counts[d.name] = len(got)
    kept, dropped = plan_merge(items)
    dropped_by_target: dict[str, int] = {}
    for _, matched in dropped:
        dropped_by_target[matched.split(":", 1)[0]] = \
            dropped_by_target.get(matched.split(":", 1)[0], 0) + 1

    rng = random.Random(seed)
    n_train = n_val = 0
    idx = 0
    per_image: dict[str, dict] = {}
    per_source_kept: dict[str, int] = {}
    for it in kept:
        split = "val" if rng.random() < val_frac else "train"
        stem = f"{it.source}__{it.split}_{idx:07d}"
        out_img = out / "images" / split / f"{stem}.jpg"
        out_lbl = out / "labels" / split / f"{stem}.txt"
        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        if link:
            try:
                out_img.hardlink_to(it.img)
            except OSError:
                out_img.write_bytes(it.img.read_bytes())
        else:
            out_img.write_bytes(it.img.read_bytes())
        mh.write_yolo_label(out_lbl, it.rows)
        per_image[out_img.name] = {"source": it.source, "original": it.name}
        per_source_kept[it.source] = per_source_kept.get(it.source, 0) + 1
        n_train += split == "train"
        n_val += split == "val"
        idx += 1

    (out / "data.yaml").write_text(
        f"# OQ-008 {out.name} — generated by perception/merge_datasets.py\n"
        f"path: {out.resolve()}  # absolute: pin to this repo (env ultralytics settings point outside)\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(TARGET_CLASSES)),
        encoding="utf-8",
    )

    manifest: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "args": {"inputs": [str(i) for i in inputs], "out": str(out),
                 "val_frac": val_frac, "seed": seed, "link": link},
        "source_composition": {name: {"offered": source_counts.get(name, 0),
                                      "kept": per_source_kept.get(name, 0)}
                               for name in source_counts},
        "dedup": {
            "method_exact": "sha256 of image bytes",
            "method_near": f"64-bit dHash, Hamming <= {mh.NEAR_DUP_TOLERANCE}",
            "dropped_total": len(dropped),
            "dropped_by_matched_source": dropped_by_target,
            "dropped_head": [list(p) for p in dropped[:20]],
        },
        "final_unique_images": len(kept),
        "split": {"train": n_train, "val": n_val},
        "class_boxes_final": mh.histogram({f"{it.source}:{it.name}": it.rows
                                           for it in kept}),
        "source_per_image": per_image,
        "license_note": LICENSE_NOTE,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"merged dataset at {out}")
    print(f"unique images: {len(kept)}  (train {n_train} / val {n_val}); "
          f"cross-input dups dropped: {len(dropped)}")
    print(f"per-source kept: {per_source_kept}")
    print(f"final class boxes: {manifest['class_boxes_final']}")
    _write_receipt(out, manifest)
    return manifest


def _write_receipt(out: Path, manifest: dict) -> Path | None:
    ev = Path("evidence/datasets")
    if not ev.is_dir():
        return None
    stamp = datetime.now().strftime("%Y%m%d")
    dest = ev / f"{out.name}_{stamp}"
    n = 1
    while dest.exists():
        dest = ev / f"{out.name}_{stamp}_{n}"
        n += 1
    dest.mkdir(parents=True)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    comp = manifest["source_composition"]
    dedup = manifest["dedup"]
    split = manifest["split"]
    final = manifest["class_boxes_final"]
    comp_md = "\n".join(
        f"- `{name}`: {v['kept']} of {v['offered']} images kept"
        for name, v in comp.items())
    md = f"""# {out.name} — merged train-scale dataset ({stamp})

**Status: merge complete.** Same 7-class vocabulary (`plate cup fork spoon
knife napkin drawer`), YOLOv5 layout, built by `perception/merge_datasets.py`
from the v2 (COCO+LVIS train) and v3 OI component hauls.  Source dataset dirs
were only READ (hardlinks into this tree) — v2 and the OI dir are intact.

## Source composition

{comp_md}

## Dedup (belt-and-braces pass on top of haul-time dedup)

- Exact: sha256 of image bytes.  Near: 64-bit dHash, Hamming <= {mh.NEAR_DUP_TOLERANCE}.
- **{dedup['dropped_total']} cross-input duplicates dropped**
  (matched against: {dedup['dropped_by_matched_source']}).

## Final numbers

- **{manifest['final_unique_images']} unique images** — train {split['train']} /
  val {split['val']} (val-frac {manifest['args']['val_frac']}, seed
  {manifest['seed']} — same seed convention as the v2 haul).
- Class boxes: {" · ".join(f"{c} {n}" for c, n in final.items())}.
- Elapsed: {manifest['elapsed_seconds']} s.

## Usage

```bash
python perception/finetune.py --data {str(out).replace(chr(92), '/')}/data.yaml ...
```

`data/` is gitignored; reproducible by re-running the command in the
manifest's `args`.

## License

{LICENSE_NOTE}

Portions derived from *The Hidden Canopy LLC* — [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
"""
    (dest / "README.md").write_text(md, encoding="utf-8")
    print(f"receipt at {dest}", flush=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", type=Path, required=True,
                    help="input YOLO dataset dir(s); repeatable, order is the "
                         "tie-break order for dedup and the split")
    ap.add_argument("--out", type=Path, default=Path("data/table_yolo_v3"))
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=13,
                    help="same convention as the v2 haul (13)")
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of hardlinking (slower, more disk)")
    args = ap.parse_args()
    merge_datasets(args.input, args.out, val_frac=args.val_frac,
                   seed=args.seed, link=not args.copy)


if __name__ == "__main__":
    main()
