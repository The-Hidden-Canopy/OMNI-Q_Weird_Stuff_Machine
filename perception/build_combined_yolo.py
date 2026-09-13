"""Build an opt-in tableware + COCO context YOLO dataset.

The current seven-class dataset is a protected baseline.  This builder reads
it and locally cached COCO images/annotations, then writes a new output tree:

    python perception/build_combined_yolo.py \
        --table data/table_yolo_v3 \
        --coco-annotations data/_haul_cache/annotations_trainval2017.zip \
        --coco-train-images data/_haul_cache/coco_train_pull \
        --coco-val-images data/_haul_cache/coco_val_extract \
        --out data/table_yolo_combined_v1

The first seven class ids are unchanged.  Exact duplicate images are retained
once but their annotations are unioned, which is important when an existing
tableware-only label set and a dense COCO label set refer to the same frame.
Near duplicates are reported and dropped.  Inputs are never modified.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import combined_classmap as cm  # noqa: E402
import multisource_haul as mh  # noqa: E402


YoloRow = tuple[int, float, float, float, float]


@dataclass
class CombinedItem:
    source: str
    split: str
    name: str
    img: Path
    rows: list[YoloRow]
    annotation_sources: list[str] = field(default_factory=list)


def _dedupe_rows(rows: list[YoloRow]) -> list[YoloRow]:
    """Remove repeated boxes while preserving deterministic row order."""

    seen: set[tuple[int, float, float, float, float]] = set()
    out: list[YoloRow] = []
    for row in rows:
        key = (row[0], *(round(float(v), 6) for v in row[1:]))
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def iter_table_items(data_dir: Path) -> list[CombinedItem]:
    """Read an existing YOLO tree and keep its original seven-class ids."""

    out: list[CombinedItem] = []
    for split in ("train", "val"):
        for img in sorted((data_dir / "images" / split).glob("*.jpg")):
            label = data_dir / "labels" / split / f"{img.stem}.txt"
            if not label.exists():
                continue
            rows = mh.parse_yolo_label(label)
            if any(row[0] < 0 or row[0] >= len(cm.TABLE_CLASSES)
                   for row in rows):
                raise ValueError(f"{label}: class id outside the seven-class baseline")
            out.append(CombinedItem(
                source=data_dir.name,
                split=split,
                name=img.name,
                img=img,
                rows=rows,
                annotation_sources=[data_dir.name],
            ))
    return out


def _coco_rows(instances: dict) -> tuple[dict[int, list[YoloRow]], dict[int, str]]:
    """Return combined rows and COCO file names keyed by image id."""

    category_names = {c["id"]: c["name"] for c in instances["categories"]}
    images = {
        image["id"]: (image["width"], image["height"], image["file_name"])
        for image in instances["images"]
    }
    rows: dict[int, list[YoloRow]] = {}
    for annotation in instances["annotations"]:
        image_id = annotation["image_id"]
        image_meta = images.get(image_id)
        target = cm.remap_coco(category_names.get(annotation["category_id"], ""))
        if image_meta is None or target is None:
            continue
        box = mh.coco_bbox_to_yolo(
            annotation["bbox"], image_meta[0], image_meta[1]
        )
        if box is None:
            continue
        rows.setdefault(image_id, []).append((cm.TARGET_INDEX[target], *box))
    return {key: _dedupe_rows(value) for key, value in rows.items()}, {
        key: value[2] for key, value in images.items()
    }


def iter_coco_items(annotation_zip: Path, image_dirs: dict[str, Path]) -> tuple[list[CombinedItem], dict]:
    """Read cached COCO train/val images and annotations without network I/O."""

    items: list[CombinedItem] = []
    stats: dict = {"splits": {}, "class_boxes": {name: 0 for name in cm.TARGET_CLASSES}}
    with zipfile.ZipFile(annotation_zip) as archive:
        for split, image_dir in (("train2017", image_dirs["train2017"]),
                                 ("val2017", image_dirs["val2017"])):
            instance_file = f"annotations/instances_{split}.json"
            instances = json.loads(archive.read(instance_file))
            rows_by_id, file_by_id = _coco_rows(instances)
            available = {path.name: path for path in image_dir.glob("*.jpg")}
            offered = 0
            missing = 0
            for image_id in sorted(rows_by_id):
                name = file_by_id[image_id]
                img = available.get(name)
                if img is None:
                    missing += 1
                    continue
                rows = rows_by_id[image_id]
                items.append(CombinedItem(
                    source=f"coco_{split}",
                    split="source",
                    name=name,
                    img=img,
                    rows=rows,
                    annotation_sources=[f"coco_{split}"],
                ))
                offered += 1
                for class_id, *_ in rows:
                    stats["class_boxes"][cm.TARGET_CLASSES[class_id]] += 1
            stats["splits"][split] = {
                "annotated_images": len(rows_by_id),
                "images_available": offered,
                "images_missing": missing,
                "image_dir": str(image_dir),
            }
    return items, stats


class _IndexedDeduper:
    """Exact SHA plus indexed dHash lookup for large combined trees.

    The legacy ``mh.Deduper`` scans every prior hash for every new image.  A
    full v3 tableware tree plus context images makes that quadratic scan
    unnecessarily expensive.  Four 16-bit bands are sufficient here: any two
    64-bit hashes at Hamming distance <= 6 must share a band within one bit,
    so querying exact and one-bit-neighbor buckets is complete for the
    configured near-duplicate threshold while keeping candidate sets small.
    """

    def __init__(self, tolerance: int = mh.NEAR_DUP_TOLERANCE) -> None:
        self.tolerance = tolerance
        self._by_sha: dict[str, str] = {}
        self._dhash_by_key: dict[str, int] = {}
        self._order: dict[str, int] = {}
        self._buckets: dict[tuple[int, int], set[str]] = {}

    @staticmethod
    def _band_values(dhash: int) -> list[int]:
        return [(dhash >> (16 * band)) & 0xFFFF for band in range(4)]

    def _candidates(self, dhash: int) -> set[str]:
        candidates: set[str] = set()
        for band, value in enumerate(self._band_values(dhash)):
            values = [value] + [value ^ (1 << bit) for bit in range(16)]
            for neighbor in values:
                candidates.update(self._buckets.get((band, neighbor), ()))
        return candidates

    def add(self, key: str, sha256: str, dhash: int) -> mh.DupDecision:
        if sha256 in self._by_sha:
            return mh.DupDecision("exact_dup", self._by_sha[sha256])
        candidates = sorted(self._candidates(dhash), key=self._order.__getitem__)
        for candidate in candidates:
            if mh.hamming(self._dhash_by_key[candidate], dhash) <= self.tolerance:
                return mh.DupDecision("near_dup", candidate)
        self._by_sha[sha256] = key
        self._dhash_by_key[key] = dhash
        self._order[key] = len(self._order)
        for band, value in enumerate(self._band_values(dhash)):
            self._buckets.setdefault((band, value), set()).add(key)
        return mh.DupDecision("unique")


def _merge_unique(items: list[CombinedItem]) -> tuple[list[CombinedItem], dict]:
    """Deduplicate images, unioning labels for exact duplicates."""

    deduper = _IndexedDeduper()
    kept: list[CombinedItem] = []
    index_by_key: dict[str, int] = {}
    decisions: list[dict[str, str]] = []
    source_stats: dict[str, dict[str, int]] = {}

    for item in items:
        src = source_stats.setdefault(item.source, {
            "offered": 0, "unique_output_images": 0,
            "exact_annotation_merges": 0, "near_duplicates_dropped": 0,
        })
        src["offered"] += 1
        key = f"{item.source}:{item.name}"
        sha = mh.sha256_of(item.img)
        with Image.open(item.img) as image:
            dhash = mh.dhash64(image)
        decision = deduper.add(key, sha, dhash)
        if decision.status == "unique":
            index_by_key[key] = len(kept)
            kept.append(item)
            src["unique_output_images"] += 1
        elif decision.status == "exact_dup":
            representative = kept[index_by_key[decision.matched or ""]]
            representative.rows = _dedupe_rows(representative.rows + item.rows)
            for annotation_source in item.annotation_sources:
                if annotation_source not in representative.annotation_sources:
                    representative.annotation_sources.append(annotation_source)
            src["exact_annotation_merges"] += 1
            decisions.append({
                "status": "exact_annotation_merge",
                "image": key,
                "matched": decision.matched or "",
            })
        else:
            src["near_duplicates_dropped"] += 1
            decisions.append({
                "status": decision.status,
                "image": key,
                "matched": decision.matched or "",
            })
    return kept, {"source_composition": source_stats, "decisions": decisions}


def _histogram(items: list[CombinedItem]) -> dict[str, int]:
    counts = {name: 0 for name in cm.TARGET_CLASSES}
    for item in items:
        for class_id, *_ in item.rows:
            counts[cm.TARGET_CLASSES[class_id]] += 1
    return counts


def _write_receipt(out: Path, manifest: dict) -> Path | None:
    evidence_root = Path("evidence/datasets")
    if not evidence_root.is_dir():
        return None
    stamp = datetime.now().strftime("%Y%m%d")
    destination = evidence_root / f"{out.name}_{stamp}"
    suffix = 1
    while destination.exists():
        destination = evidence_root / f"{out.name}_{stamp}_{suffix}"
        suffix += 1
    destination.mkdir(parents=True)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    split = manifest["split"]
    classes = manifest["class_boxes_final"]
    source_lines = "\n".join(
        f"- `{source}`: offered {values['offered']}; output images "
        f"{values['unique_output_images']}; exact annotation merges "
        f"{values['exact_annotation_merges']}; near duplicates dropped "
        f"{values['near_duplicates_dropped']}"
        for source, values in manifest["source_composition"].items()
    )
    (destination / "README.md").write_text(
        f"""# {out.name} — combined YOLO dataset ({stamp})

Status: built from the protected seven-class tableware tree plus locally
cached COCO 2017 train/validation images.  The tableware ids remain 0–6;
COCO aliases `wine glass -> cup` and `dining table -> table` are explicit.

## Composition

{source_lines}

Final images: **{manifest['final_unique_images']}** — train {split['train']} /
val {split['val']} (seed {manifest['seed']}).  Exact duplicate images were
kept once and their annotations unioned; near duplicates were dropped and
recorded in `manifest.json`.

Class boxes: {' · '.join(f"{name} {count}" for name, count in classes.items() if count)}.

Use with an opt-in combined-head training run:

```text
python perception/finetune.py --data {str(out / 'data.yaml').replace(chr(92), '/')} \\
    --base models/table_yolo_v2_ft_2026-09-11.pt --name combined_v1
```

The source images remain local/in-house artifacts.  COCO annotations are
CC-BY-4.0; image redistribution rights remain with their original owners.
""",
        encoding="utf-8",
    )
    return destination


def build_combined_dataset(
    table_dirs: list[Path],
    annotation_zip: Path,
    coco_image_dirs: dict[str, Path],
    out: Path,
    *,
    val_frac: float = 0.10,
    seed: int = 13,
    link: bool = True,
) -> dict:
    """Build a fresh combined tree without changing any input."""

    if not table_dirs:
        raise ValueError("at least one tableware YOLO dataset is required")
    for directory in table_dirs:
        if not (directory / "images").is_dir():
            raise ValueError(f"{directory}: no images/ directory")
    if not annotation_zip.is_file():
        raise FileNotFoundError(annotation_zip)
    for directory in coco_image_dirs.values():
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    if not 0.0 <= val_frac <= 1.0:
        raise ValueError("val_frac must be between 0 and 1")

    started = time.time()
    items: list[CombinedItem] = []
    table_counts: dict[str, int] = {}
    for directory in table_dirs:
        table_items = iter_table_items(directory)
        table_counts[directory.name] = len(table_items)
        items.extend(table_items)
    coco_items, coco_stats = iter_coco_items(annotation_zip, coco_image_dirs)
    items.extend(coco_items)
    kept, dedup = _merge_unique(items)

    output = mh._fresh_dir(Path(out))
    output.mkdir(parents=True)
    rng = random.Random(seed)
    n_train = n_val = 0
    per_image: dict[str, dict] = {}
    for index, item in enumerate(kept):
        split = "val" if rng.random() < val_frac else "train"
        stem = f"{item.source}__{item.split}_{index:07d}"
        output_image = output / "images" / split / f"{stem}.jpg"
        output_label = output / "labels" / split / f"{stem}.txt"
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output_label.parent.mkdir(parents=True, exist_ok=True)
        if link:
            try:
                output_image.hardlink_to(item.img)
            except OSError:
                output_image.write_bytes(item.img.read_bytes())
        else:
            output_image.write_bytes(item.img.read_bytes())
        mh.write_yolo_label(output_label, item.rows)
        per_image[output_image.name] = {
            "source": item.source,
            "original": item.name,
            "annotation_sources": item.annotation_sources,
        }
        if split == "train":
            n_train += 1
        else:
            n_val += 1

    (output / "data.yaml").write_text(
        f"# {output.name} — generated by perception/build_combined_yolo.py\n"
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n"
                  for index, name in enumerate(cm.TARGET_CLASSES)),
        encoding="utf-8",
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "builder": "perception/build_combined_yolo.py",
        "seed": seed,
        "args": {
            "table_dirs": [str(directory) for directory in table_dirs],
            "annotation_zip": str(annotation_zip),
            "coco_image_dirs": {key: str(value) for key, value in coco_image_dirs.items()},
            "out": str(output),
            "val_frac": val_frac,
            "link": link,
        },
        "taxonomy": {
            "class_count": len(cm.TARGET_CLASSES),
            "tableware_prefix": list(cm.TABLE_CLASSES),
            "coco_category_map": cm.mapped_coco_labels(),
        },
        "tableware_source_counts": table_counts,
        "coco_cache": coco_stats,
        "source_composition": dedup["source_composition"],
        "dedup": {
            "method_exact": "sha256 of image bytes; exact duplicates union labels",
            "method_near": f"64-bit dHash, Hamming <= {mh.NEAR_DUP_TOLERANCE}",
            "exact_annotation_merges": sum(
                values["exact_annotation_merges"]
                for values in dedup["source_composition"].values()
            ),
            "near_duplicates_dropped": sum(
                values["near_duplicates_dropped"]
                for values in dedup["source_composition"].values()
            ),
            "decision_head": dedup["decisions"][:50],
        },
        "final_unique_images": len(kept),
        "split": {"train": n_train, "val": n_val},
        "class_boxes_final": _histogram(kept),
        "source_per_image": per_image,
        "license_note": mh.LICENSE_NOTE,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    receipt = _write_receipt(output, manifest)
    if receipt:
        manifest["evidence_receipt"] = str(receipt)
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    print(
        f"combined dataset at {output}\n"
        f"unique images: {len(kept)} (train {n_train} / val {n_val})\n"
        f"exact annotation merges: {manifest['dedup']['exact_annotation_merges']}; "
        f"near duplicates dropped: {manifest['dedup']['near_duplicates_dropped']}\n"
        f"non-empty class boxes: "
        f"{Counter({k: v for k, v in manifest['class_boxes_final'].items() if v})}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", action="append", type=Path, required=True,
                        help="existing tableware YOLO tree; repeatable")
    parser.add_argument("--coco-annotations", type=Path, required=True,
                        help="local annotations_trainval2017.zip")
    parser.add_argument("--coco-train-images", type=Path, required=True)
    parser.add_argument("--coco-val-images", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("data/table_yolo_combined_v1"))
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--copy", action="store_true",
                        help="copy images instead of hardlinking")
    args = parser.parse_args()
    build_combined_dataset(
        args.table,
        args.coco_annotations,
        {"train2017": args.coco_train_images, "val2017": args.coco_val_images},
        args.out,
        val_frac=args.val_frac,
        seed=args.seed,
        link=not args.copy,
    )


if __name__ == "__main__":
    main()
