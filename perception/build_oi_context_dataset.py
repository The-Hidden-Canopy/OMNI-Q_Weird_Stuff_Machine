"""Build an 82-class YOLO dataset with a capped Open Images supplement.

The protected input is the existing combined 82-class tree.  This builder
adds only Open Images validation images whose box labels can be mapped to the
same target vocabulary, deduplicates against the protected tree, and writes a
new output tree.  It intentionally does not mutate the input dataset.

Open Images is only partially annotated across the full 82-class vocabulary,
so the manifest records the mapped and missing classes.  The resulting tree
is a source-augmentation candidate, not promotion evidence by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import combined_classmap as cm  # noqa: E402
import multisource_haul as mh  # noqa: E402


URL_OI_CLASSES = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
URL_OI_BOXES = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
URL_OI_IMAGE = "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"

# OI uses several display-name variants.  Deliberately keep mappings narrow;
# broad labels such as "Tableware" must not become a guessed plate/cup label.
ALIASES: dict[str, tuple[str, ...]] = {
    "cup": ("coffee cup", "mug", "wine glass", "teacup"),
    "suitcase": ("luggage",),
    "sports ball": ("ball",),
    "table": ("dining table", "kitchen & dining room table"),
    "tv": ("television", "television set", "monitor"),
    "cell phone": ("mobile phone", "cellphone"),
    "donut": ("doughnut",),
    "hair drier": ("hair dryer",),
    "bottle": ("water bottle",),
    "bowl": ("bowl/basin",),
    "drawer": ("drawer",),
}


def normalize(value: str) -> str:
    value = value.strip().lower().replace("&", "and")
    value = value.replace("/", " ").replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", value)


def target_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for target in cm.TARGET_CLASSES:
        lookup[normalize(target)] = target
        for alias in ALIASES.get(target, ()):
            lookup[normalize(alias)] = target
    return lookup


def download_file(url: str, dest: Path, label: str) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with part.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    part.replace(dest)
    print(f"downloaded {label}: {dest} ({dest.stat().st_size} bytes)", flush=True)


def load_oi_mapping(classes_csv: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    mid_to_target: dict[str, str] = {}
    resolved: dict[str, list[str]] = {name: [] for name in cm.TARGET_CLASSES}
    lookup = target_lookup()
    with classes_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            target = lookup.get(normalize(row["DisplayName"]))
            if target is not None:
                mid_to_target[row["LabelName"]] = target
                resolved[target].append(row["DisplayName"].strip())
    return mid_to_target, resolved


def oi_box(xmin: str, xmax: str, ymin: str, ymax: str) -> tuple[float, float, float, float] | None:
    x0, x1 = max(0.0, float(xmin)), min(1.0, float(xmax))
    y0, y1 = max(0.0, float(ymin)), min(1.0, float(ymax))
    if x1 <= x0 or y1 <= y0:
        return None
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def parse_candidates(boxes_csv: Path, mid_to_target: dict[str, str]) -> dict[str, list[tuple[int, float, float, float, float]]]:
    target_index = cm.TARGET_INDEX
    candidates: dict[str, list[tuple[int, float, float, float, float]]] = {}
    with boxes_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            target = mid_to_target.get(row["LabelName"])
            if target is None:
                continue
            box = oi_box(row["XMin"], row["XMax"], row["YMin"], row["YMax"])
            if box is not None:
                candidates.setdefault(row["ImageID"], []).append(
                    (target_index[target], *box)
                )
    return candidates


def ordered_candidates(
    candidates: dict[str, list[tuple[int, float, float, float, float]]],
    seed: int,
) -> list[str]:
    offered = Counter(
        cm.TARGET_CLASSES[class_id]
        for rows in candidates.values()
        for class_id, *_ in rows
    )
    rng = random.Random(seed)
    tie = list(candidates)
    rng.shuffle(tie)
    rank = {image_id: index for index, image_id in enumerate(tie)}

    def score(image_id: str) -> tuple[float, int]:
        rows = candidates[image_id]
        # Rare mapped classes are deliberately pulled earlier under a cap.
        coverage = sum(1.0 / max(1, offered[cm.TARGET_CLASSES[c]]) for c, *_ in rows)
        return (-coverage, rank[image_id])

    return sorted(candidates, key=score)


def iter_images(dataset: Path) -> Iterable[Path]:
    for split in ("train", "val"):
        yield from sorted((dataset / "images" / split).glob("*.jpg"))


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def copy_base(base: Path, out: Path) -> int:
    count = 0
    for split in ("train", "val"):
        for image in sorted((base / "images" / split).glob("*.jpg")):
            label = base / "labels" / split / f"{image.stem}.txt"
            if not label.exists():
                continue
            stem = f"base__{image.name}"
            link_or_copy(image, out / "images" / split / stem)
            link_or_copy(label, out / "labels" / split / f"{Path(stem).stem}.txt")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--oi-cap", type=int, default=2500)
    parser.add_argument("--max-downloads", type=int, default=6000)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    if not (args.base / "images").is_dir():
        raise FileNotFoundError(f"base dataset missing: {args.base}")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.out}")
    if not 0.0 <= args.val_frac <= 1.0:
        raise ValueError("--val-frac must be between 0 and 1")

    started = time.time()
    args.cache.mkdir(parents=True, exist_ok=True)
    classes_csv = args.cache / "oidv7-class-descriptions-boxable.csv"
    boxes_csv = args.cache / "validation-annotations-bbox.csv"
    download_file(URL_OI_CLASSES, classes_csv, "Open Images class descriptions")
    download_file(URL_OI_BOXES, boxes_csv, "Open Images validation boxes")

    mid_to_target, resolved = load_oi_mapping(classes_csv)
    candidates = parse_candidates(boxes_csv, mid_to_target)
    order = ordered_candidates(candidates, args.seed)
    offered = Counter(
        cm.TARGET_CLASSES[c]
        for rows in candidates.values()
        for c, *_ in rows
    )
    missing = [name for name in cm.TARGET_CLASSES if not resolved[name]]
    print(f"mapped Open Images labels: {len(mid_to_target)}", flush=True)
    print(f"candidate images: {len(candidates)}; missing classes: {missing}", flush=True)

    args.out.mkdir(parents=True)
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)
    base_count = copy_base(args.base, args.out)

    deduper = mh.Deduper()
    ref_images = mh.seed_deduper_from_dataset(deduper, args.base, prefix="base:")
    print(f"seeded deduplication from {ref_images} base images", flush=True)

    image_cache = args.cache / "oi_validation_images"
    image_cache.mkdir(parents=True, exist_ok=True)
    requested = kept = failed = exact_dups = near_dups = 0
    kept_ids: list[str] = []

    def fetch(image_id: str) -> tuple[str, Path | None]:
        path = image_cache / f"{image_id}.jpg"
        try:
            if not path.exists():
                response = requests.get(URL_OI_IMAGE.format(image_id=image_id), timeout=(15, 60))
                response.raise_for_status()
                path.write_bytes(response.content)
            return image_id, path
        except Exception:
            path.unlink(missing_ok=True)
            return image_id, None

    # Fetch in small waves so a capped build cannot create an uncontrolled
    # disk or connection storm.
    pos = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        while kept < args.oi_cap and requested < args.max_downloads and pos < len(order):
            wave = order[pos:pos + 64]
            pos += len(wave)
            for image_id, path in executor.map(fetch, wave):
                requested += 1
                if path is None:
                    failed += 1
                    continue
                try:
                    sha = mh.sha256_of(path)
                    with path.open("rb") as handle:
                        from PIL import Image
                        with Image.open(handle) as image:
                            dhash = mh.dhash64(image)
                    decision = deduper.add(f"open-images:{image_id}", sha, dhash)
                except Exception:
                    path.unlink(missing_ok=True)
                    failed += 1
                    continue
                if decision.status != "unique":
                    path.unlink(missing_ok=True)
                    if decision.status == "exact_dup":
                        exact_dups += 1
                    else:
                        near_dups += 1
                    continue
                kept_ids.append(image_id)
                kept += 1
                if kept >= args.oi_cap:
                    break
            print(f"Open Images: {kept} kept / {requested} tried / {failed} failed", flush=True)

    rng = random.Random(args.seed)
    kept_boxes = Counter()
    for image_id in kept_ids:
        split = "val" if rng.random() < args.val_frac else "train"
        stem = f"open_images_v5_val__{image_id}"
        link_or_copy(image_cache / f"{image_id}.jpg", args.out / "images" / split / f"{stem}.jpg")
        with (args.out / "labels" / split / f"{stem}.txt").open("w", encoding="utf-8") as handle:
            for class_id, cx, cy, width, height in candidates[image_id]:
                handle.write(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n")
                kept_boxes[cm.TARGET_CLASSES[class_id]] += 1

    (args.out / "data.yaml").write_text(
        f"# {args.out.name} — generated by perception/build_oi_context_dataset.py\n"
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\n\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(cm.TARGET_CLASSES)),
        encoding="utf-8",
    )
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "perception/build_oi_context_dataset.py",
        "args": {key: str(value) for key, value in vars(args).items()},
        "taxonomy": {"class_count": len(cm.TARGET_CLASSES), "classes": list(cm.TARGET_CLASSES),
                     "tableware_prefix": list(cm.TABLE_CLASSES)},
        "base_dataset": str(args.base),
        "base_images_linked": base_count,
        "open_images": {
            "classes_url": URL_OI_CLASSES, "boxes_url": URL_OI_BOXES,
            "image_url": URL_OI_IMAGE, "mapped_labels": resolved,
            "classes_without_mapped_box_label": missing,
            "candidate_images": len(candidates), "class_boxes_offered": dict(offered),
            "images_requested": requested, "images_kept_unique": kept,
            "download_failures": failed, "exact_duplicates_dropped": exact_dups,
            "near_duplicates_dropped": near_dups, "class_boxes_kept": dict(kept_boxes),
        },
        "split": {
            "train": len(list((args.out / "images" / "train").glob("*.jpg"))),
            "val": len(list((args.out / "images" / "val").glob("*.jpg"))),
        },
        "license_note": (
            "Open Images box annotations are CC BY 4.0 per the official dataset "
            "description. Image rights remain source-image specific; do not "
            "redistribute source images without preserving applicable terms."
        ),
        "promotion_boundary": "source augmentation only; requires classwise AP, deploy-view, clutter/safety holdout, and baseline regression",
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        f"# {args.out.name}\n\n"
        "Capped Open Images supplement for the protected 82-class combined YOLO taxonomy.\n\n"
        f"Base images are hardlinked from `{args.base}`; {kept} unique Open Images validation images were added.\n"
        "Open Images labels are partially mapped; see `manifest.json` for resolved and missing classes, source URLs, dedup receipts, and license boundaries.\n",
        encoding="utf-8",
    )
    print(f"built {args.out}: base={base_count}, open_images={kept}, elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
