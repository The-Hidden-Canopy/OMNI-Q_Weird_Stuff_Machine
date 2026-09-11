"""Pseudo-label HF kitchen-utensil image dumps with our own fine-tuned detector.

Self-training component (table_yolo_v3_hf): two MIT-licensed HuggingFace image
dumps — ``lrad3/kitchen_utensils_13k`` (13.4k images, no annotations) and
``rypow/kitchen_utensils_5k`` (5k images, classification labels only) — are
labeled by inference with our own ftv2 detector
(``models/table_yolo_v2_ft_2026-09-11.pt``, the 7-class plate/cup/fork/spoon/
knife/napkin/drawer model).  Both sets are fork/knife/spoon-heavy, exactly our
weakest classes.

IMPORTANT: every box in the output is PSEUDO-LABELED by our own model at
conf >= 0.5 — NOT human-annotated.  Treat it as noisy self-training signal.

Flow (download first, then inference, so a mid-way failure keeps the HF cache):

  1. ``huggingface_hub.snapshot_download(repo_id=..., repo_type="dataset",
     allow_patterns=["*.jpg"])`` into ``--work-dir/<repo_slug>/`` for BOTH
     repos (snapshot_download caches, reruns are cheap).
  2. YOLO predict over every image (CPU, stream=True, imgsz=640).  Images with
     >=1 detection: normalized-xywh YOLO label txt (class ids in OUR 7-class
     order) + the image copied into the pool.  Zero-detection images are
     skipped and counted.
  3. dHash-dedup (sha256 exact + 64-bit dHash near, Hamming <= 6 — the
     multisource_haul helpers) against data/table_yolo, data/table_yolo_v2 and
     data/table_yolo_v3_oi (when it exists), plus within-set dedup.  Drops are
     counted per reference set.
  4. Deterministic train/val split (--val-frac 0.10 --seed 13) and a YOLOv5
     layout tree at --out with data.yaml (ABSOLUTE path), manifest.json, and an
     evidence receipt under evidence/datasets/.

    python perception/pseudo_label.py \
        --weights models/table_yolo_v2_ft_2026-09-11.pt \
        --out data/table_yolo_v3_hf --work-dir data/_hf_pseudo

Portions derived from *The Hidden Canopy LLC* — [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classmap import TARGET_CLASSES, TARGET_INDEX  # noqa: E402
from multisource_haul import (  # noqa: E402
    NEAR_DUP_TOLERANCE,
    Deduper,
    YoloRow,
    dhash64,
    histogram,
    seed_deduper_from_dataset,
    sha256_of,
    write_yolo_label,
)

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

SOURCE_REPOS: tuple[dict, ...] = (
    {"repo_id": "lrad3/kitchen_utensils_13k", "license": "MIT",
     "annotations": "none (images only)"},
    {"repo_id": "rypow/kitchen_utensils_5k", "license": "MIT",
     "annotations": "classification labels only (unused here)"},
)

DEFAULT_DEDUP_DIRS = ("data/table_yolo", "data/table_yolo_v2",
                      "data/table_yolo_v3_oi")

WEIGHTS_PROVENANCE = (
    "models/table_yolo_v2_ft_2026-09-11.pt — our own YOLOv8n fine-tune on the "
    "table_yolo_v2 haul (OQ-008, perception/finetune.py); 7-class vocabulary "
    "plate/cup/fork/spoon/knife/napkin/drawer. Boxes emitted here are "
    "PSEUDO-LABELS produced BY this model (self-training), conf>=0.5, NOT "
    "human-annotated."
)

LICENSE_NOTE = (
    "Source image dumps: lrad3/kitchen_utensils_13k and rypow/kitchen_utensils_5k, "
    "both MIT-licensed on HuggingFace (verified at download). Labels are OUR "
    "OWN model's pseudo-labels, not the repos' annotations."
)


def repo_slug(repo_id: str) -> str:
    """'lrad3/kitchen_utensils_13k' -> 'lrad3__kitchen_utensils_13k' (dir-safe)."""
    return repo_id.replace("/", "__")


# ---------------------------------------------------------------------------
# Pure helpers — the testable core (no network, no weights)
# ---------------------------------------------------------------------------

def our_class_ids(names) -> list[int]:
    """Model class-name table -> our 7-class index per model class id.

    ``names`` is ultralytics' {id: name} dict (or a list).  Every name must be
    one of our TARGET_CLASSES — the ftv2 head was trained on exactly that
    vocabulary, so a mismatch means the wrong weights were passed.
    """
    seq = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    unknown = [n for n in seq if n not in TARGET_INDEX]
    if unknown:
        raise ValueError(f"weights classes {unknown} are outside our 7-class "
                         f"vocabulary {TARGET_CLASSES} — wrong weights?")
    return [TARGET_INDEX[n] for n in seq]


def rows_from_result(result, cls_map: list[int]) -> list[YoloRow]:
    """One ultralytics Results -> [(our_cls, cx, cy, w, h)] normalized; [] when no boxes.

    ``cls_map`` maps the MODEL's class ids to OUR class ids (identity for the
    ftv2 weights, but validated, never assumed).  Boxes are clamped to the
    image before normalization so pseudo-labels stay in [0, 1].
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    h, w = result.orig_shape
    if h <= 0 or w <= 0:
        return []
    xyxy = np.asarray(boxes.xyxy, dtype=float).reshape(-1, 4)
    cls = np.asarray(boxes.cls, dtype=int).reshape(-1)
    rows: list[YoloRow] = []
    for (x1, y1, x2, y2), c in zip(xyxy, cls):
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append((cls_map[int(c)], (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                     (x2 - x1) / w, (y2 - y1) / h))
    return rows


def assign_splits(keys: list[str], val_frac: float, seed: int) -> dict[str, str]:
    """Deterministic train/val assignment: sorted keys, one seeded RNG stream."""
    rng = random.Random(seed)
    return {k: ("val" if rng.random() < val_frac else "train") for k in sorted(keys)}


@dataclass
class KeptImage:
    key: str                 # "<slug>/<filename>"
    source: str              # repo slug
    rows: list[YoloRow] = field(default_factory=list)
    src_path: Path | None = None
    sha256: str = ""
    dhash: int = 0


def collect_pool(images: list[tuple[str, Path]], predict_rows, deduper: Deduper,
                 ref_prefixes: list[str], on_progress=None,
                 progress_every: int = 500) -> tuple[dict[str, KeptImage], dict]:
    """Run ``predict_rows(path) -> list[YoloRow]`` over every image; hash + dedup.

    Images with zero detections are skipped (counted, never hashed).  Dedup is
    against whatever the caller pre-seeded ``deduper`` with (cross-set refs,
    prefix-matched) plus within-set accumulation.  Returns (pool, stats).
    """
    pool: dict[str, KeptImage] = {}
    stats = {
        "images_seen": 0,
        "images_no_detections": 0,
        "cross_set_drops": {p: 0 for p in ref_prefixes},
        "within_set_dups": {"exact": 0, "near": 0},
        "unreadable": 0,
    }
    for n, (key, path) in enumerate(images):
        stats["images_seen"] = n + 1
        try:
            rows = predict_rows(path)
        except Exception:
            rows = []
        if not rows:
            stats["images_no_detections"] += 1
            if on_progress and (n + 1) % progress_every == 0:
                on_progress(n + 1, len(images), stats)
            continue
        try:
            sha = sha256_of(path)
            with Image.open(path) as im:
                dh = dhash64(im)
        except Exception:
            stats["unreadable"] += 1
            continue
        dec = deduper.add(key, sha, dh)
        if dec.status != "unique":
            hit = next((p for p in ref_prefixes
                        if dec.matched and dec.matched.startswith(p)), None)
            if hit is not None:
                stats["cross_set_drops"][hit] += 1
            else:
                stats["within_set_dups"][dec.status.split("_")[0]] += 1
            if on_progress and (n + 1) % progress_every == 0:
                on_progress(n + 1, len(images), stats)
            continue
        source = key.split("/", 1)[0]
        pool[key] = KeptImage(key=key, source=source, rows=rows,
                              src_path=path, sha256=sha, dhash=dh)
        if on_progress and (n + 1) % progress_every == 0:
            on_progress(n + 1, len(images), stats)
    return pool, stats


def persist(pool: dict[str, KeptImage], out: Path, val_frac: float, seed: int,
            manifest: dict, t_start: float) -> Path:
    """Write the YOLO tree + data.yaml (ABSOLUTE path) + manifest.json."""
    out.mkdir(parents=True, exist_ok=True)
    splits = assign_splits([k for k, v in pool.items() if v.rows], val_frac, seed)
    n_train = n_val = idx = 0
    per_image: dict[str, dict] = {}
    for key in sorted(splits, key=lambda k: (splits[k], k)):
        k = pool[key]
        split = splits[key]
        out_img = out / "images" / split / f"{split}_{idx:07d}.jpg"
        out_lbl = out / "labels" / split / f"{split}_{idx:07d}.txt"
        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        if k.src_path is not None:
            try:
                out_img.hardlink_to(k.src_path)
            except OSError:
                shutil.copyfile(k.src_path, out_img)
        write_yolo_label(out_lbl, k.rows)
        per_image[out_img.name] = {"source": k.source, "key": k.key,
                                   "sha256": k.sha256}
        n_train += split == "train"
        n_val += split == "val"
        idx += 1

    (out / "data.yaml").write_text(
        f"# OQ-008 {out.name} — PSEUDO-LABELED by perception/pseudo_label.py\n"
        f"path: {out.resolve().as_posix()}  # absolute: pin to this repo "
        "(env ultralytics settings point outside)\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(TARGET_CLASSES)),
        encoding="utf-8",
    )

    manifest["output"] = str(out)
    manifest["final_unique_images"] = len(per_image)
    manifest["split"] = {"train": n_train, "val": n_val}
    manifest["class_boxes_final"] = histogram(
        {k: v.rows for k, v in pool.items() if v.rows})
    manifest["source_per_image"] = per_image
    manifest["elapsed_seconds"] = round(time.time() - t_start, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")

    print(f"\ndataset at {out}")
    print(f"unique images: {len(per_image)}  (train {n_train} / val {n_val})")
    print(f"final class boxes: {manifest['class_boxes_final']}", flush=True)
    return out


def write_receipt(out: Path, manifest: dict, receipt_dir: Path) -> Path | None:
    """Evidence receipt: README + manifest copy, stating PSEUDO-LABELED loudly."""
    ev = Path("evidence/datasets")
    if not ev.is_dir():
        return None
    dest = ev / receipt_dir.name
    n = 1
    while dest.exists():
        dest = ev / f"{receipt_dir.name}_{n}"
        n += 1
    dest.mkdir(parents=True)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                        encoding="utf-8")

    final = manifest.get("class_boxes_final", {})
    split = manifest.get("split", {})
    srcs = manifest.get("sources", {})
    dedup = manifest.get("dedup", {})
    src_lines = "\n".join(
        f"- `{s}` ({srcs[s].get('license')}): {srcs[s].get('images_downloaded')} "
        f"images downloaded, {srcs[s].get('images_with_detections')} with "
        f"detections kept pre-dedup"
        for s in srcs)
    refs_md = "\n".join(
        f"- `{r.get('dir')}` (prefix `{r.get('prefix')}`): {r.get('images')} "
        f"reference images hashed, **{r.get('dropped')} v3_hf candidates dropped**"
        for r in dedup.get("cross_set", {}).get("references", [])) or "- (none)"

    md = f"""# {Path(manifest.get('output', 'table_yolo_v3_hf')).name} — HF pseudo-labeled component ({manifest.get('receipt_stamp')})

**Status: pseudo-labeling complete.  THIS COMPONENT IS PSEUDO-LABELED, NOT
HUMAN-ANNOTATED.**  Every bounding box was emitted by OUR OWN fine-tuned
detector (`{manifest.get('weights')}`, conf >= {manifest.get('conf_threshold')})
running inference over two MIT-licensed HuggingFace image dumps — classic
self-training.  Expect label noise; treat as training signal only, never as
ground truth.

## What was pulled

{src_lines}
- Download date (UTC): {manifest.get('created_utc')}
- Detector: {manifest.get('weights_provenance')}

## Dedup (measured, owner requirement)

- Exact: sha256 of image bytes.  Near: 64-bit dHash, Hamming <= {NEAR_DUP_TOLERANCE}
  (same helpers as multisource_haul).
- Within-set dups: {dedup.get('within_source', {})}
- Cross-set: **{dedup.get('cross_set', {}).get('dropped_total', 0)} of
  {dedup.get('cross_set', {}).get('candidates', 0)} detected candidates dropped**:
{refs_md}

## Final numbers

- **{manifest.get('final_unique_images')} unique images** — train {split.get('train')}
  / val {split.get('val')} (val-frac {manifest.get('val_frac')}, seed {manifest.get('seed')}).
- Per-class KEPT boxes: {" · ".join(f"{c} {n}" for c, n in final.items())}.
- Skipped (no detections >= conf): {manifest.get('images_no_detections')} of
  {manifest.get('images_seen')} seen; unreadable: {manifest.get('unreadable')}.
- Elapsed: {manifest.get('elapsed_seconds')} s.

## License

{manifest.get('license_note')}

Portions derived from *The Hidden Canopy LLC* — [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
"""
    (dest / "README.md").write_text(md, encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# Network + inference (kept thin; everything above is pure)
# ---------------------------------------------------------------------------

def download_repo(repo_id: str, work_dir: Path, allow_patterns: list[str]
                  ) -> tuple[Path, int]:
    """snapshot_download one HF dataset repo into work_dir/<slug>; returns (path, n jpg)."""
    from huggingface_hub import snapshot_download

    slug = repo_slug(repo_id)
    dest = work_dir / slug
    dest.mkdir(parents=True, exist_ok=True)
    print(f"== snapshot_download {repo_id} -> {dest}", flush=True)
    snapshot_download(repo_id=repo_id, repo_type="dataset",
                      allow_patterns=allow_patterns,
                      local_dir=str(dest), local_dir_use_symlinks=False)
    n = sum(1 for p in dest.rglob("*") if p.suffix.lower() == ".jpg")
    print(f"  {n} jpg files present under {dest}", flush=True)
    return dest, n


def _load_model(weights: Path):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    cls_map = our_class_ids(model.names)
    print(f"weights classes: {model.names} -> ours {cls_map}", flush=True)
    return model, cls_map


def _predict_rows_factory(model, cls_map: list[int], conf: float):
    """Single-image predict wrapper so collect_pool stays batch-agnostic."""
    def predict_rows(path: Path) -> list[YoloRow]:
        results = model.predict(str(path), stream=True, conf=conf, imgsz=640,
                                device="cpu", verbose=False)
        for r in results:           # stream yields exactly one result for one image
            return rows_from_result(r, cls_map)
        return []
    return predict_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path,
                    default=Path("models/table_yolo_v2_ft_2026-09-11.pt"))
    ap.add_argument("--conf", type=float, default=0.5,
                    help="confidence floor for kept pseudo-labels (default 0.5)")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, default=Path("data/table_yolo_v3_hf"))
    ap.add_argument("--work-dir", type=Path, default=Path("data/_hf_pseudo"),
                    help="snapshot_download cache root (reruns are cheap)")
    ap.add_argument("--dedup-vs", action="append", default=None,
                    help="existing YOLO dataset dirs to dedup against "
                         "(repeatable/comma; default: the three known sets, "
                         "existing ones only)")
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")

    t_start = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    manifest: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "receipt_stamp": stamp,
        "component": "table_yolo_v3_hf",
        "pseudo_labeled": True,
        "weights": str(args.weights),
        "weights_provenance": WEIGHTS_PROVENANCE,
        "conf_threshold": args.conf,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "license_note": LICENSE_NOTE,
        "args": {k: str(v) for k, v in vars(args).items()},
        "sources": {},
    }

    # ---- 1. Download BOTH repos (cached; a later failure keeps this) --------
    total_imgs = 0
    for spec in SOURCE_REPOS:
        dest, n = download_repo(spec["repo_id"], args.work_dir, ["*.jpg"])
        manifest["sources"][spec["repo_id"]] = {
            **spec, "images_downloaded": n, "local_dir": str(dest),
        }
        total_imgs += n
    print(f"total source images: {total_imgs}", flush=True)

    # ---- 2. Seed cross-set deduper from existing sets -----------------------
    if args.dedup_vs is None:
        refs = [Path(d) for d in DEFAULT_DEDUP_DIRS if (Path(d) / "images").is_dir()]
    else:
        from multisource_haul import parse_dedup_vs
        refs = parse_dedup_vs(args.dedup_vs)
    deduper = Deduper()
    ref_stats: list[dict] = []
    for ref in refs:
        prefix = f"{ref.name}:"
        print(f"seeding reference hashes from {ref} (prefix {prefix})", flush=True)
        n_ref = seed_deduper_from_dataset(deduper, ref, prefix=prefix)
        print(f"  {n_ref} reference images hashed", flush=True)
        ref_stats.append({"dir": str(ref), "prefix": prefix, "images": n_ref,
                          "dropped": 0})
    ref_prefixes = [s["prefix"] for s in ref_stats]

    # ---- 3. Inference over every image --------------------------------------
    model, cls_map = _load_model(args.weights)
    predict_rows = _predict_rows_factory(model, cls_map, args.conf)

    images: list[tuple[str, Path]] = []
    for spec in SOURCE_REPOS:
        slug = repo_slug(spec["repo_id"])
        for p in sorted((args.work_dir / slug).rglob("*.jpg")):
            images.append((f"{slug}/{p.name}", p))
    print(f"running inference over {len(images)} images (cpu, conf>={args.conf})",
          flush=True)

    def _progress(done: int, total: int, stats: dict):
        print(f"  {done}/{total} seen, {stats['images_no_detections']} no-det, "
              f"{stats['unreadable']} unreadable "
              f"({time.time() - t_start:.0f}s)", flush=True)

    pool, stats = collect_pool(images, predict_rows, deduper, ref_prefixes,
                               on_progress=_progress)
    for s in ref_stats:
        s["dropped"] = stats["cross_set_drops"].get(s["prefix"], 0)
    n_dets = sum(len(v.rows) for v in pool.values())
    print(f"inference done: {n_dets} boxes on {len(pool)} images; "
          f"{stats['images_no_detections']} skipped (no detections)", flush=True)
    for spec in SOURCE_REPOS:
        slug = repo_slug(spec["repo_id"])
        manifest["sources"][spec["repo_id"]]["images_with_detections"] = sum(
            1 for v in pool.values() if v.source == slug)
    manifest["images_seen"] = stats["images_seen"]
    manifest["images_no_detections"] = stats["images_no_detections"]
    manifest["unreadable"] = stats["unreadable"]
    manifest["boxes_pseudo_labeled"] = n_dets
    manifest["dedup"] = {
        "method_exact": "sha256 of image bytes",
        "method_near": f"64-bit dHash, Hamming <= {NEAR_DUP_TOLERANCE}",
        "within_source": {"v3_hf": stats["within_set_dups"]},
        "cross_set": {
            "references": ref_stats,
            "candidates": len(pool) + sum(stats["cross_set_drops"].values()),
            "dropped_total": sum(stats["cross_set_drops"].values()),
        },
    }
    manifest["class_boxes_offered"] = {
        slug: histogram({k: v.rows for k, v in pool.items() if v.source == slug})
        for slug in (repo_slug(s["repo_id"]) for s in SOURCE_REPOS)
    }

    # ---- 4. Persist + receipt -----------------------------------------------
    out = persist(pool, args.out, args.val_frac, args.seed, manifest, t_start)
    receipt = write_receipt(out, manifest, Path(f"table_yolo_v3_hf_{stamp}"))
    if receipt:
        print(f"receipt at {receipt}", flush=True)


if __name__ == "__main__":
    main()
