"""OQ-008 — multi-dataset direct-HTTP hauler for the tabletop detector.

FiftyOne is not installed anywhere (and its Open Images loader needs gcloud
auth), so this pulls the same public data over plain HTTP with no credentials:

  COCO 2017 val        images.cocodataset.org   (boxes: instances_val2017.json)
  LVIS v1 val          dl.fbaipublicfiles.com   (images ARE COCO val images ->
                       matched by image id / filename, annotations merged)
  Open Images v5 val   boxes CSV from storage.googleapis.com/openimages,
                       per-image JPEGs from the CVDF S3 bucket
                       (open-images-dataset.s3.amazonaws.com) — the only Drawer
                       source. v6/v7 validation CSVs on the same bucket are
                       auth-walled (403 at build time); v5 validation is public.
  Objects365           SKIPPED — v1 requires registration; no no-auth URL.

Dedup is a *measured* output (owner requirement), not a side effect:
  exact   sha256 of image bytes
  near    64-bit dHash (PIL grayscale 9x8 resize + numpy diff), Hamming <= 6
  cross   COCO vs LVIS by image id/filename; Open Images vs COCO/LVIS by dHash

Output: a YOLOv5-layout tree (images/ labels/ train+val, normalized
cx cy w h txt), data.yaml, and manifest.json with per-source image counts,
exact/near dup rates, per-class box counts before/after dedup, and a license
note.  data/ is gitignored.

    python perception/multisource_haul.py --out data/table_yolo_v1 \
        --oi-cap 1500 --min-per-class 1500 --val-frac 0.10 --seed 7

Portions derived from *The Hidden Canopy LLC* — [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classmap import TARGET_CLASSES, TARGET_INDEX, remap  # noqa: E402

# ---------------------------------------------------------------------------
# URLs (all verified reachable, no auth, at build time 2026-09-10)
# ---------------------------------------------------------------------------
URL_COCO_VAL_ZIP = "http://images.cocodataset.org/zips/val2017.zip"
URL_COCO_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
URL_LVIS_VAL_ZIP = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"
URL_OI_BOXES_CSV = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
URL_OI_CLASSES_CSV = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
URL_OI_IMAGE = "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"

LICENSE_NOTE = (
    "Annotations: COCO (CC-BY-4.0), LVIS v1 (CC-BY-4.0), Open Images "
    "(CC-BY-4.0). Images retain their original Flickr/owner licenses "
    "(COCO, Open Images) — redistribution of images is NOT granted by the "
    "annotation license; keep this dataset in-house."
)

NEAR_DUP_TOLERANCE = 6  # Hamming bits

# ---------------------------------------------------------------------------
# Pure helpers — the testable core (no network)
# ---------------------------------------------------------------------------

def dhash64(img: Image.Image) -> int:
    """64-bit difference hash: grayscale, 9x8 resize, horizontal diff, bitpack."""
    g = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    diff = g[:, 1:] > g[:, :-1]                       # 8x8 = 64 bools
    bits = np.packbits(diff.reshape(-1))              # 8 bytes, big-endian bits
    return int.from_bytes(bits.tobytes(), "big")


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass
class DupDecision:
    status: str            # "unique" | "exact_dup" | "near_dup"
    matched: str | None = None   # key of the earlier image it duplicates


class Deduper:
    """Exact (sha256) + near (dHash Hamming <= tol) duplicate detector.

    Every accepted image is keyed once; ``add`` returns a decision so the
    caller can count and report duplication instead of silently dropping it.
    """

    def __init__(self, tolerance: int = NEAR_DUP_TOLERANCE) -> None:
        self.tolerance = tolerance
        self._by_sha: dict[str, str] = {}
        self._hashes: list[int] = []
        self._keys: list[str] = []
        self._arr = np.zeros(0, dtype=np.uint64)

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, key: str, sha256: str, dh: int) -> DupDecision:
        if sha256 in self._by_sha:
            return DupDecision("exact_dup", self._by_sha[sha256])
        if len(self._arr):
            dist = np.bitwise_count(self._arr ^ np.uint64(dh))
            hit = int(np.argmin(dist))
            if int(dist[hit]) <= self.tolerance:
                return DupDecision("near_dup", self._keys[hit])
        self._by_sha[sha256] = key
        self._hashes.append(dh)
        self._keys.append(key)
        self._arr = np.asarray(self._hashes, dtype=np.uint64)
        return DupDecision("unique")


YoloRow = tuple[int, float, float, float, float]  # cls cx cy w h (normalized)


def coco_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> YoloRow | None:
    """COCO/LVIS [x, y, w, h] absolute -> (cls-agnostic xywh) normalized; None if degenerate."""
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None
    return (x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h


def oi_bbox_to_yolo(xmin: float, xmax: float, ymin: float, ymax: float) -> YoloRow | None:
    if xmax <= xmin or ymax <= ymin:
        return None
    return (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin


def write_yolo_label(path: Path, rows: list[YoloRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in rows) + "\n",
        encoding="utf-8",
    )


def parse_yolo_label(path: Path) -> list[YoloRow]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        c, cx, cy, w, h = ln.split()
        rows.append((int(c), float(cx), float(cy), float(w), float(h)))
    return rows


# ---------------------------------------------------------------------------
# Annotation parsing (pure: dict/CSV-text in, per-image target rows out)
# ---------------------------------------------------------------------------

def coco_target_rows(instances: dict) -> dict[int, list[YoloRow]]:
    """instances_val2017.json -> {image_id: [(cls, cx, cy, w, h), ...]} target-only."""
    cat_names = {c["id"]: c["name"] for c in instances["categories"]}
    imgs = {im["id"]: (im["width"], im["height"]) for im in instances["images"]}
    out: dict[int, list[YoloRow]] = {}
    for ann in instances["annotations"]:
        tgt = remap("coco-2017", cat_names.get(ann["category_id"], ""))
        if tgt is None or ann["image_id"] not in imgs:
            continue
        wh = coco_bbox_to_yolo(ann["bbox"], *imgs[ann["image_id"]])
        if wh is None:
            continue
        out.setdefault(ann["image_id"], []).append((TARGET_INDEX[tgt], *wh))
    return out


def lvis_target_rows(lvis: dict) -> dict[int, list[YoloRow]]:
    """lvis_v1_val.json -> {image_id: rows}; image ids match COCO val ids."""
    cat_names = {c["id"]: c["name"] for c in lvis["categories"]}
    imgs = {im["id"]: (im["width"], im["height"]) for im in lvis["images"]}
    out: dict[int, list[YoloRow]] = {}
    for ann in lvis["annotations"]:
        tgt = remap("lvis", cat_names.get(ann["category_id"], ""))
        if tgt is None or ann["image_id"] not in imgs:
            continue
        wh = coco_bbox_to_yolo(ann["bbox"], *imgs[ann["image_id"]])
        if wh is None:
            continue
        out.setdefault(ann["image_id"], []).append((TARGET_INDEX[tgt], *wh))
    return out


def oi_mid_to_display(classes_csv_text: str) -> dict[str, str]:
    """oidv7 class-descriptions-boxable.csv -> {MID: display name}."""
    return {r["LabelName"]: r["DisplayName"].strip()
            for r in csv.DictReader(io.StringIO(classes_csv_text))}


def oi_target_rows(boxes_csv_text: str, mid_to_name: dict[str, str]
                   ) -> dict[str, list[YoloRow]]:
    """Open Images validation-annotations-bbox.csv -> {ImageID: rows} target-only.

    Boxes are XMin/XMax/YMin/YMax normalized, so no image dims are needed.
    """
    out: dict[str, list[YoloRow]] = {}
    for r in csv.DictReader(io.StringIO(boxes_csv_text)):
        name = mid_to_name.get(r["LabelName"])
        if name is None:
            continue
        tgt = remap("open-images-v7", name)
        if tgt is None:
            continue
        wh = oi_bbox_to_yolo(float(r["XMin"]), float(r["XMax"]),
                             float(r["YMin"]), float(r["YMax"]))
        if wh is None:
            continue
        out.setdefault(r["ImageID"], []).append((TARGET_INDEX[tgt], *wh))
    return out


def order_oi_candidates(candidates: dict[str, list[YoloRow]],
                        initial_counts: dict[str, int],
                        min_per_class: int,
                        seed: int,
                        batch: int = 200) -> list[str]:
    """Greedy, deterministic ordering: images covering the biggest class
    deficits first, so a capped download still chases the per-class minimum.

    Scores are refreshed every ``batch`` picks (stale within a batch) — a full
    re-score per pick is O(N) and needlessly slow for thousands of candidates.
    """
    per_image: dict[str, dict[str, int]] = {}
    for iid, rows in candidates.items():
        h: dict[str, int] = {}
        for c, *_ in rows:
            h[TARGET_CLASSES[c]] = h.get(TARGET_CLASSES[c], 0) + 1
        per_image[iid] = h

    counts = dict(initial_counts)
    base = sorted(candidates)           # deterministic base order
    random.Random(seed).shuffle(base)   # seeded tie-break below the score
    rank = {iid: n for n, iid in enumerate(base)}
    remaining = set(candidates)
    picked: list[str] = []
    done = False
    while remaining and not done:
        def deficit_cov(iid: str) -> tuple[int, int]:
            cov = 0
            for c, n in per_image[iid].items():
                cov += min(n, max(0, min_per_class - counts[c]))
            return cov, len(candidates[iid])
        batch_ids = sorted(remaining,
                           key=lambda i: (-deficit_cov(i)[0], -deficit_cov(i)[1], rank[i])
                           )[:batch]
        for best in batch_ids:
            if deficit_cov(best)[0] == 0 and picked:
                done = True            # deficits exhausted; rest become extras
                break
            picked.append(best)
            remaining.discard(best)
            for c, n in per_image[best].items():
                counts[c] += n
    picked.extend(sorted(remaining, key=rank.__getitem__))  # zero-cov extras, still usable
    return picked


def histogram(rows_by_image: dict[str, list[YoloRow]]) -> dict[str, int]:
    h = {c: 0 for c in TARGET_CLASSES}
    for rows in rows_by_image.values():
        for c, *_ in rows:
            h[TARGET_CLASSES[c]] += 1
    return h


# ---------------------------------------------------------------------------
# Network (kept thin; everything above is pure)
# ---------------------------------------------------------------------------

def download(url: str, dest: Path, label: str = "", chunk: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    got = 0
    with requests.get(url, stream=True, timeout=(15, 60)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        with dest.open("wb") as f:
            for blk in r.iter_content(chunk):
                f.write(blk)
                got += len(blk)
    dt = time.time() - t0
    if label:
        print(f"  [{label}] {got / 1e6:.0f} MB in {dt:.0f}s "
              f"({got / 1e6 / max(dt, 0.1):.1f} MB/s)"
              + (f" of {total / 1e6:.0f} MB" if total else ""), flush=True)
    return dest


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# The haul
# ---------------------------------------------------------------------------

@dataclass
class KeptImage:
    key: str                 # dataset-wide unique stem
    source: str              # "coco+lvis" | "open-images"
    rows: list[YoloRow] = field(default_factory=list)
    src_path: Path | None = None   # local image file (COCO extract) ...
    oi_id: str | None = None       # ... or Open Images ImageID
    sha256: str = ""
    dhash: int = 0


def _fresh_dir(base: Path) -> Path:
    """Never overwrite: suffix a timestamp if the dir already exists."""
    if not base.exists():
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cand = base.with_name(f"{base.name}_{stamp}")
    n = 1
    while cand.exists():
        cand = base.with_name(f"{base.name}_{stamp}_{n}")
        n += 1
    return cand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/table_yolo_v1"))
    ap.add_argument("--cache", type=Path, default=Path("data/_haul_cache"))
    ap.add_argument("--coco-cap", type=int, default=None,
                    help="max COCO val images to keep (default: all with target boxes)")
    ap.add_argument("--oi-cap", type=int, default=1500,
                    help="max Open Images val images to download (0 = skip OI)")
    ap.add_argument("--min-per-class", type=int, default=1500,
                    help="greedy coverage target for the 6 tableware classes")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--budget-minutes", type=float, default=22.0,
                    help="wall-clock cap on the whole haul's download phase")
    args = ap.parse_args()

    t_start = time.time()
    rng = random.Random(args.seed)
    cache = args.cache
    manifest: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "args": {k: str(v) for k, v in vars(args).items()},
        "sources": {},
        "dedup": {"exact": [], "near": []},
        "license_note": LICENSE_NOTE,
    }

    # ---- 1. COCO val annotations + images --------------------------------
    print("== COCO 2017 val", flush=True)
    ann_zip = cache / "annotations_trainval2017.zip"
    if not ann_zip.exists():
        download(URL_COCO_ANN_ZIP, ann_zip, "coco-annotations")
    with zipfile.ZipFile(ann_zip) as zf:
        instances = json.loads(zf.read("annotations/instances_val2017.json"))
    coco_rows = coco_target_rows(instances)
    coco_ids = sorted(coco_rows)
    if args.coco_cap:
        coco_ids = coco_ids[: args.coco_cap]
    coco_file = {im["id"]: im["file_name"] for im in instances["images"]}
    want_members = {f"val2017/{coco_file[i]}": i for i in coco_ids}
    print(f"  {len(coco_ids)} val images carry target boxes", flush=True)

    val_zip = cache / "val2017.zip"
    if not val_zip.exists():
        download(URL_COCO_VAL_ZIP, val_zip, "coco-val2017")
    img_dir = cache / "coco_val_extract"
    img_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(val_zip) as zf:
        have = {n for n in zf.namelist() if n in want_members}
        for n in sorted(have):
            dst = img_dir / Path(n).name
            if not dst.exists():
                with zf.open(n) as src, dst.open("wb") as out:
                    while True:
                        blk = src.read(1 << 20)
                        if not blk:
                            break
                        out.write(blk)
    missing = [i for m, i in want_members.items() if m not in have]
    coco_ids = [i for i in coco_ids if i not in missing]
    if missing:
        print(f"  WARNING: {len(missing)} annotated images missing from the zip", flush=True)

    # ---- 2. LVIS val annotations (images ARE COCO images -> merge) --------
    # LVIS v1 val has NO file_name field — only coco_url — and its 19,809
    # images span BOTH COCO val2017 and COCO train2017.  Names/splits are
    # derived from coco_url; train2017 images are fetched per-image over HTTP
    # (images.cocodataset.org serves single files, no auth).
    print("== LVIS v1 val", flush=True)
    lvis_zip = cache / "lvis_v1_val.json.zip"
    if not lvis_zip.exists():
        download(URL_LVIS_VAL_ZIP, lvis_zip, "lvis-val")
    with zipfile.ZipFile(lvis_zip) as zf:
        lvis = json.loads(zf.read("lvis_v1_val.json"))
    lvis_rows = lvis_target_rows(lvis)
    lvis_url = {im["id"]: im.get("coco_url", "") for im in lvis["images"]}

    def _lvis_name(i: int) -> str:
        return lvis_url.get(i, "").rstrip("/").rsplit("/", 1)[-1]

    lvis_matched = sorted(i for i in lvis_rows if i in coco_file)
    lvis_added = sorted(i for i in lvis_rows if i not in coco_file)
    lvis_paths: dict[int, Path] = {}

    with zipfile.ZipFile(val_zip) as zf:
        names = {n for n in zf.namelist() if not n.endswith("/")}
        for i in lvis_matched:
            p = img_dir / coco_file[i]
            if p.exists():
                lvis_paths[i] = p
                continue
            member = f"val2017/{coco_file[i]}"
            if member in names:
                with zf.open(member) as src, p.open("wb") as out:
                    out.write(src.read())
                lvis_paths[i] = p
    # LVIS-annotated images beyond COCO's target set: pull from COCO train2017
    # per-image (bounded by the same wall-clock budget).
    from concurrent.futures import ThreadPoolExecutor
    train_dir = cache / "coco_train_pull"
    train_dir.mkdir(parents=True, exist_ok=True)
    budget_s = args.budget_minutes * 60
    todo = [i for i in lvis_added
            if not (train_dir / _lvis_name(i)).exists()][:2000]
    if todo and time.time() - t_start < budget_s:
        print(f"  fetching {len(todo)} LVIS images from COCO train2017 "
              f"(per-image HTTP)", flush=True)

        def _pull(i: int):
            url = lvis_url.get(i, "")
            if not url:
                return i, None
            dst = train_dir / _lvis_name(i)
            try:
                r = requests.get(url, timeout=(10, 60))
                r.raise_for_status()
                dst.write_bytes(r.content)
                return i, dst
            except requests.RequestException:
                dst.unlink(missing_ok=True)
                return i, None

        with ThreadPoolExecutor(8) as ex:
            for n, (i, p) in enumerate(ex.map(_pull, todo)):
                if p is not None:
                    lvis_paths[i] = p
                if n % 200 == 0:
                    print(f"    lvis pull {n}/{len(todo)} "
                          f"({time.time() - t_start:.0f}s)", flush=True)
    n_new = len([i for i in lvis_added if i in lvis_paths])
    print(f"  {len(lvis_matched)} LVIS images share COCO val ids "
          f"(cross-source dedup rate "
          f"{len(lvis_matched) / max(len(lvis_rows), 1):.1%}), "
          f"{n_new} LVIS images beyond COCO's target set fetched", flush=True)

    # ---- 3. Build the COCO+LVIS pool, hashing every kept image ----------
    print("== hashing COCO+LVIS pool", flush=True)
    deduper = Deduper()
    pool: dict[str, KeptImage] = {}
    dup_stats = {"coco": {"exact": 0, "near": 0}, "open-images": {"exact": 0, "near": 0}}
    for i in sorted(coco_ids):
        stem = Path(coco_file.get(i) or lvis_file.get(i, f"{i:012d}.jpg")).stem
        p = img_dir / f"{stem}.jpg"
        if not p.exists():
            continue
        sha = sha256_of(p)
        with Image.open(p) as im:
            dh = dhash64(im)
        dec = deduper.add(f"coco:{stem}", sha, dh)
        if dec.status != "unique":
            dup_stats["coco"][dec.status.split("_")[0]] += 1
            continue
        rows = list(coco_rows.get(i, [])) + list(lvis_rows.get(i, []))
        pool[stem] = KeptImage(key=stem, source="coco+lvis", rows=rows,
                               src_path=p, sha256=sha, dhash=dh)
    counts = histogram({k: v.rows for k, v in pool.items()})
    print(f"  {len(pool)} unique COCO+LVIS images; boxes {counts}", flush=True)
    manifest["sources"]["coco-2017"] = {
        "status": "ok", "split": "val2017",
        "annotations": "instances_val2017.json (annotations_trainval2017.zip)",
        "val_images_total": len(instances["images"]),
        "images_with_target_boxes": len(coco_rows),
        "images_kept_unique": sum(1 for v in pool.values() if v.source == "coco+lvis"),
        "url_images": URL_COCO_VAL_ZIP, "url_annotations": URL_COCO_ANN_ZIP,
    }
    manifest["sources"]["lvis"] = {
        "status": "ok", "version": "v1", "split": "val",
        "url_annotations": URL_LVIS_VAL_ZIP,
        "images_with_target_boxes": len(lvis_rows),
        "images_matched_to_coco": len(lvis_matched),
        "images_added_beyond_coco": len(lvis_new),
        "cross_source_dedup_rate": round(len(lvis_matched) / max(len(lvis_rows), 1), 4),
    }
    manifest["sources"]["objects365"] = {
        "status": "skipped",
        "reason": "v1 requires registration; no public no-auth download URL. "
                  "v2 yaml exists in ultralytics configs but images are behind "
                  "OpenDataLab sign-up. Revisit after registration.",
    }

    # ---- 4. Open Images (only Drawer source) -----------------------------
    oi_manifest: dict = {"status": "skipped"}
    if args.oi_cap > 0 and time.time() - t_start < args.budget_minutes * 60:
        print("== Open Images v5 validation", flush=True)
        try:
            oi_manifest = _haul_open_images(args, cache, pool, deduper,
                                            counts, dup_stats, t_start)
        except Exception as e:  # keep the haul alive; record honestly
            oi_manifest = {"status": "skipped", "reason": f"{type(e).__name__}: {e}"}
            print(f"  OI skipped: {oi_manifest['reason']}", flush=True)
    manifest["sources"]["open-images"] = oi_manifest

    # ---- 5. Persist -------------------------------------------------------
    out = _fresh_dir(args.out)
    n_train = n_val = 0
    idx = 0
    per_image: dict[str, dict] = {}
    for stem in sorted(pool):
        k = pool[stem]
        if not k.rows:
            continue
        split = "val" if rng.random() < args.val_frac else "train"
        out_img = out / "images" / split / f"{split}_{idx:07d}.jpg"
        out_lbl = out / "labels" / split / f"{split}_{idx:07d}.txt"
        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        if k.src_path is not None:
            try:
                out_img.hardlink_to(k.src_path)
            except OSError:
                out_img.write_bytes(k.src_path.read_bytes())
        write_yolo_label(out_lbl, k.rows)
        per_image[out_img.name] = {"source": k.source, "sha256": k.sha256}
        n_train += split == "train"
        n_val += split == "val"
        idx += 1

    (out / "data.yaml").write_text(
        "# OQ-008 table_yolo_v1 — generated by perception/multisource_haul.py\n"
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(TARGET_CLASSES)),
        encoding="utf-8",
    )

    manifest["dedup"] = {
        "method_exact": "sha256 of image bytes",
        "method_near": f"64-bit dHash, Hamming <= {NEAR_DUP_TOLERANCE}",
        "within_source": dup_stats,
        "cross_source": {
            "lvis_vs_coco": {
                "matched_by_image_id": len(lvis_matched),
                "rate": round(len(lvis_matched) / max(len(lvis_rows), 1), 4),
            },
            "openimages_vs_cocolvis": {
                "exact_dups": dup_stats["open-images"]["exact"],
                "near_dups": dup_stats["open-images"]["near"],
            },
        },
    }
    manifest["output"] = str(out)
    manifest["class_boxes_offered"] = {
        "coco-2017": histogram(coco_rows),
        "lvis": histogram(lvis_rows),
        "open-images": oi_manifest.get("class_boxes_offered", {}),
    }
    manifest["final_unique_images"] = len(per_image)
    manifest["split"] = {"train": n_train, "val": n_val}
    manifest["class_boxes_final"] = histogram({k: v.rows for k, v in pool.items() if v.rows})
    manifest["source_per_image"] = per_image
    manifest["elapsed_seconds"] = round(time.time() - t_start, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\ndataset at {out}")
    print(f"unique images: {len(per_image)}  (train {n_train} / val {n_val})")
    print(f"final class boxes: {manifest['class_boxes_final']}")
    thin = [c for c in TARGET_CLASSES[:-1] if manifest["class_boxes_final"][c] < args.min_per_class]
    if thin:
        print(f"below the {args.min_per_class}/class target: {thin} — scale-up path is in the receipt")


def _haul_open_images(args, cache: Path, pool: dict[str, KeptImage],
                      deduper: Deduper, counts: dict[str, int],
                      dup_stats: dict, t_start: float) -> dict:
    boxes_csv = cache / "validation-annotations-bbox.csv"
    if not boxes_csv.exists():
        download(URL_OI_BOXES_CSV, boxes_csv, "oi-boxes")
    classes_csv = cache / "oidv7-class-descriptions-boxable.csv"
    if not classes_csv.exists():
        download(URL_OI_CLASSES_CSV, classes_csv, "oi-classes")
    candidates = oi_target_rows(boxes_csv.read_text(encoding="utf-8"),
                                oi_mid_to_display(classes_csv.read_text(encoding="utf-8")))
    order = order_oi_candidates(candidates, counts, args.min_per_class, args.seed)
    print(f"  {len(candidates)} OI val images carry target boxes; "
          f"greedy order computed for cap {args.oi_cap}", flush=True)

    oi_dir = cache / "oi_images"
    oi_dir.mkdir(parents=True, exist_ok=True)
    kept = tried = exact = near = 0
    budget_s = args.budget_minutes * 60

    def _fetch_one(iid: str):
        dst = oi_dir / f"{iid}.jpg"
        try:
            if not dst.exists():
                r = requests.get(URL_OI_IMAGE.format(image_id=iid), timeout=(10, 60))
                r.raise_for_status()
                dst.write_bytes(r.content)
            sha = sha256_of(dst)
            with Image.open(dst) as im:
                dh = dhash64(im)
            return iid, sha, dh
        except Exception:
            dst.unlink(missing_ok=True)
            return iid, None, None

    from concurrent.futures import ThreadPoolExecutor
    WAVE = 96            # fetch in waves so the budget check stays responsive
    WORKERS = 8
    pos = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        while pos < len(order) and kept < args.oi_cap \
                and time.time() - t_start <= budget_s:
            wave = order[pos: pos + WAVE]
            pos += len(wave)
            for iid, sha, dh in ex.map(_fetch_one, wave):
                tried += 1
                if sha is None:
                    continue
                dec = deduper.add(f"oi:{iid}", sha, dh)
                if dec.status == "exact_dup":
                    exact += 1
                    dup_stats["open-images"]["exact"] += 1
                    (oi_dir / f"{iid}.jpg").unlink()
                    continue
                if dec.status == "near_dup":
                    near += 1
                    dup_stats["open-images"]["near"] += 1
                    (oi_dir / f"{iid}.jpg").unlink()
                    continue
                kept += 1
                pool[iid] = KeptImage(key=iid, source="open-images",
                                      rows=list(candidates[iid]),
                                      src_path=oi_dir / f"{iid}.jpg",
                                      oi_id=iid, sha256=sha, dhash=dh)
                for c, *_ in candidates[iid]:
                    counts[TARGET_CLASSES[c]] += 1
            print(f"  OI: {kept} kept / {tried} tried "
                  f"({time.time() - t_start:.0f}s elapsed)", flush=True)
    return {
        "status": "ok",
        "boxes_csv": URL_OI_BOXES_CSV,
        "images_csv": URL_OI_CLASSES_CSV,
        "image_base": URL_OI_IMAGE,
        "note": "v6/v7 validation CSVs on storage.googleapis.com/openimages "
                "return 403 without auth; v5 validation CSV is public and its "
                "ImageIDs match the CVDF S3 validation bucket.",
        "candidate_images_with_target_boxes": len(candidates),
        "class_boxes_offered": histogram(candidates),
        "images_requested": tried,
        "images_kept_unique": kept,
        "exact_dups_vs_existing": exact,
        "near_dups_vs_existing": near,
        "stopped_by_budget": time.time() - t_start > budget_s and kept < len(order),
    }


if __name__ == "__main__":
    main()
