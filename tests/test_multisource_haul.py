"""OQ-008 — multisource_haul pure core: dHash dedup, remap integration, YOLO writer.

No network. Synthetic images only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception"))

import classmap as cm  # noqa: E402
import multisource_haul as mh  # noqa: E402


def _img(w: int = 64, h: int = 64, seed: int = 0, fmt: str = "RGB") -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), fmt)


# ---------------------------------------------------------------------------
# dHash / Hamming
# ---------------------------------------------------------------------------

def test_dhash_is_64bit_and_deterministic():
    im = _img(seed=1)
    a, b = mh.dhash64(im), mh.dhash64(im.copy())
    assert 0 <= a < 2**64 and a == b


def test_dhash_ignores_color_channel_shifts_but_not_layout():
    rng = np.random.default_rng(3)
    base = rng.integers(0, 200, (32, 32), dtype=np.uint8)
    im1 = Image.fromarray(np.stack([base, base, base], -1))
    im2 = Image.fromarray(np.stack([base, (base + 10) % 255, base], -1))
    assert mh.dhash64(im1) == mh.dhash64(im2)  # flat color offset, same gradients
    im3 = Image.fromarray(np.ascontiguousarray(base[::-1, :]))  # vertical flip
    assert mh.hamming(mh.dhash64(im1), mh.dhash64(im3)) > 6


def test_hamming():
    assert mh.hamming(0, 0) == 0
    assert mh.hamming(0b1010, 0b0101) == 4
    assert mh.hamming(2**64 - 1, 0) == 64


# ---------------------------------------------------------------------------
# Deduper: exact sha + near dHash
# ---------------------------------------------------------------------------

def _add_img(d: mh.Deduper, key: str, im: Image.Image, tweak: bytes = b""):
    from io import BytesIO
    buf = BytesIO()
    im.save(buf, "JPEG")
    sha = mh.hashlib.sha256(buf.getvalue() + tweak).hexdigest()
    return d.add(key, sha, mh.dhash64(im))


def test_exact_dup_by_sha():
    d = mh.Deduper()
    im = _img(seed=5)
    assert _add_img(d, "a", im).status == "unique"
    dec = _add_img(d, "b", im)  # same bytes -> same sha even if hash computed fresh
    assert dec.status == "exact_dup" and dec.matched == "a"


def test_near_dup_by_dhash_within_tolerance():
    d = mh.Deduper(tolerance=6)
    im = _img(seed=7)
    assert _add_img(d, "a", im).status == "unique"
    # slightly brightened version: tiny JPEG re-encode, dHash barely moves,
    # but bytes differ -> not exact
    im2 = Image.eval(im, lambda px: min(px + 2, 255))
    dec = _add_img(d, "b", im2)
    if dec.status == "exact_dup":
        pytest.skip("re-encode happened to be byte-identical; environment quirk")
    assert dec.status == "near_dup", f"hamming={mh.hamming(mh.dhash64(im), mh.dhash64(im2))}"


def test_distinct_images_stay_unique():
    d = mh.Deduper()
    a, b = _img(seed=11), _img(seed=22)
    assert _add_img(d, "a", a).status == "unique"
    assert _add_img(d, "b", b).status == "unique"
    assert len(d) == 2


# ---------------------------------------------------------------------------
# remap integration: COCO / LVIS / OI parsing feeds the 7-class vocab
# ---------------------------------------------------------------------------

def _fake_coco():
    return {
        "categories": [
            {"id": 1, "name": "cup"}, {"id": 2, "name": "fork"},
            {"id": 3, "name": "knife"}, {"id": 4, "name": "bowl"},  # bowl -> dropped
        ],
        "images": [{"id": 10, "width": 400, "height": 200}],
        "annotations": [
            {"image_id": 10, "category_id": 1, "bbox": [100, 50, 200, 100]},
            {"image_id": 10, "category_id": 2, "bbox": [0, 0, 40, 20]},
            {"image_id": 10, "category_id": 4, "bbox": [10, 10, 50, 50]},
            {"image_id": 10, "category_id": 3, "bbox": [0, 0, 0, 10]},  # degenerate
        ],
    }


def test_coco_rows_use_classmap_and_drop_unmapped():
    rows = mh.coco_target_rows(_fake_coco())[10]
    classes = sorted(c for c, *_ in rows)
    assert classes == [cm.TARGET_INDEX["cup"], cm.TARGET_INDEX["fork"]]
    c, cx, cy, w, h = rows[0]
    assert cx == pytest.approx(0.5) and cy == pytest.approx(0.5)
    assert w == pytest.approx(0.5) and h == pytest.approx(0.5)


def test_lvis_rows_plate_family_collapses():
    lvis = {
        "categories": [{"id": 1, "name": "plate"}, {"id": 2, "name": "saucer"},
                       {"id": 3, "name": "place_mat"}],
        "images": [{"id": 5, "width": 100, "height": 100}],
        "annotations": [
            {"image_id": 5, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"image_id": 5, "category_id": 2, "bbox": [10, 10, 10, 10]},
            {"image_id": 5, "category_id": 3, "bbox": [20, 20, 10, 10]},
        ],
    }
    rows = mh.lvis_target_rows(lvis)[5]
    assert [c for c, *_ in rows] == [cm.TARGET_INDEX["plate"]] * 2


def test_oi_rows_use_display_names_via_mid_map():
    classes_csv = "LabelName,DisplayName\n/m/a,Plate\n/m/b,Kitchen knife\n/m/c,Knife\n"
    boxes_csv = (
        "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax\n"
        "img1,x,/m/a,1,0.1,0.3,0.2,0.4\n"
        "img1,x,/m/b,1,0.0,0.5,0.0,0.5\n"
        "img1,x,/m/c,1,0.0,0.1,0.0,0.1\n"   # Knife -> dropped (weapon class)
        "img2,x,/m/a,1,0.9,0.1,0.0,0.2\n"   # degenerate (xmax<xmin)
    )
    mid = mh.oi_mid_to_display(classes_csv)
    out = mh.oi_target_rows(boxes_csv, mid)
    assert set(out) == {"img1"}
    classes = sorted(c for c, *_ in out["img1"])
    assert classes == [cm.TARGET_INDEX["plate"], cm.TARGET_INDEX["knife"]]
    c, cx, cy, w, h = out["img1"][0]
    assert cx == pytest.approx(0.2) and cy == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# greedy OI ordering chases per-class minimums, deterministically
# ---------------------------------------------------------------------------

def test_order_oi_candidates_prioritizes_thin_classes_and_is_deterministic():
    P, F = cm.TARGET_INDEX["plate"], cm.TARGET_INDEX["fork"]
    candidates = {
        "a": [(P, 0.5, 0.5, 0.1, 0.1)],
        "b": [(F, 0.5, 0.5, 0.1, 0.1)] * 3,
        "c": [(F, 0.5, 0.5, 0.1, 0.1)] * 2 + [(P, 0.5, 0.5, 0.1, 0.1)],
    }
    initial = {c: 0 for c in cm.TARGET_CLASSES}
    initial["fork"] = 1500            # fork already at target -> only plate counts
    o1 = mh.order_oi_candidates(candidates, initial, 1500, seed=7)
    o2 = mh.order_oi_candidates(candidates, initial, 1500, seed=7)
    assert o1 == o2
    assert o1[0] in ("a", "c")        # images with plate first, not pure-fork "b"
    assert o1[-1] == "b" or "b" in o1


# ---------------------------------------------------------------------------
# YOLO label writer round-trip
# ---------------------------------------------------------------------------

def test_write_and_parse_yolo_label(tmp_path: Path):
    rows = [(0, 0.5, 0.25, 0.1, 0.2), (6, 1.0, 0.0, 0.05, 0.05)]
    p = tmp_path / "labels" / "train_0000000.txt"
    mh.write_yolo_label(p, rows)
    assert mh.parse_yolo_label(p) == rows
    text = p.read_text(encoding="utf-8")
    assert text.endswith("\n") and len(text.splitlines()) == 2


def test_histogram_counts_target_classes():
    rows = {f"im{i}": [(0, 0.5, 0.5, 0.1, 0.1), (6, 0.5, 0.5, 0.1, 0.1)]
            for i in range(3)}
    h = mh.histogram(rows)
    assert h["plate"] == 3 and h["drawer"] == 3
    assert all(v == 0 for k, v in h.items() if k not in ("plate", "drawer"))


# ---------------------------------------------------------------------------
# train-scale (v2): runtime category-id discovery, never hardcoded
# ---------------------------------------------------------------------------

def test_mapped_category_ids_reads_ids_from_json_not_hardcoded():
    # ids deliberately NOT the real COCO/LVIS ids — mapping must go by NAME
    coco = {"categories": [
        {"id": 77, "name": "fork"}, {"id": 12, "name": "spoon"},
        {"id": 5, "name": "bowl"}, {"id": 9, "name": "cup"},
        {"id": 40, "name": "wine glass"}, {"id": 3, "name": "dining table"},
    ]}
    m = mh.mapped_category_ids(coco, "coco-2017")
    assert m == {"cup": [9, 40], "fork": [77], "spoon": [12]}  # bowl/table dropped

    lvis = {"categories": [
        {"id": 913, "name": "plate"}, {"id": 240, "name": "saucer"},
        {"id": 601, "name": "drawer"}, {"id": 88, "name": "place_mat"},
    ]}
    m = mh.mapped_category_ids(lvis, "lvis")
    assert m == {"drawer": [601], "plate": [240, 913]}  # place_mat -> dropped


def test_coco_train_style_json_maps_by_name():
    # train2017 schema is the same instances format; ids must not be assumed
    inst = {
        "categories": [{"id": 44, "name": "knife"}, {"id": 45, "name": "fork"}],
        "images": [{"id": 99, "width": 100, "height": 50}],
        "annotations": [
            {"image_id": 99, "category_id": 44, "bbox": [0, 0, 50, 25]},
            {"image_id": 99, "category_id": 45, "bbox": [50, 25, 50, 25]},
        ],
    }
    rows = mh.coco_target_rows(inst)[99]
    assert [c for c, *_ in rows] == [cm.TARGET_INDEX["knife"], cm.TARGET_INDEX["fork"]]


def test_lvis_image_name_prefers_file_name_then_coco_url():
    assert mh.lvis_image_name({"file_name": "train2017/0000001.jpg"}) == "0000001.jpg"
    assert mh.lvis_image_name(
        {"coco_url": "http://images.cocodataset.org/train2017/000000000009.jpg"}
    ) == "000000000009.jpg"
    assert mh.lvis_image_name({}) == ""


# ---------------------------------------------------------------------------
# train-scale (v2): selective extraction plan
# ---------------------------------------------------------------------------

def test_plan_selective_extract_filters_wanted_members(tmp_path: Path):
    z = tmp_path / "train2017.zip"
    with mh.zipfile.ZipFile(z, "w") as zf:
        for n in ("train2017/a.jpg", "train2017/b.jpg", "train2017/unwanted.jpg"):
            zf.writestr(n, b"x")
        zf.writestr("train2017/", b"")  # dir entry must not confuse matching
    wanted = {"train2017/a.jpg": 1, "train2017/b.jpg": 2, "train2017/c.jpg": 3}
    with mh.zipfile.ZipFile(z) as zf:
        present, missing = mh.plan_selective_extract(zf.namelist(), wanted)
    assert present == ["train2017/a.jpg", "train2017/b.jpg"]
    assert missing == [3]


# ---------------------------------------------------------------------------
# train-scale (v2): cross-set dedup vs an existing dataset (synthetic hashes)
# ---------------------------------------------------------------------------

def test_partition_vs_reference_drops_exact_near_keeps_new():
    ref = mh.Deduper(tolerance=6)
    base = 0xFFFF_FF00_0000_0000
    # seed "v1" reference images
    for k, sha, dh in [("v1:a", "sha-a", base),
                       ("v1:b", "sha-b", base ^ 0xFFFF_0000_0000_0000)]:
        assert ref.add(k, sha, dh).status == "unique"

    items = [
        ("v2:new", "sha-new", 0x0000_00FF_0000_0000),   # far away -> keep
        ("v2:exact", "sha-a", 0x1234),                  # same sha as v1:a -> drop
        ("v2:near", "sha-near", base ^ 0x3F),           # hamming 6 vs v1:a -> drop
    ]
    kept, dropped = mh.partition_vs_reference(items, ref)
    assert [k for k, *_ in kept] == ["v2:new"]
    assert dict(dropped) == {"v2:exact": "v1:a", "v2:near": "v1:a"}

    # a second candidate colliding with an already-kept v2 image is a
    # within-set dup (matched key has no v1: prefix) — caller splits by prefix
    kept2, dropped2 = mh.partition_vs_reference(
        [("v2:twin", "sha-twin", 0x0000_00FF_0000_0001)], ref)  # hamming 1 vs v2:new
    assert kept2 == [] and dropped2[0][1] == "v2:new"
