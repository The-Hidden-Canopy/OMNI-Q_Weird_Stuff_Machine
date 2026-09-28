"""Pseudo-labeling core: class mapping, label emission, split, skip counting, dedup.

No network, no real weights.  The ultralytics import is monkeypatched with a
fake module exactly like tests/test_yolo_perception.py does; the real-weights
path is exercised only by the (skipped-without-weights) smoke test.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception"))

import classmap as cm  # noqa: E402
import multisource_haul as mh  # noqa: E402
import pseudo_label as pl  # noqa: E402

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "table_yolo_v2_ft_2026-09-11.pt"


def _img(w: int = 64, h: int = 64, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), "RGB")


def _save_jpg(path: Path, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _img(seed=seed).save(path, "JPEG")
    return path


# ---------------------------------------------------------------------------
# fake ultralytics (no torch, no weights)
# ---------------------------------------------------------------------------

def _make_fake_ultralytics(monkeypatch, weights: Path):
    """Fake YOLO: one detection (a plate) for every image EXCEPT paths
    containing 'neg' (zero detections).  Predict kwargs must be accepted
    silently."""
    weights.touch()
    calls = {"paths": []}

    class _Boxes:
        xyxy = [[10.0, 20.0, 110.0, 120.0]]   # 100x100 box in a 480x640 frame
        cls = [0]                              # model class 0 = plate

        def __len__(self):
            return len(self.cls)

    class _Result:
        orig_shape = (480, 640)
        boxes = _Boxes()

    class _EmptyResult:
        orig_shape = (480, 640)
        boxes = None

    class FakeYOLO:
        names = {i: c for i, c in enumerate(cm.TARGET_CLASSES)}

        def __init__(self, path):
            self.path = path

        def predict(self, frame, **kwargs):
            calls["paths"].append(str(frame))
            assert kwargs.get("imgsz") == 640
            assert kwargs.get("device") == "cpu"
            assert "conf" in kwargs and "stream" in kwargs
            return [_Result()] if "neg" not in str(frame) else [_EmptyResult()]

    fake = types.ModuleType("ultralytics")
    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    return calls


# ---------------------------------------------------------------------------
# class mapping
# ---------------------------------------------------------------------------

def test_our_class_ids_identity_for_ftv2_vocabulary():
    names = {i: c for i, c in enumerate(cm.TARGET_CLASSES)}
    assert pl.our_class_ids(names) == list(range(7))
    # list form works too
    assert pl.our_class_ids(list(cm.TARGET_CLASSES)) == list(range(7))


def test_our_class_ids_rejects_foreign_classes():
    with pytest.raises(ValueError, match="wrong weights"):
        pl.our_class_ids({0: "person", 1: "bicycle"})
    with pytest.raises(ValueError):
        pl.our_class_ids({0: "plate", 1: "airplane"})


# ---------------------------------------------------------------------------
# rows_from_result (pure)
# ---------------------------------------------------------------------------

def _result(xyxy, cls, shape=(480, 640)):
    class _Boxes:
        def __init__(self, xyxy, cls):
            self.xyxy = xyxy
            self.cls = cls

        def __len__(self):
            return len(self.cls)

    return types.SimpleNamespace(orig_shape=shape, boxes=_Boxes(xyxy, cls))


def test_rows_from_result_normalizes_and_maps_classes():
    ident = pl.our_class_ids(list(cm.TARGET_CLASSES))
    res = _result([[10.0, 20.0, 110.0, 120.0], [320.0, 240.0, 640.0, 480.0]],
                  [4, 1])
    rows = pl.rows_from_result(res, ident)
    assert rows == pytest.approx([(4, 60 / 640, 70 / 480, 100 / 640, 100 / 480),
                                  (1, 480 / 640, 360 / 480, 320 / 640, 240 / 480)])


def test_rows_from_result_clamps_boxes_outside_frame():
    ident = pl.our_class_ids(list(cm.TARGET_CLASSES))
    res = _result([[-20.0, -10.0, 100.0, 100.0], [600.0, 400.0, 999.0, 999.0]], [0, 2])
    rows = pl.rows_from_result(res, ident)
    for c, cx, cy, w, h in rows:
        assert 0.0 <= cx - w / 2 and cx + w / 2 <= 1.0
        assert 0.0 <= cy - h / 2 and cy + h / 2 <= 1.0
        assert 0 < w <= 1.0 and 0 < h <= 1.0


def test_rows_from_result_empty_and_degenerate():
    ident = pl.our_class_ids(list(cm.TARGET_CLASSES))
    assert pl.rows_from_result(_result([], []), ident) == []
    empty = types.SimpleNamespace(orig_shape=(480, 640), boxes=None)
    assert pl.rows_from_result(empty, ident) == []
    # zero-area box dropped
    assert pl.rows_from_result(_result([[5.0, 5.0, 5.0, 9.0]], [0]), ident) == []


# ---------------------------------------------------------------------------
# end-to-end over synthetic images: emission, copy, skip counting
# ---------------------------------------------------------------------------

def test_label_emission_and_skip_counting(monkeypatch, tmp_path):
    work = tmp_path / "hf"
    _save_jpg(work / "repoA" / "pos1.jpg", seed=1)
    _save_jpg(work / "repoA" / "pos2.jpg", seed=2)
    _save_jpg(work / "repoB" / "neg1.jpg", seed=3)   # no detections -> skipped

    fake_weights = tmp_path / "fake.pt"
    calls = _make_fake_ultralytics(monkeypatch, fake_weights)
    model, cls_map = pl._load_model(fake_weights)
    predict_rows = pl._predict_rows_factory(model, cls_map, conf=0.5)

    images = [(f"{p.parent.name}/{p.name}", p)
              for p in sorted(work.rglob("*.jpg"))]
    pool, stats = pl.collect_pool(images, predict_rows, mh.Deduper(), [])

    assert stats["images_seen"] == 3
    assert stats["images_no_detections"] == 1        # neg1 skipped + counted
    assert stats["unreadable"] == 0
    assert len(pool) == 2
    assert len(calls["paths"]) == 3     # every image went through predict

    out = tmp_path / "out"
    manifest = {"weights": "fake.pt", "weights_provenance": "test",
                "conf_threshold": 0.5, "license_note": "test",
                "sources": {}, "receipt_stamp": "20990101"}
    import time as _t
    pl.persist(pool, out, val_frac=0.0, seed=13, manifest=manifest,
               t_start=_t.time())

    lbls = sorted(out.glob("labels/*/*.txt"))
    imgs = sorted(out.glob("images/*/*.jpg"))
    assert len(lbls) == 2 and len(imgs) == 2
    for lbl in lbls:
        rows = mh.parse_yolo_label(lbl)
        assert len(rows) == 1
        c, cx, cy, w, h = rows[0]
        assert c == 0                       # plate, OUR 7-class order
        assert cx == pytest.approx(60 / 640, abs=1e-6)
        assert cy == pytest.approx(70 / 480, abs=1e-6)
    yaml_text = (out / "data.yaml").read_text(encoding="utf-8")
    assert f"path: {out.resolve().as_posix()}" in yaml_text
    for i, c in enumerate(cm.TARGET_CLASSES):
        assert f"{i}: {c}" in yaml_text
    saved = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert saved["final_unique_images"] == 2
    assert saved["class_boxes_final"]["plate"] == 2


# ---------------------------------------------------------------------------
# split determinism
# ---------------------------------------------------------------------------

def test_assign_splits_deterministic():
    keys = [f"k{i:03d}" for i in range(400)]
    a = pl.assign_splits(keys, 0.10, 13)
    b = pl.assign_splits(keys, 0.10, 13)
    assert a == b
    n_val = sum(1 for v in a.values() if v == "val")
    assert 0 < n_val < 100                          # ~40 for frac .10
    c = pl.assign_splits(keys, 0.10, 7)
    assert c != a or n_val != sum(1 for v in c.values() if v == "val")
    assert set(a) == set(keys) and set(a.values()) <= {"train", "val"}


# ---------------------------------------------------------------------------
# dedup drops: cross-set near-dup + within-set exact, counted separately
# ---------------------------------------------------------------------------

def _seed_ref(d: mh.Deduper, key: str, im: Image.Image, prefix: str = "v1:"):
    from io import BytesIO
    buf = BytesIO()
    im.save(buf, "JPEG")
    sha = mh.hashlib.sha256(buf.getvalue()).hexdigest()
    d.add(prefix + key, sha, mh.dhash64(im))


def test_dedup_drops_cross_set_near_dup_and_counts(monkeypatch, tmp_path):
    ref_im = _img(seed=42)
    d = mh.Deduper()
    _seed_ref(d, "ref.jpg", ref_im)

    work = tmp_path / "hf"
    _save_jpg(work / "a" / "near.jpg", seed=9)
    # overwrite near.jpg with a brightened copy of the REF (same layout):
    # different bytes, dHash within tolerance -> near_dup vs the v1: ref
    Image.eval(ref_im, lambda px: min(px + 2, 255)).save(work / "a" / "near.jpg", "JPEG")
    _save_jpg(work / "a" / "far.jpg", seed=77)

    fake_weights = tmp_path / "fake.pt"
    _make_fake_ultralytics(monkeypatch, fake_weights)
    model, cls_map = pl._load_model(fake_weights)
    predict_rows = pl._predict_rows_factory(model, cls_map, conf=0.5)

    images = [(f"{p.parent.name}/{p.name}", p) for p in sorted(work.rglob("*.jpg"))]
    pool, stats = pl.collect_pool(images, predict_rows, d, ["v1:"])

    assert len(pool) == 1 and "far" in next(iter(pool))
    assert stats["cross_set_drops"] == {"v1:": 1}
    assert stats["within_set_dups"] == {"exact": 0, "near": 0}


def test_dedup_counts_within_set_exact_dup(monkeypatch, tmp_path):
    work = tmp_path / "hf"
    src = work / "a" / "same.jpg"
    _save_jpg(src, seed=5)
    dup = work / "b" / "same.jpg"
    dup.parent.mkdir(parents=True)
    dup.write_bytes(src.read_bytes())            # byte-identical -> exact dup

    fake_weights = tmp_path / "fake.pt"
    _make_fake_ultralytics(monkeypatch, fake_weights)
    model, cls_map = pl._load_model(fake_weights)
    predict_rows = pl._predict_rows_factory(model, cls_map, conf=0.5)

    images = [(f"{p.parent.name}/{p.name}", p) for p in sorted(work.rglob("*.jpg"))]
    pool, stats = pl.collect_pool(images, predict_rows, mh.Deduper(), [])

    assert len(pool) == 1
    assert stats["within_set_dups"]["exact"] == 1
    assert stats["cross_set_drops"] == {}


def test_dhash_dedup_with_synthetic_dhashes():
    """Synthetic dHash values (no images): within tolerance drops, outside keeps."""
    d = mh.Deduper(tolerance=6)
    base = 0xAAAAAAAAAAAAAAAA
    assert d.add("a", "sha-a", base).status == "unique"
    near = base ^ 0b101                       # 2 bits flipped
    dec = d.add("b", "sha-b", near)
    assert dec.status == "near_dup" and dec.matched == "a"
    far = base ^ 0xFFFF                       # 16 bits flipped
    assert d.add("c", "sha-c", far).status == "unique"
    assert d.add("a2", "sha-a", base ^ 0xFF).status == "exact_dup"


# ---------------------------------------------------------------------------
# real-weights smoke — local artifact, opt-in (existing pattern)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not WEIGHTS.exists(), reason=(
    "fine-tuned weights are a local artifact (gitignored); expected at "
    f"{WEIGHTS}"))
def test_real_weights_pseudo_label_one_synthetic_image(monkeypatch, tmp_path):
    pytest.importorskip("ultralytics")
    work = tmp_path / "hf" / "real"
    _save_jpg(work / "pos_real.jpg", seed=11)
    model, cls_map = pl._load_model(WEIGHTS)
    predict_rows = pl._predict_rows_factory(model, cls_map, conf=0.05)
    pool, stats = pl.collect_pool([("real/pos_real.jpg", work / "pos_real.jpg")],
                                  predict_rows, mh.Deduper(), [])
    assert stats["images_seen"] == 1
    for rows in (v.rows for v in pool.values()):
        for c, *_ in rows:
            assert cm.TARGET_CLASSES[c] in cm.TARGET_CLASSES
