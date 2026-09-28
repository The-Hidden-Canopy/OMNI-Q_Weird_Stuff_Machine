"""Adversarial, local-only tests for combined dataset construction."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception"))

import build_combined_yolo as builder  # noqa: E402
import combined_classmap as cm  # noqa: E402
import multisource_haul as mh  # noqa: E402


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color).save(path, "JPEG")


def _coco_zip(path: Path) -> None:
    categories = [
        {"id": 1, "name": "person"},
        {"id": 47, "name": "cup"},
        {"id": 67, "name": "dining table"},
    ]
    payloads = {}
    for split, filename, image_id in (
        ("train2017", "000000000001.jpg", 1),
        ("val2017", "000000000002.jpg", 2),
    ):
        payloads[f"annotations/instances_{split}.json"] = {
            "images": [{"id": image_id, "file_name": filename,
                        "width": 100, "height": 80}],
            "annotations": [
                {"id": image_id * 10, "image_id": image_id,
                 "category_id": 1, "bbox": [10, 10, 20, 30]},
                {"id": image_id * 10 + 1, "image_id": image_id,
                 "category_id": 47, "bbox": [50, 20, 20, 20]},
                {"id": image_id * 10 + 2, "image_id": image_id,
                 "category_id": 67, "bbox": [0, 0, 100, 80]},
            ],
            "categories": categories,
        }
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, json.dumps(payload))


def _table_tree(root: Path, image_name: str, color=(255, 0, 0)) -> Path:
    image = root / "images" / "train" / image_name
    _image(image, color)
    label = root / "labels" / "train" / f"{Path(image_name).stem}.txt"
    mh.write_yolo_label(label, [(cm.TARGET_INDEX["plate"], 0.5, 0.5, 0.2, 0.2)])
    return root


def test_exact_duplicate_unions_context_labels_without_changing_plate_id(tmp_path: Path,
                                                                           monkeypatch):
    monkeypatch.chdir(tmp_path)
    table = _table_tree(tmp_path / "table_yolo_v3", "000000000001.jpg")
    # COCO copy is byte-identical, but contributes person/table boxes.
    train = tmp_path / "coco_train"
    val = tmp_path / "coco_val"
    _image(train / "000000000001.jpg", (255, 0, 0))
    _image(val / "000000000002.jpg", (0, 255, 0))
    # Use a visibly different spatial pattern so this is not classified as a
    # near duplicate merely because both fixtures are flat-color images.
    with Image.open(val / "000000000002.jpg") as image:
        for x in range(0, image.width, 4):
            for y in range(image.height):
                image.putpixel((x, y), (0, 0, 0))
        image.save(val / "000000000002.jpg", "JPEG")
    annotations = tmp_path / "annotations.zip"
    _coco_zip(annotations)

    manifest = builder.build_combined_dataset(
        [table], annotations,
        {"train2017": train, "val2017": val},
        tmp_path / "combined", val_frac=0.0,
    )
    assert manifest["final_unique_images"] == 2
    assert manifest["dedup"]["exact_annotation_merges"] == 1
    assert manifest["class_boxes_final"]["plate"] == 1
    assert manifest["class_boxes_final"]["person"] == 2
    assert manifest["class_boxes_final"]["table"] == 2
    labels = list((tmp_path / "combined" / "labels" / "train").glob("*.txt"))
    assert any(cm.TARGET_INDEX["plate"] in [row[0] for row in mh.parse_yolo_label(path)]
               for path in labels)


def test_missing_coco_image_is_reported_and_inputs_are_not_modified(tmp_path: Path,
                                                                      monkeypatch):
    monkeypatch.chdir(tmp_path)
    table = _table_tree(tmp_path / "table_yolo_v3", "existing.jpg")
    train = tmp_path / "coco_train"
    val = tmp_path / "coco_val"
    train.mkdir()
    val.mkdir()
    annotations = tmp_path / "annotations.zip"
    _coco_zip(annotations)
    manifest = builder.build_combined_dataset(
        [table], annotations,
        {"train2017": train, "val2017": val},
        tmp_path / "combined", val_frac=0.0,
    )
    assert manifest["coco_cache"]["splits"]["train2017"]["images_missing"] == 1
    assert manifest["coco_cache"]["splits"]["val2017"]["images_missing"] == 1
    assert (table / "images" / "train" / "existing.jpg").exists()


def test_indexed_deduper_preserves_near_duplicate_boundary():
    deduper = builder._IndexedDeduper(tolerance=6)
    first = 0x1234567890ABCDEF
    assert deduper.add("first", "sha-first", first).status == "unique"
    # Three changed bits stay within the documented near-duplicate boundary.
    second = first ^ (1 << 0) ^ (1 << 17) ^ (1 << 63)
    decision = deduper.add("second", "sha-second", second)
    assert decision.status == "near_dup"
    assert decision.matched == "first"
