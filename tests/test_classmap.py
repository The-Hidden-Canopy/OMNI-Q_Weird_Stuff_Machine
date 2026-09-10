"""OQ-008 — the table-setting class map (deterministic, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception"))

import classmap as cm  # noqa: E402


def test_seven_targets_indexed_in_order():
    assert cm.TARGET_CLASSES == ("plate", "cup", "fork", "spoon", "knife", "napkin", "drawer")
    assert cm.TARGET_INDEX["plate"] == 0 and cm.TARGET_INDEX["drawer"] == 6


def test_every_target_has_at_least_one_real_source():
    cov = cm.coverage()
    for tgt, srcs in cov.items():
        assert srcs, f"{tgt} has no source"
    # drawer is the known-thin one: only Open Images + LVIS carry it
    assert set(cov["drawer"]) == {"open-images-v7", "lvis"}


def test_open_images_knife_vs_kitchen_knife():
    assert cm.remap("open-images-v7", "Kitchen knife") == "knife"
    assert cm.remap("open-images-v7", "Knife") is None          # weapon class, dropped
    assert cm.remap("open-images-v7", "Drawer") == "drawer"


def test_cup_family_collapses():
    assert cm.remap("coco-2017", "wine glass") == "cup"
    assert cm.remap("open-images-v7", "Mug") == "cup"
    assert cm.remap("lvis", "wineglass") == "cup"
    assert cm.remap("objects365", "Wine Glass") == "cup"


def test_plate_family_from_lvis():
    for lbl in ("plate", "saucer", "platter"):
        assert cm.remap("lvis", lbl) == "plate"


def test_objects365_spelling_and_drops():
    assert cm.remap("objects365", "Plate") == "plate"
    assert cm.remap("objects365", "Napkin") == "napkin"
    assert cm.remap("objects365", "Dinning Table") is None      # O365's own typo
    assert cm.remap("objects365", "Bowl") is None


def test_unknown_label_drops_unknown_source_raises():
    assert cm.remap("coco-2017", "giraffe") is None
    try:
        cm.remap("nope", "plate")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown source should raise")


def test_request_labels_excludes_drops():
    req = cm.request_labels("open-images-v7")
    assert "Kitchen knife" in req and "Drawer" in req
    assert "Knife" not in req and "Tableware" not in req
    assert all(cm.remap("open-images-v7", lbl) is not None for lbl in req)
