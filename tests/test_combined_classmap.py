"""Focused contracts for the opt-in combined YOLO vocabulary."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception"))

import combined_classmap as cm  # noqa: E402


def test_tableware_prefix_is_stable():
    assert cm.TARGET_CLASSES[:7] == (
        "plate", "cup", "fork", "spoon", "knife", "napkin", "drawer"
    )
    assert cm.TARGET_INDEX["plate"] == 0
    assert cm.TARGET_INDEX["drawer"] == 6


def test_context_and_kitchen_classes_are_present():
    for name in (
        "person", "chair", "table", "bird", "cat", "dog", "bottle",
        "bowl", "microwave", "oven", "toaster", "sink", "refrigerator",
    ):
        assert name in cm.TARGET_INDEX


def test_coco_aliases_preserve_tableware_semantics():
    assert cm.remap_coco("cup") == "cup"
    assert cm.remap_coco("wine glass") == "cup"
    assert cm.remap_coco("fork") == "fork"
    assert cm.remap_coco("dining table") == "table"
    assert cm.TARGET_INDEX[cm.remap_coco("cup")] == cm.TARGET_INDEX["cup"]


def test_unknown_coco_label_fails_closed():
    assert cm.remap_coco("not-a-coco-category") is None

