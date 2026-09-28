"""Combined YOLO vocabulary for tableware plus scene and kitchen context.

The existing seven-class table-setting vocabulary is deliberately kept at the
front of this tuple.  Existing labels therefore remain valid, while a future
combined head can learn people, furniture, animals, and kitchen objects from
COCO without changing the meaning of ``plate``/``cup``/``fork``/``spoon``.

This module is separate from :mod:`classmap`: the deployed tableware model and
its seven-class contract remain backward-compatible.  The combined dataset is
an explicitly versioned opt-in artifact.
"""

from __future__ import annotations

from classmap import TARGET_CLASSES as TABLE_CLASSES


# COCO 2017 category names, in the source order.  Keeping this list here makes
# category-name mapping independent of COCO numeric ids and robust to re-id'ed
# annotation exports.
COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


# Preserve the existing tableware semantics.  In particular, COCO's wine
# glass has historically fed the tableware ``cup`` class; ``dining table`` is
# normalized to the shorter scene name ``table``.
_COCO_TO_COMBINED: dict[str, str] = {
    name: name for name in COCO_CLASSES
}
_COCO_TO_COMBINED.update({
    "wine glass": "cup",
    "dining table": "table",
})

_APPENDED_CLASSES = tuple(
    target for source in COCO_CLASSES
    for target in (_COCO_TO_COMBINED[source],)
    if target not in TABLE_CLASSES
)

# A dict/set preserves first occurrence while removing the COCO aliases that
# collapse into the existing tableware classes.
COMBINED_CLASSES: tuple[str, ...] = TABLE_CLASSES + tuple(
    dict.fromkeys(_APPENDED_CLASSES)
)
COMBINED_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(COMBINED_CLASSES)
}

# Public aliases make the combined contract read like the existing classmap
# contract without changing the old module's TARGET_CLASSES.
TARGET_CLASSES = COMBINED_CLASSES
TARGET_INDEX = COMBINED_INDEX


def remap_coco(label: str) -> str | None:
    """Map a COCO category name to the combined target, or ``None``.

    Unknown labels are dropped rather than guessed.  COCO's labels are
    expected to be lowercase, but surrounding whitespace is harmless.
    """

    return _COCO_TO_COMBINED.get(label.strip())


def mapped_coco_labels() -> dict[str, str]:
    """Return the source-to-target map as a defensive copy for manifests."""

    return dict(_COCO_TO_COMBINED)

