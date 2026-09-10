"""OQ-008 — the 7-class table-setting vocabulary and per-source label maps.

No third-party deps: this is the deterministic core the dataset builder and the
tests use. Every public-dataset label we care about is mapped to one of the 7
targets, or dropped (``None``).

    plate  cup  fork  spoon  knife  napkin  drawer
"""

from __future__ import annotations

TARGET_CLASSES: tuple[str, ...] = (
    "plate", "cup", "fork", "spoon", "knife", "napkin", "drawer",
)
TARGET_INDEX: dict[str, int] = {c: i for i, c in enumerate(TARGET_CLASSES)}


# source label (as the dataset / FiftyOne exposes it) -> target class | None
_MAPS: dict[str, dict[str, str | None]] = {
    # FiftyOne "open-images-v7" detection display names
    "open-images-v7": {
        "Plate": "plate",
        "Coffee cup": "cup",
        "Mug": "cup",
        "Wine glass": "cup",
        "Fork": "fork",
        "Spoon": "spoon",
        "Kitchen knife": "knife",
        "Napkin": "napkin",
        "Drawer": "drawer",
        # deliberately dropped: "Knife" (weapon class), "Tableware" (too broad),
        # "Bowl", "Kitchen & dining room table"
        "Knife": None,
        "Tableware": None,
        "Bowl": None,
        "Kitchen & dining room table": None,
    },
    # FiftyOne "coco-2017" (80-class, lowercase)
    "coco-2017": {
        "cup": "cup",
        "wine glass": "cup",
        "fork": "fork",
        "knife": "knife",
        "spoon": "spoon",
        "bowl": None,
        "dining table": None,
    },
    # LVIS v1 synset-ish names (snake_case, singular)
    "lvis": {
        "plate": "plate",
        "saucer": "plate",
        "platter": "plate",
        "cup": "cup",
        "mug": "cup",
        "wineglass": "cup",
        "goblet": "cup",
        "fork": "fork",
        "spoon": "spoon",
        "soupspoon": "spoon",
        "wooden_spoon": "spoon",
        "knife": "knife",
        "butter_knife": "knife",
        "table_knife": "knife",
        "napkin": "napkin",
        "drawer": "drawer",
        "place_mat": None,
        "tablecloth": None,
        "bowl": None,
    },
    # Objects365 v2 (title case, as in ultralytics/cfg/datasets/Objects365.yaml)
    "objects365": {
        "Plate": "plate",
        "Cup": "cup",
        "Wine Glass": "cup",
        "Fork": "fork",
        "Knife": "knife",
        "Spoon": "spoon",
        "Napkin": "napkin",
        "Bowl": None,
        "Dinning Table": None,   # sic - Objects365's own spelling
    },
}

SOURCES: tuple[str, ...] = tuple(_MAPS)


def remap(source: str, label: str) -> str | None:
    """Source label -> target class, or None to drop. Unknown labels drop."""
    if source not in _MAPS:
        raise KeyError(f"unknown source {source!r}; expected one of {SOURCES}")
    return _MAPS[source].get(label)


def request_labels(source: str) -> list[str]:
    """The source labels to ask the dataset/FiftyOne for (the ones that map to
    a real target). Skips the explicit ``None`` drops."""
    return sorted(k for k, v in _MAPS[source].items() if v is not None)


def coverage() -> dict[str, list[str]]:
    """target class -> which sources can supply it (for gap-spotting)."""
    out: dict[str, list[str]] = {c: [] for c in TARGET_CLASSES}
    for src, m in _MAPS.items():
        for tgt in set(v for v in m.values() if v):
            out[tgt].append(src)
    return {k: sorted(v) for k, v in out.items()}
