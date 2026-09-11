"""Demo: the real fine-tuned "eyes" wired through the FrameObserver tracker.

Runs ``yolo_perception.YoloDetector`` (the OQ-008 7-class table fine-tune,
val mAP50 0.324) over a few frames and prints **tracked** detections
``{id, class, center, conf}`` -- the same Tracker + zone map
``FrameObserver.observe`` uses per frame, exercised directly so no WorldState
is needed:

    PYTHONPATH=src python -m omni_q.demo_yolo_wired
    PYTHONPATH=src python -m omni_q.demo_yolo_wired \
        --images data/table_yolo_v2/images/val        # real val frames instead

Two frame sources:

- default: a synthetic tabletop scene (PIL-drawn plate/cup/utensils on a
  wooden table, sub-pixel drift between frames) -- no dataset needed, runs
  anywhere, and the drift is there so the IoU tracker's stable ids are
  visible even when the detector's boxes jitter;
- ``--images <dir>``: real JPEG frames (e.g. the v2 val split under
  ``data/``, gitignored but present locally).

Opt-in, per the weights-are-local rule: the .pt is never required for tests
or the default stub path; this demo exits with a clear message if it (or the
override ``OMNIQ_YOLO_WEIGHTS``) is missing.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from .frame_observer import (
    YOLO_WEIGHTS_ENV_VAR,
    Detector,
    Tracker,
    grid_zone_map,
)
from .yolo_perception import TABLE_YOLO_CLASSES

DEFAULT_WEIGHTS = Path("models/table_yolo_v2_ft_2026-09-11.pt")
DEFAULT_VAL_IMAGES = Path("data/table_yolo_v2/images/val")


def _synth_tabletop(width: int, height: int, frame_idx: int, n_frames: int):
    """A deterministic tabletop scene with smooth object drift across frames.

    Deliberately crude -- it exists so the detector sees *something* shaped
    like the training distribution without needing the dataset, and so the
    tracker's stable-id behavior is observable frame to frame."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (146, 108, 72))  # wooden table
    d = ImageDraw.Draw(img)
    phase = 2 * math.pi * frame_idx / max(n_frames, 1)

    def drift(cx: float, cy: float) -> tuple[float, float]:
        return cx + 6 * math.sin(phase), cy + 4 * math.cos(phase)

    # plate
    cx, cy = drift(width * 0.42, height * 0.52)
    d.ellipse([cx - 90, cy - 90, cx + 90, cy + 90], fill=(240, 238, 230), outline=(200, 198, 190))
    d.ellipse([cx - 55, cy - 55, cx + 55, cy + 55], outline=(210, 208, 200))
    # cup
    cx, cy = drift(width * 0.66, height * 0.34)
    d.ellipse([cx - 32, cy - 32, cx + 32, cy + 32], fill=(226, 232, 240), outline=(120, 120, 130))
    d.ellipse([cx - 24, cy - 24, cx + 24, cy + 24], fill=(90, 60, 40))
    # fork + knife lines to the plate's right
    cx, cy = drift(width * 0.62, height * 0.58)
    for i, shade in enumerate([(205, 205, 210), (215, 215, 220)]):
        x0 = cx + i * 24
        d.line([x0, cy - 70, x0, cy + 70], fill=shade, width=7)
    # napkin
    cx, cy = drift(width * 0.30, height * 0.30)
    d.rectangle([cx - 55, cy - 40, cx + 55, cy + 40], fill=(235, 230, 220))
    return img


def _load_frames(images_dir: Path, n_frames: int) -> list:
    from PIL import Image

    paths = sorted(images_dir.glob("*.jpg"))[:n_frames]
    if not paths:
        raise SystemExit(f"no .jpg frames under {images_dir}")
    return [Image.open(p).convert("RGB") for p in paths]


def _resolve_weights(arg: Path | None) -> Path:
    weights = Path(arg or os.environ.get(YOLO_WEIGHTS_ENV_VAR, DEFAULT_WEIGHTS))
    if not weights.is_file():
        raise SystemExit(
            f"weights not found: {weights}\n"
            f"the fine-tune is a local artifact (gitignored) -- pull it from "
            f"https://huggingface.co/KissTheHabit/yolov8n-table-yolo or point "
            f"--weights / {YOLO_WEIGHTS_ENV_VAR} at a local .pt")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, default=None,
                        help=f".pt path (default: {YOLO_WEIGHTS_ENV_VAR} env or {DEFAULT_WEIGHTS})")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frames", type=int, default=4, help="frames to run")
    parser.add_argument("--images", type=Path, default=None,
                        help=f"dir of real frames (default: synthetic; e.g. {DEFAULT_VAL_IMAGES})")
    args = parser.parse_args()

    from .vision import as_frame_detector
    from .yolo_perception import YoloDetector

    weights = _resolve_weights(args.weights)
    detector: Detector = as_frame_detector(
        YoloDetector(weights, conf_threshold=args.conf, imgsz=args.imgsz),
        (640, 480),
    )

    if args.images is not None:
        frames = _load_frames(args.images, args.frames)
        source = f"{args.images} (real frames)"
    else:
        frames = [_synth_tabletop(640, 480, i, args.frames) for i in range(args.frames)]
        source = "synthetic rendered tabletop"

    print("=" * 66)
    print(f"YoloDetector ({weights.name}) -> Tracker -> stable ids")
    print("=" * 66)
    print(f"frames: {source} · taxonomy: {', '.join(TABLE_YOLO_CLASSES)}")

    tracker = Tracker()
    for i, frame in enumerate(frames):
        dets = detector(frame)  # normalised Detection2D, via as_frame_detector
        zoned = [(d, grid_zone_map(*d.center)) for d in dets]
        tracks = tracker.update(i, zoned)
        print(f"\nframe {i}: {len(dets)} detection(s), {len(tracks)} live track(s)")
        for t in sorted(tracks, key=lambda t: t.object_id):
            cx, cy = (t.xyxy[0] + t.xyxy[2]) / 2, (t.xyxy[1] + t.xyxy[3]) / 2
            print(f"  id={t.object_id:<10} class={t.cls:<7} "
                  f"center=({cx:.3f},{cy:.3f}) conf={t.conf:.2f} zone={t.zone}")
        if not dets:
            print("  (no detections above --conf)")

    print(
        "\nHonesty note: labels and confidences come straight from the v2 "
        "fine-tune (val mAP50 0.324) -- a real but modest detector. Stable "
        "ids come from the IoU/class tracker, not the model."
    )


if __name__ == "__main__":
    main()
