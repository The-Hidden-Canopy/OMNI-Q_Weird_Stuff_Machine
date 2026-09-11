"""The real "eyes" — the fine-tuned 7-class tabletop YOLO, wired in.

``YoloDetector`` loads the OQ-008 fine-tune (``perception/`` pipeline, weights
published at `KissTheHabit/yolov8n-table-yolo <https://huggingface.co/KissTheHabit/yolov8n-table-yolo>`_)
through ultralytics' own ``YOLO`` class and emits the same pixel-space
:class:`omni_q.vision.RawDetection` as ``OpenVINODetector``, so
``vision.as_frame_detector`` + ``frame_observer.FrameObserver`` consume both
identically. `torch`/`ultralytics` are imported lazily inside ``__init__`` so
importing this module stays cheap (same opt-in idiom as
``omni_reasoner.OmniReferenceReasoner``).

Honesty notes (truth-in-labeling):

- The v2 fine-tune reports **val mAP50 0.324** on ``data/table_yolo_v2`` — a
  real but modest detector. Confidence and labels come from the fine-tune
  itself; nothing downstream re-weights them.
- The 7-class taxonomy is ``plate, cup, fork, spoon, knife, napkin, drawer``
  (``perception.classmap.TARGET_CLASSES``, class order == model output order,
  confirmed by the exported model's own ``metadata.yaml``).
- The weights are a **local artifact** (``*.pt`` is gitignored) — the yolo
  path is opt-in via ``OMNIQ_PERCEPTION=yolo`` (+ ``OMNIQ_YOLO_WEIGHTS``),
  see ``frame_observer.detector_from_env``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .vision import RawDetection

__all__ = ["TABLE_YOLO_CLASSES", "YoloDetector"]

# perception/classmap.TARGET_CLASSES — duplicated here so the gate and tests
# can check the taxonomy without importing the perception/ tooling package.
TABLE_YOLO_CLASSES: tuple[str, ...] = (
    "plate", "cup", "fork", "spoon", "knife", "napkin", "drawer",
)


def _to_rows(x: Any) -> list:
    """Accept torch tensors, numpy arrays, or plain lists (the latter so a
    mocked ultralytics result works without torch)."""
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    if hasattr(x, "tolist"):
        x = x.tolist()
    return list(x)


class YoloDetector:
    """Fine-tuned 7-class tabletop detector (ultralytics YOLO backend).

    Emits ``list[RawDetection]`` in the source frame's pixel coordinates —
    the exact contract ``OpenVINODetector`` keeps — so letterboxing, class
    decoding, and NMS are ultralytics' own battle-tested implementation, and
    the ``frame_observer`` seam on top is shared, not forked.

    Class names default to the loaded model's own ``names`` metadata (the
    same source ``OpenVINODetector`` uses via ``metadata.yaml``); pass
    ``class_names`` to pin the taxonomy explicitly.
    """

    def __init__(
        self,
        weights: "str | Path",
        *,
        conf_threshold: float = 0.25,
        imgsz: int = 640,
        device: str = "cpu",
        class_names: "dict[int, str] | None" = None,
    ) -> None:
        from ultralytics import YOLO  # lazy: heavy dep, opt-in backend

        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device
        self._model = YOLO(str(self.weights))
        if class_names is not None:
            self.class_names = {int(k): str(v) for k, v in class_names.items()}
        else:
            self.class_names = {int(k): str(v) for k, v in self._model.names.items()}

    def detect(self, frame) -> list[RawDetection]:
        """Run inference on one HxWx3 uint8 frame, return pixel-space
        detections above ``conf_threshold`` (ultralytics NMS applied)."""
        results = self._model.predict(
            frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = _to_rows(boxes.xyxy)
        confs = _to_rows(boxes.conf)
        clss = _to_rows(boxes.cls)

        out: list[RawDetection] = []
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            cid = int(cls)
            out.append(RawDetection(
                cls_id=cid,
                cls_name=self.class_names.get(cid, f"class_{cid}"),
                conf=round(float(conf), 4),
                bbox_xyxy=(round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
                center_xy=(round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)),
            ))
        return out
