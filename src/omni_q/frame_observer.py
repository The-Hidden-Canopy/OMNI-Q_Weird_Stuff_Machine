"""FrameObserver — perception from rendered frames, not ground truth.

Closes the OQ-004 audit's flat gap ("observer reads ground-truth state, not
rendered frames"). Implements the ``Observe`` contract on top of:

- a **detector** callable ``frame -> list[Detection2D]`` — a stub now, the
  fine-tuned 7-class table YOLO (`perception/`) later, or a remote node;
- a **zone map** ``(cx, cy) -> zone name`` placing an image-space box in a
  table region;
- an **IoU + class tracker** giving each object a **stable id across frames**
  (the brief's "stable object IDs from simulated camera frames").

The detector is injected, so swapping the stub for the real YOLO changes
nothing else. ``frame_source`` pulls the frame from the world adapter
(``lambda w: sim.render()``); for the mock, ``StubDetector`` derives boxes from
the world so the whole path runs with no model and no renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .contracts import DataStatus, Detection, Observation, WorldState


@dataclass(frozen=True)
class Detection2D:
    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]   # normalised [0,1] image coords

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.xyxy
        return (x0 + x1) / 2, (y0 + y1) / 2


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


Detector = Callable[[Any], "list[Detection2D]"]
ZoneMap = Callable[[float, float], str]
FrameSource = Callable[[WorldState], Any]


# ---------------------------------------------------------------------------
# stable-id tracker
# ---------------------------------------------------------------------------


@dataclass
class _Track:
    object_id: str
    cls: str
    xyxy: tuple[float, float, float, float]
    conf: float
    zone: str
    last_seen: int
    missed: int = 0


class Tracker:
    """Greedy per-class IoU association; ids survive occlusion for
    ``max_missed`` frames, then drop."""

    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 3,
                 center_gate: float = 0.15) -> None:
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.center_gate = center_gate      # fallback: match by centre when IoU=0
        self._tracks: dict[str, _Track] = {}
        self._counts: dict[str, int] = {}

    def _new_id(self, cls: str) -> str:
        self._counts[cls] = self._counts.get(cls, 0) + 1
        return f"{cls}_{self._counts[cls]}"

    @staticmethod
    def _dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5

    def update(self, frame_idx: int, dets: list[tuple[Detection2D, str]]) -> list[_Track]:
        by_cls: dict[str, list[tuple[Detection2D, str]]] = {}
        for d, zone in dets:
            by_cls.setdefault(d.cls, []).append((d, zone))

        matched: set[str] = set()
        for cls, group in by_cls.items():
            live = [t for t in self._tracks.values() if t.cls == cls]
            pending: list[tuple[Detection2D, str]] = []

            # pass 1 — IoU (survives small drift, disambiguates crossings)
            for det, zone in sorted(group, key=lambda g: -g[0].conf):
                best, best_iou = None, self.iou_thresh
                for t in live:
                    if t.object_id in matched:
                        continue
                    v = iou(t.xyxy, det.xyxy)
                    if v >= best_iou:
                        best, best_iou = t, v
                if best is None:
                    pending.append((det, zone))
                else:
                    best.xyxy, best.conf, best.zone = det.xyxy, det.conf, zone
                    best.last_seen, best.missed = frame_idx, 0
                    matched.add(best.object_id)

            # pass 2 — nearest-centre within the gate (fast motion, low fps)
            for det, zone in pending:
                best, best_d = None, self.center_gate
                for t in live:
                    if t.object_id in matched:
                        continue
                    d = self._dist(t.xyxy, det.xyxy)
                    if d <= best_d:
                        best, best_d = t, d
                if best is None:
                    tid = self._new_id(cls)
                    self._tracks[tid] = _Track(tid, cls, det.xyxy, det.conf, zone, frame_idx)
                    matched.add(tid)
                else:
                    best.xyxy, best.conf, best.zone = det.xyxy, det.conf, zone
                    best.last_seen, best.missed = frame_idx, 0
                    matched.add(best.object_id)

        for tid, t in list(self._tracks.items()):
            if tid not in matched:
                t.missed += 1
                if t.missed > self.max_missed:
                    del self._tracks[tid]

        return [t for t in self._tracks.values() if t.missed == 0]


# ---------------------------------------------------------------------------
# zone map + stub detector
# ---------------------------------------------------------------------------


def grid_zone_map(cx: float, cy: float) -> str:
    """Coarse 3x3 image grid → a zone label. Replace with the real camera
    homography once OQ-006/OQ-007 pin the table pose."""
    col = "left" if cx < 0.38 else "right" if cx > 0.62 else "center"
    row = "far" if cy < 0.38 else "near" if cy > 0.62 else "mid"
    return f"{row}_{col}" if (row, col) != ("mid", "center") else "center"


class StubDetector:
    """A detector fed by the world, not a model — projects each object to a
    fake box so the FrameObserver path runs end-to-end with no YOLO/renderer.
    Swap for `perception/` YOLO inference and nothing else changes."""

    def __init__(self, world_ref: Any, box: float = 0.08) -> None:
        self._world = world_ref
        self._box = box
        # deterministic pseudo-layout: hash object id -> image position
        self._pos: dict[str, tuple[float, float]] = {}

    def _place(self, oid: str) -> tuple[float, float]:
        if oid not in self._pos:
            h = abs(hash(oid))
            self._pos[oid] = (0.12 + (h % 76) / 100.0, 0.12 + (h // 97 % 76) / 100.0)
        return self._pos[oid]

    def __call__(self, _frame: Any) -> list[Detection2D]:
        state = self._world.state() if hasattr(self._world, "state") else self._world
        out: list[Detection2D] = []
        for oid, d in state.objects.items():
            cx, cy = self._place(oid)
            b = self._box / 2
            out.append(Detection2D(d.cls, float(d.conf),
                                   (cx - b, cy - b, cx + b, cy + b)))
        return out


# ---------------------------------------------------------------------------
# the Observe provider
# ---------------------------------------------------------------------------


class FrameObserver:
    def __init__(
        self,
        detector: Detector,
        *,
        zone_map: ZoneMap = grid_zone_map,
        frame_source: FrameSource | None = None,
        target_zones: Callable[[str, str], str] | None = None,
        iou_thresh: float = 0.3,
        max_missed: int = 3,
    ) -> None:
        self.detector = detector
        self.zone_map = zone_map
        self.frame_source = frame_source or (lambda w: None)
        self.target_zones = target_zones
        self.tracker = Tracker(iou_thresh=iou_thresh, max_missed=max_missed)

    def observe(self, world: WorldState) -> Observation:
        frame = self.frame_source(world)
        raw = self.detector(frame)
        zoned = [(d, self.zone_map(*d.center)) for d in raw]
        tracks = self.tracker.update(world.frame, zoned)

        dets: list[Detection] = []
        for t in sorted(tracks, key=lambda t: t.object_id):
            tgt = (self.target_zones(t.object_id, t.cls) if self.target_zones
                   else self._infer_target(world, t))
            dets.append(Detection(
                object_id=t.object_id, cls=t.cls, zone=t.zone,
                target_zone=tgt, conf=round(t.conf, 3),
                status=DataStatus.LIVE, verified_frame=world.frame,
            ))
        return Observation(
            frame=world.frame,
            detections=tuple(dets),
            workspace_clear=not any(d.misplaced for d in dets),
            raw_ref=f"frame://camera/{world.frame:06d}",
        )

    @staticmethod
    def _infer_target(world: WorldState, t: _Track) -> str:
        """If the world already knows this object, reuse its target zone;
        otherwise the detection is location-only (target unknown)."""
        known = world.objects.get(t.object_id)
        return known.target_zone if known else t.zone
