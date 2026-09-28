"""The real "eyes" wired in: YoloDetector + the OMNIQ_PERCEPTION env gate.

Tiered like tests/test_vision.py: everything except the explicitly
weights-gated smoke test runs with NO torch/ultralytics and NO weights on
disk -- the ultralytics import is monkeypatched with a fake module, the
weights-existence gate is exercised with a path that does not exist, and the
real-weights smoke test is ``skipif(not WEIGHTS.exists())``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from omni_q.frame_observer import (
    PERCEPTION_ENV_VAR,
    YOLO_WEIGHTS_ENV_VAR,
    DEFAULT_YOLO_WEIGHTS,
    StubDetector,
    Tracker,
    detector_from_env,
    grid_zone_map,
    perception_mode,
)
from omni_q.yolo_perception import TABLE_YOLO_CLASSES, YoloDetector

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "table_yolo_v2_ft_2026-09-11.pt"


# ---------------------------------------------------------------------------
# fake ultralytics (no torch, no weights)
# ---------------------------------------------------------------------------

def _make_fake_ultralytics(monkeypatch, frames: list, weights: Path):
    """Install a fake ``ultralytics`` module whose YOLO.predict returns one
    canned result per call: ``frames`` is a list of ``(xyxy, conf, cls)``
    rows in pixel coords. ``weights`` must exist on disk (YoloDetector
    validates it before handing the path to the backend)."""
    weights.touch()
    calls = {"n": 0}

    class _Boxes:
        def __init__(self, rows):
            self.xyxy = [r[0] for r in rows]
            self.conf = [r[1] for r in rows]
            self.cls = [r[2] for r in rows]

        def __len__(self):
            return len(self.conf)

    class _Result:
        def __init__(self, rows):
            self.boxes = _Boxes(rows) if rows else None

    class FakeYOLO:
        names = {i: c for i, c in enumerate(TABLE_YOLO_CLASSES)}

        def __init__(self, path):
            self.path = path

        def predict(self, frame, **kwargs):
            i = min(calls["n"], len(frames) - 1)
            calls["n"] += 1
            return [_Result(frames[i])]

    fake = types.ModuleType("ultralytics")
    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    return calls


# ---------------------------------------------------------------------------
# (a) gate parsing
# ---------------------------------------------------------------------------

def test_gate_defaults_to_stub(monkeypatch):
    from omni_q.world import MockWorld

    monkeypatch.delenv(PERCEPTION_ENV_VAR, raising=False)
    monkeypatch.delenv(YOLO_WEIGHTS_ENV_VAR, raising=False)

    assert perception_mode() == "stub"
    detector = detector_from_env(MockWorld.sample())
    assert isinstance(detector, StubDetector)


def test_gate_unknown_mode_raises(monkeypatch):
    monkeypatch.setenv(PERCEPTION_ENV_VAR, "telepathy")
    with pytest.raises(ValueError, match=PERCEPTION_ENV_VAR):
        detector_from_env(object())


def test_gate_stub_mode_requires_world_ref(monkeypatch):
    monkeypatch.setenv(PERCEPTION_ENV_VAR, "stub")
    with pytest.raises(ValueError, match="world_ref"):
        detector_from_env(None)


# ---------------------------------------------------------------------------
# (b) yolo gate with missing weights -> clear error, no silent fallback
# ---------------------------------------------------------------------------

def test_yolo_gate_with_missing_weights_names_the_env_vars(monkeypatch, tmp_path):
    missing = tmp_path / "no-such-weights.pt"
    monkeypatch.setenv(PERCEPTION_ENV_VAR, "yolo")
    monkeypatch.setenv(YOLO_WEIGHTS_ENV_VAR, str(missing))

    with pytest.raises(FileNotFoundError) as exc_info:
        detector_from_env(object())  # must raise BEFORE touching ultralytics

    message = str(exc_info.value)
    assert PERCEPTION_ENV_VAR in message
    assert YOLO_WEIGHTS_ENV_VAR in message
    assert str(missing) in message


# ---------------------------------------------------------------------------
# (c) YoloDetector over a mocked ultralytics backend
# ---------------------------------------------------------------------------

def test_yolo_detector_maps_to_the_7_class_taxonomy(monkeypatch, tmp_path):
    rows = [
        ([10.0, 20.0, 110.0, 120.0], 0.91, 0),   # plate
        ([200.0, 100.0, 260.0, 230.0], 0.66, 2),  # fork
    ]
    weights = tmp_path / "fake.pt"
    _make_fake_ultralytics(monkeypatch, [rows], weights)

    detector = YoloDetector(weights, conf_threshold=0.25, imgsz=640)
    out = detector.detect(object())  # frame content irrelevant to the fake

    assert len(out) == 2
    plate, fork = out
    assert plate.cls_name == "plate" and plate.cls_id == 0
    assert fork.cls_name == "fork" and fork.cls_id == 2
    for d in out:
        assert d.cls_name in TABLE_YOLO_CLASSES
        assert 0.0 <= d.conf <= 1.0
        x1, y1, x2, y2 = d.bbox_xyxy
        assert x1 < x2 and y1 < y2
        assert d.center_xy == pytest.approx(((x1 + x2) / 2, (y1 + y2) / 2))
    assert detector.class_names == {i: c for i, c in enumerate(TABLE_YOLO_CLASSES)}


def test_yolo_detector_returns_empty_list_when_no_boxes(monkeypatch, tmp_path):
    weights = tmp_path / "fake.pt"
    _make_fake_ultralytics(monkeypatch, [[]], weights)
    detector = YoloDetector(weights)
    assert detector.detect(object()) == []


# ---------------------------------------------------------------------------
# (d) tracker integration: YoloDetector-shaped output keeps stable ids
# ---------------------------------------------------------------------------

def test_tracker_keeps_stable_ids_across_yolo_shaped_frames(monkeypatch, tmp_path):
    """Two frames of pixel boxes with small drift + high IoU overlap must
    keep one id per object -- the exact consume path FrameObserver uses
    (as_frame_detector normalisation + Tracker IoU association)."""
    from omni_q.vision import as_frame_detector

    frame1 = [
        ([100.0, 100.0, 200.0, 200.0], 0.90, 0),   # plate, pixel space
        ([300.0, 100.0, 340.0, 260.0], 0.80, 1),   # cup
    ]
    frame2 = [  # everything drifts ~8px: IoU still well above the 0.3 gate
        ([108.0, 104.0, 208.0, 204.0], 0.88, 0),
        ([306.0, 102.0, 346.0, 262.0], 0.79, 1),
    ]
    weights = tmp_path / "fake.pt"
    _make_fake_ultralytics(monkeypatch, [frame1, frame2], weights)

    wrapped = as_frame_detector(YoloDetector(weights), (640, 480))
    tracker = Tracker()

    first = {t.object_id: t for t in tracker.update(
        0, [(d, grid_zone_map(*d.center)) for d in wrapped(object())])}
    second = {t.object_id: t for t in tracker.update(
        1, [(d, grid_zone_map(*d.center)) for d in wrapped(object())])}

    assert set(first) == set(second) != set()
    for tid, t1 in first.items():
        t2 = second[tid]
        assert t1.cls == t2.cls
        assert t2.last_seen == 1


# ---------------------------------------------------------------------------
# (e) real-weights smoke test -- local artifact, opt-in
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not WEIGHTS.exists(), reason=(
    f"fine-tuned weights are a local artifact (gitignored): pull from "
    f"KissTheHabit/yolov8n-table-yolo or set OMNIQ_YOLO_WEIGHTS; "
    f"expected at {WEIGHTS}"))
def test_real_weights_run_one_detection_on_a_synthetic_image(monkeypatch):
    pytest.importorskip("ultralytics")  # gate the heavy dep
    numpy = pytest.importorskip("numpy")

    monkeypatch.setenv(PERCEPTION_ENV_VAR, "yolo")
    monkeypatch.setenv(YOLO_WEIGHTS_ENV_VAR, str(WEIGHTS))

    from omni_q.frame_observer import detector_from_env

    # small tabletop-ish image: table colour + a light blob where a plate goes
    frame = numpy.full((480, 640, 3), (146, 108, 72), dtype=numpy.uint8)
    frame[200:340, 240:420] = (238, 236, 228)

    detector = detector_from_env(object(), conf_threshold=0.05)
    dets = detector(frame)  # gate built it; returns normalised Detection2D

    assert isinstance(dets, list)
    for d in dets:
        assert d.cls in TABLE_YOLO_CLASSES
        assert 0.0 <= d.conf <= 1.0
        x0, y0, x1, y1 = d.xyxy
        assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0


def test_default_weights_path_points_at_the_ftv2_fine_tune():
    assert DEFAULT_YOLO_WEIGHTS.name == "table_yolo_v2_ft_2026-09-11.pt"
    assert DEFAULT_YOLO_WEIGHTS.parent.name == "models"
