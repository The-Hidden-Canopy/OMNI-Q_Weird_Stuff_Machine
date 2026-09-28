"""FrameObserver — frame-based perception + stable-id tracking (OQ-004 gap, OQ-008)."""

from __future__ import annotations

from omni_q.contracts import Observe
from omni_q.frame_observer import (
    Detection2D,
    FrameObserver,
    StubDetector,
    Tracker,
    grid_zone_map,
    iou,
)
from omni_q.world import MockWorld


def _box(cx, cy, s=0.08):
    b = s / 2
    return (cx - b, cy - b, cx + b, cy + b)


def test_satisfies_observe_contract():
    obs = FrameObserver(StubDetector(MockWorld.sample()))
    assert isinstance(obs, Observe)


def test_iou_basic():
    assert iou(_box(0.5, 0.5), _box(0.5, 0.5)) == 1.0
    assert iou(_box(0.1, 0.1), _box(0.9, 0.9)) == 0.0
    assert 0.0 < iou(_box(0.5, 0.5), _box(0.53, 0.5)) < 1.0


def test_stable_id_across_frames_as_box_drifts():
    tr = Tracker(iou_thresh=0.3)
    d0 = Detection2D("plate", 0.9, _box(0.40, 0.40))
    id0 = tr.update(1, [(d0, "z")])[0].object_id
    # small drift each frame -> same id
    for f, cx in enumerate([0.42, 0.44, 0.46], start=2):
        d = Detection2D("plate", 0.9, _box(cx, 0.40))
        tracks = tr.update(f, [(d, "z")])
        assert len(tracks) == 1 and tracks[0].object_id == id0


def test_new_object_gets_new_id_and_missing_object_drops():
    tr = Tracker(iou_thresh=0.3, max_missed=2)
    a = Detection2D("cup", 0.9, _box(0.2, 0.2))
    b = Detection2D("cup", 0.9, _box(0.8, 0.8))
    tr.update(1, [(a, "z")])
    ids = {t.object_id for t in tr.update(2, [(a, "z"), (b, "z")])}
    assert len(ids) == 2
    # b disappears; still tracked (missed<=2) then dropped
    tr.update(3, [(a, "z")])
    tr.update(4, [(a, "z")])
    live = {t.object_id for t in tr.update(5, [(a, "z")])}
    assert len(live) == 1


def test_two_same_class_objects_keep_their_ids():
    tr = Tracker(iou_thresh=0.3)
    left = Detection2D("fork", 0.9, _box(0.30, 0.5))
    right = Detection2D("fork", 0.9, _box(0.70, 0.5))
    t = {x.object_id: x.xyxy[0] for x in tr.update(1, [(left, "z"), (right, "z")])}
    id_left = min(t, key=t.get)
    id_right = max(t, key=t.get)
    # nudge them toward each other, still distinct
    left2 = Detection2D("fork", 0.9, _box(0.40, 0.5))
    right2 = Detection2D("fork", 0.9, _box(0.60, 0.5))
    got = {x.object_id: x.xyxy[0] for x in tr.update(2, [(left2, "z"), (right2, "z")])}
    assert min(got, key=got.get) == id_left
    assert max(got, key=got.get) == id_right


def test_grid_zone_map():
    assert grid_zone_map(0.5, 0.5) == "center"
    assert grid_zone_map(0.1, 0.1) == "far_left"
    assert grid_zone_map(0.9, 0.9) == "near_right"


def test_observe_produces_detections_with_ids_and_zones():
    world = MockWorld.sample()
    obs = FrameObserver(StubDetector(world))
    o1 = obs.observe(world.state())
    assert o1.detections
    ids = {d.object_id for d in o1.detections}
    classes = {d.cls for d in o1.detections}
    assert "connector" in classes and "plate" in classes
    assert all(d.status.value == "live" for d in o1.detections)
    # ids stable on a second frame
    o2 = obs.observe(world.state())
    assert {d.object_id for d in o2.detections} == ids


def test_drop_in_replacement_for_fakeobserver_in_the_engine():
    from omni_q import build_mock_engine

    engine = build_mock_engine()
    engine.observer = FrameObserver(StubDetector(engine.world))
    receipt = engine.run("inspect and correct the workspace")
    # detector-driven perception still lets the loop finish and record
    assert "steps_executed" in receipt.metrics
    assert receipt.metrics["steps_executed"] >= 0
