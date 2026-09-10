"""Camera + detector perception pipeline (see src/omni_q/vision.py).

Two tiers: the projection math is pure numpy and always runs; the camera
render + real OpenVINO inference tests need mujoco/openvino and (for the
detector test) a locally-exported model, so they skip cleanly when either
is absent rather than failing CI on missing multi-GB artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_project_to_table_recovers_a_known_world_point():
    """Pure math, no MuJoCo needed: pick a camera pose + a world point,
    forward-project with the textbook pinhole formula, then check
    project_to_table's back-projection recovers the same point. This is
    the same round-trip validated against a real MuJoCo render + a known
    object position during development (see integrations/intel/README.md)."""
    import numpy as np

    from omni_q.vision import project_to_table

    cam_pos = np.array([0.0, -0.1, 1.2])
    cam_rot = np.eye(3)  # camera looking straight down, table_overhead's actual pose
    fovy = 45.0
    width, height = 640, 480
    world_point = np.array([0.16, -0.06, 0.055])

    rel = world_point - cam_pos
    cam_space = cam_rot.T @ rel
    ndc_x, ndc_y = cam_space[0] / -cam_space[2], cam_space[1] / -cam_space[2]
    half_h = np.tan(np.radians(fovy) / 2)
    half_w = half_h * (width / height)
    u = (ndc_x / half_w + 1) / 2 * width
    v = (1 - ndc_y / half_h) / 2 * height

    recovered = project_to_table((u, v), (width, height), cam_pos, cam_rot, fovy, table_z=float(world_point[2]))

    assert recovered is not None
    assert recovered[0] == pytest.approx(0.16, abs=1e-6)
    assert recovered[1] == pytest.approx(-0.06, abs=1e-6)


def test_project_to_table_returns_none_for_a_ray_parallel_to_the_table():
    import numpy as np

    from omni_q.vision import project_to_table

    cam_pos = np.array([0.0, -0.1, 1.2])
    # Rotation mapping local forward (-Z) to world +X exactly (Ry(-90 deg)):
    # a level, horizontal look -- hand-verified: rot @ [0,0,-1] == [1,0,0].
    level_rot = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
    assert level_rot @ np.array([0.0, 0.0, -1.0]) == pytest.approx([1.0, 0.0, 0.0])
    # The centre pixel's ray is then exactly horizontal (world z-component
    # zero) -- never reaches the table plane, whatever height it's at.
    center_pixel = (320, 240)
    result = project_to_table(center_pixel, (640, 480), cam_pos, level_rot, 45.0, table_z=-0.005)
    assert result is None


mujoco = pytest.importorskip("mujoco")


def test_mujoco_camera_source_renders_a_real_frame():
    from omni_q.intel_sim import IntelTableWorld
    from omni_q.vision import MuJoCoCameraSource

    world = IntelTableWorld()
    cam = MuJoCoCameraSource(world.model, world.data, "table_overhead", width=320, height=240)

    frame = cam.capture()

    assert frame.shape == (240, 320, 3)
    assert frame.dtype.name == "uint8"
    pos, rot, fovy = cam.camera_pose()
    assert fovy > 0
    assert pos.shape == (3,)
    assert rot.shape == (3, 3)


_MODEL_XML = Path(os.environ.get(
    "OMNIQ_TEST_OPENVINO_MODEL",
    "does-not-exist/model.xml",
))


@pytest.mark.skipif(not _MODEL_XML.exists(), reason=(
    "set OMNIQ_TEST_OPENVINO_MODEL to a local OpenVINO IR .xml to exercise "
    "real inference -- see evidence/benchmark_results/openvino_inference_2026-09-10/"
    "README.md for how to export one; not committed (binary, regenerable)"
))
def test_openvino_detector_runs_real_inference_on_a_real_render():
    from omni_q.intel_sim import IntelTableWorld
    from omni_q.vision import MuJoCoCameraSource, OpenVINODetector

    world = IntelTableWorld()
    cam = MuJoCoCameraSource(world.model, world.data, "table_overhead", width=640, height=480)
    detector = OpenVINODetector(_MODEL_XML, device="CPU", conf_threshold=0.1)

    frame = cam.capture()
    detections = detector.detect(frame)

    # Not asserting specific classes/counts: this is the already-published
    # thermal model, whose labels are honestly meaningless for a table
    # scene (see vision.py's module docstring). What's real and checkable:
    # inference ran, and any boxes returned are sane pixel coordinates.
    for d in detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        assert 0 <= x1 < x2 <= 640
        assert 0 <= y1 < y2 <= 480
        assert 0.0 <= d.conf <= 1.0


@pytest.mark.skipif(not _MODEL_XML.exists(), reason=(
    "set OMNIQ_TEST_OPENVINO_MODEL to a local OpenVINO IR .xml -- see "
    "evidence/benchmark_results/openvino_inference_2026-09-10/README.md"
))
def test_real_detector_and_zone_map_wire_into_frame_observer():
    """The actual point of these adapters: a real render, through real
    OpenVINO inference, through a real camera-geometry zone map, satisfies
    frame_observer.FrameObserver's Observe contract end to end -- no
    ground-truth WorldState read anywhere in this path."""
    from omni_q.contracts import DataStatus
    from omni_q.frame_observer import FrameObserver
    from omni_q.intel_sim import IntelTableWorld, ZONE_POSITIONS
    from omni_q.vision import MuJoCoCameraSource, OpenVINODetector, as_frame_detector, make_camera_zone_map

    world = IntelTableWorld()
    cam = MuJoCoCameraSource(world.model, world.data, "table_overhead", width=640, height=480)
    detector = OpenVINODetector(_MODEL_XML, device="CPU", conf_threshold=0.1)

    zones = dict(ZONE_POSITIONS)
    zones.update({"tray_plate": (-.13, -.08, .026), "tray_cup": (.16, -.06, .055)})
    observer = FrameObserver(
        as_frame_detector(detector, (640, 480)),
        zone_map=make_camera_zone_map(cam, zones, (640, 480)),
        frame_source=lambda w: cam.capture(),
    )

    observation = observer.observe(world.state())

    for d in observation.detections:
        assert d.status is DataStatus.LIVE
        assert d.zone in zones or d.zone == "unknown"
