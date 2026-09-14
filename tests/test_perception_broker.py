from __future__ import annotations

import threading

import pytest

from omni_q.perception_broker import (
    CameraDetection,
    CameraPoint,
    JointVisionNode,
    OmniPerceptionBroker,
    OmniVisionNode,
    PinholeCameraModel,
)


class Camera:
    def __init__(self, value):
        self.value = value

    def capture(self):
        return self.value


class Detector:
    def __init__(self, label, barrier=None):
        self.label = label
        self.barrier = barrier

    def detect(self, frame):
        if self.barrier:
            self.barrier.wait(timeout=1)
        return [CameraDetection(self.label, 0.9, CameraPoint(0.1, -0.2, 0.8))]


class Transform:
    def __init__(self, value):
        self.value = value

    def __matmul__(self, other):
        return Transform(f"{self.value}@{other.value}")

    def __eq__(self, other):
        return isinstance(other, Transform) and self.value == other.value


class State:
    frame = 42
    timestamp = 123.5

    def joint_position(self, arm, joint):
        return {("left", "Wrist_Pitch"): 0.75}[(arm, joint)]

    def link_transform(self, arm, joint):
        return Transform(f"world:{arm}:{joint}")


def make_broker(*, barrier=None, resolver=None):
    joint = JointVisionNode(
        "left.cam.wrist_pitch",
        Camera("joint-frame"),
        Detector("fork", barrier),
        arm="left",
        joint="Wrist_Pitch",
        mount_transform=Transform("mount"),
    )
    global_node = OmniVisionNode(
        Camera("global-frame"), Detector("plate", barrier), camera_pose="world:global"
    )
    return OmniPerceptionBroker([joint], global_node, resolver=resolver)


def test_broker_preserves_joint_and_global_provenance():
    batch = make_broker().observe(State())

    assert batch.frame_id == 42
    assert [packet.source_id for packet in batch.packets] == [
        "left.cam.wrist_pitch", "omni.global"
    ]
    joint, global_packet = batch.packets
    assert joint.role == "joint"
    assert joint.joint_position == 0.75
    assert joint.camera_pose == Transform("world:left:Wrist_Pitch@mount")
    assert joint.detections == (
        CameraDetection("fork", 0.9, CameraPoint(0.1, -0.2, 0.8)),
    )
    assert global_packet.role == "global"
    assert global_packet.arm is global_packet.joint is None
    assert global_packet.camera_pose == "world:global"


def test_nodes_can_be_toggled_by_stable_source_id():
    broker = make_broker()
    broker.disable("left.cam.wrist_pitch")
    assert broker.enabled_sources() == ("omni.global",)
    assert [p.source_id for p in broker.observe(State()).packets] == ["omni.global"]
    broker.enable("left.cam.wrist_pitch")
    assert len(broker.observe(State()).packets) == 2
    with pytest.raises(KeyError, match="unknown perception source"):
        broker.disable("missing")


def test_each_detector_runs_in_parallel_but_results_stay_ordered():
    # Both detector threads must rendezvous; sequential execution would time out.
    barrier = threading.Barrier(2)
    batch = make_broker(barrier=barrier).observe(State())
    assert [p.source_id for p in batch.packets] == [
        "left.cam.wrist_pitch", "omni.global"
    ]


def test_resolver_receives_packets_without_losing_provenance():
    broker = make_broker(resolver=lambda packets: {p.source_id: p for p in packets})
    resolved = broker.observe(State())
    assert resolved["left.cam.wrist_pitch"].joint == "Wrist_Pitch"


def test_source_ids_must_be_unique():
    joint = JointVisionNode(
        "omni.global", Camera(1), Detector("x"), arm="left", joint="Elbow"
    )
    with pytest.raises(ValueError, match="unique"):
        OmniPerceptionBroker([joint], OmniVisionNode(Camera(2), Detector("y")))


def test_pinhole_model_maps_yolo_box_center_and_depth_to_camera_xyz():
    model = PinholeCameraModel(fx=400, fy=500, cx=320, cy=240)

    point = model.back_project(u=420, v=140, depth=2.0)

    assert point == CameraPoint(x=0.5, y=-0.4, z=2.0)


@pytest.mark.parametrize(
    ("point", "message"),
    [
        ((0.0, 0.0, 0.0), "positive"),
        ((float("nan"), 0.0, 1.0), "finite"),
    ],
)
def test_camera_coordinates_reject_invalid_depth(point, message):
    with pytest.raises(ValueError, match=message):
        CameraPoint(*point)


def test_node_rejects_unlocalized_2d_yolo_results():
    class BoxOnlyDetector:
        def detect(self, frame):
            return [("fork", (10, 20, 30, 40))]

    node = OmniVisionNode(Camera("frame"), BoxOnlyDetector())

    with pytest.raises(TypeError, match="camera-relative XYZ"):
        node.observe(State())
