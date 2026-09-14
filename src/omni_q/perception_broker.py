"""Provenance-preserving perception for kinematically mounted cameras.

This module sits *below* the frozen :class:`~omni_q.contracts.Observation`
boundary.  It deliberately does not fuse detections into a synthetic image:
each detector result remains attached to the camera, link pose, and robot-state
sample that produced it.  A resolver may subsequently associate objects across
cameras and emit the compact public observation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Literal, Protocol, Sequence


@dataclass(frozen=True)
class CameraPoint:
    """Metric XYZ position in the observing camera's optical frame.

    The convention is ``+x`` right, ``+y`` down, and ``+z`` forward.  Keeping
    this value camera-relative is important: consumers can transform it into
    the world with the packet's capture-time ``camera_pose`` without confusing
    measurements from moving cameras.
    """

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x, self.y, self.z)):
            raise ValueError("camera coordinates must be finite")
        if self.z <= 0:
            raise ValueError("camera z must be positive (in front of the camera)")


@dataclass(frozen=True)
class CameraDetection:
    """A YOLO result localized in three dimensions relative to its camera."""

    label: str
    confidence: float
    position: CameraPoint
    box_xyxy: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("detection label must be non-empty")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("detection confidence must be between 0 and 1")
        if self.box_xyxy is not None:
            x1, y1, x2, y2 = self.box_xyxy
            if not all(isfinite(value) for value in self.box_xyxy) or x2 < x1 or y2 < y1:
                raise ValueError("box_xyxy must be finite and ordered")


@dataclass(frozen=True)
class PinholeCameraModel:
    """Calibrated intrinsics for back-projecting YOLO pixels plus depth."""

    fx: float
    fy: float
    cx: float
    cy: float

    def back_project(self, u: float, v: float, depth: float) -> CameraPoint:
        """Map image pixel ``(u, v)`` and metric depth to camera-frame XYZ."""
        if not all(isfinite(value) for value in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")
        return CameraPoint(
            x=(u - self.cx) * depth / self.fx,
            y=(v - self.cy) * depth / self.fy,
            z=depth,
        )


class Camera(Protocol):
    """Minimal camera interface used by a perception node."""

    def capture(self) -> Any: ...


class Detector(Protocol):
    """Camera-owned YOLO runtime with depth localization.

    Implementations combine each 2-D YOLO box with aligned depth (or another
    calibrated range source) and return metric camera-frame measurements.
    RGB-only YOLO boxes cannot supply a physically meaningful ``z`` value.
    """

    def detect(self, frame: Any) -> Sequence[CameraDetection]: ...


class RobotStateSample(Protocol):
    """Kinematic state sampled for the frame being observed."""

    frame: int
    timestamp: float

    def joint_position(self, arm: str, joint: str) -> float: ...

    def link_transform(self, arm: str, joint: str) -> Any: ...


@dataclass(frozen=True)
class PerceptionPacket:
    """One camera's detections with their complete sensor provenance."""

    source_id: str
    role: Literal["joint", "global"]
    arm: str | None
    joint: str | None
    frame_id: int
    timestamp: float
    joint_position: float | None
    camera_pose: Any | None
    detections: tuple[CameraDetection, ...]


@dataclass(frozen=True)
class PerceptionBatch:
    """Default broker result when no cross-camera resolver is installed."""

    frame_id: int
    timestamp: float
    packets: tuple[PerceptionPacket, ...]


class PerceptionNode(Protocol):
    source_id: str
    enabled: bool

    def observe(self, robot_state: RobotStateSample) -> PerceptionPacket | None: ...


def _camera_detections(detector: Detector, frame: Any) -> tuple[CameraDetection, ...]:
    detections = tuple(detector.detect(frame))
    if not all(isinstance(detection, CameraDetection) for detection in detections):
        raise TypeError(
            "detectors must return CameraDetection values with camera-relative XYZ"
        )
    return detections


class JointVisionNode:
    """Camera and private detector runtime mounted on a robot link."""

    def __init__(
        self,
        source_id: str,
        camera: Camera,
        detector: Detector,
        *,
        arm: str,
        joint: str,
        mount_transform: Any | None = None,
    ) -> None:
        if not source_id or not arm or not joint:
            raise ValueError("source_id, arm, and joint must be non-empty")
        self.source_id = source_id
        self.camera = camera
        self.detector = detector
        self.arm = arm
        self.joint = joint
        self.mount_transform = mount_transform
        self.enabled = True

    def observe(self, robot_state: RobotStateSample) -> PerceptionPacket | None:
        if not self.enabled:
            return None

        # robot_state is one coherent sample supplied to every node.  Camera
        # drivers should capture on that sample's trigger (or expose its frame
        # timestamp) rather than asking the live robot for state a second time.
        frame = self.camera.capture()
        joint_pose = robot_state.link_transform(self.arm, self.joint)
        camera_pose = (
            joint_pose @ self.mount_transform
            if self.mount_transform is not None
            else joint_pose
        )
        return PerceptionPacket(
            source_id=self.source_id,
            role="joint",
            arm=self.arm,
            joint=self.joint,
            frame_id=robot_state.frame,
            timestamp=robot_state.timestamp,
            joint_position=robot_state.joint_position(self.arm, self.joint),
            camera_pose=camera_pose,
            detections=_camera_detections(self.detector, frame),
        )


class OmniVisionNode:
    """Static/global OMNI camera with its own detector runtime."""

    def __init__(
        self,
        camera: Camera,
        detector: Detector,
        *,
        source_id: str = "omni.global",
        camera_pose: Any | Callable[[RobotStateSample], Any] | None = None,
    ) -> None:
        if not source_id:
            raise ValueError("source_id must be non-empty")
        self.source_id = source_id
        self.camera = camera
        self.detector = detector
        self.camera_pose = camera_pose
        self.enabled = True

    def observe(self, robot_state: RobotStateSample) -> PerceptionPacket | None:
        if not self.enabled:
            return None
        frame = self.camera.capture()
        pose = self.camera_pose(robot_state) if callable(self.camera_pose) else self.camera_pose
        return PerceptionPacket(
            source_id=self.source_id,
            role="global",
            arm=None,
            joint=None,
            frame_id=robot_state.frame,
            timestamp=robot_state.timestamp,
            joint_position=None,
            camera_pose=pose,
            detections=_camera_detections(self.detector, frame),
        )


Resolver = Callable[[tuple[PerceptionPacket, ...]], Any]


class OmniPerceptionBroker:
    """Run subordinate camera/YOLO nodes and preserve their provenance.

    Nodes execute concurrently by default.  Output order nevertheless follows
    construction order, making receipts and tests deterministic.  The broker
    remains the single ``intel.perception``-level device; its nodes are not
    planner-routable devices.
    """

    def __init__(
        self,
        joint_nodes: Sequence[JointVisionNode],
        global_node: OmniVisionNode,
        *,
        resolver: Resolver | None = None,
        parallel: bool = True,
    ) -> None:
        self.joint_nodes = tuple(joint_nodes)
        self.global_node = global_node
        self.resolver = resolver
        self.parallel = parallel
        self._nodes: tuple[PerceptionNode, ...] = (*self.joint_nodes, global_node)
        ids = [node.source_id for node in self._nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("perception node source_id values must be unique")
        self._by_id = {node.source_id: node for node in self._nodes}

    def enable(self, source_id: str) -> None:
        self._node(source_id).enabled = True

    def disable(self, source_id: str) -> None:
        self._node(source_id).enabled = False

    def enabled_sources(self) -> tuple[str, ...]:
        return tuple(node.source_id for node in self._nodes if node.enabled)

    def _node(self, source_id: str) -> PerceptionNode:
        try:
            return self._by_id[source_id]
        except KeyError as exc:
            raise KeyError(f"unknown perception source: {source_id}") from exc

    def observe(self, robot_state: RobotStateSample) -> Any:
        nodes = tuple(node for node in self._nodes if node.enabled)
        if self.parallel and len(nodes) > 1:
            with ThreadPoolExecutor(max_workers=len(nodes), thread_name_prefix="omni-vision") as pool:
                packets = tuple(pool.map(lambda node: node.observe(robot_state), nodes))
        else:
            packets = tuple(node.observe(robot_state) for node in nodes)
        present = tuple(packet for packet in packets if packet is not None)
        return self.resolve(present, robot_state)

    def resolve(
        self,
        packets: tuple[PerceptionPacket, ...],
        robot_state: RobotStateSample,
    ) -> Any:
        """Associate/arbitrate packets, or return an unmodified packet batch."""
        if self.resolver is not None:
            return self.resolver(packets)
        return PerceptionBatch(robot_state.frame, robot_state.timestamp, packets)
