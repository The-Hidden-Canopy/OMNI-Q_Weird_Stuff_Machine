"""Camera + detector perception pipeline for the Intel Online track.

Two swappable halves, neither one aware of the other's implementation:

    CameraSource.capture() -> HxWx3 RGB frame
    Detector.detect(frame) -> list[RawDetection]  (pixel space)

``MuJoCoCameraSource`` renders a named camera from the real, already-stepped
MuJoCo scene (simulation-first, per the brief). ``OpenCVCameraSource`` wraps
a real UVC/USB camera via OpenCV for the onsite/real-hardware track -- it is
**not validated against physical hardware in this repo** (no rig available);
it exists so the detector and downstream code never need to change when a
real camera replaces the simulated one, only the ``CameraSource`` swaps.

``OpenVINODetector`` runs a real exported YOLOv8-style OpenVINO IR model
(see ``integrations/intel/scripts/profile_yolo_openvino.py`` for the export
path) -- real inference, real decode (box regression + class sigmoid +
NMS), not a stub. The 7-class tabletop fine-tune (OQ-008,
``models/table_yolo_v2_ft_2026-09-11.pt``, val mAP50 0.324, weights at
`KissTheHabit/yolov8n-table-yolo <https://huggingface.co/KissTheHabit/yolov8n-table-yolo>`_)
is exported for BOTH runtime paths: ``models/table_yolo_v2_ft_2026-09-11.onnx``
and ``models/table_yolo_v2_ft_2026-09-11_openvino_model/`` (IR + the
``metadata.yaml`` this class reads its class names from -- plate, cup,
fork, spoon, knife, napkin, drawer). The already-published thermal
HIT-UAV YOLO remains useful as an export/benchmark stand-in; its labels
are honestly meaningless for a table scene. For the ultralytics/PyTorch
runtime of the same fine-tune, see ``yolo_perception.YoloDetector`` -- it
emits this same ``RawDetection`` contract, so ``as_frame_detector`` and
``FrameObserver`` consume both identically.

``project_to_table`` turns a detected pixel center into an approximate
world (x, y) by intersecting the camera's real pinhole ray (from its actual
MuJoCo ``fovy``/position/orientation, not assumed) with the table plane.
This assumes the detected object rests ON the table (true for tableware at
rest, not for something mid-air in a gripper) -- a stated geometric
approximation, not a second ground-truth read of the WorldState.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class CameraSource(Protocol):
    def capture(self):  # -> np.ndarray, HxWx3 uint8 RGB
        ...


class MuJoCoCameraSource:
    """Renders a named camera from a real, already-stepped MuJoCo scene."""

    def __init__(self, model, data, camera_name: str, width: int = 640, height: int = 480):
        import mujoco

        from . import gl_safety

        gl_safety.install()
        self._mujoco = mujoco
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def capture(self):
        self._renderer.update_scene(self.data, camera=self.camera_name)
        return self._renderer.render()

    def camera_pose(self) -> tuple:
        """(position[3], rotation_3x3 world_from_camera, fovy_deg) for the
        camera this source renders -- the real values MuJoCo is using, read
        fresh each call since the camera may be attached to a moving body."""
        import numpy as np

        cam_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        pos = self.data.cam_xpos[cam_id].copy()
        rot = self.data.cam_xmat[cam_id].reshape(3, 3).copy()
        fovy = float(self.model.cam_fovy[cam_id])
        return pos, rot, fovy


class OpenCVCameraSource:
    """Real camera via OpenCV ``VideoCapture`` (USB/UVC webcam or an
    industrial camera exposing the same driver interface) -- for the Intel
    onsite / real-SO-101 track. **Not exercised against physical hardware
    in this repo**; no rig available here. Exists so the rest of the vision
    pipeline (``Detector``, ``project_to_table``) is identical whether the
    frame came from MuJoCo or a real lens -- only this class changes."""

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        import cv2

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera device {device_index}")

    def capture(self):
        ok, frame_bgr = self._cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        self._cap.release()


@dataclass(frozen=True)
class RawDetection:
    cls_id: int
    cls_name: str
    conf: float
    bbox_xyxy: tuple[float, float, float, float]  # pixel coords in the source frame
    center_xy: tuple[float, float]                # pixel coords in the source frame


class OpenVINODetector:
    """YOLOv8-style detector running a real exported OpenVINO IR model.

    Expects the standard ultralytics export layout: output (1, 4+nc, N) --
    box regression already decoded to input-pixel-space (cx, cy, w, h) and
    class scores already through sigmoid, both baked into the exported
    graph (``end2end: false`` in the model's own ``metadata.yaml``, i.e. no
    NMS baked in -- this class does that part)."""

    def __init__(
        self, model_xml_path, *, device: str = "CPU",
        class_names: dict[int, str] | None = None,
        conf_threshold: float = 0.25, iou_threshold: float = 0.45,
    ) -> None:
        import openvino as ov
        import yaml

        model_xml_path = Path(model_xml_path)  # accept a plain str too, not just Path
        core = ov.Core()
        model = core.read_model(model_xml_path)
        self._compiled = core.compile_model(model, device)
        self._request = self._compiled.create_infer_request()
        input_shape = list(self._compiled.input(0).shape)
        self._imgsz = (int(input_shape[2]), int(input_shape[3]))  # (H, W)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if class_names is not None:
            self.class_names = class_names
        else:
            meta_path = model_xml_path.parent / "metadata.yaml"
            self.class_names = {}
            if meta_path.exists():
                meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
                self.class_names = {int(k): v for k, v in (meta.get("names") or {}).items()}

    def _letterbox(self, frame):
        import cv2
        import numpy as np

        h0, w0 = frame.shape[:2]
        h1, w1 = self._imgsz
        scale = min(h1 / h0, w1 / w0)
        nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((h1, w1, 3), 114, dtype=np.uint8)
        top, left = (h1 - nh) // 2, (w1 - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, scale, left, top

    def detect(self, frame) -> list[RawDetection]:
        import numpy as np

        canvas, scale, pad_x, pad_y = self._letterbox(frame)
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None, ...]  # NCHW

        output = self._request.infer({0: blob})[self._compiled.output(0)]
        preds = np.squeeze(output, axis=0).T  # (N, 4+nc)
        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_id = np.argmax(class_scores, axis=1)
        conf = class_scores[np.arange(len(class_id)), class_id]

        keep = conf > self.conf_threshold
        if not np.any(keep):
            return []
        boxes_cxcywh, class_id, conf = boxes_cxcywh[keep], class_id[keep], conf[keep]

        cx, cy, w, h = boxes_cxcywh.T
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        import cv2

        nms_boxes = np.stack([x1, y1, w, h], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(nms_boxes, conf.tolist(), self.conf_threshold, self.iou_threshold)
        indices = np.array(indices).reshape(-1) if len(indices) else np.array([], dtype=int)

        results: list[RawDetection] = []
        for i in indices:
            bx1, by1, bx2, by2 = (float(v) for v in boxes_xyxy[i])
            # undo letterbox: canvas-space -> original-frame-space
            ox1, oy1 = (bx1 - pad_x) / scale, (by1 - pad_y) / scale
            ox2, oy2 = (bx2 - pad_x) / scale, (by2 - pad_y) / scale
            cid = int(class_id[i])
            results.append(RawDetection(
                cls_id=cid, cls_name=self.class_names.get(cid, f"class_{cid}"),
                conf=round(float(conf[i]), 4),
                bbox_xyxy=(round(ox1, 1), round(oy1, 1), round(ox2, 1), round(oy2, 1)),
                center_xy=(round((ox1 + ox2) / 2, 1), round((oy1 + oy2) / 2, 1)),
            ))
        return results


def project_to_table(
    pixel_xy: tuple[float, float], frame_size: tuple[int, int],
    cam_pos, cam_rot, fovy_deg: float, table_z: float = -0.005,
):
    """Back-project a detected pixel center to an approximate world (x, y)
    by intersecting the camera's real pinhole ray with the table plane
    ``z = table_z``. Real camera geometry (the caller's actual MuJoCo
    fovy/position/orientation) -- the approximation is assuming the object
    rests on the table, not the projection math itself.

    Returns ``None`` if the ray doesn't hit the plane in front of the
    camera (e.g. pixel is above the horizon)."""
    import numpy as np

    width, height = frame_size
    u, v = pixel_xy
    ndc_x = (u / width) * 2 - 1
    ndc_y = 1 - (v / height) * 2  # image row grows down; NDC y grows up
    half_h = np.tan(np.radians(fovy_deg) / 2)
    half_w = half_h * (width / height)
    ray_cam = np.array([ndc_x * half_w, ndc_y * half_h, -1.0])
    ray_cam /= np.linalg.norm(ray_cam)
    ray_world = cam_rot @ ray_cam

    cam_pos = np.asarray(cam_pos, dtype=float)
    if abs(ray_world[2]) < 1e-9:
        return None
    t = (table_z - cam_pos[2]) / ray_world[2]
    if t <= 0:
        return None
    hit = cam_pos + t * ray_world
    return float(hit[0]), float(hit[1])


# ---------------------------------------------------------------------------
# Adapters onto omni_q.frame_observer.FrameObserver
# ---------------------------------------------------------------------------
#
# FrameObserver (OQ-004 gap / OQ-008, added independently the same session)
# implements the full Observe contract -- stable per-object ids across
# frames via IoU tracking, zone assignment, Observation construction -- and
# is deliberately built around two injectable seams so its own StubDetector
# and grid_zone_map placeholders can be swapped for something real without
# touching FrameObserver itself:
#
#     Detector  = Callable[[frame], list[Detection2D]]   (normalised [0,1] xyxy)
#     ZoneMap   = Callable[[cx, cy], str]
#
# These two functions ARE that real swap-in, built from the pieces above:
# real MuJoCo-rendered frames, real OpenVINO inference, and real camera-
# geometry back-projection in place of a coarse 3x3 image-grid guess.


def as_frame_detector(detector: "OpenVINODetector", frame_size: tuple[int, int]):
    """Wrap an ``OpenVINODetector`` (pixel-space ``RawDetection``) as a
    ``frame_observer.Detector`` (normalised-``[0,1]`` ``Detection2D``) --
    the seam ``FrameObserver`` was built to accept in place of its own
    ``StubDetector``."""
    from .frame_observer import Detection2D

    width, height = frame_size

    def run(frame) -> list:
        raw = detector.detect(frame)
        out = []
        for r in raw:
            x1, y1, x2, y2 = r.bbox_xyxy
            out.append(Detection2D(r.cls_name, r.conf, (x1 / width, y1 / height, x2 / width, y2 / height)))
        return out

    return run


def make_camera_zone_map(cam: MuJoCoCameraSource, zone_positions: dict[str, tuple[float, float, float]],
                         frame_size: tuple[int, int]):
    """A real ``frame_observer.ZoneMap`` grounded in the camera's actual
    pose instead of a coarse image-grid guess: back-projects the normalised
    detection centre through ``project_to_table`` and returns the nearest
    known zone by Euclidean distance in table-plane coordinates.

    ``zone_positions`` is the same real-position table used for tableware
    placement -- see ``intel_sim.ZONE_POSITIONS`` -- so the zone this
    function names for a *detected* object is the same coordinate space as
    where the planner intends to *place* one, not a separate guess."""
    pos, rot, fovy = cam.camera_pose()
    width, height = frame_size

    def zone_map(cx: float, cy: float) -> str:
        world_xy = project_to_table((cx * width, cy * height), frame_size, pos, rot, fovy)
        if world_xy is None or not zone_positions:
            return "unknown"
        best_name, best_dist = "unknown", float("inf")
        for name, (zx, zy, _zz) in zone_positions.items():
            dist = ((world_xy[0] - zx) ** 2 + (world_xy[1] - zy) ** 2) ** 0.5
            if dist < best_dist:
                best_name, best_dist = name, dist
        return best_name

    return zone_map


# ---------------------------------------------------------------------------
# Multi-camera fusion (2026-09-14)
# ---------------------------------------------------------------------------
#
# The scene has several cameras (overhead, third-person, table-grazing, two
# wrist cameras) and one view misses things: the overhead camera never saw
# the spoon parked near the right arm's base, so the camera-driven plan was
# built without it. Each camera's detections are back-projected through that
# camera's own geometry to the table plane (`project_to_table`), merged by
# class and world distance, and re-expressed in a canonical reference camera's
# pixel frame so the existing FrameObserver / zone map / tracker are unchanged.

def project_from_table(world_xy, cam_pos, cam_rot, fovy_deg: float, frame_size: tuple[int, int],
                       table_z: float = -0.005):
    """Inverse of project_to_table for a point on the table plane -> (u, v)."""
    import numpy as np

    width, height = frame_size
    p = np.array([world_xy[0], world_xy[1], table_z], dtype=float)
    rel = np.asarray(cam_rot).T @ (p - np.asarray(cam_pos, dtype=float))
    if rel[2] >= -1e-9:
        return None
    half_h = np.tan(np.radians(fovy_deg) / 2)
    half_w = half_h * (width / height)
    ndc_x = rel[0] / -rel[2] / half_w
    ndc_y = rel[1] / -rel[2] / half_h
    return float((ndc_x + 1) / 2 * width), float((1 - ndc_y) / 2 * height)


class MultiCameraFusion:
    """Run one detector over several MuJoCo cameras and fuse the results.

    ``cameras``: dict name -> MuJoCoCameraSource (all the same frame size).
    ``reference``: the camera whose pixel frame the fused detections are
    expressed in (the zone map and tracker keep working on that frame).
    Returns ``frame_observer.Detection2D`` in normalised reference coords.
    ``last`` keeps the per-camera raw view for receipts.
    """

    # Height of each class's visual centre above the table. A detection's
    # centre is back-projected onto the plane at that height, not the table:
    # from an oblique camera the 90 mm cup's centre landed 7 cm too far
    # along the ray when the table plane was assumed.
    # napkin: standing fan fold, 70 mm tall (2026-09-14), box centre ~35 mm up
    CLASS_CENTRE_Z = {"cup": 0.045, "plate": 0.020, "fork": 0.010, "spoon": 0.010, "napkin": 0.035, "drawer": 0.015}

    def __init__(self, detector: "OpenVINODetector", cameras: dict, *, reference: str,
                 frame_size: tuple[int, int], merge_radius_m: float = 0.06, table_z: float = -0.005,
                 min_votes: int = 2, single_view_min_conf: float = 0.9):
        self.detector = detector
        self.cameras = cameras
        self.reference = reference
        self.frame_size = frame_size
        self.merge_radius_m = merge_radius_m
        self.table_z = table_z
        # A detection seen by one camera only must be confident; two cameras
        # agreeing on class and place is accepted at the normal threshold.
        self.min_votes = min_votes
        self.single_view_min_conf = single_view_min_conf
        self.last: dict = {}

    def __call__(self, _frame=None) -> list:
        import numpy as np
        from .frame_observer import Detection2D

        width, height = self.frame_size
        hits = []          # (cls, conf, world_xy, camera)
        per_camera = {}
        for name, cam in self.cameras.items():
            frame = cam.capture()
            pos, rot, fovy = cam.camera_pose()
            rows = []
            for r in self.detector.detect(frame):
                plane_z = self.table_z + self.CLASS_CENTRE_Z.get(r.cls_name, 0.0)
                wxy = project_to_table(r.center_xy, self.frame_size, pos, rot, fovy, plane_z)
                if wxy is None:
                    continue
                hits.append((r.cls_name, float(r.conf), wxy, name))
                rows.append((r.cls_name, round(float(r.conf), 3), (round(wxy[0], 3), round(wxy[1], 3))))
            per_camera[name] = rows
        # greedy merge: highest confidence first, absorb same-class hits nearby
        hits.sort(key=lambda h: -h[1])
        fused = []
        for cls, conf, wxy, cam in hits:
            for f in fused:
                if f["cls"] == cls and np.hypot(f["xy"][0] - wxy[0], f["xy"][1] - wxy[1]) <= self.merge_radius_m:
                    # confidence-weighted position over the agreeing cameras:
                    # one oblique view can be 3-4 cm off, the mean is not
                    w0, w1 = f["weight"], conf
                    f["xy"] = ((f["xy"][0] * w0 + wxy[0] * w1) / (w0 + w1), (f["xy"][1] * w0 + wxy[1] * w1) / (w0 + w1))
                    f["weight"] = w0 + w1
                    f["votes"] += 1
                    f["cameras"].append(cam)
                    break
            else:
                fused.append({"cls": cls, "conf": conf, "xy": wxy, "weight": conf, "votes": 1, "cameras": [cam]})
        fused = [f for f in fused if f["votes"] >= self.min_votes or f["conf"] >= self.single_view_min_conf]
        ref = self.cameras[self.reference]
        pos, rot, fovy = ref.camera_pose()
        out = []
        for f in fused:
            uv = project_from_table(f["xy"], pos, rot, fovy, self.frame_size, self.table_z)
            if uv is None:
                continue
            u, v = uv
            r = 12.0  # synthetic box: the tracker only needs a stable centre and IoU overlap frame to frame
            out.append(Detection2D(f["cls"], min(1.0, f["conf"]),
                                   ((u - r) / width, (v - r) / height, (u + r) / width, (v + r) / height)))
            f["ref_uv"] = (round(u, 1), round(v, 1))
        self.last = {"per_camera": per_camera, "fused": [
            {"cls": f["cls"], "conf": round(f["conf"], 3), "world_xy": (round(f["xy"][0], 3), round(f["xy"][1], 3)),
             "votes": f["votes"], "cameras": f["cameras"]} for f in fused]}
        return out
