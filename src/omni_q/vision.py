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
NMS), not a stub. Its class *labels* are only as good as the weights it is
given: the already-published thermal YOLO used for today's benchmark knows
HIT-UAV classes (Person/Car/Bicycle/...), not tableware, so detections from
it are real model output with honestly meaningless labels for this scene --
swap in the real fine-tuned 7-class weights (OQ-008) and nothing else here
changes.

``project_to_table`` turns a detected pixel center into an approximate
world (x, y) by intersecting the camera's real pinhole ray (from its actual
MuJoCo ``fovy``/position/orientation, not assumed) with the table plane.
This assumes the detected object rests ON the table (true for tableware at
rest, not for something mid-air in a gripper) -- a stated geometric
approximation, not a second ground-truth read of the WorldState.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class CameraSource(Protocol):
    def capture(self):  # -> np.ndarray, HxWx3 uint8 RGB
        ...


class MuJoCoCameraSource:
    """Renders a named camera from a real, already-stepped MuJoCo scene."""

    def __init__(self, model, data, camera_name: str, width: int = 640, height: int = 480):
        import mujoco

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
            meta_path = model_xml_path.parent / "metadata.yaml" if hasattr(model_xml_path, "parent") else None
            self.class_names = {}
            if meta_path is not None and meta_path.exists():
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
