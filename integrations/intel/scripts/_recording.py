"""Shared MP4 recorder for the sim demos (2026-09-14).

Wraps the world's ``_mujoco`` handle so every physics step is seen, renders a
scene camera every ``every`` steps (20 steps = 40 ms of sim = 25 fps at the
2 ms timestep, i.e. real-time playback), and writes H.264-compatible MP4 via
OpenCV. Videos are demo material, not evidence: keep them out of the repo.

    rec = Recorder(world, "third_person", "C:/.../run.mp4")
    world._mujoco = rec.spy()
    ...run...
    rec.close()
"""
from __future__ import annotations

from pathlib import Path


def draw_text_block(bgr, lines, x, y, *, scales=None, color=(255, 255, 255), alpha=0.55, pad=10):
    """Lines of text over a translucent dark panel, single-pass glyphs.
    (OpenCV 5 advances thick and thin strokes differently, so the usual
    thick-black-then-thin-white outline leaves a ghost of the last glyphs.)
    ``x=None`` centres the block horizontally."""
    import cv2  # noqa: PLC0415

    scales = scales or [0.6] * len(lines)
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, sc, 1)[0] for t, sc in zip(lines, scales)]
    width = max(w for w, _ in sizes)
    height = sum(h for _, h in sizes) + 10 * len(lines)
    if x is None:
        x = max(pad, (bgr.shape[1] - width) // 2)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(bgr.shape[1], x + width + pad), min(bgr.shape[0], y + height + pad)
    panel = bgr[y0:y1, x0:x1]
    panel[:] = (panel * (1.0 - alpha)).astype(panel.dtype)
    cy = y
    for text, sc, (_, h) in zip(lines, scales, sizes):
        cy += h
        cv2.putText(bgr, text, (x, cy), font, sc, color, 1, cv2.LINE_AA)
        cy += 10


_FIXED_FREE: dict[str, str] = {}   # free-camera spec -> fixed camera name compiled into the model


def free_camera_frame(spec: str):
    """pos and xyaxes of a MuJoCo free camera 'free:az,el,dist,lx,ly,lz'
    (MuJoCo's convention: forward = (cos el cos az, cos el sin az, sin el))."""
    import math
    az, el, dist, lx, ly, lz = (float(v) for v in spec.split(":", 1)[1].split(","))
    a, e = math.radians(az), math.radians(el)
    fwd = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
    up = (-math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), math.cos(e))
    pos = (lx - dist * fwd[0], ly - dist * fwd[1], lz - dist * fwd[2])
    z = (-fwd[0], -fwd[1], -fwd[2])
    x = (up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0])
    n = math.sqrt(sum(v * v for v in x)); x = tuple(v / n for v in x)
    y = (z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0])
    return pos, (*x, *y)


def install_fixed_director_cameras(specs) -> None:
    """Compile the recording scripts' free 'director' views into the table
    model as fixed cameras (process-local wrap of the model loader). Reason,
    2026-09-15: with a VLA policy rendering its own cameras in the same
    process, the free-camera (MjvCamera) renders went black from the first
    policy step while every fixed camera stayed fine; a fixed camera at the
    same pose sidesteps it. Recorder._render uses the fixed camera when the
    spec is registered and falls back to the free camera otherwise."""
    import hashlib
    import xml.etree.ElementTree as ET
    from omni_q import intel_sim

    names = {}
    for spec in specs:
        if spec.startswith("free:") and spec not in _FIXED_FREE:
            names[spec] = "director_" + hashlib.sha1(spec.encode()).hexdigest()[:6]
    if not names:
        return
    inner = intel_sim.load_dual_so101_model

    def load(config=None):
        import mujoco
        xml = intel_sim.dual_so101_xml(config)
        root = ET.fromstring(xml)
        wb = root.find("worldbody")
        for spec, name in names.items():
            pos, xy = free_camera_frame(spec)
            ET.SubElement(wb, "camera", {"name": name, "pos": "%.5f %.5f %.5f" % pos,
                                         "xyaxes": " ".join("%.5f" % v for v in xy), "fovy": "45"})
        assets = {f"assets/{path.name}": path.read_bytes() for path in intel_sim.ARM_ASSETS.glob("*.stl")}
        return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets=assets)
    intel_sim.load_dual_so101_model = load
    _FIXED_FREE.update(names)


class _FfmpegWriter:
    """H.264 through the ffmpeg binary that imageio-ffmpeg ships (OpenCV's
    own mp4v left ghosts of earlier HUD text at its default bitrate and this
    build has no openh264). Same write()/release() shape as cv2.VideoWriter."""

    def __init__(self, path, fps: int, size) -> None:
        import subprocess
        import imageio_ffmpeg  # noqa: PLC0415

        w, h = size
        self._proc = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(path)],
            stdin=subprocess.PIPE)

    def write(self, bgr) -> None:
        self._proc.stdin.write(bgr.tobytes())

    def release(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait()


def _open_writer(path, fps: int, size):
    try:
        import imageio_ffmpeg  # noqa: F401,PLC0415
        return _FfmpegWriter(path, fps, size)
    except Exception:  # noqa: BLE001 - no bundled ffmpeg: OpenCV's mp4v
        import cv2  # noqa: PLC0415
        return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


class Recorder:
    def __init__(self, world, camera: str, path, *, every: int = 20, size=(1280, 720), fps: int = 25,
                 label: str | None = None, detector=None, annotate=(), detect_every: int = 2):
        import cv2  # noqa: PLC0415
        import mujoco  # noqa: PLC0415
        from omni_q import gl_safety  # noqa: PLC0415

        gl_safety.install()   # a stale Renderer freed mid-run must not black out this one
        self.world = world
        # one camera name -> single view; a list of 2 or 4 -> side-by-side / 2x2 grid
        self.cameras = [camera] if isinstance(camera, str) else list(camera)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.size = size
        self.label = label
        self._cv2 = cv2
        self._mujoco = mujoco
        w, h = size
        n = len(self.cameras)
        # 1 -> full frame; 2 -> side by side; 3-4 -> 2x2; 5-6 -> 3x2 (all six scene cameras)
        self._tile = (w, h) if n == 1 else ((w // 2, h) if n == 2 else ((w // 2, h // 2) if n <= 4 else (w // 3, h // 2)))
        tw, th = self._tile
        self._renderer = mujoco.Renderer(world.model, height=th, width=tw)
        self.fps = fps
        self._writer = _open_writer(self.path, fps, (w, h))
        self.frames = 0
        self._count = 0
        # vision overlay: run ``detector`` (omni_q.vision.OpenVINODetector) on
        # the tiles named in ``annotate`` and draw boxes/labels/confidence
        self.detector = detector
        self.annotate = set(annotate)
        self.detect_every = detect_every
        self._last_dets: dict = {}
        self.last_bgr = None  # most recent composed frame (montage title/result cards reuse it)
        # on-screen state: the command, what each arm is doing right now, the
        # plan's progress (see attach()); a notice is a one-off banner
        self.hud: list[str] = []
        self.hud_extra: list[str] = []   # appended after attach()'s lines
        self._notice: tuple[str, int] | None = None

    _COLORS = {"plate": (255, 255, 255), "cup": (255, 200, 60), "fork": (120, 220, 255), "spoon": (255, 150, 220),
               "napkin": (90, 90, 255), "drawer": (140, 200, 140), "knife": (200, 200, 200)}

    def _draw_detections(self, camera: str, bgr):
        if self.detector is None or camera not in self.annotate:
            return bgr
        if self.frames % self.detect_every == 0 or camera not in self._last_dets:
            rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
            self._last_dets[camera] = self.detector.detect(rgb)
        for d in self._last_dets[camera]:
            x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
            col = self._COLORS.get(d.cls_name, (0, 255, 0))
            self._cv2.rectangle(bgr, (x1, y1), (x2, y2), col, 2)
            txt = f"{d.cls_name} {d.conf:.2f}"
            (tw, th), _ = self._cv2.getTextSize(txt, self._cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            self._cv2.rectangle(bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), col, -1)
            self._cv2.putText(bgr, txt, (x1 + 2, y1 - 3), self._cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, self._cv2.LINE_AA)
        return bgr

    @staticmethod
    def _free_camera(spec: str):
        """'free:az,el,dist,lx,ly,lz' -> MjvCamera. A render-only director view;
        not a model camera, so it does not count against the sensor budget."""
        import mujoco  # noqa: PLC0415

        az, el, dist, lx, ly, lz = (float(v) for v in spec.split(":", 1)[1].split(","))
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        cam.lookat[:] = (lx, ly, lz)
        return cam

    def _render(self, camera: str, data=None):
        data = self.world.data if data is None else data
        if camera.startswith("free:"):
            fixed = _FIXED_FREE.get(camera)
            import mujoco  # noqa: PLC0415
            if fixed and mujoco.mj_name2id(self.world.model, mujoco.mjtObj.mjOBJ_CAMERA, fixed) >= 0:
                self._renderer.update_scene(data, camera=fixed)
            else:
                self._renderer.update_scene(data, camera=self._free_camera(camera))
        else:
            self._renderer.update_scene(data, camera=camera)
        bgr = self._cv2.cvtColor(self._renderer.render(), self._cv2.COLOR_RGB2BGR)
        bgr = self._draw_detections(camera, bgr)
        if len(self.cameras) > 1:
            name = "director" if camera.startswith("free:") else camera
            if camera in self.annotate and self.detector is not None:
                name += "  [YOLO/OpenVINO]"
            self._cv2.putText(bgr, name, (10, 24), self._cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, self._cv2.LINE_AA)
        return bgr

    def frame(self, data=None) -> None:
        import numpy as np  # noqa: PLC0415

        tiles = [self._render(c, data) for c in self.cameras]
        if len(tiles) == 1:
            bgr = tiles[0]
        elif len(tiles) == 2:
            bgr = np.concatenate(tiles, axis=1)
        elif len(tiles) <= 4:
            while len(tiles) < 4:
                tiles.append(np.zeros_like(tiles[0]))
            bgr = np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:4], axis=1)], axis=0)
        else:
            while len(tiles) < 6:
                tiles.append(np.zeros_like(tiles[0]))
            bgr = np.concatenate([np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:6], axis=1)], axis=0)
        if bgr.shape[1] != self.size[0] or bgr.shape[0] != self.size[1]:
            bgr = self._cv2.resize(bgr, self.size)
        if self.label:
            self._cv2.putText(bgr, self.label, (16, self.size[1] - 16), self._cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, self._cv2.LINE_AA)
        self._draw_hud(bgr)
        self._writer.write(bgr)
        self.last_bgr = bgr
        self.frames += 1

    def _draw_hud(self, bgr) -> None:
        y = 12 if len(self.cameras) == 1 else 34   # under the tile name in a grid
        if self.hud:
            draw_text_block(bgr, self.hud, 16, y, scales=[0.7] + [0.55] * (len(self.hud) - 1))
        if self._notice is not None:
            text, until = self._notice
            if self.frames <= until:
                draw_text_block(bgr, [text], None, self.size[1] - 90, scales=[0.75], color=(80, 220, 255), alpha=0.75)
            else:
                self._notice = None

    def notify(self, text: str, seconds: float = 4.0) -> None:
        """Show a one-off banner (an operator command, a fault) for ``seconds`` of video."""
        self._notice = (text, self.frames + int(seconds * self.fps))

    def attach(self, world, command: str, objects=("plate_1", "cup_1", "fork_1", "spoon_1", "napkin_1")) -> None:
        """Wrap the world's transition entry points so the HUD shows the
        command, each arm's current action and the plan's progress. Wrap
        BEFORE any demo-specific hooks so those compose on top."""
        rec = self
        rec.command = command
        state = {"left": "ready", "right": "ready", "placed": []}
        short = {o: o.split("_")[0] for o in objects}

        def text(req):
            obj = short.get(req.args.get("object"), req.args.get("object") or "")
            to = req.args.get("to")
            return f"{req.op} {obj}" + (f" -> {to}" if to and req.op in {"MOVE", "PLACE"} else "")

        def arm_of(req):
            return "right" if (req.actor and "right" in req.actor) else "left"

        def arm_status(arm: str) -> str:
            """A physically failed arm, or one withdrawn by the operator, says so
            on every frame -- not only in the few-second notice."""
            failed = getattr(world, "_failed_arms", None) or {}
            off = 0 if arm == "left" else 6
            if off in failed:
                return f"FAILED - {failed[off].get('reason', 'no response')} -> withdrawn from authority; last: {state[arm]}"
            eng = getattr(world, "_engine", None)
            pending = list(getattr(eng, "_pending_constraints", []) or []) + list(getattr(world, "_constraints", ()) or ())
            for c in pending:
                if getattr(c, "kind", "") == "prefer_arm" and getattr(c, "value", None) not in (None, arm) and getattr(c, "source", "") == "operator"                         and "voice" in str(getattr(c, "justification", "") or ""):
                    return f"WITHDRAWN by operator (voice: \"don't use the {arm} arm anymore\"); last: {state[arm]}"
            return state[arm]

        def render():
            done = " ".join(short[o] for o in objects if o in state["placed"]) or "-"
            todo = " ".join(short[o] for o in objects if o not in state["placed"]) or "-"
            lines = [f'command: "{rec.command}"',
                     f"left arm:  {arm_status('left')}",
                     f"right arm: {arm_status('right')}",
                     f"placed: {done}   remaining: {todo}"]
            if "vla_done" in state:
                lines.append(f"VLA-led steps: {state['vla_done'] + state['vla_fallback']}   finished by the policy: {state['vla_done']}   finished by the governed primitive: {state['vla_fallback']}")
            rec.hud = lines + list(rec.hud_extra)

        def begin(reqs, paired):
            for req in reqs:
                state[arm_of(req)] = text(req) + ("   [both arms at once]" if paired else "")
            render()

        def end(reqs, results):
            for req, res in zip(reqs, results):
                ok = getattr(res, "ok", False)
                if ok and req.op in {"MOVE", "PLACE"} and req.args.get("object") in short:
                    state["placed"].append(req.args.get("object"))
                detail = getattr(res, "detail", None) or {}
                tag = ""
                if detail.get("grasp") == "vla_smolvla":
                    tag = "  [VLA: SmolVLA]"
                elif detail.get("fallback"):
                    tag = "  [VLA-led -> governed completion]"
                elif detail.get("grasp") == "bimanual_edge":
                    tag = "  [two-arm primitive]"
                state[arm_of(req)] = ("done: " if ok else "FAILED: ") + text(req) + tag
                if detail.get("grasp") == "vla_smolvla" or detail.get("fallback"):
                    state["vla_done"] = state.get("vla_done", 0) + int(detail.get("grasp") == "vla_smolvla")
                    state["vla_fallback"] = state.get("vla_fallback", 0) + int(bool(detail.get("fallback")))
            render()

        orig, orig_par = world.apply_transition, world.apply_transitions_parallel

        def apply(req):
            begin([req], False)
            res = orig(req)
            end([req], [res])
            return res

        def apply_pair(reqs):
            begin(reqs, True)
            out = orig_par(reqs)
            end(reqs, out)
            return out
        world.apply_transition = apply
        world.apply_transitions_parallel = apply_pair
        rec.render_hud = render
        render()

    def spy(self):
        """A drop-in for the world's ``mujoco`` module that records a frame
        every ``every`` physics steps.

        Rendering must happen on the thread that owns the GL context (the
        main thread; rendering from another thread produces black frames).
        When both arms run at once the world steps physics from worker
        threads, so a step that lands on a frame boundary snapshots the
        physics state instead and ``main_thread_pump`` -- called by
        ``IntelTableWorld.apply_transitions_parallel`` while it waits --
        renders the queued snapshots in order.
        """
        import collections
        import threading
        import time

        real = self.world._mujoco
        rec = self
        main = threading.main_thread()
        queue: collections.deque = collections.deque()
        lock = threading.Lock()

        class Spy:
            def __getattr__(self, n):
                return getattr(real, n)

            def mj_step(self, m, d, nstep=1):
                for _ in range(nstep):
                    real.mj_step(m, d)
                    rec._count += 1
                    if rec._count % rec.every == 0:
                        if threading.current_thread() is main:
                            rec.frame()
                        else:
                            snap = rec._mujoco.MjData(m)
                            rec._mujoco.mj_copyData(snap, m, d)
                            with lock:
                                queue.append(snap)
                            # backpressure, not dropping: a 4-tile render is
                            # slower than paired physics and dropped frames
                            # made grid clips play the paired segments fast
                            # (56 s vs 115 s for the same run, 2026-09-14)
                            while len(queue) > 40:
                                time.sleep(0.002)

            def main_thread_pump(self):
                while True:
                    with lock:
                        if not queue:
                            return
                        snap = queue.popleft()
                    rec.frame(data=snap)
        return Spy()

    def close(self) -> str:
        self._writer.release()
        # free the GL context: a second live Renderer in the process renders black
        try:
            self._renderer.close()
        except Exception:  # noqa: BLE001
            pass
        return f"{self.path} ({self.frames} frames, {self.frames / 25:.0f} s)"
