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


class Recorder:
    def __init__(self, world, camera: str, path, *, every: int = 20, size=(1280, 720), fps: int = 25,
                 label: str | None = None):
        import cv2  # noqa: PLC0415
        import mujoco  # noqa: PLC0415

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
        self._tile = (w, h) if n == 1 else ((w // 2, h) if n == 2 else (w // 2, h // 2))
        tw, th = self._tile
        self._renderer = mujoco.Renderer(world.model, height=th, width=tw)
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self.frames = 0
        self._count = 0

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

    def _render(self, camera: str):
        if camera.startswith("free:"):
            self._renderer.update_scene(self.world.data, camera=self._free_camera(camera))
        else:
            self._renderer.update_scene(self.world.data, camera=camera)
        bgr = self._cv2.cvtColor(self._renderer.render(), self._cv2.COLOR_RGB2BGR)
        if len(self.cameras) > 1:
            name = "director" if camera.startswith("free:") else camera
            self._cv2.putText(bgr, name, (10, 24), self._cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, self._cv2.LINE_AA)
        return bgr

    def frame(self) -> None:
        import numpy as np  # noqa: PLC0415

        tiles = [self._render(c) for c in self.cameras]
        if len(tiles) == 1:
            bgr = tiles[0]
        elif len(tiles) == 2:
            bgr = np.concatenate(tiles, axis=1)
        else:
            bgr = np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:4], axis=1)], axis=0)
        if self.label:
            self._cv2.putText(bgr, self.label, (16, self.size[1] - 16), self._cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, self._cv2.LINE_AA)
        self._writer.write(bgr)
        self.frames += 1

    def spy(self):
        real = self.world._mujoco
        rec = self

        class Spy:
            def __getattr__(self, n):
                return getattr(real, n)

            def mj_step(self, m, d, nstep=1):
                for _ in range(nstep):
                    real.mj_step(m, d)
                    rec._count += 1
                    if rec._count % rec.every == 0:
                        rec.frame()
        return Spy()

    def close(self) -> str:
        self._writer.release()
        # free the GL context: a second live Renderer in the process renders black
        try:
            self._renderer.close()
        except Exception:  # noqa: BLE001
            pass
        return f"{self.path} ({self.frames} frames, {self.frames / 25:.0f} s)"
