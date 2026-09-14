"""Synthetic YOLO dataset from the MuJoCo table scene, auto-labelled.

The previously published detector (`table_yolo_v2_ft_2026-09-11`) finds
nothing on the current overhead render -- not on the realistic tableware
and not on the legacy proxies either, even at conf 0.05 (2026-09-14). The
scene it was trained on is not this scene. This regenerates the data from
the scene itself: every image is a real render of the simulator, and every
label is the projection of the object's own geometry through the same
camera the detector will be run on, so there is no hand labelling and no
mismatch between what the sim shows and what the label says.

Variation per image: scene seed (object positions and yaw, tableware colour,
light direction and intensity via IntelSceneConfig), a random subset of
objects moved to their set-table zones (so "before" and "after" states are
both covered), and both cameras (overhead and third-person).

    python integrations/intel/scripts/make_table_yolo_dataset.py --out data/table_yolo_v3 --n 400
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

from omni_q.intel_sim import ZONE_POSITIONS, IntelSceneConfig, IntelTableWorld  # noqa: E402

CLASSES = {"plate_1": 0, "cup_1": 1, "fork_1": 2, "spoon_1": 3, "napkin_1": 5}   # 4 = knife (absent)
NAMES = {0: "plate", 1: "cup", 2: "fork", 3: "spoon", 4: "knife", 5: "napkin"}


def project(points_world, cam_pos, cam_rot, fovy_deg, width, height):
    """World points -> pixel (u, v). Same pinhole convention as vision.project_to_table."""
    rel = (np.asarray(points_world) - cam_pos) @ cam_rot          # into the camera frame
    z = -rel[:, 2]
    ok = z > 1e-6
    half_h = math.tan(math.radians(fovy_deg) / 2)
    half_w = half_h * (width / height)
    ndc_x = rel[:, 0] / np.where(ok, z, 1.0) / half_w
    ndc_y = rel[:, 1] / np.where(ok, z, 1.0) / half_h
    u = (ndc_x + 1) / 2 * width
    v = (1 - ndc_y) / 2 * height
    return u[ok], v[ok]


def body_bbox(model, data, body_name, cam_pos, cam_rot, fovy, width, height):
    body_id = model.body(body_name).id
    corners = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id:
            continue
        c = data.geom_xpos[g]
        R = data.geom_xmat[g].reshape(3, 3)
        s = model.geom_size[g]
        t = int(model.geom_type[g])
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            half = s[:3]
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            half = np.array([s[0], s[0], s[1]])
        elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            half = s[:3]
        else:
            half = np.array([s[0]] * 3)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners.append(c + R @ (half * np.array([sx, sy, sz])))
    u, v = project(np.array(corners), cam_pos, cam_rot, fovy, width, height)
    if len(u) == 0:
        return None
    x1, x2 = float(np.clip(u.min(), 0, width)), float(np.clip(u.max(), 0, width))
    y1, y2 = float(np.clip(v.min(), 0, height)), float(np.clip(v.max(), 0, height))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return x1, y1, x2, y2


def teleport(world, obj, xy, yaw):
    """Dataset generation only: place an object at a zone for a 'set table'
    frame. Never used by the controller (the no-cheating rule is about runs)."""
    adr, _ = world._object_joints[obj]
    z = float(world.data.qpos[adr + 2])
    world.data.qpos[adr:adr + 3] = [xy[0], xy[1], z + 0.002]
    q = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    world.data.qpos[adr + 3:adr + 7] = q
    world.data.qvel[:] = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/table_yolo_v3"))
    ap.add_argument("--n", type=int, default=400, help="number of scene seeds (x2 cameras)")
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--size", default="640x480")
    args = ap.parse_args()
    width, height = (int(v) for v in args.size.lower().split("x"))

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed0)
    renderer = None  # one renderer per process: a second live Renderer renders black
    zone_of = {"plate_1": "center", "cup_1": "upper_right", "fork_1": "left", "spoon_1": "right", "napkin_1": "lower_left"}
    n_written = 0
    for i in range(args.n):
        seed = args.seed0 + i
        world = IntelTableWorld(IntelSceneConfig(randomized=True, seed=seed))
        model, data = world.model, world.data
        for _ in range(200):
            mujoco.mj_step(model, data)
        # some objects already "set": a random subset moved to their zones
        k = rng.choice([0, 0, 1, 2, 3, 5])
        for obj in rng.sample(list(zone_of), k):
            zx, zy, _ = ZONE_POSITIONS[zone_of[obj]]
            teleport(world, obj, (zx + rng.uniform(-0.02, 0.02), zy + rng.uniform(-0.02, 0.02)), rng.uniform(-math.pi, math.pi))
        for _ in range(150):
            mujoco.mj_step(model, data)
        # occasionally park an arm over the table so gripper pixels appear near objects
        if rng.random() < 0.3:
            arm = rng.choice([0, 6])
            data.ctrl[arm:arm + 5] = [rng.uniform(-0.6, 0.6), rng.uniform(-1.2, -0.6), rng.uniform(1.2, 2.2), rng.uniform(-1.0, 0.5), rng.uniform(-1.6, 1.6)]
            for _ in range(300):
                mujoco.mj_step(model, data)
        split = "val" if rng.random() < args.val_frac else "train"
        if renderer is None:
            renderer = mujoco.Renderer(model, height=height, width=width)
        for cam_name in ("table_overhead", "third_person"):
            renderer.update_scene(data, camera=cam_name)
            frame = np.asarray(renderer.render()).copy()
            cam_id = model.camera(cam_name).id
            pos = data.cam_xpos[cam_id].copy()
            rot = data.cam_xmat[cam_id].reshape(3, 3).copy()
            fovy = float(model.cam_fovy[cam_id])
            lines = []
            for obj, cls in CLASSES.items():
                bb = body_bbox(model, data, obj, pos, rot, fovy, width, height)
                if bb is None:
                    continue
                x1, y1, x2, y2 = bb
                lines.append(f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
            stem = f"s{seed}_{cam_name}"
            Image.fromarray(frame).save(args.out / "images" / split / f"{stem}.png")
            (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_written += 1
        if i % 50 == 0:
            print(f"{i}/{args.n} scenes, {n_written} images", flush=True)
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n" +
        "".join(f"  {k}: {v}\n" for k, v in NAMES.items()))
    print(f"wrote {n_written} images to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
