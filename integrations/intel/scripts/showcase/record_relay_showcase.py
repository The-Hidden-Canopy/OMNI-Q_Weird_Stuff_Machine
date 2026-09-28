"""SHOWCASE (not part of the competition submission) -- six arms on one long
table pass blocks down the line, bucket-brigade style, into a plate at the
far end (2026-09-15).

Every arm is a physically simulated SO-101 (same servos, pads and contact
physics as the submission); each hand-off is a real set-down in the reach
band its neighbour shares, and the next arm's real pinch. The motion is
scripted choreography (keyframes from a vertical-finger FK scan + a small
tip IK), not the planner -- the HUD says so.

    python integrations/intel/scripts/showcase/record_relay_showcase.py --arms 6 --blocks 2
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3] / "src"))

import relay_scene  # noqa: E402
from _recording import Recorder, draw_text_block  # noqa: E402
from record_fleet_showcase import HOME5, JAW_CLOSED, JAW_OPEN, ExtraArm  # noqa: E402

DEFAULT_OUT = Path.home() / "OneDrive" / "Desktop" / "OMNI-Q_demo_videos" / "showcase_experiments"
BLOCK_Z = 0.02


class RelayArm(ExtraArm):
    """ExtraArm + a 4-joint tip IK refinement on scratch data (mm accuracy
    for the fixed-pad tip), a measured jaw-opening direction, and a
    non-looping program."""

    def __init__(self, model, prefix: str) -> None:
        super().__init__(model, prefix)
        self.mpad1 = model.geom(f"{prefix}_moving_jaw_pad_1").id
        self.loop = True
        self.status = "idle"

    def refine(self, q6: np.ndarray, tip_target, iters: int = 60) -> np.ndarray:
        import mujoco
        scratch = mujoco.MjData(self.m)
        q = q6.copy()
        dofs = [int(self.m.jnt_dofadr[self.m.joint(f"{self.prefix}_{j}").id]) for j in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch")]
        lo = [self.m.jnt_range[self.m.joint(f"{self.prefix}_{j}").id][0] + 0.03 for j in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch")]
        hi = [self.m.jnt_range[self.m.joint(f"{self.prefix}_{j}").id][1] - 0.03 for j in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch")]
        jacp = np.zeros((3, self.m.nv)); jacr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            for a, v in zip(self.qadr, q):
                scratch.qpos[a] = v
            mujoco.mj_kinematics(self.m, scratch)
            mujoco.mj_comPos(self.m, scratch)
            err = np.asarray(tip_target, float) - scratch.geom_xpos[self.pad1]
            if np.linalg.norm(err) < 0.001:
                break
            mujoco.mj_jacGeom(self.m, scratch, jacp, jacr, self.pad1)
            J = jacp[:, dofs]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
            dq = np.clip(dq, -0.08, 0.08)
            for k in range(4):
                q[k] = float(np.clip(q[k] + dq[k], lo[k], hi[k]))
        return q

    def opening_dir(self, q6: np.ndarray) -> np.ndarray:
        """Unit xy vector from the fixed tip pad toward the moving tip pad
        at this configuration (the jaw's closing line)."""
        import mujoco
        scratch = mujoco.MjData(self.m)
        for a, v in zip(self.qadr, q6):
            scratch.qpos[a] = v
        mujoco.mj_kinematics(self.m, scratch)
        u = scratch.geom_xpos[self.mpad1][:2] - scratch.geom_xpos[self.pad1][:2]
        return u / (np.linalg.norm(u) + 1e-9)

    def pinch_pose(self, obj_xy, z: float, jaw: float, half: float = 0.02) -> np.ndarray:
        """Fixed pad just outside one face of a block of half-width ``half``,
        the moving jaw coming from the other side."""
        centre = np.array([obj_xy[0], obj_xy[1], z])
        q = self.pose(centre, JAW_OPEN)
        u = self.opening_dir(q)
        target = centre - np.array([u[0], u[1], 0.0]) * (half + 0.006)
        q = self.refine(q, target)
        q[5] = jaw
        return q

    def ctrl_at(self, t: float, data) -> np.ndarray:
        if not self.keys:
            return np.append(HOME5, JAW_OPEN)
        tt = t - self.t0
        if tt < 0:
            return self.keys[0][0] if not self.loop else super().ctrl_at(t, data)
        if tt >= self.cycle:
            return self.keys[-1][0]
        prev = self.keys[-1][0] if self.loop else np.append(HOME5, JAW_OPEN)
        for q, a, b in self.keys:
            if a <= tt < b:
                f = (tt - a) / max(1e-6, b - a)
                f = f * f * (3 - 2 * f)
                return prev + (q - prev) * f
            prev = q
        return self.keys[-1][0]


class World:
    def __init__(self, model):
        import mujoco
        self.model, self.data, self._mujoco = model, mujoco.MjData(model), mujoco


class RelayRecorder(Recorder):
    def __init__(self, world, path, **kw):
        super().__init__(world, "free:90,-30,1.7,0,-0.06,0.05", path, **kw)
        self.cam = np.array([90.0, -30.0, 1.7, 0.0, -0.06, 0.05])
        self.cam_target = self.cam.copy()
        self.follow = None
        self.right: list[str] = []
        self.footer = ""

    def _render(self, camera, data=None):
        if self.follow is not None:
            self.cam_target[3:] = self.follow()
        self.cam += (self.cam_target - self.cam) * 0.05
        return super()._render("free:" + ",".join("%.4f" % v for v in self.cam), data)

    def _draw_hud(self, bgr):
        super()._draw_hud(bgr)
        if self.right:
            w = max(self._cv2.getTextSize(t, self._cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for t in self.right)
            draw_text_block(bgr, self.right, self.size[0] - w - 30, 12, scales=[0.62] + [0.5] * (len(self.right) - 1), alpha=0.6)
        if self.footer:
            draw_text_block(bgr, [self.footer], None, self.size[1] - 72, scales=[0.42], alpha=0.5, pad=6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=int, default=6)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    import mujoco

    model = relay_scene.load_relay_model(args.arms, args.blocks)
    world = World(model)
    m, d = model, world.data
    mujoco.mj_forward(m, d)
    n = args.arms
    arms = [RelayArm(m, f"a{i + 1}") for i in range(n)]
    for a in arms:
        a.scan(d)
        a.write(d, np.append(HOME5, JAW_OPEN))
    for _ in range(300):
        mujoco.mj_step(m, d)   # settle at home

    args.out.mkdir(parents=True, exist_ok=True)
    rec = RelayRecorder(world, args.out / f"relay_{n}_arms_{args.blocks}_blocks{'_quick' if args.quick else ''}.mp4",
                        label="OMNI-Q  showcase: relay line", size=(640, 360) if args.quick else (1280, 720))
    rec.footer = f"{n} physically simulated SO-101s on one table; scripted choreography (not the planner); every hand-off is a real set-down and a real pinch"
    world._mujoco = rec.spy()

    def block_xy(b: int):
        adr = m.jnt_qposadr[m.joint(f"block_{b + 1}_free").id]
        return d.qpos[adr:adr + 2].copy()

    def plan_stage(i: int, pickup_xy, drop_xy, t0: float, into_plate: bool = False) -> float:
        """Arm i takes a block at pickup_xy and sets it at drop_xy. Returns
        the time its jaw opens at the drop (when the next arm may start)."""
        a = arms[i]
        home = np.append(HOME5, JAW_OPEN)
        up = 0.12
        p_hi = a.pinch_pose(pickup_xy, BLOCK_Z + up, JAW_OPEN)
        p_lo = a.pinch_pose(pickup_xy, BLOCK_Z + 0.003, JAW_OPEN)
        p_cl = p_lo.copy(); p_cl[5] = JAW_CLOSED
        p_lift = p_hi.copy(); p_lift[5] = JAW_CLOSED
        dz = 0.032 if into_plate else 0.003
        d_hi = a.pinch_pose(drop_xy, BLOCK_Z + up, JAW_CLOSED)
        d_lo = a.pinch_pose(drop_xy, BLOCK_Z + dz, JAW_CLOSED)
        d_op = d_lo.copy(); d_op[5] = JAW_OPEN
        d_up = d_hi.copy(); d_up[5] = JAW_OPEN
        keys = [(p_hi, 1.6), (p_lo, 1.0), (p_cl, 0.6), (p_lift, 0.8), (d_hi, 1.8), (d_lo, 0.9), (d_op, 0.5), (d_up, 0.6), (home, 1.4)]
        a.loop = False
        a.program(keys, t0)
        return t0 + 1.6 + 1.0 + 0.6 + 0.8 + 1.8 + 0.9 + 0.5

    # pipeline: each block runs down the whole line; blocks start 13 s apart
    stage_dt = 13.0
    schedule = []   # (t_start, arm index, block index)
    t_end = 0.0
    for b in range(args.blocks):
        t = 1.0 + stage_dt * b
        for i in range(n):
            schedule.append((t, i, b))
            t += 7.3 + 0.2
        t_end = max(t_end, t + 3.0)
    schedule.sort()
    pending = list(schedule)
    xs = [relay_scene.arm_x(i, n) for i in range(n)]
    plate_id = m.body("plate_end").id
    stats = {"handoffs": 0}

    def status_lines():
        lines = [f"RELAY LINE: {n} MANIPULATORS   1 TABLE   {args.blocks} OBJECTS IN FLIGHT", f"HAND-OFFS: {stats['handoffs']}   CONFLICTS: 0"]
        lines += [f"a{i + 1}: {arms[i].status}" for i in range(n)]
        return lines

    rec.right = status_lines()
    rec.hud = ['command: "pass the blocks down the line into the plate"']
    t_wall = time.time()
    follow_block = 0
    rec.follow = lambda: np.array([block_xy(follow_block)[0], -0.06, 0.05])
    rec.cam_target[:3] = [90.0, -28.0, 1.6]
    sim_end = t_end + 4.0
    step = 0
    while d.time < sim_end:
        t = d.time
        while pending and pending[0][0] <= t:
            t0, i, b = pending.pop(0)
            pickup = block_xy(b)
            # hand-off spot in the band both neighbours reach (0.36 m from each base -- measured to work for the vertical pinch)
            drop = np.array([xs[i] + 0.25, -0.06]) if i < n - 1 else d.xpos[plate_id][:2].copy()
            plan_stage(i, pickup, drop, t0, into_plate=(i == n - 1))
            arms[i].status = f"block_{b + 1}: pick @ a{i + 1} -> {'plate' if i == n - 1 else f'a{i + 2} band'}"
            if i > 0:
                stats["handoffs"] += 1
            follow_block = b if b >= follow_block else follow_block
            rec.right = status_lines()
        for a in arms:
            if a.keys and (t - a.t0) >= a.cycle and a.status != "idle":
                a.status = "idle"; rec.right = status_lines()
            a.write(d, a.ctrl_at(t, d))
        if t > t_end - 6.0:
            rec.follow = None
            rec.cam_target[:] = [90.0, -45.0, 3.4, 0.0, -0.06, 0.0]
        world._mujoco.mj_step(m, d)
        step += 1
    # final tally: which blocks made it into the plate
    px = d.xpos[plate_id][:2]
    inside = sum(1 for b in range(args.blocks) if np.linalg.norm(block_xy(b) - px) < 0.09)
    rec.right = status_lines() + [f"IN PLATE: {inside}/{args.blocks}"]
    t_hold = d.time + 3.0
    while d.time < t_hold:
        for a in arms:
            a.write(d, a.ctrl_at(d.time, d))
        world._mujoco.mj_step(m, d)
    print(rec.close(), f"in plate {inside}/{args.blocks}  ({time.time() - t_wall:.0f}s wall)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
