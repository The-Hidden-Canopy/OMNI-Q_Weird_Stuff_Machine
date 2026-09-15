"""VLA-first manipulation on the submission's table world (2026-09-15).

``VLAWorld`` is the submission's ``IntelTableWorld`` with the two single-arm
motor primitives -- PICK and MOVE/PLACE -- driven by a fine-tuned SmolVLA
policy instead of the scripted IK controller:

    cameras (overhead, front, wrist) + arm state + language instruction
        -> SmolVLA (src/omni_q/skills/controllers/smolvla.py, the governed seam)
        -> seven Cartesian deltas per 10 Hz tick
        -> workspace-safety check (the world's own)
        -> the world's pad-pose IK applies the delta (a few iterations)
        -> physics; the world's own grasp / placement verifiers decide

The VLA is the dominant policy: it produces every motion command of a
single-arm step. IK is only the joint-space realisation of the VLA's
Cartesian delta. If the policy has not achieved the step within its time
budget, the world rewinds that arm and the governed scripted primitive
finishes the step -- and the receipt says so (``vla_attempt``), so the VLA
share of every run is measurable, never implied.

The plate (a two-arm rim pinch) stays on the governed bimanual primitive:
the policy is single-arm.
"""
from __future__ import annotations

import collections
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from omni_q import intel_sim  # noqa: E402
from omni_q.intel_sim import (  # noqa: E402
    BIMANUAL_OBJECTS, GRIPPER_CLOSED, GRIPPER_OPEN, LIFT_VERIFY_MIN, ZONE_POSITIONS, IntelTableWorld,
)
from omni_q.contracts import ActionAuthorization, AuthorizationVerdict  # noqa: E402
from omni_q.skills.contracts import SkillRequest  # noqa: E402
from omni_q.skills.controllers.smolvla import SmolVLAController  # noqa: E402

W, H = 320, 240
NAMES = {"plate_1": "plate", "cup_1": "cup", "fork_1": "fork", "spoon_1": "spoon", "napkin_1": "napkin"}
ZONE_TEXT = {"center": "the centre", "upper_right": "the upper right", "left": "the left", "right": "the right", "lower_left": "the lower left"}


def _rot_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)))
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)))
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)))
    return rz @ ry @ rx


class VLAWorld(IntelTableWorld):
    checkpoint: str = os.environ.get("OMNIQ_VLA_CHECKPOINT", "")
    device: str = os.environ.get("OMNIQ_VLA_DEVICE", "cuda")
    pick_budget_s: float = float(os.environ.get("OMNIQ_VLA_PICK_S", "14"))
    place_budget_s: float = float(os.environ.get("OMNIQ_VLA_PLACE_S", "14"))
    hz: int = 10
    fallback: bool = os.environ.get("OMNIQ_VLA_FALLBACK", "1") not in {"", "0", "false", "no"}
    # the absjaw dataset stores the 7th action as the absolute jaw command (a
    # per-tick jaw delta is a sparse spike that regression averages away)
    jaw_absolute: bool = os.environ.get("OMNIQ_VLA_JAW_ABS", "1") not in {"", "0", "false", "no"}

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        import mujoco
        self._vla_renderer = mujoco.Renderer(self.model, height=H, width=W)
        self._vla_render_requests: collections.deque = collections.deque()
        self._vla_render_lock = threading.Lock()
        self._vla_ctrl: dict[int, SmolVLAController] = {}
        self._vla_instruction: dict[int, str] = {0: "", 6: ""}
        self.vla_stats: list[dict[str, Any]] = []
        self.vla_tick_hook = None    # optional callable(arm_offset, tick, action) for recordings

    # -- observation providers --------------------------------------------
    def _cam(self, name: str):
        def capture():
            if threading.current_thread() is threading.main_thread():
                self._vla_renderer.update_scene(self.data, camera=name)
                return self._vla_renderer.render().copy()
            # both arms at once: this arm's step runs on a worker thread and the
            # GL context lives on the main thread -- ask it to render for us
            done = threading.Event()
            holder: dict = {}
            with self._vla_render_lock:
                self._vla_render_requests.append((name, done, holder))
            done.wait()
            return holder["img"]
        return capture

    def _service_renders(self) -> None:
        while True:
            with self._vla_render_lock:
                if not self._vla_render_requests:
                    return
                name, done, holder = self._vla_render_requests.popleft()
            self._vla_renderer.update_scene(self.data, camera=name)
            holder["img"] = self._vla_renderer.render().copy()
            done.set()

    def apply_transitions_parallel(self, requests):
        inner = self._mujoco
        world = self

        class Pump:
            def __getattr__(self, n):
                return getattr(inner, n)

            def main_thread_pump(self):
                world._service_renders()
                p = getattr(inner, "main_thread_pump", None)
                if p is not None:
                    p()
        self._mujoco = Pump()
        try:
            return super().apply_transitions_parallel(requests)
        finally:
            self._mujoco = inner
            self._service_renders()

    def _state(self, arm_offset: int):
        def provider():
            pos = self.data.geom_xpos[self._pad_geom[arm_offset]]
            return np.concatenate([self.data.qpos[arm_offset:arm_offset + 5], [float(self.data.ctrl[arm_offset + 5])], pos]).astype(np.float32)
        return provider

    def _controller(self, arm_offset: int) -> SmolVLAController:
        if arm_offset not in self._vla_ctrl:
            if not self.checkpoint:
                raise RuntimeError("OMNIQ_VLA_CHECKPOINT is not set")
            arm = "left" if arm_offset == 0 else "right"
            self._vla_ctrl[arm_offset] = SmolVLAController(
                self.checkpoint,
                # the base policy's camera slots: camera1 = overhead, camera2 = front, camera3 = wrist (train_smolvla.sh rename_map)
                camera_sources={"observation.images.camera1": self._cam("table_overhead"),
                                "observation.images.camera2": self._cam("third_person"),
                                "observation.images.camera3": self._cam(f"{arm}_wrist")},
                state_provider=self._state(arm_offset),
                instruction_provider=lambda req, a=arm_offset: self._vla_instruction[a],
                device=self.device, robot_type="so101", skill_id=f"vla_{arm}")
            # the fine-tuned checkpoint's normalizer stats are 9-d (this dataset's state);
            # its inherited config.json still declares the base's 6 -- trust the stats
            self._vla_ctrl[arm_offset].state_dim = 9
        return self._vla_ctrl[arm_offset]

    # -- one VLA-driven step -----------------------------------------------
    def _vla_run(self, arm_offset: int, instruction: str, success, budget_s: float) -> dict[str, Any]:
        ctrl = self._controller(arm_offset)
        self._vla_instruction[arm_offset] = instruction
        ctrl.reset()
        # The engine authorized this step before it reached the world (apply_transition
        # runs only after the mission-envelope check); bind the VLA request to that step.
        step = getattr(self, "_current_request", None)
        auth = ActionAuthorization(run_id="engine", step_id=getattr(step, "step_id", "vla"), op=getattr(step, "op", "PICK"),
                                   verdict=AuthorizationVerdict.ALLOW, reason="governed step reached the world adapter (engine authorized)",
                                   state_revision=int(self.revision), envelope_digest="engine:apply_transition")
        request = SkillRequest(request_id=f"vla-{int(time.time() * 1000)}", org_id="omni-q", capability="manipulate",
                               operation=instruction, arm_id="left" if arm_offset == 0 else "right",
                               expected_world_revision=int(self.revision), authorization=auth,
                               skill_id=f"vla_{'left' if arm_offset == 0 else 'right'}")
        iters = max(4, int(round(1.0 / (self.hz * float(self.model.opt.timestep)))) // 3)
        lo, hi = self.model.jnt_range[arm_offset + 5]
        ticks = int(budget_s * self.hz)
        t0 = time.time()
        for tick in range(ticks):
            proposal = ctrl.propose(request, None)
            a = dict(proposal.values)
            pad = self._pad_geom[arm_offset]
            pos = self.data.geom_xpos[pad].copy()
            R = self.data.geom_xmat[pad].reshape(3, 3).copy()
            target = pos + np.array([a["dx_mm"], a["dy_mm"], a["dz_mm"]]) / 1000.0
            jaw = float(np.clip(a["gripper_delta"] if self.jaw_absolute else self.data.ctrl[arm_offset + 5] + a["gripper_delta"], lo, hi))
            self.data.ctrl[arm_offset + 5] = jaw
            # Realise the delta the way the expert's own transit does: the
            # position-only, roll-pinned pad solve (the 6-D pose solve does
            # not move from the folded ready posture -- wrist-pitch limit).
            # The world yaw delta becomes the wrist-roll command.
            rlo, rhi = self.model.jnt_range[arm_offset + 4]
            roll = float(np.clip(self.data.ctrl[arm_offset + 4] + math.radians(a["dyaw_deg"]), rlo + 0.03, rhi - 0.03))
            self._ik_reach_pad(arm_offset, target, iters=iters, roll=roll, track_tcp=False,
                               geom_id=self._pad_geom[arm_offset], tol=0.002, max_dq=0.04)
            if self.vla_tick_hook:
                self.vla_tick_hook(arm_offset, tick, a)
            if tick % 10 == 9:
                safety = self._workspace_safety(arm_offset)
                if not safety["safe"]:
                    return {"ok": False, "ticks": tick + 1, "reason": f"safety: {safety['reason']}", "wall_s": round(time.time() - t0, 1)}
            if success():
                return {"ok": True, "ticks": tick + 1, "reason": None, "wall_s": round(time.time() - t0, 1)}
        return {"ok": False, "ticks": ticks, "reason": "budget exhausted", "wall_s": round(time.time() - t0, 1)}

    def apply_transition(self, request):
        self._current_request = request
        try:
            return super().apply_transition(request)
        finally:
            self._current_request = None

    def _snapshot(self):
        return (self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(), float(self.data.time), self._controller_steps)

    # -- PICK / PLACE -------------------------------------------------------
    def _do_pick(self, arm_offset: int, obj: str) -> dict[str, Any]:
        if obj in BIMANUAL_OBJECTS or not self.checkpoint:
            return super()._do_pick(arm_offset, obj)
        qpos_adr, _ = self._object_joints[obj]
        start_z = float(self.data.qpos[qpos_adr + 2])
        arm = "left" if arm_offset == 0 else "right"
        snap = self._snapshot()
        self._go_home(arm_offset)
        self._set_gripper(arm_offset, GRIPPER_OPEN)

        def held() -> bool:
            lift = float(self.data.qpos[qpos_adr + 2]) - start_z
            return lift >= LIFT_VERIFY_MIN and float(self.data.ctrl[arm_offset + 5]) < 0.6 and self._jaw_object_force(arm_offset, obj) > 0.3
        attempt = self._vla_run(arm_offset, f"pick up the {NAMES.get(obj, obj)} with the {arm} arm", held, self.pick_budget_s)
        lift = float(self.data.qpos[qpos_adr + 2]) - start_z
        record = {"op": "PICK", "object": obj, "arm": arm, **attempt, "lift_height_m": round(lift, 4)}
        self.vla_stats.append(record)
        if attempt["ok"]:
            return {"grasp": "vla_smolvla", "held": True, "lift_height_m": round(lift, 4), "vla_attempt": attempt,
                    "grasp_sensor": {"jaw_commanded_rad": float(self.data.ctrl[arm_offset + 5]),
                                     "max_contact_force_n": round(self._jaw_object_force(arm_offset, obj), 2)}}
        if not self.fallback:
            return {"grasp": "vla_smolvla", "held": False, "lift_height_m": round(lift, 4), "vla_attempt": attempt, "reason": attempt["reason"]}
        self._restore_physics(snap, arm_offset=arm_offset, obj=obj)
        self._set_gripper(arm_offset, GRIPPER_OPEN)
        out = super()._do_pick(arm_offset, obj)
        out["vla_attempt"] = attempt
        out["fallback"] = "governed primitive"
        return out

    def _do_place(self, arm_offset: int, obj: str, to_zone: str | None) -> dict[str, Any]:
        if obj in BIMANUAL_OBJECTS or not self.checkpoint or to_zone not in ZONE_POSITIONS:
            return super()._do_place(arm_offset, obj, to_zone)
        qpos_adr, _ = self._object_joints[obj]
        target = np.asarray(ZONE_POSITIONS[to_zone])
        arm = "left" if arm_offset == 0 else "right"
        snap = self._snapshot()

        def down() -> bool:
            xy = self.data.qpos[qpos_adr:qpos_adr + 2]
            near = float(np.linalg.norm(xy - target[:2])) < 0.05
            low = float(self.data.qpos[qpos_adr + 2]) < target[2] + 0.02
            return near and low and float(self.data.ctrl[arm_offset + 5]) > 0.9
        attempt = self._vla_run(arm_offset, f"move the {NAMES.get(obj, obj)} to {ZONE_TEXT.get(to_zone, to_zone)} with the {arm} arm", down, self.place_budget_s)
        if attempt["ok"]:
            # retreat, home, and let the world's own verdict decide
            pad = self.data.geom_xpos[self._pad_geom[arm_offset]].copy()
            self._ik_reach_pad_pose(arm_offset, pad + np.array([0, 0, 0.12]), self.data.geom_xmat[self._pad_geom[arm_offset]].reshape(3, 3).copy(),
                                    roll_hint=float(self.data.ctrl[arm_offset + 4]), iters=120, orientation_tol=0.5)
            self._go_home(arm_offset)
            settle = self._settle_released_object(obj)
            offset = float(np.linalg.norm(self.data.qpos[qpos_adr:qpos_adr + 2] - target[:2]))
            tilt = self._object_tilt(obj)
            placed = bool(offset < 0.06 and settle["settled"] and self._placed_upright(obj, tilt))
            record = {"op": "MOVE", "object": obj, "arm": arm, **attempt, "placement_error_m": round(offset, 4), "placed": placed}
            self.vla_stats.append(record)
            if placed:
                return {"grasp": "vla_smolvla", "placed": True, "placement_error_m": round(offset, 4), "tilt_rad": round(tilt, 4),
                        "settle": settle, "vla_attempt": attempt, "reason": None}
            attempt = {**attempt, "ok": False, "reason": "released off target" if offset >= 0.06 else "unstable placement"}
        else:
            self.vla_stats.append({"op": "MOVE", "object": obj, "arm": arm, **attempt})
        if not self.fallback:
            return {"grasp": "vla_smolvla", "placed": False, "vla_attempt": attempt, "reason": attempt["reason"],
                    "placement_error_m": float(np.linalg.norm(self.data.qpos[qpos_adr:qpos_adr + 2] - target[:2]))}
        # the object is still in the jaw (or was set down wrong): rewind this arm + object and let the governed primitive place it
        self._restore_physics(snap, arm_offset=arm_offset, obj=obj)
        out = super()._do_place(arm_offset, obj, to_zone)
        out["vla_attempt"] = attempt
        out["fallback"] = "governed primitive"
        return out


def install() -> None:
    """Every engine built in this process gets a VLAWorld (process-local)."""
    intel_sim.IntelTableWorld = VLAWorld
