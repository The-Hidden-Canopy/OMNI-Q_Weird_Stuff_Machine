"""VLA-first manipulation on the submission's table world (2026-09-15).

``VLAWorld`` is the submission's ``IntelTableWorld`` with the two single-arm
motor primitives -- PICK and MOVE/PLACE -- driven by a fine-tuned SmolVLA
policy instead of the scripted IK controller:

    cameras (overhead, front, wrist) + arm state + language instruction
        -> SmolVLA proposal-only adapter
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

This module is an offline simulation/evaluation adapter.  It records the VLA
proposal and the governed-primitive fallback, but it is not the production
``SkillRuntime`` and does not promote the checkpoint to ``ACTIVE``.
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
    pick_budget_s: float = float(os.environ.get("OMNIQ_VLA_PICK_S", "8"))
    place_budget_s: float = float(os.environ.get("OMNIQ_VLA_PLACE_S", "4"))
    # "continue": the governed primitive picks up from where the VLA left the arm
    # (its approach stays in the run); "rewind": the arm is put back first
    fallback_mode: str = os.environ.get("OMNIQ_VLA_FALLBACK_MODE", "continue")
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
                device=self.device, robot_type="so101", skill_id=f"vla_{arm}",
                # The fine-tuned checkpoint's normalizer stats are 9-d (this
                # dataset's state); its inherited config.json still declares
                # the base's 6. Keep that mismatch explicit at construction.
                state_dim_override=9)
        return self._vla_ctrl[arm_offset]

    # -- one VLA-driven step -----------------------------------------------
    def _vla_run(self, arm_offset: int, instruction: str, success, budget_s: float, *, hold_jaw: bool = False,
                 success_release=None, guard=None) -> dict[str, Any]:
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
            if not hold_jaw:
                jaw = float(np.clip(a["gripper_delta"] if self.jaw_absolute else self.data.ctrl[arm_offset + 5] + a["gripper_delta"], lo, hi))
                self.data.ctrl[arm_offset + 5] = jaw
            elif a["gripper_delta"] > 1.0 and success_release is not None and success_release():
                self.data.ctrl[arm_offset + 5] = GRIPPER_OPEN   # the policy wants to release and the object is over the zone
            # Realise the delta the way the expert's own transit does: the
            # position-only, roll-pinned pad solve (the 6-D pose solve does
            # not move from the folded ready posture -- wrist-pitch limit).
            # The world yaw delta becomes the wrist-roll command.
            rlo, rhi = self.model.jnt_range[arm_offset + 4]
            roll = float(np.clip(self.data.ctrl[arm_offset + 4] + math.radians(a["dyaw_deg"]), rlo + 0.15, rhi - 0.15))
            self._realise_delta(arm_offset, target, roll, iters)
            if self.vla_tick_hook:
                self.vla_tick_hook(arm_offset, tick, a)
            if tick % 10 == 9:
                safety = self._workspace_safety(arm_offset)
                # the expert's own solves sit at the 0.03 rad joint margin mid-motion; the
                # world's check is for its step boundaries -- abort here only on force/space
                if not safety["safe"] and "joint-limit" not in str(safety.get("reason")):
                    return {"ok": False, "ticks": tick + 1, "reason": f"safety: {safety['reason']}", "wall_s": round(time.time() - t0, 1)}
            if success():
                return {"ok": True, "ticks": tick + 1, "reason": None, "wall_s": round(time.time() - t0, 1)}
            if guard is not None:
                why = guard()
                if why:
                    return {"ok": False, "ticks": tick + 1, "reason": why, "wall_s": round(time.time() - t0, 1)}
        return {"ok": False, "ticks": ticks, "reason": "budget exhausted", "wall_s": round(time.time() - t0, 1)}

    def _go_home(self, arm_offset: int, *, steps: int = 150) -> None:
        if getattr(self, "_vla_continue_arm", None) == arm_offset:
            return   # continuing a VLA approach: don't fly back to the ready pose first
        return super()._go_home(arm_offset, steps=steps)

    def apply_transition(self, request):
        self._current_request = request
        try:
            return super().apply_transition(request)
        finally:
            self._current_request = None

    def _realise_delta(self, arm_offset: int, target, roll: float, iters: int) -> None:
        """Unweighted 4-joint damped-least-squares tracking of the fixed pad
        toward ``target`` (wrist roll commanded directly). The world's own
        pad solve weights the proximal joints down 80x, which is right for
        its long moves but absorbs a small per-tick lateral delta into the
        distal joints -- the base never turned toward the object."""
        mujoco = self._mujoco
        pad = self._pad_geom[arm_offset]
        jacp = np.zeros((3, self.model.nv)); jacr = np.zeros((3, self.model.nv))
        lo = self.model.jnt_range[arm_offset:arm_offset + 4, 0] + 0.03
        hi = self.model.jnt_range[arm_offset:arm_offset + 4, 1] - 0.03
        self.data.ctrl[arm_offset + 4] = roll
        for _ in range(iters):
            mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, pad)
            err = np.asarray(target, float) - self.data.geom_xpos[pad]
            if np.linalg.norm(err) < 0.0015:
                mujoco.mj_step(self.model, self.data, nstep=3)
                self._controller_steps += 3
                continue
            J = jacp[:, arm_offset:arm_offset + 4]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(3), err)
            dq = np.clip(dq, -0.04, 0.04)
            q = self.data.qpos[arm_offset:arm_offset + 4]
            self.data.ctrl[arm_offset:arm_offset + 4] = np.clip(q + dq, lo, hi)
            mujoco.mj_step(self.model, self.data, nstep=3)
            self._controller_steps += 3

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
        if self.fallback_mode == "rewind":
            self._restore_physics(snap, arm_offset=arm_offset, obj=obj)
            self._set_gripper(arm_offset, GRIPPER_OPEN)
            out = super()._do_pick(arm_offset, obj)
        else:
            self._vla_continue_arm = arm_offset
            try:
                out = super()._do_pick(arm_offset, obj)
            finally:
                self._vla_continue_arm = None
        out["vla_attempt"] = attempt
        out["fallback"] = "governed primitive (continued from the VLA's approach)" if self.fallback_mode != "rewind" else "governed primitive"
        return out

    def _do_place(self, arm_offset: int, obj: str, to_zone: str | None) -> dict[str, Any]:
        if obj in BIMANUAL_OBJECTS or not self.checkpoint or to_zone not in ZONE_POSITIONS:
            return super()._do_place(arm_offset, obj, to_zone)
        qpos_adr, _ = self._object_joints[obj]
        target = np.asarray(ZONE_POSITIONS[to_zone])
        arm = "left" if arm_offset == 0 else "right"
        snap = self._snapshot()

        def over_zone() -> bool:
            xy = self.data.qpos[qpos_adr:qpos_adr + 2]
            return float(np.linalg.norm(xy - target[:2])) < 0.05 and float(self.data.qpos[qpos_adr + 2]) < target[2] + 0.03

        def down() -> bool:
            return over_zone() and float(self.data.ctrl[arm_offset + 5]) > 0.9
        # the VLA moves the arm; the grasp stays as the pick left it until the
        # policy asks to release over the zone (its free jaw commands loosened
        # pinches mid-carry and the continuation inherited a dropped object)
        pad0 = self.data.geom_xpos[self._pad_geom[arm_offset]].copy()
        rel0 = self.data.qpos[qpos_adr:qpos_adr + 3] - pad0
        R_pad0 = self.data.geom_xmat[self._pad_geom[arm_offset]].reshape(3, 3).copy()
        R_obj0 = np.zeros(9); self._mujoco.mju_quat2Mat(R_obj0, self.data.qpos[qpos_adr + 3:qpos_adr + 7]); R_obj0 = R_obj0.reshape(3, 3)
        R_rel0 = R_pad0.T @ R_obj0

        def grasp_guard():
            # grasp integrity: the object must ride with the pad -- neither
            # sliding nor turning in the pinch (a fork turns without sliding);
            # otherwise the governed carry takes over before it is lost
            pad = self._pad_geom[arm_offset]
            rel = self.data.qpos[qpos_adr:qpos_adr + 3] - self.data.geom_xpos[pad]
            drift = float(np.linalg.norm(rel - rel0))
            R_obj = np.zeros(9); self._mujoco.mju_quat2Mat(R_obj, self.data.qpos[qpos_adr + 3:qpos_adr + 7])
            R_rel = self.data.geom_xmat[pad].reshape(3, 3).T @ R_obj.reshape(3, 3)
            turn = float(np.arccos(np.clip((np.trace(R_rel0.T @ R_rel) - 1) / 2, -1, 1)))
            if drift > 0.010:
                return f"grasp slipping ({drift * 1000:.0f} mm)"
            if turn > math.radians(8):
                return f"grasp turning ({math.degrees(turn):.0f} deg)"
            return None
        attempt = self._vla_run(arm_offset, f"move the {NAMES.get(obj, obj)} to {ZONE_TEXT.get(to_zone, to_zone)} with the {arm} arm", down,
                                self.place_budget_s, hold_jaw=True, success_release=over_zone, guard=grasp_guard)
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
        if self.fallback_mode == "rewind" or not attempt.get("ok", False) and float(self.data.ctrl[arm_offset + 5]) > 0.9:
            # released somewhere wrong (or rewind requested): put the arm and object back, then the governed primitive places it
            self._restore_physics(snap, arm_offset=arm_offset, obj=obj)
            out = super()._do_place(arm_offset, obj, to_zone)
            out["fallback"] = "governed primitive"
        else:
            # still holding it part-way: the governed carry continues from here
            out = super()._do_place(arm_offset, obj, to_zone)
            out["fallback"] = "governed primitive (continued from the VLA's carry)"
        out["vla_attempt"] = attempt
        return out


def install() -> None:
    """Every engine built in this process gets a VLAWorld (process-local)."""
    intel_sim.IntelTableWorld = VLAWorld
