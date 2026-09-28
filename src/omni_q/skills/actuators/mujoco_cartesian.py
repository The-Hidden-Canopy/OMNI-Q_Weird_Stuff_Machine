"""MuJoCo Cartesian-delta actuator behind the skill supervisor."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..contracts import SkillObservation

__all__ = ["MuJoCoCartesianIKActuator"]


_ACTION_FIELDS = frozenset({
    "dx_mm",
    "dy_mm",
    "dz_mm",
    "droll_deg",
    "dpitch_deg",
    "dyaw_deg",
    "gripper_delta",
})


def _rotation_xyz(roll: float, pitch: float, yaw: float, np: Any) -> Any:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)))
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)))
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)))
    return rz @ ry @ rx


def _rotation_error(target: Any, current: Any, np: Any) -> Any:
    r = target @ current.T
    return 0.5 * np.array((
        r[2, 1] - r[1, 2],
        r[0, 2] - r[2, 0],
        r[1, 0] - r[0, 1],
    ))


class MuJoCoCartesianIKActuator:
    """Convert an approved Cartesian proposal into position-actuator targets.

    The controller never receives this object and the actuator never performs
    authorization.  It assumes the caller already passed the proposal through
    :class:`omni_q.skills.supervisor.SkillSupervisor`.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        arm_joint_names: tuple[str, ...],
        arm_actuator_names: tuple[str, ...],
        ee_site_name: str,
        gripper_actuator_name: str | None = None,
        ik_iterations: int = 40,
        physics_substeps: int = 5,
        damping: float = 0.015,
        max_joint_step_rad: float = 0.08,
        position_tolerance_m: float = 0.0025,
        rotation_tolerance_rad: float = 0.03,
    ) -> None:
        try:
            import mujoco
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo Cartesian IK requires the optional mujoco dependency"
            ) from exc

        if len(arm_joint_names) == 0:
            raise ValueError("arm_joint_names must not be empty")
        if len(arm_joint_names) != len(arm_actuator_names):
            raise ValueError("joint/actuator lengths must match")
        if len(set(arm_joint_names)) != len(arm_joint_names):
            raise ValueError("arm_joint_names must not contain duplicates")
        if len(set(arm_actuator_names)) != len(arm_actuator_names):
            raise ValueError("arm_actuator_names must not contain duplicates")
        if not isinstance(ee_site_name, str) or not ee_site_name.strip():
            raise ValueError("ee_site_name must be a non-empty string")
        for name, value in (
            ("ik_iterations", ik_iterations),
            ("physics_substeps", physics_substeps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("damping", damping),
            ("max_joint_step_rad", max_joint_step_rad),
            ("position_tolerance_m", position_tolerance_m),
            ("rotation_tolerance_rad", rotation_tolerance_rad),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a positive finite number")

        self.model = model
        self.data = data
        self._mujoco = mujoco
        self._np = np

        self.joint_ids = tuple(self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name, "joint") for name in arm_joint_names)
        self.qpos_indices = tuple(int(model.jnt_qposadr[j]) for j in self.joint_ids)
        self.dof_indices = tuple(int(model.jnt_dofadr[j]) for j in self.joint_ids)
        if any(int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE) for j in self.joint_ids):
            raise ValueError("Cartesian IK currently requires scalar hinge arm joints")

        self.actuator_ids = tuple(self._named_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name, "actuator") for name in arm_actuator_names)
        for actuator_id, joint_id in zip(self.actuator_ids, self.joint_ids):
            if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                raise ValueError("arm actuators must transmit directly to joints")
            if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
                raise ValueError("each arm actuator must bind to its named joint")

        self.ee_site_id = self._named_id(mujoco.mjtObj.mjOBJ_SITE, ee_site_name, "site")
        self.gripper_actuator_id = None if gripper_actuator_name is None else self._named_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            gripper_actuator_name,
            "gripper actuator",
        )

        self.ik_iterations = ik_iterations
        self.physics_substeps = physics_substeps
        self.damping = float(damping)
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.position_tolerance_m = float(position_tolerance_m)
        self.rotation_tolerance_rad = float(rotation_tolerance_rad)

    def _named_id(self, object_type: Any, name: str, kind: str) -> int:
        identifier = int(self._mujoco.mj_name2id(self.model, object_type, name))
        if identifier < 0:
            raise ValueError(f"MuJoCo {kind} {name!r} was not found")
        return identifier

    def policy_state(self) -> Any:
        return self._np.asarray(self.data.qpos[list(self.qpos_indices)], dtype=self._np.float32).copy()

    def tcp_position_mm(self) -> tuple[float, float, float]:
        self._mujoco.mj_forward(self.model, self.data)
        position = self.data.site_xpos[self.ee_site_id]
        return tuple(float(value * 1000.0) for value in position)

    def observation(
        self,
        *,
        org_id: str,
        arm_id: str,
        world_revision: int,
        safety_stop: bool = False,
        contact_force_n: float = 0.0,
        action_interval_ms: int = 50,
    ) -> SkillObservation:
        return SkillObservation(
            org_id=org_id,
            arm_id=arm_id,
            world_revision=world_revision,
            tcp_position_mm=self.tcp_position_mm(),
            contact_force_n=contact_force_n,
            action_interval_ms=action_interval_ms,
            safety_stop=safety_stop,
        )

    def _clamp_joint(self, joint_id: int, value: float) -> float:
        if not bool(self.model.jnt_limited[joint_id]):
            return value
        lower, upper = self.model.jnt_range[joint_id]
        return float(self._np.clip(value, lower, upper))

    def apply(
        self,
        action: Mapping[str, float],
        observation: SkillObservation,
    ) -> Mapping[str, Any]:
        if not isinstance(action, Mapping):
            raise TypeError("actuator action must be a mapping")
        unknown = sorted(set(action) - _ACTION_FIELDS)
        if unknown:
            raise ValueError(f"unsupported actuator action fields: {', '.join(unknown)}")
        values = {name: float(action.get(name, 0.0)) for name in _ACTION_FIELDS}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("actuator action must contain only finite values")
        del observation  # The supervisor already validated the observation.

        dx, dy, dz = (values[name] for name in ("dx_mm", "dy_mm", "dz_mm"))
        delta_rotation = _rotation_xyz(
            math.radians(values["droll_deg"]),
            math.radians(values["dpitch_deg"]),
            math.radians(values["dyaw_deg"]),
            self._np,
        )

        self._mujoco.mj_forward(self.model, self.data)
        start_position = self.data.site_xpos[self.ee_site_id].copy()
        start_rotation = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        target_position = start_position + self._np.array((dx, dy, dz), dtype=float) / 1000.0
        target_rotation = delta_rotation @ start_rotation

        jacp = self._np.zeros((3, self.model.nv))
        jacr = self._np.zeros((3, self.model.nv))
        position_error_norm = math.inf
        rotation_error_norm = math.inf
        iterations_used = 0

        for iteration in range(self.ik_iterations):
            iterations_used = iteration + 1
            self._mujoco.mj_forward(self.model, self.data)
            current_position = self.data.site_xpos[self.ee_site_id].copy()
            current_rotation = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
            position_error = target_position - current_position
            rotation_error = _rotation_error(target_rotation, current_rotation, self._np)
            position_error_norm = float(self._np.linalg.norm(position_error))
            rotation_error_norm = float(self._np.linalg.norm(rotation_error))
            if position_error_norm <= self.position_tolerance_m and rotation_error_norm <= self.rotation_tolerance_rad:
                break

            jacp.fill(0.0)
            jacr.fill(0.0)
            self._mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
            joint_dofs = list(self.dof_indices)
            jacobian = self._np.vstack((jacp[:, joint_dofs], 0.5 * jacr[:, joint_dofs]))
            error = self._np.concatenate((position_error, 0.5 * rotation_error))
            lhs = jacobian @ jacobian.T + self._np.eye(6) * self.damping**2
            delta_q = jacobian.T @ self._np.linalg.solve(lhs, error)
            delta_q = self._np.clip(delta_q, -self.max_joint_step_rad, self.max_joint_step_rad)
            current_q = self.data.qpos[list(self.qpos_indices)].copy()
            target_q = current_q + delta_q
            for index, joint_id in enumerate(self.joint_ids):
                target_q[index] = self._clamp_joint(joint_id, float(target_q[index]))

            self.data.ctrl[list(self.actuator_ids)] = target_q
            for _ in range(self.physics_substeps):
                self._mujoco.mj_step(self.model, self.data)

        if self.gripper_actuator_id is not None and values["gripper_delta"] != 0.0:
            actuator_id = self.gripper_actuator_id
            target = float(self.data.ctrl[actuator_id]) + values["gripper_delta"]
            if bool(self.model.actuator_ctrllimited[actuator_id]):
                lower, upper = self.model.actuator_ctrlrange[actuator_id]
                target = float(self._np.clip(target, lower, upper))
            self.data.ctrl[actuator_id] = target
            for _ in range(self.physics_substeps):
                self._mujoco.mj_step(self.model, self.data)

        self._mujoco.mj_forward(self.model, self.data)
        final_position = self.data.site_xpos[self.ee_site_id].copy()
        return {
            "controller": "smolvla+cartesian-ik",
            "iterations": iterations_used,
            "position_error_m": position_error_norm,
            "rotation_error_rad": rotation_error_norm,
            "start_position_m": start_position.tolist(),
            "target_position_m": target_position.tolist(),
            "final_position_m": final_position.tolist(),
            "sim_time": float(self.data.time),
        }
