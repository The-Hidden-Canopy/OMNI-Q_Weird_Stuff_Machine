"""Governed Unitree G1 locomotion provider for OMNI-Q.

This provider is intentionally separate from the arm fleet runtime.  OMNI-Q
selects the high-level locomotion capability and supplies a bounded velocity
intent; the Unitree TorchScript policy supplies leg targets; this adapter owns
only the MuJoCo controller loop.  It does not mutate :class:`WorldState` or
choose a task objective.

The observation construction and PD loop follow Unitree Robotics'
``deploy/deploy_mujoco/deploy_mujoco.py`` at the pinned source revision listed
in ``THIRD_PARTY_NOTICES.md``.  The joint and actuator bindings are resolved
by name and validated against the policy dimensions before stepping physics.
That validation is deliberate: silently applying a 12-DOF policy to a
different G1 MJCF would be a dangerous false positive.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Optional provider dependencies; the core package stays dependency-free.
    import mujoco as _mujoco
except ImportError:  # pragma: no cover - exercised on minimal installations
    _mujoco = None

try:
    import numpy as _np
except ImportError:  # pragma: no cover - exercised on minimal installations
    _np = None

try:
    import torch as _torch
except ImportError:  # pragma: no cover - exercised on minimal installations
    _torch = None

try:
    import yaml as _yaml
except ImportError:  # pragma: no cover - exercised on minimal installations
    _yaml = None


class G1Error(RuntimeError):
    """Raised when the G1 provider cannot establish a safe model boundary."""


# The official 12-DOF G1 deployment model uses this policy order.  The
# provider still resolves every address from the loaded MJCF rather than
# assuming that qpos[7:] and ctrl[:] are interchangeable for every model.
G1_LEG_JOINTS: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

G1_CAPABILITIES: tuple[str, ...] = (
    "LOCOMOTION",
    "HUMANOID_BODY",
    "MARCH",
)


def _require_numpy():
    if _np is None:
        raise G1Error("G1 provider requires numpy; install the g1 optional dependencies")
    return _np


def gravity_orientation(quaternion: Any):
    """Return Unitree's gravity-vector observation for ``w, x, y, z``."""

    np = _require_numpy()
    q = np.asarray(quaternion, dtype=np.float32)
    if q.shape != (4,):
        raise ValueError("quaternion must contain exactly four values in wxyz order")

    qw, qx, qy, qz = q
    return np.asarray(
        (
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ),
        dtype=np.float32,
    )


def pd_control(target_q: Any, q: Any, kp: Any, target_dq: Any, dq: Any, kd: Any):
    """Calculate bounded-policy position-control torques."""

    np = _require_numpy()
    target = np.asarray(target_q, dtype=np.float32)
    current = np.asarray(q, dtype=np.float32)
    proportional = np.asarray(kp, dtype=np.float32)
    target_velocity = np.asarray(target_dq, dtype=np.float32)
    velocity = np.asarray(dq, dtype=np.float32)
    derivative = np.asarray(kd, dtype=np.float32)
    return (target - current) * proportional + (target_velocity - velocity) * derivative


@dataclass(frozen=True)
class HumanoidState:
    """Provider telemetry returned to the OMNI observation boundary."""

    humanoid_id: str
    online: bool
    sim_time: float
    position: tuple[float, float, float]
    command: tuple[float, float, float]
    capability: tuple[str, ...] = G1_CAPABILITIES


def _dependency_error() -> G1Error | None:
    missing = tuple(
        name
        for name, module in (
            ("mujoco", _mujoco),
            ("numpy", _np),
            ("torch", _torch),
            ("pyyaml", _yaml),
        )
        if module is None
    )
    if missing:
        return G1Error(
            "G1 provider requires optional dependencies: "
            + ", ".join(missing)
            + "; install the 'g1' extra"
        )
    return None


def _resolve_root(unitree_root: str | Path | None) -> Path:
    value = unitree_root or os.environ.get("UNITREE_RL_GYM")
    if not value:
        raise G1Error("Set UNITREE_RL_GYM to a unitree_rl_gym checkout")

    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise G1Error(f"Unitree G1 root does not exist: {root}")
    return root


def _resolve_config_path(root: Path, config_path: str | Path | None) -> Path:
    if config_path is None:
        return root / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml"

    candidate = Path(config_path).expanduser()
    if candidate.is_absolute():
        return candidate
    direct = root / candidate
    if direct.is_file():
        return direct
    return root / "deploy" / "deploy_mujoco" / "configs" / candidate


def _resolve_asset(root: Path, value: Any, *, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise G1Error(f"G1 config field {key!r} must be a non-empty path")

    resolved = value.replace("{LEGGED_GYM_ROOT_DIR}", str(root))
    path = Path(resolved).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _as_vector(config: Mapping[str, Any], key: str, length: int):
    np = _require_numpy()
    value = config.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise G1Error(f"G1 config field {key!r} must contain {length} values")
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise G1Error(f"G1 config field {key!r} contains non-finite values")
    return array


def _name_id(model: Any, object_type: Any, name: str, kind: str) -> int:
    object_id = int(_mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise G1Error(f"G1 MJCF is missing {kind} {name!r}")
    return object_id


class G1Provider:
    """OMNI-facing Unitree G1 locomotion provider.

    OMNI supplies bounded ``vx, vy, yaw_rate`` intent.  The pretrained
    Unitree controller owns leg-level locomotion.  No method here accepts a
    joint target or torque from OMNI.
    """

    def __init__(
        self,
        *,
        humanoid_id: str = "humanoid_01",
        unitree_root: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        if not isinstance(humanoid_id, str) or not humanoid_id.strip():
            raise ValueError("humanoid_id must be a non-empty string")

        dependency_error = _dependency_error()
        if dependency_error is not None:
            raise dependency_error

        np = _np
        torch = _torch
        mujoco = _mujoco
        yaml = _yaml
        assert np is not None and torch is not None and mujoco is not None and yaml is not None

        self.humanoid_id = humanoid_id
        root = _resolve_root(unitree_root)
        config_file = _resolve_config_path(root, config_path)
        if not config_file.is_file():
            raise G1Error(f"missing G1 MuJoCo config: {config_file}")

        with config_file.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, Mapping):
            raise G1Error(f"G1 config is not a mapping: {config_file}")

        self.policy_path = _resolve_asset(root, config.get("policy_path"), key="policy_path")
        self.xml_path = _resolve_asset(root, config.get("xml_path"), key="xml_path")
        if not self.policy_path.is_file():
            raise G1Error(f"missing G1 policy: {self.policy_path}")
        if not self.xml_path.is_file():
            raise G1Error(f"missing G1 MJCF: {self.xml_path}")

        self.dt = float(config.get("simulation_dt", 0.0))
        self.decimation = int(config.get("control_decimation", 0))
        if self.dt <= 0 or self.decimation <= 0:
            raise G1Error("G1 simulation_dt and control_decimation must be positive")

        self.num_actions = int(config.get("num_actions", 0))
        self.num_obs = int(config.get("num_obs", 0))
        if self.num_actions != len(G1_LEG_JOINTS):
            raise G1Error(
                f"unsupported G1 policy action count {self.num_actions}; "
                f"expected {len(G1_LEG_JOINTS)}"
            )
        expected_obs = 3 + 3 + 3 + (3 * self.num_actions) + 2
        if self.num_obs != expected_obs:
            raise G1Error(
                f"unsupported G1 policy observation count {self.num_obs}; "
                f"expected {expected_obs}"
            )

        self.kps = _as_vector(config, "kps", self.num_actions)
        self.kds = _as_vector(config, "kds", self.num_actions)
        self.default_angles = _as_vector(config, "default_angles", self.num_actions)
        self.cmd_scale = _as_vector(config, "cmd_scale", 3)

        self.ang_vel_scale = float(config.get("ang_vel_scale", 0.0))
        self.dof_pos_scale = float(config.get("dof_pos_scale", 0.0))
        self.dof_vel_scale = float(config.get("dof_vel_scale", 0.0))
        self.action_scale = float(config.get("action_scale", 0.0))
        if not all(
            np.isfinite(value)
            for value in (
                self.ang_vel_scale,
                self.dof_pos_scale,
                self.dof_vel_scale,
                self.action_scale,
            )
        ):
            raise G1Error("G1 policy scales must be finite")

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.dt
        if self.model.nq < 7 or self.model.nv < 6:
            raise G1Error("G1 MJCF does not expose a free floating base")
        if self.model.nu < self.num_actions:
            raise G1Error(
                f"G1 MJCF exposes {self.model.nu} actuators for "
                f"{self.num_actions} policy actions"
            )

        self.qpos_indices: tuple[int, ...] = tuple(
            self._joint_qpos_index(name) for name in G1_LEG_JOINTS
        )
        self.qvel_indices: tuple[int, ...] = tuple(
            self._joint_qvel_index(name) for name in G1_LEG_JOINTS
        )
        self.actuator_indices: tuple[int, ...] = tuple(
            self._actuator_index(name) for name in G1_LEG_JOINTS
        )
        if len(set(self.qpos_indices)) != self.num_actions:
            raise G1Error("G1 policy joint bindings contain duplicate qpos addresses")
        if len(set(self.qvel_indices)) != self.num_actions:
            raise G1Error("G1 policy joint bindings contain duplicate qvel addresses")
        if len(set(self.actuator_indices)) != self.num_actions:
            raise G1Error("G1 policy actuator bindings contain duplicates")

        try:
            self.policy = torch.jit.load(str(self.policy_path), map_location="cpu")
            self.policy.eval()
        except Exception as exc:  # pragma: no cover - depends on external artifact
            raise G1Error(f"could not load G1 TorchScript policy {self.policy_path}: {exc}") from exc

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_q = self.default_angles.copy()
        self.command = np.zeros(3, dtype=np.float32)
        self.counter = 0
        self._online = True

        mujoco.mj_forward(self.model, self.data)

    def _joint_qpos_index(self, name: str) -> int:
        mujoco = _mujoco
        assert mujoco is not None
        joint_id = _name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name, "joint")
        if int(self.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise G1Error(f"G1 policy joint {name!r} is not a scalar hinge joint")
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_qvel_index(self, name: str) -> int:
        mujoco = _mujoco
        assert mujoco is not None
        joint_id = _name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name, "joint")
        return int(self.model.jnt_dofadr[joint_id])

    def _actuator_index(self, name: str) -> int:
        mujoco = _mujoco
        assert mujoco is not None
        actuator_id = _name_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name, "actuator")
        joint_id = _name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name, "joint")
        if int(self.model.actuator_trnid[actuator_id, 0]) != joint_id:
            raise G1Error(f"G1 actuator {name!r} is not bound to its policy joint")
        return actuator_id

    @property
    def online(self) -> bool:
        return self._online

    @property
    def capabilities(self) -> tuple[str, ...]:
        return G1_CAPABILITIES

    # ------------------------------------------------------------------
    # OMNI capability boundary
    # ------------------------------------------------------------------

    def set_velocity(
        self,
        forward_mps: float,
        lateral_mps: float = 0.0,
        yaw_rate: float = 0.0,
    ) -> None:
        """Set bounded high-level locomotion intent in m/s and rad/s."""

        if not self.online:
            raise G1Error(f"{self.humanoid_id} is offline")
        values = (float(forward_mps), float(lateral_mps), float(yaw_rate))
        if not all(_np.isfinite(value) for value in values):
            raise ValueError("G1 velocity intent must be finite")
        self.command[:] = (
            _np.clip(values[0], -0.8, 0.8),
            _np.clip(values[1], -0.5, 0.5),
            _np.clip(values[2], -1.57, 1.57),
        )

    def stop(self) -> None:
        """Remove locomotion intent without taking the provider offline."""

        self.command[:] = 0.0

    def set_online(self, online: bool) -> None:
        """Change provider availability; going offline first removes intent."""

        if not isinstance(online, bool):
            raise TypeError("online must be a bool")
        if not online:
            self.stop()
            self.data.ctrl[list(self.actuator_indices)] = 0.0
        self._online = online

    def march(self, speed: float = 0.5) -> None:
        self.set_velocity(speed, 0.0, 0.0)

    def turn(self, yaw_rate: float) -> None:
        self.set_velocity(0.0, 0.0, yaw_rate)

    # ------------------------------------------------------------------
    # Unitree controller loop
    # ------------------------------------------------------------------

    def _policy_update(self) -> None:
        np = _np
        torch = _torch
        assert np is not None and torch is not None

        qj = self.data.qpos[list(self.qpos_indices)].copy()
        dqj = self.data.qvel[list(self.qvel_indices)].copy()
        quat = self.data.qpos[3:7].copy()
        omega = self.data.qvel[3:6].copy()

        qj_norm = (qj - self.default_angles) * self.dof_pos_scale
        dqj_norm = dqj * self.dof_vel_scale
        omega_norm = omega * self.ang_vel_scale

        period = 0.8
        phase = ((self.counter * self.dt) % period) / period
        sin_phase = np.sin(2.0 * np.pi * phase)
        cos_phase = np.cos(2.0 * np.pi * phase)

        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[0:3] = omega_norm
        obs[3:6] = gravity_orientation(quat)
        obs[6:9] = self.command * self.cmd_scale
        offset = 9
        obs[offset : offset + self.num_actions] = qj_norm
        offset += self.num_actions
        obs[offset : offset + self.num_actions] = dqj_norm
        offset += self.num_actions
        obs[offset : offset + self.num_actions] = self.action
        offset += self.num_actions
        obs[offset : offset + 2] = (sin_phase, cos_phase)

        with torch.inference_mode():
            output = self.policy(torch.from_numpy(obs).unsqueeze(0))
        if not hasattr(output, "detach"):
            raise G1Error("G1 TorchScript policy returned a non-tensor output")
        action = output.detach().cpu().numpy().squeeze()
        if action.shape != (self.num_actions,):
            raise G1Error(
                f"G1 policy returned shape {action.shape}; "
                f"expected {(self.num_actions,)}"
            )
        if not np.all(np.isfinite(action)):
            raise G1Error("G1 policy returned non-finite action values")

        self.action = np.asarray(action, dtype=np.float32)
        self.target_q = self.action * self.action_scale + self.default_angles

    def step(self) -> None:
        """Advance one MuJoCo physics step using the active policy target."""

        if not self.online:
            self.data.ctrl[list(self.actuator_indices)] = 0.0
            return

        np = _np
        mujoco = _mujoco
        assert np is not None and mujoco is not None
        current_q = self.data.qpos[list(self.qpos_indices)]
        current_dq = self.data.qvel[list(self.qvel_indices)]
        tau = pd_control(
            self.target_q,
            current_q,
            self.kps,
            np.zeros_like(self.kds),
            current_dq,
            self.kds,
        )
        self.data.ctrl[list(self.actuator_indices)] = tau
        mujoco.mj_step(self.model, self.data)

        self.counter += 1
        if self.counter % self.decimation == 0:
            self._policy_update()

    def state(self) -> HumanoidState:
        return HumanoidState(
            humanoid_id=self.humanoid_id,
            online=self.online,
            sim_time=float(self.data.time),
            position=tuple(float(value) for value in self.data.qpos[:3]),
            command=tuple(float(value) for value in self.command),
        )
