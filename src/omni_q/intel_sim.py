"""Intel Online dual-SO-101 MuJoCo adapters.

This module uses the pinned SO-ARM100 Menagerie MJCF as the documented
six-joint mechanical proxy for SO-101.  It creates one real MuJoCo scene with
two independently actuated arms, a table-setting object pack (plate, cup,
fork, spoon, napkin -- distinct masses/friction/collision per OQ-007), and a
passive slide-jointed drawer holding the cutlery, matching the brief's
"open the top drawer, retrieve spoons and forks" scenario.

PICK/MOVE/PLACE on tracked tableware (OQ-010) use a real weighted
damped-least-squares differential IK controller (``IntelTableWorld._ik_reach``)
driving the arm's position servos toward the object/target -- real joint
motion via mj_step, no teleport, no velocity override -- and a genuine
contact-driven grasp attempt (close the gripper and see what actually
happens), with success/failure grounded in the measured outcome (lift height
on PICK, final position error on PLACE) rather than an assumed label: a
failed grasp/placement reverts the WorldState change and reports the step as
failed, so the engine's normal replan loop actually retries instead of the
receipt silently claiming success.

**Current fidelity, measured, not assumed:** IK position convergence is
reliable (~1cm) once aimed at a target that clears the object's own volume.
The pinch itself is not: this adapter only solves 3-DOF position, so wrist
orientation is whatever the redundant IK null-space happens to settle into --
not controlled to face the jaws at the object. Position-only IK also turned
out to be dangerous near the table: with 5 joints solving a 3-task position,
the unweighted minimum-norm solution was measured swinging the shoulder/
forearm through tableware even for a small vertical lift where that swing
wasn't geometrically necessary, which is why ``_IK_JOINT_WEIGHTS`` and the
safe-transit-height waypointing in ``_move_to``/``_ik_track_line`` exist.
Net result: grasp attempts are real physics with an honest pass/fail signal,
but the pass rate is currently low -- reliable grasping needs orientation-
aware (6-DOF) IK, not more tuning of this 3-DOF controller. See
integrations/intel/README.md for what that would take.

This is still a proxy, not hardware evidence -- no vision-guided grasp point,
no force control. The separate OQ-010/OQ-011 contact adapter uses only MuJoCo
contact dynamics for a bounded ``cup_1`` handoff. It is a SO-ARM100
mechanical-proxy simulation, not evidence of perception, VLA control,
hardware, complete table setting, or concurrent execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .contracts import (
    Detection,
    DeviceSpec,
    ManipResult,
    PlanDecision,
    PlanGraph,
    Step,
    TransitionRejected,
    TransitionRequest,
    TransitionResult,
    VerifyResult,
)
from .devices import DeviceRouter
from .engine import OmniQ
from .fakes import FakeManipulator, FakeObserver, FakeRecorder, FakeVerifier, RulePlanner
from .world import MockWorld


ROOT = Path(__file__).resolve().parents[2]
ARM_XML = ROOT / "integrations" / "intel" / "assets" / "menagerie_so_arm100" / "so_arm100.xml"
ARM_ASSETS = ARM_XML.parent / "assets"
HOME = (0.0, -1.57, 1.57, 1.57, -1.57, 0.0)

# OQ-010 primitive targets. GRIPPER_OPEN/CLOSED drive the real IK grasp
# (IntelTableWorld._do_pick/_do_place); the rest are coarse fixed poses for
# ops without dedicated IK behaviour (OPEN/CLOSE/ROTATE/PRESENT/HANDOFF, and
# any PICK/MOVE/PLACE on an object this adapter doesn't track).
GRIPPER_OPEN = 1.5     # Jaw joint, rad -- near the SO-101 open end of its range
GRIPPER_CLOSED = 0.0
DRAWER_OPEN = 0.12     # drawer_slide qpos, m -- matches its MJCF range max
DRAWER_CLOSED = 0.0
TRANSIT_HEIGHT = 0.28  # m -- above the table/drawer/tableware envelope, within reach (see _move_to)

# Half-height (m) of each tableware geom in dual_so101_xml() -- used to place
# the IK grasp/place target just above the object's actual TOP surface, not
# its centre. Targeting centre + a small fixed offset put the gripper target
# *inside* tall objects (the cup's half-height alone is 0.055m, more than
# the old fixed 0.012m clearance), which doesn't converge -- the position
# servo just pushes into the object's own volume, dragging it around instead
# of approaching it cleanly. Not derived from the model at runtime because
# _do_pick/_do_place need it before the arm ever gets there.
OBJECT_HALF_HEIGHT: dict[str, float] = {
    "plate_1": 0.016, "cup_1": 0.055, "fork_1": 0.004, "spoon_1": 0.004, "napkin_1": 0.003,
}
GRASP_CLEARANCE = 0.015  # m -- gap kept above an object's top surface before closing on it

# Real MuJoCo (x, y, z) table-setting target per zone name, keyed by the same
# strings IntelTableWorld/IntelTablePlanner use as Detection target_zones.
# IK targets for _do_place -- unrelated to scheduler.DEFAULT_LAYOUT, which is
# an abstract reachability space, not a physical coordinate frame (see that
# dict's comment).
ZONE_POSITIONS: dict[str, tuple[float, float, float]] = {
    "center": (0.00, -0.10, 0.02),        # plate
    "upper_right": (0.16, 0.02, 0.055),   # cup
    "left": (-0.16, -0.10, 0.015),        # fork
    "right": (0.16, -0.10, 0.015),        # spoon
    "lower_left": (-0.16, 0.05, 0.012),   # napkin
}


class IntelSimulationUnavailable(RuntimeError):
    """Raised when the optional MuJoCo integration dependency is absent."""


def _mujoco():
    try:
        import mujoco
    except ImportError as exc:  # allow the pure-Python core to remain runnable
        raise IntelSimulationUnavailable(
            "MuJoCo is required for the Intel simulation; install integrations/intel/requirements.txt"
        ) from exc
    return mujoco


def _prefixed(element: ET.Element, prefix: str) -> ET.Element:
    """Copy a Menagerie arm subtree while preserving shared default/mesh names."""
    clone = copy.deepcopy(element)
    for node in clone.iter():
        if "name" in node.attrib:
            node.set("name", f"{prefix}_{node.attrib['name']}")
    return clone


def _body(name: str, pos: str, geom: dict[str, str]) -> ET.Element:
    body = ET.Element("body", {"name": name, "pos": pos})
    ET.SubElement(body, "freejoint", {"name": f"{name}_free"})
    ET.SubElement(body, "geom", {"name": name, **geom})
    return body


def _drawer(pos: str) -> ET.Element:
    """A shallow drawer on a slide joint (OQ-007). Passive -- no actuator, so
    it doesn't change ``model.nu``. Opens toward the arms along +y; a future
    OQ-010 ``OPEN``/``CLOSE`` primitive drives ``data.qpos`` for
    ``drawer_slide`` (or contacts a handle, once grasping is contact-driven)."""
    body = ET.Element("body", {"name": "drawer", "pos": pos})
    ET.SubElement(body, "joint", {
        "name": "drawer_slide", "type": "slide", "axis": "0 1 0",
        "range": "0 .12", "limited": "true", "damping": "3",
    })
    ET.SubElement(body, "geom", {
        "name": "drawer", "type": "box", "size": ".12 .045 .015",
        "rgba": ".30 .19 .11 1", "mass": ".2", "friction": "0.4 .004 .0001",
    })
    return body


def dual_so101_xml() -> str:
    """Return a dual-arm, table-setting MJCF built from the pinned asset."""
    source = ET.parse(ARM_XML).getroot()
    root = ET.Element("mujoco", {"model": "omni_q_dual_so101_table"})
    for tag in ("compiler", "option", "asset", "default"):
        node = source.find(tag)
        if node is not None:
            root.append(copy.deepcopy(node))

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"azimuth": "125", "elevation": "-28"})
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "key", "pos": "0 -0.3 1.3", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane", "size": "0 0 .05", "rgba": ".08 .12 .12 1"})
    ET.SubElement(worldbody, "geom", {
        "name": "table", "type": "box", "pos": "0 -0.10 -0.055", "size": ".42 .36 .05",
        "rgba": ".23 .14 .08 1", "friction": "1 .005 .0001",
    })
    ET.SubElement(worldbody, "camera", {"name": "third_person", "pos": "0 -1.15 .85", "euler": "1.05 0 0"})
    ET.SubElement(worldbody, "camera", {"name": "table_overhead", "pos": "0 -.10 1.20", "euler": "0 0 0"})

    base = source.find("./worldbody/body[@name='Base']")
    if base is None:  # static source validation, not a recoverable runtime state
        raise RuntimeError("pinned SO-ARM100 MJCF is missing Base")
    # The source model excludes self-collision between adjacent links (its
    # own <contact><exclude .../></contact>, e.g. Base/Rotation_Pitch)
    # because those meshes overlap by construction at the joint. Dropping
    # this (as an earlier version of this function did) doesn't just look
    # wrong -- with that self-collision active, rotating the shoulder joint
    # drives it straight into the excluded contact, and MuJoCo generates a
    # resisting force that saturates the actuator's forcerange, jamming the
    # joint anywhere off its rest angle (silently, since nothing before the
    # real IK controller ever commanded it away from home). Re-declare each
    # source exclude pair once per arm, prefixed to match _prefixed()'s
    # renaming.
    source_excludes = source.findall("./contact/exclude")
    contact = ET.SubElement(root, "contact") if source_excludes else None
    for arm, pos in (("left", "-.26 .20 .0"), ("right", ".26 .20 .0")):
        arm_body = _prefixed(base, arm)
        arm_body.set("pos", pos)
        worldbody.append(arm_body)
        for exclude in source_excludes:
            ET.SubElement(contact, "exclude", {
                "body1": f"{arm}_{exclude.attrib['body1']}",
                "body2": f"{arm}_{exclude.attrib['body2']}",
            })

    worldbody.append(_drawer("0 -.40 .01"))
    worldbody.extend([
        # Rim half-height .016 (32mm full thickness), not the original .007
        # (14mm): the SO-101 gripper's own fully-closed pad gap is 21.3mm
        # (integrations/intel/so101_capability_map.md) -- a 14mm rim is
        # geometrically thinner than the gripper can ever close to, so the
        # jaws would sweep past it without contact. 32mm sits inside the
        # gripper's 21.3-77mm graspable range.
        _body("plate_1", "-.13 -.08 .026", {
            "type": "cylinder", "size": ".095 .016", "rgba": ".93 .93 .91 1",
            "mass": ".18", "friction": "0.35 .003 .0001",  # ceramic
        }),
        _body("cup_1", ".16 -.06 .055", {
            "type": "cylinder", "size": ".032 .055", "rgba": ".22 .58 .78 1",
            "mass": ".12", "friction": "0.45 .004 .0001",  # ceramic, needs grip for the pour scenario
        }),
        # fork/spoon start inside the drawer -- retrieval is gated on OPEN, matching
        # the brief's scenario ("open the top drawer, retrieve spoons and forks").
        _body("fork_1", "-.04 -.40 .035", {
            "type": "box", "size": ".012 .075 .004", "rgba": ".72 .73 .75 1",
            "mass": ".04", "friction": "0.5 .003 .0001",  # metal cutlery, small grasp footprint
        }),
        _body("spoon_1", ".04 -.40 .035", {
            "type": "box", "size": ".013 .07 .004", "rgba": ".72 .73 .75 1",
            "mass": ".04", "friction": "0.5 .003 .0001",
        }),
        _body("napkin_1", "-.22 .02 .006", {
            "type": "box", "size": ".07 .05 .003", "rgba": ".90 .40 .38 1",
            "mass": ".02", "friction": "0.9 .006 .0002",  # cloth
        }),
    ])

    actuators = ET.SubElement(root, "actuator")
    source_actuators = source.find("actuator")
    if source_actuators is None:
        raise RuntimeError("pinned SO-ARM100 MJCF is missing actuators")
    for arm in ("left", "right"):
        for actuator in source_actuators:
            copied = copy.deepcopy(actuator)
            copied.set("name", f"{arm}_{actuator.attrib['name']}")
            copied.set("joint", f"{arm}_{actuator.attrib['joint']}")
            actuators.append(copied)
    return ET.tostring(root, encoding="unicode")


def load_dual_so101_model():
    """Load the dual-arm model without writing generated MJCF into the repo."""
    mujoco = _mujoco()
    assets = {
        f"assets/{path.name}": path.read_bytes()
        for path in ARM_ASSETS.glob("*.stl")
    }
    return mujoco.MjModel.from_xml_string(dual_so101_xml(), assets=assets)


class IntelTableWorld(MockWorld):
    """Authoritative table world backed by a real MuJoCo model and timestep."""

    mode = "simulation-scripted-manipulation"

    def __init__(self) -> None:
        # Distinct starting zones (OQ-007): a shared literal zone string for
        # every object collapses the scheduler's workspace-conflict check
        # into full serialization regardless of arm (see
        # integrations/intel/README.md). fork_1/spoon_1 sharing "drawer" is
        # the one intentional exception -- they really do start in the same
        # physical drawer cavity (see dual_so101_xml()).
        super().__init__([
            Detection("plate_1", "plate", "tray_plate", "center"),
            Detection("cup_1", "cup", "tray_cup", "upper_right"),
            Detection("fork_1", "fork", "drawer", "left"),
            Detection("spoon_1", "spoon", "drawer", "right"),
            Detection("napkin_1", "napkin", "tray_napkin", "lower_left"),
            # The drawer itself, so OPEN/CLOSE(object="drawer") passes
            # MockWorld.apply_transition's object-registration check.
            # zone == target_zone: it's a fixture, never "misplaced", so
            # RulePlanner never tries to PICK/MOVE it like tableware.
            Detection("drawer", "fixture", "closed", "closed"),
        ])
        mujoco = _mujoco()
        self.model = load_dual_so101_model()
        self.data = mujoco.MjData(self.model)
        self._mujoco = mujoco
        self._controller_steps = 0
        drawer_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        self._drawer_qpos_adr = self.model.jnt_qposadr[drawer_joint_id]
        # TCP reference per arm -- the same body the capability-map probe
        # (integrations/intel/scripts/probe_so101.py) already uses.
        self._tcp_body = {
            0: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_Fixed_Jaw"),
            6: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_Fixed_Jaw"),
        }
        # (qpos address, dof/qvel address) per tableware freejoint -- used by
        # the IK grasp/place below; a freejoint is 7 qpos (xyz + quat) but
        # only 6 dof (linvel + angvel), so the two addresses are not
        # interchangeable.
        self._object_joints: dict[str, tuple[int, int]] = {}
        for obj_id in ("plate_1", "cup_1", "fork_1", "spoon_1", "napkin_1"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj_id}_free")
            self._object_joints[obj_id] = (self.model.jnt_qposadr[jid], self.model.jnt_dofadr[jid])
        for index, value in enumerate(HOME * 2):
            self.data.ctrl[index] = value
        mujoco.mj_forward(self.model, self.data)
        # mj_forward only computes kinematics for the CURRENT qpos (still all
        # zeros -- MjData starts zeroed); it does not integrate toward ctrl.
        # Actually settle the arms into HOME before anything else touches
        # this world, otherwise every qpos-based read (including the IK
        # solver's "current configuration") starts from a pose the arm was
        # never really in.
        mujoco.mj_step(self.model, self.data, nstep=400)
        self._controller_steps += 400

    def apply_transition(self, request: TransitionRequest) -> TransitionResult:
        op = request.op
        obj = request.args.get("object")
        physics_snapshot = None
        if op in {"PICK", "MOVE", "PLACE"} and obj in self._object_joints:
            # A failed real attempt must not leave the next governed retry
            # starting from a disturbed arm/object configuration. WorldState
            # rollback alone is insufficient because MuJoCo has already
            # integrated joint and free-body dynamics.
            physics_snapshot = (
                self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(),
                float(self.data.time), self._controller_steps,
            )
        # captured before super() mutates WorldState, so a failed real grasp
        # can be reverted below rather than leaving a scripted "success" that
        # the physics never actually delivered.
        prev_zone = self._objects[obj].zone if obj in self._objects else None
        prev_owner = self._ownership.get(obj) if obj in self._ownership else None

        result = super().apply_transition(request)
        arm_offset = 6 if request.actor and "right" in request.actor else 0
        grasp_info: dict[str, Any] = {}

        if op == "PICK" and obj in self._object_joints:
            grasp_info = self._do_pick(arm_offset, obj)
            if not grasp_info["held"]:
                self._ownership[obj] = prev_owner  # revert the scripted grasp claim
                self._restore_physics(physics_snapshot)
                result = replace(result, ok=False)
        elif op in {"MOVE", "PLACE"} and obj in self._object_joints:
            grasp_info = self._do_place(arm_offset, obj, request.args.get("to"))
            if not grasp_info["placed"]:
                if prev_zone is not None:
                    self._objects[obj] = replace(self._objects[obj], zone=prev_zone)
                self._ownership[obj] = prev_owner
                self._restore_physics(physics_snapshot)
                result = replace(result, ok=False)
        else:
            # OQ-010 fallback pose for ops with no dedicated IK behaviour, or
            # a PICK/MOVE/PLACE on an object this adapter doesn't track
            # (e.g. the "drawer" fixture): a fixed coarse target, same as
            # before real grasping existed for tableware.
            target = list(HOME)
            if op in {"PICK", "MOVE"}:
                target[2], target[3], target[5] = 1.20, 0.85, GRIPPER_CLOSED
            elif op == "PLACE":
                target[2], target[3], target[5] = 1.20, 0.85, GRIPPER_OPEN
            elif op == "OPEN" and obj == "drawer":
                target[2], target[3] = 0.9, 0.3
                self.data.qpos[self._drawer_qpos_adr] = DRAWER_OPEN
            elif op == "CLOSE" and obj == "drawer":
                target[2], target[3] = 0.9, 0.3
                self.data.qpos[self._drawer_qpos_adr] = DRAWER_CLOSED
            elif op == "OPEN":
                target[5] = GRIPPER_OPEN
            elif op == "CLOSE":
                target[5] = GRIPPER_CLOSED
            elif op == "ROTATE":
                target[4] = -0.8 if arm_offset else 0.8
            elif op == "PRESENT":
                target[1], target[3], target[5] = -0.8, 0.3, GRIPPER_CLOSED
            elif op == "HANDOFF":
                target[4] = 1.20 if arm_offset else -1.20
            for index, value in enumerate(target, start=arm_offset):
                self.data.ctrl[index] = value
            self._mujoco.mj_step(self.model, self.data, nstep=20)
            self._controller_steps += 20

        detail = {
            **result.detail,
            **grasp_info,
            "sim_time": round(float(self.data.time), 4),
            "controller_steps": self._controller_steps,
            "simulation_mode": self.mode,
            "physics_note": (
                "PICK/MOVE/PLACE on tracked tableware use real IK + a "
                "contact-driven grasp (no teleport, no velocity override); "
                "everything else is still a coarse scripted pose"
            ),
        }
        return replace(result, detail=detail)

    def _restore_physics(self, snapshot) -> None:
        """Restore a pre-attempt MuJoCo state after a rejected transition."""
        if snapshot is None:
            return
        qpos, qvel, ctrl, sim_time, controller_steps = snapshot
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = ctrl
        self.data.time = sim_time
        self._mujoco.mj_forward(self.model, self.data)
        self._controller_steps = controller_steps

    # -- OQ-010: real IK + contact grasp for tracked tableware ----------

    # Weighted (damped, weighted-pseudoinverse) IK joint costs: heavily
    # discourage the proximal joints (shoulder Rotation, then Pitch) relative
    # to the distal ones (Elbow, Wrist_Pitch, Wrist_Roll). The 5-joint chain
    # is kinematically redundant for a 3D position task, and the *unweighted*
    # minimum-norm solution is free to route error reduction through
    # whichever joint the local Jacobian favours -- measured swinging the
    # shoulder/upper-arm through a wide arc and knocking tableware aside even
    # for a small, purely vertical TCP motion where that swing wasn't
    # geometrically necessary. Proximal joints move a much larger swept
    # volume per radian (long lever arm back to the base) than distal ones,
    # so biasing the redundant solution toward the distal joints keeps the
    # arm's own body closer to the path the *end effector* traces instead of
    # ranging far from it.
    _IK_JOINT_WEIGHTS = (80.0, 25.0, 1.0, 1.0, 1.0)

    def _ik_reach(
        self, arm_offset: int, target_pos, *, iters: int = 500,
        max_dq: float = 0.05, sub_steps: int = 3, tol: float = 0.012, weighted: bool = True,
    ) -> float:
        """Damped-least-squares differential IK on this arm's first 5 joints
        (excluding the gripper) driving the ``Fixed_Jaw`` body toward
        ``target_pos``. Each iteration nudges the position servo targets by a
        small joint-space step and lets ``mj_step`` actually move the arm --
        real joint motion, not a jump. Returns the final position error (m);
        callers decide what error counts as "close enough".

        ``weighted=True`` (default) uses ``_IK_JOINT_WEIGHTS`` to discourage
        the proximal joints -- needed near the table, where the unweighted
        minimum-norm solution was measured swinging the shoulder/upper-arm
        through the tableware even for small vertical motions where that
        swing wasn't geometrically necessary. It also converges much slower,
        so callers that are already at a safe height and only need
        horizontal reach (nothing nearby to hit) should pass
        ``weighted=False``."""
        import numpy as np

        mujoco = self._mujoco
        tcp_id = self._tcp_body[arm_offset]
        jacp = np.zeros((3, self.model.nv))
        lo = self.model.jnt_range[arm_offset:arm_offset + 5, 0]
        hi = self.model.jnt_range[arm_offset:arm_offset + 5, 1]
        weights = self._IK_JOINT_WEIGHTS if weighted else (1.0, 1.0, 1.0, 1.0, 1.0)
        w_inv = np.diag([1.0 / w for w in weights])
        target = np.asarray(target_pos, dtype=float)
        err_norm = float("inf")
        for _ in range(iters):
            mujoco.mj_jacBody(self.model, self.data, jacp, None, tcp_id)
            jac = jacp[:, arm_offset:arm_offset + 5]
            err = target - self.data.body(tcp_id).xpos
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                break
            damping = 0.06
            jac_w = jac @ w_inv
            dq = w_inv @ jac_w.T @ np.linalg.solve(
                jac_w @ jac_w.T + damping * damping * np.eye(3), err)
            dq = np.clip(dq, -max_dq, max_dq)
            q_now = self.data.qpos[arm_offset:arm_offset + 5]
            self.data.ctrl[arm_offset:arm_offset + 5] = np.clip(q_now + dq, lo, hi)
            mujoco.mj_step(self.model, self.data, nstep=sub_steps)
            self._controller_steps += sub_steps
        return err_norm

    def _set_gripper(self, arm_offset: int, value: float, *, settle_steps: int = 15) -> None:
        self.data.ctrl[arm_offset + 5] = value
        self._mujoco.mj_step(self.model, self.data, nstep=settle_steps)
        self._controller_steps += settle_steps

    def _ik_track_line(
        self, arm_offset: int, start_xyz, end_xyz, *,
        step: float = 0.03, iters_per_waypoint: int = 180, weighted: bool = True,
    ) -> float:
        """Follow a straight Cartesian line in small (~3cm) waypoints rather
        than one direct IK reach to a distant target -- keeps each solve
        close to its start, which keeps the path close to the line, on top
        of whatever ``weighted`` already buys (see ``_ik_reach``)."""
        import numpy as np

        start = np.asarray(start_xyz, dtype=float)
        end = np.asarray(end_xyz, dtype=float)
        n = max(1, int(np.linalg.norm(end - start) / step))
        err = 0.0
        for i in range(1, n + 1):
            waypoint = start + (end - start) * (i / n)
            err = self._ik_reach(arm_offset, waypoint, iters=iters_per_waypoint, tol=0.012, weighted=weighted)
        return err

    def _move_to(self, arm_offset: int, target_xyz, *, transit_z: float = TRANSIT_HEIGHT) -> float:
        """Up-over-down waypoint motion, each leg tracked in small steps
        (see ``_ik_track_line``): straight up to a safe transit height,
        translate horizontally at that height, then descend. The vertical
        legs stay weighted (near the table/drawer/other tableware); the
        horizontal leg at the already-safe transit height doesn't need it --
        measured no disturbance either way up there, and it converges far
        faster unweighted (proximal joints do most large lateral reaches).
        Not a full collision-aware planner -- it avoids the table, nothing
        else."""
        x, y, z = target_xyz
        cur = self.data.body(self._tcp_body[arm_offset]).xpos.copy()
        self._ik_track_line(arm_offset, cur, (cur[0], cur[1], transit_z))
        cur = self.data.body(self._tcp_body[arm_offset]).xpos.copy()
        self._ik_track_line(arm_offset, cur, (x, y, transit_z), weighted=False)
        cur = self.data.body(self._tcp_body[arm_offset]).xpos.copy()
        return self._ik_track_line(arm_offset, cur, (x, y, z))

    def _do_pick(self, arm_offset: int, obj: str) -> dict[str, Any]:
        """Approach from above (via a safe transit height), close the
        gripper on the real object, lift, and report whether it's actually
        being carried (measured height gain) -- not just whether the motion
        finished."""
        qpos_adr, _ = self._object_joints[obj]
        start_z = float(self.data.qpos[qpos_adr + 2])
        xy = self.data.qpos[qpos_adr:qpos_adr + 2].copy()
        half_h = OBJECT_HALF_HEIGHT.get(obj, 0.01)
        clear_z = start_z + half_h + GRASP_CLEARANCE  # just above the object's real top

        self._set_gripper(arm_offset, GRIPPER_OPEN)
        self._move_to(arm_offset, (xy[0], xy[1], clear_z))
        cur = self.data.body(self._tcp_body[arm_offset]).xpos.copy()
        # centre height, not clear_z: the old fixed "centre + 1.2cm" put the
        # target *inside* tall objects (cup half-height alone is 5.5cm) --
        # the servo just pushed into the object's own volume instead of
        # converging. Centre height is what a side pinch actually needs.
        reach_err = self._ik_track_line(arm_offset, cur, (xy[0], xy[1], start_z))
        self._set_gripper(arm_offset, GRIPPER_CLOSED, settle_steps=60)  # let the grip actually settle
        cur = self.data.body(self._tcp_body[arm_offset]).xpos.copy()
        self._ik_track_line(arm_offset, cur, (xy[0], xy[1], clear_z))  # lift straight up

        lifted_z = float(self.data.qpos[qpos_adr + 2])
        lift = lifted_z - start_z
        return {
            "grasp": "contact", "reach_error_m": round(reach_err, 4),
            "lift_height_m": round(lift, 4), "held": lift > 0.02,
        }

    def _do_place(self, arm_offset: int, obj: str, to_zone: str | None) -> dict[str, Any]:
        """Carry (still gripping -- no teleport), via a safe transit height,
        to the target zone's real table position, release, and report
        whether it actually ended up there, not just whether the arm reached
        the coordinate."""
        import numpy as np

        target = ZONE_POSITIONS.get(to_zone)
        if target is None:
            return {"grasp": "contact", "placed": False, "reason": f"unknown zone {to_zone!r}"}
        target_arr = np.asarray(target)  # z here is the object's intended resting *centre* height
        qpos_adr, _ = self._object_joints[obj]
        half_h = OBJECT_HALF_HEIGHT.get(obj, 0.01)
        clear = np.array([0.0, 0.0, half_h + GRASP_CLEARANCE])

        self._move_to(arm_offset, tuple(target_arr + clear))
        cur = self.data.body(self._tcp_body[arm_offset]).xpos.copy()
        place_err = self._ik_track_line(arm_offset, cur, target_arr)  # centre height, not centre + fixed offset
        self._set_gripper(arm_offset, GRIPPER_OPEN, settle_steps=40)  # let it drop/settle
        cur = self.data.body(self._tcp_body[arm_offset]).xpos.copy()
        self._ik_track_line(arm_offset, cur, target_arr + clear)  # retract straight up

        final_xy = self.data.qpos[qpos_adr:qpos_adr + 2].copy()
        offset = float(np.linalg.norm(final_xy - target_arr[:2]))
        return {
            "grasp": "contact", "reach_error_m": round(place_err, 4),
            "placement_error_m": round(offset, 4), "placed": offset < 0.06,
        }

    def simulation_summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "nq": self.model.nq,
            "nv": self.model.nv,
            "nu": self.model.nu,
            "cameras": self.model.ncam,
            "time": float(self.data.time),
            "controller_steps": self._controller_steps,
            "drawer_qpos": round(float(self.data.qpos[self._drawer_qpos_adr]), 4),
        }


class IntelTablePlanner(RulePlanner):
    """Deterministic role assignment for the first dual-arm table-setting slice."""

    _RIGHT_OBJECTS = {"cup_1", "spoon_1"}
    # objects that physically start inside the drawer (dual_so101_xml) -- their
    # PICK depends on an OPEN step first, matching the brief's literal
    # scenario ("open the top drawer, retrieve spoons and forks").
    _DRAWER_OBJECTS = {"fork_1", "spoon_1"}

    def plan(self, goal, world):
        graph = super().plan(goal, world)

        needs_drawer = any(
            s.op == "PICK" and s.args.get("object") in self._DRAWER_OBJECTS
            for s in graph.steps
        )
        if needs_drawer:
            open_drawer = Step(
                "open_drawer", "manipulate", "OPEN", args={"object": "drawer"},
                rationale="fork/spoon start in the drawer; open it before retrieving them",
            )
            graph.steps.insert(0, open_drawer)
            for s in graph.steps:
                if s.op == "PICK" and s.args.get("object") in self._DRAWER_OBJECTS:
                    s.deps = tuple(sorted(set(s.deps) | {open_drawer.id}))

        for step in graph.steps:
            object_id = step.args.get("object")
            if object_id in self._RIGHT_OBJECTS:
                step.arm = "right"
            elif object_id:
                step.arm = "left"
        return graph


class IntelTableObserver(FakeObserver):
    """Simulation-grounded observation reference; it is never labelled camera live."""

    def observe(self, world):
        observation = super().observe(world)
        return replace(
            observation,
            raw_ref=f"mujoco://dual-so101/state/{world.frame:06d}",
        )


def intel_devices() -> DeviceRouter:
    return DeviceRouter([
        DeviceSpec("intel.perception.sim", ("perception", "verify"), local=True),
        DeviceSpec("intel.cpu.sim", ("reasoning",), local=True),
        DeviceSpec("intel.left_arm", ("arm",), local=True),
        DeviceSpec("intel.right_arm", ("arm",), local=True),
    ])


def build_intel_sim_engine() -> OmniQ:
    """Construct the explicitly labelled Intel simulation path."""
    world = IntelTableWorld()
    return OmniQ(
        world=world,
        observer=IntelTableObserver(),
        planner=IntelTablePlanner(),
        manipulator=FakeManipulator(world),
        verifier=FakeVerifier(),
        device=intel_devices(),
        recorder=FakeRecorder(),
    )


# ---------------------------------------------------------------------------
# OQ-010/OQ-011: bounded contact-physics cup handoff
# ---------------------------------------------------------------------------

CONTACT_HANDOFF_MODE = "simulation-contact-handoff"
CONTACT_HANDOFF_SCHEMA_VERSION = 1
_CUP_GEOM = "cup_1_contact"
_TABLE_GEOM = "handoff_table"
_PAD_MARKER = "jaw_pad_"


class ContactHandoffRejected(RuntimeError):
    """A physics or evidence boundary rejected a proposed handoff."""


@dataclass(frozen=True)
class ContactHandoffConfig:
    """Bounded controller and scene parameters for one reproducible trial.

    The randomized mode intentionally keeps perturbations narrow.  It is a
    robustness characterization of this particular proxy scene, not a
    perception, policy, or hardware claim.
    """

    seed: int = 19
    randomized: bool = False
    position_jitter_m: float = 0.002
    orientation_jitter_rad: float = 0.025
    ik_damping: float = 0.001
    ik_iterations: int = 700
    ik_tolerance_m: float = 0.002
    interpolation_segments: int = 8
    motion_steps_per_segment: int = 75
    motion_timeout_steps: int = 120
    # 0.6 s at the pinned MuJoCo timestep; enough for a perturbed cup to
    # dissipate post-release sliding before stable-placement admission.
    settle_steps: int = 300
    joint_limit_margin_rad: float = 0.03
    # Measured against the proxy's compliant finger pads.  This remains a
    # controller stop bound, not a hardware-safe force calibration.
    max_contact_force_n: float = 80.0
    max_relative_cup_distance_m: float = 0.105
    shared_workspace_x_limit_m: float = 0.18
    gripper_open_rad: float = 1.65
    # The proxy cup is 52 mm across; this leaves a compliant, contact-rich
    # clamp instead of driving the 21 mm hard-stop through the object.
    gripper_closed_rad: float = 0.35

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContactHandoffReceipt:
    """Tamper-evident, controller-level evidence for one contact trial."""

    schema_version: int
    mode: str
    mjcf_sha256: str
    controller: dict[str, Any]
    seed: int
    deterministic: bool
    randomized: bool
    source_state_revision: int
    phase_timings: tuple[dict[str, Any], ...]
    contact_transitions: tuple[dict[str, Any], ...]
    state_samples: tuple[dict[str, Any], ...]
    final_cup_pose: dict[str, Any]
    final_owner: str | None
    final_stable: bool
    success: bool
    failure_reason: str | None
    content_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "mjcf_sha256": self.mjcf_sha256,
            "controller": self.controller,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "randomized": self.randomized,
            "source_state_revision": self.source_state_revision,
            "phase_timings": list(self.phase_timings),
            "contact_transitions": list(self.contact_transitions),
            "state_samples": list(self.state_samples),
            "final_cup_pose": self.final_cup_pose,
            "final_owner": self.final_owner,
            "final_stable": self.final_stable,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "content_hash": self.content_hash,
        }


def _contact_receipt_hash(payload: dict[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "content_hash"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_contact_handoff_receipt(receipt: ContactHandoffReceipt | dict[str, Any]) -> None:
    """Fail closed when a controller receipt has been altered or relabelled."""
    payload = receipt.as_dict() if isinstance(receipt, ContactHandoffReceipt) else dict(receipt)
    if payload.get("schema_version") != CONTACT_HANDOFF_SCHEMA_VERSION:
        raise ContactHandoffRejected("unsupported contact-handoff receipt schema")
    if payload.get("mode") != CONTACT_HANDOFF_MODE:
        raise ContactHandoffRejected("receipt is not contact-handoff evidence")
    if not payload.get("mjcf_sha256"):
        raise ContactHandoffRejected("receipt is missing an MJCF hash")
    if payload.get("content_hash") != _contact_receipt_hash(payload):
        raise ContactHandoffRejected("contact-handoff receipt content hash mismatch")


def write_contact_handoff_receipt(receipt: ContactHandoffReceipt, path: str | Path) -> Path:
    """Atomically persist one immutable controller receipt at a caller-owned path."""
    verify_contact_handoff_receipt(receipt)
    target = Path(path)
    if target.exists():
        raise ContactHandoffRejected(
            "refusing to overwrite existing contact-handoff receipt"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    encoded = json.dumps(receipt.as_dict(), sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def _contact_scene_cup_pose(config: ContactHandoffConfig) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Build-time perturbation only; runtime never writes the free-joint pose."""
    # Keep the cup clear of the parked left finger pads at simulator reset.
    # It is then approached dynamically from the front; this avoids treating
    # initial interpenetration as a grasp or as a contact-force success.
    position = [-0.260, -0.150, 0.050]
    orientation = [0.0, 0.0, 0.0]
    if config.randomized:
        rng = random.Random(config.seed)
        position[0] += rng.uniform(-config.position_jitter_m, config.position_jitter_m)
        position[1] += rng.uniform(-config.position_jitter_m, config.position_jitter_m)
        orientation = [
            rng.uniform(-config.orientation_jitter_rad, config.orientation_jitter_rad),
            rng.uniform(-config.orientation_jitter_rad, config.orientation_jitter_rad),
            rng.uniform(-config.orientation_jitter_rad, config.orientation_jitter_rad),
        ]
    return tuple(position), tuple(orientation)


def _configure_contact_arm(arm_body: ET.Element) -> None:
    """Keep only named finger-pad contacts active in the contact test scene.

    The group mask deliberately permits cup↔pad and cup↔table contacts while
    suppressing pad↔pad and unmodelled mesh contacts.  That makes the receipt's
    contact ownership calculation auditable rather than a side effect of mesh
    tessellation.
    """
    for geom in arm_body.iter("geom"):
        name = geom.attrib.get("name", "")
        if _PAD_MARKER in name:
            geom.set("contype", "4")
            geom.set("conaffinity", "0")
            geom.set("group", "3")
            geom.set("friction", "3.00 0.020 0.001")
            geom.set("solref", ".050 1")
            geom.set("solimp", ".80 .95 .010")
        else:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def contact_handoff_xml(config: ContactHandoffConfig | None = None) -> str:
    """Return the isolated, contact-physics MJCF for the ``cup_1`` handoff.

    This is intentionally not an edit to :func:`dual_so101_xml`: the legacy
    table-setting route must remain a visibly distinct scripted route.
    """
    config = config or ContactHandoffConfig()
    cup_pos, cup_euler = _contact_scene_cup_pose(config)
    source = ET.parse(ARM_XML).getroot()
    root = ET.Element("mujoco", {"model": "omni_q_dual_so101_contact_handoff"})
    for tag in ("compiler", "option", "asset", "default"):
        node = source.find(tag)
        if node is not None:
            root.append(copy.deepcopy(node))

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"azimuth": "125", "elevation": "-28"})
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {
        "name": "key", "pos": "0 -0.25 1.2", "dir": "0 0 -1", "directional": "true",
    })
    ET.SubElement(worldbody, "geom", {
        "name": _TABLE_GEOM,
        "type": "box",
        "pos": "0 -0.10 -0.05",
        "size": ".42 .36 .05",
        "rgba": ".23 .14 .08 1",
        "friction": "1.20 0.006 0.0002",
        "contype": "2",
        "conaffinity": "16",
    })
    # Keep failed trials bounded when a cup is knocked beyond the table edge.
    # This is containment only: the controller still fails closed on a carried
    # cup below the drop bound, and final success requires cup↔table contact.
    ET.SubElement(worldbody, "geom", {
        "name": "handoff_floor",
        "type": "plane",
        "pos": "0 0 -0.12",
        "size": "0 0 .05",
        "rgba": ".08 .12 .12 1",
        "friction": "0.80 0.005 0.0001",
        "contype": "2",
        "conaffinity": "16",
    })
    ET.SubElement(worldbody, "camera", {
        "name": "handoff_third_person", "pos": "0 -1.15 .85", "euler": "1.05 0 0",
    })

    base = source.find("./worldbody/body[@name='Base']")
    if base is None:
        raise RuntimeError("pinned SO-ARM100 MJCF is missing Base")
    for arm, pos in (("left", "-.26 .20 .0"), ("right", ".26 .20 .0")):
        arm_body = _prefixed(base, arm)
        arm_body.set("pos", pos)
        _configure_contact_arm(arm_body)
        worldbody.append(arm_body)

    cup = ET.SubElement(worldbody, "body", {
        "name": "cup_1",
        "pos": "%.6f %.6f %.6f" % cup_pos,
        "euler": "%.6f %.6f %.6f" % cup_euler,
    })
    ET.SubElement(cup, "freejoint", {"name": "cup_1_free"})
    ET.SubElement(cup, "geom", {
        "name": _CUP_GEOM,
        "type": "cylinder",
        "size": ".022 .050",
        "mass": ".010",
        "rgba": ".22 .58 .78 1",
        "friction": "3.00 0.020 0.001",
        "solref": ".050 1",
        "solimp": ".80 .95 .010",
        "contype": "16",
        "conaffinity": "4",
        "group": "1",
    })

    actuators = ET.SubElement(root, "actuator")
    source_actuators = source.find("actuator")
    if source_actuators is None:
        raise RuntimeError("pinned SO-ARM100 MJCF is missing actuators")
    for arm in ("left", "right"):
        for actuator in source_actuators:
            copied = copy.deepcopy(actuator)
            copied.set("name", f"{arm}_{actuator.attrib['name']}")
            copied.set("joint", f"{arm}_{actuator.attrib['joint']}")
            actuators.append(copied)
    return ET.tostring(root, encoding="unicode")


def load_contact_handoff_model(config: ContactHandoffConfig | None = None):
    """Load the generated contact scene without creating a generated MJCF file."""
    mujoco = _mujoco()
    assets = {f"assets/{path.name}": path.read_bytes() for path in ARM_ASSETS.glob("*.stl")}
    return mujoco.MjModel.from_xml_string(contact_handoff_xml(config), assets=assets)


class IntelContactHandoffWorld(MockWorld):
    """Authoritative world for one atomic, verified physical handoff.

    The physical controller has no authority to alter ``WorldState``.  It can
    only attach a proposed contact receipt; this world admits the represented
    handoff after validating the receipt, org scope, and state revision.
    """

    mode = CONTACT_HANDOFF_MODE

    def __init__(self, config: ContactHandoffConfig | None = None) -> None:
        self.config = config or ContactHandoffConfig()
        super().__init__([
            Detection("cup_1", "cup", "left_table", "right_table"),
        ])
        self._mujoco = _mujoco()
        self.model_xml = contact_handoff_xml(self.config)
        assets = {f"assets/{path.name}": path.read_bytes() for path in ARM_ASSETS.glob("*.stl")}
        self.model = self._mujoco.MjModel.from_xml_string(self.model_xml, assets=assets)
        self.data = self._mujoco.MjData(self.model)
        self.mjcf_sha256 = "sha256:" + hashlib.sha256(self.model_xml.encode("utf-8")).hexdigest()
        self.pending_contact_receipt: ContactHandoffReceipt | None = None
        self.last_admitted_receipt: ContactHandoffReceipt | None = None
        self.contact_events: list[dict[str, Any]] = []

        # Initializing only the arm joints is a normal simulator reset, not a
        # cup pose write.  The free-joint state remains the pose compiled from
        # ``contact_handoff_xml`` (including any randomized build-time offset).
        self.data.qpos[:12] = list(HOME) * 2
        self.data.ctrl[:12] = list(HOME) * 2
        self.data.qpos[5] = self.config.gripper_open_rad
        self.data.qpos[11] = self.config.gripper_open_rad
        self.data.ctrl[5] = self.config.gripper_open_rad
        self.data.ctrl[11] = self.config.gripper_open_rad
        self._mujoco.mj_forward(self.model, self.data)

    def apply_transition(self, request: TransitionRequest) -> TransitionResult:
        if request.expected_revision != self.revision:
            raise TransitionRejected(
                f"{request.step_id}: expected revision {request.expected_revision}, current revision is {self.revision}"
            )
        if request.org_id != self.org_id:
            raise TransitionRejected(
                f"{request.step_id}: request org {request.org_id} does not match world org {self.org_id}"
            )
        if request.op != "HANDOFF" or request.args.get("object") != "cup_1":
            raise TransitionRejected(f"{request.step_id}: contact world only admits HANDOFF of cup_1")
        if request.actor != "intel.left_arm" or request.args.get("to_actor") != "intel.right_arm":
            raise TransitionRejected(f"{request.step_id}: contact handoff requires left giver and right receiver")
        receipt = self.pending_contact_receipt
        if receipt is None:
            raise TransitionRejected(f"{request.step_id}: no contact receipt was proposed")
        try:
            verify_contact_handoff_receipt(receipt)
        except ContactHandoffRejected as exc:
            raise TransitionRejected(f"{request.step_id}: {exc}") from exc
        if not receipt.success or not receipt.final_stable or receipt.final_owner is not None:
            raise TransitionRejected(f"{request.step_id}: contact handoff did not reach a stable released cup")
        if receipt.source_state_revision != self.revision:
            raise TransitionRejected(f"{request.step_id}: contact evidence has a stale state revision")

        self._objects["cup_1"] = replace(self._objects["cup_1"], zone="right_table")
        self._ownership["cup_1"] = None
        self.revision += 1
        self.last_admitted_receipt = receipt
        self.pending_contact_receipt = None
        event = {
            "kind": "contact_handoff.admitted",
            "state_revision": self.revision,
            "receipt_hash": receipt.content_hash,
            "mjcf_sha256": receipt.mjcf_sha256,
        }
        self.contact_events.append(event)
        return TransitionResult(
            step_id=request.step_id,
            ok=True,
            state_revision=self.revision,
            detail={
                "handed_off": "cup_1",
                "to_actor": "intel.right_arm",
                "simulation_mode": self.mode,
                "contact_receipt": receipt.as_dict(),
                "domain_event": event,
            },
        )

    def move_object(self, obj_id: str, zone: str) -> None:
        """Reject an unaudited background mutation in the contact world.

        A real external disturbance must enter through an integration that
        records its causal domain event first; inheriting ``MockWorld``'s
        convenience mutator here would silently bypass that boundary.
        """
        raise TransitionRejected(
            f"background mutation of {obj_id} requires a recorded contact domain event"
        )

    def simulation_summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "nq": self.model.nq,
            "nv": self.model.nv,
            "nu": self.model.nu,
            "time": float(self.data.time),
            "mjcf_sha256": self.mjcf_sha256,
            "admitted_contact_events": len(self.contact_events),
        }


class _ContactHandoffController:
    """Sequential, contact-verified primitive controller for a single cup.

    Every arm target is calculated with damped least squares and is sent as an
    interpolated position-actuator command.  It intentionally contains no
    equality/weld attachment, no mocap body, and no write to the cup freejoint.
    """

    def __init__(self, world: IntelContactHandoffWorld) -> None:
        self.world = world
        self.model = world.model
        self.data = world.data
        self.mujoco = world._mujoco
        self.config = world.config
        try:
            import numpy as np
        except ImportError as exc:  # MuJoCo itself requires numpy, but stay explicit
            raise IntelSimulationUnavailable("NumPy is required by the MuJoCo handoff controller") from exc
        self.np = np
        self.ik_data = self.mujoco.MjData(self.model)
        self.phase_timings: list[dict[str, Any]] = []
        self.contact_transitions: list[dict[str, Any]] = []
        self.state_samples: list[dict[str, Any]] = []
        self._last_contact_signature: tuple[str, ...] = ()
        self._carry_required = False
        self._phase = "initial"

    def run(self, source_state_revision: int) -> ContactHandoffReceipt:
        failure: str | None = None
        final_stable = False
        try:
            self._sample("initial")
            self._run_phase("initial_settle", lambda: self._advance(self.config.settle_steps, "initial_settle"))

            cup0 = self._cup_position()
            left_grasp = self._grasp_target("left", cup0, wrist_roll=1.65)
            self._run_phase("left_approach", lambda: self._move_to(
                "left", self._raised_target("left", left_grasp, 0.075), 1.65, "left_approach",
            ))
            self._run_phase("left_descend", lambda: self._move_to("left", left_grasp, 1.65, "left_descend"))
            self._run_phase("left_grasp", lambda: self._command_gripper("left", self.config.gripper_closed_rad, "left_grasp"))
            self._require_owner("left", "left_grasp")
            self._carry_required = True

            # The shared target was selected inside both arm workspaces during
            # the measured OQ-003 proxy sweep.  A tighter operator-configured
            # bound rejects entry before either arm crosses it.
            left_shared = self.np.array([-0.040, -0.110, 0.134])
            if abs(float(left_shared[0])) > self.config.shared_workspace_x_limit_m:
                raise ContactHandoffRejected("unsafe shared-workspace entry")
            self._run_phase("left_lift_transfer", lambda: self._move_to(
                "left", left_shared, 1.65, "left_lift_transfer",
            ))
            self._require_owner("left", "left_lift_transfer")

            cup_shared = self._cup_position()
            right_grasp = self._grasp_target("right", cup_shared, wrist_roll=-1.65)
            self._run_phase("right_approach", lambda: self._move_to(
                "right", self._raised_target("right", right_grasp, 0.035), -1.65, "right_approach",
            ))
            self._run_phase("right_descend", lambda: self._move_to("right", right_grasp, -1.65, "right_descend"))
            self._run_phase("right_grasp", lambda: self._command_gripper("right", self.config.gripper_closed_rad, "right_grasp"))
            self._require_owner("dual", "dual_contact_confirmation")

            def release_left() -> None:
                self._command_gripper("left", self.config.gripper_open_rad, "left_release")
                # Opening alone can leave a compliant pad touching the cup;
                # withdraw the giver before admitting right-only ownership.
                self._move_to("left", self.np.array([-0.115, -0.075, 0.180]), 1.65, "left_release")

            self._run_phase("left_release", release_left)
            self._require_owner("right", "left_release")

            self._run_phase("right_retreat", lambda: self._move_to(
                "right", self.np.array([0.115, -0.105, 0.145]), -1.65, "right_retreat",
            ))
            self._require_owner("right", "right_retreat")
            self._carry_required = False
            self._run_phase("right_place", lambda: self._move_to(
                "right", self.np.array([0.145, -0.085, 0.050]), -1.65, "right_place",
            ))
            def release_right() -> None:
                # Open to a clearance gap before lifting the pads away from
                # the cup. Fully opening while the compliant pads are still
                # pressed against a tilted, perturbed cup can impart a large
                # lateral impulse.
                preopen = self.config.gripper_closed_rad + 0.45
                self._command_gripper("right", preopen, "right_release_preopen")
                self._move_to("right", self.np.array([0.145, -0.085, 0.090]), -1.65, "right_release_lift")
                self._command_gripper("right", self.config.gripper_open_rad, "right_release")
                self._move_to("right", self.np.array([0.190, -0.060, 0.155]), -1.65, "right_release")

            self._run_phase("right_release", release_right)
            self._run_phase("final_settle", lambda: self._advance(self.config.settle_steps, "final_settle"))
            if self._ownership() is not None:
                raise ContactHandoffRejected("cup remains held after the requested release")
            final_stable = self._cup_is_stable_on_table()
            if not final_stable:
                raise ContactHandoffRejected("cup did not reach stable final placement")
        except ContactHandoffRejected as exc:
            failure = str(exc)
        finally:
            self._sample("final")

        receipt = ContactHandoffReceipt(
            schema_version=CONTACT_HANDOFF_SCHEMA_VERSION,
            mode=CONTACT_HANDOFF_MODE,
            mjcf_sha256=self.world.mjcf_sha256,
            controller=self.config.as_dict(),
            seed=self.config.seed,
            deterministic=not self.config.randomized,
            randomized=self.config.randomized,
            source_state_revision=source_state_revision,
            phase_timings=tuple(self.phase_timings),
            contact_transitions=tuple(self.contact_transitions),
            state_samples=tuple(self.state_samples),
            final_cup_pose=self._cup_pose(),
            final_owner=self._ownership(),
            final_stable=final_stable,
            success=failure is None and final_stable,
            failure_reason=failure,
        )
        payload = receipt.as_dict()
        return ContactHandoffReceipt(**{**payload, "content_hash": _contact_receipt_hash(payload)})

    # -- bounded motion -------------------------------------------------
    def _arm_slice(self, arm: str) -> slice:
        return slice(0, 6) if arm == "left" else slice(6, 12)

    def _arm_ids(self, arm: str) -> tuple[int, int, int]:
        offset = 0 if arm == "left" else 6
        return (
            self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, f"{arm}_fixed_jaw_pad_4"),
            self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, f"{arm}_moving_jaw_pad_4"),
            offset,
        )

    def _solve_ik(self, arm: str, target: Any, wrist_roll: float) -> Any:
        fixed_id, _, offset = self._arm_ids(arm)
        q = self.data.qpos[offset:offset + 5].copy()
        q[4] = wrist_roll
        desired = self.np.asarray(target, dtype=float)
        joint_low = self.model.jnt_range[offset:offset + 4, 0] + self.config.joint_limit_margin_rad
        joint_high = self.model.jnt_range[offset:offset + 4, 1] - self.config.joint_limit_margin_rad
        dofs = self.np.arange(offset, offset + 4)
        for _ in range(self.config.ik_iterations):
            self.ik_data.qpos[:] = self.data.qpos
            self.ik_data.qpos[offset:offset + 5] = q
            self.ik_data.qpos[offset + 5] = self.config.gripper_open_rad
            self.mujoco.mj_forward(self.model, self.ik_data)
            error = desired - self.ik_data.geom_xpos[fixed_id]
            if float(self.np.linalg.norm(error)) <= self.config.ik_tolerance_m:
                return q
            jacobian = self.np.zeros((3, self.model.nv))
            self.mujoco.mj_jacGeom(self.model, self.ik_data, jacobian, None, fixed_id)
            active = jacobian[:, dofs]
            damped = active @ active.T + (self.config.ik_damping ** 2) * self.np.eye(3)
            delta = active.T @ self.np.linalg.solve(damped, error)
            q[:4] = self.np.clip(q[:4] + self.np.clip(delta, -0.075, 0.075), joint_low, joint_high)
        raise ContactHandoffRejected(f"controller timeout: {arm} inverse-kinematics target not reached")

    def _grasp_target(self, arm: str, cup_position: Any, wrist_roll: float) -> Any:
        """Choose a fixed-pad target so open pads bracket the observed cup."""
        desired_cup = self.np.asarray(cup_position, dtype=float)
        if arm == "right":
            # The mirrored SO-ARM100 jaw opens mainly along +y at this wrist
            # roll.  A small front offset puts the cup inside the closed gap;
            # solving the exact open-gap midpoint would ask the right arm for
            # a near-limit pose and would be less robust to table settling.
            target = desired_cup + self.np.array([0.006, -0.012, 0.0])
            self._solve_ik(arm, target, wrist_roll)
            return target
        target = desired_cup.copy()
        for _ in range(4):
            q = self._solve_ik(arm, target, wrist_roll)
            fixed_id, moving_id, offset = self._arm_ids(arm)
            self.ik_data.qpos[:] = self.data.qpos
            self.ik_data.qpos[offset:offset + 5] = q
            self.ik_data.qpos[offset + 5] = self.config.gripper_open_rad
            self.mujoco.mj_forward(self.model, self.ik_data)
            opening = self.ik_data.geom_xpos[moving_id] - self.ik_data.geom_xpos[fixed_id]
            next_target = desired_cup - 0.5 * opening
            if float(self.np.linalg.norm(next_target - target)) < self.config.ik_tolerance_m:
                return next_target
            target = next_target
        return target

    def _raised_target(self, arm: str, grasp_target: Any, dz: float) -> Any:
        target = self.np.asarray(grasp_target, dtype=float).copy()
        target[2] += dz
        return target

    def _move_to(self, arm: str, target: Any, wrist_roll: float, phase: str) -> None:
        if self.config.motion_timeout_steps <= 0:
            raise ContactHandoffRejected(f"controller timeout: {phase} has no allowed control steps")
        q = self._solve_ik(arm, target, wrist_roll)
        arm_slice = self._arm_slice(arm)
        start = self.data.ctrl[arm_slice].copy()
        destination = self.np.concatenate((q, [self.data.ctrl[arm_slice][-1]]))
        for segment in range(1, self.config.interpolation_segments + 1):
            fraction = segment / self.config.interpolation_segments
            self.data.ctrl[arm_slice] = start + fraction * (destination - start)
            self._advance(self.config.motion_steps_per_segment, phase)
        fixed_id, _, _ = self._arm_ids(arm)
        error = float(self.np.linalg.norm(self.data.geom_xpos[fixed_id] - self.np.asarray(target)))
        if error > 0.035:
            raise ContactHandoffRejected(f"controller timeout: {phase} target error {error:.3f} m")

    def _command_gripper(self, arm: str, target: float, phase: str) -> None:
        arm_slice = self._arm_slice(arm)
        start = float(self.data.ctrl[arm_slice][-1])
        for segment in range(1, self.config.interpolation_segments + 1):
            self.data.ctrl[arm_slice][-1] = start + (target - start) * segment / self.config.interpolation_segments
            self._advance(max(1, self.config.motion_steps_per_segment // 2), phase)

    def _advance(self, steps: int, phase: str) -> None:
        for step in range(steps):
            self.mujoco.mj_step(self.model, self.data)
            if step % 10 == 0:
                self._record_contact_transition(phase)
                self._guard(phase)
        self._record_contact_transition(phase)

    def _guard(self, phase: str) -> None:
        for index in range(12):
            joint_id = int(self.model.actuator_trnid[index, 0])
            low, high = self.model.jnt_range[joint_id]
            value = self.data.qpos[self.model.jnt_qposadr[joint_id]]
            if min(value - low, high - value) < self.config.joint_limit_margin_rad:
                raise ContactHandoffRejected(f"joint limit approached during {phase}")
        contacts = self._contact_snapshot()
        if contacts["max_gripper_force_n"] > self.config.max_contact_force_n:
            raise ContactHandoffRejected(f"unsafe collision during {phase}")
        if self._carry_required and self._cup_position()[2] < 0.020:
            raise ContactHandoffRejected(f"cup dropped during {phase}")

    # -- contact/state verification ------------------------------------
    def _contact_snapshot(self) -> dict[str, Any]:
        cup_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, _CUP_GEOM)
        table_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, _TABLE_GEOM)
        by_arm = {"left": 0, "right": 0}
        labels: list[str] = []
        max_force = 0.0
        cup_on_table = False
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            first, second = int(contact.geom1), int(contact.geom2)
            if cup_id not in {first, second}:
                continue
            other = second if first == cup_id else first
            other_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, other) or "unnamed"
            force = self.np.zeros(6)
            self.mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            if other == table_id:
                cup_on_table = True
                labels.append("cup:table")
                continue
            if other_name.startswith("left_") and _PAD_MARKER in other_name:
                by_arm["left"] += 1
                max_force = max(max_force, abs(float(force[0])))
                labels.append(f"cup:{other_name}")
            elif other_name.startswith("right_") and _PAD_MARKER in other_name:
                by_arm["right"] += 1
                max_force = max(max_force, abs(float(force[0])))
                labels.append(f"cup:{other_name}")
            else:
                labels.append(f"cup:unexpected:{other_name}")
                max_force = max(max_force, abs(float(force[0])))
        return {
            "by_arm": by_arm,
            "labels": tuple(sorted(set(labels))),
            "max_gripper_force_n": max_force,
            "cup_on_table": cup_on_table,
        }

    def _ownership(self) -> str | None:
        contacts = self._contact_snapshot()["by_arm"]
        cup = self._cup_position()
        candidates: list[str] = []
        for arm in ("left", "right"):
            fixed_id, moving_id, _ = self._arm_ids(arm)
            center = 0.5 * (self.data.geom_xpos[fixed_id] + self.data.geom_xpos[moving_id])
            if contacts[arm] >= 1 and float(self.np.linalg.norm(cup - center)) <= self.config.max_relative_cup_distance_m:
                candidates.append(arm)
        if candidates == ["left"]:
            return "left"
        if candidates == ["right"]:
            return "right"
        if candidates == ["left", "right"]:
            return "dual"
        return None

    def _require_owner(self, expected: str, phase: str) -> None:
        actual = self._ownership()
        if actual == expected:
            return
        if actual is None:
            raise ContactHandoffRejected(f"missing contact during {phase}")
        raise ContactHandoffRejected(f"ambiguous ownership during {phase}: observed {actual}, expected {expected}")

    def _cup_position(self) -> Any:
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "cup_1")
        return self.data.xpos[body_id].copy()

    def _cup_pose(self) -> dict[str, Any]:
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "cup_1")
        return {
            "position": [round(float(value), 6) for value in self.data.xpos[body_id]],
            "quaternion": [round(float(value), 6) for value in self.data.xquat[body_id]],
            "linear_velocity": [round(float(value), 6) for value in self._cup_velocity()],
        }

    def _cup_velocity(self) -> Any:
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "cup_1")
        velocity = self.np.zeros(6)
        self.mujoco.mj_objectVelocity(self.model, self.data, self.mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
        return velocity[3:]

    def _cup_is_stable_on_table(self) -> bool:
        contacts = self._contact_snapshot()
        velocity = self._cup_velocity()
        return bool(
            contacts["cup_on_table"]
            and float(self.np.linalg.norm(velocity)) < 0.025
            and self._cup_position()[2] > 0.020
        )

    # -- receipt trace --------------------------------------------------
    def _run_phase(self, name: str, operation: Any) -> None:
        self._phase = name
        start = float(self.data.time)
        operation()
        self.phase_timings.append({
            "phase": name,
            "start_s": round(start, 6),
            "end_s": round(float(self.data.time), 6),
            "duration_s": round(float(self.data.time) - start, 6),
        })
        self._sample(name)

    def _record_contact_transition(self, phase: str) -> None:
        signature = self._contact_snapshot()["labels"]
        if signature == self._last_contact_signature:
            return
        self.contact_transitions.append({
            "phase": phase,
            "sim_time_s": round(float(self.data.time), 6),
            "from": list(self._last_contact_signature),
            "to": list(signature),
        })
        self._last_contact_signature = signature

    def _sample(self, phase: str) -> None:
        contacts = self._contact_snapshot()
        limits: list[float] = []
        for index in range(12):
            joint_id = int(self.model.actuator_trnid[index, 0])
            low, high = self.model.jnt_range[joint_id]
            value = self.data.qpos[self.model.jnt_qposadr[joint_id]]
            limits.append(float(min(value - low, high - value)))
        self.state_samples.append({
            "phase": phase,
            "sim_time_s": round(float(self.data.time), 6),
            "cup": self._cup_pose(),
            "ownership": self._ownership(),
            "contacts": list(contacts["labels"]),
            "contact_counts": dict(contacts["by_arm"]),
            "cup_on_table": contacts["cup_on_table"],
            "max_gripper_force_n": round(float(contacts["max_gripper_force_n"]), 6),
            "min_joint_margin_rad": round(min(limits), 6),
        })


class IntelContactHandoffManipulator:
    """Physical proposal provider; final world authority remains external."""

    def __init__(self, world: IntelContactHandoffWorld) -> None:
        self.world = world
        self._attempted = False
        self._attempt_receipt: ContactHandoffReceipt | None = None

    def supports(self, op: str) -> bool:
        return op == "HANDOFF"

    def execute(self, step: Step, world: Any) -> ManipResult:
        if step.op != "HANDOFF" or step.args.get("object") != "cup_1":
            return ManipResult(step.id, False, {"error": "contact controller only supports HANDOFF cup_1"})
        # MuJoCo state is mutable even when authoritative WorldState is not.
        # Re-entering a failed controller from OmniQ's generic revision loop
        # would therefore retry on a dirty scene and make the receipt's causal
        # history ambiguous. A caller that wants another physical attempt must
        # construct a fresh contact world/engine.
        if self._attempted:
            detail: dict[str, Any] = {
                "error": "contact handoff is one-shot; retry requires a fresh contact world",
            }
            if self._attempt_receipt is not None:
                detail["contact_handoff"] = self._attempt_receipt.as_dict()
            return ManipResult(step.id, False, detail)
        self._attempted = True
        receipt = _ContactHandoffController(self.world).run(source_state_revision=world.revision)
        self._attempt_receipt = receipt
        self.world.pending_contact_receipt = receipt
        return ManipResult(step.id, receipt.success, {"contact_handoff": receipt.as_dict()})


class IntelContactHandoffPlanner:
    """One narrow proposal: no AI planner may widen this controller's scope."""

    def __init__(self, world: IntelContactHandoffWorld | None = None) -> None:
        self.world = world
        self.last_decision: PlanDecision | None = None

    def plan(self, goal: str, world: Any) -> PlanGraph:
        cup = world.objects.get("cup_1")
        if cup is None or not cup.authoritative:
            reason = "cup_1 lacks a live authoritative observation"
            graph = PlanGraph(goal=goal)
            self.last_decision = PlanDecision(
                goal=goal,
                revision=graph.revision,
                selected_ops=(),
                candidates_considered=1 if cup is not None else 0,
                candidates_feasible=0,
                governing_constraints=("live_observation",),
                rejected={"cup_1": reason},
                state_hash="contact-handoff",
                reason=reason,
            )
            return graph
        if cup.zone == cup.target_zone:
            reason = "cup_1 is already at its target zone; no second handoff is authorized"
            graph = PlanGraph(goal=goal)
            self.last_decision = PlanDecision(
                goal=goal,
                revision=graph.revision,
                selected_ops=(),
                candidates_considered=1,
                candidates_feasible=0,
                governing_constraints=("single_handoff", "live_observation"),
                rejected={"cup_1": reason},
                state_hash="contact-handoff",
                reason=reason,
            )
            return graph
        if self.world is not None:
            prior = self.world.pending_contact_receipt or self.world.last_admitted_receipt
            if prior is not None:
                reason = "contact handoff was already attempted; retry requires a fresh contact world"
                graph = PlanGraph(goal=goal)
                self.last_decision = PlanDecision(
                    goal=goal,
                    revision=graph.revision,
                    selected_ops=(),
                    candidates_considered=1,
                    candidates_feasible=0,
                    governing_constraints=("single_handoff", "contact_receipt"),
                    rejected={"cup_1": reason},
                    state_hash="contact-handoff",
                    reason=reason,
                )
                return graph
        step = Step(
            "contact_handoff_cup_1",
            "manipulate",
            "HANDOFF",
            args={"object": "cup_1", "to_actor": "intel.right_arm"},
            arm="left",
            rationale="bounded physical transfer of cup_1; authority waits for contact receipt verification",
        )
        graph = PlanGraph(goal=goal, steps=[step, Step(
            "verify_contact_handoff",
            "verify",
            "VERIFY",
            deps=(step.id,),
            rationale="admit success only after the world validates the physics receipt",
        )])
        self.last_decision = PlanDecision(
            goal=goal,
            revision=graph.revision,
            selected_ops=("HANDOFF", "VERIFY"),
            candidates_considered=1,
            candidates_feasible=1,
            governing_constraints=("contact_receipt", "state_revision", "org_scope"),
            state_hash="contact-handoff",
            reason="one fixed cup handoff is within this evidence slice",
        )
        return graph

    def replan(self, current: PlanGraph, world: Any, reason: str) -> PlanGraph:
        fresh = self.plan(current.goal, world)
        fresh.revision = current.revision + 1
        if self.last_decision is not None:
            self.last_decision.revision = fresh.revision
            self.last_decision.reason = f"replan: {reason}"
        return fresh


class IntelContactHandoffObserver(FakeObserver):
    def observe(self, world: Any):
        observation = super().observe(world)
        return replace(observation, raw_ref=f"mujoco://dual-so101/contact-state/{world.frame:06d}")


class IntelContactHandoffVerifier:
    """Verification reads the admitted receipt, never a controller self-claim."""

    def __init__(self, world: IntelContactHandoffWorld) -> None:
        self.world = world

    def check(self, step: Step, observation: Any) -> VerifyResult:
        receipt = self.world.last_admitted_receipt
        ok = bool(
            receipt is not None
            and receipt.success
            and receipt.final_stable
            and receipt.final_owner is None
            and self.world.state().objects["cup_1"].zone == "right_table"
        )
        return VerifyResult(
            ok=ok,
            expected={"cup_1": "right_table", "contact_receipt": "stable_released"},
            observed={
                "cup_1": self.world.state().objects["cup_1"].zone,
                "receipt_hash": receipt.content_hash if receipt else None,
                "final_stable": receipt.final_stable if receipt else False,
            },
            mismatch=() if ok else ("contact-handoff-verification",),
        )


def build_intel_contact_handoff_engine(
    config: ContactHandoffConfig | None = None,
    *,
    recorder: Any | None = None,
) -> OmniQ:
    """Construct the separate governed contact-handoff path."""
    world = IntelContactHandoffWorld(config)
    return OmniQ(
        world=world,
        observer=IntelContactHandoffObserver(),
        planner=IntelContactHandoffPlanner(world),
        manipulator=IntelContactHandoffManipulator(world),
        verifier=IntelContactHandoffVerifier(world),
        device=intel_devices(),
        recorder=recorder or FakeRecorder(),
    )


def run_contact_handoff(
    config: ContactHandoffConfig | None = None,
    *,
    receipt_path: str | Path | None = None,
) -> ContactHandoffReceipt:
    """Execute one governed physical proposal and optionally retain its receipt."""
    engine = build_intel_contact_handoff_engine(config)
    engine.run("transfer cup_1 from left arm to right arm")
    receipt = engine.world.last_admitted_receipt
    if receipt is None:
        pending = engine.world.pending_contact_receipt
        if pending is None:
            raise ContactHandoffRejected("controller returned no handoff receipt")
        receipt = pending
    if receipt_path is not None:
        write_contact_handoff_receipt(receipt, receipt_path)
    return receipt


def run_randomized_contact_handoff_report(
    root: str | Path,
    *,
    trials: int = 20,
    seed: int = 700,
) -> dict[str, Any]:
    """Retain every bounded randomized trial; report outcomes without promotion."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    outcomes = {"success": 0, "drop": 0, "timeout": 0, "collision": 0, "ambiguity": 0, "other_failure": 0}
    for index in range(trials):
        trial_seed = seed + index
        receipt = run_contact_handoff(
            ContactHandoffConfig(seed=trial_seed, randomized=True),
            receipt_path=root_path / f"trial-{index:02d}-seed-{trial_seed}.json",
        )
        if receipt.success:
            category = "success"
        else:
            reason = (receipt.failure_reason or "").lower()
            category = next((name for name in ("drop", "timeout", "collision", "ambiguity") if name in reason), "other_failure")
        outcomes[category] += 1
        receipts.append({
            "trial": index,
            "seed": trial_seed,
            "success": receipt.success,
            "failure_reason": receipt.failure_reason,
            "receipt": f"trial-{index:02d}-seed-{trial_seed}.json",
        })
    report = {
        "schema_version": CONTACT_HANDOFF_SCHEMA_VERSION,
        "mode": CONTACT_HANDOFF_MODE,
        "kind": "exploratory robustness report; not a promotion claim",
        "trials": trials,
        "seed_start": seed,
        "outcomes": outcomes,
        "receipts": receipts,
    }
    report_path = root_path / "report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    return report
