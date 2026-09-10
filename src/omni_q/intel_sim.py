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
        # captured before super() mutates WorldState, so a failed real grasp
        # can be reverted below rather than leaving a scripted "success" that
        # the physics never actually delivered.
        prev_zone = self._objects[obj].zone if obj in self._objects else None

        result = super().apply_transition(request)
        arm_offset = 6 if request.actor and "right" in request.actor else 0
        grasp_info: dict[str, Any] = {}

        if op == "PICK" and obj in self._object_joints:
            grasp_info = self._do_pick(arm_offset, obj)
            if not grasp_info["held"]:
                self._ownership[obj] = None  # revert the scripted grasp claim
                result = replace(result, ok=False)
        elif op in {"MOVE", "PLACE"} and obj in self._object_joints:
            grasp_info = self._do_place(arm_offset, obj, request.args.get("to"))
            if not grasp_info["placed"]:
                if prev_zone is not None:
                    self._objects[obj] = replace(self._objects[obj], zone=prev_zone)
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
