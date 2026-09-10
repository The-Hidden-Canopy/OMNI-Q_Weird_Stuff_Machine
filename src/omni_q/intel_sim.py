"""Intel Online dual-SO-101 MuJoCo adapters.

This module uses the pinned SO-ARM100 Menagerie MJCF as the documented
six-joint mechanical proxy for SO-101.  It creates one real MuJoCo scene with
two independently actuated arms, a table-setting object pack (plate, cup,
fork, spoon, napkin -- distinct masses/friction/collision per OQ-007), and a
passive slide-jointed drawer holding the cutlery, matching the brief's
"open the top drawer, retrieve spoons and forks" scenario.  Tableware state
transitions remain explicitly scripted in the table adapter and are labelled
``simulation-scripted-manipulation``.  The separate OQ-010/OQ-011 contact
adapter uses only MuJoCo contact dynamics for a bounded ``cup_1`` handoff. It
is a SO-ARM100 mechanical-proxy simulation, not evidence of perception, VLA
control, hardware, complete table setting, or concurrent execution.
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

# OQ-010 primitive targets. Coarse proxy values (not IK), consistent with this
# module's existing "simulation-scripted-manipulation" framing -- distinct
# enough per op to make PICK/PLACE/MOVE/OPEN/CLOSE/ROTATE/PRESENT observably
# different controller behaviour, not evidence of contact-rich grasping.
GRIPPER_OPEN = 1.5     # Jaw joint, rad -- near the SO-101 open end of its range
GRIPPER_CLOSED = 0.0
DRAWER_OPEN = 0.12     # drawer_slide qpos, m -- matches its MJCF range max
DRAWER_CLOSED = 0.0


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
    for arm, pos in (("left", "-.26 .20 .0"), ("right", ".26 .20 .0")):
        arm_body = _prefixed(base, arm)
        arm_body.set("pos", pos)
        worldbody.append(arm_body)

    worldbody.append(_drawer("0 -.40 .01"))
    worldbody.extend([
        _body("plate_1", "-.13 -.08 .018", {
            "type": "cylinder", "size": ".095 .007", "rgba": ".93 .93 .91 1",
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
        for index, value in enumerate(HOME * 2):
            self.data.ctrl[index] = value
        mujoco.mj_forward(self.model, self.data)

    def apply_transition(self, request: TransitionRequest) -> TransitionResult:
        result = super().apply_transition(request)
        arm_offset = 6 if request.actor and "right" in request.actor else 0
        op = request.op
        obj = request.args.get("object")
        target = list(HOME)

        # OQ-010: each primitive gets a distinct controller target instead of
        # PICK/MOVE/PLACE sharing one pose and everything else silently
        # falling through to HOME. The drawer has no actuator (see
        # dual_so101_xml._drawer) -- OPEN/CLOSE on it is a kinematic qpos
        # override, the same "explicitly scripted" honesty this adapter
        # already applies to tableware placement.
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
            "sim_time": round(float(self.data.time), 4),
            "controller_steps": self._controller_steps,
            "simulation_mode": self.mode,
            "physics_note": "arm controller stepped; table-object placement is an explicit scripted transition",
        }
        return replace(result, detail=detail)

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
