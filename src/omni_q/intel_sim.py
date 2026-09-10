"""Intel Online dual-SO-101 MuJoCo adapters.

This module uses the pinned SO-ARM100 Menagerie MJCF as the documented
six-joint mechanical proxy for SO-101.  It creates one real MuJoCo scene with
two independently actuated arms and tableware.  Tableware state transitions
remain explicitly scripted in the table adapter and are labelled
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
    ET.SubElement(body, "geom", geom)
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

    worldbody.extend([
        _body("plate_1", "-.13 -.08 .018", {"type": "cylinder", "size": ".095 .007", "rgba": ".93 .93 .91 1", "mass": ".18"}),
        _body("cup_1", ".16 -.06 .055", {"type": "cylinder", "size": ".032 .055", "rgba": ".22 .58 .78 1", "mass": ".12"}),
        _body("fork_1", "-.21 -.20 .011", {"type": "box", "size": ".012 .075 .004", "rgba": ".72 .73 .75 1", "mass": ".04"}),
        _body("spoon_1", ".22 -.20 .011", {"type": "box", "size": ".013 .07 .004", "rgba": ".72 .73 .75 1", "mass": ".04"}),
        _body("napkin_1", "-.22 .02 .006", {"type": "box", "size": ".07 .05 .003", "rgba": ".90 .40 .38 1", "mass": ".02"}),
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
        super().__init__([
            Detection("plate_1", "plate", "staging", "center"),
            Detection("cup_1", "cup", "staging", "upper_right"),
            Detection("fork_1", "fork", "staging", "left"),
            Detection("spoon_1", "spoon", "staging", "right"),
            Detection("napkin_1", "napkin", "staging", "lower_left"),
        ])
        mujoco = _mujoco()
        self.model = load_dual_so101_model()
        self.data = mujoco.MjData(self.model)
        self._mujoco = mujoco
        self._controller_steps = 0
        for index, value in enumerate(HOME * 2):
            self.data.ctrl[index] = value
        mujoco.mj_forward(self.model, self.data)

    def apply_transition(self, request: TransitionRequest) -> TransitionResult:
        result = super().apply_transition(request)
        arm_offset = 6 if request.actor and "right" in request.actor else 0
        target = list(HOME)
        if request.op in {"PICK", "MOVE", "PLACE"}:
            target[2] = 1.20
            target[3] = 0.85
        if request.op == "HANDOFF":
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
        }


class IntelTablePlanner(RulePlanner):
    """Deterministic role assignment for the first dual-arm table-setting slice."""

    _RIGHT_OBJECTS = {"cup_1", "spoon_1"}

    def plan(self, goal, world):
        graph = super().plan(goal, world)
        for step in graph.steps:
            object_id = step.args.get("object")
            if object_id:
                step.arm = "right" if object_id in self._RIGHT_OBJECTS else "left"
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
