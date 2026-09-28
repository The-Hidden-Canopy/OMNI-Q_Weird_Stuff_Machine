"""Shared-world fleet actuator substrate, not an object-manipulation policy.

MOVE_JOINTS is a simulation-only primitive. CO_ROTATE/PICK/PLACE deliberately
remain unsupported until contact-based object verification is implemented.
Only ctrl is written during execution; WorldState admission stays with OMNI-Q.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
import xml.etree.ElementTree as ET

from .contracts import ManipResult, WorldState
from .fleet import FleetError
from .fleet_runtime import FleetBackendFault, FleetCommand
from .intel_sim import ARM_ASSETS, ARM_XML, HOME, _mujoco, _prefixed


@dataclass(frozen=True)
class ArmBinding:
    arm_id: str
    qpos_indices: tuple[int, ...]
    actuator_indices: tuple[int, ...]
    joint_names: tuple[str, ...]
    gripper_actuator: int
    dof_indices: tuple[int, ...]


def fleet_xml(arm_ids: tuple[str, ...]) -> str:
    """Clone pinned articulated arms, preserving their collision defaults."""
    if not arm_ids or len(set(arm_ids)) != len(arm_ids):
        raise ValueError("arm ids must be nonempty and unique")
    if any(not arm or not arm.replace('_', '').isalnum() for arm in arm_ids):
        raise ValueError("arm ids must contain only letters, digits and underscores")
    source = ET.parse(ARM_XML).getroot()
    root = ET.Element("mujoco", model="omni_q_shared_fleet_joint_demo")
    for tag in ("compiler", "option", "asset", "default"):
        root.append(copy.deepcopy(source.find(tag)))
    root.find("option").set("timestep", "0.002")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 .05",
                  pos="0 0 -.02", rgba=".2 .2 .2 1")
    actuators = ET.SubElement(root, "actuator")
    contacts = ET.SubElement(root, "contact")
    columns = max(1, math.ceil(math.sqrt(len(arm_ids))))
    for index, arm_id in enumerate(arm_ids):
        body = _prefixed(source.find("./worldbody/body[@name='Base']"), arm_id)
        # Deliberately separated stations for an actuator-concurrency test.
        # This layout does not claim that pairs can grasp a common object.
        body.set("pos", f"{(index % columns) * .8} {(index // columns) * .9} 0")
        world.append(body)
        for actuator in source.find("actuator"):
            node = copy.deepcopy(actuator)
            node.set("name", f"{arm_id}_{actuator.get('name')}")
            node.set("joint", f"{arm_id}_{actuator.get('joint')}")
            actuators.append(node)
        # Preserve only the pinned model's adjacent-body exclusions.
        for exclusion in source.findall("./contact/exclude"):
            node = copy.deepcopy(exclusion)
            for attr in ("body1", "body2"):
                node.set(attr, f"{arm_id}_{node.get(attr)}")
            contacts.append(node)
    return ET.tostring(root, encoding="unicode")


class IntelFleetWorld:
    """Exactly one MjModel and one physical MjData for all fleet members."""

    def __init__(self, arm_ids=("arm_1", "arm_2", "arm_3", "arm_4", "arm_5"),
                 *, org_id="local-demo"):
        if not isinstance(org_id, str) or not org_id.strip():
            raise ValueError("org_id must be nonempty")
        self.org_id = org_id
        self.mj = _mujoco()
        self.arm_ids: tuple[str, ...] = ()
        self.model_xml = ""
        self.model_sha256 = ""
        self.model = None
        self.data = None
        self.bindings: dict[str, ArmBinding] = {}
        self._build(tuple(arm_ids))

    def _build(self, arm_ids: tuple[str, ...]) -> None:
        """Compile a model before execution; never mutate a live model."""
        self.model_xml = fleet_xml(arm_ids)
        assets = {f"assets/{p.name}": p.read_bytes() for p in ARM_ASSETS.glob("*.stl")}
        digest = hashlib.sha256(self.model_xml.encode("utf-8"))
        for name, payload in sorted(assets.items()):
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(payload).digest())
        self.model_sha256 = digest.hexdigest()
        self.model = self.mj.MjModel.from_xml_string(self.model_xml, assets=assets)
        self.data = self.mj.MjData(self.model)
        self.bindings = {}
        source = ET.parse(ARM_XML).getroot()
        joint_suffixes = tuple(a.get("joint") for a in source.find("actuator"))
        for arm_id in arm_ids:
            names = tuple(f"{arm_id}_{suffix}" for suffix in joint_suffixes)
            joint_ids = tuple(self.model.joint(name).id for name in names)
            actuator_ids = tuple(self.model.actuator(name).id for name in names)
            self.bindings[arm_id] = ArmBinding(
                arm_id, tuple(int(self.model.jnt_qposadr[j]) for j in joint_ids),
                actuator_ids, names, self.model.actuator(f"{arm_id}_Jaw").id,
                tuple(int(self.model.jnt_dofadr[j]) for j in joint_ids),
            )
        # Initial condition only. No qpos/qvel writes in the execution loop.
        for binding in self.bindings.values():
            self.data.qpos[list(binding.qpos_indices)] = HOME
            self.data.ctrl[list(binding.actuator_indices)] = HOME
        self.mj.mj_forward(self.model, self.data)
        self.arm_ids = arm_ids

    def ensure_arms(self, arm_ids: tuple[str, ...]) -> None:
        """Provision newly scheduled arm IDs at a run boundary.

        MuJoCo models are immutable. Before the first physical step, rebuilding
        with a superset is safe and the resulting execution still uses only one
        model/data pair. After any step it fails closed for strict replanning.
        """
        requested = tuple(dict.fromkeys(arm_ids))
        additions = tuple(arm for arm in requested if arm not in self.bindings)
        if not additions:
            return
        if self.data is not None and self.data.time != 0:
            raise FleetError("cannot provision an arm after physical execution began")
        self._build(self.arm_ids + additions)


class IntelFleetBackend:
    """Bounded shared-loop joint servo with measured receipts and fault latch.

    It does not publish governed events or mutate authoritative WorldState.
    Callers must acquire fleet authority first (FleetExecutor) and retain the
    returned evidence. A latched fault requires explicit recovery outside this
    substrate; merely calling execute_wave again cannot restart motion.
    """

    def __init__(self, simulation: IntelFleetWorld, *, max_steps=1000,
                 position_tolerance=.06, velocity_tolerance=.12, settle_steps=10):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if isinstance(settle_steps, bool) or not isinstance(settle_steps, int) or settle_steps <= 0:
            raise ValueError("settle_steps must be a positive integer")
        if any(not math.isfinite(x) or x <= 0
               for x in (position_tolerance, velocity_tolerance)):
            raise ValueError("tolerances must be finite and positive")
        self.simulation = simulation
        self.max_steps = max_steps
        self.position_tolerance = position_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.settle_steps = settle_steps
        self.trace: list[dict] = []
        self._pending_fault: FleetBackendFault | None = None
        self._latched_fault: FleetBackendFault | None = None

    def prepare_run(self, resource_ids: tuple[str, ...], world: WorldState) -> None:
        """Accept newly scheduled resources before the run's first timestep."""
        if world.org_id != self.simulation.org_id:
            raise FleetError("cross-org fleet execution denied")
        self.simulation.ensure_arms(resource_ids)

    def report_fault(self, resource_ids: tuple[str, ...], reason: str) -> None:
        """Queue an observation; never write ctrl or step physics here."""
        if not resource_ids or len(set(resource_ids)) != len(resource_ids):
            raise FleetError("fault resources must be nonempty and unique")
        if any(r not in self.simulation.bindings for r in resource_ids):
            raise FleetError("unknown fault resource")
        if not isinstance(reason, str) or not reason.strip():
            raise FleetError("fault reason must be nonempty")
        if self._pending_fault is not None or self._latched_fault is not None:
            raise FleetError("fault already pending or latched")
        self._pending_fault = FleetBackendFault(resource_ids, reason)

    def _check_fault(self):
        if self._pending_fault is not None:
            self._latched_fault = self._pending_fault
            self._pending_fault = None
        if self._latched_fault is not None:
            # Hold at measured positions without advancing physics or assigning
            # replacement arms. This is a simulation hold, not hardware e-stop.
            for b in self.simulation.bindings.values():
                self.simulation.data.ctrl[list(b.actuator_indices)] = (
                    self.simulation.data.qpos[list(b.qpos_indices)])
            raise self._latched_fault

    def _validate(self, commands, world):
        if world.org_id != self.simulation.org_id:
            raise FleetError("cross-org fleet execution denied")
        ids, assigned, targets = set(), set(), {}
        for command in commands:
            if not command.step_id or command.step_id in ids:
                raise FleetError("step ids must be nonempty and unique")
            ids.add(command.step_id)
            if command.op != "MOVE_JOINTS":
                raise FleetError(f"unsupported physical operation: {command.op}")
            if not command.participants or len(set(command.participants)) != len(command.participants):
                raise FleetError("participants must be nonempty and unique")
            if command.primary is not None and command.primary not in command.participants:
                raise FleetError("primary must be a participant")
            requested = command.args.get("targets")
            if not isinstance(requested, dict) or set(requested) != set(command.participants):
                raise FleetError("targets must match participants exactly")
            for arm in command.participants:
                if arm in assigned or arm not in self.simulation.bindings:
                    raise FleetError("duplicate or unknown arm assignment")
                assigned.add(arm)
                b = self.simulation.bindings[arm]
                values = requested[arm]
                if not isinstance(values, dict) or set(values) != set(b.joint_names):
                    raise FleetError("joint targets must match named binding exactly")
                ordered = []
                for name, actuator in zip(b.joint_names, b.actuator_indices):
                    value = values[name]
                    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                        raise FleetError("joint targets must be finite numbers")
                    lo, hi = self.simulation.model.actuator_ctrlrange[actuator]
                    if not lo <= value <= hi:
                        raise FleetError(f"joint target out of range: {name}")
                    ordered.append(float(value))
                targets[arm] = tuple(ordered)
        return targets

    def execute_wave(self, commands: tuple[FleetCommand, ...], world: WorldState):
        import numpy as np

        targets = self._validate(commands, world)  # Entire wave before writes.
        self._check_fault()
        self.trace = []
        if not commands:
            return ()
        sim = self.simulation
        start = float(sim.data.time)
        consecutive = {c.step_id: 0 for c in commands}
        errors, velocities = {}, {}
        initial_warnings = sim.data.warning.number.copy()
        for tick in range(self.max_steps):
            self._check_fault()
            # Every arm gets its target BEFORE this single shared physics step.
            for arm, target in targets.items():
                sim.data.ctrl[list(sim.bindings[arm].actuator_indices)] = target
            sim.mj.mj_step(sim.model, sim.data)
            if (not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all()
                    or np.any(sim.data.warning.number > initial_warnings)):
                self.report_fault(tuple(targets), "MuJoCo numerical fault")
                self._check_fault()
            sample = {}
            for arm, target in targets.items():
                b = sim.bindings[arm]
                actual = sim.data.qpos[list(b.qpos_indices)]
                errors[arm] = float(np.max(np.abs(actual - target)))
                velocities[arm] = float(np.max(np.abs(sim.data.qvel[list(b.dof_indices)])))
                sample[arm] = actual.tolist()
            self.trace.append({"tick": tick + 1, "sim_time": float(sim.data.time),
                               "arms": sample, "contact_count": int(sim.data.ncon)})
            for c in commands:
                reached = all(errors[a] <= self.position_tolerance and
                              velocities[a] <= self.velocity_tolerance for a in c.participants)
                consecutive[c.step_id] = consecutive[c.step_id] + 1 if reached else 0
            if all(count >= self.settle_steps for count in consecutive.values()):
                break
        return tuple(ManipResult(c.step_id, consecutive[c.step_id] >= self.settle_steps, {
            "mode": "shared-world-joint-motion", "physical_object_verified": False,
            "model_sha256": sim.model_sha256, "mujoco_version": sim.mj.__version__,
            "participants": list(c.participants), "physics_steps": len(self.trace),
            "sim_start": start, "sim_end": float(sim.data.time),
            "max_joint_error_rad": {a: errors[a] for a in c.participants},
            "max_joint_velocity_rad_s": {a: velocities[a] for a in c.participants},
            "reason": "reached" if consecutive[c.step_id] >= self.settle_steps else "timeout",
        }) for c in commands)
