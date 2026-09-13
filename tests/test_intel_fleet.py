import copy
import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("mujoco")
import numpy as np

from omni_q.contracts import WorldState
from omni_q.fleet import FleetError
from omni_q.fleet_runtime import FleetBackendFault, FleetCommand
from omni_q.intel_fleet import IntelFleetBackend, IntelFleetWorld, fleet_xml
from omni_q.intel_sim import HOME


@pytest.fixture
def simulation():
    return IntelFleetWorld()


def commands(sim):
    result = []
    for name, participants in (("north", ("arm_1", "arm_2")),
                               ("south", ("arm_3", "arm_4"))):
        targets = {}
        for arm in participants:
            target = list(HOME)
            target[0] = .25
            targets[arm] = dict(zip(sim.bindings[arm].joint_names, target))
        result.append(FleetCommand(name, "MOVE_JOINTS", {"targets": targets},
                                   participants, f"table.{name}", participants[0]))
    return tuple(result)


def test_two_teams_move_in_same_physics_ticks(simulation):
    backend = IntelFleetBackend(simulation)
    state = WorldState(0, {})
    before = copy.deepcopy(state.as_dict())
    results = backend.execute_wave(commands(simulation), state)
    assert all(r.ok for r in results), results
    assert state.as_dict() == before
    assert all(not r.detail["physical_object_verified"] for r in results)
    assert all(r.detail["model_sha256"] == simulation.model_sha256 for r in results)
    assert results[0].detail["physics_steps"] == results[1].detail["physics_steps"]
    assert simulation.data.time == pytest.approx(len(backend.trace) * simulation.model.opt.timestep)
    assert all(set(s["arms"]) == {"arm_1", "arm_2", "arm_3", "arm_4"} for s in backend.trace)
    # Measured simultaneous movement, not merely simultaneous ctrl assignments.
    assert any(all(abs(s["arms"][a][0]) > .05 for a in s["arms"])
               for s in backend.trace)
    all_indices = [i for b in simulation.bindings.values() for i in b.qpos_indices]
    assert len(all_indices) == len(set(all_indices)) == 30


def test_fifty_arms_move_as_twenty_five_teams_in_one_world():
    from omni_q.demo_intel_fleet import arm_ids, demo_plan
    from omni_q.fleet_runtime import FleetExecutor
    ids = arm_ids(50)
    sim = IntelFleetWorld(ids)
    backend = IntelFleetBackend(sim, max_steps=400)
    graph, schedule, fleet = demo_plan(sim, ids)
    result = FleetExecutor(backend).execute(graph, schedule, WorldState(0, {}), fleet)
    assert result.ok
    assert sim.model.nq == sim.model.nv == sim.model.nu == 300
    assert len(result.waves) == 1
    assert len(result.waves[0].commands) == 25
    assert {arm for c in result.waves[0].commands for arm in c.participants} == set(ids)
    assert all(set(sample["arms"]) == set(ids) for sample in backend.trace)
    assert all(receipt.ok for receipt in result.waves[0].results)
    assert result.fleet.leases == result.fleet.reservations == ()


def test_backend_provisions_new_scheduled_arm_before_first_step():
    sim = IntelFleetWorld()
    backend = IntelFleetBackend(sim)
    backend.prepare_run(("arm_1", "arm_6"), WorldState(0, {}))
    assert sim.arm_ids == ("arm_1", "arm_2", "arm_3", "arm_4", "arm_5", "arm_6")
    target = list(HOME)
    target[0] = .25
    command = FleetCommand(
        "new-arm", "MOVE_JOINTS",
        {"targets": {"arm_6": dict(zip(sim.bindings["arm_6"].joint_names, target))}},
        ("arm_6",), "station.new", "arm_6")
    result = backend.execute_wave((command,), WorldState(0, {}))
    assert result[0].ok
    with pytest.raises(FleetError, match="after physical execution"):
        sim.ensure_arms(("arm_7",))


def test_executor_auto_provisions_online_arm_from_schedule():
    from omni_q.contracts import PlanGraph, Step
    from omni_q.fleet import ManipulationFleet, ManipulatorSpec
    from omni_q.fleet_runtime import FleetExecutor
    from omni_q.scheduler import Schedule, ScheduledStep, Wave
    sim = IntelFleetWorld()
    target = list(HOME)
    target[0] = .25
    names = tuple(f"arm_6_{suffix}" for suffix in
                  ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"))
    step = Step("sixth", "manipulate", "MOVE_JOINTS",
                {"targets": {"arm_6": dict(zip(names, target))}})
    schedule = Schedule(
        waves=[Wave(0, [ScheduledStep("sixth", "arm_6", "station.six", participants=("arm_6",))])],
        barriers=[], arm_timeline={}, metrics={}, assignment={"sixth": "arm_6"},
        regions={"sixth": "station.six"}, resource_assignment={"sixth": ("arm_6",)},
    )
    fleet = ManipulationFleet.from_specs(tuple(
        ManipulatorSpec(f"arm_{index}", capabilities=frozenset({"MOVE_JOINTS"}),
                        workspace_regions=frozenset({"station.six"}))
        for index in range(1, 7)))
    result = FleetExecutor(IntelFleetBackend(sim)).execute(
        PlanGraph("accept online new arm", [step]), schedule, WorldState(0, {}), fleet)
    assert result.ok
    assert "arm_6" in sim.bindings
    assert result.waves[0].results[0].ok


def test_named_bindings_survive_non_numeric_arm_order():
    sim = IntelFleetWorld(("spare", "north_b", "north_a"))
    for name, binding in sim.bindings.items():
        assert all(j.startswith(name + "_") for j in binding.joint_names)
        assert sim.model.actuator(binding.gripper_actuator).name == f"{name}_Jaw"
        for joint, qpos in zip(binding.joint_names, binding.qpos_indices):
            assert sim.model.jnt_qposadr[sim.model.joint(joint).id] == qpos


@pytest.mark.parametrize("kind", ["org", "nan", "range", "missing", "unknown",
                                   "duplicate", "unsupported", "primary"])
def test_invalid_second_command_cannot_move_first_arm(simulation, kind):
    request = list(commands(simulation))
    state = WorldState(0, {}, org_id="other" if kind == "org" else "local-demo")
    c = request[1]
    args = copy.deepcopy(c.args)
    participants, op, primary = c.participants, c.op, c.primary
    first = simulation.bindings["arm_3"].joint_names[0]
    if kind == "nan": args["targets"]["arm_3"][first] = float("nan")
    if kind == "range": args["targets"]["arm_3"][first] = 999
    if kind == "missing": del args["targets"]["arm_3"][first]
    if kind == "unknown": participants = ("missing",)
    if kind == "duplicate":
        participants = request[0].participants
        args = copy.deepcopy(request[0].args)
        primary = participants[0]
    if kind == "unsupported": op = "CO_ROTATE"
    if kind == "primary": primary = "arm_5"
    request[1] = FleetCommand(c.step_id, op, args, participants, c.region_id, primary)
    initial_ctrl = simulation.data.ctrl.copy()
    initial_qpos = simulation.data.qpos.copy()
    with pytest.raises(FleetError):
        IntelFleetBackend(simulation).execute_wave(tuple(request), state)
    np.testing.assert_array_equal(simulation.data.ctrl, initial_ctrl)
    np.testing.assert_array_equal(simulation.data.qpos, initial_qpos)
    assert simulation.data.time == 0


def test_timeout_is_failed_evidence(simulation):
    result = IntelFleetBackend(simulation, max_steps=1).execute_wave(
        commands(simulation), WorldState(0, {}))
    assert all(not r.ok and r.detail["reason"] == "timeout" for r in result)


def test_fault_observation_waits_for_loop_boundary_and_latches(simulation, monkeypatch):
    backend = IntelFleetBackend(simulation)
    original_step = simulation.mj.mj_step
    calls = []

    def step(model, data):
        assert model is simulation.model and data is simulation.data
        original_step(model, data)
        calls.append(float(data.time))
        if len(calls) == 20:
            ctrl = data.ctrl.copy()
            backend.report_fault(("arm_1",), "test-injected servo fault")
            np.testing.assert_array_equal(data.ctrl, ctrl)

    monkeypatch.setattr(simulation.mj, "mj_step", step)
    with pytest.raises(FleetBackendFault, match="test-injected"):
        backend.execute_wave(commands(simulation), WorldState(0, {}))
    assert len(calls) == 20
    with pytest.raises(FleetBackendFault):
        backend.execute_wave(commands(simulation), WorldState(0, {}))
    assert len(calls) == 20
    for b in simulation.bindings.values():
        np.testing.assert_array_equal(simulation.data.ctrl[list(b.actuator_indices)],
                                      simulation.data.qpos[list(b.qpos_indices)])
    assert len(backend.trace) == 20  # A refused restart must not erase fault evidence.


def test_scene_keeps_collision_defaults_and_only_pinned_exclusions():
    root = ET.fromstring(fleet_xml(("arm_1", "arm_2")))
    assert len(root.findall("./contact/exclude")) == 2
    for geom in root.findall("./worldbody/body//geom"):
        assert "contype" not in geom.attrib and "conaffinity" not in geom.attrib
    assert not root.findall("equality")


def test_executor_releases_real_wave_leases(simulation):
    from omni_q.demo_intel_fleet import demo_plan
    from omni_q.fleet_runtime import FleetExecutor
    graph, schedule, fleet = demo_plan(simulation)
    result = FleetExecutor(IntelFleetBackend(simulation)).execute(
        graph, schedule, WorldState(0, {}), fleet)
    assert result.ok
    assert len(result.waves) == 1
    assert len(result.waves[0].commands) == 2
    assert result.fleet.leases == result.fleet.reservations == ()


def test_real_mid_wave_fault_returns_reallocation_without_continuing(simulation, monkeypatch):
    from omni_q.demo_intel_fleet import demo_plan
    from omni_q.fleet_runtime import FleetExecutor
    from omni_q.scheduler import Wave
    graph, schedule, fleet = demo_plan(simulation)
    schedule.waves.append(Wave(1, schedule.waves[0].steps))
    backend = IntelFleetBackend(simulation)
    original_step = simulation.mj.mj_step
    ticks = []

    def step(model, data):
        original_step(model, data)
        ticks.append(data.time)
        if len(ticks) == 20:
            backend.report_fault(("arm_1",), "test-injected servo fault")

    monkeypatch.setattr(simulation.mj, "mj_step", step)
    result = FleetExecutor(backend).execute(graph, schedule, WorldState(0, {}), fleet)
    assert not result.ok and result.failed_wave == 0
    assert len(ticks) == 20
    assert "arm_1" not in result.fleet.online_ids
    assert result.fleet.leases == result.fleet.reservations == ()
    assert result.detail["requires_replan"]
    assert set(result.reallocation.replacements[0][2]) == {"arm_2", "arm_5"}
    assert fleet.online_ids == tuple(simulation.bindings)
