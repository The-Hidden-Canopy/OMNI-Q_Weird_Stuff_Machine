"""Run the bounded shared-world joint-motion demo (not object cooperation).

    python -m omni_q.demo_intel_fleet
"""
from __future__ import annotations

import argparse
import json

from .contracts import PlanGraph, Step, WorldState
from .fleet import ManipulationFleet, ManipulatorSpec
from .fleet_runtime import FleetExecutor
from .intel_fleet import IntelFleetBackend, IntelFleetWorld
from .intel_sim import HOME
from .scheduler import Schedule, ScheduledStep, Wave


def arm_ids(count: int) -> tuple[str, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 2 or count % 2:
        raise ValueError("arm count must be an even integer of at least two")
    return tuple(f"arm_{index}" for index in range(1, count + 1))


def demo_plan(simulation, active_arm_ids=None):
    active = tuple(active_arm_ids or tuple(simulation.bindings)[:4])
    if len(active) < 2 or len(active) % 2 or len(set(active)) != len(active):
        raise ValueError("active arms must be unique, known bimanual pairs")
    if any(arm not in simulation.bindings for arm in active):
        raise ValueError("active arms must belong to this world")
    teams = {f"team_{index // 2 + 1:02d}": (active[index], active[index + 1])
             for index in range(0, len(active), 2)}
    steps, scheduled = [], []
    for name, arms in teams.items():
        targets = {}
        for arm in arms:
            pose = list(HOME)
            pose[0] = .25
            targets[arm] = dict(zip(simulation.bindings[arm].joint_names, pose))
        steps.append(Step(name, "manipulate", "MOVE_JOINTS", {"targets": targets}))
        scheduled.append(ScheduledStep(name, arms[0], f"station.{name}", participants=arms))
    regions = frozenset(f"station.{name}" for name in teams)
    fleet = ManipulationFleet.from_specs(tuple(
        ManipulatorSpec(arm, capabilities=frozenset({"MOVE_JOINTS"}),
                        workspace_regions=regions)
        for arm in simulation.bindings
    ), preferred_pairs=tuple(teams.values()))
    schedule = Schedule(
        waves=[Wave(0, scheduled)], barriers=[], arm_timeline={}, metrics={},
        assignment={s.step_id: s.arm for s in scheduled},
        regions={s.step_id: s.region_id for s in scheduled}, resource_assignment=teams,
    )
    return PlanGraph("shared-loop actuator verification", steps), schedule, fleet


def main(argv=None):
    parser = argparse.ArgumentParser(description="run one shared-world bimanual fleet wave")
    parser.add_argument("--arms", type=int, default=50,
                        help="even number of arms to activate; defaults to 50")
    parser.add_argument("--full-receipts", action="store_true",
                        help="include every command target and receipt detail")
    options = parser.parse_args(argv)
    ids = arm_ids(options.arms)
    simulation = IntelFleetWorld(ids)
    backend = IntelFleetBackend(simulation)
    graph, schedule, fleet = demo_plan(simulation, ids)
    result = FleetExecutor(backend).execute(graph, schedule, WorldState(0, {}), fleet)
    report = {
        "ok": result.ok, "mode": "shared-world-joint-motion",
        "physical_object_verified": False,
        "model_count": 1, "physical_data_count": 1,
        "configured_arms": len(simulation.bindings),
        "active_arms": len(ids), "bimanual_teams": len(ids) // 2,
        "physics_steps": len(backend.trace),
        "sim_seconds": float(simulation.data.time),
        "simultaneous_motion_ticks": sum(
            all(abs(q[0]) > .05 for q in sample["arms"].values())
            for sample in backend.trace),
        "wave_count": len(result.waves),
        "command_count": sum(len(receipt.commands) for receipt in result.waves),
        "successful_command_count": sum(
            result.ok for receipt in result.waves for result in receipt.results),
        "remaining_leases": len(result.fleet.leases),
        "remaining_reservations": len(result.fleet.reservations),
    }
    if options.full_receipts:
        report["waves"] = [receipt.as_dict() for receipt in result.waves]
    print(json.dumps(report, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
