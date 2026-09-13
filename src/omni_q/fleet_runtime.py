"""Lease-aware execution seam for OMNI-Q manipulation fleets.

Bridges the scheduler and resource model to a backend executing an entire
wave. Physical concurrency belongs to the backend: for MuJoCo, one MjModel,
one MjData and one control loop step all active manipulators together.
Independent MuJoCo worlds in threads do not establish fleet concurrency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .contracts import ManipResult, PlanGraph, Step, WorldState
from .fleet import (
    CapabilityLease, FleetError, FleetFault, FleetReallocation,
    ManipulationFleet, ReallocationRequest, WorkspaceReservation,
)
from .scheduler import Schedule, ScheduledStep, Wave


@dataclass(frozen=True)
class FleetCommand:
    """One governed manipulation command inside a scheduled wave."""
    step_id: str
    op: str
    args: dict[str, Any]
    participants: tuple[str, ...]
    region_id: str | None
    primary: str | None = None


@dataclass(frozen=True)
class FleetWaveReceipt:
    wave_index: int
    commands: tuple[FleetCommand, ...]
    results: tuple[ManipResult, ...]
    leases: tuple[str, ...]
    ok: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "wave_index": self.wave_index,
            "commands": [
                {"step_id": c.step_id, "op": c.op, "args": dict(c.args),
                 "participants": list(c.participants), "region_id": c.region_id,
                 "primary": c.primary} for c in self.commands
            ],
            "results": [
                {"step_id": r.step_id, "ok": r.ok, "detail": dict(r.detail)}
                for r in self.results
            ],
            "leases": list(self.leases), "ok": self.ok,
        }


@dataclass(frozen=True)
class FleetRunResult:
    fleet: ManipulationFleet
    waves: tuple[FleetWaveReceipt, ...]
    ok: bool
    reallocation: FleetReallocation | None = None
    failed_wave: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)


class FleetBackendFault(RuntimeError):
    """Raised when the backend observes a real resource fault mid-wave."""
    def __init__(self, resource_ids: tuple[str, ...], reason: str) -> None:
        super().__init__(reason)
        if not resource_ids:
            raise ValueError("FleetBackendFault requires at least one resource id")
        self.resource_ids = tuple(resource_ids)
        self.reason = reason


class FleetWaveBackend(Protocol):
    """Execute all commands in one shared control loop until terminal."""
    def execute_wave(
        self, commands: tuple[FleetCommand, ...], world: WorldState,
    ) -> tuple[ManipResult, ...]: ...


class FleetExecutor:
    """Acquire authority, execute scheduled waves, and fail closed."""
    def __init__(self, backend: FleetWaveBackend, *, wave_window_ms: int = 3_000):
        if wave_window_ms <= 0:
            raise ValueError("wave_window_ms must be positive")
        self.backend = backend
        self.wave_window_ms = wave_window_ms

    def execute(
        self, graph: PlanGraph, schedule: Schedule, world: WorldState,
        fleet: ManipulationFleet, *, start_ms: int = 0, run_id: str = "fleet-run",
    ) -> FleetRunResult:
        if start_ms < 0:
            raise ValueError("start_ms must be non-negative")
        if not run_id:
            raise ValueError("run_id must be non-empty")
        graph_steps = {step.id: step for step in graph.steps}
        # A physical backend may provision all named fleet members here, before
        # leases are issued or a timestep runs. This must be a run-boundary
        # operation: it is never used to splice a resource into an active wave.
        prepare_run = getattr(self.backend, "prepare_run", None)
        if callable(prepare_run):
            requested = tuple(dict.fromkeys(
                resource
                for wave in schedule.waves
                for scheduled_step in wave.steps
                for resource in self._resources_for(scheduled_step, schedule)
            ))
            unknown = set(requested).difference(fleet.online_ids)
            if unknown:
                raise FleetError(
                    f"schedule names resources absent or offline in fleet: {sorted(unknown)}"
                )
            prepare_run(requested, world)
        current = fleet
        receipts: list[FleetWaveReceipt] = []
        clock_ms = start_ms
        for wave in schedule.waves:
            scheduled = tuple(s for s in wave.steps if self._resources_for(s, schedule))
            if not scheduled:
                clock_ms += self.wave_window_ms
                continue
            commands: list[FleetCommand] = []
            leases: list[CapabilityLease] = []
            lease_for_step: dict[str, CapabilityLease] = {}
            wave_start = clock_ms
            wave_end = wave_start + self.wave_window_ms
            working = current
            for scheduled_step in scheduled:
                step = self._resolve_step(scheduled_step, graph_steps)
                resources = self._resources_for(scheduled_step, schedule)
                region = scheduled_step.region_id
                self._validate_capability(current, step, resources, region)
                lease_id = f"{run_id}:w{wave.index}:{step.id}"
                lease = CapabilityLease(
                    lease_id=lease_id, owner=step.id, resources=resources,
                    start_ms=wave_start, end_ms=wave_end, purpose=step.op,
                )
                reservation = None
                if region is not None:
                    reservation = WorkspaceReservation(
                        reservation_id=f"{lease_id}:space", owner=step.id,
                        regions=(region,), start_ms=wave_start, end_ms=wave_end,
                        lease_id=lease_id,
                    )
                # Reserve the whole wave before anything moves.
                working = working.reserve(lease, reservation)
                leases.append(lease)
                commands.append(FleetCommand(
                    step_id=step.id, op=step.op, args=dict(step.args),
                    participants=resources, region_id=region, primary=scheduled_step.arm,
                ))
                lease_for_step[step.id] = lease
            try:
                results = tuple(self.backend.execute_wave(tuple(commands), world))
            except FleetBackendFault as exc:
                return self._fault_result(
                    current=working, fault=exc, leases=tuple(leases),
                    commands=tuple(commands), lease_for_step=lease_for_step,
                    wave=wave, observed_ms=wave_start, receipts=tuple(receipts),
                    org_id=world.org_id, run_id=run_id,
                )
            self._validate_results(tuple(commands), results)
            ok = all(result.ok for result in results)
            receipts.append(FleetWaveReceipt(
                wave_index=wave.index, commands=tuple(commands), results=results,
                leases=tuple(lease.lease_id for lease in leases), ok=ok,
            ))
            current = self._release_all(working, tuple(l.lease_id for l in leases))
            clock_ms = wave_end
            if not ok:
                return FleetRunResult(
                    fleet=current, waves=tuple(receipts), ok=False, failed_wave=wave.index,
                    detail={"reason": "backend reported manipulation failure"},
                )
        return FleetRunResult(fleet=current, waves=tuple(receipts), ok=True)

    @staticmethod
    def _resources_for(step: ScheduledStep, schedule: Schedule) -> tuple[str, ...]:
        resources = step.participants or schedule.resource_assignment.get(step.step_id, ())
        if not resources and step.arm is not None:
            resources = (step.arm,)
        return tuple(resources)

    @staticmethod
    def _resolve_step(scheduled: ScheduledStep, graph_steps: dict[str, Step]) -> Step:
        existing = graph_steps.get(scheduled.step_id)
        if existing is not None:
            return existing
        if not scheduled.op:
            raise FleetError(f"{scheduled.step_id}: scheduled step has no graph step or op")
        return Step(id=scheduled.step_id, contract="manipulate", op=scheduled.op,
                    args=dict(scheduled.args), arm=scheduled.arm)

    @staticmethod
    def _validate_capability(fleet, step, resources, region) -> None:
        if not resources:
            raise FleetError(f"{step.id}: no fleet resources assigned")
        for resource in resources:
            if not fleet.spec(resource).supports(step.op, region):
                raise FleetError(
                    f"{step.id}: {resource} cannot perform {step.op} in "
                    f"{region or 'unscoped workspace'}"
                )

    @staticmethod
    def _validate_results(commands, results) -> None:
        expected = [c.step_id for c in commands]
        actual = [r.step_id for r in results]
        if len(set(actual)) != len(actual):
            raise FleetError("backend returned duplicate step results")
        if set(actual) != set(expected):
            raise FleetError(
                f"backend result mismatch: expected {sorted(expected)}, got {sorted(actual)}"
            )

    @staticmethod
    def _release_all(fleet, lease_ids) -> ManipulationFleet:
        for lease_id in lease_ids:
            fleet = fleet.release(lease_id)
        return fleet

    def _fault_result(self, *, current, fault, leases, commands, lease_for_step,
                      wave: Wave, observed_ms, receipts, org_id, run_id) -> FleetRunResult:
        failed = frozenset(fault.resource_ids)
        requests = []
        for command in commands:
            lease = lease_for_step[command.step_id]
            if failed.intersection(lease.resources):
                requests.append(ReallocationRequest(
                    lease_id=lease.lease_id, capability=command.op, region=command.region_id,
                ))
        fault_event = FleetFault(
            fault_id=f"{run_id}:fault:w{wave.index}:{observed_ms}",
            resource_ids=tuple(fault.resource_ids), reason=fault.reason,
            observed_ms=observed_ms, org_id=org_id, source="fleet-backend",
        )
        reallocated, event = current.reallocate(fault_event, requests=requests)
        # Return evidence for governed replanning; never execute the stale graph.
        unwind_ids = {lease.lease_id for lease in leases}
        unwind_ids.update(new_id for _, new_id, _ in event.replacements)
        reallocated = self._release_all(reallocated, tuple(sorted(unwind_ids)))
        return FleetRunResult(
            fleet=reallocated, waves=receipts, ok=False, reallocation=event,
            failed_wave=wave.index, detail={"reason": fault.reason,
                "failed_resources": list(fault.resource_ids), "requires_replan": True},
        )
