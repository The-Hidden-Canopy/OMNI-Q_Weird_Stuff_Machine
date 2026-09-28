# Manipulation fleet substrate

The current Intel executor is a dual-SO-101 path. Multi-biarm support starts
below the planner with a resource-neutral fleet snapshot in
`src/omni_q/fleet.py`.

```text
ManipulationFleet
├── ManipulatorSpec × N
├── preferred BiArmUnit pairs
├── dynamically formable online pairs
├── CapabilityLease records
└── WorkspaceReservation records
```

## Boundary

`ManipulationFleet` is immutable. `with_online()`, `invalidate_resources()`, `reserve()`, and `release()`
return new snapshots. It does not write `WorldState`, execute a motor command,
or claim that multiple arms are moving simultaneously.

An offline transition fails closed when the arm still has an active lease. The
fault/replan path can call `reallocate(FleetFault, requests=...)`, which returns
a degraded snapshot plus event-ready `FleetReallocation` evidence. It first
invalidates affected leases and marks failed resources offline, then recreates
only the leases whose explicit capability context has an eligible replacement.
The caller still owns publication to `EventBus` and the governed plan/world
transition. Calling `invalidate_resources()` followed by
`with_online(..., False)` remains available when no reallocation is wanted.

Preferred pairs are only a selection ranking. If `arm_2` and `arm_3` are both
online and capable, `biarm:arm_2+arm_3` is available even when their normal
groups are `(arm_1, arm_2)` and `(arm_3, arm_4)`.

Capability selection fails closed: an arm with no declared capabilities cannot
be selected for an operation. A bimanual unit is eligible only when both
members are online, both declare the requested capability, and both cover the
requested named region.

Leases and workspace reservations are separate checks. Two pairs may have
different manipulators but still conflict when their time-overlapping named
workspace regions intersect:

```python
from omni_q.fleet import CapabilityLease, ManipulationFleet, WorkspaceReservation

lease = CapabilityLease("lease-17", "step-17", ("arm_1", "arm_2"), 14200, 16800)
space = WorkspaceReservation("space-17", "step-17", ("table.NW",), 14200, 16800)
next_fleet = fleet.reserve(lease, space)
```

`omni_q.scheduler.schedule(..., fleet=fleet)` now consumes the fleet's online
manipulator ids and capability/region declarations, chooses a preferred or
dynamic pair, and exposes all participants in `Schedule.resource_assignment`
and `ScheduledStep.participants`. It does not yet acquire
`CapabilityLease`/`WorkspaceReservation` objects or drive a real-time N-arm
executor. The pure fault reallocation seam exists, but lease-aware scheduler
acquisition, governed engine integration, and physical concurrency are
separate gates and are not implied by this planning slice.

## Wave runtime bridge

`omni_q.fleet_runtime.FleetExecutor` acquires capability leases and named
workspace reservations for a complete scheduled wave before calling
`FleetWaveBackend.execute_wave(commands, world)` once. Successful and reported
failed waves return receipts and release their temporary authority. A
`FleetBackendFault` returns a degraded fleet and reallocation evidence, releases
both original and replacement leases, and stops for governed replanning; it
does not resume the old graph with replacement arms.

This is a standalone seam, not yet wired into the governed engine. The
`IntelFleetBackend` joint-motion demo below exercises it. The caller owns event publication and governed state
transitions. Wave windows use a logical clock, not enforced physical deadlines.
Inputs must come from a validated scheduler: no-resource steps are skipped,
and scheduled synthetic operations are accepted when absent from the graph.
Malformed backend receipts raise `FleetError`; they are not successful runs.

The shared-world actuator demo below supplies named bindings and one physics
loop. Concurrent contact-based object manipulation remains the next physical
gate. Recording backend tests establish only one-call dispatch.

Fault identities are scoped to the executor run and retain the supplied
WorldState org rather than defaulting tenant evidence to `local-demo`.

## Shared-world actuator demo

`omni_q.intel_fleet` now provides `IntelFleetWorld`, named `ArmBinding` records,
and `IntelFleetBackend`. It creates five articulated SO-ARM100 proxy arms in
one model and one physical data instance. Joint/actuator addresses are resolved
by name, not left/right offsets. Each tick writes every active arm's controls,
steps the shared model once, then reads joint positions, velocities and contact
count. There are no execution-time qpos/qvel writes, object teleportation,
extra collision-mask exemptions, independent worlds, or controller threads.
The old two-arm implementation remains unchanged.
Receipts include a hash covering generated MJCF and mesh assets plus the
MuJoCo version; joint outcomes are measured rather than inferred from controls.

Run using the repository's existing MuJoCo environment from PowerShell:

```powershell
$env:PYTHONPATH = 'src'
& .venv/Scripts/python.exe -m omni_q.demo_intel_fleet
& .venv/Scripts/python.exe -m omni_q.demo_intel_fleet --arms 50
& .venv/Scripts/python.exe -m pytest tests/test_intel_fleet.py -q
```

This isolated demo uses `MOVE_JOINTS` and an explicit bimanual-team schedule. The
arms occupy separated stations; this is not yet a common-object bimanual
layout. `CO_ROTATE`, `PICK`, and `PLACE` are rejected. Receipt success means
joint position and velocity tolerances held for consecutive ticks, and every
receipt explicitly sets `physical_object_verified=False`. The measured initial
demo completed in 161 shared physics ticks (0.322 simulated seconds), with
four arms displaced more than 0.05 rad from their initial rotation in 133 ticks.
This is neither hardware timing nor an object-grasp result.

The default demo is now 50 arms arranged as 25 bimanual teams. This changes
the shared simulation's size and schedule cardinality, not the evidence claim:
all arms still receive controls before one MuJoCo step, but the stations remain
separated and there is no common-object cooperative manipulation. `--arms`
accepts any even count of two or more; the 50-arm regression asserts one model,
one data instance, 300 qpos/velocity/control entries, one 25-command wave, and
every active arm present in every recorded shared timestep.
The CLI prints a compact run summary by default; pass `--full-receipts` when
the full command/target and result evidence is needed.

New arm IDs are accepted automatically at run preparation: `FleetExecutor`
collects the full schedule's resources before leases or physics execution and
`IntelFleetBackend` compiles any missing arm bindings into that run's single
world. The IDs must already be online members of the authoritative fleet; a
schedule cannot manufacture resources. An attempted addition after a physics
step fails closed and must return through fault/replan into a new run, because
MuJoCo cannot safely modify a compiled world in place.

Adversarial tests cover cross-org requests, unsupported operations, malformed
or out-of-range targets, duplicate assignments, timeouts, and a test-injected
fault at tick 20. Fault intake queues only; the next loop boundary holds controls
at measured joint positions, stops physics and latches the backend. Through
`FleetExecutor`, the fault marks arm_1 offline, returns arm_2+arm_5 replacement
evidence and unwinds leases without executing the next wave. The fault is an
injection, not evidence of a naturally occurring servo failure. Hold is not a
hardware emergency-stop implementation. A restart cannot clear the fault latch.

Remaining gates: common-object contact controllers and per-object evidence,
collision safety monitoring (contact counts alone are not collision avoidance),
deadline enforcement, governed event publication, and explicit recovery after
strict replanning. No automatic replacement motion is enabled by this demo.

## Evidence boundary

The tests prove data-model invariants, dynamic pairing, fail-closed capability
selection, resource/workspace conflict detection, and pure fault reallocation
selection. They do not prove:

- real-time concurrent execution;
- multi-arm collision avoidance in MuJoCo;
- physical robot control for more than the existing two-arm path; or
- throughput or business performance.
