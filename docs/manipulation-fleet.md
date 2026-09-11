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
separate backlog gates and are not implied by this planning slice.

## Evidence boundary

The tests prove data-model invariants, dynamic pairing, fail-closed capability
selection, resource/workspace conflict detection, and pure fault reallocation
selection. They do not prove:

- real-time concurrent execution;
- multi-arm collision avoidance in MuJoCo;
- physical robot control for more than the existing two-arm path; or
- throughput or business performance.
