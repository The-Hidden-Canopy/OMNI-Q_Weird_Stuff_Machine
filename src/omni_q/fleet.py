"""Resource-neutral manipulation fleet primitives.

This module is the substrate for multi-biarm scheduling.  It deliberately does
not execute motors or mutate :class:`~omni_q.contracts.WorldState`:

* a :class:`ManipulatorSpec` describes one available physical resource;
* a :class:`BiArmUnit` describes a preferred or dynamically formed pair;
* a :class:`CapabilityLease` reserves manipulator ids for a bounded interval;
* a :class:`WorkspaceReservation` reserves named physical regions for that
  interval; and
* :class:`ManipulationFleet` returns a new snapshot for every reservation or
  online/offline change.

The immutable snapshot boundary is intentional.  A future scheduler can turn
these decisions into governed graph transitions without letting a planner
directly mutate authoritative world state.  Preferred pairs are only a
ranking signal: every online pair can be formed when its members and
capabilities permit it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
import hashlib
import json
from typing import Any, Iterable


class FleetError(ValueError):
    """Base class for invalid fleet definitions or resource requests."""


class FleetUnavailable(FleetError):
    """A requested manipulator or pair is not currently online."""


class FleetConflict(FleetError):
    """A lease or workspace reservation overlaps an active reservation."""


@dataclass(frozen=True)
class FleetFault:
    """A bounded resource-health observation that requires reallocation."""

    fault_id: str
    resource_ids: tuple[str, ...]
    reason: str
    observed_ms: int
    org_id: str = "local-demo"
    source: str = "proprioception"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.fault_id, str)
            or not self.fault_id
            or not isinstance(self.reason, str)
            or not self.reason.strip()
        ):
            raise FleetError("fault_id and reason must be non-empty")
        if (
            not isinstance(self.org_id, str)
            or not self.org_id.strip()
            or not isinstance(self.source, str)
            or not self.source.strip()
        ):
            raise FleetError("fault org_id and source must be non-empty strings")
        _unique_ids(self.resource_ids, field_name="fault resources")
        if (
            not isinstance(self.observed_ms, int)
            or isinstance(self.observed_ms, bool)
            or self.observed_ms < 0
        ):
            raise FleetError("fault observed_ms must be a non-negative integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "fleet.fault",
            "fault_id": self.fault_id,
            "resource_ids": list(self.resource_ids),
            "reason": self.reason,
            "observed_ms": self.observed_ms,
            "org_id": self.org_id,
            "source": self.source,
        }


@dataclass(frozen=True)
class ReallocationRequest:
    """The capability context needed to safely retry one invalidated lease."""

    lease_id: str
    capability: str
    region: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.lease_id, str)
            or not self.lease_id
            or not isinstance(self.capability, str)
            or not self.capability
        ):
            raise FleetError("reallocation lease_id and capability must be non-empty")
        if self.region is not None and (
            not isinstance(self.region, str) or not self.region
        ):
            raise FleetError("reallocation region must be a non-empty string")


@dataclass(frozen=True)
class FleetReallocation:
    """Event-ready evidence for a fault-driven fleet snapshot change.

    This is deliberately a value object rather than an EventBus side effect.
    The governed runtime owns publication and any WorldState/plan transition.
    """

    fault_id: str
    reason: str
    observed_ms: int
    failed_resources: tuple[str, ...]
    invalidated_lease_ids: tuple[str, ...]
    affected_owners: tuple[str, ...]
    replacements: tuple[tuple[str, str, tuple[str, ...]], ...]
    unresolved_lease_ids: tuple[str, ...]
    prior_fleet_digest: str
    resulting_fleet_digest: str
    requires_replan: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "fleet.reallocation",
            "fault_id": self.fault_id,
            "reason": self.reason,
            "observed_ms": self.observed_ms,
            "failed_resources": list(self.failed_resources),
            "invalidated_lease_ids": list(self.invalidated_lease_ids),
            "affected_owners": list(self.affected_owners),
            "replacements": [
                {
                    "old_lease_id": old_id,
                    "new_lease_id": new_id,
                    "resources": list(resources),
                }
                for old_id, new_id, resources in self.replacements
            ],
            "unresolved_lease_ids": list(self.unresolved_lease_ids),
            "prior_fleet_digest": self.prior_fleet_digest,
            "resulting_fleet_digest": self.resulting_fleet_digest,
            "requires_replan": self.requires_replan,
        }


def _unique_ids(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    ids = tuple(values)
    if not ids or any(not value for value in ids):
        raise FleetError(f"{field_name} must contain non-empty ids")
    if len(set(ids)) != len(ids):
        raise FleetError(f"{field_name} must not contain duplicates")
    return ids


def _intervals_overlap(
    start_ms: int,
    end_ms: int,
    other_start_ms: int,
    other_end_ms: int,
) -> bool:
    return start_ms < other_end_ms and other_start_ms < end_ms


def _fleet_digest(fleet: "ManipulationFleet") -> str:
    encoded = json.dumps(
        fleet.as_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ManipulatorSpec:
    """Static identity and declared capability of one manipulator."""

    id: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    workspace_regions: frozenset[str] = field(default_factory=frozenset)
    online: bool = True
    preferred_group: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise FleetError("manipulator id must be non-empty")
        if any(not capability for capability in self.capabilities):
            raise FleetError(f"{self.id}: capabilities must be non-empty strings")
        if any(not region for region in self.workspace_regions):
            raise FleetError(f"{self.id}: workspace regions must be non-empty strings")

    def supports(self, capability: str, region: str | None = None) -> bool:
        """Return whether this online manipulator declares a capability.

        An empty capability set is intentionally *not* treated as universal
        access.  Callers must declare capabilities before a resource can be
        selected for an action.
        """
        if not self.online or capability not in self.capabilities:
            return False
        return region is None or region in self.workspace_regions


@dataclass(frozen=True)
class BiArmUnit:
    """Two-manipulator resource, preferred or formed dynamically."""

    id: str
    members: tuple[str, str]
    preferred: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise FleetError("biarm unit id must be non-empty")
        if len(self.members) != 2 or len(set(self.members)) != 2:
            raise FleetError("a biarm unit must contain two distinct manipulators")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "members": list(self.members),
            "preferred": self.preferred,
        }


@dataclass(frozen=True)
class CapabilityLease:
    """A time-bounded reservation of one or more manipulator resources."""

    lease_id: str
    owner: str
    resources: tuple[str, ...]
    start_ms: int
    end_ms: int
    purpose: str = ""

    def __post_init__(self) -> None:
        if not self.lease_id or not self.owner:
            raise FleetError("lease_id and owner must be non-empty")
        _unique_ids(self.resources, field_name="lease resources")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise FleetError("lease interval must satisfy 0 <= start_ms < end_ms")

    def overlaps(self, other: "CapabilityLease") -> bool:
        return bool(set(self.resources) & set(other.resources)) and _intervals_overlap(
            self.start_ms, self.end_ms, other.start_ms, other.end_ms
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "owner": self.owner,
            "resources": list(self.resources),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "purpose": self.purpose,
        }


@dataclass(frozen=True)
class WorkspaceReservation:
    """A time-bounded reservation of named workspace regions."""

    reservation_id: str
    owner: str
    regions: tuple[str, ...]
    start_ms: int
    end_ms: int
    lease_id: str | None = None

    def __post_init__(self) -> None:
        if not self.reservation_id or not self.owner:
            raise FleetError("reservation_id and owner must be non-empty")
        _unique_ids(self.regions, field_name="workspace regions")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise FleetError("reservation interval must satisfy 0 <= start_ms < end_ms")

    def overlaps(self, other: "WorkspaceReservation") -> bool:
        return bool(set(self.regions) & set(other.regions)) and _intervals_overlap(
            self.start_ms, self.end_ms, other.start_ms, other.end_ms
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "reservation_id": self.reservation_id,
            "owner": self.owner,
            "regions": list(self.regions),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "lease_id": self.lease_id,
        }


@dataclass(frozen=True)
class ManipulationFleet:
    """Immutable fleet snapshot used by planning and resource arbitration."""

    manipulators: tuple[ManipulatorSpec, ...]
    preferred_units: tuple[BiArmUnit, ...] = ()
    leases: tuple[CapabilityLease, ...] = ()
    reservations: tuple[WorkspaceReservation, ...] = ()

    def __post_init__(self) -> None:
        ids = _unique_ids((m.id for m in self.manipulators), field_name="manipulators")
        known = set(ids)
        for unit in self.preferred_units:
            if not set(unit.members) <= known:
                raise FleetError(f"{unit.id}: member is not in the fleet")
        lease_ids = _unique_ids((lease.lease_id for lease in self.leases), field_name="leases") \
            if self.leases else ()
        if len(lease_ids) != len(self.leases):  # defensive; _unique_ids already checks
            raise FleetError("lease ids must be unique")
        for lease in self.leases:
            if not set(lease.resources) <= known:
                raise FleetError(f"{lease.lease_id}: resource is not in the fleet")
        for first, second in combinations(self.leases, 2):
            if first.overlaps(second):
                raise FleetConflict(f"capability leases conflict: {first.lease_id}, {second.lease_id}")
        reservation_ids = (
            _unique_ids(
                (reservation.reservation_id for reservation in self.reservations),
                field_name="workspace reservations",
            )
            if self.reservations else ()
        )
        if len(reservation_ids) != len(self.reservations):  # defensive
            raise FleetError("workspace reservation ids must be unique")
        lease_by_id = {lease.lease_id: lease for lease in self.leases}
        for reservation in self.reservations:
            if reservation.lease_id not in lease_by_id:
                raise FleetError(
                    f"{reservation.reservation_id}: reservation must reference an active lease"
                )
            if reservation.owner != lease_by_id[reservation.lease_id].owner:
                raise FleetError(
                    f"{reservation.reservation_id}: reservation owner does not match its lease"
                )
        for first, second in combinations(self.reservations, 2):
            if first.overlaps(second):
                raise FleetConflict(
                    f"workspace reservations conflict: {first.reservation_id}, {second.reservation_id}"
                )

    @classmethod
    def from_specs(
        cls,
        manipulators: Iterable[ManipulatorSpec],
        *,
        preferred_pairs: Iterable[tuple[str, str]] = (),
    ) -> "ManipulationFleet":
        specs = tuple(manipulators)
        known = {spec.id for spec in specs}
        units = tuple(
            BiArmUnit(
                id=f"biarm:{left}+{right}",
                members=(left, right),
                preferred=True,
            )
            for left, right in preferred_pairs
        )
        if any(member not in known for unit in units for member in unit.members):
            raise FleetError("preferred pair contains an unknown manipulator")
        return cls(specs, units)

    @classmethod
    def from_ids(
        cls,
        ids: Iterable[str],
        *,
        capabilities: dict[str, frozenset[str]] | None = None,
        workspace_regions: dict[str, frozenset[str]] | None = None,
        preferred_pairs: Iterable[tuple[str, str]] = (),
    ) -> "ManipulationFleet":
        all_ids = _unique_ids(ids, field_name="manipulators")
        capability_map = capabilities or {}
        region_map = workspace_regions or {}
        specs = tuple(
            ManipulatorSpec(
                id=manipulator_id,
                capabilities=capability_map.get(manipulator_id, frozenset()),
                workspace_regions=region_map.get(manipulator_id, frozenset()),
            )
            for manipulator_id in all_ids
        )
        return cls.from_specs(specs, preferred_pairs=preferred_pairs)

    @property
    def manipulator_ids(self) -> tuple[str, ...]:
        return tuple(manipulator.id for manipulator in self.manipulators)

    @property
    def online_ids(self) -> tuple[str, ...]:
        return tuple(manipulator.id for manipulator in self.manipulators if manipulator.online)

    def digest(self) -> str:
        """Return the stable digest of this immutable fleet snapshot."""
        return _fleet_digest(self)

    def spec(self, manipulator_id: str) -> ManipulatorSpec:
        for manipulator in self.manipulators:
            if manipulator.id == manipulator_id:
                return manipulator
        raise FleetError(f"unknown manipulator: {manipulator_id}")

    def units(self, *, include_dynamic: bool = True) -> tuple[BiArmUnit, ...]:
        """Return online preferred units followed by all online dynamic pairs."""
        online = set(self.online_ids)
        preferred = tuple(
            unit for unit in self.preferred_units if set(unit.members) <= online
        )
        if not include_dynamic:
            return preferred
        seen = {frozenset(unit.members) for unit in preferred}
        dynamic = tuple(
            BiArmUnit(
                id=f"biarm:{left}+{right}",
                members=(left, right),
                preferred=False,
            )
            for left, right in combinations(self.online_ids, 2)
            if frozenset((left, right)) not in seen
        )
        return preferred + dynamic

    def form_unit(self, members: tuple[str, str]) -> BiArmUnit:
        """Form a cross-group pair if both requested members are online."""
        unit = BiArmUnit(id=f"biarm:{members[0]}+{members[1]}", members=members)
        if not set(unit.members) <= set(self.online_ids):
            missing = sorted(set(unit.members) - set(self.online_ids))
            raise FleetUnavailable(f"biarm unit unavailable: {missing}")
        for preferred in self.preferred_units:
            if frozenset(preferred.members) == frozenset(unit.members):
                return preferred
        return unit

    def eligible_units(
        self,
        capability: str,
        *,
        region: str | None = None,
        exclude: frozenset[str] = frozenset(),
    ) -> tuple[BiArmUnit, ...]:
        """Return pairs whose two online members support the request."""
        return tuple(
            unit for unit in self.units()
            if not (set(unit.members) & set(exclude))
            and all(self.spec(member).supports(capability, region) for member in unit.members)
        )

    def eligible_manipulators(
        self,
        capability: str,
        *,
        region: str | None = None,
        exclude: frozenset[str] = frozenset(),
    ) -> tuple[ManipulatorSpec, ...]:
        """Return online single-arm resources eligible for an operation."""
        return tuple(
            manipulator for manipulator in self.manipulators
            if manipulator.id not in exclude
            and manipulator.supports(capability, region)
        )

    def reserve(
        self,
        lease: CapabilityLease,
        workspace: WorkspaceReservation | None = None,
    ) -> "ManipulationFleet":
        """Return a new snapshot with a validated lease and optional workspace."""
        if not set(lease.resources) <= set(self.online_ids):
            unavailable = sorted(set(lease.resources) - set(self.online_ids))
            raise FleetUnavailable(f"lease {lease.lease_id} uses offline/unknown resources: {unavailable}")
        if any(existing.overlaps(lease) for existing in self.leases):
            raise FleetConflict(f"capability lease conflict: {lease.lease_id}")
        if workspace is not None:
            if workspace.owner != lease.owner:
                raise FleetError("workspace owner must match capability lease owner")
            if workspace.lease_id is not None and workspace.lease_id != lease.lease_id:
                raise FleetError("workspace reservation lease_id must match capability lease")
            if (
                workspace.start_ms < lease.start_ms
                or workspace.end_ms > lease.end_ms
            ):
                raise FleetError(
                    "workspace reservation must be contained within its capability lease"
                )
            if any(
                region not in self.spec(resource).workspace_regions
                for resource in lease.resources
                for region in workspace.regions
            ):
                raise FleetUnavailable(
                    f"lease {lease.lease_id} does not have declared workspace coverage"
                )
            if any(existing.overlaps(workspace) for existing in self.reservations):
                raise FleetConflict(f"workspace reservation conflict: {workspace.reservation_id}")
            workspace = replace(workspace, lease_id=lease.lease_id)
        return replace(
            self,
            leases=self.leases + (lease,),
            reservations=self.reservations + ((workspace,) if workspace is not None else ()),
        )

    def release(self, lease_id: str) -> "ManipulationFleet":
        """Return a new snapshot with a lease and its owned reservations removed."""
        if not lease_id:
            raise FleetError("lease_id must be non-empty")
        leases = tuple(lease for lease in self.leases if lease.lease_id != lease_id)
        reservations = tuple(
            reservation for reservation in self.reservations
            if reservation.lease_id != lease_id
        )
        return replace(self, leases=leases, reservations=reservations)

    def with_online(self, manipulator_id: str, online: bool) -> "ManipulationFleet":
        """Return a new snapshot with one manipulator's health state changed."""
        self.spec(manipulator_id)
        if not online and any(
            manipulator_id in lease.resources for lease in self.leases
        ):
            raise FleetConflict(
                f"cannot take {manipulator_id} offline while a capability lease is active; "
                "invalidate or release the lease first"
            )
        specs = tuple(
            replace(spec, online=online) if spec.id == manipulator_id else spec
            for spec in self.manipulators
        )
        return replace(self, manipulators=specs)

    def invalidate_resources(self, resource_ids: Iterable[str]) -> "ManipulationFleet":
        """Return a snapshot with leases touching failed resources removed.

        This is deliberately separate from :meth:`with_online`: the caller can
        record the invalidation/replan event before marking a resource offline.
        No authoritative world mutation happens here.
        """
        failed = set(_unique_ids(resource_ids, field_name="failed resources"))
        known = set(self.manipulator_ids)
        if not failed <= known:
            raise FleetError(f"unknown failed resources: {sorted(failed - known)}")
        invalidated = {
            lease.lease_id for lease in self.leases if failed & set(lease.resources)
        }
        return replace(
            self,
            leases=tuple(lease for lease in self.leases if lease.lease_id not in invalidated),
            reservations=tuple(
                reservation for reservation in self.reservations
                if reservation.lease_id not in invalidated
            ),
        )

    def reallocate(
        self,
        fault: FleetFault,
        *,
        requests: Iterable[ReallocationRequest] = (),
    ) -> tuple["ManipulationFleet", FleetReallocation]:
        """Return a degraded snapshot and event-ready reallocation evidence.

        The method is intentionally a pure resource transition.  It first
        invalidates every lease touching a failed resource, marks those
        resources offline in the returned snapshot, then attempts to recreate
        each affected one- or two-resource lease using the explicitly supplied
        capability context.  Missing capability context, occupied resources,
        workspace conflicts, or an unavailable replacement leave that lease
        unresolved instead of guessing.

        Lease owners and purposes are copied exactly.  A replacement receives
        a new id so evidence can distinguish the invalidated authority from
        the replacement authority.  The caller must publish the returned
        :class:`FleetReallocation` and request the governed plan/world
        transition; this method performs neither side effect.
        """
        failed = set(_unique_ids(fault.resource_ids, field_name="fault resources"))
        known = set(self.manipulator_ids)
        if not failed <= known:
            raise FleetError(f"unknown failed resources: {sorted(failed - known)}")

        affected = tuple(
            lease for lease in self.leases if failed & set(lease.resources)
        )
        request_list = tuple(requests)
        request_ids = _unique_ids(
            (request.lease_id for request in request_list),
            field_name="reallocation requests",
        ) if request_list else ()
        affected_ids = {lease.lease_id for lease in affected}
        unknown_requests = set(request_ids) - affected_ids
        if unknown_requests:
            raise FleetError(
                f"reallocation request is not for an invalidated lease: "
                f"{sorted(unknown_requests)}"
            )
        request_by_lease = {request.lease_id: request for request in request_list}

        prior_digest = self.digest()
        working = self.invalidate_resources(fault.resource_ids)
        for resource_id in sorted(failed):
            working = working.with_online(resource_id, False)

        replacements: list[tuple[str, str, tuple[str, ...]]] = []
        unresolved: list[str] = []

        for lease in affected:
            request = request_by_lease.get(lease.lease_id)
            if request is None or len(lease.resources) not in {1, 2}:
                unresolved.append(lease.lease_id)
                continue

            workspaces = tuple(
                reservation for reservation in self.reservations
                if reservation.lease_id == lease.lease_id
            )
            # The public reserve seam currently accepts one workspace
            # reservation per lease.  Refuse to collapse multiple regions.
            if len(workspaces) > 1:
                unresolved.append(lease.lease_id)
                continue

            required_regions: tuple[str, ...] = (
                tuple(workspaces[0].regions) if workspaces else ()
            )
            if request.region is not None and request.region not in required_regions:
                required_regions += (request.region,)

            busy = {
                resource
                for active in working.leases
                if _intervals_overlap(
                    lease.start_ms, lease.end_ms, active.start_ms, active.end_ms
                )
                for resource in active.resources
            }
            exclude = frozenset(failed | busy)

            def supports(resources: tuple[str, ...]) -> bool:
                return all(
                    working.spec(resource).supports(request.capability, region)
                    for resource in resources
                    for region in required_regions
                )

            if len(lease.resources) == 2:
                candidates = tuple(
                    unit for unit in working.units()
                    if not (set(unit.members) & set(exclude))
                    and supports(unit.members)
                )
                unit = min(
                    candidates,
                    key=lambda candidate: (
                        -len(set(candidate.members) & set(lease.resources)),
                        -int(candidate.preferred),
                        candidate.id,
                    ),
                    default=None,
                )
                replacement_resources = unit.members if unit is not None else None
            else:
                candidates = tuple(
                    spec for spec in working.manipulators
                    if spec.id not in exclude
                    and spec.supports(request.capability, None)
                    and all(spec.supports(request.capability, region)
                            for region in required_regions)
                )
                spec = min(
                    candidates,
                    key=lambda candidate: (
                        -int(candidate.id in lease.resources), candidate.id
                    ),
                    default=None,
                )
                replacement_resources = (spec.id,) if spec is not None else None

            if replacement_resources is None:
                unresolved.append(lease.lease_id)
                continue

            new_lease_id = f"{lease.lease_id}:replan:{fault.fault_id}"
            replacement_lease = CapabilityLease(
                lease_id=new_lease_id,
                owner=lease.owner,
                resources=replacement_resources,
                start_ms=lease.start_ms,
                end_ms=lease.end_ms,
                purpose=lease.purpose,
            )
            workspace = None
            if workspaces:
                workspace = replace(
                    workspaces[0],
                    reservation_id=f"{workspaces[0].reservation_id}:replan:{fault.fault_id}",
                    lease_id=new_lease_id,
                )
            try:
                working = working.reserve(replacement_lease, workspace)
            except FleetError:
                unresolved.append(lease.lease_id)
                continue
            replacements.append((lease.lease_id, new_lease_id, replacement_resources))

        evidence = FleetReallocation(
            fault_id=fault.fault_id,
            reason=fault.reason,
            observed_ms=fault.observed_ms,
            failed_resources=tuple(fault.resource_ids),
            invalidated_lease_ids=tuple(lease.lease_id for lease in affected),
            affected_owners=tuple(dict.fromkeys(lease.owner for lease in affected)),
            replacements=tuple(replacements),
            unresolved_lease_ids=tuple(unresolved),
            prior_fleet_digest=prior_digest,
            resulting_fleet_digest=working.digest(),
        )
        return working, evidence

    def as_dict(self) -> dict[str, Any]:
        return {
            "manipulators": [
                {
                    "id": spec.id,
                    "capabilities": sorted(spec.capabilities),
                    "workspace_regions": sorted(spec.workspace_regions),
                    "online": spec.online,
                    "preferred_group": spec.preferred_group,
                }
                for spec in self.manipulators
            ],
            "preferred_units": [unit.as_dict() for unit in self.preferred_units],
            "online_units": [unit.as_dict() for unit in self.units()],
            "leases": [lease.as_dict() for lease in self.leases],
            "reservations": [reservation.as_dict() for reservation in self.reservations],
        }
