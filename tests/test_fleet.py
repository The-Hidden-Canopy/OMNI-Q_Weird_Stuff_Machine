"""Boundaries for resource-neutral multi-biarm fleet scheduling."""

from __future__ import annotations

import pytest

from omni_q.fleet import (
    CapabilityLease,
    FleetConflict,
    FleetError,
    FleetFault,
    FleetReallocation,
    FleetUnavailable,
    ManipulationFleet,
    ManipulatorSpec,
    ReallocationRequest,
    WorkspaceReservation,
)


CAPS = frozenset({"PICK", "PLACE", "CO_ROTATE"})


def _fleet() -> ManipulationFleet:
    return ManipulationFleet.from_specs(
        [
            ManipulatorSpec("arm_1", CAPS, frozenset({"north", "center"})),
            ManipulatorSpec("arm_2", CAPS, frozenset({"north", "center"})),
            ManipulatorSpec("arm_3", CAPS, frozenset({"south", "center"})),
            ManipulatorSpec("arm_4", CAPS, frozenset({"south", "center"})),
        ],
        preferred_pairs=(("arm_1", "arm_2"), ("arm_3", "arm_4")),
    )


def test_preferred_pairs_are_ranked_but_cross_group_pairs_are_available():
    fleet = _fleet()

    units = fleet.units()
    assert [unit.id for unit in units[:2]] == ["biarm:arm_1+arm_2", "biarm:arm_3+arm_4"]
    assert any(set(unit.members) == {"arm_2", "arm_3"} for unit in units[2:])
    assert all(unit.preferred for unit in units[:2])
    assert all(not unit.preferred for unit in units[2:])


def test_form_unit_rejects_offline_member_without_mutating_snapshot():
    fleet = _fleet()
    degraded = fleet.with_online("arm_3", False)

    with pytest.raises(FleetUnavailable, match="arm_3"):
        degraded.form_unit(("arm_2", "arm_3"))

    assert fleet.spec("arm_3").online is True
    assert degraded.spec("arm_3").online is False
    assert len(fleet.units(include_dynamic=False)) == 2
    assert len(degraded.units(include_dynamic=False)) == 1


def test_offline_transition_requires_lease_invalidation_first():
    fleet = _fleet()
    lease = CapabilityLease("lease-1", "step-1", ("arm_1", "arm_2"), 0, 100)
    space = WorkspaceReservation("space-1", "step-1", ("north",), 0, 100)
    reserved = fleet.reserve(lease, space)

    with pytest.raises(FleetConflict, match="invalidate or release"):
        reserved.with_online("arm_2", False)

    invalidated = reserved.invalidate_resources(("arm_2",))
    degraded = invalidated.with_online("arm_2", False)
    assert degraded.leases == ()
    assert degraded.reservations == ()
    assert reserved.spec("arm_2").online is True


def test_eligibility_requires_both_members_capability_and_region():
    fleet = _fleet()

    north = fleet.eligible_units("CO_ROTATE", region="north")
    south = fleet.eligible_units("CO_ROTATE", region="south")

    assert [unit.id for unit in north] == ["biarm:arm_1+arm_2"]
    assert [unit.id for unit in south] == ["biarm:arm_3+arm_4"]


def test_empty_capabilities_fail_closed():
    fleet = ManipulationFleet((ManipulatorSpec("arm_1"), ManipulatorSpec("arm_2")))
    assert fleet.eligible_units("PICK") == ()


def test_overlapping_resource_lease_is_rejected_but_adjacent_time_is_allowed():
    fleet = _fleet()
    first = CapabilityLease("lease-1", "step-1", ("arm_1", "arm_2"), 0, 100)
    second = CapabilityLease("lease-2", "step-2", ("arm_2", "arm_3"), 50, 150)
    adjacent = CapabilityLease("lease-3", "step-3", ("arm_2", "arm_3"), 100, 150)

    reserved = fleet.reserve(first)
    with pytest.raises(FleetConflict, match="lease-2"):
        reserved.reserve(second)

    next_snapshot = reserved.reserve(adjacent)
    assert [lease.lease_id for lease in next_snapshot.leases] == ["lease-1", "lease-3"]
    assert fleet.leases == ()


def test_workspace_conflict_is_independent_of_manipulator_pair():
    fleet = _fleet()
    first = CapabilityLease("lease-1", "step-1", ("arm_1", "arm_2"), 0, 100)
    second = CapabilityLease("lease-2", "step-2", ("arm_3", "arm_4"), 20, 80)
    first_space = WorkspaceReservation("space-1", "step-1", ("center",), 0, 100)
    second_space = WorkspaceReservation("space-2", "step-2", ("center",), 20, 80)

    reserved = fleet.reserve(first, first_space)
    with pytest.raises(FleetConflict, match="space-2"):
        reserved.reserve(second, second_space)


def test_workspace_reservation_cannot_outlive_its_capability_lease():
    fleet = _fleet()
    lease = CapabilityLease("lease-1", "step-1", ("arm_1", "arm_2"), 100, 200)
    early = WorkspaceReservation("space-early", "step-1", ("north",), 90, 150)
    late = WorkspaceReservation("space-late", "step-1", ("north",), 150, 210)

    with pytest.raises(FleetError, match="contained"):
        fleet.reserve(lease, early)
    with pytest.raises(FleetError, match="contained"):
        fleet.reserve(lease, late)


def test_release_returns_new_snapshot_and_removes_owned_workspace():
    fleet = _fleet()
    lease = CapabilityLease("lease-1", "step-1", ("arm_1", "arm_2"), 0, 100)
    space = WorkspaceReservation("space-1", "step-1", ("north",), 0, 100)
    reserved = fleet.reserve(lease, space)
    released = reserved.release("lease-1")

    assert reserved.leases == (lease,)
    assert reserved.reservations[0].lease_id == "lease-1"
    assert released.leases == ()
    assert released.reservations == ()


def test_release_does_not_remove_a_sibling_lease_for_the_same_owner():
    fleet = _fleet()
    first = CapabilityLease("lease-1", "same-owner", ("arm_1", "arm_2"), 0, 100)
    second = CapabilityLease("lease-2", "same-owner", ("arm_3", "arm_4"), 0, 100)
    first_space = WorkspaceReservation("space-1", "same-owner", ("north",), 0, 100)
    second_space = WorkspaceReservation("space-2", "same-owner", ("south",), 0, 100)
    reserved = fleet.reserve(first, first_space).reserve(second, second_space)

    released = reserved.release("lease-1")

    assert [lease.lease_id for lease in released.leases] == ["lease-2"]
    assert [space.reservation_id for space in released.reservations] == ["space-2"]


def test_invalid_workspace_owner_and_unknown_resource_fail_closed():
    fleet = _fleet()
    lease = CapabilityLease("lease-1", "step-1", ("arm_9",), 0, 100)
    with pytest.raises(FleetUnavailable, match="arm_9"):
        fleet.reserve(lease)

    valid = CapabilityLease("lease-2", "step-2", ("arm_1", "arm_2"), 0, 100)
    wrong_owner = WorkspaceReservation("space-2", "other-step", ("north",), 0, 100)
    with pytest.raises(FleetError, match="owner"):
        fleet.reserve(valid, wrong_owner)


def test_fault_reallocation_preserves_authority_and_uses_a_surviving_pair():
    fleet = ManipulationFleet.from_specs(
        [
            ManipulatorSpec("arm_1", CAPS, frozenset({"north"})),
            ManipulatorSpec("arm_2", CAPS, frozenset({"north"})),
            ManipulatorSpec("arm_3", CAPS, frozenset({"south"})),
            ManipulatorSpec("arm_4", CAPS, frozenset({"south"})),
            ManipulatorSpec("arm_5", CAPS, frozenset({"north"})),
        ],
        preferred_pairs=(("arm_1", "arm_2"), ("arm_3", "arm_4")),
    )
    lease = CapabilityLease(
        "lease-1", "task:place-plate", ("arm_1", "arm_2"), 0, 100,
        purpose="place plate in north",
    )
    space = WorkspaceReservation("space-1", "task:place-plate", ("north",), 0, 100)
    reserved = fleet.reserve(lease, space)

    degraded, evidence = reserved.reallocate(
        FleetFault("fault-1", ("arm_1",), "motor health unavailable", 25),
        requests=(ReallocationRequest("lease-1", "CO_ROTATE", "north"),),
    )

    assert isinstance(evidence, FleetReallocation)
    assert degraded.spec("arm_1").online is False
    assert [item.lease_id for item in degraded.leases] == ["lease-1:replan:fault-1"]
    replacement = degraded.leases[0]
    assert replacement.resources == ("arm_2", "arm_5")
    assert replacement.owner == lease.owner
    assert replacement.purpose == lease.purpose
    assert degraded.reservations[0].lease_id == replacement.lease_id
    assert evidence.invalidated_lease_ids == ("lease-1",)
    assert evidence.affected_owners == ("task:place-plate",)
    assert evidence.observed_ms == 25
    assert evidence.replacements == (
        ("lease-1", "lease-1:replan:fault-1", ("arm_2", "arm_5")),
    )
    assert evidence.unresolved_lease_ids == ()
    assert evidence.as_dict()["kind"] == "fleet.reallocation"
    assert evidence.as_dict()["requires_replan"] is True
    assert reserved.spec("arm_1").online is True
    assert reserved.leases == (lease,)


def test_fault_without_capability_context_fails_closed_and_reports_unresolved_lease():
    fleet = _fleet()
    lease = CapabilityLease("lease-1", "task-1", ("arm_1", "arm_2"), 0, 100)
    reserved = fleet.reserve(
        lease,
        WorkspaceReservation("space-1", "task-1", ("north",), 0, 100),
    )

    degraded, evidence = reserved.reallocate(
        FleetFault("fault-1", ("arm_1",), "fault signal", 50),
    )

    assert degraded.spec("arm_1").online is False
    assert degraded.leases == ()
    assert degraded.reservations == ()
    assert evidence.invalidated_lease_ids == ("lease-1",)
    assert evidence.unresolved_lease_ids == ("lease-1",)
    assert evidence.replacements == ()


def test_reallocation_request_rejects_an_invalid_region_before_planning():
    with pytest.raises(FleetError, match="region"):
        ReallocationRequest("lease-1", "PICK", "")


def test_fault_preserves_unrelated_leases_and_does_not_reuse_busy_resources():
    fleet = ManipulationFleet.from_specs(
        [
            ManipulatorSpec("arm_1", CAPS, frozenset({"north"})),
            ManipulatorSpec("arm_2", CAPS, frozenset({"north"})),
            ManipulatorSpec("arm_3", CAPS, frozenset({"north"})),
            ManipulatorSpec("arm_4", CAPS, frozenset({"north"})),
        ],
        preferred_pairs=(("arm_1", "arm_2"), ("arm_3", "arm_4")),
    )
    affected = CapabilityLease("lease-1", "task-1", ("arm_1", "arm_2"), 0, 100)
    unrelated = CapabilityLease("lease-2", "task-2", ("arm_3", "arm_4"), 0, 100)
    reserved = fleet.reserve(affected).reserve(unrelated)

    degraded, evidence = reserved.reallocate(
        FleetFault("fault-1", ("arm_1",), "joint fault", 10),
        requests=(ReallocationRequest("lease-1", "CO_ROTATE", "north"),),
    )

    assert [lease.lease_id for lease in degraded.leases] == ["lease-2"]
    assert evidence.unresolved_lease_ids == ("lease-1",)
    assert evidence.replacements == ()
    assert degraded.spec("arm_3").online is True
    assert degraded.spec("arm_4").online is True


def test_reallocation_rejects_a_request_for_a_lease_not_touched_by_the_fault():
    fleet = _fleet()

    with pytest.raises(FleetError, match="not for an invalidated lease"):
        fleet.reallocate(
            FleetFault("fault-1", ("arm_1",), "joint fault", 10),
            requests=(ReallocationRequest("unknown-lease", "PICK", "north"),),
        )
