import pytest

from omni_q.contracts import ManipResult, PlanGraph, Step, WorldState
from omni_q.fleet import ManipulationFleet, ManipulatorSpec, FleetError
from omni_q.fleet_runtime import FleetBackendFault, FleetCommand, FleetExecutor
from omni_q.scheduler import Schedule, ScheduledStep, Wave

CAPS = frozenset({"CO_ROTATE", "PICK", "PLACE"})
REGIONS = frozenset({"table.N", "table.S"})


def make_fleet(n=5):
    return ManipulationFleet.from_specs(
        tuple(ManipulatorSpec(id=f"arm_{i}", capabilities=CAPS,
                              workspace_regions=REGIONS) for i in range(1, n + 1)),
        preferred_pairs=(("arm_1", "arm_2"), ("arm_3", "arm_4")),
    )


class RecordingBackend:
    def __init__(self):
        self.waves: list[tuple[FleetCommand, ...]] = []

    def execute_wave(self, commands, world):
        self.waves.append(tuple(commands))
        return tuple(ManipResult(c.step_id, True, {"participants": list(c.participants)})
                     for c in commands)


class FaultBackend:
    def execute_wave(self, commands, world):
        raise FleetBackendFault(("arm_1",), "servo fault")


def world():
    return WorldState(frame=0, objects={})


def plan(two=True, south_resources=("arm_3", "arm_4"), south_region="table.S"):
    steps = [Step("north", "manipulate", "CO_ROTATE", {"to": "table.N"})]
    scheduled = [ScheduledStep("north", "arm_1", "table.N",
                               participants=("arm_1", "arm_2"))]
    if two:
        steps.append(Step("south", "manipulate", "CO_ROTATE", {"to": south_region}))
        scheduled.append(ScheduledStep("south", south_resources[0], south_region,
                                       participants=south_resources))
    return PlanGraph(goal="two teams move together", steps=steps), Schedule(
        waves=[Wave(0, scheduled)], barriers=[], arm_timeline={},
        metrics={"max_resource_parallelism": 4 if two else 2},
        assignment={s.step_id: s.arm for s in scheduled},
        regions={s.step_id: s.region_id for s in scheduled},
        resource_assignment={s.step_id: s.participants for s in scheduled},
    )


def test_executes_two_biarm_units_as_one_backend_wave():
    graph, schedule = plan()
    backend = RecordingBackend()
    result = FleetExecutor(backend).execute(graph, schedule, world(), make_fleet())
    assert result.ok
    assert len(backend.waves) == 1
    assert {r for c in backend.waves[0] for r in c.participants} == {
        "arm_1", "arm_2", "arm_3", "arm_4"}
    assert result.fleet.leases == ()
    assert result.fleet.reservations == ()
    assert result.waves[0].as_dict()["ok"] is True


def test_fault_takes_arm_offline_and_returns_replan_evidence():
    graph, schedule = plan(two=False)
    result = FleetExecutor(FaultBackend()).execute(
        graph, schedule, world(), make_fleet(), start_ms=100)
    assert not result.ok
    assert result.reallocation is not None
    assert result.reallocation.requires_replan
    assert "arm_1" not in result.fleet.online_ids
    assert result.fleet.leases == ()
    assert result.fleet.reservations == ()
    assert result.reallocation.replacements
    _, _, resources = result.reallocation.replacements[0]
    assert "arm_2" in resources
    assert "arm_1" not in resources


@pytest.mark.parametrize("resources,region", [
    (("arm_2", "arm_3"), "table.S"),
    (("arm_3", "arm_4"), "table.N"),
    (("arm_3", "arm_4"), "forbidden"),
])
def test_invalid_wave_never_reaches_backend(resources, region):
    graph, schedule = plan(south_resources=resources, south_region=region)
    backend = RecordingBackend()
    fleet = make_fleet()
    before = fleet.digest()
    with pytest.raises(FleetError):
        FleetExecutor(backend).execute(graph, schedule, world(), fleet)
    assert backend.waves == []
    assert fleet.digest() == before


@pytest.mark.parametrize("ids", [(), ("north", "north"), ("unknown",)])
def test_malformed_results_rejected(ids):
    class BadBackend:
        def execute_wave(self, commands, world):
            return tuple(ManipResult(i, True, {}) for i in ids)
    graph, schedule = plan(two=False)
    with pytest.raises(FleetError):
        FleetExecutor(BadBackend()).execute(graph, schedule, world(), make_fleet())


def test_fault_halts_before_later_wave_and_preserves_input():
    graph, schedule = plan(two=False)
    schedule.waves.append(Wave(1, schedule.waves[0].steps))
    class CountFault(FaultBackend):
        calls = 0
        def execute_wave(self, commands, world):
            self.calls += 1
            return super().execute_wave(commands, world)
    backend = CountFault()
    fleet = make_fleet()
    result = FleetExecutor(backend).execute(graph, schedule, world(), fleet)
    assert backend.calls == 1
    assert result.failed_wave == 0
    assert result.detail["requires_replan"]
    assert "arm_1" in fleet.online_ids
    assert fleet.leases == ()


def test_reported_failure_releases_authority_and_stops_later_waves():
    class FailedBackend(RecordingBackend):
        def execute_wave(self, commands, world):
            super().execute_wave(commands, world)
            return tuple(ManipResult(c.step_id, False, {"reason": "not reached"})
                         for c in commands)

    graph, schedule = plan(two=False)
    schedule.waves.append(Wave(1, schedule.waves[0].steps))
    backend = FailedBackend()
    result = FleetExecutor(backend).execute(graph, schedule, world(), make_fleet())
    assert not result.ok
    assert result.failed_wave == 0
    assert len(backend.waves) == 1
    assert len(result.waves) == 1
    assert not result.waves[0].ok
    assert result.waves[0].results[0].detail == {"reason": "not reached"}
    assert result.fleet.leases == ()
    assert result.fleet.reservations == ()


def test_successful_waves_release_resources_for_next_wave():
    graph, schedule = plan(two=False)
    schedule.waves.append(Wave(1, schedule.waves[0].steps))
    backend = RecordingBackend()
    result = FleetExecutor(backend).execute(graph, schedule, world(), make_fleet())
    assert result.ok
    assert len(backend.waves) == 2
    assert [r.wave_index for r in result.waves] == [0, 1]
    assert result.waves[0].leases != result.waves[1].leases
    assert result.fleet.leases == ()
    assert result.fleet.reservations == ()


def test_fault_provenance_uses_world_org_and_run_identity(monkeypatch):
    observed = []
    original = ManipulationFleet.reallocate

    def capture(self, fault, **kwargs):
        observed.append(fault)
        return original(self, fault, **kwargs)

    monkeypatch.setattr(ManipulationFleet, "reallocate", capture)
    graph, schedule = plan(two=False)
    for run_id in ("first-run", "second-run"):
        result = FleetExecutor(FaultBackend()).execute(
            graph, schedule, WorldState(0, {}, org_id="tenant-a"), make_fleet(),
            run_id=run_id)
        assert not result.ok
    assert [f.org_id for f in observed] == ["tenant-a", "tenant-a"]
    assert observed[0].fault_id != observed[1].fault_id


def test_preparation_is_given_the_complete_fleet_schedule_before_any_wave():
    seen = []

    class PreparingBackend(RecordingBackend):
        def prepare_run(self, resources, state):
            seen.append((resources, state.org_id, len(self.waves)))

    graph, schedule = plan()
    result = FleetExecutor(PreparingBackend()).execute(
        graph, schedule, WorldState(0, {}, org_id="tenant-a"), make_fleet())
    assert result.ok
    assert seen == [(("arm_1", "arm_2", "arm_3", "arm_4"), "tenant-a", 0)]
