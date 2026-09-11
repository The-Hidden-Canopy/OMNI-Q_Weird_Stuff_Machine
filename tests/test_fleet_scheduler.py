"""Fleet-aware scheduler acceptance and fail-closed boundaries."""

from __future__ import annotations

import pytest

from omni_q.contracts import Detection, PlanGraph, Step, WorldState
from omni_q.fleet import FleetError, ManipulationFleet, ManipulatorSpec
from omni_q.scheduler import Region, schedule


CAPS = frozenset({"PICK", "MOVE", "PLACE", "CO_ROTATE"})
LAYOUT = {
    "north": Region("north", 0.30),
    "south": Region("south", -0.30),
    "center": Region("center", 0.0),
}


def _fleet(*, center: bool = False) -> ManipulationFleet:
    regions = frozenset({"north", "south", "center"} if center else {"north"})
    south_regions = frozenset({"south", "center"} if center else {"south"})
    return ManipulationFleet.from_specs(
        [
            ManipulatorSpec("arm_1", CAPS, regions),
            ManipulatorSpec("arm_2", CAPS, regions),
            ManipulatorSpec("arm_3", CAPS, south_regions),
            ManipulatorSpec("arm_4", CAPS, south_regions),
        ],
        preferred_pairs=(("arm_1", "arm_2"), ("arm_3", "arm_4")),
    )


def _world(*objects: Detection) -> WorldState:
    return WorldState(
        frame=1,
        objects={obj.object_id: obj for obj in objects},
        ownership={obj.object_id: None for obj in objects},
    )


def test_disjoint_biarm_units_share_a_wave_and_consume_only_their_pair():
    graph = PlanGraph(goal="coordinate two teams")
    graph.steps = [
        Step("co_north", "manipulate", "CO_ROTATE", args={"object": "plate_north"}),
        Step("co_south", "manipulate", "CO_ROTATE", args={"object": "plate_south"}),
        Step("verify", "verify", "VERIFY", deps=("co_north", "co_south")),
    ]
    world = _world(
        Detection("plate_north", "plate", zone="north", target_zone="north"),
        Detection("plate_south", "plate", zone="south", target_zone="south"),
    )

    planned = schedule(graph, world, fleet=_fleet(), layout=LAYOUT)

    assert planned.assignment["co_north"] == "arm_1"
    assert planned.assignment["co_south"] == "arm_3"
    assert planned.resource_assignment["co_north"] == ("arm_1", "arm_2")
    assert planned.resource_assignment["co_south"] == ("arm_3", "arm_4")
    assert planned.metrics["max_resource_parallelism"] == 4
    waves = {
        next(w.index for w in planned.waves if any(s.step_id == step_id for s in w.steps))
        for step_id in ("co_north", "co_south")
    }
    assert len(waves) == 1


def test_explicit_cross_group_pair_is_schedulable_resource():
    graph = PlanGraph(goal="cross-pair assist")
    graph.steps = [
        Step(
            "co_center", "manipulate", "CO_ROTATE",
            args={"object": "plate_center", "participants": ("arm_2", "arm_3")},
        ),
        Step("verify", "verify", "VERIFY", deps=("co_center",)),
    ]
    world = _world(Detection(
        "plate_center", "plate", zone="center", target_zone="center"))

    planned = schedule(graph, world, fleet=_fleet(center=True), layout=LAYOUT)

    assert planned.assignment["co_center"] == "arm_2"
    assert planned.resource_assignment["co_center"] == ("arm_2", "arm_3")
    assert planned.waves[0].steps[0].participants == ("arm_2", "arm_3")


def test_fleet_capability_gap_fails_closed_instead_of_using_another_arm():
    limited = ManipulationFleet.from_specs([
        ManipulatorSpec("arm_1", frozenset({"CO_ROTATE"}), frozenset({"center"})),
        ManipulatorSpec("arm_2", frozenset({"CO_ROTATE"}), frozenset({"center"})),
    ], preferred_pairs=(("arm_1", "arm_2"),))
    graph = PlanGraph(goal="pick")
    graph.steps = [Step("pick", "manipulate", "PICK", args={"object": "x"})]
    world = _world(Detection("x", "part", zone="center", target_zone="center"))

    with pytest.raises(FleetError, match="cannot perform PICK"):
        schedule(graph, world, fleet=limited, layout=LAYOUT)


def test_explicit_primary_arm_cannot_be_silently_replaced_by_another_pair():
    limited = ManipulationFleet.from_specs([
        ManipulatorSpec("arm_1", CAPS, frozenset({"center"})),
        ManipulatorSpec("arm_2", CAPS, frozenset({"center"})),
        ManipulatorSpec("arm_3", frozenset({"PICK"}), frozenset({"center"})),
        ManipulatorSpec("arm_4", frozenset({"PICK"}), frozenset({"center"})),
    ], preferred_pairs=(("arm_1", "arm_2"), ("arm_3", "arm_4")))
    graph = PlanGraph(goal="explicit pair")
    graph.steps = [
        Step(
            "co_center", "manipulate", "CO_ROTATE",
            args={"object": "plate_center"}, arm="arm_3",
        ),
        Step("verify", "verify", "VERIFY", deps=("co_center",)),
    ]
    world = _world(Detection(
        "plate_center", "plate", zone="center", target_zone="center"))

    with pytest.raises(FleetError, match="explicit primary arm arm_3"):
        schedule(graph, world, fleet=limited, layout=LAYOUT)
