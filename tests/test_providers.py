"""OQ-031 / OQ-034 — sponsor providers under one Omni graph."""

from __future__ import annotations

import pytest

from omni_q.contracts import Constraint, Step
from omni_q.fakes import RulePlanner
from omni_q.providers import (
    INTEL_PROVIDER,
    QUALCOMM_PROVIDER,
    NoDeviceForStep,
    Provider,
    ProviderRouter,
)
from omni_q.scheduler import ScheduledPlanner, schedule
from omni_q.world import MockWorld


def _scheduled_graph(goal: str = "inspect and correct the workspace"):
    world = MockWorld.sample().state()
    base = RulePlanner().plan(goal, world)
    return schedule(base, world).annotate(base), world


# ---------------------------------------------------------------------------
# routing (Intel track)
# ---------------------------------------------------------------------------


def test_arm_steps_route_to_the_matching_physical_arm():
    router = ProviderRouter([INTEL_PROVIDER])
    _g, world = _scheduled_graph()
    left = Step("s1", "manipulate", "PICK", args={"object": "x"}, arm="left")
    right = Step("s2", "manipulate", "MOVE", args={"object": "x", "to": "bin"}, arm="right")
    assert router.route(left, world) == "intel.arm_left"
    assert router.route(right, world) == "intel.arm_right"


def test_perception_and_reasoning_go_to_their_own_devices():
    router = ProviderRouter([INTEL_PROVIDER])
    _g, world = _scheduled_graph()
    assert router.route(Step("v", "verify", "VERIFY"), world) == "intel.perception"
    assert router.route(Step("o", "observe", "observe"), world) == "intel.perception"
    assert router.route(Step("l", "manipulate", "LOCATE"), world) == "intel.reason"


def test_router_assigns_complementary_work_not_one_serial_stream():
    # OQ-031: a routed plan touches several distinct devices, and verification
    # never shares a device with manipulation.
    router = ProviderRouter([INTEL_PROVIDER])
    graph, world = _scheduled_graph()
    routing = router.route_graph(graph, world)

    manip = {routing[s.id] for s in graph.steps if s.contract == "manipulate"}
    verify = {routing[s.id] for s in graph.steps if s.contract == "verify"}
    assert len(set(routing.values())) >= 3
    assert manip.isdisjoint(verify)
    assert {"intel.arm_left", "intel.arm_right"} <= manip


# ---------------------------------------------------------------------------
# OQ-034 — one graph, two tracks
# ---------------------------------------------------------------------------


def test_same_graph_runs_on_either_track():
    graph, world = _scheduled_graph()
    router = ProviderRouter([INTEL_PROVIDER, QUALCOMM_PROVIDER])  # qualcomm off by default

    intel_routing = router.route_graph(graph, world)
    assert all(router.provider_of(d) == "intel" for d in intel_routing.values())

    router.set_available("intel", False)
    router.set_available("qualcomm", True)
    qc_routing = router.route_graph(graph, world)          # SAME graph object
    assert all(router.provider_of(d) == "qualcomm" for d in qc_routing.values())
    assert set(intel_routing) == set(qc_routing)           # every step still placed


def test_keep_local_excludes_the_cloud_device():
    router = ProviderRouter([QUALCOMM_PROVIDER])
    router.set_available("qualcomm", True)
    graph, _w = _scheduled_graph()
    local_world = MockWorld(
        objects=list(MockWorld.sample()._objects.values()),
        constraints=(Constraint("keep_local", None, justification="op"),),
    ).state()
    routed = router.route_graph(graph, local_world)
    assert "qualcomm.cloud" not in routed.values()
    assert "UNROUTED" not in routed.values()


def test_no_available_provider_is_a_placement_failure():
    router = ProviderRouter([INTEL_PROVIDER])
    router.set_available("intel", False)
    _g, world = _scheduled_graph()
    with pytest.raises(NoDeviceForStep):
        router.route(Step("s", "manipulate", "PICK", args={"object": "x"}), world)
    graph, _w = _scheduled_graph()
    assert set(router.route_graph(graph, world).values()) == {"UNROUTED"}


def test_to_dict_carries_providers_routing_and_grouping():
    router = ProviderRouter([INTEL_PROVIDER, QUALCOMM_PROVIDER])
    graph, world = _scheduled_graph()
    d = router.to_dict(graph, world)
    assert [p["name"] for p in d["providers"]] == ["intel", "qualcomm"]
    assert d["routing"] and d["by_provider"].get("intel")


# ---------------------------------------------------------------------------
# engine integration — the full Intel stack
# ---------------------------------------------------------------------------


def test_intel_stack_runs_end_to_end_through_the_engine():
    from omni_q import build_mock_engine

    engine = build_mock_engine()
    engine.planner = ScheduledPlanner(RulePlanner())
    engine.device = ProviderRouter([INTEL_PROVIDER])
    receipt = engine.run("inspect and correct the workspace")

    assert receipt.metrics["resolved"] is True
    devices = {a["device"] for a in engine._actions if a["device"]}
    assert devices and all(d.startswith("intel.") for d in devices)
    assert {"intel.arm_left", "intel.arm_right"} <= devices


def test_engine_replans_when_a_track_goes_offline_mid_run():
    from omni_q import build_mock_engine

    engine = build_mock_engine()
    engine.planner = ScheduledPlanner(RulePlanner())
    router = ProviderRouter([INTEL_PROVIDER, QUALCOMM_PROVIDER])
    router.set_available("qualcomm", True)
    engine.device = router

    original = engine.manipulator.execute
    fired = {"n": 0}

    def hook(step, world):
        if fired["n"] == 0 and step.op == "PICK":
            fired["n"] = 1
            router.set_available("intel", False)     # Intel drops out mid-run
        return original(step, world)

    engine.manipulator.execute = hook  # type: ignore[method-assign]
    engine.run("inspect and correct the workspace")

    devices = {a["device"] for a in engine._actions if a["device"]}
    assert any(d.startswith("qualcomm.") for d in devices)   # work continued on the other track
