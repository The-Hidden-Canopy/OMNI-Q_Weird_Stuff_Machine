"""OQ-012 / OQ-013 / OQ-044 — bimanual task scheduler."""

from __future__ import annotations

import pytest

from omni_q.contracts import Constraint, Detection, PlanGraph, Step, WorldState
from omni_q.fakes import RulePlanner
from omni_q.scheduler import DEFAULT_LAYOUT, Region, ScheduledPlanner, schedule
from omni_q.world import MockWorld


def _world(objects: list[Detection], constraints: tuple[Constraint, ...] = ()) -> WorldState:
    return WorldState(
        frame=1,
        objects={o.object_id: o for o in objects},
        constraints=constraints,
        ownership={o.object_id: None for o in objects},
    )


def _chain(oid: str) -> list[Step]:
    return [
        Step(f"pick_{oid}", "manipulate", "PICK", args={"object": oid}),
        Step(f"move_{oid}", "manipulate", "MOVE", args={"object": oid, "to": _TARGET[oid]},
             deps=(f"pick_{oid}",)),
    ]


_TARGET: dict[str, str] = {}


def _graph(*chains: list[Step]) -> PlanGraph:
    g = PlanGraph(goal="tidy the workspace")
    move_ids = []
    for steps in chains:
        g.steps.extend(steps)
        move_ids.extend(s.id for s in steps if s.op == "MOVE")
    g.steps.append(Step("verify_final", "verify", "VERIFY", deps=tuple(move_ids)))
    return g


# ---------------------------------------------------------------------------
# parallelism vs serialisation
# ---------------------------------------------------------------------------


def test_opposite_regions_run_both_arms_in_parallel():
    _TARGET.update(x="bin", y="setting_1")
    world = _world([
        Detection("x", "connector", zone="A", target_zone="bin"),      # right side
        Detection("y", "plate", zone="tray", target_zone="setting_1"),  # left side
    ])
    sch = schedule(_graph(_chain("x"), _chain("y")), world)

    assert sch.metrics["max_parallelism"] == 2
    assert sch.assignment["pick_x"] == "right"
    assert sch.assignment["pick_y"] == "left"
    assert sch.metrics["serialized_conflicts"] == 0
    # picks share a wave, moves share a wave
    assert _wave_of(sch, "pick_x") == _wave_of(sch, "pick_y")
    assert _wave_of(sch, "move_x") == _wave_of(sch, "move_y")


def test_same_centre_region_is_serialised_with_a_workspace_barrier():
    _TARGET.update(p="center", q="center")
    world = _world([
        Detection("p", "cup", zone="drawer", target_zone="center"),
        Detection("q", "fork", zone="drawer", target_zone="center"),
    ])
    sch = schedule(_graph(_chain("p"), _chain("q")), world)

    assert sch.metrics["max_parallelism"] == 1
    assert sch.metrics["serialized_conflicts"] >= 1
    assert any(b.kind == "workspace" for b in sch.barriers)
    # even though the two picks were handed to different arms, they never coexist
    assert _wave_of(sch, "pick_p") != _wave_of(sch, "pick_q")


# ---------------------------------------------------------------------------
# arm assignment (OQ-044)
# ---------------------------------------------------------------------------


def test_role_follows_reach_not_task_order():
    _TARGET.update(a="tray", b="bin")
    world = _world([
        Detection("a", "plate", zone="B", target_zone="tray"),   # left-only
        Detection("b", "bolt", zone="A", target_zone="bin"),      # right-only
    ])
    sch = schedule(_graph(_chain("a"), _chain("b")), world)
    # first-declared object is NOT forced onto "left"
    assert sch.assignment["pick_a"] == "left"
    assert sch.assignment["pick_b"] == "right"


def test_explicit_step_arm_is_honoured_even_when_unreachable():
    _TARGET.update(z="bin")
    world = _world([Detection("z", "part", zone="A", target_zone="bin")])  # right-only
    steps = _chain("z")
    steps[0].arm = "left"
    steps[1].arm = "left"
    sch = schedule(_graph(steps), world)
    assert sch.assignment["pick_z"] == "left"
    assert any(b.kind == "reach" for b in sch.barriers)


def test_prefer_arm_constraint_wins_when_reachable():
    _TARGET.update(c="center")
    world = _world(
        [Detection("c", "mug", zone="drawer", target_zone="center")],
        constraints=(Constraint("prefer_arm", "left", justification="operator said so"),),
    )
    sch = schedule(_graph(_chain("c")), world)
    assert sch.assignment["pick_c"] == "left"


# ---------------------------------------------------------------------------
# gripper occupancy & barriers (OQ-013)
# ---------------------------------------------------------------------------


def test_an_arm_cannot_pick_a_second_object_while_holding_one():
    _TARGET.update(m="bin", n="bin")
    world = _world([
        Detection("m", "clip", zone="A", target_zone="bin"),          # right
        Detection("n", "washer", zone="setting_2", target_zone="bin"),  # right
    ])
    sch = schedule(_graph(_chain("m"), _chain("n")), world)
    assert sch.assignment["pick_m"] == "right" and sch.assignment["pick_n"] == "right"
    # the two picks are never in the same wave (one hand, one object at a time)
    assert _wave_of(sch, "pick_m") != _wave_of(sch, "pick_n")
    # and a pick never precedes its own move on the same arm within a wave
    for w in sch.waves:
        arms_used = [ss.arm for ss in w.steps if ss.arm]
        assert len(arms_used) == len(set(arms_used))


def test_verify_step_gets_its_own_trailing_wave():
    _TARGET.update(x="bin", y="setting_1")
    world = _world([
        Detection("x", "connector", zone="A", target_zone="bin"),
        Detection("y", "plate", zone="tray", target_zone="setting_1"),
    ])
    sch = schedule(_graph(_chain("x"), _chain("y")), world)
    last = sch.waves[-1]
    assert [ss.step_id for ss in last.steps] == ["verify_final"]
    assert last.steps[0].arm is None
    assert any(b.kind == "verify" for b in sch.barriers)


def test_no_wave_holds_two_conflicting_regions():
    world = MockWorld.sample().state()
    sch = schedule(RulePlanner().plan("tidy the workspace", world), world)
    layout = DEFAULT_LAYOUT
    for w in sch.waves:
        regs = [layout.get(ss.region_id, Region(ss.region_id or "center", 0.0))
                for ss in w.steps if ss.region_id]
        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                assert not regs[i].conflicts_with(regs[j], 0.15)


# ---------------------------------------------------------------------------
# hand-off
# ---------------------------------------------------------------------------


def test_handoff_crosses_arms_without_a_false_workspace_barrier():
    world = MockWorld.sample().state()
    g = PlanGraph(goal="spin and place")
    g.steps = [
        Step("pick_p", "manipulate", "PICK", args={"object": "plate_1"}),
        Step("handoff_p", "manipulate", "HANDOFF",
             args={"object": "plate_1", "to_actor": "right"}, deps=("pick_p",)),
        Step("place_p", "manipulate", "PLACE",
             args={"object": "plate_1", "to": "A"}, deps=("handoff_p",)),
        Step("verify_final", "verify", "VERIFY", deps=("place_p",)),
    ]
    sch = schedule(g, world)
    assert sch.metrics["handoffs"] == 1
    assert sch.assignment["pick_p"] == "left"
    assert sch.assignment["place_p"] == "right"
    assert "pick_p" in sch.arm_timeline["left"]
    assert "place_p" in sch.arm_timeline["right"]
    assert not any(b.kind == "workspace" for b in sch.barriers)


# ---------------------------------------------------------------------------
# annotate() integration seam
# ---------------------------------------------------------------------------


def test_annotate_fills_arms_and_keeps_a_valid_dag():
    world = MockWorld.sample().state()
    graph = RulePlanner().plan("tidy the workspace", world)
    annotated = schedule(graph, world).annotate(graph)

    for s in annotated.steps:
        if s.contract == "manipulate":
            assert s.arm in ("left", "right")
    # still a DAG
    annotated.topo_order()
    # a later-wave step now also depends on the previous wave
    move = annotated.by_id("move_connector_2")
    assert "pick_connector_2" in move.deps


def test_annotated_graph_runs_through_the_engine():
    from omni_q import build_mock_engine

    engine = build_mock_engine()
    world = engine.world.state()
    graph = RulePlanner().plan("inspect and correct the workspace", world)
    annotated = schedule(graph, world).annotate(graph)

    class _FixedPlanner(RulePlanner):
        def plan(self, goal, world):  # noqa: D401 - test stub
            super().plan(goal, world)
            return annotated

    engine.planner = _FixedPlanner()
    receipt = engine.run("inspect and correct the workspace")
    assert receipt.metrics["resolved"] is True
    arms = {a["arm"] for a in engine._actions if a["op"] in {"PICK", "MOVE"}}
    assert arms <= {"left", "right"} and arms


# ---------------------------------------------------------------------------
# style constraints / flourishes (OQ-015)
# ---------------------------------------------------------------------------


def _present(oid: str) -> Step:
    return Step(f"present_{oid}", "manipulate", "PRESENT",
               args={"object": oid}, deps=(f"move_{oid}",))


def test_no_style_means_no_flourish_and_no_extra_wave():
    _TARGET.update(x="bin", y="setting_1")
    world = _world([
        Detection("x", "connector", zone="A", target_zone="bin"),
        Detection("y", "plate", zone="tray", target_zone="setting_1"),
    ])
    sch = schedule(_graph(_chain("x"), _chain("y")), world)
    assert sch.metrics["flourishes_scheduled"] == 0
    assert sch.metrics["flourish_waves_added"] == 0


def test_show_off_adds_one_flourish_wave_before_verify():
    _TARGET.update(x="bin", y="setting_1")
    world = _world([
        Detection("x", "connector", zone="A", target_zone="bin"),
        Detection("y", "plate", zone="tray", target_zone="setting_1"),
    ])
    g = _graph(_chain("x") + [_present("x")], _chain("y") + [_present("y")])
    sch = schedule(g, world)

    assert sch.metrics["flourishes_scheduled"] == 2
    assert sch.metrics["flourishes_dropped"] == 0
    assert sch.metrics["flourish_waves_added"] == 1
    assert sch.metrics["max_parallelism"] == 2          # goal path unchanged
    assert [ss.step_id for ss in sch.waves[-1].steps] == ["verify_final"]
    flourish_wave = sch.waves[-2]
    assert all(ss.flourish for ss in flourish_wave.steps)


def test_flourish_is_never_a_dependency_of_verify():
    _TARGET.update(x="bin")
    world = _world([Detection("x", "connector", zone="A", target_zone="bin")])
    g = _graph(_chain("x") + [_present("x")])
    annotated = schedule(g, world).annotate(g)
    verify = annotated.by_id("verify_final")
    assert not any(d.startswith("present_") for d in verify.deps)


def test_flourish_keeps_its_object_chain_arm():
    _TARGET.update(x="bin")
    world = _world([Detection("x", "connector", zone="A", target_zone="bin")])  # right
    g = _graph(_chain("x") + [_present("x")])
    sch = schedule(g, world)
    assert sch.assignment["present_x"] == sch.assignment["move_x"] == "right"


def test_flourish_dropped_when_no_slack_and_wave_disallowed():
    _TARGET.update(x="bin", y="setting_1")
    world = _world([
        Detection("x", "connector", zone="A", target_zone="bin"),
        Detection("y", "plate", zone="tray", target_zone="setting_1"),
    ])
    g = _graph(_chain("x") + [_present("x")], _chain("y") + [_present("y")])
    sch = schedule(g, world, allow_flourish_wave=False)

    assert sch.metrics["flourishes_dropped"] == 2
    assert set(sch.dropped) == {"present_x", "present_y"}
    assert any(b.kind == "flourish" for b in sch.barriers)
    annotated = sch.annotate(g)
    assert not any(s.id.startswith("present_") for s in annotated.steps)


def test_flourish_slots_into_an_existing_slack_wave():
    # y's pick waits on x's move, so the wave that runs pick_y leaves the other
    # arm idle -> present_x fills it, no new wave.
    _TARGET.update(x="bin", y="tray")
    world = _world([
        Detection("x", "connector", zone="A", target_zone="bin"),   # right side
        Detection("y", "plate", zone="B", target_zone="tray"),       # left side
    ])
    g = PlanGraph(goal="tidy")
    g.steps = [
        Step("pick_x", "manipulate", "PICK", args={"object": "x"}),
        Step("move_x", "manipulate", "MOVE", args={"object": "x", "to": "bin"}, deps=("pick_x",)),
        Step("present_x", "manipulate", "PRESENT", args={"object": "x"}, deps=("move_x",)),
        Step("pick_y", "manipulate", "PICK", args={"object": "y"}, deps=("move_x",)),
        Step("move_y", "manipulate", "MOVE", args={"object": "y", "to": "tray"}, deps=("pick_y",)),
        Step("verify_final", "verify", "VERIFY", deps=("move_x", "move_y")),
    ]
    sch = schedule(g, world)
    assert sch.metrics["flourish_waves_added"] == 0
    assert sch.metrics["flourishes_scheduled"] == 1
    assert _wave_of(sch, "present_x") == _wave_of(sch, "pick_y")


# ---------------------------------------------------------------------------
# dance-while-working (speech -> style mode -> scheduler)
# ---------------------------------------------------------------------------


def _sequential_graph() -> PlanGraph:
    # y's pick waits on x's move -> every wave has exactly one arm working
    g = PlanGraph(goal="tidy")
    g.steps = [
        Step("pick_x", "manipulate", "PICK", args={"object": "x"}),
        Step("move_x", "manipulate", "MOVE", args={"object": "x", "to": "bin"}, deps=("pick_x",)),
        Step("pick_y", "manipulate", "PICK", args={"object": "y"}, deps=("move_x",)),
        Step("move_y", "manipulate", "MOVE", args={"object": "y", "to": "tray"}, deps=("pick_y",)),
        Step("verify_final", "verify", "VERIFY", deps=("move_x", "move_y")),
    ]
    return g


def _seq_world(*styles: str) -> WorldState:
    _TARGET.clear()
    return WorldState(
        frame=1,
        objects={
            "x": Detection("x", "connector", zone="A", target_zone="bin"),
            "y": Detection("y", "plate", zone="B", target_zone="tray"),
        },
        constraints=tuple(Constraint("style", s, justification="spoken") for s in styles),
        ownership={"x": None, "y": None},
    )


def test_dance_mode_fills_idle_arm_slack_without_delaying_the_goal():
    g = _sequential_graph()
    plain = schedule(g, _seq_world())
    danced = schedule(g, _seq_world("dance"))

    assert danced.style_mode == "dance"
    assert danced.metrics["idle_flourishes"] > 0
    # the goal path is untouched: same wave count, same real-work order
    assert danced.metrics["waves"] == plain.metrics["waves"]
    real = lambda s: [ss.step_id for w in s.waves for ss in w.steps if not ss.flourish]
    assert real(danced) == real(plain)
    # every injected step is a non-blocking flourish on an otherwise-idle arm
    for w in danced.waves:
        busy_real = {ss.arm for ss in w.steps if not ss.flourish and ss.arm}
        for ss in w.steps:
            if ss.flourish:
                assert ss.arm not in busy_real


def test_synchronized_mode_uses_one_primitive():
    danced = schedule(_sequential_graph(), _seq_world("show_off", "together"))
    assert danced.style_mode == "synchronized"
    injected = [ss.step_id for w in danced.waves for ss in w.steps
                if ss.flourish and ss.step_id.startswith("idle_")]
    assert injected and all("sway" in i for i in injected)


def test_last_style_constraint_wins_and_minimum_time_strips_flourishes():
    _TARGET.update(x="bin")
    world = WorldState(
        frame=1,
        objects={"x": Detection("x", "connector", zone="A", target_zone="bin")},
        constraints=(Constraint("style", "show_off", justification="a"),
                     Constraint("style", "back to work", justification="b")),
        ownership={"x": None},
    )
    g = _graph(_chain("x") + [_present("x")])
    sch = schedule(g, world)
    assert sch.style_mode == "minimum_time"
    assert sch.metrics["flourishes_scheduled"] == 0
    assert sch.metrics["flourish_waves_added"] == 0
    assert "present_x" in sch.dropped


def test_execute_flourishes_emits_runnable_non_blocking_leaf_steps():
    # OQ-HAND-011: annotate(..., execute_flourishes=True) turns the
    # scheduler-invented idle-slack flourishes into real Steps the engine runs.
    from omni_q.fakes import FakeManipulator

    g = _sequential_graph()
    sch = schedule(g, _seq_world("dance"))
    assert sch.metrics["idle_flourishes"] > 0

    display = sch.annotate(g)                       # default: display-only
    execute = sch.annotate(g, execute_flourishes=True)
    assert len(execute.steps) == len(display.steps) + sch.metrics["idle_flourishes"]

    extra = [s for s in execute.steps if s.id.startswith("idle_")]
    assert extra and all(s.contract == "manipulate" and s.op and s.arm for s in extra)
    # every emitted op is one the manipulator can actually run
    fm = FakeManipulator(None)
    assert all(fm.supports(s.op) for s in extra)
    # nothing depends on a flourish -> the goal path / wave count is untouched
    all_deps = {d for s in execute.steps for d in s.deps}
    assert not any(s.id in all_deps for s in extra)
    # the graph is still acyclic
    ordered = execute.topo_order()
    assert {getattr(s, "id", s) for s in ordered} == {s.id for s in execute.steps}


def test_bimanual_op_takes_both_arms_in_its_wave():
    g = PlanGraph(goal="co-rotate demo")
    g.steps = [
        Step("pick_p", "manipulate", "PICK", args={"object": "plate_1"}),
        Step("co", "manipulate", "CO_ROTATE", args={"object": "plate_1", "degrees": 140},
             deps=("pick_p",)),
        Step("place_p", "manipulate", "PLACE", args={"object": "plate_1", "to": "A"}, deps=("co",)),
        Step("verify_final", "verify", "VERIFY", deps=("place_p",)),
    ]
    sch = schedule(g, MockWorld.sample().state())
    co_wave = next(w for w in sch.waves if any(ss.step_id == "co" for ss in w.steps))
    assert len(co_wave.steps) == 1        # nothing else runs alongside a bimanual op


# ---------------------------------------------------------------------------
# ScheduledPlanner — live integration
# ---------------------------------------------------------------------------


def test_scheduled_planner_drives_the_engine_with_arm_assignments():
    from omni_q import build_mock_engine

    engine = build_mock_engine()
    engine.planner = ScheduledPlanner(RulePlanner())
    receipt = engine.run("inspect and correct the workspace")

    assert receipt.metrics["resolved"] is True
    arms = {a["arm"] for a in engine._actions if a["op"] in {"PICK", "MOVE"}}
    assert arms and arms <= {"left", "right"}
    assert engine.planner.last_schedule is not None
    assert engine.planner.last_schedule.waves


def test_scheduled_planner_reschedules_on_a_mid_run_constraint():
    from omni_q import build_mock_engine
    from omni_q.mutation import RuntimeMutator

    engine = build_mock_engine()
    engine.planner = ScheduledPlanner(RulePlanner())
    mut = RuntimeMutator(engine)
    original = engine.manipulator.execute
    fired = {"n": 0}

    def hook(step, world):
        if fired["n"] == 0 and step.op == "PICK":
            fired["n"] = 1
            mut.apply("don't touch the plate")
        return original(step, world)

    engine.manipulator.execute = hook  # type: ignore[method-assign]
    engine.run("inspect and correct the workspace")

    assert "graph.recompiled" in [e.kind for e in engine.bus.log]
    assert engine.planner.last_schedule is not None
    moved = [a["result"].get("moved") for a in engine._actions if a["op"] == "MOVE"]
    assert "plate_1" not in moved


def test_scheduled_planner_degrades_when_scheduling_raises(monkeypatch):
    import omni_q.scheduler as sched_mod

    world = MockWorld.sample().state()
    inner = RulePlanner()
    sp = ScheduledPlanner(inner)

    def boom(*a, **k):
        raise RuntimeError("scheduler exploded")

    monkeypatch.setattr(sched_mod, "schedule", boom)
    out = sp.plan("inspect and correct the workspace", world)

    assert sp.last_schedule is None
    assert "scheduler exploded" in sp.last_error
    # falls back to the inner planner's graph, unmodified
    assert [s.id for s in out.steps] == [s.id for s in inner.plan(
        "inspect and correct the workspace", world).steps]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _wave_of(sch, step_id: str) -> int:
    for w in sch.waves:
        if any(ss.step_id == step_id for ss in w.steps):
            return w.index
    raise AssertionError(f"{step_id} not scheduled")
