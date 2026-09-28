"""OQ-HAND-006 + OQ-025 executable — graph rewrites."""

from __future__ import annotations

from omni_q.contracts import Constraint, Detection, PlanGraph, Step, WorldState
from omni_q.fakes import RulePlanner
from omni_q.rewrite import RewritingPlanner, _same_surface, apply_mutation, optimize
from omni_q.scheduler import DEFAULT_LAYOUT
from omni_q.world import MockWorld

_ALL = lambda _op: True  # noqa: E731


def _world(objs, ownership=None, constraints=()):
    return WorldState(
        frame=1, objects={o.object_id: o for o in objs}, constraints=constraints,
        ownership=ownership or {o.object_id: None for o in objs},
    )


def _chain_graph(oid, to):
    g = PlanGraph(goal="tidy")
    g.steps = [
        Step(f"pick_{oid}", "manipulate", "PICK", args={"object": oid}),
        Step(f"move_{oid}", "manipulate", "MOVE", args={"object": oid, "to": to},
             deps=(f"pick_{oid}",)),
        Step("verify_final", "verify", "VERIFY", deps=(f"move_{oid}",)),
    ]
    return g


# -- OQ-HAND-006: cheaper op when the move is small -------------------------


def test_same_surface_move_becomes_slide():
    w = _world([Detection("connector_2", "connector", zone="A", target_zone="bin")])
    g = _chain_graph("connector_2", "bin")            # A(-0.25) -> bin(-0.30): same side
    out, rw = optimize(g, w, can_run=_ALL)
    ops = [s.op for s in out.steps]
    assert "SLIDE" in ops and "PICK" not in ops and "MOVE" not in ops
    assert rw[0].kind == "slide"
    out.topo_order()                                  # still a valid DAG
    assert "slide_connector_2" in out.by_id("verify_final").deps


def test_cross_table_move_keeps_pick_and_place():
    w = _world([Detection("p", "plate", zone="A", target_zone="tray")])
    g = _chain_graph("p", "tray")                     # A(-0.25) -> tray(+0.30): needs a lift
    out, rw = optimize(g, w, can_run=_ALL)
    assert rw == []
    assert [s.op for s in out.steps] == ["PICK", "MOVE", "VERIFY"]


def test_move_to_current_zone_becomes_nudge():
    w = _world([Detection("f", "fork", zone="left", target_zone="left")])
    g = _chain_graph("f", "left")
    out, rw = optimize(g, w, can_run=_ALL)
    assert rw[0].kind == "nudge" and "NUDGE" in [s.op for s in out.steps]


def test_held_object_is_not_rewritten():
    w = _world([Detection("c", "cup", zone="A", target_zone="bin")],
               ownership={"c": "left"})
    g = _chain_graph("c", "bin")
    out, rw = optimize(g, w, can_run=_ALL)
    assert rw == []


def test_can_run_gate_skips_unsupported_ops():
    w = _world([Detection("connector_2", "connector", zone="A", target_zone="bin")])
    g = _chain_graph("connector_2", "bin")
    out, rw = optimize(g, w, can_run=lambda op: op in {"PICK", "MOVE", "VERIFY"})
    assert rw == [] and [s.op for s in out.steps] == ["PICK", "MOVE", "VERIFY"]


def test_optimize_on_the_real_planner_graph():
    w = MockWorld.sample().state()
    g = RulePlanner().plan("inspect and correct the workspace", w)
    out, rw = optimize(g, w, can_run=_ALL)
    assert {r.kind for r in rw} <= {"slide", "nudge"}
    assert any(r.kind == "slide" for r in rw)
    out.topo_order()


# -- OQ-025 executable: spin / spin_on_place / nudge ----------------------


def test_spin_inserts_spin_before_the_terminal_move():
    w = MockWorld.sample().state()
    g = RulePlanner().plan("inspect and correct the workspace", w)
    out, rw = apply_mutation(g, w, "spin", {"object": "plate_1", "degrees": 180}, can_run=_ALL)
    spin = out.by_id("spin_plate_1")
    assert spin.op == "SPIN" and spin.args["degrees"] == 180
    assert "spin_plate_1" in out.by_id("move_plate_1").deps
    out.topo_order()


def test_spin_on_place_hits_every_matching_class():
    w = _world([
        Detection("plate_1", "plate", zone="tray", target_zone="setting_1"),
        Detection("plate_2", "plate", zone="tray", target_zone="setting_2"),
        Detection("cup_1", "cup", zone="tray", target_zone="upper_right"),
    ])
    g = PlanGraph(goal="set the table")
    for oid in ("plate_1", "plate_2", "cup_1"):
        g.steps += [
            Step(f"pick_{oid}", "manipulate", "PICK", args={"object": oid}),
            Step(f"move_{oid}", "manipulate", "MOVE",
                 args={"object": oid, "to": w.objects[oid].target_zone}, deps=(f"pick_{oid}",)),
        ]
    g.steps.append(Step("verify_final", "verify", "VERIFY",
                        deps=("move_plate_1", "move_plate_2", "move_cup_1")))
    out, rw = apply_mutation(g, w, "spin_on_place", {"object_class": "plate"}, can_run=_ALL)
    spun = {r.object for r in rw}
    assert spun == {"plate_1", "plate_2"}            # not cup_1


def test_nudge_with_a_chain_collapses_to_nudge():
    w = MockWorld.sample().state()
    g = RulePlanner().plan("inspect and correct the workspace", w)
    out, rw = apply_mutation(g, w, "nudge",
                             {"object": "connector_2", "direction": "left", "amount": "a bit"},
                             can_run=_ALL)
    n = out.by_id("nudge_connector_2")
    assert n.op == "NUDGE" and n.args["direction"] == "left"
    assert not any(s.id == "pick_connector_2" for s in out.steps)


def test_nudge_without_a_chain_adds_a_standalone_step_before_verify():
    w = MockWorld.sample().state()
    g = RulePlanner().plan("inspect and correct the workspace", w)
    out, rw = apply_mutation(g, w, "nudge",
                             {"object": "sleeve_1", "direction": "right", "amount": "slightly"},
                             can_run=_ALL)
    assert "nudge_sleeve_1" in out.by_id("verify_final").deps
    out.topo_order()


# -- RewritingPlanner + RuntimeMutator ----------------------------------


def test_rewriting_planner_gates_to_the_manipulator_and_still_resolves():
    from omni_q import build_mock_engine

    engine = build_mock_engine()
    engine.planner = RewritingPlanner(RulePlanner(), can_run=engine.manipulator.supports)
    receipt = engine.run("inspect and correct the workspace")
    # FakeManipulator has no SLIDE -> rewrites gated off -> normal resolution
    assert engine.planner.last_rewrites == []
    assert receipt.metrics["resolved"] is True


def test_rewriting_planner_degrades_on_error(monkeypatch):
    import omni_q.rewrite as rw_mod

    sp = RewritingPlanner(RulePlanner())

    def boom(*a, **k):
        raise RuntimeError("rewrite exploded")

    monkeypatch.setattr(rw_mod, "optimize", boom)
    out = sp.plan("inspect and correct the workspace", MockWorld.sample().state())
    assert sp.last_error and "exploded" in sp.last_error
    assert out.steps          # inner graph returned


def test_runtime_mutator_routes_spin_to_the_planner_when_it_can_run():
    from omni_q import build_mock_engine
    from omni_q.mutation import RuntimeMutator

    engine = build_mock_engine()
    engine.planner = RewritingPlanner(RulePlanner(), can_run=lambda op: True)
    res = RuntimeMutator(engine, planner=engine.planner).apply("spin that plate")

    assert ("rewrite:spin", "plate_1") in res.applied
    assert not res.deferred
    # the registered mutation shows up as a rewrite on the next plan
    engine.planner.plan("inspect and correct the workspace", engine.world.state())
    assert any(r.kind == "spin" for r in engine.planner.last_rewrites)


def test_runtime_mutator_defers_when_manipulator_cannot_run_the_op():
    from omni_q import build_mock_engine
    from omni_q.mutation import RuntimeMutator

    engine = build_mock_engine()
    engine.planner = RewritingPlanner(RulePlanner())     # gate = engine.manipulator.supports
    res = RuntimeMutator(engine, planner=engine.planner).apply("spin that plate")
    assert not res.applied
    assert res.deferred and "can't run SPIN" in res.deferred[0][2]


def test_same_surface_helper():
    assert _same_surface("A", "bin", DEFAULT_LAYOUT)          # both right
    assert _same_surface("tray", "setting_1", DEFAULT_LAYOUT)  # both left
    assert not _same_surface("A", "tray", DEFAULT_LAYOUT)      # cross table
