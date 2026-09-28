"""Reactive control — ontology attention drives halt / re-observe / resume."""

from __future__ import annotations

from omni_q.contracts import Observation, Observe, Plan
from omni_q.ontology import OntologyObserver, stub_swarm
from omni_q.reactor import ReactiveObserver, ReactivePlanner, wait_graph
from omni_q.world import MockWorld


class _Obs:
    """Minimal Observe stub whose workspace_clear the test flips."""

    def __init__(self):
        self.clear = True
        self.ontology = None

    def observe(self, world):
        return Observation(frame=world.frame, detections=(),
                           workspace_clear=self.clear, raw_ref=None)


def test_reactive_observer_satisfies_observe():
    assert isinstance(ReactiveObserver(_Obs()), Observe)


def test_halt_then_resume_events():
    inner = _Obs()
    ro = ReactiveObserver(inner)
    w = MockWorld.sample()

    ro.observe(w.state())
    assert not ro.blocked and ro.events == []

    inner.clear = False                       # a hand enters
    ro.observe(w.state())
    assert ro.blocked and ro.events[-1].kind == "HALT"

    ro.observe(w.state())                     # still blocked -> no duplicate event
    assert [e.kind for e in ro.events] == ["HALT"]

    inner.clear = True                        # workspace clears
    ro.observe(w.state())
    assert not ro.blocked and ro.events[-1].kind == "RESUME"


def test_observer_without_engine_never_crashes():
    ro = ReactiveObserver(_Obs())            # engine=None
    ro.inner.clear = False
    ro.observe(MockWorld.sample().state())   # must not raise
    assert ro.blocked


def test_reactive_observer_queues_a_recompile_on_the_engine():
    class _Eng:
        def __init__(self):
            self.constraints = []

        def add_constraint(self, kind, value=None, *, justification=None):
            self.constraints.append((kind, value))

    eng = _Eng()
    ro = ReactiveObserver(_Obs(), engine=eng)
    ro.inner.clear = False
    ro.observe(MockWorld.sample().state())
    assert ("style", "freeze") in eng.constraints
    ro.inner.clear = True
    ro.observe(MockWorld.sample().state())
    assert ("style", "minimum_time") in eng.constraints


# -- planner --------------------------------------------------------


def test_reactive_planner_holds_while_blocked():
    from omni_q.fakes import RulePlanner

    ro = ReactiveObserver(_Obs())
    rp = ReactivePlanner(RulePlanner(), observer=ro)
    assert isinstance(rp, Plan)
    w = MockWorld.sample().state()

    g = rp.plan("inspect and correct the workspace", w)
    assert any(s.op == "PICK" for s in g.steps)      # real plan

    ro.blocked = True
    held = rp.plan("inspect and correct the workspace", w)
    assert [s.op for s in held.steps] == ["STABILIZE", "VERIFY"]
    assert rp.holding

    ro.blocked = False
    assert any(s.op == "PICK" for s in rp.replan(held, w, "resume").steps)


def test_wait_graph_shape():
    g = wait_graph("x")
    assert [s.op for s in g.steps] == ["STABILIZE", "VERIFY"]
    g.topo_order()


def test_reactive_planner_forwards_decorator_hooks():
    class _Inner:
        def __init__(self):
            self.registered = []
            self.can_run = None

        def register_mutation(self, k, p):
            self.registered.append((k, p))

        def plan(self, g, w):
            return None

        def replan(self, c, w, r):
            return None

    inner = _Inner()
    rp = ReactivePlanner(inner, observer=ReactiveObserver(_Obs()))
    rp.register_mutation("spin", {"object": "plate_1"})
    rp.can_run = lambda op: True
    assert inner.registered == [("spin", {"object": "plate_1"})]
    assert inner.can_run("SPIN")


# -- end to end ----------------------------------------------------


def test_engine_halts_for_a_hand_and_resumes_to_finish():
    from omni_q import build_mock_engine
    from omni_q.fakes import RulePlanner

    engine = build_mock_engine()
    engine.world.hand_xy = None
    ro = ReactiveObserver(OntologyObserver(stub_swarm(engine.world)), engine=engine)
    engine.observer = ro
    engine.planner = ReactivePlanner(RulePlanner(), observer=ro)
    engine.max_revisions = 40

    original = engine.manipulator.execute
    seq = {"n": 0}

    def hook(step, world):
        seq["n"] += 1
        if seq["n"] == 2:
            engine.world.hand_xy = (0.30, 0.50)          # hand into the left-arm zone
        if sum(1 for a in engine._actions if a["op"] == "STABILIZE") >= 3:
            engine.world.hand_xy = None                  # hand leaves
        return original(step, world)

    engine.manipulator.execute = hook
    receipt = engine.run("inspect and correct the workspace")

    kinds = [e.kind for e in ro.events]
    assert kinds == ["HALT", "RESUME"]
    assert "STABILIZE" in [a["op"] for a in engine._actions]     # it held
    assert receipt.metrics["resolved"] is True                    # then finished
