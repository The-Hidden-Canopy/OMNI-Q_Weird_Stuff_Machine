"""OQ-001 acceptance: interfaces compile and fake capabilities execute end-to-end."""

from __future__ import annotations

import omni_q
from omni_q import build_mock_engine
from omni_q.contracts import CONTRACTS, Observe, Plan, Manipulate, Verify, Device, Receipt, World
from omni_q.devices import default_devices
from omni_q.fakes import FakeManipulator, FakeObserver, FakeRecorder, FakeVerifier, RulePlanner
from omni_q.world import MockWorld


def test_import_surface():
    assert omni_q.__version__
    assert set(CONTRACTS) == {"World", "Observe", "Plan", "Manipulate", "Verify", "Device", "Receipt"}


def test_fakes_satisfy_contracts():
    world = MockWorld.sample()
    assert isinstance(FakeObserver(), Observe)
    assert isinstance(world, World)
    assert isinstance(RulePlanner(), Plan)
    assert isinstance(FakeManipulator(world), Manipulate)
    assert isinstance(FakeVerifier(), Verify)
    assert isinstance(default_devices(), Device)
    assert isinstance(FakeRecorder(), Receipt)


def test_end_to_end_normal_resolves_and_receipts():
    engine = build_mock_engine()
    receipt = engine.run("inspect and correct the workspace")

    assert receipt.metrics["resolved"] is True
    assert receipt.metrics["steps_executed"] > 0
    assert set(receipt.hashes) == {"inputs", "plan", "actions"}
    assert all(len(h) == 64 for h in receipt.hashes.values())
    # nothing left misplaced in the world
    assert engine.world.state().misplaced() == []


def test_constraint_change_recompiles():
    engine = build_mock_engine()
    engine.add_constraint("keep_local", justification="test operator command")
    engine.device.set_online("arduino.left_arm", False)
    engine.add_constraint("prefer_arm", "left", justification="test operator preference")
    receipt = engine.run("inspect and correct the workspace")

    kinds = [e.kind for e in engine.bus.log]
    assert "constraint.added" in kinds
    assert "graph.recompiled" in kinds
    assert receipt.metrics["revisions"] >= 1
    assert receipt.metrics["resolved"] is True


def test_world_change_forces_replan_then_resolves():
    world = MockWorld.sample()
    from omni_q.engine import OmniQ

    engine = OmniQ(
        world=world,
        observer=FakeObserver(),
        planner=RulePlanner(),
        manipulator=FakeManipulator(world),
        verifier=FakeVerifier(),
        device=default_devices(),
        recorder=FakeRecorder(),
    )

    original = engine.manipulator.execute
    fired = {"n": 0}

    def perturbing(step, wstate):
        res = original(step, wstate)
        if step.op == "MOVE" and fired["n"] == 0:
            fired["n"] = 1
            world.perturb("connector_2", "floor")
        return res

    engine.manipulator.execute = perturbing  # type: ignore[method-assign]
    receipt = engine.run("inspect and correct the workspace")

    assert receipt.metrics["revisions"] >= 1
    assert receipt.metrics["resolved"] is True
    assert "graph.recompiled" in [e.kind for e in engine.bus.log]


def test_forbidden_object_is_left_alone():
    engine = build_mock_engine()
    engine.add_constraint("forbid_object", "connector_2", justification="test boundary")
    engine.run("inspect and correct the workspace")

    moved = [a["result"].get("moved") for a in engine._actions if a["op"] == "MOVE"]
    assert "connector_2" not in moved
