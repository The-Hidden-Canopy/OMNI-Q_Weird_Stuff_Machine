"""OQ-025 — runtime constraint mutation."""

from __future__ import annotations

import pytest

from omni_q import build_mock_engine
from omni_q.mutation import MutationResult, RuntimeMutator


# ---------------------------------------------------------------------------
# constraint-shaped mutations (applied safely through the engine)
# ---------------------------------------------------------------------------


def test_forbid_before_run_is_honoured():
    engine = build_mock_engine()
    res = RuntimeMutator(engine).apply("don't touch the red connector")
    assert res.applied == [("forbid_object", "connector_2")]
    assert res.ok

    engine.run("inspect and correct the workspace")
    moved = [a["result"].get("moved") for a in engine._actions if a["op"] == "MOVE"]
    assert "connector_2" not in moved


def test_keep_local_mutation_escalates_mode():
    engine = build_mock_engine()
    RuntimeMutator(engine).apply("keep everything local")
    receipt = engine.run("inspect and correct the workspace")
    assert receipt.metrics["mode"] == "LOCAL_ONLY"
    assert receipt.metrics["resolved"] is True


def test_prefer_arm_mutation_reaches_the_planner():
    engine = build_mock_engine()
    RuntimeMutator(engine).apply("from now on use the left arm")
    engine.run("inspect and correct the workspace")
    arms = {a["arm"] for a in engine._actions if a["op"] in {"PICK", "MOVE"}}
    assert arms == {"left"}


def test_mid_run_mutation_recompiles_the_live_graph():
    engine = build_mock_engine()
    mut = RuntimeMutator(engine)
    original = engine.manipulator.execute
    fired = {"n": 0}

    def hook(step, world):
        if fired["n"] == 0 and step.op == "PICK":
            fired["n"] = 1
            mut.apply("actually, don't touch the plate")  # plate_1
        return original(step, world)

    engine.manipulator.execute = hook  # type: ignore[method-assign]
    engine.run("inspect and correct the workspace")

    kinds = [e.kind for e in engine.bus.log]
    assert "constraint.added" in kinds
    assert "graph.recompiled" in kinds
    moved = [a["result"].get("moved") for a in engine._actions if a["op"] == "MOVE"]
    assert "plate_1" not in moved


def test_rejected_constraint_is_reported_not_raised(monkeypatch):
    from omni_q.contracts import ConstraintValidationError

    engine = build_mock_engine()

    def boom(*a, **k):
        raise ConstraintValidationError("nope")

    monkeypatch.setattr(engine, "add_constraint", boom)
    res = RuntimeMutator(engine).apply("don't touch the red connector")
    assert res.applied == []
    assert res.rejected and res.rejected[0][0] == "forbid_object"
    assert not res.ok


# ---------------------------------------------------------------------------
# structural mutations are parsed but deferred
# ---------------------------------------------------------------------------


def test_spin_is_deferred_with_a_reason():
    res = RuntimeMutator(build_mock_engine()).apply("spin that plate")
    assert res.applied == []
    assert res.deferred and res.deferred[0][0] == "spin"
    assert res.deferred[0][1]["object"] == "plate_1"
    assert "OQ-014" in res.deferred[0][2] or "primitive" in res.deferred[0][2]


def test_nudge_is_deferred():
    res = RuntimeMutator(build_mock_engine()).apply("move the fork farther left")
    assert [k for k, _p, _r in res.deferred] == ["nudge"]
    assert res.deferred[0][1] == {"object": "fork", "direction": "left", "amount": "farther"}


def test_compound_instruction_splits_applied_and_deferred():
    res = RuntimeMutator(build_mock_engine()).apply(
        "don't touch the red connector and spin the plate")
    assert ("forbid_object", "connector_2") in res.applied
    assert [k for k, _p, _r in res.deferred] == ["spin"]
    assert "applied" in res.summary() and "deferred" in res.summary()


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


def test_empty_instruction_is_a_no_op():
    res = RuntimeMutator(build_mock_engine()).apply("hello there")
    assert res.summary() == "no change"
    assert not res.ok


def test_history_and_as_dict_round_trip():
    mut = RuntimeMutator(build_mock_engine())
    mut.apply_all(["keep it local", "spin that plate"])
    assert len(mut.history) == 2
    d = mut.history[0].as_dict()
    assert d["applied"] == [["keep_local", None]] and d["ok"] is True
