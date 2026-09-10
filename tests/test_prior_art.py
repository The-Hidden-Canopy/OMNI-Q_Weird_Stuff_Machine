"""Patterns lifted from sibling repos (see docs/prior-art.md)."""

from __future__ import annotations

import dataclasses

import pytest

from omni_q import build_mock_engine
from omni_q.contracts import (
    AutonomyMode,
    DataStatus,
    Detection,
    MissionEnvelope,
    content_hash_of,
    merge_mode,
    verify_chain,
)
from omni_q.engine import OmniQ
from omni_q.devices import default_devices
from omni_q.fakes import FakeManipulator, FakeObserver, FakeRecorder, FakeVerifier, RulePlanner
from omni_q.world import MockWorld


# -- SOCOM_REACT: monotone degradation ---------------------------------------

def test_merge_mode_never_downgrades_without_reset():
    m = AutonomyMode.NOMINAL
    m = merge_mode(m, AutonomyMode.DEGRADED)
    m = merge_mode(m, AutonomyMode.NOMINAL)      # ignored
    assert m is AutonomyMode.DEGRADED
    m = merge_mode(m, AutonomyMode.LOCAL_ONLY)
    m = merge_mode(m, AutonomyMode.DEGRADED)     # ignored
    assert m is AutonomyMode.LOCAL_ONLY


def test_keep_local_escalates_mode():
    engine = build_mock_engine()
    engine.add_constraint("keep_local", justification="test operator command")
    receipt = engine.run("inspect and correct the workspace")
    assert receipt.metrics["mode"] == "LOCAL_ONLY"
    assert "mode.changed" in [e.kind for e in engine.bus.log]


# -- SOCOM_REACT: PlanDecision reason object --------------------------------

def test_plan_decision_records_rejected_forbidden_object():
    engine = build_mock_engine()
    engine.add_constraint("forbid_object", "connector_2", justification="test boundary")
    receipt = engine.run("inspect and correct the workspace")

    assert receipt.decisions, "every compile should emit a PlanDecision"
    assert any(
        d["rejected"].get("connector_2", "").startswith("forbidden")
        for d in receipt.decisions
    )
    assert {"target": "connector_2", "reason": "forbidden by constraint"} in receipt.rejected


def test_plan_decision_carries_envelope_digest_and_mode():
    engine = build_mock_engine()
    receipt = engine.run("inspect and correct the workspace")
    d = receipt.decisions[0]
    assert d["envelope_digest"].startswith("sha256:")
    assert d["mode"] in {m.value for m in AutonomyMode}


# -- SOCOM_REACT: signed envelope authority ---------------------------------

def test_mission_envelope_digest_is_stable_and_content_sensitive():
    a = MissionEnvelope(mission_id="m1", forbidden_objects=("x",))
    b = MissionEnvelope(mission_id="m1", forbidden_objects=("x",))
    c = MissionEnvelope(mission_id="m1", forbidden_objects=("y",))
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()


def test_envelope_max_revisions_is_the_engine_bound():
    env = MissionEnvelope(mission_id="tight", max_revisions=1)
    world = MockWorld.sample()
    engine = OmniQ(
        world=world, observer=FakeObserver(), planner=RulePlanner(),
        manipulator=FakeManipulator(world), verifier=FakeVerifier(),
        device=default_devices(), recorder=FakeRecorder(), envelope=env,
    )
    assert engine.max_revisions == 1


# -- FALCON + VIGIL: parent-chained, verifiable receipts -------------------

def test_receipt_chain_verifies_across_runs():
    engine = build_mock_engine()
    engine.run("inspect and correct the workspace")
    engine.run("set the table")
    engine.run("tidy the workspace")

    records = engine.recorder.records
    assert len(records) == 3
    assert records[0].parent_hash == "GENESIS"
    assert records[1].parent_hash == records[0].content_hash
    assert records[2].parent_hash == records[1].content_hash
    verify_chain(records)   # raises on any inconsistency


def test_receipt_tamper_is_detected():
    engine = build_mock_engine()
    engine.run("inspect and correct the workspace")
    rec = engine.recorder.records[0]
    tampered = dataclasses.replace(rec, goal="something else")
    with pytest.raises(ValueError):
        verify_chain([tampered])


def test_receipt_provenance_block_present():
    engine = build_mock_engine()
    receipt = engine.run("inspect and correct the workspace")
    prov = receipt.provenance
    assert prov["generated_by"] == "omni_q.engine"
    assert "python" in prov and "platform" in prov and "package_versions" in prov


# -- Open-World-Model-Harness: stale knowledge is not authoritative --------

def test_planner_will_not_act_on_a_non_live_detection():
    world = MockWorld.sample()
    # mark the misplaced connector as STALE
    stale = dataclasses.replace(world._objects["connector_2"], status=DataStatus.STALE)
    world._objects["connector_2"] = stale

    engine = OmniQ(
        world=world, observer=FakeObserver(), planner=RulePlanner(),
        manipulator=FakeManipulator(world), verifier=FakeVerifier(),
        device=default_devices(), recorder=FakeRecorder(),
    )
    engine.run("inspect and correct the workspace")

    decided = engine.planner.last_decision
    assert "connector_2" in decided.rejected
    assert "non-authoritative" in decided.rejected["connector_2"]
    moved = [a["result"].get("moved") for a in engine._actions if a["op"] == "MOVE"]
    assert "connector_2" not in moved
