"""OQ-010 -- single-arm primitives get distinct controller targets.

Skipped in the pure-Python core environment; run with the project venv after
installing the ``intel`` extra (``pip install -e ".[dev,intel]"``).
"""

from __future__ import annotations

import pytest


mujoco = pytest.importorskip("mujoco")

from omni_q.contracts import TransitionRequest
from omni_q.intel_sim import (
    DRAWER_CLOSED,
    DRAWER_OPEN,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    IntelTableWorld,
    build_intel_sim_engine,
)

GRIPPER_QPOS_ADR = 5  # Jaw is joint index 5 within one arm's 6-joint block


def _send(world: IntelTableWorld, op: str, args: dict, actor: str = "intel.left_arm"):
    request = TransitionRequest(
        step_id=f"test_{op.lower()}", op=op, args=args,
        expected_revision=world.state().revision, actor=actor,
    )
    return world.apply_transition(request)


def test_open_then_close_drawer_moves_its_qpos_both_ways():
    world = IntelTableWorld()

    _send(world, "OPEN", {"object": "drawer"})
    assert world.simulation_summary()["drawer_qpos"] == pytest.approx(DRAWER_OPEN, abs=1e-6)

    _send(world, "CLOSE", {"object": "drawer"})
    assert world.simulation_summary()["drawer_qpos"] == pytest.approx(DRAWER_CLOSED, abs=1e-6)


def test_pick_closes_gripper_and_place_opens_it():
    world = IntelTableWorld()

    _send(world, "PICK", {"object": "fork_1"})
    assert world.data.ctrl[GRIPPER_QPOS_ADR] == pytest.approx(GRIPPER_CLOSED)

    _send(world, "PLACE", {"object": "fork_1", "to": "left"})
    assert world.data.ctrl[GRIPPER_QPOS_ADR] == pytest.approx(GRIPPER_OPEN)


def test_open_close_rotate_present_execute_individually_and_differ():
    """OQ-010 done-when: PICK/PLACE/MOVE/OPEN/CLOSE/ROTATE/PRESENT each
    execute; this asserts they aren't all silently collapsing onto the same
    HOME-derived controller target."""
    world = IntelTableWorld()
    targets: dict[str, tuple[float, ...]] = {}
    for op, args in (
        ("OPEN", {}), ("CLOSE", {}), ("ROTATE", {}), ("PRESENT", {}),
    ):
        _send(world, op, args)
        targets[op] = tuple(round(v, 4) for v in world.data.ctrl[:6])

    assert len(set(targets.values())) == len(targets), targets


def test_failed_grasp_reverts_worldstate_instead_of_claiming_success():
    """The actual point of the OQ-010 rework: a real grasp/placement failure
    must not leave the scripted WorldState update (ownership/zone) committed
    -- that would silently claim success the physics never delivered. cup_1
    is a reliable real-world repro today (SO-101's gripper margin against a
    64mm-diameter cup is tight, and this adapter has no orientation-aware
    IK), which makes it a good regression case for the revert path itself,
    independent of whether/when the grasp geometry improves."""
    world = IntelTableWorld()

    request = TransitionRequest(
        step_id="t", op="PICK", args={"object": "cup_1"},
        expected_revision=world.state().revision, actor="intel.right_arm",
    )
    result = world.apply_transition(request)

    assert result.detail["held"] is False
    assert result.ok is False
    # MockWorld.apply_transition assigns ownership unconditionally before
    # this adapter's real-physics grasp check runs; a failed grasp must
    # revert that claim, not leave the object silently "held" by an arm
    # that never actually gripped it.
    assert world.state().ownership["cup_1"] is None


def test_full_run_opens_the_drawer_before_retrieving_cutlery():
    """Not asserting resolved is True: full-run grasp success needs
    orientation-aware IK this adapter doesn't have yet (see
    test_intel_sim.py). OPEN is independent of the grasp/place IK path, so
    it stays reliable regardless -- that's what this test actually covers."""
    engine = build_intel_sim_engine()

    receipt = engine.run("set the table")

    ops_in_order = [a["op"] for a in receipt.actions]
    assert ops_in_order[0] == "OPEN"
    assert ops_in_order.index("OPEN") < ops_in_order.index("PICK")
    assert engine.world.simulation_summary()["drawer_qpos"] == pytest.approx(DRAWER_OPEN, abs=1e-6)
