"""OQ-010 -- single-arm primitives get distinct controller targets.

Skipped in the pure-Python core environment; run with the project venv after
installing the ``intel`` extra (``pip install -e ".[dev,intel]"``).
"""

from __future__ import annotations

import json

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from omni_q.contracts import TransitionRequest
from omni_q.intel_sim import (
    DRAWER_CLOSED,
    DRAWER_OPEN,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    IntelTableObserver,
    IntelTableVerifier,
    IntelTableWorld,
    build_intel_sim_engine,
    dual_so101_xml,
    IntelSceneConfig,
    run_intel_table_evaluation_report,
)

GRIPPER_QPOS_ADR = 5  # Jaw is joint index 5 within one arm's 6-joint block


def _send(world: IntelTableWorld, op: str, args: dict, actor: str = "intel.left_arm"):
    request = TransitionRequest(
        step_id=f"test_{op.lower()}", op=op, args=args,
        expected_revision=world.state().revision, actor=actor,
    )
    return world.apply_transition(request)


def _disable_gripper_collision(world: IntelTableWorld, arm: str) -> None:
    """Adversarial test fixture: force a real grasp failure by making one
    arm's whole gripper (Fixed_Jaw + Moving_Jaw bodies -- every geom they
    own, not just the named jaw_pad_* markers) unable to collide with
    anything. Zeroing only the 4 named pad geoms isn't enough on its own:
    with every geom in this scene on plain MuJoCo defaults (contype=
    conaffinity=1, see intel_sim.py's no-clip-exemption revert), the
    gripper's own unnamed collision meshes (Fixed_Jaw_Collision_1/2,
    Moving_Jaw_Collision_1/2/3 from the source MJCF) still register real
    contact even with the pad markers disabled -- checked directly, this
    was silently letting the "adversarial" fixture grasp succeed anyway."""
    mujoco = world._mujoco
    for body_name in (f"{arm}_Fixed_Jaw", f"{arm}_Moving_Jaw"):
        body_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for geom_id in range(world.model.ngeom):
            if world.model.geom_bodyid[geom_id] == body_id:
                world.model.geom_contype[geom_id] = 0
                world.model.geom_conaffinity[geom_id] = 0
    mujoco.mj_forward(world.model, world.data)


def test_open_then_close_drawer_moves_its_qpos_both_ways():
    world = IntelTableWorld()

    _send(world, "OPEN", {"object": "drawer"})
    assert world.simulation_summary()["drawer_qpos"] == pytest.approx(DRAWER_OPEN, abs=1e-6)

    _send(world, "CLOSE", {"object": "drawer"})
    assert world.simulation_summary()["drawer_qpos"] == pytest.approx(DRAWER_CLOSED, abs=1e-6)


def test_drawer_fixture_open_does_not_retarget_an_arm():
    """A passive fixture transition must not disturb the next arm command."""
    world = IntelTableWorld()
    before_ctrl = world.data.ctrl.copy()

    _send(world, "OPEN", {"object": "drawer"})

    np.testing.assert_allclose(world.data.ctrl, before_ctrl, atol=0.0)
    assert world.simulation_summary()["drawer_qpos"] == pytest.approx(DRAWER_OPEN, abs=1e-6)


def test_drawer_postcondition_verification_is_physics_backed_and_fail_closed():
    world = IntelTableWorld()
    verifier = IntelTableVerifier(world)
    observer = IntelTableObserver()

    _send(world, "OPEN", {"object": "drawer"})
    open_step = TransitionRequest(
        step_id="verify_open", op="OPEN", args={"object": "drawer"},
        expected_revision=world.state().revision, actor="intel.left_arm",
    )
    assert verifier.check(open_step, observer.observe(world.state())).ok

    # A stale/fabricated drawer label must not pass merely because the OPEN
    # operation was requested.  Perturb the simulator, then re-run the same
    # postcondition check against the observed state.
    world.data.qpos[world._drawer_qpos_adr] = DRAWER_CLOSED
    world._mujoco.mj_forward(world.model, world.data)
    assert not verifier.check(open_step, observer.observe(world.state())).ok

    _send(world, "CLOSE", {"object": "drawer"})
    close_step = TransitionRequest(
        step_id="verify_close", op="CLOSE", args={"object": "drawer"},
        expected_revision=world.state().revision, actor="intel.left_arm",
    )
    assert verifier.check(close_step, observer.observe(world.state())).ok


def test_legacy_scene_randomization_is_seeded_at_build_time():
    base = IntelSceneConfig()
    randomized = IntelSceneConfig(seed=701, randomized=True)
    assert dual_so101_xml(base) == dual_so101_xml(IntelSceneConfig())
    assert dual_so101_xml(randomized) == dual_so101_xml(randomized)
    assert dual_so101_xml(randomized) != dual_so101_xml(
        IntelSceneConfig(seed=702, randomized=True)
    )


def test_legacy_scene_randomization_bounds_fail_closed():
    with pytest.raises(ValueError, match="position_jitter_m"):
        IntelSceneConfig(position_jitter_m=0.011)
    with pytest.raises(ValueError, match="yaw_jitter_rad"):
        IntelSceneConfig(yaw_jitter_rad=0.26)


def test_randomized_legacy_report_retains_hashed_receipts(tmp_path):
    report = run_intel_table_evaluation_report(tmp_path, trials=2, seed=701)

    assert report["kind"].endswith("not a promotion claim")
    assert len(report["receipts"]) == 2
    assert len({entry["run_id"] for entry in report["receipts"]}) == 2
    assert sum(report["outcomes"].values()) == 2
    for entry in report["receipts"]:
        persisted = json.loads((tmp_path / entry["receipt"]).read_text(encoding="utf-8"))
        assert persisted["content_hash"] == entry["content_hash"]
        assert persisted["inputs"]["mode"] == "simulation-scripted-manipulation"
        assert len(persisted["actions"]) > 0
    assert (tmp_path / "report.json").exists()


def test_randomized_legacy_report_rejects_empty_trial_count(tmp_path):
    with pytest.raises(ValueError, match="trials must be positive"):
        run_intel_table_evaluation_report(tmp_path, trials=0)


def test_pick_and_place_keep_gripper_command_honest_on_failure():
    world = IntelTableWorld()

    pick = _send(world, "PICK", {"object": "fork_1"})
    assert world.data.ctrl[GRIPPER_QPOS_ADR] == pytest.approx(GRIPPER_CLOSED)

    place = _send(world, "PLACE", {"object": "fork_1", "to": "left"})
    if place.ok:
        assert world.data.ctrl[GRIPPER_QPOS_ADR] == pytest.approx(GRIPPER_OPEN)
    else:
        # A rejected placement restores the pre-attempt physics snapshot,
        # whose gripper command is the closed command left by PICK.
        assert place.detail["placed"] is False
        assert world.data.ctrl[GRIPPER_QPOS_ADR] == pytest.approx(GRIPPER_CLOSED)


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
    """A real grasp failure must not leave a scripted ownership claim.

    The calibrated cup now succeeds on the nominal scene (see
    intel_sim.py's module docstring: real orientation-aware IK + a resized
    cup_1 produce a genuine held grasp for the first time this
    investigation has seen). This adversarial fixture (see
    _disable_gripper_collision) disables the right gripper's own contact
    geoms, forcing the physical check to fail and exercising the same
    rollback boundary regardless of which object currently succeeds.
    """
    world = IntelTableWorld()
    _disable_gripper_collision(world, "right")

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


def test_pad_tracked_ik_converges_tighter_than_body_tracked_ik():
    """Protect the measured pad-tracking improvement in the legacy path.

    This is not a full table-setting gate: placement and all-object
    robustness remain separate evidence.  It only prevents the final
    object-facing approach from regressing to the old body-tracked solve.
    """
    world = IntelTableWorld()

    result = world._do_pick(6, "cup_1")

    assert result["reach_error_m"] < 0.06


def test_grasp_frame_reads_object_yaw_without_mutating_freejoint():
    """The orientation target is derived from observed scene state only."""
    world = IntelTableWorld()
    qpos_before = world.data.qpos.copy()

    rotation, roll_hint = world._grasp_frame(6, "cup_1")

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-10)
    assert -1.65 <= roll_hint <= 1.65
    np.testing.assert_allclose(world.data.qpos, qpos_before, atol=0.0)


def test_oriented_pad_solver_returns_pose_metrics_and_respects_arm_only_scope():
    world = IntelTableWorld()
    rotation, roll_hint = world._grasp_frame(6, "cup_1")
    target = world.data.geom_xpos[world._pad_geom[6]].copy()
    object_qpos_before = world.data.qpos[world._object_joints["cup_1"][0]:world._object_joints["cup_1"][0] + 7].copy()

    result = world._ik_reach_pad_pose(6, target, rotation, roll_hint=roll_hint, iters=1)

    assert set(result) == {"position_error_m", "orientation_error_rad", "orientation_satisfied"}
    assert result["position_error_m"] >= 0.0
    assert result["orientation_error_rad"] >= 0.0
    assert isinstance(result["orientation_satisfied"], bool)
    np.testing.assert_allclose(
        world.data.qpos[world._object_joints["cup_1"][0]:world._object_joints["cup_1"][0] + 7],
        object_qpos_before,
        atol=1e-10,
    )


def test_legacy_scene_exposes_calibrated_cup_and_no_collision_exemptions():
    """cup_1's real calibration survives, but the earlier version of this
    test also asserted a contype/conaffinity scheme that exempted the arm
    from colliding with the table/drawer/objects -- a no-clip cheat, not a
    control improvement (see intel_sim.py's module docstring). Reverted:
    every geom in this scene uses plain MuJoCo defaults (contype=
    conaffinity=1) and collides with everything. The narrower, real fix
    for the order-dependence bug that scheme was also solving --
    tableware not shoving other tableware before its own governed PICK --
    is now explicit named <exclude> pairs between the 5 tableware bodies,
    not a bitmask that also happened to exempt the arm."""
    world = IntelTableWorld()
    mujoco = world._mujoco
    cup_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, "cup_1")
    pad_id = world._pad_geom[0]
    assert tuple(world.model.geom_size[cup_id][:2]) == pytest.approx((0.022, 0.050), abs=1e-6)
    assert tuple(world.model.geom_friction[cup_id]) == pytest.approx((3.0, 0.020, 0.001), abs=1e-6)
    # Every geom -- pads included -- uses plain MuJoCo defaults, not a
    # custom contype/conaffinity scheme (the pad's own friction/solref/
    # solimp overrides are real material tuning, not a collision mask).
    for geom_id in (pad_id, cup_id):
        assert int(world.model.geom_contype[geom_id]) == 1
        assert int(world.model.geom_conaffinity[geom_id]) == 1
    for object_id in ("plate_1", "cup_1", "fork_1", "spoon_1", "napkin_1"):
        geom_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, object_id)
        assert int(world.model.geom_contype[geom_id]) == 1
        assert int(world.model.geom_conaffinity[geom_id]) == 1
    # The real, narrower fix: explicit exclude pairs between all 5
    # tableware bodies (10 pairs), on top of whatever arm self-collision
    # excludes the source MJCF already declared.
    assert world.model.nexclude >= 10


def test_legacy_cup_place_uses_observed_carry_offset_and_settles():
    """Placement is admitted only after measured release stability."""
    world = IntelTableWorld()

    pick = world._do_pick(6, "cup_1")
    assert pick["held"] is True

    place = world._do_place(6, "cup_1", "upper_right")

    assert place["placed"] is True
    assert place["placement_error_m"] < 0.06
    assert place["settle"]["table_supported"] is True
    assert place["settle"]["settled"] is True
    assert place["safety"]["before"]["relative_object_pad_distance_m"] < 0.10


def test_legacy_workspace_guard_fails_closed_for_limit_and_shared_entry():
    """A controller proposal cannot enter a proven unsafe boundary."""
    world = IntelTableWorld()
    world.data.qpos[0] = world.model.jnt_range[0, 1] - 0.005
    world._mujoco.mj_forward(world.model, world.data)
    limit_guard = world._workspace_safety(0, obj="cup_1")
    assert limit_guard["safe"] is False
    assert limit_guard["reason"] == "joint-limit proximity"

    world = IntelTableWorld()
    other_pad = world.data.geom_xpos[world._pad_geom[6]].copy()
    shared_guard = world._workspace_safety(0, target_xy=other_pad)
    assert shared_guard["safe"] is False
    assert shared_guard["reason"] == "unsafe shared-workspace entry"


def test_legacy_pick_stops_before_motion_when_shared_entry_is_unsafe():
    world = IntelTableWorld()
    before_time = float(world.data.time)

    world._workspace_safety = lambda *args, **kwargs: {
        "safe": False,
        "reason": "unsafe shared-workspace entry",
    }
    result = world._do_pick(6, "cup_1")

    assert result["held"] is False
    assert result["reason"] == "unsafe shared-workspace entry"
    assert float(world.data.time) == pytest.approx(before_time)


def test_legacy_planner_reserves_napkin_before_cutlery_retrieval():
    """The shared workspace order is explicit and remains governed."""
    engine = build_intel_sim_engine()
    graph = engine.planner.plan("set the table", engine.world.state())
    pick_ids = {
        step.args.get("object"): index
        for index, step in enumerate(graph.steps)
        if step.op == "PICK"
    }

    assert pick_ids["napkin_1"] < pick_ids["fork_1"]
    assert pick_ids["napkin_1"] < pick_ids["spoon_1"]


def test_failed_grasp_restores_mujoco_state_for_a_clean_retry():
    """A rejected real attempt must roll back physics as well as WorldState.

    Disable the right gripper's own contact geoms (see
    _disable_gripper_collision) to force a physical failure regardless of
    which object currently succeeds. Compare all state that the
    transition advances, not just the ownership label that the MockWorld
    layer reverts.
    """
    world = IntelTableWorld()
    _disable_gripper_collision(world, "right")
    before_qpos = world.data.qpos.copy()
    before_qvel = world.data.qvel.copy()
    before_ctrl = world.data.ctrl.copy()
    before_time = float(world.data.time)
    before_steps = world.simulation_summary()["controller_steps"]

    result = _send(world, "PICK", {"object": "cup_1"}, actor="intel.right_arm")

    assert result.ok is False
    np.testing.assert_allclose(world.data.qpos, before_qpos, atol=1e-10)
    np.testing.assert_allclose(world.data.qvel, before_qvel, atol=1e-10)
    np.testing.assert_allclose(world.data.ctrl, before_ctrl, atol=1e-10)
    assert float(world.data.time) == pytest.approx(before_time, abs=1e-12)
    assert world.simulation_summary()["controller_steps"] == before_steps
    assert world.state().ownership["cup_1"] is None


def test_failed_place_restores_the_pre_attempt_owner():
    """A failed carry must not silently turn a held object into released."""
    world = IntelTableWorld()
    world._ownership["cup_1"] = "intel.right_arm"  # fixture: prior verified PICK
    before_qpos = world.data.qpos.copy()

    result = _send(
        world, "PLACE", {"object": "cup_1", "to": "not-a-zone"},
        actor="intel.right_arm",
    )

    assert result.ok is False
    assert result.detail["placed"] is False
    assert world.state().ownership["cup_1"] == "intel.right_arm"
    assert world.state().objects["cup_1"].zone == "tray_cup"
    np.testing.assert_allclose(world.data.qpos, before_qpos, atol=1e-10)


def test_full_run_opens_the_drawer_before_retrieving_cutlery():
    """Not asserting resolved is True: real orientation-aware IK exists
    (see intel_sim.py's module docstring) but only cup_1 reliably holds
    today, so a full "set the table" run still doesn't resolve. OPEN is
    independent of the grasp/place IK path, so it stays reliable
    regardless -- that's what this test actually covers."""
    engine = build_intel_sim_engine()

    receipt = engine.run("set the table")

    ops_in_order = [a["op"] for a in receipt.actions]
    assert ops_in_order[0] == "OPEN"
    assert ops_in_order.index("OPEN") < ops_in_order.index("PICK")
    assert engine.world.simulation_summary()["drawer_qpos"] == pytest.approx(DRAWER_OPEN, abs=1e-6)


def test_a_persistently_failing_object_does_not_exhaust_the_whole_run_alone():
    """A found and fixed real problem, not a hypothetical: before this fix,
    once the first still-misplaced object in _OBJECT_ORDER failed to grasp,
    the engine's replan-on-any-failure loop kept regenerating the exact
    same graph with that object still first, so it alone consumed the
    entire max_revisions budget -- napkin_1 (the first object after cup_1
    reliably succeeds) was measured failing 6 times in a row while
    plate_1/fork_1/spoon_1 never got a single real attempt. This doesn't
    change whether the run resolves (every object still has to actually
    succeed for that), but it means the revision budget gets spent trying
    different objects instead of hammering whichever one happens to be
    stuck first -- both for real evidence richness and for what a live
    demo actually looks like."""
    engine = build_intel_sim_engine()

    engine.run("set the table")

    pick_objects = [
        action["result"].get("grasped") or action.get("step", "").removeprefix("pick_")
        for action in engine._actions
        if action["op"] == "PICK"
    ]
    distinct_objects_attempted = {obj for obj in pick_objects if obj}
    # cup_1 succeeds and drops out immediately; among the objects that keep
    # failing, more than one should have been genuinely attempted within
    # the same run, not just the first one in priority order.
    assert len(distinct_objects_attempted - {"cup_1"}) > 1
