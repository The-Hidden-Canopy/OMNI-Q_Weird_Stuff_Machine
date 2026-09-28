"""OQ-007 -- table-setting scene/object pack coverage.

Skipped in the pure-Python core environment; run with the project venv after
installing the ``intel`` extra (``pip install -e ".[dev,intel]"``).
"""

from __future__ import annotations

import pytest


mujoco = pytest.importorskip("mujoco")

from omni_q.intel_sim import build_intel_sim_engine, load_dual_so101_model


def test_drawer_is_a_passive_slide_joint_not_a_new_actuator():
    model = load_dual_so101_model()

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    assert joint_id >= 0
    assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE
    # passive: opening it is still an OQ-010 primitive, not a 13th actuator
    assert model.nu == 12


def test_tableware_objects_have_distinct_friction_by_material():
    model = load_dual_so101_model()

    def friction_of(geom_name: str) -> float:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        assert geom_id >= 0
        return float(model.geom_friction[geom_id][0])

    napkin, fork, plate = friction_of("napkin_1"), friction_of("fork_1"), friction_of("plate_1")
    # cloth grips more than metal cutlery
    assert napkin > fork
    assert napkin > plate


def test_fork_and_spoon_start_inside_the_drawer():
    engine = build_intel_sim_engine()
    objects = engine.world.state().objects

    assert objects["fork_1"].zone == "drawer"
    assert objects["spoon_1"].zone == "drawer"
    # every other object has its own zone -- the scheduler regression this
    # guards against is *all five* sharing one literal string
    other_zones = {objects[o].zone for o in ("plate_1", "cup_1", "napkin_1")}
    assert len(other_zones) == 3
    assert "drawer" not in other_zones

def test_worldstate_carries_real_metric_poses_from_mujoco():
    """OQ-009: IntelTableWorld.state() stamps each tracked object's live
    free-joint pose onto its Detection; the drawer fixture stays pose-less."""
    engine = build_intel_sim_engine()
    objects = engine.world.state().objects

    for oid in ("plate_1", "cup_1", "fork_1", "spoon_1", "napkin_1"):
        det = objects[oid]
        assert det.located and det.pose is not None
        # settled on the table surface (top face at world z = 0), not the floor
        assert 0.0 < det.pose.z < 0.10
        assert -0.45 < det.pose.x < 0.45 and -0.5 < det.pose.y < 0.3

    # two different objects are at different places
    assert objects["fork_1"].pose.planar_distance_to(objects["spoon_1"].pose) > 0.1
    # untracked fixture -> no metric pose
    assert objects["drawer"].pose is None
    # pose is read live: stepping the sim updates it
    engine.world.data.qpos[engine.world._object_joints["cup_1"][0]] += 0.05
    engine.world._mujoco.mj_forward(engine.world.model, engine.world.data)
    assert engine.world.state().objects["cup_1"].pose.x != objects["cup_1"].pose.x
