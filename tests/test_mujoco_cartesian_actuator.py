from __future__ import annotations

import pytest

from omni_q.skills.actuators import MuJoCoCartesianIKActuator


mujoco = pytest.importorskip("mujoco")


ARM_XML = """
<mujoco>
  <option timestep="0.002" />
  <worldbody>
    <body name="base" pos="0 0 0">
      <body name="link1">
        <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14" />
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02" />
        <body name="link2" pos="0.1 0 0">
          <joint name="j2" type="hinge" axis="0 0 1" range="-3.14 3.14" />
          <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02" />
          <site name="ee" pos="0.1 0 0" size="0.01" />
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="20" />
    <position name="a2" joint="j2" kp="20" />
  </actuator>
</mujoco>
"""


def make_actuator() -> MuJoCoCartesianIKActuator:
    model = mujoco.MjModel.from_xml_string(ARM_XML)
    data = mujoco.MjData(model)
    return MuJoCoCartesianIKActuator(
        model,
        data,
        arm_joint_names=("j1", "j2"),
        arm_actuator_names=("a1", "a2"),
        ee_site_name="ee",
        ik_iterations=3,
        physics_substeps=1,
    )


def test_cartesian_actuator_requires_real_named_bindings() -> None:
    model = mujoco.MjModel.from_xml_string(ARM_XML)
    data = mujoco.MjData(model)

    with pytest.raises(ValueError, match="site"):
        MuJoCoCartesianIKActuator(
            model,
            data,
            arm_joint_names=("j1", "j2"),
            arm_actuator_names=("a1", "a2"),
            ee_site_name="missing-ee",
        )


def test_cartesian_actuator_steps_physics_and_returns_receipt() -> None:
    actuator = make_actuator()
    observation = actuator.observation(
        org_id="org_a",
        arm_id="arm_1",
        world_revision=4,
    )
    before = actuator.data.time
    detail = actuator.apply({"dx_mm": 5.0}, observation)

    assert actuator.data.time > before
    assert detail["controller"] == "smolvla+cartesian-ik"
    assert detail["iterations"] == 3
    assert len(detail["final_position_m"]) == 3


def test_cartesian_actuator_rejects_unknown_or_nonfinite_actions() -> None:
    actuator = make_actuator()
    observation = actuator.observation(
        org_id="org_a",
        arm_id="arm_1",
        world_revision=4,
    )

    with pytest.raises(ValueError, match="unsupported"):
        actuator.apply({"joint_1": 0.2}, observation)

    with pytest.raises(ValueError, match="finite"):
        actuator.apply({"dx_mm": float("nan")}, observation)
