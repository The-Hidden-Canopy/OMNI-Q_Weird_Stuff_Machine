from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

import omni_q.humanoid_g1 as g1_module
from omni_q.humanoid_g1 import G1Error, G1Provider, gravity_orientation, pd_control


def unitree_root() -> Path:
    root = Path(__file__).parents[1] / "external" / "unitree_rl_gym"
    if not root.is_dir():
        pytest.skip("Unitree checkout is not present")
    return root


def provider_dependencies_available() -> bool:
    return g1_module._dependency_error() is None


def test_gravity_orientation_matches_unitree_wxyz_identity_convention():
    np.testing.assert_allclose(
        gravity_orientation(np.array([1.0, 0.0, 0.0, 0.0])),
        # Unitree's policy convention points body-frame gravity down for the
        # identity orientation; preserve the official formula exactly.
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )


def test_pd_control_is_elementwise_position_and_velocity_control():
    np.testing.assert_allclose(
        pd_control(
            np.array([2.0, 1.0]),
            np.array([1.0, 0.5]),
            np.array([3.0, 4.0]),
            np.array([0.0, 0.0]),
            np.array([0.25, -0.5]),
            np.array([2.0, 2.0]),
        ),
        np.array([2.5, 3.0], dtype=np.float32),
    )


def test_missing_unitree_root_fails_closed(monkeypatch):
    if not provider_dependencies_available():
        pytest.skip("G1 optional dependencies are not installed")
    monkeypatch.delenv("UNITREE_RL_GYM", raising=False)
    with pytest.raises(G1Error, match="UNITREE_RL_GYM"):
        G1Provider()


def test_actual_unitree_policy_and_mjcf_bind_before_stepping():
    if not provider_dependencies_available():
        pytest.skip("G1 optional dependencies are not installed")
    provider = G1Provider(unitree_root=unitree_root())

    assert provider.num_actions == 12
    assert provider.num_obs == 47
    assert provider.model.nq == 19
    assert provider.model.nv == 18
    assert provider.model.nu == 12
    assert provider.policy_path.name == "motion.pt"
    assert provider.xml_path.name == "scene.xml"
    assert provider.qpos_indices == tuple(range(7, 19))
    assert provider.qvel_indices == tuple(range(6, 18))
    assert provider.actuator_indices == tuple(range(12))

    provider.set_velocity(3.0, -2.0, 4.0)
    np.testing.assert_allclose(provider.command, [0.8, -0.5, 1.57])
    for _ in range(provider.decimation):
        provider.step()

    assert provider.counter == provider.decimation
    assert provider.data.time == pytest.approx(provider.dt * provider.decimation)
    assert np.isfinite(provider.action).all()
    assert np.isfinite(provider.target_q).all()


def test_offline_provider_cannot_accept_new_velocity_intent():
    if not provider_dependencies_available():
        pytest.skip("G1 optional dependencies are not installed")
    provider = G1Provider(unitree_root=unitree_root())
    provider.set_online(False)
    assert not provider.online
    np.testing.assert_allclose(provider.command, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(provider.data.ctrl, np.zeros(provider.model.nu))
    with pytest.raises(G1Error, match="offline"):
        provider.march()
