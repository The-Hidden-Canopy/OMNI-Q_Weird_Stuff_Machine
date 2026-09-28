from __future__ import annotations

from dataclasses import replace
import sys
import types

import numpy as np
import pytest

from omni_q import ControllerType, SkillContractError
from integrations.intel.vla.record_expert_demos import expert_action
from omni_q.skills.controllers.smolvla import ACTION_FIELDS, SmolVLAController
from scripts.record_smolvla_dataset import SmolVLADatasetRecorder

from test_skills import manifest, observation, request


def test_vla_is_a_learned_skill_with_provenance_and_verification() -> None:
    vla = replace(manifest(), controller_type=ControllerType.VLA)
    assert vla.controller_type is ControllerType.VLA

    with pytest.raises(SkillContractError, match="sha256"):
        replace(vla, artifact_digest="checkpoint.bin")

    with pytest.raises(SkillContractError, match="verification"):
        replace(vla, verification=())


def test_smolvla_module_is_import_safe_and_has_fixed_action_contract() -> None:
    assert ACTION_FIELDS == (
        "dx_mm",
        "dy_mm",
        "dz_mm",
        "droll_deg",
        "dpitch_deg",
        "dyaw_deg",
        "gripper_delta",
    )
    assert SmolVLAController.backend == "lerobot-smolvla"


def test_smolvla_controller_only_returns_a_proposal(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    class Feature:
        def __init__(self, shape):
            self.shape = shape

    class FakePolicy:
        config = types.SimpleNamespace(
            input_features={
                "observation.state": Feature((2,)),
                "observation.images.overhead": Feature((2, 3, 3)),
            },
            action_feature=Feature((7,)),
        )

        @classmethod
        def from_pretrained(cls, _checkpoint):
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def reset(self):
            self.reset_count = getattr(self, "reset_count", 0) + 1

        def select_action(self, _batch):
            return torch.zeros((1, 7), dtype=torch.float32)

    lerobot = types.ModuleType("lerobot")
    policies = types.ModuleType("lerobot.policies")
    smolvla = types.ModuleType("lerobot.policies.smolvla")
    utils = types.ModuleType("lerobot.policies.utils")
    policies.make_pre_post_processors = lambda *_args, **_kwargs: (lambda x: x, lambda x: x)
    smolvla.SmolVLAPolicy = FakePolicy
    utils.prepare_observation_for_inference = lambda raw, *_args, **_kwargs: raw
    lerobot.policies = policies
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.policies", policies)
    monkeypatch.setitem(sys.modules, "lerobot.policies.smolvla", smolvla)
    monkeypatch.setitem(sys.modules, "lerobot.policies.utils", utils)

    controller = SmolVLAController(
        "org/skill",
        camera_sources={"observation.images.overhead": lambda: np.zeros((2, 3, 3), dtype=np.uint8)},
        state_provider=lambda: np.array([0.0, 1.0], dtype=np.float32),
        instruction_provider=lambda _request: "pick the cup",
        device="cpu",
        skill_id="skill.v1",
    )
    proposal = controller.propose(request(), observation())

    assert proposal.skill_id == "skill.v1"
    assert proposal.controller_backend == "lerobot-smolvla"
    assert dict(proposal.values) == dict.fromkeys(ACTION_FIELDS, 0.0)
    assert not hasattr(controller, "actuator")


def test_smolvla_state_dimension_override_is_explicit(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    class Feature:
        def __init__(self, shape):
            self.shape = shape

    class FakePolicy:
        config = types.SimpleNamespace(
            input_features={
                "observation.state": Feature((2,)),
                "observation.images.overhead": Feature((2, 3, 3)),
            },
            action_feature=Feature((7,)),
        )

        @classmethod
        def from_pretrained(cls, _checkpoint):
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def reset(self):
            pass

        def select_action(self, _batch):
            return torch.zeros((1, 7), dtype=torch.float32)

    lerobot = types.ModuleType("lerobot")
    policies = types.ModuleType("lerobot.policies")
    smolvla = types.ModuleType("lerobot.policies.smolvla")
    utils = types.ModuleType("lerobot.policies.utils")
    policies.make_pre_post_processors = lambda *_args, **_kwargs: (lambda x: x, lambda x: x)
    smolvla.SmolVLAPolicy = FakePolicy
    utils.prepare_observation_for_inference = lambda raw, *_args, **_kwargs: raw
    lerobot.policies = policies
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.policies", policies)
    monkeypatch.setitem(sys.modules, "lerobot.policies.smolvla", smolvla)
    monkeypatch.setitem(sys.modules, "lerobot.policies.utils", utils)

    controller = SmolVLAController(
        "org/skill",
        camera_sources={"observation.images.overhead": lambda: np.zeros((2, 3, 3), dtype=np.uint8)},
        state_provider=lambda: np.array([0.0, 1.0, 2.0], dtype=np.float32),
        instruction_provider=lambda _request: "pick the cup",
        device="cpu",
        skill_id="skill.v1",
        state_dim_override=3,
    )

    assert controller.config_state_dim == 2
    assert controller.state_dim == 3
    proposal = controller.propose(request(), observation())
    assert dict(proposal.values) == dict.fromkeys(ACTION_FIELDS, 0.0)


class FakeDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.frames = []
        self.calls = []

    def add_frame(self, frame):
        self.calls.append("add_frame")
        self.frames.append(frame)

    def save_episode(self):
        self.calls.append("save_episode")

    def clear_episode_buffer(self):
        self.calls.append("clear_episode_buffer")

    def finalize(self):
        self.calls.append("finalize")

    def push_to_hub(self):
        self.calls.append("push_to_hub")
        return "pushed"


def test_dataset_recorder_finalizes_before_push_and_rejects_bad_frames() -> None:
    dataset = None

    def factory(**kwargs):
        nonlocal dataset
        dataset = FakeDataset(**kwargs)
        return dataset

    recorder = SmolVLADatasetRecorder(
        repo_id="org/omni-q-vla",
        root="data/vla",
        fps=20,
        state_names=("joint_1", "joint_2"),
        camera_sources={
            "observation.images.overhead": lambda: np.zeros((2, 3, 3), dtype=np.uint8),
        },
        width=3,
        height=2,
        dataset_factory=factory,
    )
    assert dataset is not None

    recorder.add(
        state=np.array([0.0, 1.0], dtype=np.float32),
        action={"dx_mm": 1.0},
        task="pick the cup",
    )
    recorder.finish_episode()
    assert recorder.push() == "pushed"
    assert dataset.calls == ["add_frame", "save_episode", "finalize", "push_to_hub"]

    with pytest.raises(RuntimeError, match="finalization"):
        recorder.add(
            state=np.array([0.0, 1.0], dtype=np.float32),
            action={},
            task="pick the cup",
        )


def test_dataset_recorder_does_not_resize_or_accept_nonfinite_state() -> None:
    recorder = SmolVLADatasetRecorder(
        repo_id="org/omni-q-vla",
        root="data/vla",
        fps=20,
        state_names=("joint_1",),
        camera_sources={
            "observation.images.overhead": lambda: np.zeros((2, 3, 3), dtype=np.uint8),
        },
        width=3,
        height=2,
        dataset_factory=FakeDataset,
    )

    with pytest.raises(ValueError, match="non-finite"):
        recorder.add(
            state=np.array([np.nan], dtype=np.float32),
            action={},
            task="pick the cup",
        )

    bad_camera = SmolVLADatasetRecorder(
        repo_id="org/omni-q-vla",
        root="data/vla",
        fps=20,
        state_names=("joint_1",),
        camera_sources={
            "observation.images.overhead": lambda: np.zeros((1, 3, 3), dtype=np.uint8),
        },
        width=3,
        height=2,
        dataset_factory=FakeDataset,
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        bad_camera.add(
            state=np.array([0.0], dtype=np.float32),
            action={},
            task="pick the cup",
        )


def test_dataset_recorder_accepts_explicit_rendered_images() -> None:
    recorder = SmolVLADatasetRecorder(
        repo_id="org/omni-q-vla",
        root="data/vla",
        fps=20,
        state_names=("joint_1",),
        camera_sources={
            "observation.images.overhead": None,
            "observation.images.wrist": None,
        },
        width=3,
        height=2,
        dataset_factory=FakeDataset,
    )

    images = {
        "observation.images.overhead": np.zeros((2, 3, 3), dtype=np.uint8),
        "observation.images.wrist": np.full((2, 3, 3), 127, dtype=np.uint8),
    }
    recorder.add(
        state=np.array([0.0], dtype=np.float32),
        action={"dx_mm": 1.0},
        task="pick the cup",
        images=images,
    )

    assert len(recorder.dataset.frames) == 1
    assert np.array_equal(
        recorder.dataset.frames[0]["observation.images.wrist"],
        images["observation.images.wrist"],
    )


def test_dataset_recorder_rejects_incomplete_explicit_images() -> None:
    recorder = SmolVLADatasetRecorder(
        repo_id="org/omni-q-vla",
        root="data/vla",
        fps=20,
        state_names=("joint_1",),
        camera_sources={
            "observation.images.overhead": None,
            "observation.images.wrist": None,
        },
        width=3,
        height=2,
        dataset_factory=FakeDataset,
    )

    with pytest.raises(ValueError, match="exactly the configured camera names"):
        recorder.add(
            state=np.array([0.0], dtype=np.float32),
            action={},
            task="pick the cup",
            images={
                "observation.images.overhead": np.zeros((2, 3, 3), dtype=np.uint8),
            },
        )


def test_expert_action_defaults_to_absolute_jaw_and_supports_legacy_delta() -> None:
    previous = {
        "pos": np.array([0.1, 0.2, 0.3]),
        "R": np.eye(3),
        "jaw": 0.4,
    }
    current = {
        "pos": np.array([0.101, 0.2, 0.3]),
        "R": np.eye(3),
        "jaw": 0.7,
    }

    absolute = expert_action(previous, current)
    delta = expert_action(previous, current, jaw_absolute=False)

    assert absolute["dx_mm"] == pytest.approx(1.0)
    assert absolute["gripper_delta"] == pytest.approx(0.7)
    assert delta["gripper_delta"] == pytest.approx(0.3)
