"""Proposal-only SmolVLA integration for governed Cartesian skills.

The optional LeRobot dependency is deliberately loaded only when a controller
instance is created.  Importing :mod:`omni_q.skills` therefore remains safe on
machines that only run deterministic or mock skills.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..contracts import ProposedAction, SkillObservation, SkillRequest

__all__ = ["ACTION_FIELDS", "SmolVLAController"]


ACTION_FIELDS = (
    "dx_mm",
    "dy_mm",
    "dz_mm",
    "droll_deg",
    "dpitch_deg",
    "dyaw_deg",
    "gripper_delta",
)


class SmolVLAController:
    """Turn images, proprioception, and language into one safe proposal.

    This adapter deliberately has no actuator reference.  Its only output is
    :class:`~omni_q.skills.contracts.ProposedAction`; the skill supervisor and
    actuator boundary remain responsible for authorization and motor access.
    """

    backend = "lerobot-smolvla"

    def __init__(
        self,
        checkpoint: str,
        *,
        camera_sources: Mapping[str, Any],
        state_provider: Callable[[], Any],
        instruction_provider: Callable[[SkillRequest], str],
        device: str | None = None,
        robot_type: str = "so101",
        skill_id: str | None = None,
    ) -> None:
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("checkpoint must be a non-empty string")
        if not camera_sources:
            raise ValueError("camera_sources must not be empty")
        if not callable(state_provider):
            raise TypeError("state_provider must be callable")
        if not callable(instruction_provider):
            raise TypeError("instruction_provider must be callable")
        if not isinstance(robot_type, str) or not robot_type.strip():
            raise ValueError("robot_type must be a non-empty string")

        try:
            import numpy as np
            import torch
            from lerobot.policies import make_pre_post_processors
            from lerobot.policies.smolvla import SmolVLAPolicy
            from lerobot.policies.utils import (
                prepare_observation_for_inference,
            )
        except ImportError as exc:
            raise RuntimeError(
                "SmolVLA support is optional. Install the VLA extra with "
                '`python -m pip install -e "[smolvla]"`.'
            ) from exc

        self._np = np
        self._torch = torch
        self._prepare_observation = prepare_observation_for_inference
        self.device = self._resolve_device(torch, device)
        self.checkpoint = checkpoint.strip()
        self.robot_type = robot_type.strip()
        if skill_id is not None and (not isinstance(skill_id, str) or not skill_id.strip()):
            raise ValueError("skill_id must be non-empty when provided")
        self.skill_id = skill_id.strip() if skill_id is not None else None
        self.camera_sources = dict(camera_sources)
        self.state_provider = state_provider
        self.instruction_provider = instruction_provider

        try:
            self.policy = SmolVLAPolicy.from_pretrained(self.checkpoint)
            self.policy.to(self.device)
            self.policy.eval()
            self.preprocess, self.postprocess = make_pre_post_processors(
                self.policy.config,
                self.checkpoint,
                preprocessor_overrides={
                    "device_processor": {"device": str(self.device)},
                },
            )
        except Exception as exc:  # noqa: BLE001 - optional backend boundary
            raise RuntimeError(
                f"could not load SmolVLA checkpoint {self.checkpoint!r}"
            ) from exc

        self._validate_config()
        self._last_instruction: str | None = None

    @staticmethod
    def _resolve_device(torch: Any, requested: str | None) -> Any:
        if requested is not None:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _validate_config(self) -> None:
        config = self.policy.config
        input_features = getattr(config, "input_features", {})
        if not isinstance(input_features, Mapping):
            raise ValueError("SmolVLA checkpoint has invalid input_features")

        expected_images = {
            name
            for name in input_features
            if name.startswith("observation.images.")
        }
        missing = expected_images - set(self.camera_sources)
        if missing:
            raise ValueError(
                "VLA checkpoint expects camera inputs that were not provided: "
                f"{sorted(missing)}"
            )

        state_feature = input_features.get("observation.state")
        state_shape = getattr(state_feature, "shape", None)
        if state_shape is None or len(state_shape) != 1:
            raise ValueError(
                "SmolVLA checkpoint must expose a one-dimensional "
                "observation.state feature"
            )
        self.state_dim = int(state_shape[0])

        action_feature = getattr(config, "action_feature", None)
        if action_feature is None:
            output_features = getattr(config, "output_features", {})
            if isinstance(output_features, Mapping):
                action_feature = output_features.get("action")
        action_shape = getattr(action_feature, "shape", None)
        if action_shape is None or len(action_shape) != 1 or int(action_shape[0]) != len(ACTION_FIELDS):
            raise ValueError(
                "OMNI SmolVLA checkpoints must emit exactly seven action values"
            )

        self.expected_images = tuple(sorted(expected_images))

    @staticmethod
    def _capture(source: Any, np: Any) -> Any:
        frame = source.capture() if hasattr(source, "capture") else source()
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("camera source must return an HxWx3 RGB frame")
        if not np.issubdtype(frame.dtype, np.number):
            raise ValueError("camera source must return numeric RGB data")
        if not np.isfinite(frame).all():
            raise ValueError("camera source returned non-finite RGB data")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def _make_batch(self, instruction: str) -> dict[str, Any]:
        np = self._np
        state = np.asarray(self.state_provider(), dtype=np.float32).reshape(-1)
        if state.shape != (self.state_dim,):
            raise ValueError(
                f"VLA state dimension mismatch: got {state.shape[0]}, "
                f"expected {self.state_dim}"
            )
        if not np.isfinite(state).all():
            raise ValueError("VLA state contains non-finite values")

        batch: dict[str, Any] = {"observation.state": state}
        for name in self.expected_images:
            batch[name] = self._capture(self.camera_sources[name], np)
        return batch

    def reset(self) -> None:
        self.policy.reset()
        if hasattr(self.preprocess, "reset"):
            self.preprocess.reset()
        if hasattr(self.postprocess, "reset"):
            self.postprocess.reset()
        self._last_instruction = None

    def _reset_for_instruction_change(self) -> None:
        # Current LeRobot exposes reset() as the stable queue boundary.  Keep
        # support for an explicit queue method if a future version provides it.
        drop_queue = getattr(self.policy, "drop_queued_actions", None)
        if callable(drop_queue):
            drop_queue()
        else:
            self.policy.reset()

    def propose(
        self,
        request: SkillRequest,
        observation: SkillObservation,
    ) -> ProposedAction:
        del observation  # The provider is already bound to the current sensors.

        instruction = self.instruction_provider(request)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("SmolVLA requires a non-empty language instruction")
        instruction = instruction.strip()

        if instruction != self._last_instruction:
            self._reset_for_instruction_change()
            self._last_instruction = instruction

        raw = self._make_batch(instruction)
        batch = self._prepare_observation(
            raw,
            self.device,
            task=instruction,
            robot_type=self.robot_type,
        )
        batch = self.preprocess(batch)

        with self._torch.inference_mode():
            action = self.policy.select_action(batch)
        action = self.postprocess(action)

        if hasattr(action, "detach"):
            values = action.detach().float().cpu().numpy()
        else:
            values = self._np.asarray(action, dtype=self._np.float32)
        values = self._np.asarray(values).squeeze().reshape(-1)
        if values.shape != (len(ACTION_FIELDS),):
            raise RuntimeError(
                "OMNI SmolVLA checkpoint must emit exactly seven actions; "
                f"got {values.shape}"
            )
        if not self._np.isfinite(values).all():
            raise RuntimeError("VLA produced non-finite action")

        skill_id = self.skill_id or request.skill_id
        if not skill_id:
            raise ValueError(
                "SmolVLAController needs a configured skill_id when the "
                "request was selected by capability"
            )

        return ProposedAction.from_mapping(
            request_id=request.request_id,
            skill_id=skill_id,
            values=dict(zip(ACTION_FIELDS, values.tolist())),
            controller_backend=self.backend,
        )
