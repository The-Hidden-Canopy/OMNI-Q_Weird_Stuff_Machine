"""Record supervisor-approved Cartesian demonstrations for SmolVLA.

This module is an optional data-plane tool.  It never promotes a skill and it
does not infer actions: callers provide the action that the governed expert
actually approved and applied.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

ACTION_FIELDS = (
    "dx_mm",
    "dy_mm",
    "dz_mm",
    "droll_deg",
    "dpitch_deg",
    "dyaw_deg",
    "gripper_delta",
)


class SmolVLADatasetRecorder:
    """Write camera/state/action frames using the current LeRobot API."""

    def __init__(
        self,
        *,
        repo_id: str,
        root: str | Path,
        fps: int,
        state_names: tuple[str, ...],
        camera_sources: Mapping[str, Any],
        width: int = 640,
        height: int = 480,
        dataset_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(repo_id, str) or not repo_id.strip():
            raise ValueError("repo_id must be a non-empty string")
        if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive integer")
        if not state_names or any(not name.strip() for name in state_names):
            raise ValueError("state_names must contain at least one named state")
        if not camera_sources:
            raise ValueError("camera_sources must not be empty")
        if any(not name.startswith("observation.images.") for name in camera_sources):
            raise ValueError("camera source names must use observation.images.*")
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if dataset_factory is None:
            try:
                from lerobot.datasets import LeRobotDataset
            except ImportError as exc:
                raise RuntimeError(
                    "Dataset recording requires LeRobot. Install the smolvla extra."
                ) from exc
            dataset_factory = LeRobotDataset.create

        self.camera_sources = dict(camera_sources)
        self.state_names = tuple(state_names)
        self.width = width
        self.height = height
        self._finalized = False

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.state_names),),
                "names": list(self.state_names),
            },
            "action": {
                "dtype": "float32",
                "shape": (len(ACTION_FIELDS),),
                "names": list(ACTION_FIELDS),
            },
        }
        for feature_name in self.camera_sources:
            features[feature_name] = {
                "dtype": "video",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            }

        self.dataset = dataset_factory(
            repo_id=repo_id.strip(),
            root=Path(root),
            fps=fps,
            robot_type="omni_q_so101",
            features=features,
            use_videos=True,
            image_writer_threads=4,
        )

    @staticmethod
    def _capture(source: Any) -> np.ndarray:
        frame = source.capture() if hasattr(source, "capture") else source()
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("camera source must return an HxWx3 RGB frame")
        if not np.issubdtype(frame.dtype, np.number) or not np.isfinite(frame).all():
            raise ValueError("camera source returned invalid RGB data")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def add(
        self,
        *,
        state: np.ndarray,
        action: Mapping[str, float],
        task: str,
    ) -> None:
        if self._finalized:
            raise RuntimeError("cannot add frames after dataset finalization")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.shape != (len(self.state_names),):
            raise ValueError(
                f"state shape mismatch: got {state_array.shape[0]}, "
                f"expected {len(self.state_names)}"
            )
        if not np.isfinite(state_array).all():
            raise ValueError("state contains non-finite values")

        action_vector = np.asarray(
            [float(action.get(name, 0.0)) for name in ACTION_FIELDS],
            dtype=np.float32,
        )
        if not np.isfinite(action_vector).all():
            raise ValueError("action contains non-finite values")

        frame: dict[str, Any] = {
            "observation.state": state_array,
            "action": action_vector,
            "task": task.strip(),
        }
        for name, source in self.camera_sources.items():
            image = self._capture(source)
            if image.shape != (self.height, self.width, 3):
                raise ValueError(
                    f"{name} shape mismatch: got {image.shape}, "
                    f"expected {(self.height, self.width, 3)}"
                )
            frame[name] = image

        self.dataset.add_frame(frame)

    def finish_episode(self) -> None:
        if self._finalized:
            raise RuntimeError("cannot finish an episode after finalization")
        self.dataset.save_episode()

    def discard_episode(self) -> None:
        if self._finalized:
            raise RuntimeError("cannot discard an episode after finalization")
        self.dataset.clear_episode_buffer()

    def finalize(self) -> None:
        if not self._finalized:
            self.dataset.finalize()
            self._finalized = True

    def push(self) -> Any:
        # LeRobot requires finalization before the dataset is publishable.
        self.finalize()
        return self.dataset.push_to_hub()


__all__ = ["ACTION_FIELDS", "SmolVLADatasetRecorder"]
