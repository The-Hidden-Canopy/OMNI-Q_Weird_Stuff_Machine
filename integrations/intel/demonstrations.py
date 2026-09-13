"""Governed command-level demonstration capture for the Intel arm world.

This recorder is deliberately smaller than a LeRobot motor trajectory.  It
captures the transition requests that the governed planner sent to a world,
the authoritative before/after snapshots, and optional MuJoCo telemetry.  It
never writes world state itself: every attempted action goes through
``world.apply_transition`` and stale, cross-organization, or otherwise
rejected requests remain visible in the episode.

The resulting JSONL is a provenance-bearing intermediate artifact for
inspection and later conversion.  It is not evidence of a trained policy,
physical hardware, or a complete LeRobot dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from omni_q.contracts import (
    TransitionRequest,
    TransitionResult,
    WorldState,
)

SCHEMA_VERSION = "omni-q.command-demonstration.v1"


@runtime_checkable
class DemonstrationWorld(Protocol):
    """Minimum world boundary needed by the recorder."""

    org_id: str
    revision: int

    def state(self) -> WorldState: ...

    def apply_transition(self, request: TransitionRequest) -> TransitionResult: ...


@dataclass(frozen=True)
class SimulationTelemetry:
    """Optional read-only simulator values captured around one command."""

    sim_time_s: float | None = None
    controller_steps: int | None = None
    joint_position: tuple[float, ...] = ()
    joint_velocity: tuple[float, ...] = ()
    actuator_command: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DemonstrationFrame:
    """One attempted governed transition, including failures."""

    episode_id: str
    frame_index: int
    request: dict[str, Any]
    before: dict[str, Any]
    after: dict[str, Any]
    accepted: bool
    result: dict[str, Any] | None = None
    rejection: dict[str, str] | None = None
    telemetry_before: SimulationTelemetry = field(default_factory=SimulationTelemetry)
    telemetry_after: SimulationTelemetry = field(default_factory=SimulationTelemetry)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "frame_index": self.frame_index,
            "request": dict(self.request),
            "before": dict(self.before),
            "after": dict(self.after),
            "accepted": self.accepted,
            "result": dict(self.result) if self.result is not None else None,
            "rejection": dict(self.rejection) if self.rejection is not None else None,
            "telemetry_before": self.telemetry_before.as_dict(),
            "telemetry_after": self.telemetry_after.as_dict(),
        }


class DemonstrationRecorder:
    """Record governed commands without introducing a second mutation path."""

    def __init__(self, world: DemonstrationWorld, *, episode_id: str, goal: str) -> None:
        if not isinstance(world, DemonstrationWorld):
            raise TypeError("world must implement DemonstrationWorld")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be non-empty")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be non-empty")
        self.world = world
        self.episode_id = episode_id.strip()
        self.goal = goal.strip()
        self._frames: list[DemonstrationFrame] = []

    @property
    def frames(self) -> tuple[DemonstrationFrame, ...]:
        return tuple(self._frames)

    def record(self, request: TransitionRequest) -> DemonstrationFrame:
        """Attempt one request through the authoritative world boundary.

        A rejection is recorded and re-raised.  That keeps caller behavior
        unchanged while ensuring an unsuccessful human/scripted attempt is
        not silently lost from the demonstration evidence.
        """

        if not isinstance(request, TransitionRequest):
            raise TypeError("record expects TransitionRequest")
        before = self.world.state()
        telemetry_before = self._telemetry()
        try:
            result = self.world.apply_transition(request)
        except Exception as exc:  # noqa: BLE001 - preserve governed rejection evidence
            after = self.world.state()
            frame = DemonstrationFrame(
                episode_id=self.episode_id,
                frame_index=len(self._frames),
                request=self._request_dict(request),
                before=before.as_dict(),
                after=after.as_dict(),
                accepted=False,
                rejection={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                telemetry_before=telemetry_before,
                telemetry_after=self._telemetry(),
            )
            self._frames.append(frame)
            raise

        after = self.world.state()
        frame = DemonstrationFrame(
            episode_id=self.episode_id,
            frame_index=len(self._frames),
            request=self._request_dict(request),
            before=before.as_dict(),
            after=after.as_dict(),
            accepted=bool(result.ok),
            result=self._result_dict(result),
            telemetry_before=telemetry_before,
            telemetry_after=self._telemetry(),
        )
        self._frames.append(frame)
        return frame

    def as_dict(self) -> dict[str, Any]:
        frames = [frame.as_dict() for frame in self._frames]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "goal": self.goal,
            "org_id": self.world.org_id,
            "mode": getattr(self.world, "mode", "unknown"),
            "frames": frames,
        }
        payload["content_hash"] = _content_hash(payload)
        return payload

    def write_jsonl(self, path: str | Path, *, overwrite: bool = False) -> str:
        """Write one episode header and one frame per line.

        The default is exclusive creation so a prior evidence artifact cannot
        be overwritten accidentally.  The returned digest covers the exact
        serialized episode payload, not filesystem metadata.
        """

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "x"
        payload = self.as_dict()
        lines = [{key: value for key, value in payload.items() if key != "frames"}]
        lines.extend(frame.as_dict() for frame in self._frames)
        with target.open(mode, encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")
        return str(payload["content_hash"])

    @staticmethod
    def _request_dict(request: TransitionRequest) -> dict[str, Any]:
        return {
            "step_id": request.step_id,
            "op": request.op,
            "args": dict(request.args),
            "expected_revision": request.expected_revision,
            "actor": request.actor,
            "org_id": request.org_id,
        }

    @staticmethod
    def _result_dict(result: TransitionResult) -> dict[str, Any]:
        return {
            "step_id": result.step_id,
            "ok": result.ok,
            "state_revision": result.state_revision,
            "detail": dict(result.detail),
        }

    def _telemetry(self) -> SimulationTelemetry:
        """Read optional MuJoCo arrays without requiring MuJoCo at import time."""

        data = getattr(self.world, "data", None)
        if data is None:
            return SimulationTelemetry()

        def values(name: str) -> tuple[float, ...]:
            value = getattr(data, name, ())
            return tuple(float(item) for item in value)

        controller_steps = getattr(self.world, "_controller_steps", None)
        if controller_steps is not None:
            controller_steps = int(controller_steps)
        sim_time = getattr(data, "time", None)
        return SimulationTelemetry(
            sim_time_s=float(sim_time) if sim_time is not None else None,
            controller_steps=controller_steps,
            joint_position=values("qpos"),
            joint_velocity=values("qvel"),
            actuator_command=values("ctrl"),
        )


def _content_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "DemonstrationFrame",
    "DemonstrationRecorder",
    "DemonstrationWorld",
    "SCHEMA_VERSION",
    "SimulationTelemetry",
]
