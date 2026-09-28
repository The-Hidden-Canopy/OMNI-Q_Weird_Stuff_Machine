"""Executable TableOps event profiles.

The profile is the product-level desired state.  This module deliberately
stops at symbolic placement zones when the observer has no metric pose; it
never turns a matching zone into proof that a millimetre tolerance was met.
The existing OMNI-Q planner and governed engine remain responsible for the
actual graph, authorization, execution, verification, and receipt.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    ActionAuthorization,
    Detection,
    MissionEnvelope,
    PlanGraph,
    ReceiptRecord,
    WorldState,
)
from .devices import default_devices
from .engine import OmniQ
from .events import EventBus
from .fakes import FakeManipulator, FakeObserver, FakeRecorder, FakeVerifier, RulePlanner
from .omni_planner import OmniPlanner
from .omni_reasoner import MockReasoner, OmniReferenceReasoner


PROFILE_ROOT = Path(__file__).resolve().parents[2] / "profiles"
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class ProfileError(ValueError):
    """The operator selected an invalid or malformed event profile."""


@dataclass(frozen=True)
class ProfileItem:
    cls: str
    relationship: str
    enabled: bool = True
    tolerance_mm: float | None = None
    spacing_mm: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DesiredPlacement:
    requirement_id: str
    object_id: str
    cls: str
    seat: int
    target_zone: str
    relationship: str
    tolerance_mm: float | None = None
    spacing_mm: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProfileResidual:
    kind: str
    requirement_id: str
    object_id: str
    cls: str
    seat: int
    current_zone: str | None
    target_zone: str
    reason: str
    actionable: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProfileDiff:
    profile_id: str
    required_count: int
    satisfied_count: int
    residuals: tuple[ProfileResidual, ...]
    metric_verification_gaps: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        return not self.residuals

    @property
    def actionable_count(self) -> int:
        return sum(item.actionable for item in self.residuals)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "required_count": self.required_count,
            "satisfied_count": self.satisfied_count,
            "residual_count": len(self.residuals),
            "actionable_count": self.actionable_count,
            "resolved": self.resolved,
            "residuals": [item.as_dict() for item in self.residuals],
            "metric_verification_gaps": list(self.metric_verification_gaps),
        }


@dataclass(frozen=True)
class EventProfile:
    profile_id: str
    venue: str
    guest_count: int
    place_setting: dict[str, ProfileItem]
    constraints: dict[str, Any] = field(default_factory=dict)

    @property
    def desired_placements(self) -> tuple[DesiredPlacement, ...]:
        placements: list[DesiredPlacement] = []
        for seat in range(1, self.guest_count + 1):
            target_zone = f"setting_{seat}"
            for cls, item in self.place_setting.items():
                if not item.enabled:
                    continue
                requirement_id = f"seat_{seat}.{cls}"
                placements.append(
                    DesiredPlacement(
                        requirement_id=requirement_id,
                        object_id=f"{cls}_{seat}",
                        cls=cls,
                        seat=seat,
                        target_zone=target_zone,
                        relationship=item.relationship,
                        tolerance_mm=item.tolerance_mm,
                        spacing_mm=item.spacing_mm,
                        attributes=dict(item.attributes),
                    )
                )
        return tuple(placements)

    @property
    def required_count(self) -> int:
        return len(self.desired_placements)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile_id,
            "venue": self.venue,
            "guest_count": self.guest_count,
            "required_count": self.required_count,
            "place_setting": {
                key: value.as_dict() for key, value in self.place_setting.items()
            },
            "constraints": dict(self.constraints),
        }

    def diff(self, world: WorldState) -> ProfileDiff:
        residuals: list[ProfileResidual] = []
        satisfied = 0
        metric_gaps: list[str] = []
        for required in self.desired_placements:
            observed = world.objects.get(required.object_id)
            if observed is None:
                residuals.append(ProfileResidual(
                    "missing", required.requirement_id, required.object_id,
                    required.cls, required.seat, None, required.target_zone,
                    "required object was not observed", False,
                ))
                continue
            if not observed.authoritative:
                residuals.append(ProfileResidual(
                    "stale", required.requirement_id, required.object_id,
                    required.cls, required.seat, observed.zone, required.target_zone,
                    f"observation status is {observed.status.value}", False,
                ))
                continue
            if observed.zone != required.target_zone:
                residuals.append(ProfileResidual(
                    "misplaced", required.requirement_id, required.object_id,
                    required.cls, required.seat, observed.zone, required.target_zone,
                    "observed zone differs from desired seat zone", True,
                ))
                continue
            satisfied += 1
            if required.tolerance_mm is not None and observed.pose is None:
                metric_gaps.append(
                    f"{required.requirement_id}: metric pose unavailable for "
                    f"{required.tolerance_mm:g}mm tolerance"
                )
        return ProfileDiff(
            profile_id=self.profile_id,
            required_count=self.required_count,
            satisfied_count=satisfied,
            residuals=tuple(residuals),
            metric_verification_gaps=tuple(metric_gaps),
        )

    def report(
        self,
        initial: ProfileDiff,
        final: ProfileDiff,
        *,
        metrics: Mapping[str, Any],
        actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        failed_actions = sum(action.get("state") == "failed" for action in actions)
        human_interventions = sum(
            event.get("kind") == "human.intervention" for event in actions
        )
        corrected = max(0, len(initial.residuals) - len(final.residuals))
        return {
            "profile": self.profile_id,
            "venue": self.venue,
            "guest_count": self.guest_count,
            "profile_constraints": dict(self.constraints),
            "status": "COMPLETE" if final.resolved else "HUMAN ASSISTANCE REQUIRED",
            "required_placements": self.required_count,
            "first_pass_correct": initial.satisfied_count,
            "self_corrected": corrected,
            "human_interventions": human_interventions,
            "replans": metrics.get("revisions", 0),
            "grasp_retries": failed_actions,
            "unresolved": len(final.residuals),
            "final_compliance": f"{final.satisfied_count}/{self.required_count}",
            "verification_scope": "metric" if not final.metric_verification_gaps else "symbolic-zone",
            "metric_verification_gaps": list(final.metric_verification_gaps),
            "initial_diff": initial.as_dict(),
            "final_diff": final.as_dict(),
        }


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return {}
    if value in {"{}", "[]"}:
        return {} if value == "{}" else []
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"null", "none"}:
            return None
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return value.strip("\"'")


def _minimal_yaml(text: str) -> dict[str, Any]:
    """Parse the small mapping-only subset used by checked-in profiles."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in line:
            raise ProfileError(f"unsupported profile YAML line: {raw!r}")
        key, raw_value = line.strip().split(":", 1)
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        key = key.strip().strip("\"'")
        value = _scalar(raw_value)
        if raw_value.strip() == "":
            value = {}
            stack.append((indent, value))
        parent[key] = value
    return root


def _read_profile_data(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = _minimal_yaml(text)
    else:
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ProfileError(f"profile {path.name} must contain a mapping")
    return value


def _profile_path(profile_id: str, root: Path = PROFILE_ROOT) -> Path:
    if not isinstance(profile_id, str) or not _PROFILE_NAME.fullmatch(profile_id):
        raise ProfileError("profile id must contain only letters, numbers, '_' or '-'")
    path = root / f"{profile_id}.yaml"
    if not path.is_file():
        raise ProfileError(f"unknown profile: {profile_id}")
    return path


def load_profile(profile_id: str, *, root: Path = PROFILE_ROOT) -> EventProfile:
    data = _read_profile_data(_profile_path(profile_id, root))
    declared = data.get("profile", profile_id)
    if declared != profile_id:
        raise ProfileError(f"profile id mismatch: requested {profile_id}, file declares {declared}")
    try:
        guest_count = int(data["guest_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProfileError("profile guest_count must be a positive integer") from exc
    if guest_count <= 0:
        raise ProfileError("profile guest_count must be positive")
    raw_items = data.get("place_setting")
    if not isinstance(raw_items, dict) or not raw_items:
        raise ProfileError("profile place_setting must be a non-empty mapping")
    items: dict[str, ProfileItem] = {}
    for cls, raw in raw_items.items():
        if not isinstance(raw, dict):
            raise ProfileError(f"place_setting.{cls} must be a mapping")
        items[str(cls)] = ProfileItem(
            cls=str(cls),
            relationship=str(raw.get("relationship", "seat_datum")),
            enabled=bool(raw.get("enabled", True)),
            tolerance_mm=(float(raw["tolerance_mm"]) if "tolerance_mm" in raw else None),
            spacing_mm=(float(raw["spacing_mm"]) if "spacing_mm" in raw else None),
            attributes={
                key: value for key, value in raw.items()
                if key not in {"relationship", "enabled", "tolerance_mm", "spacing_mm"}
            },
        )
    constraints = data.get("constraints", {})
    if not isinstance(constraints, dict):
        raise ProfileError("profile constraints must be a mapping")
    return EventProfile(
        profile_id=profile_id,
        venue=str(data.get("venue", "unspecified")),
        guest_count=guest_count,
        place_setting=items,
        constraints=dict(constraints),
    )


def list_profiles(*, root: Path = PROFILE_ROOT) -> list[EventProfile]:
    return [
        load_profile(path.stem, root=root)
        for path in sorted(root.glob("*.yaml"))
    ]


class ProfileWorld:
    """Profile-backed mock workspace with all inventory objects present.

    The first run intentionally leaves most objects in ``staging`` so the
    existing RulePlanner has real residual work. This is a product workflow
    fixture, not a claim that perception or physical tolerances are solved.
    """

    def __init__(self, profile: EventProfile) -> None:
        from .world import MockWorld

        objects: list[Detection] = []
        for index, required in enumerate(profile.desired_placements):
            current_zone = required.target_zone if index % 5 == 0 else "staging"
            objects.append(Detection(
                required.object_id,
                required.cls,
                zone=current_zone,
                target_zone=required.target_zone,
                conf=0.98,
            ))
        self._world = MockWorld(objects)
        self.mode = "tableops-profile"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._world, name)


class ProfileRecorder:
    """Adds profile diff/report data before the normal receipt is hashed."""

    def __init__(self, delegate: Any, profile: EventProfile, world: Any) -> None:
        self.delegate = delegate
        self.profile = profile
        self.world = world
        self.initial: ProfileDiff | None = None

    def set_initial(self, diff: ProfileDiff) -> None:
        self.initial = diff

    def authorize(self, authorization: ActionAuthorization) -> ActionAuthorization:
        return self.delegate.authorize(authorization)

    def record(
        self,
        run_id: str,
        goal: str,
        inputs: dict[str, Any],
        plan: PlanGraph,
        actions: list[dict[str, Any]],
        metrics: dict[str, Any],
        decisions: list[dict[str, Any]] | None = None,
        rejected: list[dict[str, Any]] | None = None,
    ) -> ReceiptRecord:
        initial = self.initial or self.profile.diff(self.world.state())
        final = self.profile.diff(self.world.state())
        report = self.profile.report(initial, final, metrics=metrics, actions=actions)
        enriched_metrics = {
            **metrics,
            # A profile run is resolved only when the selected profile's
            # desired state is resolved.  This prevents an Intel world with
            # unrelated objects from being presented as a completed profile.
            "resolved": final.resolved,
            "profile_report": report,
        }
        enriched_inputs = {
            **inputs,
            "profile": self.profile.as_dict(),
            "profile_initial_diff": initial.as_dict(),
        }
        return self.delegate.record(
            run_id, goal, enriched_inputs, plan, actions, enriched_metrics,
            decisions=decisions, rejected=rejected,
        )


class ProfileMission:
    """Expose a normal OmniQ session plus profile lifecycle evidence."""

    def __init__(self, engine: OmniQ, profile: EventProfile) -> None:
        self._engine = engine
        self.profile = profile
        self.bus = engine.bus
        self.world = engine.world
        self._profile_recorder = engine.recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def run(self, goal: str) -> ReceiptRecord:
        initial = self.profile.diff(self.world.state())
        if isinstance(self._profile_recorder, ProfileRecorder):
            self._profile_recorder.set_initial(initial)
        self.bus.publish(
            "profile.loaded",
            profile=self.profile.as_dict(),
            desired_count=self.profile.required_count,
        )
        self.bus.publish(
            "mission.diff",
            profile_id=self.profile.profile_id,
            diff=initial.as_dict(),
            state_revision=self.world.revision,
        )
        return self._engine.run(goal)


def attach_profile(engine: OmniQ, profile: EventProfile) -> ProfileMission:
    engine.recorder = ProfileRecorder(engine.recorder, profile, engine.world)
    return ProfileMission(engine, profile)


def build_profile_engine(
    profile_id: str,
    bus: EventBus,
    *,
    reasoner_mode: str = "off",
    checkpoint: str | Path | None = None,
    receipt: str | Path | None = None,
    device: str = "cpu",
) -> ProfileMission:
    profile = load_profile(profile_id)
    world = ProfileWorld(profile)
    fallback = RulePlanner()
    if reasoner_mode in {"", "0", "off", "rule"}:
        planner: Any = fallback
    elif reasoner_mode == "mock":
        planner = OmniPlanner(MockReasoner(), fallback=fallback)
    elif reasoner_mode == "omni":
        if not checkpoint or not receipt:
            raise ProfileError(
                "OMNIQ_OMNI_REASONER=omni requires checkpoint and receipt"
            )
        planner = OmniPlanner(
            OmniReferenceReasoner(checkpoint, receipt, device=device),
            fallback=fallback,
        )
    else:
        raise ProfileError(f"unknown OMNIQ_OMNI_REASONER mode: {reasoner_mode!r}")
    engine = OmniQ(
        world=world,
        observer=FakeObserver(),
        planner=planner,
        manipulator=FakeManipulator(world),
        verifier=FakeVerifier(),
        device=default_devices(),
        recorder=FakeRecorder(),
        bus=bus,
        envelope=MissionEnvelope(mission_id=f"profile-{profile.profile_id}"),
    )
    return attach_profile(engine, profile)


def profile_from_engine(engine: Any) -> EventProfile | None:
    return getattr(engine, "profile", None)


__all__ = [
    "DesiredPlacement",
    "EventProfile",
    "ProfileDiff",
    "ProfileError",
    "ProfileItem",
    "ProfileMission",
    "ProfileResidual",
    "attach_profile",
    "build_profile_engine",
    "list_profiles",
    "load_profile",
]
