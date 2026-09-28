"""OQ-019 — table-layout evaluator.

Scores a finished table setting against a target layout and returns
positional/orientation errors plus an overall PASS/FAIL — the acceptance signal
for OQ-016/017 (did the arms actually set the table?), the breakage tests in
OQ-021, and the acceptance suite in OQ-038.

Dependency posture (OQ-007 scene pack and OQ-009 pose-state are unbuilt): the
evaluator consumes a neutral, frozen *input* shape — :class:`PlacedObject` — and
does not import world state. Whoever observes the scene (mock fixture today,
YOLO+MuJoCo adapter when OQ-007 lands, world-state adapter when OQ-009 lands)
converts their representation into ``PlacedObject`` and calls
:func:`evaluate_layout`. Metres and degrees everywhere; yaw is rotation about
the vertical axis in [-180, 180).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

DEFAULT_POS_TOL_M = 0.01
DEFAULT_ORI_TOL_DEG = 5.0


@dataclass(frozen=True)
class PlacedObject:
    """One object as observed on the table."""

    object_id: str
    cls: str
    position: tuple[float, float, float]     # metres, scene frame
    orientation_deg: float = 0.0             # yaw about vertical
    zone: str | None = None


@dataclass(frozen=True)
class PlacementSpec:
    """Where one object must end up. Any subset of checks may be specified:
    position, orientation, zone — each given check must pass."""

    object_id: str
    target_zone: str | None = None
    position: tuple[float, float, float] | None = None
    orientation_deg: float | None = None
    pos_tol_m: float = DEFAULT_POS_TOL_M
    ori_tol_deg: float = DEFAULT_ORI_TOL_DEG

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlacementSpec":
        pos = d.get("position")
        return cls(
            object_id=d["object_id"],
            target_zone=d.get("target_zone"),
            position=tuple(pos) if pos is not None else None,
            orientation_deg=d.get("orientation_deg"),
            pos_tol_m=d.get("pos_tol_m", DEFAULT_POS_TOL_M),
            ori_tol_deg=d.get("ori_tol_deg", DEFAULT_ORI_TOL_DEG),
        )


def load_layout_spec(specs: Iterable[dict[str, Any]]) -> list[PlacementSpec]:
    return [PlacementSpec.from_dict(s) for s in specs]


@dataclass(frozen=True)
class ObjectVerdict:
    object_id: str
    cls: str
    passed: bool
    pos_err_m: float | None          # euclidean distance to target (None if no target)
    ori_err_deg: float | None        # wraparound-aware yaw error (None if no target)
    zone_ok: bool | None             # None if no zone requirement
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class LayoutReport:
    passed: bool
    objects: tuple[ObjectVerdict, ...]
    missing: tuple[str, ...]                       # in spec, not on the table
    max_pos_err_m: float | None
    mean_pos_err_m: float | None
    max_ori_err_deg: float | None
    n_passed: int
    n_failed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def yaw_error(actual_deg: float, target_deg: float) -> float:
    """Minimal signed difference between two yaws, in degrees."""
    return (actual_deg - target_deg + 180.0) % 360.0 - 180.0


def _pos_error(actual: tuple[float, float, float],
               target: tuple[float, float, float]) -> float:
    return math.dist(actual, target)


def evaluate_layout(
    setting: Iterable[PlacedObject],
    spec: Iterable[PlacementSpec],
    *,
    allow_extras: bool = False,
) -> LayoutReport:
    """Score ``setting`` against ``spec``.

    Every specified check must pass for an object to pass. Extras (objects on
    the table that the layout does not mention) fail the report by default —
    "set the table" means nothing left lying around.
    """
    by_id = {o.object_id: o for o in setting}
    specs = list(spec)
    verdicts: list[ObjectVerdict] = []
    missing: list[str] = []

    for want in specs:
        obj = by_id.get(want.object_id)
        if obj is None:
            missing.append(want.object_id)
            continue

        reasons: list[str] = []
        pos_err = ori_err = zone_ok = None

        if want.position is not None:
            pos_err = _pos_error(obj.position, want.position)
            if pos_err > want.pos_tol_m:
                reasons.append(
                    f"position error {pos_err * 1000:.1f} mm > tol {want.pos_tol_m * 1000:.1f} mm"
                )
        if want.orientation_deg is not None:
            ori_err = abs(yaw_error(obj.orientation_deg, want.orientation_deg))
            if ori_err > want.ori_tol_deg:
                reasons.append(
                    f"orientation error {ori_err:.1f} deg > tol {want.ori_tol_deg:.1f} deg"
                )
        if want.target_zone is not None:
            zone_ok = obj.zone == want.target_zone
            if not zone_ok:
                reasons.append(f"zone {obj.zone!r} != target {want.target_zone!r}")

        passed = not reasons
        verdicts.append(ObjectVerdict(
            object_id=obj.object_id, cls=obj.cls, passed=passed,
            pos_err_m=round(pos_err, 5) if pos_err is not None else None,
            ori_err_deg=round(ori_err, 2) if ori_err is not None else None,
            zone_ok=zone_ok, reasons=tuple(reasons),
        ))

    extra_ids = [oid for oid in by_id if oid not in {w.object_id for w in specs}]
    if not allow_extras:
        for oid in extra_ids:
            obj = by_id[oid]
            verdicts.append(ObjectVerdict(
                object_id=oid, cls=obj.cls, passed=False,
                pos_err_m=None, ori_err_deg=None, zone_ok=None,
                reasons=("object on table is not part of the layout",),
            ))

    pos_errors = [v.pos_err_m for v in verdicts if v.pos_err_m is not None]
    ori_errors = [v.ori_err_deg for v in verdicts if v.ori_err_deg is not None]
    n_failed = sum(1 for v in verdicts if not v.passed) + len(missing)

    return LayoutReport(
        passed=not missing and n_failed == 0,
        objects=tuple(verdicts),
        missing=tuple(missing),
        max_pos_err_m=max(pos_errors) if pos_errors else None,
        mean_pos_err_m=(round(sum(pos_errors) / len(pos_errors), 5)
                        if pos_errors else None),
        max_ori_err_deg=max(ori_errors) if ori_errors else None,
        n_passed=len(verdicts) - sum(1 for v in verdicts if not v.passed),
        n_failed=n_failed,
    )
