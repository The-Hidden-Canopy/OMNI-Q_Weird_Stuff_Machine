"""Mock world (OQ-009 seed).

Stands in for camera + scene until the Intel/MuJoCo environment (OQ-006/007)
lands. Holds :class:`Detection` objects, produces a :class:`WorldState`
snapshot, and can be perturbed mid-run to force a replan.
"""

from __future__ import annotations

from dataclasses import replace

from .contracts import Constraint, Detection, WorldState


class MockWorld:
    def __init__(self, objects: list[Detection], constraints: tuple[Constraint, ...] = ()) -> None:
        self._objects: dict[str, Detection] = {o.object_id: o for o in objects}
        self._constraints: tuple[Constraint, ...] = constraints
        self.frame = 0
        self.goal: str | None = None

    # -- construction --------------------------------------------------
    @classmethod
    def sample(cls) -> "MockWorld":
        return cls([
            Detection("connector_2", "connector", zone="A", target_zone="bin"),
            Detection("sleeve_1", "sleeve", zone="bin", target_zone="bin"),
            Detection("cable_4", "cable", zone="B", target_zone="B"),
            Detection("plate_1", "plate", zone="tray", target_zone="setting_1"),
        ])

    # -- snapshot ----------------------------------------------------
    def state(self) -> WorldState:
        self.frame += 1
        return WorldState(
            frame=self.frame,
            objects=dict(self._objects),
            goal=self.goal,
            constraints=self._constraints,
        )

    # -- mutation --------------------------------------------------
    def move_object(self, obj_id: str, zone: str) -> None:
        self._objects[obj_id] = replace(self._objects[obj_id], zone=zone)

    def perturb(self, obj_id: str, zone: str) -> None:
        """Someone nudged a part while Omni was working."""
        self.move_object(obj_id, zone)

    def add_constraint(self, constraint: Constraint) -> None:
        self._constraints = self._constraints + (constraint,)

    def clear_constraints(self) -> None:
        self._constraints = ()
