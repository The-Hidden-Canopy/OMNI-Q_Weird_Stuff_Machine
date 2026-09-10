"""Mock world (OQ-009 seed).

Stands in for camera + scene until the Intel/MuJoCo environment (OQ-006/007)
lands. Holds :class:`Detection` objects, produces a :class:`WorldState`
snapshot, and can be perturbed mid-run to force a replan.
"""

from __future__ import annotations

from dataclasses import replace

from .contracts import (
    Constraint,
    Detection,
    TransitionRejected,
    TransitionRequest,
    TransitionResult,
    WorldState,
)


class MockWorld:
    def __init__(
        self,
        objects: list[Detection],
        constraints: tuple[Constraint, ...] = (),
        *,
        org_id: str = "local-demo",
    ) -> None:
        self._objects: dict[str, Detection] = {o.object_id: o for o in objects}
        self._ownership: dict[str, str | None] = {o.object_id: None for o in objects}
        self._constraints: tuple[Constraint, ...] = constraints
        self.frame = 0
        self.revision = 0
        self.goal: str | None = None
        self.org_id = org_id

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
            revision=self.revision,
            ownership=dict(self._ownership),
            org_id=self.org_id,
        )

    # -- authoritative transitions ---------------------------------
    def start_mission(self, goal: str) -> None:
        self.goal = goal
        self.revision += 1

    def apply_transition(self, request: TransitionRequest) -> TransitionResult:
        """Apply one checked command to the mock's authoritative state.

        The method is intentionally the only production path that alters the
        simulated scene.  External perturbations below represent the world
        changing independently of Omni Q and still advance the revision.
        """
        if request.expected_revision != self.revision:
            raise TransitionRejected(
                f"{request.step_id}: expected revision {request.expected_revision}, "
                f"current revision is {self.revision}"
            )
        if request.org_id != self.org_id:
            raise TransitionRejected(
                f"{request.step_id}: request org {request.org_id} does not match world org {self.org_id}"
            )

        obj_id = request.args.get("object")
        if obj_id is not None and obj_id not in self._objects:
            raise TransitionRejected(f"{request.step_id}: unknown object {obj_id}")

        detail: dict[str, object] = {"before_revision": self.revision}
        if request.op == "PICK":
            if not obj_id:
                raise TransitionRejected(f"{request.step_id}: PICK requires object")
            if self._ownership[obj_id] is not None:
                raise TransitionRejected(f"{request.step_id}: {obj_id} is already held")
            self._ownership[obj_id] = request.actor or "unassigned"
            detail["grasped"] = obj_id
        elif request.op in {"MOVE", "PLACE"}:
            if not obj_id or not request.args.get("to"):
                raise TransitionRejected(f"{request.step_id}: {request.op} requires object and to")
            owner = self._ownership[obj_id]
            if owner is not None and request.actor is not None and owner != request.actor:
                raise TransitionRejected(
                    f"{request.step_id}: {obj_id} is held by {owner}, not {request.actor}"
                )
            self._objects[obj_id] = replace(self._objects[obj_id], zone=request.args["to"])
            self._ownership[obj_id] = None
            detail.update({"moved": obj_id, "to": request.args["to"]})
        elif request.op == "HANDOFF":
            if not obj_id or not request.args.get("to_actor"):
                raise TransitionRejected(f"{request.step_id}: HANDOFF requires object and to_actor")
            if self._ownership[obj_id] != request.actor:
                raise TransitionRejected(f"{request.step_id}: handoff actor does not hold {obj_id}")
            self._ownership[obj_id] = request.args["to_actor"]
            detail.update({"handed_off": obj_id, "to_actor": request.args["to_actor"]})
        else:
            detail["op"] = request.op

        self.revision += 1
        detail["state_revision"] = self.revision
        return TransitionResult(
            step_id=request.step_id,
            ok=True,
            state_revision=self.revision,
            detail=detail,
        )

    # -- independent world changes ---------------------------------
    def move_object(self, obj_id: str, zone: str) -> None:
        self._objects[obj_id] = replace(self._objects[obj_id], zone=zone)
        self._ownership[obj_id] = None
        self.revision += 1

    def perturb(self, obj_id: str, zone: str) -> None:
        """Someone nudged a part while Omni was working."""
        self.move_object(obj_id, zone)

    def add_constraint(self, constraint: Constraint) -> None:
        self._constraints = self._constraints + (constraint,)
        self.revision += 1

    def clear_constraints(self) -> None:
        self._constraints = ()
        self.revision += 1
