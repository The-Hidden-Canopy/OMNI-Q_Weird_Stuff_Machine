"""OQ-012 / OQ-013 / OQ-044 — bimanual task scheduler.

Takes a (mostly linear) :class:`~omni_q.contracts.PlanGraph` from the planner
plus a :class:`~omni_q.contracts.WorldState` and works out:

- **which arm** runs each manipulate step (OQ-044 — chosen by reach + load,
  never a fixed left/right role),
- **which steps run concurrently** (wave layering + gripper occupancy), and
- **where a shared-workspace collision forced serialisation** (OQ-013 barriers).

Standalone by design: it only *reads* ``PlanGraph`` / ``Step`` / ``WorldState``
and hands work back through :meth:`Schedule.annotate`, which returns a *new*
graph the current engine executes unchanged.

Design authority: ``integrations/intel/so101_capability_map.md`` — two SO-101
arms are a mirrored pair about the table centreline; reach is rarely the binding
constraint, so scheduling worries about **gripper occupancy and shared-workspace
collisions**, not reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .contracts import PlanGraph, Step, WorldState

ARMS: tuple[str, str] = ("left", "right")


# ---------------------------------------------------------------------------
# table layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """A named table area. ``x_center`` is signed across the centreline:
    ``< 0`` right side, ``> 0`` left side, ``~0`` shared centre."""

    id: str
    x_center: float

    def conflicts_with(self, other: "Region", tol: float) -> bool:
        return self.id == other.id or abs(self.x_center - other.x_center) <= tol


# Covers the MockWorld.sample zones plus generic table zones. An unknown zone is
# treated as the contention-prone centre (see ``_region_for``).
DEFAULT_LAYOUT: dict[str, Region] = {
    "A": Region("A", -0.25),
    "bin": Region("bin", -0.30),
    "setting_2": Region("setting_2", -0.10),
    "B": Region("B", 0.25),
    "tray": Region("tray", 0.30),
    "setting_1": Region("setting_1", 0.10),
    "drawer": Region("drawer", 0.0),
    "center": Region("center", 0.0),
    "centre": Region("centre", 0.0),
    "table": Region("table", 0.0),
    "home_left": Region("home_left", 0.35),
    "home_right": Region("home_right", -0.35),
}


def arm_reaches(arm: str, region: Region, overlap: float) -> bool:
    """A mirrored pair: ``left`` covers ``x >= -overlap``, ``right`` covers
    ``x <= +overlap``; the centre band is reachable by both."""
    if arm == "left":
        return region.x_center >= -overlap
    if arm == "right":
        return region.x_center <= overlap
    return False


# ---------------------------------------------------------------------------
# schedule value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduledStep:
    step_id: str
    arm: str | None          # None for observe/verify barrier steps
    region_id: str | None
    wave: int


@dataclass
class Wave:
    index: int
    steps: list[ScheduledStep]


@dataclass(frozen=True)
class Barrier:
    between: tuple[str, str]
    kind: str                # workspace | gripper | verify | reach
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"between": list(self.between), "kind": self.kind, "reason": self.reason}


@dataclass
class Schedule:
    waves: list[Wave]
    barriers: list[Barrier]
    arm_timeline: dict[str, list[str]]
    metrics: dict[str, int]
    assignment: dict[str, str | None]      # step_id -> arm
    regions: dict[str, str | None]         # step_id -> region id

    # -- integration seam ------------------------------------------------
    def annotate(self, graph: PlanGraph) -> PlanGraph:
        """Return a copy of ``graph`` with each manipulate ``Step.arm`` filled
        and every step also depending on the whole previous wave, so the
        current dep-gated engine executes the waves in order."""
        wave_of = {ss.step_id: ss.wave for w in self.waves for ss in w.steps}
        wave_members: dict[int, list[str]] = {}
        for w in self.waves:
            wave_members[w.index] = [ss.step_id for ss in w.steps]

        out = PlanGraph(goal=graph.goal, revision=graph.revision)
        for s in graph.steps:
            deps = set(s.deps)
            wv = wave_of.get(s.id)
            if wv is not None and wv > 0:
                deps.update(wave_members.get(wv - 1, ()))
            deps.discard(s.id)
            out.steps.append(replace(
                s,
                arm=self.assignment.get(s.id, s.arm) if s.contract == "manipulate" else s.arm,
                deps=tuple(sorted(deps)),
            ))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "waves": [
                {"index": w.index,
                 "steps": [vars(ss) for ss in w.steps]}
                for w in self.waves
            ],
            "barriers": [b.as_dict() for b in self.barriers],
            "arm_timeline": {k: list(v) for k, v in self.arm_timeline.items()},
            "metrics": dict(self.metrics),
        }


# ---------------------------------------------------------------------------
# scheduling
# ---------------------------------------------------------------------------


def _region_for(step: Step, world: WorldState, layout: dict[str, Region]) -> Region:
    """MOVE/PLACE -> destination zone; anything else with an object -> that
    object's current zone; otherwise the centre."""
    def region(zone: str | None) -> Region:
        if zone is None:
            return Region("center", 0.0)
        return layout.get(zone, Region(zone, 0.0))

    to_zone = step.args.get("to")
    if to_zone is not None:
        return region(to_zone)
    obj = step.args.get("object")
    if obj is not None and obj in world.objects:
        return region(world.objects[obj].zone)
    return Region("center", 0.0)


def _object_chains(graph: PlanGraph) -> dict[str, list[Step]]:
    """Group manipulate steps by the object they act on, in graph order."""
    chains: dict[str, list[Step]] = {}
    for s in graph.steps:
        if s.contract != "manipulate":
            continue
        obj = s.args.get("object")
        if obj is None:
            continue
        chains.setdefault(obj, []).append(s)
    return chains


def _prefer_arm(world: WorldState) -> str | None:
    return next((c.value for c in world.constraints if c.kind == "prefer_arm"), None)


def _assign_arms(
    graph: PlanGraph,
    world: WorldState,
    layout: dict[str, Region],
    arms: tuple[str, str],
    overlap: float,
) -> tuple[dict[str, str | None], dict[str, str | None], list[Barrier]]:
    assignment: dict[str, str | None] = {}
    region_id: dict[str, str | None] = {}
    barriers: list[Barrier] = []
    load = {a: 0 for a in arms}
    prefer = _prefer_arm(world)
    chains = _object_chains(graph)
    chained_ids = {s.id for steps in chains.values() for s in steps}

    def reachable(arm: str, regions: list[Region]) -> bool:
        return all(arm_reaches(arm, r, overlap) for r in regions)

    def choose(regions: list[Region], explicit: str | None) -> tuple[str, list[Barrier]]:
        notes: list[Barrier] = []
        if explicit in arms:
            if not reachable(explicit, regions):
                notes.append(Barrier((explicit, explicit), "reach",
                                     f"explicit arm {explicit} cannot reach "
                                     f"{[r.id for r in regions]}"))
            return explicit, notes
        if prefer in arms and reachable(prefer, regions):
            return prefer, notes
        fit = [a for a in arms if reachable(a, regions)]
        if not fit:
            # chain spans both arms -> needs a handoff (out of scope here)
            notes.append(Barrier(("left", "right"), "reach",
                                 f"{[r.id for r in regions]} spans both arms; needs handoff"))
            fit = list(arms)
        arm = min(fit, key=lambda a: (load[a], arms.index(a)))
        return arm, notes

    # object chains first (pick + move + present share one arm)
    for obj, steps in chains.items():
        regions = []
        for s in steps:
            r = _region_for(s, world, layout)
            region_id[s.id] = r.id
            regions.append(r)
        explicit = next((s.arm for s in steps if s.arm in arms), None)
        arm, notes = choose(regions, explicit)
        barriers.extend(notes)
        for s in steps:
            assignment[s.id] = arm
        load[arm] += len(steps)

    # standalone manipulate steps (e.g. a bare HANDOFF / PRESENT)
    for s in graph.steps:
        if s.contract != "manipulate" or s.id in chained_ids:
            if s.contract != "manipulate":
                assignment[s.id] = None
                region_id[s.id] = None
            continue
        r = _region_for(s, world, layout)
        region_id[s.id] = r.id
        arm, notes = choose([r], s.arm if s.arm in arms else None)
        barriers.extend(notes)
        assignment[s.id] = arm
        load[arm] += 1

    return assignment, region_id, barriers


def schedule(
    graph: PlanGraph,
    world: WorldState,
    *,
    layout: dict[str, Region] | None = None,
    arms: tuple[str, str] = ARMS,
    overlap: float = 0.15,
) -> Schedule:
    layout = layout or DEFAULT_LAYOUT
    assignment, region_id, barriers = _assign_arms(graph, world, layout, arms, overlap)
    region_obj = {
        s.id: (layout.get(region_id[s.id], Region(region_id[s.id] or "center", 0.0)))
        for s in graph.steps if region_id.get(s.id) is not None
    }

    by_id = {s.id: s for s in graph.steps}
    done: set[str] = set()
    waves: list[Wave] = []
    arm_hold: dict[str, str | None] = {a: None for a in arms}
    handoffs = 0
    guard = 0

    while len(done) < len(graph.steps):
        guard += 1
        if guard > len(graph.steps) + 5:
            raise RuntimeError("scheduler did not converge")

        ready = [s for s in graph.steps
                 if s.id not in done and all(d in done for d in s.deps)]
        if not ready:
            raise ValueError("unschedulable graph: unmet deps or cycle")

        # observe/verify quiesce both arms -> solo wave
        barrier_step = next((s for s in ready if s.contract != "manipulate"), None)
        if barrier_step is not None:
            waves.append(Wave(len(waves),
                              [ScheduledStep(barrier_step.id, None, None, len(waves))]))
            if len(waves) >= 2:
                barriers.append(Barrier((barrier_step.id, barrier_step.id), "verify",
                                        f"{barrier_step.id} requires both arms idle"))
            done.add(barrier_step.id)
            continue

        placed: list[ScheduledStep] = []
        used_arms: set[str] = set()
        used_regions: list[tuple[Region, str]] = []

        for s in ready:
            arm = assignment[s.id]
            if s.op == "HANDOFF":
                if used_arms:
                    continue
                obj = s.args.get("object")
                placed.append(ScheduledStep(s.id, arm, region_id.get(s.id), len(waves)))
                used_arms.update(arms)
                to_actor = s.args.get("to_actor") or s.args.get("to")
                if obj is not None:
                    arm_hold = {a: (obj if a == to_actor else (None if arm_hold[a] == obj else arm_hold[a]))
                                for a in arms}
                handoffs += 1
                done.add(s.id)
                continue

            if arm in used_arms:
                continue
            if s.op == "PICK" and arm_hold[arm] is not None:
                continue  # gripper already occupied

            reg = region_obj.get(s.id, Region("center", 0.0))
            clash = next(((r, owner) for (r, owner) in used_regions
                          if r.conflicts_with(reg, overlap)), None)
            if clash is not None:
                barriers.append(Barrier((clash[1], s.id), "workspace",
                                        f"{s.id} in {reg.id} conflicts with {clash[1]} in {clash[0].id}"))
                continue

            placed.append(ScheduledStep(s.id, arm, reg.id, len(waves)))
            used_arms.add(arm)
            used_regions.append((reg, s.id))
            done.add(s.id)
            if s.op == "PICK":
                arm_hold[arm] = s.args.get("object")
            elif s.op in {"MOVE", "PLACE"}:
                arm_hold[arm] = None

        if not placed:
            # every ready manipulate step conflicts with another - force one
            s = ready[0]
            reg = region_obj.get(s.id, Region("center", 0.0))
            placed.append(ScheduledStep(s.id, assignment[s.id], reg.id, len(waves)))
            done.add(s.id)
            if s.op == "PICK":
                arm_hold[assignment[s.id]] = s.args.get("object")
            elif s.op in {"MOVE", "PLACE"}:
                arm_hold[assignment[s.id]] = None

        waves.append(Wave(len(waves), placed))

    arm_timeline: dict[str, list[str]] = {a: [] for a in arms}
    for w in waves:
        for ss in w.steps:
            if ss.arm in arm_timeline:
                arm_timeline[ss.arm].append(ss.step_id)

    manipulate_ids = {s.id for s in graph.steps if s.contract == "manipulate"}
    metrics = {
        "waves": len(waves),
        "max_parallelism": max(
            (sum(1 for ss in w.steps if ss.step_id in manipulate_ids) for w in waves),
            default=0,
        ),
        "handoffs": handoffs,
        "serialized_conflicts": sum(1 for b in barriers if b.kind == "workspace"),
    }
    return Schedule(waves, barriers, arm_timeline, metrics, assignment, region_id)


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def _main() -> None:  # pragma: no cover - manual
    from .fakes import RulePlanner
    from .world import MockWorld

    world = MockWorld.sample().state()
    graph = RulePlanner().plan("tidy the workspace", world)
    sch = schedule(graph, world)
    print("metrics    :", sch.metrics)
    for w in sch.waves:
        print(f"wave {w.index}: " + ", ".join(
            f"{ss.step_id}[{ss.arm or '-'}@{ss.region_id or '-'}]" for ss in w.steps))
    print("timelines  :", sch.arm_timeline)
    for b in sch.barriers:
        print("barrier    :", b.kind, b.reason)


if __name__ == "__main__":
    _main()
