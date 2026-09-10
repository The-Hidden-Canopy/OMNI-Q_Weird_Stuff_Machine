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

from . import actions as _actions
from .contracts import PlanGraph, Step, WorldState

ARMS: tuple[str, str] = ("left", "right")

# Steps that exist only for showmanship (OQ-015). The action registry
# (OQ-HAND-*) is authoritative; this literal set is a fallback for ops it does
# not know. Flourishes never change the goal and are dropped rather than delay
# it past one slack wave.
FLOURISH_OPS: frozenset[str] = frozenset({"PRESENT", "FLOURISH", "SHOWCASE", "SPIN_SHOW"})

# ops that occupy *both* arms for their wave
_BOTH_ARMS = _actions.ActionCategory.BIMANUAL

# ops that free / occupy the assigned arm's gripper
_RELEASES = {"MOVE", "PLACE", "RELEASE", "DRAG", "CENTER_ON_MARK", "PLACE_LEFT_OF",
             "PLACE_RIGHT_OF", "PLACE_ABOVE_RIGHT_OF", "NEST", "STACK", "SPREAD"}
_GRABS = {"PICK", "PINCH_GRIP", "WIDE_GRIP", "EDGE_GRIP", "ASSIST_GRASP"}


def _is_flourish(step: Step) -> bool:
    return step.contract == "manipulate" and (
        step.op in FLOURISH_OPS or _actions.is_style(step.op))


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
# treated as the contention-prone centre (see ``_region_for``) -- every unmapped
# zone name falls back to x_center=0.0, so *any* two unmapped zones "conflict"
# regardless of how different their names are. Real x-centers below fix that
# for the Intel table-setting pack (OQ-007, ``intel_sim.IntelTableWorld``).
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
    # Intel table-setting object pack (OQ-007, IntelTableWorld) -- start and
    # target zones, spread out so the scheduler's conflict/parallelism
    # reasoning reflects the scene instead of every unmapped zone colliding
    # at x_center=0. Signs follow THIS module's convention (< 0 right, > 0
    # left, matching arm_reaches()), matched to IntelTablePlanner's fixed
    # arm choice per object (plate_1/fork_1/napkin_1 -> left, cup_1/spoon_1
    # -> right). NOTE: this is the opposite sign of dual_so101_xml()'s real
    # MJCF x-coordinates, where the "left" arm body is mounted at x<0 -- the
    # scheduler's regions are a reachability abstraction, not a physical
    # coordinate frame, and nothing in intel_sim.py reads Region.x_center for
    # real motion (apply_transition only branches on the actor string).
    "tray_plate": Region("tray_plate", 0.13),
    "tray_cup": Region("tray_cup", -0.16),
    "tray_napkin": Region("tray_napkin", 0.22),
    "upper_right": Region("upper_right", -0.22),
    "left": Region("left", 0.20),
    "right": Region("right", -0.30),
    "lower_left": Region("lower_left", 0.28),
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
    flourish: bool = False


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
    dropped: list[str] = field(default_factory=list)   # flourishes with no slack
    style_mode: str = "unset"              # last STYLE constraint, normalised

    # -- integration seam ------------------------------------------------
    def annotate(self, graph: PlanGraph) -> PlanGraph:
        """Return a copy of ``graph`` with each manipulate ``Step.arm`` filled
        and every step also depending on the whole previous wave, so the
        current dep-gated engine executes the waves in order. Dropped
        flourishes are omitted."""
        wave_of = {ss.step_id: w.index for w in self.waves for ss in w.steps}
        # Flourishes never *block* another step: they are excluded from the
        # previous-wave dependency injection (OQ-015 — goal path unchanged).
        wave_members: dict[int, list[str]] = {
            w.index: [ss.step_id for ss in w.steps if not ss.flourish]
            for w in self.waves
        }
        dropped = set(self.dropped)

        out = PlanGraph(goal=graph.goal, revision=graph.revision)
        for s in graph.steps:
            if s.id in dropped:
                continue
            deps = set(s.deps) - dropped
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
                {"index": w.index, "steps": [vars(ss) for ss in w.steps]}
                for w in self.waves
            ],
            "barriers": [b.as_dict() for b in self.barriers],
            "arm_timeline": {k: list(v) for k, v in self.arm_timeline.items()},
            "dropped": list(self.dropped),
            "style_mode": self.style_mode,
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

    # every step gets a region up front
    for s in graph.steps:
        region_id[s.id] = _region_for(s, world, layout).id if s.contract == "manipulate" else None

    def regions_of(steps: list[Step]) -> list[Region]:
        return [layout.get(region_id[s.id], Region(region_id[s.id] or "center", 0.0))
                for s in steps]

    def reachable(arm: str, regions: list[Region]) -> bool:
        return all(arm_reaches(arm, r, overlap) for r in regions)

    def other(arm: str) -> str:
        return arms[1] if arm == arms[0] else arms[0]

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
            notes.append(Barrier((arms[0], arms[1]), "reach",
                                 f"{[r.id for r in regions]} spans both arms; needs handoff"))
            fit = list(arms)
        return min(fit, key=lambda a: (load[a], arms.index(a))), notes

    for _obj, steps in chains.items():
        hand_idx = next((i for i, s in enumerate(steps) if s.op == "HANDOFF"), None)
        if hand_idx is None:
            explicit = next((s.arm for s in steps if s.arm in arms), None)
            arm, notes = choose(regions_of(steps), explicit)
            barriers.extend(notes)
            for s in steps:
                assignment[s.id] = arm
            load[arm] += len(steps)
            continue

        # a hand-off splits the chain: giver keeps the pre-steps + the HANDOFF,
        # the named receiver takes everything after it (OQ-044 across arms).
        pre, post = steps[: hand_idx + 1], steps[hand_idx + 1:]
        giver_explicit = next((s.arm for s in pre if s.arm in arms), None)
        giver, notes = choose(regions_of(pre[:-1]) or regions_of(pre), giver_explicit)
        barriers.extend(notes)
        for s in pre:
            assignment[s.id] = giver
        load[giver] += len(pre)

        req = steps[hand_idx].args.get("to_actor") or steps[hand_idx].args.get("to")
        receiver = req if req in arms else other(giver)
        if post and not reachable(receiver, regions_of(post)):
            barriers.append(Barrier((giver, receiver), "reach",
                                    f"post-handoff arm {receiver} cannot reach "
                                    f"{[r.id for r in regions_of(post)]}"))
        for s in post:
            assignment[s.id] = receiver
        load[receiver] += len(post)

    for s in graph.steps:
        if s.contract != "manipulate":
            assignment[s.id] = None
            continue
        if s.id in chained_ids:
            continue
        arm, notes = choose(regions_of([s]), s.arm if s.arm in arms else None)
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
    allow_flourish_wave: bool = True,
) -> Schedule:
    layout = layout or DEFAULT_LAYOUT
    assignment, region_id, barriers = _assign_arms(graph, world, layout, arms, overlap)
    region_obj = {
        s.id: (layout.get(region_id[s.id], Region(region_id[s.id] or "center", 0.0)))
        for s in graph.steps if region_id.get(s.id) is not None
    }

    # A flourish is a droppable *leaf*: if another step depends on it (e.g. a
    # CO_ROTATE inside a spin routine) it stays mandatory.
    depended_on = {d for s in graph.steps for d in s.deps}
    flourishes = [s for s in graph.steps if _is_flourish(s) and s.id not in depended_on]
    _flourish_ids = {s.id for s in flourishes}
    mandatory = [s for s in graph.steps if s.id not in _flourish_ids]

    done: set[str] = set()
    waves: list[Wave] = []
    arm_hold: dict[str, str | None] = {a: None for a in arms}
    handoffs = 0
    guard = 0

    while len(done) < len(mandatory):
        guard += 1
        if guard > len(mandatory) + 5:
            raise RuntimeError("scheduler did not converge")

        ready = [s for s in mandatory
                 if s.id not in done and all(d in done for d in s.deps)]
        if not ready:
            raise ValueError("unschedulable graph: unmet deps or cycle")

        # observe/verify quiesce both arms -> solo wave
        barrier_step = next((s for s in ready if s.contract != "manipulate"), None)
        if barrier_step is not None:
            waves.append(Wave(len(waves),
                              [ScheduledStep(barrier_step.id, None, None)]))
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
            if _actions.category_of(s.op) is _BOTH_ARMS:   # HANDOFF, CO_ROTATE, ...
                if used_arms:
                    continue
                obj = s.args.get("object")
                placed.append(ScheduledStep(s.id, arm, region_id.get(s.id)))
                used_arms.update(arms)
                if s.op == "HANDOFF":
                    to_actor = s.args.get("to_actor") or s.args.get("to")
                    if obj is not None:
                        arm_hold = {a: (obj if a == to_actor
                                        else (None if arm_hold[a] == obj else arm_hold[a]))
                                    for a in arms}
                    handoffs += 1
                done.add(s.id)
                continue

            if arm in used_arms:
                continue
            if s.op in _GRABS and arm_hold[arm] is not None:
                continue  # gripper already occupied
            obj = s.args.get("object")
            if _actions.needs_grasp(s.op) and obj and obj in arm_hold.values() \
                    and arm_hold[arm] != obj:
                continue  # a different arm is holding the object

            reg = region_obj.get(s.id, Region("center", 0.0))
            clash = next(((r, owner) for (r, owner) in used_regions
                          if r.conflicts_with(reg, overlap)), None)
            if clash is not None:
                barriers.append(Barrier((clash[1], s.id), "workspace",
                                        f"{s.id} in {reg.id} conflicts with {clash[1]} in {clash[0].id}"))
                continue

            placed.append(ScheduledStep(s.id, arm, reg.id))
            used_arms.add(arm)
            used_regions.append((reg, s.id))
            done.add(s.id)
            if s.op in _GRABS:
                arm_hold[arm] = obj
            elif s.op in _RELEASES:
                arm_hold[arm] = None

        if not placed:
            # every ready manipulate step conflicts with another - force one
            s = ready[0]
            reg = region_obj.get(s.id, Region("center", 0.0))
            placed.append(ScheduledStep(s.id, assignment[s.id], reg.id))
            done.add(s.id)
            if s.op in _GRABS:
                arm_hold[assignment[s.id]] = s.args.get("object")
            elif s.op in _RELEASES:
                arm_hold[assignment[s.id]] = None

        waves.append(Wave(len(waves), placed))

    # -- OQ-015: place flourishes in slack, never delaying the goal --------
    mode = _style_mode(world)
    # "stop screwing around and finish" -> drop every flourish, keep only work
    strip_flourishes = mode in {"minimum_time", "freeze"}
    dropped, flourish_waves_added = _place_flourishes(
        waves, [] if strip_flourishes else flourishes,
        assignment, region_obj, overlap, arms, allow_flourish_wave, barriers)
    if strip_flourishes:
        dropped = [s.id for s in flourishes]

    # -- dance-while-working: fill idle-arm slack (never adds/reorders) ----
    idle_flourishes = _fill_idle_slack(waves, arms, mode) if _actions.wants_idle_flourish(mode) else 0

    arm_timeline: dict[str, list[str]] = {a: [] for a in arms}
    for w in waves:
        for ss in w.steps:
            if ss.arm in arm_timeline:
                arm_timeline[ss.arm].append(ss.step_id)

    mand_manip = {s.id for s in mandatory if s.contract == "manipulate"}
    metrics = {
        "waves": len(waves),
        "max_parallelism": max(
            (sum(1 for ss in w.steps if ss.step_id in mand_manip) for w in waves),
            default=0,
        ),
        "handoffs": handoffs,
        "serialized_conflicts": sum(1 for b in barriers if b.kind == "workspace"),
        "flourishes_scheduled": len(flourishes) - len(dropped),
        "flourishes_dropped": len(dropped),
        "flourish_waves_added": flourish_waves_added,
        "idle_flourishes": idle_flourishes,
    }
    return Schedule(waves, barriers, arm_timeline, metrics, assignment, region_id,
                    dropped, style_mode=mode)


def _style_mode(world: WorldState) -> str:
    """The last STYLE constraint wins ("make it fancy" ... "back to work").

    ``"unset"`` means no operator style was given — the planner still decides
    whether to emit `PRESENT`/flourish steps; the scheduler just schedules
    whatever is there.
    """
    styles = [c.value for c in world.constraints if c.kind == "style"]
    return _actions.style_mode(styles[-1]) if styles else "unset"


def _fill_idle_slack(waves: list[Wave], arms: tuple[str, str], mode: str) -> int:
    """Add non-blocking IDLE_FLOURISH steps to arms that are idle in a wave.

    Only touches already-idle arms in already-existing non-final waves, so the
    dependency graph and wave count are untouched — "dance occupies available
    slack but cannot violate the table-setting dependency graph".
    """
    ops = _actions.IDLE_FLOURISH_OPS
    if not ops or len(waves) < 2:
        return 0
    injected = 0
    for w in waves[:-1]:                       # never the trailing verify wave
        if any(ss.arm is None for ss in w.steps):   # barrier wave -> skip
            continue
        busy = {ss.arm for ss in w.steps if ss.arm}
        idle = [a for a in arms if a not in busy]
        if not idle:
            continue
        if mode == "synchronized":
            op = "SWAY"
            picks = {a: op for a in idle}
        elif mode == "mirrored":
            picks = {a: "MIRROR" for a in idle}
        else:                                  # show_off / dance / take_turns
            picks = {a: ops[(w.index + i) % len(ops)] for i, a in enumerate(idle)}
        for a, op in picks.items():
            w.steps.append(ScheduledStep(f"idle_{op.lower()}_w{w.index}_{a}", a, None,
                                         flourish=True))
            injected += 1
    return injected


def _place_flourishes(
    waves: list[Wave],
    flourishes: list[Step],
    assignment: dict[str, str | None],
    region_obj: dict[str, Region],
    overlap: float,
    arms: tuple[str, str],
    allow_flourish_wave: bool,
    barriers: list[Barrier],
) -> tuple[list[str], int]:
    if not flourishes:
        return [], 0

    wave_of: dict[str, int] = {ss.step_id: w.index for w in waves for ss in w.steps}
    barrier_idx = next((w.index for w in reversed(waves)
                        if any(ss.arm is None for ss in w.steps)), None)

    def region(fs: Step) -> Region:
        return region_obj.get(fs.id, Region("center", 0.0))

    def free(fs: Step, w: Wave) -> bool:
        if barrier_idx is not None and w.index >= barrier_idx:
            return False
        for d in fs.deps:
            dw = wave_of.get(d)
            if dw is None or dw >= w.index:
                return False
        arm = assignment[fs.id]
        if any(ss.arm == arm for ss in w.steps):
            return False
        reg = region(fs)
        return not any(
            region_obj[ss.step_id].conflicts_with(reg, overlap)
            for ss in w.steps if ss.step_id in region_obj
        )

    leftovers: list[Step] = []
    for fs in flourishes:
        slot = next((w for w in waves if free(fs, w)), None)
        if slot is None:
            leftovers.append(fs)
            continue
        slot.steps.append(ScheduledStep(fs.id, assignment[fs.id], region(fs).id, flourish=True))
        wave_of[fs.id] = slot.index

    added = 0
    if leftovers and allow_flourish_wave:
        insert_at = barrier_idx if barrier_idx is not None else len(waves)
        new_wave = Wave(insert_at, [])
        used_arms: set[str] = set()
        used_regions: list[Region] = []
        still: list[Step] = []
        for fs in leftovers:
            if any((wave_of.get(d) is None or wave_of[d] >= insert_at) for d in fs.deps):
                still.append(fs)
                continue
            arm = assignment[fs.id]
            reg = region(fs)
            if arm in used_arms or any(r.conflicts_with(reg, overlap) for r in used_regions):
                still.append(fs)
                continue
            new_wave.steps.append(ScheduledStep(fs.id, arm, reg.id, flourish=True))
            used_arms.add(arm)
            used_regions.append(reg)
            wave_of[fs.id] = insert_at
        if new_wave.steps:
            waves.insert(insert_at, new_wave)
            for i, w in enumerate(waves):
                w.index = i
            added = 1
        leftovers = still

    for fs in leftovers:
        barriers.append(Barrier((fs.id, fs.id), "flourish",
                                f"{fs.id} dropped: no slack wave, tempo preserved"))
    return [fs.id for fs in leftovers], added


# ---------------------------------------------------------------------------
# live integration — wrap any Plan provider
# ---------------------------------------------------------------------------


class ScheduledPlanner:
    """Decorator that makes any ``Plan`` provider emit *scheduled* graphs.

    ``ScheduledPlanner(RulePlanner())`` plugs straight into ``OmniQ`` — it runs
    the inner planner, then ``schedule(...).annotate(...)`` so the graph the
    engine executes already carries arm assignments and wave ordering. The last
    ``Schedule`` is kept on ``.last_schedule`` for the UI (OQ-020). If
    scheduling raises, the inner graph is used unchanged and the error is kept
    on ``.last_error`` — a scheduler bug can never break a run.
    """

    def __init__(self, inner: Any, **schedule_kwargs: Any) -> None:
        self.inner = inner
        self._kwargs = schedule_kwargs
        self.last_schedule: Schedule | None = None
        self.last_error: str | None = None

    @property
    def last_decision(self) -> Any:  # the engine reads this off the planner
        return getattr(self.inner, "last_decision", None)

    def _apply(self, graph: PlanGraph, world: WorldState) -> PlanGraph:
        try:
            self.last_schedule = schedule(graph, world, **self._kwargs)
            self.last_error = None
            return self.last_schedule.annotate(graph)
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the run
            self.last_schedule = None
            self.last_error = f"{type(exc).__name__}: {exc}"
            return graph

    def plan(self, goal: str, world: WorldState) -> PlanGraph:
        return self._apply(self.inner.plan(goal, world), world)

    def replan(self, current: PlanGraph, world: WorldState, reason: str) -> PlanGraph:
        return self._apply(self.inner.replan(current, world, reason), world)


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def _main() -> None:  # pragma: no cover - manual
    from .contracts import Constraint
    from .fakes import RulePlanner
    from .world import MockWorld

    for label, extra in (("plain", ()),
                         ("show off", (Constraint("style", "show_off", justification="demo"),))):
        mw = MockWorld.sample()
        for c in extra:
            mw.add_constraint(c)
        world = mw.state()
        graph = RulePlanner().plan("inspect and correct the workspace", world)
        sch = schedule(graph, world)
        print(f"\n=== {label} ===")
        print("metrics   :", sch.metrics)
        for w in sch.waves:
            print(f"wave {w.index}: " + ", ".join(
                f"{ss.step_id}[{ss.arm or '-'}@{ss.region_id or '-'}"
                f"{'/F' if ss.flourish else ''}]" for ss in w.steps))
        print("timelines :", sch.arm_timeline)
        for b in sch.barriers:
            print("barrier   :", b.kind, "-", b.reason)


if __name__ == "__main__":
    _main()
