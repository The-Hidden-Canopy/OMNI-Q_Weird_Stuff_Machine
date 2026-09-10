"""Runnable Intel Online demo: real MuJoCo dual-SO-101 physics + the bimanual
scheduler (OQ-012/013/044), composed with zero changes to either module.

``build_intel_sim_engine()`` (OQ-006) gives a real MuJoCo world and a planner
that assigns each object a fixed arm. ``ScheduledPlanner`` (OQ-012, see
``docs/scheduler.md``) is designed as a decorator specifically so it can wrap
that planner without editing ``intel_sim.py`` or ``scheduler.py`` while both
are still under active development elsewhere:

    engine.planner = ScheduledPlanner(engine.planner)

That's the whole integration. The scheduler now decides which steps run in the
same wave (both arms acting concurrently) and where a shared-workspace barrier
forces serialisation; the engine executes the annotated graph unchanged and
each manipulate step still steps real MuJoCo physics via
``IntelTableWorld.apply_transition``.

    PYTHONPATH=src python -m omni_q.demo_intel_sim

Known gap surfaced by actually running this composition: every tableware item
in ``IntelTableWorld`` starts with the same ``zone="staging"`` string (see
``intel_sim.py``), so the scheduler's workspace-conflict check (correctly)
treats every pickup as contending for one region and serialises all of them
(``max_parallelism`` stays 1) even though ``cup_1``/``spoon_1`` and the rest
are assigned to different arms. Real concurrent bimanual execution (OQ-017)
needs distinct per-object staging positions from the OQ-007 scene/object pack,
not a scheduler change.
"""

from __future__ import annotations

from .events import Event, EventBus
from .intel_sim import IntelSimulationUnavailable, build_intel_sim_engine
from .scheduler import ScheduledPlanner

GOAL = "set the table"


def _printer(event: Event) -> None:
    d = event.data
    if event.kind == "step.started":
        return
    if event.kind == "step.finished":
        print(f"  - {d['id']:<16} {d['op']:<10} arm={d.get('arm') or '-':<6} "
              f"@{d.get('device') or '-':<18} -> {d['state']}")
    elif event.kind == "graph.compiled":
        print(f"  graph rev{d['graph']['revision']}: "
              f"{[(s['op'], s['arm']) for s in d['graph']['steps']]}")
    elif event.kind == "verified":
        print(f"  verified: ok={d['ok']} mismatch={d['mismatch']}")
    elif event.kind == "run.finished":
        print(f"  == metrics: {d['metrics']}")


def main() -> None:
    try:
        engine = build_intel_sim_engine()
    except IntelSimulationUnavailable as exc:
        raise SystemExit(
            f"{exc}\ninstall the Intel stack first: pip install -e \".[dev,intel]\""
        ) from exc

    scheduled = ScheduledPlanner(engine.planner)
    engine.planner = scheduled
    engine.bus.subscribe(_printer)

    print("=" * 66)
    print(f"Intel Online - real MuJoCo dual-SO-101 + scheduler, goal: {GOAL!r}")
    print("=" * 66)
    receipt = engine.run(GOAL)

    if scheduled.last_error:
        print(f"\n! scheduler degraded to unscheduled graph: {scheduled.last_error}")
    elif scheduled.last_schedule is not None:
        sch = scheduled.last_schedule
        print(f"\nschedule: {sch.metrics}")
        for w in sch.waves:
            print(f"  wave {w.index}: " + ", ".join(
                f"{ss.step_id}[{ss.arm or '-'}@{ss.region_id or '-'}]" for ss in w.steps))
        for b in sch.barriers:
            print(f"  barrier: {b.kind} - {b.reason}")

    print(f"\nreal MuJoCo state: {engine.world.simulation_summary()}")
    print(f"resolved={receipt.metrics['resolved']} steps={receipt.metrics['steps_executed']} "
          f"revisions={receipt.metrics['revisions']}")


if __name__ == "__main__":
    main()
