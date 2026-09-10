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

    PYTHONPATH=src python -m omni_q.demo_intel_sim                        # headless text trace
    PYTHONPATH=src python -m omni_q.demo_intel_sim --viewer               # + live interactive MuJoCo window
    PYTHONPATH=src python -m omni_q.demo_intel_sim --viewer --step-delay 2  # slower, easier to follow
    PYTHONPATH=src python -m omni_q.demo_intel_sim --render-dir out/      # + PNG per step + assembled trace.gif

Object placement is still an explicit scripted WorldState transition (see
``intel_sim.py``): PICK/MOVE/PLACE kinematically teleport the object to
follow the plan (lift in place, then snap to the target zone), so a rendered
frame or the live viewer shows a real table setting forming -- but nothing
here grips or carries anything via IK or contact, so don't mistake the motion
for evidence of real grasping.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

from .events import Event
from .intel_sim import IntelSimulationUnavailable, build_intel_sim_engine
from .scheduler import ScheduledPlanner

GOAL = "set the table"
RENDER_SIZE = (640, 480)  # (width, height) -- MuJoCo's default offscreen framebuffer cap
VIEWER_STEP_DELAY = 0.6   # seconds paused after each step so a human can follow along


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


def _frame_recorder(engine, out_dir: Path) -> tuple[Callable[[Event], None], Callable[[], Path | None]]:
    """Subscribe the returned callback to the bus to save a numbered PNG after
    every finished step; call the returned ``finish()`` once, after the run,
    to assemble everything captured into ``out_dir/trace.gif``."""
    import mujoco
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(engine.world.model, height=RENDER_SIZE[1], width=RENDER_SIZE[0])
    frames: list = []

    def on_event(event: Event) -> None:
        if event.kind != "step.finished":
            return
        renderer.update_scene(engine.world.data, camera="third_person")
        frame = renderer.render().copy()
        frames.append(frame)
        Image.fromarray(frame).save(out_dir / f"frame_{len(frames):02d}_{event.data.get('op', '')}.png")

    def finish() -> Path | None:
        if not frames:
            return None
        gif_path = out_dir / "trace.gif"
        images = [Image.fromarray(f) for f in frames]
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)
        return gif_path

    return on_event, finish


def _run_with_viewer(engine, step_delay: float = VIEWER_STEP_DELAY):
    """Open a real, interactive MuJoCo window synced after every step (paced
    with a short sleep so the motion is actually watchable, not a ~10ms
    flash), then leave it open and responsive to mouse orbit/zoom until the
    user closes it by hand."""
    import mujoco.viewer

    with mujoco.viewer.launch_passive(engine.world.model, engine.world.data) as viewer:
        def sync(event: Event) -> None:
            if event.kind == "step.finished":
                viewer.sync()
                time.sleep(step_delay)

        unsubscribe = engine.bus.subscribe(sync)
        try:
            receipt = engine.run(GOAL)
        finally:
            unsubscribe()

        print("\nInteractive viewer open -- close the window when you're done looking.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Intel Online dual-SO-101 demo")
    parser.add_argument("--viewer", action="store_true",
                        help="open a live, interactive MuJoCo window synced to each step")
    parser.add_argument("--step-delay", type=float, default=VIEWER_STEP_DELAY,
                        help=f"seconds paused after each step in --viewer mode (default {VIEWER_STEP_DELAY})")
    parser.add_argument("--render-dir", type=Path, default=None,
                        help="save a PNG per step plus an assembled trace.gif into this directory")
    args = parser.parse_args()

    try:
        engine = build_intel_sim_engine()
    except IntelSimulationUnavailable as exc:
        raise SystemExit(
            f"{exc}\ninstall the Intel stack first: pip install -e \".[dev,intel]\""
        ) from exc

    scheduled = ScheduledPlanner(engine.planner)
    engine.planner = scheduled
    engine.bus.subscribe(_printer)

    finish_gif: Callable[[], Path | None] | None = None
    if args.render_dir is not None:
        on_frame, finish_gif = _frame_recorder(engine, args.render_dir)
        engine.bus.subscribe(on_frame)

    print("=" * 66)
    print(f"Intel Online - real MuJoCo dual-SO-101 + scheduler, goal: {GOAL!r}")
    print("=" * 66)

    receipt = _run_with_viewer(engine, args.step_delay) if args.viewer else engine.run(GOAL)

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

    if finish_gif is not None:
        gif_path = finish_gif()
        if gif_path:
            print(f"\nframe trace + gif written to {gif_path}")


if __name__ == "__main__":
    main()
