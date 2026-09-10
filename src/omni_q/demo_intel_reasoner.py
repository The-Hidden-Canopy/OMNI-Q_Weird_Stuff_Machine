"""Intel Online demo with the IDA Omni reasoner advising the planner.

Same composition rule as ``demo_intel_sim`` — no edits to ``intel_sim.py``
(it is under active development elsewhere)::

    engine = build_intel_sim_engine(...)
    engine.planner = OmniPlanner(reasoner, fallback=ScheduledPlanner(engine.planner))

The Omni model receives the structured scene state (IntelTableWorld's
detections, rendered by OmniPlanner — never raw frames) and advises plan
steps in the constrained PLAN…END grammar; the governed core validates every
step before it reaches the MuJoCo arms, and the deterministic scheduled
planner governs whenever the reasoner is unavailable or proposes nothing
valid. Every decision's reason is labeled with the backend that produced it,
so receipts and the UI's "why" display (OQ-047) can tell model advice from
deterministic fallback at a glance.

Backend selection (env):
    OMNIQ_OMNI_REASONER=mock   OmniPlanner over MockReasoner (default here)
    OMNIQ_OMNI_REASONER=omni   identity-gated IDA Omni reference body
                               (+ OMNIQ_OMNI_CHECKPOINT / OMNIQ_OMNI_RECEIPT)
    unset/off                  no OmniPlanner; plain scheduled demo
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .demo_intel_sim import GOAL, _printer, _run_with_viewer
from .intel_sim import IntelSimulationUnavailable, build_intel_sim_engine
from .scheduler import ScheduledPlanner

VIEWER_STEP_DELAY = 0.6


def _reasoner_from_env():
    mode = os.environ.get("OMNIQ_OMNI_REASONER", "mock").strip().lower()
    if mode in {"", "0", "off", "rule"}:
        return None
    if mode == "mock":
        from .omni_reasoner import MockReasoner
        return MockReasoner()
    if mode == "omni":
        from .omni_reasoner import OmniReferenceReasoner
        checkpoint = os.environ.get("OMNIQ_OMNI_CHECKPOINT")
        receipt = os.environ.get("OMNIQ_OMNI_RECEIPT")
        if not checkpoint or not receipt:
            raise SystemExit(
                "OMNIQ_OMNI_REASONER=omni requires OMNIQ_OMNI_CHECKPOINT "
                "and OMNIQ_OMNI_RECEIPT")
        return OmniReferenceReasoner(
            checkpoint, receipt,
            device=os.environ.get("OMNIQ_OMNI_DEVICE", "cpu"))
    raise SystemExit(f"unknown OMNIQ_OMNI_REASONER mode: {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intel Online dual-SO-101 demo + IDA Omni reasoner")
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

    reasoner = _reasoner_from_env()
    scheduled = ScheduledPlanner(engine.planner)
    if reasoner is not None:
        from .omni_planner import OmniPlanner
        engine.planner = OmniPlanner(reasoner, fallback=scheduled)
        planner, label = engine.planner, f"omni[{reasoner.backend}]"
    else:
        engine.planner = scheduled
        planner, label = scheduled, "deterministic"

    def _print_reason(event):
        if event.kind == "plan.decision" and event.data.get("reason"):
            print(f"  why: {event.data['reason']}")

    engine.bus.subscribe(_printer)
    engine.bus.subscribe(_print_reason)

    finish_gif = None
    if args.render_dir is not None:
        from .demo_intel_sim import _frame_recorder
        on_frame, finish_gif = _frame_recorder(engine, args.render_dir)
        engine.bus.subscribe(on_frame)

    print("=" * 66)
    print(f"Intel Online - MuJoCo dual-SO-101 + IDA Omni reasoner, goal: {GOAL!r}")
    print(f"planner: {label} (validated fallback: scheduled IntelTablePlanner)")
    print("=" * 66)

    receipt = (_run_with_viewer(engine, args.step_delay)
               if args.viewer else engine.run(GOAL))

    if scheduled.last_error:
        print(f"\n! scheduler degraded to unscheduled graph: {scheduled.last_error}")
    elif scheduled.last_schedule is not None and reasoner is None:
        sch = scheduled.last_schedule
        print(f"\nschedule: {sch.metrics}")

    if reasoner is not None and getattr(planner, "last_decision", None) is not None:
        dec = planner.last_decision
        print(f"\nomni decision: {dec.reason}")
        if dec.rejected:
            for target, why in dec.rejected.items():
                print(f"  rejected {target}: {why}")

    print(f"\nreal MuJoCo state: {engine.world.simulation_summary()}")
    print(f"resolved={receipt.metrics['resolved']} steps={receipt.metrics['steps_executed']} "
          f"revisions={receipt.metrics['revisions']}")

    if finish_gif is not None:
        gif_path = finish_gif()
        if gif_path:
            print(f"\nframe trace + gif written to {gif_path}")


if __name__ == "__main__":
    main()
