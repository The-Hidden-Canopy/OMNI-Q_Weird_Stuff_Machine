"""End-to-end mock demo (OQ-001 "Done when").

Runs the Observe -> Plan -> Manipulate -> Verify -> Receipt loop three times,
showing the three observable states from DEMO.md:

    1. NORMAL            goal -> graph -> execute -> verify -> success
    2. CONSTRAINT CHANGE "keep everything local" + "don't use the left arm"
    3. FAILURE / WORLD CHANGE  object nudged mid-run -> verify fails -> replan

No hardware, no third-party deps:  python -m omni_q.demo

Set OMNIQ_RECEIPTS_DIR to persist every run's receipt to disk (OQ-037):
each invocation writes a parent-chained evidence bundle under
``$OMNIQ_RECEIPTS_DIR/demo-<timestamp>/`` via omni_q.evidence.EvidenceLedger.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from .events import Event, EventBus
from . import build_mock_engine
from .world import MockWorld
from .engine import OmniQ
from .evidence import EvidenceLedger
from .fakes import FakeManipulator, FakeObserver, FakeRecorder, FakeVerifier, RulePlanner
from .devices import default_devices

GOAL = "inspect and correct the workspace"


def _durable_recorder_factory() -> "object | None":
    """Opt-in durable receipts (OQ-037). Returns a factory producing one fresh
    ledger per scenario -- deterministic run ids collide when two scenarios
    share a world+goal config (scenario 3 repeats scenario 1's), and each
    scenario is its own evidence bundle anyway."""
    root = os.environ.get("OMNIQ_RECEIPTS_DIR")
    if not root:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def make(tag: str) -> EvidenceLedger:
        ledger = EvidenceLedger(Path(root) / f"demo-{stamp}-{tag}")
        print(f"  receipts -> {ledger.root}")
        return ledger

    return make


def _printer(event: Event) -> None:
    d = event.data
    if event.kind in {"step.started"}:
        return
    if event.kind == "step.finished":
        print(f"  - {d['id']:<16} {d['op']:<10} @{d.get('device') or '-':<18} -> {d['state']}")
    elif event.kind == "graph.compiled":
        print(f"  graph rev{d['graph']['revision']}: "
              f"{[s['op'] for s in d['graph']['steps']]}")
    elif event.kind == "graph.recompiled":
        print(f"  ~ recompiled ({d.get('reason')}) -> rev{d['graph']['revision']}: "
              f"{[s['op'] for s in d['graph']['steps']]}")
    elif event.kind == "observed":
        print(f"  observed frame, misplaced={d['misplaced']}")
    elif event.kind == "verified":
        print(f"  verified: ok={d['ok']} mismatch={d['mismatch']}")
    elif event.kind in {"capability.lost", "placement.failed", "constraint.added"}:
        print(f"  ! {event.kind}: {d}")
    elif event.kind == "run.finished":
        print(f"  == metrics: {d['metrics']}")


def _banner(title: str) -> None:
    print("\n" + "=" * 66 + f"\n{title}\n" + "=" * 66)


def scenario_normal(recorder: EvidenceLedger | None = None) -> None:
    _banner("1. NORMAL")
    bus = EventBus()
    bus.subscribe(_printer)
    engine = build_mock_engine(bus, recorder=recorder)
    engine.run(GOAL)


def scenario_constraint_change(recorder: EvidenceLedger | None = None) -> None:
    _banner("2. CONSTRAINT CHANGE  (keep-local, then lose the left arm)")
    bus = EventBus()
    bus.subscribe(_printer)
    engine = build_mock_engine(bus, recorder=recorder)
    engine.add_constraint("keep_local", justification="spoken operator command: keep inference local")
    engine.add_constraint("prefer_arm", "left", justification="spoken operator preference")
    # the left arm disappears before the run
    engine.device.set_online("arduino.left_arm", False)
    engine.run(GOAL)


def scenario_world_change(recorder: EvidenceLedger | None = None) -> None:
    _banner("3. FAILURE / WORLD CHANGE  (object nudged mid-run)")
    bus = EventBus()
    bus.subscribe(_printer)
    world = MockWorld.sample()
    engine = OmniQ(
        world=world,
        observer=FakeObserver(),
        planner=RulePlanner(),
        manipulator=FakeManipulator(world),
        verifier=FakeVerifier(),
        device=default_devices(),
        recorder=recorder if recorder is not None else FakeRecorder(),
        bus=bus,
    )

    original_execute = engine.manipulator.execute
    state = {"perturbed": False}

    def perturbing_execute(step, wstate):
        res = original_execute(step, wstate)
        if step.op == "MOVE" and not state["perturbed"]:
            state["perturbed"] = True
            world.perturb("connector_2", "floor")   # someone knocked it off
            bus.publish("world.perturbed", object="connector_2", to="floor")
        return res

    engine.manipulator.execute = perturbing_execute  # type: ignore[method-assign]
    engine.run(GOAL)


def main() -> None:
    factory = _durable_recorder_factory()
    scenario_normal(factory("normal") if factory else None)
    scenario_constraint_change(factory("constraint") if factory else None)
    scenario_world_change(factory("world") if factory else None)
    print("\nAll scenarios completed.\n")


if __name__ == "__main__":
    main()
