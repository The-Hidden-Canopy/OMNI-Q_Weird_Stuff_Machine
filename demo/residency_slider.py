#!/usr/bin/env python
"""Residency slider demo - born-compressed OMNI.

Steps a fake memory envelope down the owner's path

    8 GB -> 6 GB -> 3 GB -> 1.8 GB -> 1.2 GB -> 1.2 GB + novel task -> 3 GB

printing the ASCII slider (state, resident bytes, capability/body/placement
retained, autonomy, graph revision) at each step, then writes the transition
log + final decision as a validated evidence bundle under
``evidence/benchmark_results/``.

OMNI is born compressed: the model never lives in FP32/BF16. The slider is
MXFP8 -> MXFP4 -> MXFP2 -> CORE_ONLY, selected by available memory. BF16 is
plumbing only. Below the MXFP2 envelope the neural reasoner is evicted:
OMNI-Q core remains, autonomy is DEGRADED, and novel tasks HOLD.

Run from the repo root:

    PYTHONPATH=src .venv/Scripts/python demo/residency_slider.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for candidate in (REPO_ROOT / "src", REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from omni_q.residency import GB, ResidencyManager, ResidencyState  # noqa: E402

STEPS_GB = (8.0, 6.0, 3.0, 1.8, 1.2, 1.2, 3.0)
BUNDLE_ROOT = REPO_ROOT / "evidence" / "benchmark_results"

SLIDER_LABELS = ("MXFP8", "MXFP4", "MXFP2", "CORE_ONLY")


def render_slider(state: ResidencyState) -> str:
    cells = [
        f"[{label}]" if label == state.name else f" {label} "
        for label in SLIDER_LABELS
    ]
    return "--".join(cells)


def show(manager: ResidencyManager, envelope_gib: float, note: str = "") -> None:
    d = manager.current
    resident_gib = d.resident_bytes / GB
    print(f"envelope {envelope_gib:5.2f} GiB {note}")
    print(f"  {render_slider(d.state)}")
    fmt = d.format if d.format is not None else "- (reasoner evicted)"
    print(
        f"  state={d.state.name:<10} format={fmt:<20} "
        f"resident={resident_gib:5.2f} GiB"
    )
    print(
        f"  capability={'RETAINED' if d.capability_retained else 'LOST':<9} "
        f"body={'RESIDENT' if d.state.is_mx else 'EVICTED':<8} "
        f"placement={d.placement:<6} autonomy={d.autonomy.name}"
    )
    print(f"  GRAPH REVISION {d.graph_revision}   {d.reason}")
    print()


def main() -> Path:
    manager = ResidencyManager(int(STEPS_GB[0] * GB))
    print("=== OMNI-Q RESIDENCY SLIDER - born-compressed OMNI ===")
    print("the model never lives in FP32/BF16; BF16 is plumbing only\n")

    for index, gib in enumerate(STEPS_GB):
        envelope = int(gib * GB)
        if index == 0:
            show(manager, gib, "(initial)")
            continue
        manager.update_envelope(envelope)
        note = ""
        if index == len(STEPS_GB) - 1:
            note = "(memory restored)"
        show(manager, gib, note)
        if index == 5:  # second 1.2 GB step: a novel task arrives at CORE_ONLY
            verdict = manager.admit_task(novel=True)
            print(f"  >>> NOVEL TASK at {gib} GiB -> {verdict} "
                  "(queued, not degraded-executed)")
            if manager.current.state is ResidencyState.CORE_ONLY:
                print("  >>> routine task would still EXECUTE on the core "
                      f"({manager.autonomy.name})")
            print()

    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = manager.write_receipt(BUNDLE_ROOT, run_id=f"residency_demo_{stamp}")
    print(f"receipt bundle: {run_dir.relative_to(REPO_ROOT)}")

    from omni_q.evidence_bundle import validate_evidence_bundle

    ok, errors = validate_evidence_bundle(run_dir)
    print(f"bundle validation: {'PASS' if ok else 'FAIL ' + '; '.join(errors)}")
    return run_dir


if __name__ == "__main__":
    main()
