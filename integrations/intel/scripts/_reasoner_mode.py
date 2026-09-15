"""Reasoner mode for the recording scripts (2026-09-15).

With ``OMNIQ_OMNI_REASONER=omni`` (plus ``OMNIQ_OMNI_CHECKPOINT`` /
``OMNIQ_OMNI_RECEIPT``), every engine the recording scripts build gets its
planner wrapped: the IDA Omni reference body advises in the plan grammar
with its identifier slots fenced to the world's legal ids
(omni_reasoner.OmniReferenceReasoner, fenced decode), the governed core
validates each proposal (target zone, duplicates, PICK-before-MOVE, arm by
reach / operator authority) and completes the objects the model did not
address. The HUD shows every plan decision: what the model proposed, what
was accepted, what was rejected and why, what the core completed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

_REASONER = None


def enabled() -> bool:
    return os.environ.get("OMNIQ_OMNI_REASONER", "").strip().lower() == "omni"


def reasoner():
    global _REASONER
    if _REASONER is None:
        from omni_q.omni_reasoner import OmniReferenceReasoner
        _REASONER = OmniReferenceReasoner(
            os.environ["OMNIQ_OMNI_CHECKPOINT"], os.environ["OMNIQ_OMNI_RECEIPT"],
            device=os.environ.get("OMNIQ_OMNI_DEVICE", "cuda"))
    return _REASONER


def compose(engine, rec=None) -> None:
    """Wrap ``engine.planner`` with the reasoner-advised planner and, if a
    recorder is given, feed its HUD with each plan decision."""
    from omni_q.omni_planner import OmniPlanner

    engine.planner = OmniPlanner(reasoner(), fallback=engine.planner, complete_with_fallback=True, max_new_tokens=40)
    state = {"decisions": 0, "model_steps": 0, "rejected": 0, "fallback": 0}
    if rec is None:
        return

    def on(ev):
        if ev.kind != "plan.decision":
            return
        d = ev.data
        reason = str(d.get("reason") or "")
        ops = d.get("selected_ops") or ()
        rej = d.get("rejected") or {}
        state["decisions"] += 1
        state["rejected"] += len(rej)
        if "fallback" in reason:
            state["fallback"] += 1
            head = f"OMNI reasoner decision {state['decisions']}: no usable proposal -> governed planner"
        else:
            state["model_steps"] += int(d.get("candidates_feasible") or 0)
            head = (f"OMNI reasoner decision {state['decisions']}: proposed {d.get('candidates_considered', 0)} steps, "
                    f"{d.get('candidates_feasible', 0)} accepted, {len(rej)} rejected; core completed the rest")
        why = next(iter(rej.values()), "") if rej else ""
        rec.hud_extra = [head, f"plan: {' -> '.join(str(o) for o in ops)[:150]}"] + ([f"rejected e.g.: {why[:110]}"] if why else [])
        if hasattr(rec, "render_hud"):
            rec.render_hud()
        rec.notify(f"OMNI (IDA Omni body, fenced grammar) advised: {', '.join(str(o) for o in ops[:6])}", 5)
    engine.bus.subscribe(on)
    engine._reasoner_hud_state = state
