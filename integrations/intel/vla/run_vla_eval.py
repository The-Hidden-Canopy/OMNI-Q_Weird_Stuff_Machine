"""Evaluate (and optionally record) the VLA-first table setting (2026-09-15).

    OMNIQ_VLA_CHECKPOINT=<dir> python integrations/intel/vla/run_vla_eval.py --seeds 900 901 902 --record

Per seed: the submission engine runs on a VLAWorld (fine-tuned SmolVLA drives
every single-arm PICK / MOVE; the governed primitive is the fallback and is
counted). Writes evidence/benchmark_results/vla_smolvla_<date>/summary.json
with, per op, how many steps the VLA completed itself vs. handed to the
fallback, plus the usual task outcome. ``--record`` also writes a director
clip per seed with the executing policy on the HUD.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "integrations" / "intel" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("OMNIQ_PARALLEL_ARMS", "0")   # cameras render on the main thread

import vla_world  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[900, 901, 902, 903, 904])
    ap.add_argument("--checkpoint", default=os.environ.get("OMNIQ_VLA_CHECKPOINT", ""))
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "evidence" / "benchmark_results" / f"vla_smolvla_{datetime.date.today().isoformat()}")
    ap.add_argument("--video-out", type=Path, default=Path.home() / "OneDrive" / "Desktop" / "OMNI-Q_demo_videos" / "vla_experiments")
    ap.add_argument("--no-fallback", action="store_true")
    args = ap.parse_args()
    if not args.checkpoint:
        raise SystemExit("pass --checkpoint or set OMNIQ_VLA_CHECKPOINT")
    vla_world.VLAWorld.checkpoint = args.checkpoint
    if args.no_fallback:
        vla_world.VLAWorld.fallback = False
    vla_world.install()
    from omni_q.intel_sim import IntelSceneConfig, TABLE_SETTING_PHRASINGS, build_intel_sim_engine, _per_object_pick_place_outcomes

    args.out.mkdir(parents=True, exist_ok=True)
    trials = []
    for seed in args.seeds:
        goal = TABLE_SETTING_PHRASINGS[(seed - 900) % len(TABLE_SETTING_PHRASINGS)]
        eng = build_intel_sim_engine(IntelSceneConfig(seed=seed, randomized=True))
        w = eng.world
        rec = None
        if args.record:
            from _recording import Recorder
            args.video_out.mkdir(parents=True, exist_ok=True)
            rec = Recorder(w, "free:150,-30,0.95,0,-0.12,0.05", args.video_out / f"vla_smolvla_seed{seed}.mp4",
                           label=f"OMNI-Q  VLA-first: SmolVLA drives the arms  seed {seed}")
            rec.attach(w, goal)
            rec.hud_extra = ["policy: SmolVLA (fine-tuned on the governed expert) -> Cartesian deltas -> IK realisation; fallback: governed primitive"]
            w._mujoco = rec.spy()

            def hook(arm, tick, a, rec=rec):
                if tick % 5 == 0:
                    rec.hud_extra = [f"policy: SmolVLA  tick {tick}  d=({a['dx_mm']:+.0f},{a['dy_mm']:+.0f},{a['dz_mm']:+.0f}) mm  jaw {a['gripper_delta']:+.2f}"]
                    rec.render_hud()
            w.vla_tick_hook = hook
        t0 = time.time()
        r = eng.run(goal)
        po = _per_object_pick_place_outcomes(r)
        fs = w.final_state_check()
        by_op = collections.Counter((s["op"], "vla" if s["ok"] else "fallback") for s in w.vla_stats)
        trial = {"seed": seed, "goal": goal, "resolved": bool(r.metrics.get("resolved")),
                 "placed": sum(int(v["placed"]) for v in po.values()), "final_all_in_zone_upright": fs["all_in_zone_upright"],
                 "vla_steps": {f"{op}:{k}": n for (op, k), n in by_op.items()}, "vla_attempts": w.vla_stats,
                 "wall_s": round(time.time() - t0)}
        trials.append(trial)
        (args.out / f"trial-seed-{seed}.json").write_text(json.dumps({"trial": trial, "receipt": r.as_dict()}, indent=1))
        print({k: v for k, v in trial.items() if k != "vla_attempts"}, flush=True)
        if rec is not None:
            print(rec.close())
    total = collections.Counter()
    for t in trials:
        total.update(t["vla_steps"])
    summary = {"checkpoint": args.checkpoint, "trials": len(trials), "resolved": sum(t["resolved"] for t in trials),
               "placements": sum(t["placed"] for t in trials), "final_ok": sum(t["final_all_in_zone_upright"] for t in trials),
               "vla_steps": dict(total), "fallback_enabled": not args.no_fallback, "seeds": args.seeds}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
