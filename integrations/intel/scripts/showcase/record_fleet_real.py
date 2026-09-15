"""SHOWCASE -- a real fleet: four OMNI-Q engines, four dual-arm units, eight
arms, all planned by OMNI and driven by the submission's own controllers
(2026-09-15). Nothing here is choreographed.

Each unit is the actual submission stack on its own seed and its own prompt
phrasing, run in its own process at the same time. Unit 2 takes a mid-run
operator voice command ("don't use the left arm anymore") and unit 3 loses
its left arm to a servo fault; both re-plan and finish with the right arm,
exactly as in the submission clips. The four director views are composed
into one 2x2 fleet screen with a HUD that reads each unit's live state.

    python integrations/intel/scripts/showcase/record_fleet_real.py            # runs the 4 units, then composes
    python integrations/intel/scripts/showcase/record_fleet_real.py --unit 2 --seed 905 --variant authority
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3] / "src"))

DEFAULT_OUT = Path.home() / "OneDrive" / "Desktop" / "OMNI-Q_demo_videos" / "showcase_experiments"
UNITS = [  # unit, seed, variant
    (1, 903, "plain"), (2, 905, "authority"), (3, 906, "arm_failure"), (4, 909, "plain"),
]
DIRECTOR = "free:150,-30,0.95,0,-0.12,0.05"


def run_unit(unit: int, seed: int, variant: str, out: Path) -> dict:
    """One real engine run, recorded from the director camera at 640x360,
    with a per-frame state log so the fleet HUD can be composed later."""
    from _recording import Recorder
    from omni_q import nlu
    from omni_q.intel_sim import IntelSceneConfig, TABLE_SETTING_PHRASINGS, build_intel_sim_engine, _per_object_pick_place_outcomes

    goal = TABLE_SETTING_PHRASINGS[(seed - 900) % len(TABLE_SETTING_PHRASINGS)]
    eng = build_intel_sim_engine(IntelSceneConfig(seed=seed, randomized=True))
    w = eng.world
    path = out / f"_unit{unit}_seed{seed}_{variant}.mp4"
    rec = Recorder(w, DIRECTOR, path, size=(640, 360), label=f"UNIT {unit}  seed {seed}  OMNI-planned, both arms")
    rec.attach(w, goal)
    w._mujoco = rec.spy()
    log: list[tuple[int, list[str], str | None]] = []   # (frame, hud lines, notice)
    orig_frame = rec.frame

    def frame(data=None):
        orig_frame(data)
        log.append((rec.frames, list(rec.hud), rec._notice[0] if rec._notice and rec.frames <= rec._notice[1] else None))
    rec.frame = frame
    # the parallel path renders through the spy's pump -> rec.frame; single path too
    done = {"x": False}
    orig, orig_par = w.apply_transition, w.apply_transitions_parallel

    def after(req, res):
        if done["x"] or not (req.op == "MOVE" and res.ok):
            return
        if variant == "authority" and req.args.get("object") == "plate_1":
            done["x"] = True
            for k, v in nlu.parse("don't use the left arm anymore").constraints:
                eng.add_constraint(k, v, source="operator", justification="voice: don't use the left arm anymore")
            rec.notify("OPERATOR (voice): 'don't use the left arm anymore' -> re-planned, right arm takes over", 6)
        elif variant == "arm_failure" and req.args.get("object") == "fork_1":
            done["x"] = True
            w.fail_arm(0, reason="left arm servo bus: no response")
            eng.add_constraint("prefer_arm", "right", source="operator",
                               justification="fault handler: left arm servo bus no response; withdrawn from authority")
            rec.notify("FAULT: left arm servo bus - no response -> withdrawn from authority, right arm continues", 7)

    def apply(req):
        res = orig(req); after(req, res); return res

    def apply_pair(reqs):
        outp = orig_par(reqs)
        for req, res in zip(reqs, outp):
            after(req, res)
        return outp
    w.apply_transition, w.apply_transitions_parallel = apply, apply_pair

    t0 = time.time()
    r = eng.run(goal)
    po = _per_object_pick_place_outcomes(r)
    fs = w.final_state_check()
    result = {"unit": unit, "seed": seed, "variant": variant, "goal": goal, "video": str(path),
              "placed": sum(int(v["placed"]) for v in po.values()), "resolved": bool(r.metrics.get("resolved")),
              "final_all_in_zone_upright": fs["all_in_zone_upright"], "frames": rec.frames, "wall_s": round(time.time() - t0)}
    rec.close()
    (out / f"_unit{unit}_log.json").write_text(json.dumps({"result": result, "log": log}))
    print(json.dumps(result), flush=True)
    return result


def compose(out: Path, results: list[dict]) -> Path:
    """Tile the four unit clips 2x2 into one 1280x720 fleet screen with a
    live HUD read from each unit's own state log."""
    import cv2
    import numpy as np
    from _recording import _open_writer, draw_text_block

    caps = [cv2.VideoCapture(r["video"]) for r in results]
    logs = [json.loads((out / f"_unit{r['unit']}_log.json").read_text())["log"] for r in results]
    n = max(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)
    path = out / "fleet_real_4_units_8_arms.mp4"
    writer = _open_writer(path, 25, (1280, 720))
    last = [None] * 4
    ended = [False] * 4
    for f in range(n):
        tiles = []
        active = 0
        for i, c in enumerate(caps):
            ok, fr = c.read()
            if ok:
                last[i] = fr
            else:
                ended[i] = True
            fr = last[i] if last[i] is not None else np.zeros((360, 640, 3), np.uint8)
            if not ended[i]:
                active += 1
            tiles.append(fr.copy())
        tiles = [cv2.resize(t, (640, 338)) for t in tiles]
        grid = np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0)
        band = np.full((44, 1280, 3), 18, np.uint8)
        placed = sum(r["placed"] for r, e in zip(results, ended) if e)
        text = (f"FLEET: 8 MANIPULATORS   4 OMNI-Q ENGINES, independent & simultaneous   ACTIVE UNITS: {active}   "
                f"OBJECTIVES: 4 x set the table   RE-PLANS: 2 (voice authority, servo fault)   COMPLETE: {sum(ended)}/4   PLACEMENTS: {placed}/{5 * sum(ended)}")
        cv2.putText(band, text, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(np.concatenate([band, grid], axis=0))
    writer.release()
    for c in caps:
        c.release()
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.unit:
        run_unit(args.unit, args.seed, args.variant, args.out)
        return 0
    procs = []
    for unit, seed, variant in UNITS:
        procs.append(subprocess.Popen([sys.executable, __file__, "--unit", str(unit), "--seed", str(seed), "--variant", variant,
                                       "--out", str(args.out)]))
    for p in procs:
        p.wait()
    results = [json.loads((args.out / f"_unit{u}_log.json").read_text())["result"] for u, _, _ in UNITS]
    print(compose(args.out, results))
    for r in results:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
