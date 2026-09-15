"""The brief's "10 randomized seeds" demonstration, as one MP4 (2026-09-14).

For each seed: a title card over the randomized starting scene (command,
seed, what varies), the full engine-driven run at 4x speed from the director
view, then a held final frame with the per-object outcome and the running
tally. Every outcome shown is the engine's own receipt for that seed -- the
same numbers the harness reports -- so command, scene variation and robot
outcome are verifiable from the video alone.

    python integrations/intel/scripts/record_seed_montage.py --seed0 900 --n 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _recording import Recorder, _open_writer, draw_text_block  # noqa: E402

# VLA-first mode (2026-09-15): with OMNIQ_VLA_CHECKPOINT set, every engine in
# this process runs on integrations/intel/vla/vla_world.VLAWorld -- SmolVLA
# drives the single-arm PICK/MOVE steps; arms run one step at a time so the
# policy's cameras render on the main thread.
if os.environ.get("OMNIQ_VLA_CHECKPOINT"):
    os.environ.setdefault("OMNIQ_PARALLEL_ARMS", "0")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vla"))
    import vla_world  # noqa: E402
    vla_world.install()

DEFAULT_OUT = Path.home() / "OneDrive" / "Desktop" / "OMNI-Q_demo_videos"
DIRECTOR = "free:150,-30,0.95,0,-0.12,0.05"
from _recording import install_fixed_director_cameras  # noqa: E402
install_fixed_director_cameras([DIRECTOR])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", type=int, default=900)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="explicit seed list (overrides --seed0/--n)")
    ap.add_argument("--speed", type=int, default=4, help="sim-time speedup of the run segments")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=None,
                    help="per-seed segment cache dir (default: <out>/_montage_cache[_<mode>]); a seed whose cached run "
                         "passed is reused instead of re-recorded, so a failed seed costs one seed's re-run, not ten")
    ap.add_argument("--no-reuse", action="store_true", help="ignore cached seed segments")
    args = ap.parse_args()

    import cv2  # noqa: PLC0415
    from omni_q.intel_sim import (  # noqa: PLC0415
        IntelSceneConfig, TABLE_SETTING_PHRASINGS, _per_object_pick_place_outcomes, build_intel_sim_engine,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds or list(range(args.seed0, args.seed0 + args.n))
    args.n = len(seeds)
    path = args.out / (f"09_seeds_{seeds[0]}_{seeds[-1]}_montage.mp4" if seeds == list(range(seeds[0], seeds[-1] + 1))
                       else "09_seeds_" + "_".join(str(x) for x in seeds) + "_montage.mp4")
    mode = ("_omni" if os.environ.get("OMNIQ_OMNI_REASONER", "").lower() == "omni" else "") + \
           ("_vla" if os.environ.get("OMNIQ_VLA_CHECKPOINT") else "")
    cache = args.cache or (args.out / f"_montage_cache{mode}")
    cache.mkdir(parents=True, exist_ok=True)
    part = args.out / "_montage_part.mp4"
    writer = None
    tally = {"trials": 0, "resolved": 0, "placed": 0}
    import _reasoner_mode
    # in OMNI mode the live HUD (7 lines: arms, tally, reasoner decision, plan) fills the top
    # of the frame; the cards go under it instead of over it
    card_y = 235 if _reasoner_mode.enabled() else 40

    def card(frame, lines, hold_frames):
        f = frame.copy()
        draw_text_block(f, [t for t, _ in lines], 40, card_y, scales=[sc for _, sc in lines], alpha=0.6, pad=16)
        for _ in range(hold_frames):
            writer.write(f)

    import json
    import mujoco

    def variation_text(model, seed):
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, "omniq_variation")
        if tid < 0:
            return ""
        adr, size = model.text_adr[tid], model.text_size[tid]
        v = json.loads(bytes(model.text_data[adr:adr + size]).decode().rstrip("\x00"))
        return "  ".join(f"{k.split('_')[0]}: mass x{d['mass_factor']:.2f} friction x{d['friction_factor']:.2f}" for k, d in v.items())

    def append_part(src: Path) -> None:
        cap = cv2.VideoCapture(str(src))
        while True:
            ok, f = cap.read()
            if not ok:
                break
            writer.write(f)
        cap.release()

    def result_card(end, seed, final_n, placed, resolved, status, vla_line):
        card(end, [(f"Seed {seed}: final state {final_n}/5 in zone & upright   (engine: {placed}/5 placed, resolved={resolved})", 0.8),
                   (status, 0.5)] + [(t, 0.5) for t in vla_line] + [
                   (f"running: {tally['final_ok']}/{tally['trials']} trials fully set, {tally['resolved']}/{tally['trials']} resolved", 0.6)], 75)

    for i, seed in enumerate(seeds):
        goal = TABLE_SETTING_PHRASINGS[(seed - 900) % len(TABLE_SETTING_PHRASINGS)]   # the harness's phrasing for this seed
        meta_p, seg_p = cache / f"seed{seed}.json", cache / f"seed{seed}.mp4"
        if not args.no_reuse and meta_p.exists() and seg_p.exists():
            m = json.loads(meta_p.read_text())
            if m.get("passed") and m.get("speed") == args.speed:
                start = cv2.imread(str(cache / f"seed{seed}_start.png")); end = cv2.imread(str(cache / f"seed{seed}_end.png"))
                if writer is None:
                    writer = _open_writer(path, 25, (start.shape[1], start.shape[0]))
                card(start, [(t, sc) for t, sc in m["title"]], 75)
                append_part(seg_p)
                tally["trials"] += 1; tally["resolved"] += int(m["resolved"]); tally["placed"] += m["placed"]
                tally["final_ok"] = tally.get("final_ok", 0) + int(m["final_ok"])
                result_card(end, seed, m["final_n"], m["placed"], m["resolved"], m["status"], m["vla_line"])
                print(f"seed {seed}: reused cached segment (final {m['final_n']}/5, placed {m['placed']}/5)", flush=True)
                continue
        engine = build_intel_sim_engine(IntelSceneConfig(seed=seed, randomized=True))
        world = engine.world
        import _reasoner_mode
        reasoner_on = _reasoner_mode.enabled()
        # each seed's run is recorded to a part file, then appended to the montage
        rec = Recorder(world, DIRECTOR, part, every=20 * args.speed,
                       label=f"seed {seed}   \"{goal}\"   {args.speed}x")
        if writer is None:
            writer = _open_writer(path, 25, rec.size)
        if reasoner_on:
            _reasoner_mode.compose(engine, rec)
            rec.attach(world, goal)
        rec.frame()
        start = rec.last_bgr.copy()
        title = [(f"Seed {seed} ({i + 1} of {args.n})   command: \"{goal}\"", 0.9),
                 ("randomized: placement (+/-12 mm) & yaw (+/-11 deg), object mass & friction, colour, light angle & intensity, background, prompt", 0.55),
                 (variation_text(world.model, seed), 0.5),
                 (("two SO-101 arms, MuJoCo, real contact physics -- SmolVLA (fine-tuned VLA) drives each arm's motion; governed primitive as counted fallback"
                   if os.environ.get("OMNIQ_VLA_CHECKPOINT") else
                   "two SO-101 arms, MuJoCo, real contact physics -- both arms work at once"), 0.55)]
        card(start, title, 75)
        world._mujoco = rec.spy()
        t0 = time.time()
        receipt = engine.run(goal)
        rec.frame()
        end = rec.last_bgr.copy()
        rec.close()
        vr = getattr(world, "_vla_renderer", None)   # free the policy's renderer now, not at GC time
        if vr is not None:
            vr.close()
        append_part(part)
        po = _per_object_pick_place_outcomes(receipt)
        placed = sum(int(v["placed"]) for v in po.values())
        resolved = bool(receipt.metrics.get("resolved"))
        final = world.final_state_check()
        final_n = sum(int(v["in_zone_upright"]) for k, v in final.items() if k != "all_in_zone_upright")
        tally["trials"] += 1
        tally["resolved"] += int(resolved)
        tally["placed"] += placed
        tally["final_ok"] = tally.get("final_ok", 0) + int(final["all_in_zone_upright"])
        status = "  ".join(f"{k.split('_')[0]}: {'in zone, upright' if v['in_zone_upright'] else 'NOT in zone'} ({v['distance_m'] * 100:.0f} cm)"
                           for k, v in final.items() if k != "all_in_zone_upright")
        vla_line = []
        stats = getattr(world, "vla_stats", None)
        if stats:
            ok = sum(1 for a in stats if a.get("ok")); fb = sum(1 for a in stats if not a.get("ok"))
            vla_line = [f"VLA (SmolVLA) completed {ok} single-arm steps itself; {fb} handed to the governed primitive"]
        result_card(end, seed, final_n, placed, resolved, status, vla_line)
        print(f"seed {seed}: final {final_n}/5, placed {placed}/5 resolved={resolved} ({time.time() - t0:.0f}s)", flush=True)
        # cache this seed's segment so a later montage can reuse it if it passed
        passed = bool(final["all_in_zone_upright"]) and placed == 5 and resolved
        import shutil
        shutil.copyfile(part, seg_p)
        cv2.imwrite(str(cache / f"seed{seed}_start.png"), start); cv2.imwrite(str(cache / f"seed{seed}_end.png"), end)
        meta_p.write_text(json.dumps({"seed": seed, "goal": goal, "passed": passed, "speed": args.speed, "title": title,
                                      "final_n": final_n, "placed": placed, "resolved": resolved, "final_ok": bool(final["all_in_zone_upright"]),
                                      "status": status, "vla_line": vla_line}, indent=1))
    writer.release()
    part.unlink(missing_ok=True)
    print(f"wrote {path}   {tally}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
