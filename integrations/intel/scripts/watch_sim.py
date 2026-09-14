"""Watch the dual-SO-101 manipulation live, or record it from any scene camera.

The SO-101 RL write-up the hosts circulated found its grasp-frame bug only by
*looking*: "it only sees coordinates and grasp state in the console log ... I
highly recommend visualizing the grasp point and seeing it for yourself." This
is that tool for our scene.

Live (opens MuJoCo's interactive viewer; drag to orbit, scroll to zoom, ctrl+
right-drag to pan; the sim runs the requested picks while you watch):

    .venv/Scripts/python integrations/intel/scripts/watch_sim.py --live \
        --pick cup_1 --pick fork_1 --seed 701

Record a clip from a scene camera (third_person, table_overhead, left_flank, right_flank,
left_wrist, right_wrist) to an animated GIF you can share:

    .venv/Scripts/python integrations/intel/scripts/watch_sim.py \
        --record tmp/cup_pick.gif --camera table_grazing --pick cup_1 --seed 701

Record the full randomized harness trial (every object, engine-driven order)
instead of hand-picked primitives:

    .venv/Scripts/python integrations/intel/scripts/watch_sim.py \
        --record tmp/trial.gif --camera third_person --trial --seed 701

Nothing here changes the physics: the recorder wraps mj_step only to grab a
frame every N steps, and the live viewer is MuJoCo's passive mode.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402,F401  (registers the viewer submodule)

from omni_q.intel_sim import (  # noqa: E402
    ContactHandoffConfig,
    IntelContactHandoffWorld,
    IntelSceneConfig,
    IntelTableWorld,
)

ARM_FOR = {"cup_1": 6, "spoon_1": 6, "plate_1": 0, "fork_1": 0, "napkin_1": 0}


class _StepSpy:
    """Forward everything to the mujoco module; observe mj_step."""

    def __init__(self, module, on_step) -> None:
        self._m = module
        self._on_step = on_step

    def __getattr__(self, name):
        return getattr(self._m, name)

    def mj_step(self, model, data, nstep: int = 1):
        if nstep == 1:
            self._m.mj_step(model, data)
            self._on_step(1)
            return
        # keep frame cadence honest for batched steps
        for _ in range(nstep):
            self._m.mj_step(model, data)
            self._on_step(1)


def _run_actions(world: IntelTableWorld, picks: list[str], do_move: bool) -> None:
    for obj in picks:
        arm = ARM_FOR.get(obj, 0)
        res = world._do_pick(arm, obj)
        g = res.get("grasp_sensor", {})
        print(f"PICK {obj:8} arm={'right' if arm else 'left':5} held={res['held']!s:5} "
              f"lift={res['lift_height_m'] * 1000:5.1f}mm pads={g.get('contact_pad_count')} "
              f"F={g.get('max_contact_force_n')}N retries={len(res.get('retries', []))}", flush=True)
        if do_move and res["held"]:
            target = world._objects[obj].target_zone
            res2 = world._do_place(arm, obj, target)
            print(f"MOVE {obj:8} -> {target:10} placed={res2.get('placed')!s:5} "
                  f"reason={res2.get('reason')}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="open the interactive viewer")
    ap.add_argument("--record", type=Path, help="write an animated GIF here")
    ap.add_argument("--camera", default="third_person")
    ap.add_argument("--pick", action="append", default=[], metavar="OBJECT")
    ap.add_argument("--move", action="store_true", help="also carry each held object to its target")
    ap.add_argument("--trial", action="store_true", help="run one full randomized harness trial")
    ap.add_argument("--handoff", action="store_true",
                    help="run the two-arm cup handoff (left grasps, lifts to shared point, right takes it, left releases)")
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--every", type=int, default=8, help="record one frame per N sim steps")
    ap.add_argument("--size", default="640x480")
    args = ap.parse_args()
    if not args.live and not args.record:
        ap.error("choose --live and/or --record")
    picks = args.pick or ["cup_1", "fork_1", "spoon_1", "napkin_1", "plate_1"]

    frames = []
    counter = {"n": 0}
    wall0 = {"t": time.time()}
    # Everything that touches model/data lives in this box so the trial path
    # can swap in the harness's own world. The first --trial version launched
    # the viewer on a world built here, then run_intel_table_evaluation_report
    # built and stepped a *different* world -- the window showed a scene
    # nobody was moving ("it opened but nothing moved at all").
    view: dict[str, Any] = {"world": None, "model": None, "data": None,
                            "viewer": None, "renderer": None}

    def attach(world: Any) -> None:
        if view["viewer"] is not None:
            view["viewer"].close()
        view["world"], view["model"], view["data"] = world, world.model, world.data
        if args.record:
            w, h = (int(v) for v in args.size.lower().split("x"))
            view["renderer"] = mujoco.Renderer(world.model, height=h, width=w)
        if args.live:
            view["viewer"] = mujoco.viewer.launch_passive(world.model, world.data)
        counter["n"] = 0
        wall0["t"] = time.time()

    def on_step(n: int) -> None:
        counter["n"] += n
        if counter["n"] % args.every:
            return
        renderer, viewer, data = view["renderer"], view["viewer"], view["data"]
        dt = float(view["model"].opt.timestep)
        if renderer is not None:
            renderer.update_scene(data, camera=args.camera)
            frames.append(renderer.render().copy())
        if viewer is not None:
            viewer.sync()
            # Pace to REAL time so motion plays at true speed. The first
            # version slept a fixed 2 ms per synced frame, so the sim ran far
            # faster than wall-clock and a sub-second place -- descend,
            # release, retract -- flashed past as an apparent teleport. It was
            # measured: zero single-step object jumps over 10 mm in 6,253
            # steps. The physics is continuous; the playback was not.
            due = counter["n"] * dt
            ahead = due - (time.time() - wall0["t"])
            if ahead > 0:
                time.sleep(min(ahead, 0.05))

    started = time.time()
    if args.trial:
        from omni_q.intel_sim import run_intel_table_evaluation_report  # noqa: PLC0415
        # The harness builds its own world (one per trial); attach the viewer
        # and the step spy to *that* instance the moment it exists.
        original = IntelTableWorld.__init__

        def spied_init(self, *a, **k):
            original(self, *a, **k)
            attach(self)
            self._mujoco = _StepSpy(mujoco, on_step)
        IntelTableWorld.__init__ = spied_init
        try:
            # Unique per run: the harness refuses to overwrite existing evidence.
            out = Path("tmp") / f"watch_trial_{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
            report = run_intel_table_evaluation_report(out, trials=1, seed=args.seed)
            print("per_object:", report.get("per_object_summary"))
        finally:
            IntelTableWorld.__init__ = original
    elif args.handoff:
        from omni_q.intel_sim import run_contact_handoff  # noqa: PLC0415
        # Same trick for the contact-handoff world: it steps through its own
        # ``_mujoco`` handle, so a spied instance is enough.
        original_h = IntelContactHandoffWorld.__init__

        def spied_handoff_init(self, *a, **k):
            original_h(self, *a, **k)
            attach(self)
            self._mujoco = _StepSpy(mujoco, on_step)
        IntelContactHandoffWorld.__init__ = spied_handoff_init
        try:
            receipt = run_contact_handoff(ContactHandoffConfig(seed=args.seed, randomized=True))
            print("handoff:", "success" if receipt.success else f"failed ({receipt.failure_reason})",
                  "| phases:", [t["phase"] for t in receipt.phase_timings])
        finally:
            IntelContactHandoffWorld.__init__ = original_h
    else:
        world = IntelTableWorld(scene_config=IntelSceneConfig(randomized=True, seed=args.seed))
        attach(world)
        world._mujoco = _StepSpy(mujoco, on_step)
        _run_actions(world, picks, args.move)
    print(f"done in {time.time() - started:.1f}s, {counter['n']} sim steps", flush=True)

    viewer, model, data = view["viewer"], view["model"], view["data"]
    dt = float(model.opt.timestep)
    if viewer is not None:
        # Keep PHYSICS running while the window is open, at roughly real time.
        # The first version only redrew, so the world froze the instant the
        # scripted actions ended -- a cup released by a failed placement hung
        # in mid-air, which read as "it let go and it floated". It had let go;
        # the sim just was not being stepped any more.
        print("actions done -- physics keeps running; close the window to exit", flush=True)
        while viewer.is_running():
            t0 = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            lag = dt - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)

    if args.record and frames:
        from PIL import Image  # noqa: PLC0415
        args.record.parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=40, loop=0)
        print(f"wrote {args.record} ({len(imgs)} frames)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
