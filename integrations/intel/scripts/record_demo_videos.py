"""Record the demo runs as MP4s for the submission video (2026-09-14).

Writes to a folder OUTSIDE the repo (default: Desktop/OMNI-Q_demo_videos).
Each run is recorded twice: a single camera and a multi-camera grid. Nothing
here is evidence; the receipts for these runs live under evidence/.

    python integrations/intel/scripts/record_demo_videos.py            # everything
    python integrations/intel/scripts/record_demo_videos.py --only table_903 plate_900
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _recording import Recorder  # noqa: E402

DEFAULT_OUT = Path.home() / "OneDrive" / "Desktop" / "OMNI-Q_demo_videos"
GRID = ["third_person", "table_overhead", "left_flank", "right_flank"]
WRIST_GRID = ["third_person", "table_overhead", "left_wrist", "right_wrist"]
# render-only free camera, front-left and closer than the fixed third_person view
DIRECTOR = "free:150,-30,0.95,0,-0.12,0.05"
DIRECTOR_GRID = [DIRECTOR, "table_overhead", "left_flank", "right_flank"]
SIX = ["third_person", "table_overhead", "left_flank", "right_flank", "left_wrist", "right_wrist"]


def _detector():
    from omni_q.vision import OpenVINODetector
    model = sorted(Path("models").glob("table_yolo_v*_openvino_model/*.xml"))[-1]
    return OpenVINODetector(str(model), device="CPU", conf_threshold=0.4)


def _with_recorders(world, out: Path, stem: str, run, views):
    """Run ``run(world)`` once per view set, recording each. A view tag
    starting with "vision" draws the detector's boxes on every scene camera."""
    results = []
    for tag, cams in views:
        vision = tag.startswith("vision")
        rec = Recorder(world(), cams, out / f"{stem}_{tag}.mp4",
                       label=f"OMNI-Q  {stem.replace('_', ' ')}  [{tag}]",
                       detector=_detector() if vision else None,
                       annotate=[c for c in ([cams] if isinstance(cams, str) else cams) if not c.startswith("free:")] if vision else ())
        w = rec.world
        w._mujoco = rec.spy()
        t0 = time.time()
        info = run(w)
        results.append(f"{rec.close()}  {info}  ({time.time() - t0:.0f}s wall)")
    return results


def table_trial(seed: int):
    from omni_q.intel_sim import (IntelSceneConfig, IntelTableWorld, TABLE_SETTING_PHRASINGS,
                                  build_intel_sim_engine, _per_object_pick_place_outcomes)

    def world():
        # the engine builds its own world; return a factory-made one and swap
        eng = build_intel_sim_engine(IntelSceneConfig(seed=seed, randomized=True))
        eng.world._engine = eng
        return eng.world

    def run(w):
        r = w._engine.run(TABLE_SETTING_PHRASINGS[0])
        po = _per_object_pick_place_outcomes(r)
        return "placed " + "".join("P" if v["placed"] else "-" for v in po.values())
    return world, run


def plate_only(seed: int):
    from omni_q.intel_sim import IntelSceneConfig, IntelTableWorld

    def world():
        return IntelTableWorld(IntelSceneConfig(randomized=True, seed=seed))

    def run(w):
        r = w._do_pick(0, "plate_1")
        r2 = w._do_place(0, "plate_1", "center") if r["held"] else {"placed": False}
        for _ in range(500):
            w._mujoco.mj_step(w.model, w.data)
        return f"held={r['held']} placed={r2['placed']}"
    return world, run


def authority(seed: int):
    from omni_q import nlu
    from omni_q.intel_sim import IntelSceneConfig, TABLE_SETTING_PHRASINGS, build_intel_sim_engine

    def world():
        eng = build_intel_sim_engine(IntelSceneConfig(seed=seed, randomized=True))
        eng.world._engine = eng
        return eng.world

    def run(w):
        eng = w._engine
        orig = w.apply_transition
        done = {"x": False}

        def apply(req):
            res = orig(req)
            if not done["x"] and req.op == "MOVE" and req.args.get("object") == "plate_1" and res.ok:
                done["x"] = True
                for k, v in nlu.parse("don't use the left arm anymore").constraints:
                    eng.add_constraint(k, v, source="operator", justification="voice: don't use the left arm anymore")
            return res
        w.apply_transition = apply
        r = eng.run(TABLE_SETTING_PHRASINGS[0])
        arms = [a.get("arm") for a in r.as_dict()["actions"] if a["op"] in ("PICK", "MOVE")]
        return f"arms in order: {arms}"
    return world, run


def arm_failure(seed: int):
    from omni_q.intel_sim import IntelSceneConfig, TABLE_SETTING_PHRASINGS, build_intel_sim_engine, _per_object_pick_place_outcomes

    def world():
        eng = build_intel_sim_engine(IntelSceneConfig(seed=seed, randomized=True))
        eng.world._engine = eng
        return eng.world

    def run(w):
        eng = w._engine
        orig = w.apply_transition
        done = {"x": False}

        def apply(req):
            res = orig(req)
            if not done["x"] and req.op == "MOVE" and req.args.get("object") == "fork_1" and res.ok:
                done["x"] = True
                w.fail_arm(0, reason="left arm servo bus: no response")
                eng.add_constraint("prefer_arm", "right", source="operator",
                                   justification="fault handler: left arm servo bus no response; withdrawn from authority")
            return res
        w.apply_transition = apply
        r = eng.run(TABLE_SETTING_PHRASINGS[0])
        po = _per_object_pick_place_outcomes(r)
        return "placed " + "".join("P" if v["placed"] else "-" for v in po.values()) + f" resolved={r.metrics.get('resolved')}"
    return world, run


def handoff(seed: int):
    from omni_q.intel_sim import ContactHandoffConfig, build_intel_contact_handoff_engine

    def world():
        eng = build_intel_contact_handoff_engine(ContactHandoffConfig(seed=seed))
        eng.world._engine = eng
        return eng.world

    def run(w):
        w._engine.run("transfer cup_1 from left arm to right arm")
        rec = w.last_admitted_receipt or w.pending_contact_receipt
        for _ in range(400):
            w._mujoco.mj_step(w.model, w.data)
        return f"handoff success={getattr(rec, 'success', None)}"
    return world, run


def camera_e2e(seed: int):
    from omni_q.frame_observer import FrameObserver
    from omni_q.intel_sim import (IntelSceneConfig, TABLE_SETTING_PHRASINGS, ZONE_POSITIONS,
                                  build_intel_sim_engine, _per_object_pick_place_outcomes)
    from omni_q.vision import MuJoCoCameraSource, MultiCameraFusion, OpenVINODetector, make_camera_zone_map
    model = sorted(Path("models").glob("table_yolo_v*_openvino_model/*.xml"))[-1]
    target = {"plate": "center", "cup": "upper_right", "fork": "left", "spoon": "right", "napkin": "lower_left"}

    def world():
        eng = build_intel_sim_engine(IntelSceneConfig(seed=seed, randomized=True))
        w = eng.world
        size = (640, 480)
        cams = {n: MuJoCoCameraSource(w.model, w.data, n, *size) for n in ("table_overhead", "third_person", "left_flank", "right_flank")}
        zones = dict(ZONE_POSITIONS)
        for oid, det in w.state().objects.items():
            if det.pose is not None and det.zone not in zones:
                zones[det.zone] = (det.pose.x, det.pose.y, det.pose.z)
        fusion = MultiCameraFusion(OpenVINODetector(str(model), device="CPU", conf_threshold=0.4), cams,
                                   reference="table_overhead", frame_size=size)
        eng.observer = FrameObserver(fusion, zone_map=make_camera_zone_map(cams["table_overhead"], zones, size),
                                     frame_source=lambda ws: None, target_zones=lambda oid, cls: target.get(cls, "unknown"))
        w._engine = eng
        return w

    def run(w):
        r = w._engine.run(TABLE_SETTING_PHRASINGS[0])
        po = _per_object_pick_place_outcomes(r)
        return "placed " + "".join("P" if v["placed"] else "-" for v in po.values()) + f" resolved={r.metrics.get('resolved')}"
    return world, run


def handshake(seed: int):
    """Both arms meet over the table's shared band, fingers horizontal and
    pointing at each other, close on each other's fingertips and pump
    together. Real contact, real servos -- just not a table-setting step."""
    import numpy as np
    from omni_q.intel_sim import GRIPPER_OPEN, IntelSceneConfig, IntelTableWorld

    def world():
        return IntelTableWorld(IntelSceneConfig(randomized=True, seed=seed))

    def run(w):
        meet = np.array([0.0, 0.02, 0.16])
        tips = {a: w.model.geom(f"{'left' if a == 0 else 'right'}_fixed_jaw_pad_1").id for a in (0, 6)}
        for a in (0, 6):
            w._go_home(a)
            w._set_gripper(a, 0.6)
        # approach: each tip stops 35 mm short of the meeting point on its own side
        targets = {0: meet + np.array([-0.035, 0.0, 0.0]), 6: meet + np.array([0.035, 0.0, 0.0])}
        plans = {}
        for a in (0, 6):
            base = w.data.xpos[w.model.body("left_Base" if a == 0 else "right_Base").id][:2]
            outward = targets[a][:2] - base
            q, _ = w._edge_seed_joints(a, targets[a] + np.array([0, 0, 0.05]), outward / np.linalg.norm(outward))
            plans[a] = q
        w._drive_joints_both(plans, steps=400)
        for a in (0, 6):
            w._ik_reach_pad(a, targets[a], iters=200, roll=0.0, track_tcp=False, geom_id=tips[a], tol=0.004, max_dq=0.02)
        # slide in so the fingertips overlap by ~25 mm, then close on each other
        for k in range(1, 7):
            for a, sign in ((0, 1.0), (6, -1.0)):
                goal = targets[a] + np.array([sign * 0.01 * k, 0.0, 0.0])
                w._ik_reach_pad(a, goal, iters=30, roll=0.0, track_tcp=False, geom_id=tips[a], tol=0.003, max_dq=0.015)
        for a in (0, 6):
            w.data.ctrl[a + 5] = 0.05
        for _ in range(150):
            w._mujoco.mj_step(w.model, w.data)
        # pump: three shakes, both arms together, 25 mm amplitude
        base_q = {a: w.data.ctrl[a:a + 5].copy() for a in (0, 6)}
        for cycle in range(3):
            for dz in (-0.025, 0.025):
                for _ in range(4):
                    for a in (0, 6):
                        cur = w.data.geom_xpos[tips[a]].copy()
                        w._ik_reach_pad(a, cur + np.array([0, 0, dz / 4]), iters=12, roll=0.0, track_tcp=False,
                                        geom_id=tips[a], tol=0.003, max_dq=0.015)
        for _ in range(100):
            w._mujoco.mj_step(w.model, w.data)
        for a in (0, 6):
            w._set_gripper(a, GRIPPER_OPEN, settle_steps=40)
        for a in (0, 6):
            w._go_home(a)
        return "handshake"
    return world, run


RUNS = {
    "table_903": (table_trial, 903, [("third_person", "third_person"), ("grid", GRID), ("director", DIRECTOR), ("six_cameras", SIX)]),
    "table_911": (table_trial, 911, [("overhead", "table_overhead"), ("wrists", WRIST_GRID), ("director", DIRECTOR)]),
    "plate_901": (plate_only, 901, [("third_person", "third_person"), ("grid", GRID), ("director", DIRECTOR)]),
    "authority_901": (authority, 901, [("third_person", "third_person"), ("grid", GRID), ("director", DIRECTOR)]),
    "handoff_19": (handoff, 19, [("third_person", "handoff_third_person"), ("director", "free:160,-25,0.8,0,-0.10,0.08")]),
    "handshake_903": (handshake, 903, [("director", "free:180,-15,0.7,0,0.02,0.12"), ("grid", GRID)]),
    "arm_failure_903": (arm_failure, 903, [("director", DIRECTOR), ("grid", GRID)]),
    "camera_e2e_903": (camera_e2e, 903, [("third_person", "third_person"), ("grid", GRID), ("director_grid", DIRECTOR_GRID), ("six_cameras", SIX),
                                         ("vision_grid", [DIRECTOR, "table_overhead", "left_flank", "right_flank"]),
                                         ("vision_overhead", "table_overhead")]),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--views", nargs="*", default=None, help="record only these view tags (e.g. director)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (factory, seed, views) in RUNS.items():
        if args.only and name not in args.only:
            continue
        world, run = factory(seed)
        if args.views:
            views = [v for v in views if v[0] in args.views]
        if not views:
            continue
        print(f"== {name}", flush=True)
        for line in _with_recorders(world, args.out, name, run, views):
            print("   ", line, flush=True)
    print(f"videos in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
