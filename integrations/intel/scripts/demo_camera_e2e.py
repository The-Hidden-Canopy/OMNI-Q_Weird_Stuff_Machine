"""End to end with real perception: camera -> detector -> zones -> OMNI plan ->
dual-arm controllers -> camera verification.

The engine's observer is replaced by `FrameObserver` fed by real renders of
all four scene cameras (overhead, third-person, left and right flank) through
the OpenVINO detector trained on this scene (`models/table_yolo_v*`), fused by
back-projecting each camera's detections through its own geometry
(`vision.MultiCameraFusion`), with zones assigned in the overhead frame. No ground-truth object state
reaches the planner: what OMNI plans from is what the camera saw. At the
end the camera looks again and reports, per object, whether it is in its
target zone -- that verification is independent of the controllers' own
receipts and is compared against them.

    python integrations/intel/scripts/demo_camera_e2e.py --seed 903
    python integrations/intel/scripts/demo_camera_e2e.py --seed 903 --live
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from omni_q.frame_observer import FrameObserver  # noqa: E402
from omni_q.intel_sim import (  # noqa: E402
    TABLE_SETTING_PHRASINGS, ZONE_POSITIONS, IntelSceneConfig, IntelTableWorld,
    _per_object_pick_place_outcomes, build_intel_sim_engine,
)
from omni_q.vision import MuJoCoCameraSource, MultiCameraFusion, OpenVINODetector, make_camera_zone_map  # noqa: E402

TARGET_ZONE = {"plate": "center", "cup": "upper_right", "fork": "left", "spoon": "right", "napkin": "lower_left"}


def latest_model() -> Path:
    cands = sorted(Path("models").glob("table_yolo_v*_openvino_model/*.xml"))
    if not cands:
        raise SystemExit("no models/table_yolo_v*_openvino_model/*.xml -- run train_table_yolo.py first")
    return cands[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=903)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("tmp") / "camera_e2e")
    args = ap.parse_args()

    engine = build_intel_sim_engine(IntelSceneConfig(seed=args.seed, randomized=True))
    world: IntelTableWorld = engine.world  # type: ignore[assignment]
    size = (640, 480)
    # All four scene cameras, one renderer each is fine here (they are only
    # captured through the fusion, never two at once elsewhere).
    cam_names = ("table_overhead", "third_person", "left_flank", "right_flank")
    cams = {n: MuJoCoCameraSource(world.model, world.data, n, *size) for n in cam_names}
    cam = cams["table_overhead"]
    detector = OpenVINODetector(str(args.model or latest_model()), device="CPU", conf_threshold=args.conf)
    # Zones the camera can name: the set-table zones plus where things start.
    zones = dict(ZONE_POSITIONS)
    for oid, det in world.state().objects.items():
        if det.pose is not None and det.zone not in zones:
            zones[det.zone] = (det.pose.x, det.pose.y, det.pose.z)
    zone_map = make_camera_zone_map(cam, zones, size)
    fusion = MultiCameraFusion(detector, cams, reference="table_overhead", frame_size=size)
    looks: list = []

    def look():
        """One fused multi-camera observation as (class, zone, conf, votes) rows --
        independent of the world state. Per-camera raw rows are kept for the receipt."""
        raw = fusion(None)
        rows = sorted((d.cls, zone_map(*d.center), round(float(d.conf), 2)) for d in raw)
        looks.append({"fused": fusion.last["fused"], "per_camera": fusion.last["per_camera"], "zoned": rows})
        return rows

    engine.observer = FrameObserver(
        fusion, zone_map=zone_map, frame_source=lambda w: None,
        target_zones=lambda object_id, cls: TARGET_ZONE.get(cls, "unknown"),
    )

    viewer = None
    if args.live:
        import mujoco  # noqa: PLC0415
        import mujoco.viewer  # noqa: PLC0415
        viewer = mujoco.viewer.launch_passive(world.model, world.data)
        real = world._mujoco
        wall0, count = time.time(), [0]
        dt = float(world.model.opt.timestep)

        class Spy:
            def __getattr__(self, n):
                return getattr(real, n)

            def mj_step(self, m, d, nstep=1):
                for _ in range(nstep):
                    real.mj_step(m, d)
                    count[0] += 1
                    if count[0] % 8 == 0:
                        viewer.sync()
                        ahead = count[0] * dt - (time.time() - wall0)
                        if ahead > 0:
                            time.sleep(min(ahead, 0.05))
        world._mujoco = Spy()

    before = look()
    print("camera before:", before, flush=True)
    orig_apply = world.apply_transition

    def apply_and_print(request):
        result = orig_apply(request)
        print(f"   {request.op:5s} {str(request.args.get('object') or ''):9s} {str(request.actor)[-9:]:9s} {'ok' if result.ok else 'FAIL'}", flush=True)
        return result
    world.apply_transition = apply_and_print

    goal = TABLE_SETTING_PHRASINGS[0]
    receipt = engine.run(goal)
    after = look()
    print("camera after: ", after, flush=True)

    # Independent verification: per class, does the camera see it in its target zone?
    seen_in_target = {cls: any(c == cls and z == TARGET_ZONE[cls] for c, z, _ in after) for cls in TARGET_ZONE}
    controller = {oid.split("_")[0]: v["placed"] for oid, v in _per_object_pick_place_outcomes(receipt).items()}
    agreement = {cls: (seen_in_target[cls], controller.get(cls)) for cls in TARGET_ZONE}
    summary = {
        "seed": args.seed, "goal": goal, "model": str(args.model or latest_model()),
        "cameras": list(cam_names),
        "camera_before": before, "camera_after": after,
        "camera_before_detail": looks[0], "camera_after_detail": looks[-1],
        "camera_says_in_target_zone": seen_in_target,
        "controller_says_placed": controller,
        "agree": all(a == b for a, b in agreement.values() if b is not None),
        "resolved": bool(receipt.metrics.get("resolved")),
        "run_id": receipt.run_id, "content_hash": receipt.content_hash,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"seed-{args.seed}.json").write_text(json.dumps({"summary": summary, "receipt": receipt.as_dict()}, indent=1, sort_keys=True))
    print("\n=== camera end-to-end ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "model"}, indent=1))
    if viewer is not None:
        import mujoco  # noqa: PLC0415
        print("physics keeps running; close the viewer to exit", flush=True)
        while viewer.is_running():
            mujoco.mj_step(world.model, world.data)
            viewer.sync()
            time.sleep(float(world.model.opt.timestep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
