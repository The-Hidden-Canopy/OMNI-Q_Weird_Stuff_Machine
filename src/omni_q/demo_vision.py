"""Demo: camera + detector + table-plane projection, all real, end to end.

Renders the Intel table scene's ``table_overhead`` camera through a real
OpenVINO detector and back-projects each detection's pixel center onto the
table plane using real camera geometry (see ``src/omni_q/vision.py``).

    PYTHONPATH=src python -m omni_q.demo_vision --model <exported ov ir .xml>

Not wired into the main Observe contract yet: a raw detection has a class
label and a pixel position, but the WorldState's ``Detection`` also carries
a ``target_zone`` -- where the object *should* end up -- which is a goal/
task fact, not something a camera can see. Fusing this into the engine's
observe loop needs that design question settled first (see
integrations/intel/README.md); this script demonstrates the perception
half in isolation, honestly, rather than fabricating the other half.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path, help="OpenVINO IR .xml path")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--camera", default="table_overhead", choices=["table_overhead", "third_person"])
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--save-frame", type=Path, default=None, help="optionally save the rendered frame as a PNG")
    args = parser.parse_args()

    from .intel_sim import IntelSimulationUnavailable, IntelTableWorld
    from .vision import MuJoCoCameraSource, OpenVINODetector, project_to_table

    try:
        world = IntelTableWorld()
    except IntelSimulationUnavailable as exc:
        raise SystemExit(f"{exc}\ninstall the Intel stack first: pip install -e \".[dev,intel]\"") from exc

    cam = MuJoCoCameraSource(world.model, world.data, args.camera, width=640, height=480)
    detector = OpenVINODetector(args.model, device=args.device, conf_threshold=args.conf)

    print("=" * 66)
    print(f"Camera {args.camera!r} -> OpenVINO ({args.device}) -> table-plane projection")
    print("=" * 66)
    print(f"detector class names (from the exported model's own metadata): {detector.class_names}")

    frame = cam.capture()
    if args.save_frame:
        from PIL import Image
        args.save_frame.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(args.save_frame)
        print(f"saved rendered frame -> {args.save_frame}")

    detections = detector.detect(frame)
    print(f"\n{len(detections)} detection(s):")
    pos, rot, fovy = cam.camera_pose()
    for d in detections:
        world_xy = project_to_table(d.center_xy, (640, 480), pos, rot, fovy)
        print(f"  {d.cls_name:<14} conf={d.conf:.2f} pixel={d.center_xy} "
              f"-> table_xy={tuple(round(v, 3) for v in world_xy) if world_xy else None}")

    if not detections:
        print("  (none above --conf threshold)")

    print(
        "\nNote: class labels are only as meaningful as the weights loaded above. "
        "The already-published thermal YOLO knows HIT-UAV classes "
        "(Person/Car/Bicycle/...), not tableware -- this proves the "
        "render->infer->project pipeline is real, not that the labels are "
        "correct for this scene. The fine-tuned 7-class table weights (OQ-008) "
        "now exist: models/table_yolo_v2_ft_2026-09-11_openvino_model/ -- pass "
        "its .xml here and the labels become the real plate/cup/fork/spoon/"
        "knife/napkin/drawer taxonomy; nothing else in this demo changes."
    )


if __name__ == "__main__":
    main()
