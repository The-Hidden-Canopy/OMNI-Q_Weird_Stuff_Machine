"""OQ-008 step 4 — export the fine-tuned detector for both sponsor tracks.

    python perception/export.py --weights runs/table_yolo/ft/weights/best.pt

Produces:
  best.onnx                     opset 12, simplified   (portable)
  best_openvino_model/          Intel track  (OpenVINO IR — CPU / iGPU / NPU)
  best.onnx  ->  QAIRT          Qualcomm track: run qairt-converter on the ONNX
                                (Snapdragon X Elite, via AI Hub / QAIRT)

The detector's contract for Omni: emit {class, bbox, center, conf}; assign a
stable per-object id downstream by IoU + class across frames.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--no-openvino", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    onnx_path = model.export(format="onnx", opset=args.opset, simplify=True,
                             imgsz=args.imgsz, dynamic=False)
    print(f"ONNX: {onnx_path}")

    if not args.no_openvino:
        ov = model.export(format="openvino", imgsz=args.imgsz, half=True)
        print(f"OpenVINO IR: {ov}")

    print("\nQualcomm (QAIRT) — run on a Linux box or the AI Hub:")
    print(f"  qairt-converter --input_network {onnx_path} \\")
    print("     --output_path best_qnn.dlc --input_dtype image float32")
    print("  # then qairt-quantizer for INT8 with a small calib set")

    stem = Path(onnx_path).with_suffix("")
    print(f"\nbaselines: python -m perception.bench {onnx_path}  "
          f"(cf. evidence/benchmark_results/yolo_host_baseline_*)")
    _ = stem  # placeholder for a follow-up bench hook


if __name__ == "__main__":
    main()
