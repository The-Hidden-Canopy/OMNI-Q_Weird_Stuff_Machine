"""Evaluate one 82-class checkpoint against both 82-class validation sets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _resolved_config(data_yaml: str) -> dict:
    import yaml

    path = Path(data_yaml).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    config["path"] = str(root)
    return config


def _metrics_dict(metrics, config: dict, data_yaml: str) -> dict:
    values = metrics.results_dict
    result = {
        "data_yaml": data_yaml,
        "data_root": config["path"],
        "classes": [str(v) for _, v in sorted(config["names"].items())],
        "precision": float(values["metrics/precision(B)"]),
        "recall": float(values["metrics/recall(B)"]),
        "map50": float(values["metrics/mAP50(B)"]),
        "map50_95": float(values["metrics/mAP50-95(B)"]),
    }
    try:
        result["per_class"] = [
            {
                "class": str(row["Class"]),
                "precision": float(row["Box-P"]),
                "recall": float(row["Box-R"]),
                "map50": float(row["mAP50"]),
                "map50_95": float(row["mAP50-95"]),
                "instances": int(row["Instances"]),
            }
            for row in metrics.summary()
        ]
    except (AttributeError, KeyError, TypeError, ValueError):
        result["per_class"] = None
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(args.model)
    rows = []
    for data_yaml in args.data:
        config = _resolved_config(data_yaml)
        metrics = model.val(
            data=config,
            split="val",
            imgsz=args.imgsz,
            device=args.device,
            plots=False,
            verbose=False,
        )
        rows.append(_metrics_dict(metrics, config, data_yaml))

    receipt = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(Path(args.model)),
        "imgsz": args.imgsz,
        "device": args.device,
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
