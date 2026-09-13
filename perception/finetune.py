"""OQ-008 step 3 — fine-tune the table detector from OUR thermal YOLOv8n.

Base weights: ``KissTheHabit/yolov8n-hituav-thermal-finetune`` (a trained
YOLOv8n we own), or a packaged low-bit master (``*.mxfp8.npz`` produced by
``perception/package_quant_weights.py``). The head is re-initialised for 7
classes automatically by Ultralytics when ``data.yaml`` has 7 names.

    python perception/finetune.py --epochs 4 --freeze 10 --imgsz 640
    python perception/finetune.py --base runs/last.pt --epochs 2 --freeze 0
    python perception/finetune.py --base masters/table_yolo.mxfp8.npz \
        --w-master mxfp8 --epochs 4

Packed optimizer state (``--w-master mxfp8``): the authoritative trainable
weight state is an MXFP8 E4M3 payload plus UE8M0 K32 scales, and Lion momentum
is BF16. The stock YOLO graph retains a floating-point compute view because
its Conv2d/Linear/autograd path requires floating parameters; this mode does
not claim zero floating-point model residency. Repacking happens inside the
optimizer step, not in a batch-end callback. Design source: the IDA-TRAIN-V2
MXFP4-master simulation ablation (``docs/mxfp4-master-sim-ablation-2026-09-09.md``
in IDA-TRAIN-V2) — per-step RNE re-encode of the master is stable; stochastic
rounding *on the master* is a refuted unbounded random walk (SR is for
gradients only). MXFP4/MXFP2 masters are sim-only and refused at admission.

Breadth over repetition: the variety is in the real+synth data, so keep passes
few and early-stop on val mAP50.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from package_quant_weights import decode_npz  # noqa: E402

HF_BASE_REPO = "KissTheHabit/yolov8n-hituav-thermal-finetune"
_BASE_CANDIDATES = ("best.pt", "weights/best.pt", "yolov8n-hituav.pt", "model.pt")

ABLACTION_REF = (
    "IDA-TRAIN-V2 docs/mxfp4-master-sim-ablation-2026-09-09.md"
)
MASTER_TRAIN_FORMAT = "mxfp8"
MASTER_SIM_ONLY_FORMATS = ("mxfp4", "mxfp2")
W_MASTER_CHOICES = ("none", MASTER_TRAIN_FORMAT)
PACKED_LION_DEFAULT_LR = 3e-4

_REFUSAL_MSG = (
    "--w-master {fmt!r} is refused at admission: MXFP4/MXFP2 masters are "
    "SIM-ONLY residency states, not trainable masters. The IDA-TRAIN-V2 sim "
    "ablation (" + ABLACTION_REF + ") shows per-step RNE re-encode of the "
    "master is stable while stochastic rounding on the master is a refuted "
    "unbounded random walk (SR is for gradients only). Use --w-master mxfp8."
)


def resolve_base(base: str | None) -> str:
    if base:
        return base
    from huggingface_hub import HfApi, hf_hub_download

    files = {f for f in HfApi().list_repo_files(HF_BASE_REPO)}
    for cand in _BASE_CANDIDATES:
        if cand in files:
            return hf_hub_download(HF_BASE_REPO, cand)
    pt = next((f for f in files if f.endswith(".pt")), None)
    if pt is None:
        raise SystemExit(f"no .pt in {HF_BASE_REPO}; pass --base explicitly")
    return hf_hub_download(HF_BASE_REPO, pt)


def validate_w_master(fmt: str) -> str:
    """Validate ``--w-master``; refuse sim-only formats with the ablation cited."""
    if fmt == "none":
        return fmt
    if fmt == MASTER_TRAIN_FORMAT:
        return fmt
    raise SystemExit(_REFUSAL_MSG.format(fmt=fmt))


def manifest_path_for(npz_path: str | Path) -> Path:
    """Sibling manifest of a packed master: ``<stem>.<fmt>.manifest.json``."""
    p = Path(npz_path)
    stem = p.name[:-len(".npz")] if p.name.endswith(".npz") else p.stem
    return p.with_name(f"{stem}.manifest.json")


def check_master_format_contract(npz_path: str | Path, requested: str) -> dict:
    """Resume-gate contract: master format must equal the requested residency.

    PrecisionContract-shaped check (vendored IDEA, not code) — design source:
    IDA-TRAIN-V2 ``native/include/ida_native/omni_precision.hpp:33-45``
    (``OmniMasterFormat`` + ``omni_master_format_valid``: the master format is
    a checked contract at the checkpoint boundary, refused otherwise). When a
    sibling ``.manifest.json`` exists its ``format`` must match ``requested``;
    a mismatch is refused with a clear error.
    """
    manifest = read_master_manifest(npz_path)
    if not manifest:
        return {}
    actual = manifest.get("format")
    if actual != requested:
        raise ValueError(
            f"master format contract violated: {manifest_path_for(npz_path).name} "
            f"says format={actual!r} but --w-master requests {requested!r}. "
            "Refusing to train a master under the wrong residency format "
            "(contract shape: IDA-TRAIN-V2 omni_precision.hpp:33-45)."
        )
    return manifest


def decode_master_state_dict(npz_path: str | Path) -> "dict[str, object]":
    """Decode a packed master npz to a torch state_dict (fp32 tensors)."""
    import numpy as np

    torch = _torch()
    return {
        name: torch.from_numpy(np.ascontiguousarray(value).copy())
        for name, value in decode_npz(npz_path).items()
    }


def read_master_manifest(npz_path: str | Path) -> dict:
    """Sibling ``.manifest.json`` of a packed master, or ``{}`` if absent."""
    manifest_path = manifest_path_for(npz_path)
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def infer_detection_nc(state: "dict[str, object]") -> int | None:
    """Infer YOLOv8 detection classes from a packed state dict when present."""
    candidates = {
        int(value.shape[0])
        for name, value in state.items()
        if name.endswith("cv3.0.2.bias")
        and hasattr(value, "ndim")
        and value.ndim == 1
    }
    if len(candidates) > 1:
        raise ValueError(f"packed state has conflicting detection class counts: {sorted(candidates)}")
    return next(iter(candidates), None)


def load_packed_master(npz_path: str | Path):
    """Build a YOLOv8n from a packaged master: decode npz → load_state_dict.

    Any tier loads (a lower-tier master is just a degraded fp32 init); the
    *residency* contract is enforced separately via ``--w-master`` — keeping
    an mxfp4/mxfp2 master resident during training is refused at admission.
    The detector class count is inferred from the packed head so a 7-class
    artifact is not accidentally loaded into the YAML's default 80-class head.
    """
    from ultralytics import YOLO

    manifest = read_master_manifest(npz_path)
    state = decode_master_state_dict(npz_path)
    model = YOLO("yolov8n.yaml")
    nc = infer_detection_nc(state)
    existing_nc = getattr(model.model, "nc", None)
    if existing_nc is None:
        graph = getattr(model.model, "model", None)
        head = graph[-1] if graph is not None and len(graph) else None
        existing_nc = getattr(head, "nc", None)
    if nc is not None and existing_nc != nc:
        from ultralytics.nn.tasks import DetectionModel

        model.model = DetectionModel("yolov8n.yaml", nc=nc, verbose=False)
    model.model.load_state_dict(state, strict=True)
    name = str(npz_path)
    fmt = manifest.get("format") or (
        name.rsplit(".", 2)[-2] if name.endswith(".npz") else "unknown")
    cos = manifest.get("totals", {}).get("mean_weight_cosine")
    cos_txt = f", mean weight_cosine {cos:.6f}" if cos is not None else ""
    print(f"packed master: format {fmt}{cos_txt} (from {Path(npz_path).name})")
    return model


def build_model(base: str):
    """Suffix dispatch: ``.npz`` → packed master, otherwise ultralytics load."""
    if base.endswith(".npz"):
        return load_packed_master(base)
    from ultralytics import YOLO

    return YOLO(base)


def _class_name_map(names):
    """Normalize Ultralytics class names to a name-to-index mapping."""
    if isinstance(names, dict):
        return {str(name): int(index) for index, name in names.items()}
    if isinstance(names, (list, tuple)):
        return {str(name): index for index, name in enumerate(names)}
    return {}


def expand_detection_head(preloaded_model, target_nc: int, target_names=None):
    """Expand a loaded detector head while preserving compatible class rows.

    The v4 combined dataset carries the seven tableware classes plus COCO's
    additional names.  Keep the source detector's classifier rows intact and
    initialize only the newly introduced rows from a fresh matching model.
    Build the destination from the source model's own YAML so a larger
    YOLOv8s/m/l/x base is not accidentally rebuilt as YOLOv8n.  Classifier rows
    are copied by class name when available; this prevents COCO row 0 (person)
    from becoming the tableware plate row in the expanded head.
    """
    current_nc = int(getattr(preloaded_model, "nc", 0) or 0)
    if current_nc <= 0:
        detection_head = getattr(preloaded_model, "model", None)
        if detection_head is not None and len(detection_head):
            current_nc = int(getattr(detection_head[-1], "nc", 0) or 0)
    if current_nc == target_nc:
        return preloaded_model
    if current_nc <= 0 or target_nc < current_nc:
        raise ValueError(
            f"cannot adapt detection head from nc={current_nc} to nc={target_nc}"
        )
    from ultralytics.nn.tasks import DetectionModel

    model_cfg = copy.deepcopy(getattr(preloaded_model, "yaml", None))
    if not isinstance(model_cfg, dict):
        model_cfg = "yolov8n.yaml"
    else:
        model_cfg["nc"] = target_nc
    expanded = DetectionModel(model_cfg, nc=target_nc, verbose=False)
    source = preloaded_model.state_dict()
    destination = expanded.state_dict()
    source_names = _class_name_map(getattr(preloaded_model, "names", None))
    destination_names = _class_name_map(target_names)
    copied = 0
    for name, value in source.items():
        if name not in destination:
            continue
        target = destination[name]
        if target.shape == value.shape:
            target.copy_(value)
            copied += 1
            continue
        if (
            ".cv3." in name
            and (name.endswith(".2.weight") or name.endswith(".2.bias"))
        ) and value.ndim == target.ndim and value.shape[1:] == target.shape[1:]:
            if source_names and destination_names:
                rows_copied = 0
                for class_name, target_index in destination_names.items():
                    source_index = source_names.get(class_name)
                    if (
                        source_index is not None
                        and source_index < value.shape[0]
                        and target_index < target.shape[0]
                    ):
                        target[target_index].copy_(value[source_index])
                        rows_copied += 1
                copied += int(rows_copied > 0)
            else:
                target[: min(current_nc, target.shape[0])].copy_(
                    value[: min(current_nc, value.shape[0])]
                )
                copied += 1
    if copied == 0:
        raise ValueError("loaded detector had no compatible state to expand")
    expanded.load_state_dict(destination, strict=True)
    expanded.nc = target_nc
    return expanded


def make_packed_master_trainer(preloaded_model, packed_compute_pattern=None):
    """Build a trainer that keeps MXFP8 state inside the optimizer step."""
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    from packed_optimizer import PackedMXFP8Lion
    from packed_compute import install_packed_fp4_pilot

    class PackedMasterDetectionTrainer(DetectionTrainer):
        """DetectionTrainer using packed MXFP8 Lion state."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._resume_ckpt = None
            if self.resume:
                from ultralytics.nn.tasks import load_checkpoint

                _, self._resume_ckpt = load_checkpoint(self.args.resume)
            # A packaged .npz is already decoded into this exact model. Keeping
            # the module here prevents BaseTrainer from rebuilding a random
            # model from the YAML-only args path.
            target_nc = int(self.data["nc"])
            self.model = expand_detection_head(
                preloaded_model, target_nc, self.data.get("names")
            )
            self._packed_compute_modules = []
            if packed_compute_pattern:
                selected = install_packed_fp4_pilot(
                    self.model, packed_compute_pattern
                )
                self._packed_compute_modules = list(
                    module for module in self.model.modules()
                    if getattr(module, "precision_contract", None)
                    == "mxfp4_e2m1_ue8m0_k32"
                )
                print(
                    "packed compute pilot: MXFP4 E2M1/UE8M0 K32 storage, "
                    "stock Conv2d/autocast arithmetic; selected="
                    + ",".join(selected)
                )

        def check_resume(self, overrides):
            """Preserve an explicit total-epoch target when resuming."""
            requested_epochs = overrides.get("epochs")
            super().check_resume(overrides)
            if self.resume and requested_epochs is not None:
                self.args.epochs = requested_epochs

        def setup_model(self):
            """Return the checkpoint while retaining the preloaded model graph."""
            if self._resume_ckpt is not None:
                return self._resume_ckpt
            return super().setup_model()

        def final_eval(self):
            """Validate without stripping the packed optimizer checkpoint."""
            # Ultralytics' default final_eval() calls strip_optimizer() on
            # last.pt and best.pt. That is correct for release-only YOLO
            # weights, but it destroys the packed MXFP8 payload/scales and
            # BF16 Lion momentum needed for an exact continuation. Keep the
            # normal final validation/callback behavior and leave the
            # training checkpoint resumable.
            model = self.best if self.best.exists() else None
            if model:
                self.validator.args.plots = self.args.plots
                self.validator.args.compile = False
                self.metrics = self.validator(model=model)
                self.metrics.pop("fitness", None)
                self.epoch += 1
                self.run_callbacks("on_fit_epoch_end")
                self.epoch -= 1

        def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9,
                            decay=1e-5, iterations=1e5):
            torch = _torch()
            target = unwrap_model(model)
            norm_types = tuple(
                value for key, value in torch.nn.__dict__.items()
                if "Norm" in key
            )
            decay_params = []
            no_decay_params = []
            for module_name, module in target.named_modules():
                for param_name, parameter in module.named_parameters(recurse=False):
                    if not parameter.requires_grad:
                        continue
                    fullname = f"{module_name}.{param_name}" if module_name else param_name
                    if ("bias" in fullname or isinstance(module, norm_types)
                            or parameter.ndim == 1):
                        no_decay_params.append(parameter)
                    else:
                        decay_params.append(parameter)
            optimizer = PackedMXFP8Lion(
                [
                    {"params": decay_params, "weight_decay": decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=lr,
                betas=(momentum, 0.99),
            )
            optimizer.initialize_state()
            optimizer.packed_compute_modules = self._packed_compute_modules
            for module in self._packed_compute_modules:
                module.refresh_packed()
            return optimizer

    return PackedMasterDetectionTrainer


def _torch():
    import torch

    return torch


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).with_name("data.yaml")))
    ap.add_argument("--base", default=None,
                    help="local .pt or packaged master .npz; default = pull from HF")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume; --epochs is the new total target")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--freeze", type=int, default=10, help="freeze first N layers")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr0", type=float, default=None,
                    help="initial learning rate; packed Lion defaults to 3e-4")
    ap.add_argument("--lrf", type=float, default=0.01,
                    help="final LR fraction; transfer stages may use a higher floor")
    ap.add_argument("--nbs", type=int, default=64,
                    help="nominal batch size: ultralytics accumulates nbs/batch "
                         "micro-batches per optimizer step (gradient accumulation)")
    ap.add_argument("--patience", type=int, default=2, help="early-stop patience")
    ap.add_argument("--workers", type=int, default=8, help="dataloader workers")
    ap.add_argument("--project", default="runs/table_yolo")
    ap.add_argument("--name", default="ft")
    ap.add_argument("--w-master", default="none",
                    help="{none,mxfp8} use packed MXFP8 Lion optimizer state; "
                         "mxfp4/mxfp2 are sim-only and refused "
                         "(IDA-TRAIN-V2 sim ablation)")
    ap.add_argument(
        "--packed-compute",
        choices=("none", "mxfp4"),
        default="none",
        help="selected-layer MXFP4 packed consumer pilot; requires --w-master mxfp8",
    )
    ap.add_argument(
        "--packed-compute-pattern",
        default=r"^model\.(2|4|6|8|12|15|18|21)\.",
        help="regex over module names for the selected Conv2d pilot layers",
    )
    args = ap.parse_args(argv)

    w_master = validate_w_master(args.w_master)
    if args.packed_compute != "none" and w_master != MASTER_TRAIN_FORMAT:
        raise SystemExit(
            "--packed-compute mxfp4 requires --w-master mxfp8 so the "
            "authoritative optimizer state remains MXFP8 + BF16 Lion"
        )

    base = resolve_base(args.base)
    if base.endswith(".npz") and w_master != "none":
        check_master_format_contract(base, w_master)
    print(f"base weights: {base}")
    model = build_model(base)
    if w_master != "none":
        trainer = make_packed_master_trainer(
            model.model,
            args.packed_compute_pattern if args.packed_compute == "mxfp4" else None,
        )
        print(f"w-master {w_master}: packed optimizer-state training — "
              "MXFP8 payload/scales + BF16 Lion momentum; FP32 compute view; "
              "repack occurs inside every successful optimizer step")
        if args.packed_compute == "mxfp4":
            print(
                "packed compute mxfp4: selected Conv2d weights are decoded "
                "from FP4 payloads at the consumer boundary; this is not "
                "native Hopper WGMMA or Blackwell NVFP4 MMA"
            )
    else:
        trainer = None
    train_kwargs = {
        "trainer": trainer,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "nbs": args.nbs,
        "lrf": args.lrf,
        "freeze": args.freeze,
        "patience": args.patience,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "pretrained": True,
        "exist_ok": True,
    }
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    elif w_master != "none":
        train_kwargs["lr0"] = PACKED_LION_DEFAULT_LR
    if args.resume is not None:
        train_kwargs["resume"] = args.resume
        train_kwargs["save_dir"] = str(Path(args.project) / args.name)
    model.train(
        **train_kwargs,
    )
    print(f"best: {args.project}/{args.name}/weights/best.pt")
    print("next: python perception/export.py --weights "
          f"{args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
