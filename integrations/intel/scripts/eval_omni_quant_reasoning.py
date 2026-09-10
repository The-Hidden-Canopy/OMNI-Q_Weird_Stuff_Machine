"""Phase-2 harness: 2-bit weight-only quantization formats on the IDA Omni reasoner.

Evaluates three arms on equal footing, with measurements only and no
preassigned verdicts:

* ``fp32``       — baseline, no quantization
* ``mxfp2_rne``  — W-only RNE through ``ida_train.native.mxfp2_codec``
                   (MXFP2_INT2_UE8M0_K32)
* ``nvint2_rne`` — W-only RNE through the same codec
                   (NVINT2_INT2_E4M3_K16)

W-only policy (mirrors IDA-TRAIN-V2 ``scripts/sim_mxfp2_gemm_compute.py``):
for every ``nn.Linear`` weight in the model, encode with the codec's
``encode_tensor(..., mode="rne")``, decode back to float32 with
``decode_tensor``, load the dequantized weight in place, then run the
evaluation prompt. Nothing is reimplemented here; the codec owns the
numerics. **Format simulation, numpy dequantize — no hardware 2-bit
execution exists.**

A format is a format: both quantized arms are reported symmetrically with
measurements only. No accuracy or quality verdict is drawn by this harness —
in fixture mode the run is explicitly labeled "plumbing validation on
numerical fixture, not a capability result".

Read-only consumer of IDA-TRAIN-V2 and Ask_IDA_CLI: nothing in those repos
is created or modified. All evidence lands under
``evidence/benchmark_results/omni_quant_2bit_<YYYYMMDD>/`` and the directory
is never overwritten (a suffixed timestamp variant is created instead).

Environment:
    IDA_TRAIN_V2_ROOT   (default E:/HiddenCanopy/IDA-TRAIN-V2)
    ASK_IDA_CLI_ROOT    (default E:/HiddenCanopy/Ask_IDA_CLI)
    OMNIQ_OMNI_CHECKPOINT / OMNIQ_OMNI_RECEIPT
                        (unset -> tiny CPU fixture under IDA-TRAIN-V2
                         artifacts/omni_cuda_audit_p5200_20260908/)

Portions derived from *The Hidden Canopy LLC* — [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ATTRIBUTION = (
    "Portions derived from *The Hidden Canopy LLC* — "
    "[`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). "
    "Used with permission."
)

OMNIQ_ROOT = Path(__file__).resolve().parents[3]

ARMS = {
    "fp32": None,  # baseline: no quantization
    "mxfp2_rne": ("mxfp2", "rne"),
    "nvint2_rne": ("nvint2", "rne"),
}

FIXTURE_LABEL = (
    "plumbing validation on numerical fixture, not a capability result"
)
FORMAT_SIM_LABEL = (
    "format simulation, numpy dequantize — no hardware 2-bit execution exists"
)
STANDING_NOTE = (
    "a format is a format — both quantized arms are reported symmetrically "
    "with measurements only; no quality verdict is assigned by this harness"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_paths() -> dict:
    ida_root = Path(os.environ.get("IDA_TRAIN_V2_ROOT", "E:/HiddenCanopy/IDA-TRAIN-V2"))
    ask_root = Path(os.environ.get("ASK_IDA_CLI_ROOT", "E:/HiddenCanopy/Ask_IDA_CLI"))
    fixture_dir = ida_root / "artifacts" / "omni_cuda_audit_p5200_20260908"
    checkpoint = Path(os.environ.get("OMNIQ_OMNI_CHECKPOINT", fixture_dir / "checkpoint-000001.pt"))
    receipt = Path(os.environ.get("OMNIQ_OMNI_RECEIPT", fixture_dir / "receipt.json"))
    return {
        "ida_root": ida_root,
        "ask_root": ask_root,
        "checkpoint": checkpoint,
        "receipt": receipt,
        "fixture_mode": "OMNIQ_OMNI_CHECKPOINT" not in os.environ,
    }


def _ensure_syspath(paths: dict) -> None:
    for entry in (str(OMNIQ_ROOT), str(paths["ask_root"]), str(paths["ida_root"] / "src"),
                  str(OMNIQ_ROOT / "integrations" / "qualcomm" / "lowbit")):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _read_receipt_metadata(receipt: Path) -> dict:
    try:
        meta = json.loads(receipt.read_text(encoding="utf-8"))
    except Exception as exc:  # missing/corrupt receipt: record, don't guess
        return {"error": f"receipt unreadable: {exc}"}
    cfg = meta.get("config", {})
    arch = cfg.get("architecture", {})
    return {
        "promotion_status": meta.get("promotion_status"),
        "training_status": meta.get("training_status"),
        "capability_status": meta.get("capability_status"),
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
        "receipt_tokenizer_path": cfg.get("tokenizer_path"),
        "architecture_family": arch.get("architecture_family"),
        "omni_architecture": arch.get("omni_architecture"),
        "hidden_size": arch.get("hidden_size"),
        "num_hidden_layers": arch.get("num_hidden_layers"),
        "vocab_size": arch.get("vocab_size"),
    }


def _quantize_linear_weights(model, fmt: str) -> dict:
    """W-only RNE: encode/decode every nn.Linear weight through the codec
    and load the dequantized float32 tensor back in place. Biases untouched.
    NOTE: this model ties lm_head.weight to embed_tokens.weight, so the
    lm_head pass also dequantizes the shared embedding tensor (recorded)."""
    import numpy as np
    import torch
    # Vendored numerical twin (integrations/qualcomm/lowbit/vendor/) — the
    # dependency lives in this repo, IDA-TRAIN-V2 is not imported (owner
    # requirement). See vendor/SOURCE.md.
    from vendor.mxfp2_codec import decode_tensor, encode_tensor

    stats = {"linear_modules": 0, "elements": 0, "tied_lm_head_embedding": False}
    lm_head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if module.weight is lm_head_weight:
                stats["tied_lm_head_embedding"] = True
            w = module.weight.detach().to(torch.float32).cpu().numpy()
            flat = np.ascontiguousarray(w.reshape(-1), dtype=np.float32)
            if fmt == "mxfp2":
                payload, scales = encode_tensor(flat, fmt=fmt, mode="rne")
                deq = decode_tensor(payload, scales, flat.size, fmt=fmt)
            elif fmt == "nvint2":
                payload, scales, tensor_scale = encode_tensor(flat, fmt=fmt, mode="rne")
                deq = decode_tensor(payload, scales, flat.size, fmt=fmt,
                                    tensor_scale=tensor_scale)
            else:  # pragma: no cover - guarded by ARMS table
                raise ValueError(f"unknown 2-bit format {fmt!r}")
            module.weight.copy_(
                torch.from_numpy(deq.reshape(w.shape).astype(np.float32))
            )
            stats["linear_modules"] += 1
            stats["elements"] += int(flat.size)
    return stats


def _format_parameters() -> dict:
    """Format parameters copied verbatim from the vendored codec."""
    from vendor import mxfp2_codec as c

    return {
        "source_module": "integrations.qualcomm.lowbit.vendor.mxfp2_codec (vendored twin of ida_train.native.mxfp2_codec)",
        "mxfp2": {
            "weights_dtype": c.MXFP2_WEIGHTS_DTYPE,
            "block_size": c.BLOCK_SIZE_MXFP2,
            "ladder_max": c.TERNARY_MAX,
            "payload_levels": [float(v) for v in c.TERNARY_MAGNITUDES],
            "scale": "UE8M0 power-of-two byte per block",
            "mode": "rne (tie-to-even-code)",
        },
        "nvint2": {
            "weights_dtype": c.NVINT2_WEIGHTS_DTYPE,
            "block_size": c.BLOCK_SIZE_NVINT2,
            "ladder_max": c.INT2_MAX,
            "payload_levels": [float(v) for v in c.INT2_LEVELS],
            "scale": "E4M3 byte per block + FP32 per-tensor second-level scale",
            "mode": "rne (tie-to-even-code)",
        },
        "policy": "weight-only: quantize weight, dequantize, load back; "
                  "biases and non-Linear parameters untouched",
    }


def run_arm(arm: str, spec, prompt: str, paths: dict, args) -> dict:
    """Fresh model load per arm (equal footing), optional W-only quantize,
    then N timed generation repeats on fresh sessions."""
    import numpy as np
    import torch
    from omni_q.omni_reasoner import OmniReferenceReasoner

    receipt_meta = _read_receipt_metadata(paths["receipt"])
    tokenizer_path = None
    rel = receipt_meta.get("receipt_tokenizer_path")
    if rel:
        candidate = paths["ida_root"] / rel
        if candidate.exists():
            tokenizer_path = candidate

    reasoner = OmniReferenceReasoner(
        paths["checkpoint"], paths["receipt"],
        ask_ida_cli_root=paths["ask_root"],
        ida_train_root=paths["ida_root"],
        tokenizer_path=tokenizer_path,
        device="cpu",
        max_new_tokens=args.max_new_tokens,
    )
    reasoner._load()  # identity-gated; raises on digest/arch mismatch
    model = reasoner._inference.model

    arm_result = {
        "quantization": None,
        "repeats": [],
        "wall_seconds": {},
        "error": None,
    }

    if spec is not None:
        fmt, _mode = spec
        arm_result["quantization"] = _quantize_linear_weights(model, fmt)

    repeats = []
    for i in range(args.repeats):
        session = f"omniq-quant-eval-{arm}-{i}"
        t0 = time.perf_counter()
        result = reasoner.reason(session, prompt, max_new_tokens=args.max_new_tokens)
        wall = time.perf_counter() - t0
        repeats.append({
            "session": session,
            "wall_seconds": wall,
            "token_count": len(result.token_ids),
            "token_ids": [int(t) for t in result.token_ids],
            "decoded_text": result.text,
        })

    walls = [r["wall_seconds"] for r in repeats]
    arm_result["repeats"] = repeats
    arm_result["wall_seconds"] = {
        "mean": float(np.mean(walls)),
        "min": float(np.min(walls)),
        "max": float(np.max(walls)),
    }
    return {
        "arm": arm_result,
        "artifact_identity": reasoner.artifact_identity,
        "model_class": type(model).__name__,
        "tokenizer": {
            "class": type(reasoner._tokenizer).__name__,
            "name_or_path": str(getattr(reasoner._tokenizer, "name_or_path", "")),
        },
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repeats", type=int, default=3,
                   help="timed generation repeats per arm (fixture runs may lower this)")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--out-root", type=Path,
                   default=OMNIQ_ROOT / "evidence" / "benchmark_results")
    args = p.parse_args(argv)

    paths = _resolve_paths()
    _ensure_syspath(paths)

    import numpy as np
    import torch
    import transformers
    from omni_q.omni_planner import OmniPlanner
    from omni_q.world import MockWorld

    receipt_meta = _read_receipt_metadata(paths["receipt"])

    # Render the evaluation prompt exactly as the planner would.
    world = MockWorld.sample().state()
    planner = OmniPlanner(reasoner=None)  # used only for _render_prompt
    prompt = planner._render_prompt("tidy the workspace", world, None)

    arms: dict[str, dict] = {}
    shared: dict[str, dict] = {}
    for arm, spec in ARMS.items():
        print(f"[eval] arm={arm} ...", flush=True)
        try:
            out = run_arm(arm, spec, prompt, paths, args)
        except Exception as exc:
            arms[arm] = {"error": f"{type(exc).__name__}: {exc}", "repeats": [],
                         "wall_seconds": {}, "quantization": None}
            print(f"[eval] arm={arm} FAILED: {exc}", flush=True)
            continue
        arms[arm] = out["arm"]
        shared[arm] = {
            "artifact_identity": out["artifact_identity"],
            "model_class": out["model_class"],
            "tokenizer": out["tokenizer"],
        }
        print(f"[eval] arm={arm} done: "
              f"{[r['token_count'] for r in out['arm']['repeats']]} tokens, "
              f"mean wall {out['arm']['wall_seconds'].get('mean', 0):.3f}s",
              flush=True)

    # Evidence directory: refuse to overwrite.
    stamp = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_root / f"omni_quant_2bit_{stamp}"
    if out_dir.exists():
        out_dir = args.out_root / f"omni_quant_2bit_{stamp}_{datetime.now().strftime('%H%M%S')}"
    out_dir.mkdir(parents=True)

    labels = {
        "execution": FORMAT_SIM_LABEL,
        "standing_note": STANDING_NOTE,
        "quality": "no accuracy/quality claims: quality verdicts are out of "
                   "scope for this harness",
    }
    if paths["fixture_mode"]:
        labels["artifact"] = FIXTURE_LABEL
        labels["receipt_status"] = (
            f"receipt.json: promotion_status={receipt_meta.get('promotion_status')}, "
            f"training_status={receipt_meta.get('training_status')}, "
            f"capability_status={receipt_meta.get('capability_status')} — "
            "checkpoint is an unverified numerical fixture, NOT a promoted/"
            "trained body"
        )
    else:
        labels["artifact"] = (
            "checkpoint supplied via OMNIQ_OMNI_CHECKPOINT; promoted/trained "
            "status per receipt metadata below"
        )

    results = {
        "harness": "integrations/intel/scripts/eval_omni_quant_reasoning.py",
        "question": "What do MXFP2_INT2_UE8M0_K32 and NVINT2_INT2_E4M3_K16 "
                    "weight-only RNE do to the IDA Omni reference reasoner's "
                    "prompted generation, vs the fp32 baseline, on equal footing?",
        "created_utc": _utc_now(),
        "attribution": ATTRIBUTION,
        "labels": labels,
        "environment": {
            "interpreter": sys.executable,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": transformers.__version__,
        },
        "artifact": {
            "checkpoint": str(paths["checkpoint"]),
            "receipt": str(paths["receipt"]),
            "fixture_mode": paths["fixture_mode"],
            "receipt_metadata": receipt_meta,
            "model_class": next(iter(shared.values()), {}).get("model_class"),
            "artifact_identity": next(iter(shared.values()), {}).get("artifact_identity"),
            "tokenizer": next(iter(shared.values()), {}).get("tokenizer"),
        },
        "prompt": {
            "goal": "tidy the workspace",
            "world": "omni_q.world.MockWorld.sample().state()",
            "rendered_by": "omni_q.omni_planner.OmniPlanner._render_prompt",
            "text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "text": prompt,
        },
        "protocol": {
            "arms": list(ARMS),
            "quantization": "W-only RNE per quantized arm: encode_tensor -> "
                            "decode_tensor through ida_train.native.mxfp2_codec, "
                            "dequantized fp32 weight loaded in place; fp32 arm "
                            "loads the checkpoint unmodified",
            "tied_weight_note": "OmniMorphableForCausalLM ties "
                                "lm_head.weight to embed_tokens.weight; the "
                                "lm_head Linear pass therefore also "
                                "dequantizes the shared embedding tensor",
            "fresh_model_load_per_arm": True,
            "fresh_session_per_repeat": True,
            "repeats_per_arm": args.repeats,
            "max_new_tokens": args.max_new_tokens,
            "device": "cpu",
        },
        "format_parameters": _format_parameters(),
        "arms": arms,
        "per_arm_identity": shared,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=1),
                                          encoding="utf-8")

    _write_readme(out_dir, results)

    print(f"\nevidence: {out_dir}")
    for arm, r in arms.items():
        ws = r.get("wall_seconds") or {}
        counts = [rep["token_count"] for rep in r.get("repeats", [])]
        print(f"  {arm:12s} tokens={counts} "
              f"wall mean/min/max = {ws.get('mean', 0):.3f}/{ws.get('min', 0):.3f}/"
              f"{ws.get('max', 0):.3f}s" + (f"  ERROR: {r['error']}" if r.get("error") else ""))
    return 0


def _write_readme(out_dir: Path, results: dict) -> None:
    art = results["artifact"]
    env = results["environment"]
    labels = results["labels"]
    arms = results["arms"]
    fmtp = results["format_parameters"]

    def _row(arm: str) -> str:
        r = arms.get(arm, {})
        ws = r.get("wall_seconds") or {}
        counts = [rep["token_count"] for rep in r.get("repeats", [])]
        if r.get("error"):
            return f"| {arm} | ERROR: {r['error']} |"
        return (f"| {arm} | {counts} | {ws.get('mean', 0):.3f} | "
                f"{ws.get('min', 0):.3f} | {ws.get('max', 0):.3f} |")

    lines = [
        "# Omni 2-bit W-only quantization harness — Phase 2",
        "",
        f"Run: {results['created_utc']} by `integrations/intel/scripts/eval_omni_quant_reasoning.py`.",
        "",
        f"> **{labels['execution']}.** {labels['standing_note']}.",
        "",
        "## Artifact",
        "",
        f"- Checkpoint: `{art['checkpoint']}`",
        f"- Receipt: `{art['receipt']}`",
        f"- Model class: `{art['model_class']}`",
        f"- Interpreter: `{env['interpreter']}` (torch {env['torch']}, "
        f"transformers {env['transformers']}, numpy {env['numpy']})",
        f"- Checkpoint sha256 (verified by the identity-gated loader): "
        f"`{(art.get('artifact_identity') or {}).get('checkpoint_sha256', 'n/a')}`",
        "",
        f"**Artifact status:** {labels['artifact']}.",
        "",
        f"_{labels.get('receipt_status', '')}_",
        "",
        "Quality verdicts are **out of scope** for this harness: no accuracy or",
        "capability claim is made from these generations. If the checkpoint is",
        "a random-init or unverified body, these numbers describe plumbing only.",
        "",
        "## Protocol",
        "",
        f"- {results['protocol']['quantization']}.",
        f"- Fresh model load per arm; fresh session per repeat; "
        f"N={results['protocol']['repeats_per_arm']} repeats/arm; "
        f"max_new_tokens={results['protocol']['max_new_tokens']}; device=cpu.",
        f"- Prompt: OmniPlanner._render_prompt, goal \"tidy the workspace\", "
        f"MockWorld.sample().state() (sha256 {results['prompt']['text_sha256'][:16]}…).",
        f"- **Tied-weight note:** {results['protocol']['tied_weight_note']}.",
        "",
        "## Formats (parameters copied from ida_train.native.mxfp2_codec)",
        "",
        f"- **mxfp2_rne** — `{fmtp['mxfp2']['weights_dtype']}`, block K{fmtp['mxfp2']['block_size']}, "
        f"payload levels {fmtp['mxfp2']['payload_levels']}, {fmtp['mxfp2']['scale']}.",
        f"- **nvint2_rne** — `{fmtp['nvint2']['weights_dtype']}`, block K{fmtp['nvint2']['block_size']}, "
        f"payload levels {fmtp['nvint2']['payload_levels']}, {fmtp['nvint2']['scale']}.",
        "",
        "## Results (per arm)",
        "",
        "| arm | tokens per repeat | wall mean (s) | wall min (s) | wall max (s) |",
        "|---|---|---|---|---|",
        *[_row(a) for a in results["protocol"]["arms"]],
        "",
        "Generated token ids and decoded text per repeat are in `results.json`.",
        "Timing is per generation packet (one prompt, up to 32 greedy tokens).",
        "",
        "Files: `results.json` (full per-arm metrics, honest labels, format",
        "parameters, artifact identity).",
        "",
        f"_{ATTRIBUTION}_",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
