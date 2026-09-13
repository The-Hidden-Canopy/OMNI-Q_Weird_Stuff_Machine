"""A/B a trained planner checkpoint with and without grammar-constrained decoding.

Same weights, same prompts, same greedy rule -- the only difference is whether
identifier slots are fenced to the world's legal object ids and zones
(``omni_q.plan_grammar``). Run r1 diagnosed 0.00 acceptance to a single slot
confusion (``object=setting_1``, a zone name); this measures whether fencing
that slot recovers real plans from weights that already learned the grammar.

The vendored ``omni_reference`` harness is used to *load* (so the digest,
architecture hash, execution revision, precision and checkpoint revision are all
verified exactly as the demo path verifies them) and is then left alone: the
decode loop here is local, because constraining requires per-token control that
``OmniInference.generate`` does not expose. Nothing in
``integrations/intel/vendor/`` is modified or monkey-patched.

Usage:
    PYTHONPATH=src .venv/Scripts/python reasoner/constrained_decode_probe.py \\
        --checkpoint models/omni_planner_r1_best.pt \\
        --receipt models/omni_planner_r1_best_receipt.json \\
        --n 12 --out evidence/constrained_decode_r1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reasoner"))

from omni_q.omni_planner import OmniPlanner  # noqa: E402
from omni_q.omni_reasoner import MockReasoner  # noqa: E402
from omni_q.plan_grammar import PlanGrammarConstraint, vocabulary_from_world  # noqa: E402
from train_planner_reasoner import plan_signature, world_from_meta  # noqa: E402

VENDOR = ROOT / "integrations" / "intel" / "vendor"


def load_reference(checkpoint: str, receipt: str, device: str):
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from omni_reference.omni_inference import load_omni_reference  # noqa: PLC0415
    return load_omni_reference(checkpoint, receipt, device=device)


@torch.no_grad()
def generate(model, weight, prompt_ids, max_new, device, tok, *,
             constraint: PlanGrammarConstraint | None = None, eos_id: int):
    """Greedy decode, optionally fenced to the plan grammar.

    Identical to the trainer's loop except for the masking step, so the A/B
    isolates the fence rather than comparing two different decoders.
    """
    seq = list(prompt_ids)
    boundary = torch.tensor([len(prompt_ids)], device=device)
    step = torch.zeros(1, dtype=torch.long, device=device)
    out_ids: list[int] = []
    for _ in range(max_new):
        ids = torch.tensor([seq], device=device)
        out = model(input_ids=ids, omni_prediction_boundary=boundary,
                    omni_stream_ids=("probe",), omni_step_index=step)
        hidden = out.logits[0, len(seq) - 1]
        logits = F.linear(hidden, weight)
        if constraint is not None:
            allowed = constraint.allowed_tokens()
            if allowed:
                mask = torch.full_like(logits, float("-inf"))
                index = torch.tensor(sorted(allowed), device=logits.device)
                mask[index] = logits[index]
                logits = mask
        token = int(logits.argmax())
        out_ids.append(token)
        seq.append(token)
        if constraint is not None:
            constraint.accept(token, tok.decode([token]))
        if token == eos_id:
            break
    return out_ids


def score(planner, text, row, world):
    proposals, _ = planner._parse(text)
    steps, rejected = planner._validate(proposals, world)
    manip = [s for s in steps if s.contract == "manipulate"]
    return {
        "proposed": len(proposals),
        "accepted": len(manip),
        "usable": bool(manip),
        "exact": plan_signature(text) == plan_signature(row["target"]),
        "rejected": list(rejected) if rejected else [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--data", default="reasoner/data/planner_pairs.jsonl")
    ap.add_argument("--tokenizer", default=r"C:/Users/damio/AppData/Local/IDA_DATA/tokenizer_256k")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", help="write the comparison to this JSON path")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    heldout = [r for r in rows if r["split"] == "eval"][: args.n]

    device = torch.device(args.device)
    model = load_reference(args.checkpoint, args.receipt, args.device)
    model.eval()
    weight = model.get_input_embeddings().weight
    import torch.nn as nn
    model.lm_head = nn.Identity()          # hidden states come out as .logits
    planner = OmniPlanner(MockReasoner())
    eos_id = int(tok.eos_token_id)

    totals = {"free": {"proposed": 0, "accepted": 0, "usable": 0, "exact": 0},
              "fenced": {"proposed": 0, "accepted": 0, "usable": 0, "exact": 0}}
    per_row = []
    started = time.time()
    for i, row in enumerate(heldout):
        pids = tok(row["prompt"], add_special_tokens=False)["input_ids"]
        world = world_from_meta(row["meta"])
        vocab = vocabulary_from_world(world)

        free_text = tok.decode(
            generate(model, weight, pids, args.max_new, device, tok, eos_id=eos_id),
            skip_special_tokens=True)
        fence = PlanGrammarConstraint(
            vocab, lambda s: tok(s, add_special_tokens=False)["input_ids"])
        fenced_text = tok.decode(
            generate(model, weight, pids, args.max_new, device, tok,
                     constraint=fence, eos_id=eos_id),
            skip_special_tokens=True)

        free = score(planner, free_text, row, world)
        fenced = score(planner, fenced_text, row, world)
        for key, result in (("free", free), ("fenced", fenced)):
            for field in totals[key]:
                totals[key][field] += int(result[field])
        per_row.append({"index": i, "free": free, "fenced": fenced,
                        "free_text": free_text[:200], "fenced_text": fenced_text[:200],
                        "legal_objects": list(vocab.objects)})
        print(f"[{i}] free: {free['accepted']}/{free['proposed']} accepted   "
              f"fenced: {fenced['accepted']}/{fenced['proposed']} accepted", flush=True)
        print(f"     fenced text: {fenced_text[:120]!r}", flush=True)

    n = max(1, len(heldout))
    summary = {"checkpoint": args.checkpoint, "held_out_prompts": len(heldout),
               "max_new_tokens": args.max_new, "wall_s": round(time.time() - started, 1)}
    for key in ("free", "fenced"):
        t = totals[key]
        summary[key] = {
            "acceptance": t["accepted"] / max(1, t["proposed"]),
            "usable": t["usable"] / n,
            "exact": t["exact"] / n,
            "proposed_per_prompt": t["proposed"] / n,
        }
    print("\n--- summary ---")
    for key in ("free", "fenced"):
        s = summary[key]
        print(f"{key:7} acceptance {s['acceptance']:.3f}  usable {s['usable']:.3f}  "
              f"exact {s['exact']:.3f}  proposed/prompt {s['proposed_per_prompt']:.2f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"summary": summary, "rows": per_row}, indent=2), encoding="utf-8")
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
