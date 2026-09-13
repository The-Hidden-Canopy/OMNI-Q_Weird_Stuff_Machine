"""Train the IDA Omni reference body to advise OmniPlanner, and export it for the loader.

What this applies, deliberately, from the native-trainer investigation
(IDA-TRAIN-V2, 2026-09-12/13):

- The [T, 256k] FP32 logits were the thing that oversubscribed a 6 GiB card
  and paged over PCIe (20x). Here the LM head is bypassed inside the model
  (``lm_head`` -> Identity) and logits are computed only at the supervised
  target positions -- ~60-100 rows instead of ~400 -- against the tied
  embedding weight. A hard peak-memory gate warns when the run nears the
  5.9 GiB the driver leaves usable.
- Lion (one state tensor, the native trainer's optimizer), lr 1e-4, with the
  router parameters at 0.005x so top-1 expert assignment cannot churn.
- Gradient accumulation to a real effective batch (GA=32, micro-batch 1),
  the production recipe's reason: no single example owns an update.
- Best-by-metric and periodic (crash-safety) checkpoints, kept separately.
- The metric is what the governed core measures, not loss: the fraction of
  proposed steps ``OmniPlanner._validate`` accepts, and exact match against
  the oracle plan on held-out worlds.

Initialization comes from the natively pretrained checkpoint through
IDA-TRAIN-V2's bridge (embeddings, expert banks, norms, router); components
the native trainer never touched (attention, controls, LRSS/PSS, position
embeddings) start at init and are trained here.

Export matches ``omni_reference.omni_inference.load_omni_reference`` exactly:
``{"identity": {...}, "model": state_dict, "state_boundary": "completed_example",
"carried_history": None}`` plus a receipt carrying ``config.architecture`` (the
dict the loader hashes and rebuilds ``IDALatticeConfig`` from) and
``checkpoint_sha256``.

Usage (OMNI-Q venv, has CUDA torch):
    PYTHONPATH=src .venv/Scripts/python reasoner/train_planner_reasoner.py \
        --data reasoner/data/planner_pairs.jsonl --init-native <omni_fp32_checkpoint.bin> \
        --out models/omni_planner --epochs 3
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "integrations" / "intel" / "vendor"))
IDA_TRAIN = Path(r"C:/Users/damio/OneDrive/Documents/GitHub/IDA-TRAIN-V2")
sys.path.insert(0, str(IDA_TRAIN))
sys.path.insert(0, str(IDA_TRAIN / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from omni_q.contracts import Constraint, Detection  # noqa: E402
from omni_q.omni_planner import OmniPlanner  # noqa: E402
from omni_q.omni_reasoner import MockReasoner  # noqa: E402
from omni_q.world import MockWorld  # noqa: E402

EOS_ID = 2
USABLE_VRAM = 5.9e9
MEM_GATE = 5.6e9
TOKENIZER_DIR = ROOT / "integrations" / "intel" / "vendor" / "omni_reference" / "tokenizers" / "omni_prism_bpe_256k"


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99)):
        super().__init__(params, dict(lr=lr, betas=betas))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                m = self.state[p].setdefault("m", torch.zeros_like(p))
                p.add_(torch.sign(m.mul(b1).add(p.grad, alpha=1 - b1)), alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)


def world_from_meta(meta):
    objs = [Detection(o["object_id"], o["cls"], zone=o["zone"], target_zone=o["target_zone"]) for o in meta["objects"]]
    cons = tuple(Constraint(c["kind"], c["value"], c.get("source", "operator"), c.get("justification", "")) for c in meta["constraints"])
    w = MockWorld(objs, cons)
    w.goal = meta["goal"]
    for oid, holder in meta["ownership"].items():
        w._ownership[oid] = holder
    return w.state()


def plan_signature(text: str):
    sig = []
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("STEP "):
            parts = line[5:].split()
            op = parts[0].upper() if parts else ""
            kv = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
            sig.append((op, kv.get("object", ""), kv.get("to", "")))
    return tuple(sig)


def load_model(init_native: Path, device):
    from scripts.load_omni_native_checkpoint import load_into_model
    model, report = load_into_model(init_native, init_native.with_name("receipt.json"), device="cpu")
    print(f"init from native: {len(report['loaded'])} tensors mapped, "
          f"{len(report['unmapped_native'])} native tensors without a target, "
          f"{len(report['skipped_no_target'])} python params at init")
    return model.to(device)


def selective_ce(hidden, weight, labels):
    """CE at positions p whose next label is supervised: logits only for those rows."""
    lab = labels[0]
    sel = torch.nonzero(lab[1:] != -100).flatten()          # positions p with label p+1
    if sel.numel() == 0:
        return hidden.sum() * 0
    h = hidden[0, sel]
    logits = F.linear(h, weight).float()
    return F.cross_entropy(logits, lab[sel + 1])


@torch.no_grad()
def generate(model, weight, prompt_ids, max_new, device):
    seq = list(prompt_ids)
    boundary = torch.tensor([len(prompt_ids)], device=device)
    step = torch.zeros(1, dtype=torch.long, device=device)
    out_ids = []
    for _ in range(max_new):
        ids = torch.tensor([seq], device=device)
        out = model(input_ids=ids, omni_prediction_boundary=boundary,
                    omni_stream_ids=("eval",), omni_step_index=step)
        h = out.logits[0, len(seq) - 1]
        tok = int(F.linear(h, weight).argmax())
        out_ids.append(tok)
        seq.append(tok)
        if tok == EOS_ID:
            break
    return out_ids


def evaluate(model, weight, tok, rows, device, max_new=64, limit=24):
    model.eval()
    planner = OmniPlanner(MockReasoner())
    accepted = proposed = exact = usable = 0
    samples = []
    for row in rows[:limit]:
        pids = tok(row["prompt"], add_special_tokens=False)["input_ids"]
        gen = generate(model, weight, pids, max_new, device)
        text = tok.decode(gen, skip_special_tokens=True)
        world = world_from_meta(row["meta"])
        proposals, _ = planner._parse(text)
        steps, rejected = planner._validate(proposals, world)
        manip = [s for s in steps if s.contract == "manipulate"]
        proposed += len(proposals)
        accepted += len(manip)
        usable += bool(manip)
        exact += plan_signature(text) == plan_signature(row["target"])
        if len(samples) < 2:
            samples.append(text)
    model.train()
    n = max(1, min(limit, len(rows)))
    return {"acceptance": accepted / max(1, proposed), "usable": usable / n, "exact": exact / n,
            "proposed_per_prompt": proposed / n, "samples": samples}


def architecture_dict(config):
    from omni_reference.ida_lattice.config import IDALatticeConfig
    names = [p for p in inspect.signature(IDALatticeConfig.__init__).parameters if p not in ("self", "kwargs")]
    arch = {}
    for k in names:
        if hasattr(config, k):
            v = getattr(config, k)
            try:
                json.dumps(v)
            except TypeError:
                continue
            arch[k] = v
    arch["omni_scan_backend"] = "reference"
    return arch


def export(model, weight, out_stem: Path, tok_name: str, extra: dict):
    from omni_reference.ida_lattice.omni_state_model import OMNI_EXECUTION_REVISION
    real_head = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    real_head.weight = model.embed_tokens.weight
    saved_head, model.lm_head = model.lm_head, real_head
    arch = architecture_dict(model.config)
    arch_hash = hashlib.sha256(json.dumps(arch, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    identity = {"architecture_hash": arch_hash, "execution_revision": OMNI_EXECUTION_REVISION,
                "precision": "torch_reference_fp32", "checkpoint_revision": 2,
                "lineage": "omni_planner_advisor_synthetic_v1"}
    state = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    ckpt = out_stem.with_suffix(".pt")
    torch.save({"identity": identity, "model": state, "state_boundary": "completed_example",
                "carried_history": None}, ckpt)
    model.lm_head = saved_head
    with ckpt.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    receipt = {"format": "omni_planner_advisor_receipt_v1",
               "config": {"architecture": arch, "tokenizer_path": f"tokenizers/{tok_name}"},
               "checkpoint_sha256": digest, "identity": identity, **extra}
    out_stem.with_name(out_stem.name + "_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return ckpt, digest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--init-native", required=True)
    ap.add_argument("--out", default="models/omni_planner")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--ga", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--router-lr-scale", type=float, default=0.005)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-limit", type=int, default=8)
    ap.add_argument("--final-eval-limit", type=int, default=40)
    ap.add_argument("--eval-on-train", action="store_true",
                    help="evaluate on the TRAINING rows instead of held-out ones. For an "
                         "overfit probe: a setup that cannot reproduce examples it has "
                         "already seen is mis-wired, and that is worth knowing in minutes "
                         "rather than after a multi-hour run.")
    ap.add_argument("--train-embeddings", action="store_true",
                    help="also train the 197M-param tied vocab table (default frozen: it is the one "
                         "component the native pretraining initialized, and training it costs 1.57 GiB "
                         "of grad+optimizer state on a 6 GiB card)")
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    args = ap.parse_args()
    # The 256k byte-level BPE decodes to arbitrary Unicode, and an undertrained
    # model emits plenty of it. Windows' cp1252 console raises
    # UnicodeEncodeError mid-print and takes the whole run down for a display
    # reason -- it killed a probe at its first eval after the metrics had
    # already been computed. Same guard as roundtrip_check.py / omni_chat.py.
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    train = [r for r in rows if r["split"] == "train"]
    heldout = [r for r in rows if r["split"] == "eval"]
    if args.max_train:
        train = train[:args.max_train]
    if args.eval_on_train:
        heldout = list(train)
    enc = []
    for r in train:
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        t = tok(r["target"], add_special_tokens=False)["input_ids"] + [EOS_ID]
        if len(p) + len(t) > args.max_len:
            continue
        enc.append((p, t))
    print(f"train {len(enc)} (of {len(train)}), heldout {len(heldout)}; "
          f"mean prompt {sum(len(p) for p,_ in enc)/len(enc):.0f} tok, mean target {sum(len(t) for _,t in enc)/len(enc):.0f} tok")

    model = load_model(Path(args.init_native), device)
    model.train(); model.requires_grad_(True)
    weight = model.embed_tokens.weight            # tied lm_head weight
    model.lm_head = nn.Identity()                 # hidden states come out as .logits
    if not args.train_embeddings:
        weight.requires_grad_(False)
    if not args.no_grad_checkpoint:
        model.gradient_checkpointing_enable()
    router_params = [p for n, p in model.named_parameters() if p.requires_grad and ("router" in n or "pressure" in n)]
    router_ids = {id(p) for p in router_params}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in router_ids]
    opt = Lion([{"params": other, "lr": args.lr},
                {"params": router_params, "lr": args.lr * args.router_lr_scale}])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M total, {n_train/1e6:.1f}M trainable "
          f"(embeddings {'trained' if args.train_embeddings else 'frozen'}, grad-checkpoint "
          f"{'off' if args.no_grad_checkpoint else 'on'}), router group {sum(p.numel() for p in router_params)/1e6:.2f}M at {args.router_lr_scale}x lr")

    out_stem = Path(args.out); out_stem.parent.mkdir(parents=True, exist_ok=True)
    best = {"exact": -1.0, "acceptance": -1.0}
    opt_step = 0; micro = 0; t0 = time.time(); run_loss = 0.0; run_n = 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        order = list(range(len(enc))); random.shuffle(order)
        for i in order:
            p, t = enc[i]
            ids = torch.tensor([p + t], device=device)
            labels = torch.tensor([[-100] * len(p) + t], device=device)
            out = model(input_ids=ids, omni_prediction_boundary=torch.tensor([len(p)], device=device),
                        omni_stream_ids=(f"tr{i}",), omni_step_index=torch.zeros(1, dtype=torch.long, device=device))
            loss = selective_ce(out.logits, weight, labels)
            (loss / args.ga).backward()
            run_loss += float(loss); run_n += 1; micro += 1
            if micro % args.ga == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); opt_step += 1
                peak = torch.cuda.max_memory_allocated()
                flag = "  !! PEAK NEAR VRAM LIMIT (paging risk)" if peak > MEM_GATE else ""
                print(f"[step {opt_step} ep {epoch} ex {micro}] loss {run_loss/run_n:.4f}  "
                      f"{(time.time()-t0)/micro:.2f} s/ex  peak {peak/1e9:.2f} GB{flag}", flush=True)
                run_loss = 0.0; run_n = 0
                if opt_step % args.save_every == 0:
                    ck, dg = export(model, weight, out_stem.with_name(out_stem.name + "_latest"), TOKENIZER_DIR.name,
                                    {"training": {"opt_step": opt_step, "epoch": epoch}})
                    print(f"  periodic checkpoint -> {ck.name} {dg[:12]}", flush=True)
                if opt_step % args.eval_every == 0:
                    m = evaluate(model, weight, tok, heldout, device, limit=args.eval_limit)
                    print(f"  eval: acceptance {m['acceptance']:.2f}  usable {m['usable']:.2f}  exact {m['exact']:.2f}  "
                          f"proposed/prompt {m['proposed_per_prompt']:.1f}", flush=True)
                    print("  sample:", repr(m["samples"][0][:200]) if m["samples"] else "-", flush=True)
                    if (m["exact"], m["acceptance"]) > (best["exact"], best["acceptance"]):
                        best = {"exact": m["exact"], "acceptance": m["acceptance"], "opt_step": opt_step}
                        ck, dg = export(model, weight, out_stem.with_name(out_stem.name + "_best"), TOKENIZER_DIR.name,
                                        {"training": {"opt_step": opt_step, "epoch": epoch}, "eval": {k: v for k, v in m.items() if k != "samples"}})
                        print(f"  best checkpoint -> {ck.name} {dg[:12]}", flush=True)
    m = evaluate(model, weight, tok, heldout, device, limit=args.final_eval_limit)
    print(f"final eval on {min(args.final_eval_limit, len(heldout))} held-out: acceptance {m['acceptance']:.3f}  usable {m['usable']:.3f}  exact {m['exact']:.3f}")
    ck, dg = export(model, weight, out_stem.with_name(out_stem.name + "_final"), TOKENIZER_DIR.name,
                    {"training": {"opt_step": opt_step, "epochs": args.epochs}, "eval": {k: v for k, v in m.items() if k != "samples"}})
    print(f"final checkpoint -> {ck} {dg[:12]}; best {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
