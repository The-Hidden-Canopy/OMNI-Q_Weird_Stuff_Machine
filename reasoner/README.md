# Planner-advisor training for the IDA Omni body

Trains the model `OmniPlanner` actually loads to answer in the reasoner
grammar, so `OMNIQ_OMNI_REASONER=omni` stops falling back to the
deterministic planner. Status as of 2026-09-13: **pipeline works end to end;
no real training run has been done yet.**

```
generate_planner_data.py   randomized worlds -> (OmniPlanner prompt, RulePlanner plan) pairs
train_planner_reasoner.py  trains the vendored reference body, exports a loader-format .pt + receipt
roundtrip_check.py         GATE: the exported artifact through OmniReferenceReasoner -> OmniPlanner
```

Data and `.pt` artifacts are gitignored; regenerate with
`--seed 7` (deterministic) and retrain.

## Why the Python body and not the native trainer's checkpoint

IDA-TRAIN-V2's native trainer produces a real checkpoint, but it trains only
part of this model: 97 tensors map across (embeddings, expert banks, norms,
router), 14 native tensors have no verified Python target (the SSM coupling
core `U/V/gamma/decay/sources/weights`), and **278 Python parameters — 7.2%,
including all attention, controls, LRSS/PSS and position embeddings — have no
native source at all.** A natively-pretrained checkpoint loaded into OMNI-Q is
therefore a partly-random model; that, not undertraining, is why CLI output was
gibberish. The native checkpoint is used here as *initialization*, and the rest
is trained in this venv (torch 2.11 + CUDA 12.6 on the RTX 4050).

## Measured facts that shape the design

- **Training must use the inference path.** Processing prompt+target as one
  evidence packet (boundary = full length) is ~4x faster than
  prompt-as-evidence + per-token continuation, but it is *a different
  computation*: measured cosine similarity between the two paths' hidden
  states at target positions is **0.02**. The router picks experts per packet,
  so a 1-token continuation packet routes differently than a 392-token
  evidence packet. Training the fast way would optimize a path inference never
  runs. The trainer therefore keeps `boundary = len(prompt)`; cost is ~10-12
  s/example and that is inherent.
- **Full-vocabulary projections are the recurring hazard on a 6 GiB card.**
  Three instances found: the native trainer's `[T, 256000]` logits buffer
  (fixed in IDA-TRAIN-V2 `f4196c7e`, 22x step speedup), this trainer's loss
  (only supervised rows are projected — see `selective_ce`), and generation
  (fixed in `omni_reasoner._last_position_head`: 12.29 GiB reserved -> 1.11
  GiB). Watch for a fourth.
- **Cumulative RTX 4050 training speedup:** the same 300-packet, GA=1 shape
  went from a first-attempt estimate of 33 hours, through a 4.25-hour v2 run,
  to an estimated 11.5 minutes after today's memory fix: about **170x end to
  end**. The last stage alone is about 22x versus v2 (82 -> ~1,900 tokens/s
  in the corresponding throughput comparison). Full arithmetic and provenance:
  [`evidence/benchmark_results/omni_reasoner_speed_2026-09-13/`](../evidence/benchmark_results/omni_reasoner_speed_2026-09-13/).
- **Generation costs ~0.9 s/token through the harness**, so a 48-token plan is
  ~45 s per planning call, and the engine calls the planner once per plan and
  once per replan. Worth budgeting for in the demo, or trimming `max_new_tokens`
  / the grammar.
- Training peak is ~2.2-2.4 GB with embeddings frozen + gradient checkpointing,
  well clear of the paging cliff.

## Run it

```bash
PYTHONPATH=src .venv/Scripts/python reasoner/generate_planner_data.py \
    --out reasoner/data/planner_pairs.jsonl --n 3300 --seed 7

PYTHONPATH=src .venv/Scripts/python reasoner/train_planner_reasoner.py \
    --data reasoner/data/planner_pairs.jsonl \
    --init-native <IDA_DATA>/runs/<run>/output/omni_fp32_checkpoint.bin \
    --out models/omni_planner --epochs 2 --ga 16

PYTHONPATH=src .venv/Scripts/python reasoner/roundtrip_check.py \
    --checkpoint models/omni_planner_best.pt \
    --receipt models/omni_planner_best_receipt.json --n 3 --device cuda
```

The metric is not loss: `roundtrip_check` and the in-training eval report the
fraction of proposed steps `OmniPlanner._validate` accepts, how many prompts
yield a usable plan, and exact match against the oracle — the same validation
the governed core applies at run time.

## Next

1. Size and launch a real run (a 500-example x 2-epoch pass is ~3 h at the
   measured rate; the full 3,300 x 2 is ~20 h — pick against the demo date).
2. Judge it on acceptance/usable/exact, not loss.
3. If it promotes, point `OMNIQ_OMNI_CHECKPOINT`/`OMNIQ_OMNI_RECEIPT` at the
   best artifact and run `demo_intel_reasoner` with `OMNIQ_OMNI_REASONER=omni`.

Open question worth deciding before (1): the rationale sentence is currently
templated from the oracle, so the model learns *our* phrasing rather than
fluent English. Fluency would need a prose pretraining stage; the grammar and
the state->plan mapping do not.
