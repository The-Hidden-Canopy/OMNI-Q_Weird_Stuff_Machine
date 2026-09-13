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

## r1 diagnosis: the blocker is slot vocabulary, not decoding (2026-09-13)

Run r1 (1000 examples x 2 epochs, GA=8, lr 1e-4 cosine) drove loss from **12.55
to 2.43** while acceptance went **0.11 -> 0.00 -> 0.00 -> 0.00** at steps
50/100/150/200. Loss and the metric that matters moved in opposite directions,
which is exactly why this pipeline judges on acceptance.

The logged samples show what happened:

```
step  50: 'PLA\nSTEP PICK object=setting_1\nSTEP MOVE object=_1\nSTEP Carys_1 ...'
step 100: 'PLAN\nSTEP PICK object=setting_1\nSTEP MOVE object=setting_1 to=setting_1 ...'
step 200: 'PLAN\nSTEP PICK object=setting_1\nSTEP MOVE object=setting_1 to=setting_1 to=setting_ ...'
```

The **grammar is learned** — `PLAN`, `STEP PICK object=`, `STEP MOVE object= to=`
are all correct by step 100, where step 50 still produced `PLA` and `Carys_1`.
Two things then break it, and only one of them matters.

### Decoding is not the blocker (measured, not assumed)

Feeding text straight to the real `OmniPlanner._parse` + `_validate` against a
held-out world:

| text | proposed | accepted |
| --- | --- | --- |
| raw step-200 output | 2 | **0** |
| **perfect decode, repetition removed, same vocabulary** | 3 | **0** |
| identical shape with a real object id | 3 | **2** |
| oracle | 4 | 3 |

Greedy `argmax` with no repetition penalty does produce degenerate
`to=setting_1 to=setting_1 ...` tails, and that is worth fixing eventually. But
repairing it changes acceptance by **nothing**. Every step is rejected as
`omni:pick:setting_1` because `setting_1` is not an object.

### The blocker is one slot confusion

`setting_*` appears as an `object=` value **0 times in 14,583 training object
slots**. The model is not copying a frequent pattern -- it cannot distinguish
the two identifier slots:

- `object=` takes one of **26** ids; `fork_1` alone appears 1,915 times;
- `to=` takes one of **8** zones; `setting_1` appears 967 times.

Not a tokenization artifact either: every identifier is three tokens,
`[word, '_', digit]` (` cup`=9318, ` setting`=7056, ` napkin`=129628). After
`object=` the model emits ` setting` rather than ` cup`. Given that **all
attention parameters are randomly initialized** (the 7.2% of Python params with
no native source), and that a generally-pretrained prior favours "setting" over
"napkin", weak slot conditioning is the coherent explanation.

### What to do instead of another blind run

**Constrained decoding is the obvious lever, and it needs no retraining.**
`OmniPlanner` already knows the legal object ids for the current world -- that is
precisely what `_validate` checks *after* generation. Applying the same
knowledge *during* generation (restrict the token after `object=` to the legal
ids' first tokens, and after `to=` to the legal zones) removes this entire
failure class by construction. The table above bounds the payoff: same model,
same weights, real ids instead of `setting_1` -> 2 of 3 steps accepted.

Only after that is it worth asking whether the model needs more data (3,300
pairs exist, 1,000 were used), more epochs, or a better-initialized attention
stack.

### Constrained decoding measured on r1 (2026-09-13)

Same weights (`omni_planner_r1_final.pt`), same prompts, same greedy rule; the
only difference is whether identifier slots are fenced to the world's legal ids.
12 held-out prompts, `evidence/constrained_decode_r1.json`:

| | acceptance | usable | exact | proposed/prompt |
| --- | --- | --- | --- | --- |
| free | 0.121 | 0.333 | 0.000 | 2.75 |
| **fenced** | **0.463** | **0.667** | 0.000 | 3.42 |

**3.8x acceptance and 2x usable plans with no retraining.** For the demo that is
the difference between one prompt in three producing an executable step and two
in three.

What it does **not** fix, stated plainly:

- `exact` stays 0.000 — the oracle plan is never reproduced;
- the fence supplies vocabulary, not coherence. One row reads
  `PICK object=cup_1 / MOVE object=knife_1 ...` — every identifier legal, the
  plan incoherent. Only training fixes that;
- repetition persists (`to=setting_1 to=setting_1 ...`), exactly as the earlier
  measurement predicted (removing it changes acceptance by nothing);
- 12 prompts is a small sample.

So: fencing converts a model that learned plan *shape* into one that emits
*executable* steps. It does not make it a planner.
