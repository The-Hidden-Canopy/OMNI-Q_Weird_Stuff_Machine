---
license: apache-2.0
library_name: tokenizers
tags:
  - tokenizer
  - bpe
  - byte-level-bpe
---

# omni_prism_bpe_256k

A 256,000-entry byte-level BPE tokenizer trained for the **Omni** model family
(seat PRISM, `state_coupled_v1`).

Omni previously borrowed the *lattice* family's 32k tokenizer. Its merges were
learned from a different corpus, and the mismatch was measurable, so this one
was fit in-domain.

## Why it exists — measured, not asserted

Evaluated on 4,000 **held-out** rows from the corpus's `eval` split. Neither
tokenizer was fit on any of them.

| Tokenizer | Vocab | chars/token | Tokens on the sample | vs. borrowed |
|---|---:|---:|---:|---:|
| `ida_lattice_bpe_32k` (borrowed) | 32,000 | 3.6015 | 1,745,754 | — |
| **`omni_prism_bpe_256k`** | 256,000 | **4.8222** | 1,303,826 | **−25.31%** |

A matched-vocabulary control isolates how much of that is merges rather than
vocabulary size: retrained at 32k on the same corpus, the same measurement gives
4.3418 chars/token, i.e. **−17.05%** against the borrowed tokenizer. So roughly
two thirds of the gain is domain fit and the remainder comes from the larger
vocabulary.

25% fewer tokens for the same text is 25% less compute per epoch and 25% more
text inside a fixed context window.

## Properties

- **Byte-level BPE.** No reachable `<unk>`; arbitrary bytes round-trip. Verified
  lossless on 1,000 held-out rows (0 mismatches).
- **Special tokens are pinned:** `<pad>`=0, `<s>`=1, `</s>`=2, `<unk>`=3. The
  training pipeline asserts these ids rather than trusting them.
- `add_prefix_space=false`, regex pre-tokenization, ByteLevel decoder.

## Training

Fit on the **train split only**. The source corpus has lineage-isolated
train/eval/test splits; a merge table fitted on held-out text encodes which
strings are common there and quietly flatters every later perplexity number on
that split, so eval and test were excluded outright.

| | |
|---|---|
| Rows | 998,950 |
| Text | 1,369,960,157 characters |
| Domain | Encyclopedic prose, task-oriented dialogue, JSON-structured targets |
| min_frequency | 2 |
| Train time | 269 s |

The training corpus is private and is not distributed with this tokenizer.

## Native 4096 block companion

The matching fixed-length native block pack for
`omni_780ma_source_400m_4096_r3` is stored under
`native/omni_780ma_source_400m_4096_r3/` in the private Hub repository. It
contains row-major `tokens.u32` and `labels.i32` files for train/eval/test,
plus `native_blocks_manifest.json` with the tokenizer and block hashes. The
pack was produced with `--allow-unadmitted` because the source release still
records `review_required_before_training_admission`; it is therefore a
preparation artifact, not a training-admission decision.

## Cost of the vocabulary

256k is not free. At `hidden_size=2560` the embedding is 655M parameters — in
FP32 with gradient and momentum, ~7.9 GB before any other tensor. Models using
this tokenizer need block-scaled (MXFP8/MXFP4) embedding masters, vocabulary
sharding, or both.

## Usage

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")
ids = tok.encode("The quick brown fox.").ids
assert tok.decode(ids) == "The quick brown fox."
```

## Provenance

`omni_tokenizer_manifest.json` carries the full build record: corpus release
identifier and SHA-256, row and character counts, the split used, and the
SHA-256 of `tokenizer.json` itself.
