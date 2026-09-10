# Qualcomm integration (bonus, not an entry)

We only qualify to enter the **Intel online** track — see
[`integrations/intel/README.md`](../intel/README.md) and the top-level
[`README.md`](../../README.md). This Qualcomm work is exploratory/bonus, not
part of what we're actually submitting.

**Challenge:** use GenieX, select and deploy a model from Hugging Face or
Qualcomm AI Hub, then bring it to life through on-device AI, hardware
integration, and seamless device-to-device collaboration.

**Hardware provided:** Snapdragon X Elite platform + Arduino UNO Q.

## Role in the loop

Two Qualcomm technologies, two distinct jobs:

| Tech | Job |
| --- | --- |
| AI Hub / QAIRT | perception acceleration — runs the thermal YOLO on the Hexagon NPU |
| GenieX | on-device reasoning / VLM node |

```
camera → YOLO on Qualcomm (AI Hub/QAIRT) → structured scene state
       → GenieX local model → OMNI Q task decision → Arduino UNO Q (device collaboration)
```

## Device split as capability nodes

```
SNAPDRAGON X ELITE          ARDUINO UNO Q
├── YOLO / vision           ├── sensors
├── Omni Q reasoning        ├── actuators
├── local model            ├── physical I/O
└── task routing           └── realtime device state
```

Device-to-device collaboration is intrinsic: *"Watch this workspace. When the
blue object enters region B, inspect it. If it matches the target, trigger the
actuator. If the connection disappears, continue locally."*

## TODO

- [ ] Export thermal YOLO → QAIRT / QNN, run on X Elite NPU
- [ ] Pick + deploy GenieX model (HF or AI Hub)
- [ ] Snapdragon ↔ Arduino transport for capability calls + device state
- [ ] Structured scene-state schema emitted by the perception node

## Low-bit CPU formats

OMNI-Q's own evaluation layer for sub-4-bit weight formats on plain CPUs
(OQ-029). It answers one question with measurements only: what do 2-bit
weight masters cost and preserve when inference stays on an ordinary CPU?

### The vendored codec

`lowbit/` contains a vendored numerical twin of the IDA-TRAIN-V2 2-bit
master-weight codecs (MXFP2 `INT2_UE8M0_K32` ternary payload + power-of-two
block scale; NVINT2 `INT2_E4M3_K16` four-level payload + E4M3 block scale +
FP32 per-tensor scale), plus OMNI-Q's own storage layer on top: 4×2-bit
(2×4-bit) per-byte payload packing and an MXFP4 E2M1/UE8M0 K32 reference arm.
Provenance and the parity-check policy: [`lowbit/vendor/SOURCE.md`](lowbit/vendor/SOURCE.md).
The public surface is `quantize_tensor` / `dequantize_tensor` /
`PackedTensor` (payload, scales, tensor_scale, stats, bits_per_weight).

### Evaluation tiers (`scripts/eval_yolo_2bit_cpu.py`)

Run against the HIT-UAV YOLOv8n with four arms — fp32 baseline, mxfp4,
nvint2, mxfp2 — weights-only RNE through the codec, then ordinary torch FP32
CPU execution (**software dequantize; no native 2-bit hardware exists on this
box**):

1. **Footprint + latency** — packed payload bytes vs fp32 per arm; forward
   latency (warmup + timed runs, mean/p50/p95) at `torch.set_num_threads(1)`
   and default threads; peak RSS.
2. **Bandwidth** — weights re-dequantized from `PackedTensor` right before
   each forward; dequant timed separately, so latency is reported with the
   restore included and excluded. This is the honest "decode on the fly"
   column for a resident packed master.
3. **Ternary prototype (mxfp2 only)** — multiply-free `nn.Linear` via the
   `{-1, 0, +1}` payload (block-scale einsum form) on a
   `[512,1024] × [1024,1024]` case, checked against the fp32 matmul.
   Prototype, not production.

Output agreement is output-tensor cosine vs the fp32 arm over fixed-seed
synthetic thermal-like inputs, plus a detection-count parity row via the
ultralytics predict path. mAP50 is blocked (HIT-UAV val split not local);
the published 0.825 is cited as reference. Latest bundle:
`evidence/benchmark_results/yolo_2bit_cpu_*/`.

### Why packed masters matter for the on-device constraint

The "keep inference on-device" rule runs into a wall with resident FP32
weights: the Omni reasoner class at ~6B parameters is ~24 GB fp32 — it does
not fit a normal PC. The same weights as packed 2-bit masters are ~1.7 GB,
which does. The codec and all three tiers above are weight-format-generic:
the same `quantize_tensor` round-trip and Tier 1/2 measurements apply
verbatim to the `OmniMorphableForCausalLM` Linear stacks once a checkpoint
is available locally.

### Re-run

```bash
# unit tests (OMNI-Q venv, numpy only)
TMPDIR=$PWD/tmp/pytest-tmp PYTHONPATH=src ./.venv/Scripts/python.exe \
    -m pytest tests/test_lowbit.py -q

# parity vs the source repo (optional oracle)
IDA_TRAIN_V2_ROOT=E:/HiddenCanopy/IDA-TRAIN-V2 ./.venv/Scripts/python.exe \
    -m pytest tests/test_lowbit.py -q

# the CPU evaluation itself (ISR venv: torch + ultralytics)
E:/HiddenCanopy/Semantically-Aware_ISR/.venv/Scripts/python.exe \
    integrations/qualcomm/scripts/eval_yolo_2bit_cpu.py
```
