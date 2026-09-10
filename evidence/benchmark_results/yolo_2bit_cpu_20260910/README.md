# YOLO 2-bit CPU evaluation — MXFP4 / NVINT2 / MXFP2 weight-only RNE (2026-09-10)

Measured by Gerron/Kimi (OQ-029). Companion to the host baseline
(`../yolo_host_baseline_2026-09-10/`); same honesty rules: **latency and
agreement probe on synthetic inputs, not an accuracy claim.**

**Canonical bundle — reproduced fully in-repo 2026-09-10 (~21:48Z): OMNI-Q
own venv (`./.venv/Scripts/python.exe`, torch 2.14.0+cpu) against the
in-repo model copy `models/hituav_yolov8n_finetuned_2026-08-26.pt`
(sha256 `168b8c06…e48d2`, byte-identical to the Semantically-Aware_ISR
artifact). An earlier run of the same script via the ISR venv
(torch 2.6.0+cu124) produced identical weight/output cosines and latencies
within run-to-run noise; it is superseded by this in-repo run.**

## Question

What do weight-only RNE 2-bit (and 4-bit reference) weight masters do to the
HIT-UAV YOLOv8n's CPU forward latency, memory footprint, and output agreement
vs the fp32 baseline — when inference stays on an ordinary CPU?

## Model

`models/hituav_yolov8n_finetuned_2026-08-26.pt` (3,011,823 params, 12.0 MB
fp32), ultralytics YOLOv8n 8.4.146, torch load on CPU. Arms: **fp32**
(baseline), **mxfp4** (E2M1 payload, UE8M0 K32 — the 4-bit ladder reference),
**nvint2** (INT2 `{-1.5,-.5,+.5,+1.5}`, E4M3 K16 + FP32 per-tensor scale),
**mxfp2** (ternary `{-1,0,+1}`, UE8M0 K32). Weights-only RNE through
`integrations/qualcomm/lowbit` (`quantize_tensor` → `dequantize_tensor` →
load back with `torch.no_grad`), every Conv2d/Linear weight.

## Environment

- Hardware: Intel Core i7-8750H @ 2.20 GHz (6C/12T), 40 GB RAM, Windows 11
  dev box (shared; this run made on a quiet box, load ~5%)
- Interpreter: `E:\HiddenCanopy\OMNI-Q_Weird_Stuff_Machine\.venv\Scripts\python.exe`
  (Python 3.12.10, torch 2.14.0+cpu, ultralytics 8.4.146, numpy 2.5.3,
  psutil 7.2.2) — the OMNI-Q repo's own venv
- Protocol: batch 1, fixed-seed (7) synthetic thermal-like inputs
  (`rng.normal` 640x640 grayscale broadcast to 3ch, N=32), 10 warmup +
  100 timed runs, wall-clock per forward. Full script wall time ~6.4 min.

## Honest labels

- **Software dequantize; no native 2-bit hardware.** After the restore every
  arm executes identical fp32 torch CPU kernels with identical FLOPs — per-arm
  latency spread below is timing noise (and co-residency when the box is
  shared). The discriminative rows are **footprint** and **Tier 2 dequant
  bandwidth**.
- **RSS** is process-level peak, dominated by the framework/runtime (~0.5 GB
  in this venv), not the 12 MB model.
- **Detection-count parity is degenerate here**: synthetic gaussian-noise
  inputs yield zero detections at conf=0.25 in every arm, so the count row is
  trivially 1.0. The informative agreement metric is **output cosine**.
- **mAP50 blocked**: HIT-UAV val set not on disk; 0.825 mAP50 / 0.538
  mAP50-95 cited as the published reference (see host baseline bundle).
- **Tier 3 is a prototype, not production.**

## Tier 1 — footprint + latency (per arm)

| Arm | Packed bytes | Ratio vs fp32 | Weight cosine (mean/min) | Output cosine vs fp32 (mean +/- std) | Latency mean/p50/p95 ms (default threads) | Latency mean/p50/p95 ms (1 thread) | Peak RSS MB |
|---|---|---|---|---|---|---|---|
| fp32 | 12,004,864 | 1.00 | — | 1.0 | 126.9 / 122.8 / 147.7 | 230.8 / 220.5 / 300.4 | 506.3 |
| mxfp4 | 1,594,397 | 0.133 | 0.992 / 0.988 | 0.999457 +/- 0.000035 | 116.5 / 112.7 / 138.5 | 186.1 / 172.1 / 249.4 | 516.5 |
| nvint2 | 938,136 | 0.078 | 0.910 / 0.827 | 0.955414 +/- 0.000076 | 136.5 / 135.6 / 150.6 | 278.2 / 265.2 / 352.5 | 519.8 |
| mxfp2 | 844,093 | 0.070 | 0.716 / 0.626 | 0.999325 +/- 0.000043 | 131.5 / 130.7 / 143.9 | 241.1 / 229.7 / 314.4 | 524.7 |

Reading:

- A resident packed master is **7.5x (mxfp4) to 14.2x (mxfp2)** smaller
  than fp32 — 7.0-13.3% of fp32 bytes; mxfp2 total is 2.25 bits/weight
  including UE8M0 block scales.
- End-to-end output agreement stays high for the power-of-two-scale arms
  (mxfp4/mxfp2, cosine > 0.999); nvint2's E4M3 block scale + per-tensor
  factor lands at 0.955 output cosine on this model — measured, no verdict.
- Latency is fp32-kernel latency in every arm; nothing here claims a 2-bit
  speedup (there is no 2-bit kernel on this box).

## Tier 2 — bandwidth (weights re-dequantized right before each forward)

| Arm | default threads: restore+forward mean ms | forward only (excluded) | dequant mean ms | 1 thread: restore+forward | forward only | dequant |
|---|---|---|---|---|---|---|
| mxfp4 | 246.1 | 116.5 | 119.3 | 254.2 | 186.1 | 65.3 |
| nvint2 | 237.8 | 136.5 | 97.6 | 352.9 | 278.2 | 65.0 |
| mxfp2 | 246.4 | 131.5 | 120.1 | 346.9 | 241.1 | 91.3 |

Reading: a numpy software decode of the full 3M-param master costs ~65-120
ms per forward — same order as the forward itself. Decode-on-the-fly from a
packed resident master is plausible only with a fast native/kernel decode;
that's exactly the gap a QAIRT/Hexagon or ternary-kernel path would close.

## Tier 3 — ternary multiply-free Linear prototype (mxfp2, nn.Linear only)

Case `[512,1024] x [1024,1024]`: y = per-block-scale einsum over the
`{-1,0,+1}` sign payload.

- Correctness vs fp32 matmul on the **same dequantized ternary weights**:
  cosine 1.0 (max abs diff 3.1e-05, fp32 accumulation order only).
- Vs the **pre-quantization fp32 weight**: cosine 0.612, max abs diff 134.96 —
  that gap is the mxfp2 weight quantization itself (weight cosine 0.613), not
  the multiply-free path.
- Rough timing: prototype einsum 48.6 ms vs torch fp32 matmul 6.3 ms.
- **Prototype, not production** — pure-torch emulation of the add-only
  payload; no claim about real ternary hardware.

## Numeric drift vs the ISR-venv run (superseded)

The deterministic codec outputs (weight cosines, output cosines, packed byte
counts, detection counts) are **identical** across the two runs. Latency and
RSS differ by interpreter/stack and run-to-run noise: fp32 default-threads
mean 126.9 ms here vs 120.6 ms (ISR venv); fp32 single-thread 230.8 vs
187.9 ms; Tier 2 dequant 65-120 ms vs 89-134 ms; RSS ~0.52 GB vs ~0.72 GB
(smaller runtime footprint in the OMNI-Q venv). No conclusion in this bundle
depends on those deltas.

## Omni-reasoner reuse note

The same tiers apply verbatim to the Omni reasoner class
(`OmniMorphableForCausalLM` Linear stacks): a ~6B-parameter resident FP32
master is ~24 GB — it does not fit a normal PC. Packed 2-bit, the same
weights are ~1.7 GB resident, which does. This YOLO run is the small,
fast measurement proxy for that claim; the codec
(`integrations/qualcomm/lowbit`) is weight-format-generic and takes any
numpy weight tensor.

## Files + re-run

- `results.json` — full machine-readable receipt (stats, environment, labels).
- Harness: `integrations/qualcomm/scripts/eval_yolo_2bit_cpu.py`
  (default model path: in-repo `models/hituav_yolov8n_finetuned_2026-08-26.pt`)
- Codec + provenance: `integrations/qualcomm/lowbit/` (+ `vendor/SOURCE.md`)
- Unit tests: `tests/test_lowbit.py` (43 tests; parity vs IDA-TRAIN-V2 source
  when `IDA_TRAIN_V2_ROOT` is set)

```bash
./.venv/Scripts/python.exe integrations/qualcomm/scripts/eval_yolo_2bit_cpu.py
```
