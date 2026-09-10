# Omni 2-bit W-only quantization harness — Phase 2

Run: 2026-09-10T20:53:23Z by `integrations/intel/scripts/eval_omni_quant_reasoning.py`.

> **format simulation, numpy dequantize — no hardware 2-bit execution exists.** a format is a format — both quantized arms are reported symmetrically with measurements only; no quality verdict is assigned by this harness.

## Artifact

- Checkpoint: `E:\HiddenCanopy\IDA-TRAIN-V2\artifacts\omni_cuda_audit_p5200_20260908\checkpoint-000001.pt`
- Receipt: `E:\HiddenCanopy\IDA-TRAIN-V2\artifacts\omni_cuda_audit_p5200_20260908\receipt.json`
- Model class: `OmniMorphableForCausalLM`
- Interpreter: `E:\HiddenCanopy\IDA-TRAIN-V2\.venv\Scripts\python.exe` (torch 2.14.0+cpu, transformers 5.16.1, numpy 2.4.6)
- Checkpoint sha256 (verified by the identity-gated loader): `2114fc078d4ecc6e9985a38f811fc41f3990c01a39cb68bff27e2197072c2cbd`

**Artifact status:** plumbing validation on numerical fixture, not a capability result.

_receipt.json: promotion_status=not_promoted, training_status=numerical_run_completed, capability_status=not_established — checkpoint is an unverified numerical fixture, NOT a promoted/trained body_

Quality verdicts are **out of scope** for this harness: no accuracy or
capability claim is made from these generations. If the checkpoint is
a random-init or unverified body, these numbers describe plumbing only.

## Protocol

- W-only RNE per quantized arm: encode_tensor -> decode_tensor through ida_train.native.mxfp2_codec, dequantized fp32 weight loaded in place; fp32 arm loads the checkpoint unmodified.
- Fresh model load per arm; fresh session per repeat; N=3 repeats/arm; max_new_tokens=32; device=cpu.
- Prompt: OmniPlanner._render_prompt, goal "tidy the workspace", MockWorld.sample().state() (sha256 11c74dd178b63b95…).
- **Tied-weight note:** OmniMorphableForCausalLM ties lm_head.weight to embed_tokens.weight; the lm_head Linear pass therefore also dequantizes the shared embedding tensor.

## Formats (parameters copied from ida_train.native.mxfp2_codec)

- **mxfp2_rne** — `MXFP2_INT2_UE8M0_K32`, block K32, payload levels [0.0, 1.0], UE8M0 power-of-two byte per block.
- **nvint2_rne** — `NVINT2_INT2_E4M3_K16`, block K16, payload levels [-1.5, -0.5, 0.5, 1.5], E4M3 byte per block + FP32 per-tensor second-level scale.

## Results (per arm)

| arm | tokens per repeat | wall mean (s) | wall min (s) | wall max (s) |
|---|---|---|---|---|
| fp32 | [32, 32, 32] | 27.081 | 24.422 | 28.436 |
| mxfp2_rne | [32, 32, 32] | 25.400 | 23.709 | 27.211 |
| nvint2_rne | [32, 32, 32] | 30.679 | 24.420 | 35.169 |

Generated token ids and decoded text per repeat are in `results.json`.
Timing is per generation packet (one prompt, up to 32 greedy tokens).

Files: `results.json` (full per-arm metrics, honest labels, format
parameters, artifact identity).

_Portions derived from *The Hidden Canopy LLC* — [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission._
