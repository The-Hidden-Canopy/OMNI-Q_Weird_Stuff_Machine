# Selected-layer packed-FP4 pilot

This experiment adapts the useful part of IDA-TRAIN-V2's Hopper recipe to the
YOLO detector without changing the active control run.

The pilot keeps MXFP8 E4M3/UE8M0 K32 payloads and BF16 Lion momentum as the
authoritative optimizer state. Selected YOLO Conv2d modules retain an E2M1
MXFP4/UE8M0 K32 payload and decode it at the convolution consumer boundary.
Gradients use a straight-through estimator and the packed FP4 payloads are
refreshed after each successful optimizer step.

The pilot is deliberately narrower than v2's Hopper WGMMA attention path:
stock PyTorch Conv2d/autocast performs the arithmetic, so this receipt must not
be described as native Hopper WGMMA, native Blackwell NVFP4 MMA, or a real
MXFP4-master Lion run. It measures whether packed FP4 consumer quantization is
tolerable for the 82-class detector while preserving a stable MXFP8 optimizer
control.

Launch from the remote queue after the current run records checkpoint 30:

```bash
bash scripts/queue_fp4_pilot_after_checkpoint30.sh
```

The default selected modules are backbone/neck stages `model.2`, `4`, `6`,
`8`, `12`, `15`, `18`, and `21`; the detection head and stem remain at the
control precision. The pilot starts from the checkpoint-30 weights with a
fresh optimizer state, so the source run remains an independent rollback and
comparison artifact.
