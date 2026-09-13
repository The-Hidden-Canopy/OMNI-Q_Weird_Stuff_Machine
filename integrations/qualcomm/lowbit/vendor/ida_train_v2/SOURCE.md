# Vendored source — IDA-TRAIN-V2 MXFP8 CUDA futures

Future torch-extension sources, vendored so the OMNI-Q side has **no
dependency on a live IDA-TRAIN-V2 checkout** (per project owner instruction).
Same vendoring pattern as `../` (numpy codecs) and
`integrations/intel/assets/menagerie_so_arm100/`.

**Not built here.** These files are staged for a future CUDA torch extension
(`rne_reproject` at torch level); this machine has no CUDA toolchain. Do not
compile them in CI.

| File | Source | Date vendored | What changed |
|---|---|---|---|
| `fp8_e4m3.hpp` | `IDA-TRAIN-V2/native/include/ida_native/fp8_e4m3.hpp` | 2026-09-11 | attribution header only |
| `mxfp8.cuh` | `IDA-TRAIN-V2/native/include/ida_native/mxfp8.cuh` | 2026-09-11 | attribution header only |
| `mxfp8.cu` | `IDA-TRAIN-V2/native/kernels/mxfp8.cu` | 2026-09-11 | attribution header only |

- **Parity note:** the numpy fast path `rne_reproject_tensor`
  (`perception/package_quant_weights.py`) is already bit-exact against the
  OMNI-Q codec round-trip and is the parity oracle for the future extension:
  `k_lion_mxfp8` (sr_seed=0) and `fp8_e4m3::pack` must reproduce its outputs
  element-for-element before the extension can replace it. Scale constants
  match (block 32, E4M3 max 448, UE8M0 + 127 bias); note OMNI-Q's codec floors
  `safe_scale` at 1e-30, which binds for UE8M0 codes < 28, not just code 0 —
  the CUDA parity harness must cover floored-scale blocks.
- **License / governance:** source is The Hidden Canopy LLC proprietary; usage
  per `E:/HiddenCanopy/AGENT.md` (attribution + permission). Do not upstream
  changes here without syncing intent with IDA-TRAIN-V2 — if a bug is found,
  fix it in both places and say so in the receipt.
