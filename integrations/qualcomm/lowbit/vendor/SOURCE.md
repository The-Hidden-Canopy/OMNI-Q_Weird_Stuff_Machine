# Vendored source — 2-bit weight codecs

Self-contained numerical twins, vendored so OMNI-Q has **no dependency on
IDA-TRAIN-V2** (per project owner instruction). Same vendoring pattern as
`integrations/intel/assets/menagerie_so_arm100/`.

| File | Source | Date vendored |
|---|---|---|
| `mxfp2_codec.py` | `IDA-TRAIN-V2/src/ida_train/native/mxfp2_codec.py` | 2026-09-10 |
| `mxfp_scales.py`  | `IDA-TRAIN-V2/src/ida_train/native/mxfp8_codec.py` — only the helpers `mxfp2_codec.py` imports (`FP8_E4M3_MAX`, `e4m3_pack`, `e4m3_unpack`, `encode_scale`, `safe_scale`) | 2026-09-10 |

- **What changed:** imports redirected to the local `mxfp_scales.py`; docstring
  references to sibling V2 files trimmed. **No numeric code changed** — encode
  and decode behavior is intended to be byte-for-byte identical to source.
- **Parity check:** `tests/test_lowbit.py` diffs these twins against the source
  repo's codec when `IDA_TRAIN_V2_ROOT` is set, and skips otherwise. The
  vendored variant is the dependency, the source repo is only a test oracle.
- **Numerical twin, not a storage-layout twin:** payloads are one uint8 code
  (0..3) per element here, as in the source. The 4×2-bit-per-byte storage
  packing used for on-disk artifacts lives in `../lowbit_formats.py` (OMNI-Q's
  own layer).
- **License / governance:** source is The Hidden Canopy LLC proprietary; usage
  per `E:/HiddenCanopy/AGENT.md` (attribution + permission). Do not upstream
  changes here without syncing intent with IDA-TRAIN-V2 — if a bug is found in
  the numeric behavior, fix it in both places and say so in the receipt.
