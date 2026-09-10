# Vendored source — Omni reference inference harness + model closure

Self-contained vendor of the IDA Omni inference harness and the model
implementation it loads, so OMNI-Q has **no dependency on Ask_IDA_CLI or
IDA-TRAIN-V2** (house rule: outside repos are read-only oracles; everything
needed is vendored). Same vendoring pattern as
`integrations/qualcomm/lowbit/vendor/` and
`integrations/intel/assets/menagerie_so_arm100/`.

## Files

| File | Source | Date vendored |
|---|---|---|
| `omni_inference.py` | `Ask_IDA_CLI/ask_ida_cli/omni_inference.py` (`OmniReply`, `load_omni_reference`, `OmniInference`) | 2026-09-10 |
| `ida_lattice/omni_state_model.py` | `IDA-TRAIN-V2/src/ida_train/models/ida_lattice/omni_state_model.py` (`OMNI_EXECUTION_REVISION`, `OmniMorphableForCausalLM`) | 2026-09-10 |
| `ida_lattice/omni_state.py` | same dir — `OmniEvidenceBatch`, `OmniLayerState`, `StateExpert`, `MorphableStateLayer`, `masked_mean`, `candidate_scan` | 2026-09-10 |
| `ida_lattice/omni.py` | same dir — `OmniEvidenceBridge`, `OmniStateHead`, `compute_omni_losses` | 2026-09-10 |
| `ida_lattice/omni_coupling.py` | same dir — `CoupledOperationControl`, `coupled_scan` (needed at runtime for `state_coupled_v1`, imported lazily from `omni_state.py`) | 2026-09-10 |
| `ida_lattice/config.py` | same dir — `IDALatticeConfig` (+ `DEFAULT_COGNITIVE_ROUTES`, `SEAT_COGNITIVE_BIAS`, vendored whole) | 2026-09-10 |
| `ida_lattice/constitutional_router.py` | same dir — import-time dep of `omni_state.py` | 2026-09-10 |
| `ida_lattice/local_workspace.py` | same dir — import-time dep of `omni_state.py` | 2026-09-10 |
| `ida_lattice/projections.py` | same dir — import-time dep of `local_workspace.py` | 2026-09-10 |
| `ida_lattice/fp8_linear.py` | same dir — import-time dep of `projections.py` (`Fp8Linear`; unused on the fp32 reference path but required to import `projections`) | 2026-09-10 |
| `ida_lattice/multiscale_memory.py` | same dir — import-time dep of `omni_state.py` | 2026-09-10 |
| `ida_lattice/tensor_contracts.py` | same dir — import-time dep of `multiscale_memory.py` | 2026-09-10 |
| `ida_lattice/__init__.py` | **new, replaces source `__init__.py`** — see trims below | 2026-09-10 |
| `tokenizers/omni_prism_bpe_256k/` | `IDA-TRAIN-V2/artifacts/tokenizers/omni_prism_bpe_256k/` (`tokenizer.json` ~18 MB, `tokenizer_config.json`, `omni_tokenizer_manifest.json`, `README.md`) — byte-identical copies | 2026-09-10 |
| `tokenizers/ida_lattice_bpe_32k/` | `IDA-TRAIN-V2/artifacts/tokenizers/ida_lattice_bpe_32k/` (`tokenizer.json` ~2.3 MB, `tokenizer_config.json`, `README.md`) — byte-identical copy; the `.cache/` HF download cache and `.gitattributes` from the source dir were **not** vendored | 2026-09-10 |

Both tokenizer dirs are vendored because the right tokenizer is a property of
the checkpoint: the fixture receipt (`artifacts/omni_cuda_audit_p5200_20260908/`)
records `tokenizer_path: artifacts/tokenizers/ida_lattice_bpe_32k` (vocab
32000) — feeding it the 256k tokenizer fails the harness's vocab check.
`OmniReferenceReasoner` therefore resolves its vendored default from the
receipt's recorded `tokenizer_path` basename, falling back to
`omni_prism_bpe_256k`.

All eleven `ida_lattice` modules are **byte-identical** to source (verified by
`cmp` after vendoring); the harness body is unchanged apart from the rewires
below.

## What changed (rewires)

- **`omni_inference.py`**
  - `from .errors import CLIError` → vendored `class OmniHarnessError(RuntimeError)`
    defined in this module; the `code="..."` keyword usage maps to an
    `OmniHarnessError.code` attribute. All `raise CLIError(...)` sites renamed
    1:1, no other edit.
  - `from ida_train.models.ida_lattice.{config,omni_state_model} import ...`
    → `from .ida_lattice.{config,omni_state_model} import ...` (relative, so
    the package is self-contained on any `sys.path`).
  - Docstring updated to point at the vendored model instead of instructing to
    install IDA-TRAIN-V2.
- **`ida_lattice/__init__.py` — trimmed.** The source `__init__` re-exported
  the full lattice family (`IDALatticeForCausalLM`, governed memory, thalamic
  router, temporal memory/trace emitters, prefrontal workspace, ...). None of
  those modules are vendored; the replacement `__init__` performs **no eager
  re-exports** (consumers import submodules directly, as the harness does).

## Trims (deliberately NOT vendored)

- **Full lattice family tree** — everything the source `ida_lattice/__init__.py`
  re-exported beyond the closure above (`model.py`, `governed_memory.py`,
  `thalamic_router.py`, `temporal_memory.py`, `temporal_trace_emitter.py`,
  `prefrontal_workspace.py`, `family_state_head.py`, `future_token_head.py`,
  `action_gate.py`, `controlled_merge.py`, `control_policy.py`,
  `control_summary.py`, `development_policy.py`, `memory_gate.py`,
  `pressure_field.py`, `gh_head.py`, `student_experts.py`,
  `low_rank_state_supersampler.py`, `lrss_vocab.py`, `recurrent_state.py`,
  `omni_native_scan.py`). Not reachable from `omni_state_model`'s import graph.
- **`omni_native_scan.py`** — runtime-lazy import in `omni_state.py` /
  `omni_coupling.py`, executed only when `omni_scan_backend` is
  `native_cuda`/`native_cuda_packed` (ctypes native library loader). The
  harness always constructs the config with `omni_scan_backend="reference"`,
  so the pure-torch scan path is used and the native scan module is never
  imported. Choosing a native backend with this vendor package raises
  `ModuleNotFoundError` by design — the native engine is out of scope here.
- **No `ida_train.*` imports needed inlining.** The closure was checked for
  imports of utilities outside `models/ida_lattice` (telemetry, native
  codecs, orchestration): there are none — every vendored module imports only
  torch / transformers / stdlib and sibling `ida_lattice` modules.

## Consumers

- `src/omni_q/omni_reasoner.py` (`OmniReferenceReasoner`) loads through this
  package by default. The cross-repo path survives only as an explicit
  opt-in escape hatch (`OMNIQ_OMNI_REFERENCE_EXTERNAL=1`, with
  `OMNIQ_ASK_IDA_CLI_ROOT` / `OMNIQ_IDA_TRAIN_V2_ROOT` roots), documented as
  **dev parity check only**.
- `integrations/intel/scripts/eval_omni_quant_reasoning.py` — same default.
- Checkpoints / receipts remain artifacts: they are passed as paths (env
  args), never imported as code.

## Parity / oracle

- `transformers` / `torch` stay third-party pip dependencies of the consumer
  environment (OMNI-Q's own venv or the IDA-TRAIN-V2 venv).
- The fixture end-to-end run (`evidence/benchmark_results/omni_quant_2bit_*`)
  exercises this vendored package: identity-gated load, greedy generation,
  2-bit W-only quantization.

## License / governance

Source is The Hidden Canopy LLC proprietary (all rights reserved — no LICENSE
file in either repo); usage per `E:/HiddenCanopy/AGENT.md` (attribution +
permission, no removal of authorship/notices — the attribution header is in
every vendored file). Do not upstream changes here without syncing intent
with Ask_IDA_CLI / IDA-TRAIN-V2; if a bug is found in the harness or model
numerics, fix it in both places and say so in the receipt.
