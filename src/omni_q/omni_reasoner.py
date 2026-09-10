"""Reasoner backends — the bridge between the IDA Omni body and Omni Q planning.

A *reasoner* turns a rendered prompt into text advice. Two backends:

- :class:`MockReasoner` — deterministic stand-in so the whole Omni Q loop is
  testable while the IDA Omni body is still in pretrain.
- :class:`OmniReferenceReasoner` — identity-gated loader around the
  ``OmniInference`` harness (vendored from Ask_IDA_CLI under
  ``integrations/intel/vendor/omni_reference/``), which itself loads the
  ``OmniMorphableForCausalLM`` reference body (vendored model closure from
  IDA-TRAIN-V2, same package). Nothing here reimplements either: this module
  is a thin, truthful adapter.

Truth-in-labeling (Ask_IDA_CLI law 3): an unavailable backend raises
:class:`ReasonerUnavailable`; it never silently degrades to something else.
The planner layer records which backend produced every decision.

Portions derived from *The Hidden Canopy LLC* —
[`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2) and
[`Ask_IDA_CLI`](https://github.com/The-Hidden-Canopy/Ask_IDA_CLI).
Used with permission.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class ReasonerUnavailable(RuntimeError):
    """The configured reasoner backend cannot serve requests right now."""


@dataclass(frozen=True)
class ReasonerResult:
    """One completed reasoner turn."""

    text: str
    token_ids: tuple[int, ...]
    backend: str                                  # "omni_torch_reference" | "mock"
    prompt_tokens: int
    artifact_identity: dict[str, Any] = field(default_factory=dict)


class Reasoner(Protocol):
    """Prompt in, advice out. Session history is backend-owned."""

    backend: str

    def reason(self, session_id: str, prompt: str, *,
               max_new_tokens: int | None = None) -> ReasonerResult: ...


class MockReasoner:
    """Deterministic canned stand-in for the pretrain window.

    Emits a rationale-only response by default, which deliberately sends
    :class:`omni_q.omni_planner.OmniPlanner` down its validated-fallback path;
    tests inject structured ``PLAN … END`` text to exercise model-step parsing
    without any weights on disk.
    """

    backend = "mock"

    def __init__(self, canned_text: str | None = None) -> None:
        self._text = canned_text or (
            "RATIONALE: mock reasoner stand-in; the deterministic planner "
            "governs the graph until the IDA Omni body promotes."
        )
        self.calls: list[tuple[str, str]] = []

    def reason(self, session_id: str, prompt: str, *,
               max_new_tokens: int | None = None) -> ReasonerResult:
        self.calls.append((session_id, prompt))
        return ReasonerResult(
            text=self._text,
            token_ids=(),
            backend=self.backend,
            prompt_tokens=len(prompt.split()),
        )


_VENDOR_PARENT = Path(__file__).resolve().parents[2] / "integrations" / "intel" / "vendor"
VENDORED_OMNI_REFERENCE = _VENDOR_PARENT / "omni_reference"
VENDORED_TOKENIZERS = VENDORED_OMNI_REFERENCE / "tokenizers"
VENDORED_TOKENIZER = VENDORED_TOKENIZERS / "omni_prism_bpe_256k"
EXTERNAL_ENV_VAR = "OMNIQ_OMNI_REFERENCE_EXTERNAL"
_EXTERNAL_TRUTHY = {"1", "true", "yes", "on"}


class OmniReferenceReasoner:
    """Identity-gated IDA Omni reference inference.

    Loads through the vendored harness
    (:func:`omni_reference.omni_inference.load_omni_reference`, see
    ``integrations/intel/vendor/omni_reference/`` — checkpoint digest,
    architecture hash, execution revision, FP32 precision, and
    completed-example boundary are all verified by that loader; a mismatched
    artifact is refused by name) and generates through the session-isolated,
    serial-turn-publication ``OmniInference`` harness. Generation is greedy
    fixed-boundary recompute; answer tokens are V2 private continuation, so
    committed evidence history is never mutated by decoding.

    Dependencies (``torch``, ``transformers``) are imported lazily so Omni Q
    stays dependency-free unless this backend is actually configured. The
    tokenizer defaults to the vendored copy under
    ``integrations/intel/vendor/omni_reference/tokenizers/``, resolved from
    the receipt's recorded ``tokenizer_path`` (explicit ``tokenizer_path``
    always wins).

    Escape hatch — **dev parity check only**: passing ``ask_ida_cli_root`` /
    ``ida_train_root`` (or setting ``OMNIQ_OMNI_REFERENCE_EXTERNAL=1`` with
    ``OMNIQ_ASK_IDA_CLI_ROOT`` / ``OMNIQ_IDA_TRAIN_V2_ROOT``) loads the
    harness and model from those external checkouts instead of the vendored
    package, purely to diff vendored-vs-source behavior. It is not the
    supported path; the vendored package is the dependency.
    """

    backend = "omni_torch_reference"

    def __init__(
        self,
        checkpoint: "str | Path",
        receipt: "str | Path",
        *,
        ask_ida_cli_root: "str | Path | None" = None,
        ida_train_root: "str | Path | None" = None,
        tokenizer_path: "str | Path | None" = None,
        device: str = "cpu",
        max_new_tokens: int = 64,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.receipt = Path(receipt)
        self.external = bool(
            ask_ida_cli_root or ida_train_root
            or os.environ.get(EXTERNAL_ENV_VAR, "").strip().lower() in _EXTERNAL_TRUTHY)
        self.ask_ida_cli_root = Path(ask_ida_cli_root or os.environ.get(
            "OMNIQ_ASK_IDA_CLI_ROOT", "E:/HiddenCanopy/Ask_IDA_CLI"))
        self.ida_train_root = Path(ida_train_root or os.environ.get(
            "OMNIQ_IDA_TRAIN_V2_ROOT", "E:/HiddenCanopy/IDA-TRAIN-V2"))
        if tokenizer_path:
            self.tokenizer_path = Path(tokenizer_path)
        elif self.external:
            self.tokenizer_path = (self.ida_train_root / "artifacts" / "tokenizers"
                                   / "omni_prism_bpe_256k")
        else:
            self.tokenizer_path = self._vendored_tokenizer_default()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._inference: Any = None
        self._tokenizer: Any = None
        self._artifact_identity: dict[str, Any] = {}

    def _vendored_tokenizer_default(self) -> Path:
        """Vendored tokenizer dir matching the checkpoint receipt's record.

        The receipt names the tokenizer the artifact was built with (e.g.
        ``ida_lattice_bpe_32k``); a mismatch against the model's vocab is
        refused by the harness as invalid evidence tokens, so the default is
        resolved from the receipt, falling back to ``omni_prism_bpe_256k``.
        """
        name = None
        try:
            import json
            meta = json.loads(self.receipt.read_text(encoding="utf-8"))
            rel = meta.get("config", {}).get("tokenizer_path")
            name = Path(rel).name if rel else None
        except Exception:
            name = None
        if name and (VENDORED_TOKENIZERS / name).is_dir():
            return VENDORED_TOKENIZERS / name
        return VENDORED_TOKENIZER

    @property
    def artifact_identity(self) -> dict[str, Any]:
        return dict(self._artifact_identity)

    def _ensure_on_path(self, root: Path) -> None:
        root_s = str(root.resolve())
        if root_s not in sys.path:
            sys.path.insert(0, root_s)

    def _load(self) -> None:
        from transformers import AutoTokenizer  # lazy: heavy dep, opt-in backend

        if self.external:
            # Dev parity check only (see class docstring): load from the
            # external Ask_IDA_CLI / IDA-TRAIN-V2 checkouts.
            self._ensure_on_path(self.ask_ida_cli_root)
            self._ensure_on_path(self.ida_train_root / "src")
            from ask_ida_cli.omni_inference import (  # noqa: PLC0415
                OmniInference,
                load_omni_reference,
            )
        else:
            self._ensure_on_path(VENDORED_OMNI_REFERENCE.parent)
            from omni_reference.omni_inference import (  # noqa: PLC0415
                OmniInference,
                load_omni_reference,
            )

        model = load_omni_reference(self.checkpoint, self.receipt,
                                    device=self.device)
        self._inference = OmniInference(model)
        self._artifact_identity = dict(getattr(model, "ida_artifact_identity", {}))
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))

    def reason(self, session_id: str, prompt: str, *,
               max_new_tokens: int | None = None) -> ReasonerResult:
        if self._inference is None:
            try:
                self._load()
            except Exception as exc:  # deps missing, identity mismatch, ...
                raise ReasonerUnavailable(
                    f"omni reference reasoner unavailable: {exc}") from exc
        ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ReasonerUnavailable("prompt produced no evidence tokens")
        reply = self._inference.generate(
            session_id, [int(t) for t in ids],
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        text = self._tokenizer.decode(list(reply.token_ids), skip_special_tokens=True)
        return ReasonerResult(
            text=text,
            token_ids=tuple(int(t) for t in reply.token_ids),
            backend=self.backend,
            prompt_tokens=len(ids),
            artifact_identity=dict(self._artifact_identity),
        )
