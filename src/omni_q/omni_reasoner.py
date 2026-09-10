"""Reasoner backends — the bridge between the IDA Omni body and Omni Q planning.

A *reasoner* turns a rendered prompt into text advice. Two backends:

- :class:`MockReasoner` — deterministic stand-in so the whole Omni Q loop is
  testable while the IDA Omni body is still in pretrain.
- :class:`OmniReferenceReasoner` — identity-gated loader around the
  ``OmniInference`` harness owned by Ask_IDA_CLI, which itself loads the
  ``OmniMorphableForCausalLM`` reference body owned by IDA-TRAIN-V2. Nothing
  here reimplements either: this module is a thin, truthful adapter.

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


class OmniReferenceReasoner:
    """Identity-gated IDA Omni reference inference.

    Loads through :func:`ask_ida_cli.omni_inference.load_omni_reference`
    (checkpoint digest, architecture hash, execution revision, FP32
    precision, and completed-example boundary are all verified by that
    loader — a mismatched artifact is refused by name) and generates through
    the session-isolated, serial-turn-publication ``OmniInference`` harness.
    Generation is greedy fixed-boundary recompute; answer tokens are V2
    private continuation, so committed evidence history is never mutated by
    decoding.

    Dependencies (``ask_ida_cli``, ``ida_train``, ``torch``,
    ``transformers``) are imported lazily so Omni Q stays dependency-free
    unless this backend is actually configured. Repository roots default to
    ``ASK_IDA_CLI_ROOT`` / ``IDA_TRAIN_V2_ROOT`` env vars.
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
        self.ask_ida_cli_root = Path(ask_ida_cli_root or os.environ.get(
            "ASK_IDA_CLI_ROOT", "E:/HiddenCanopy/Ask_IDA_CLI"))
        self.ida_train_root = Path(ida_train_root or os.environ.get(
            "IDA_TRAIN_V2_ROOT", "E:/HiddenCanopy/IDA-TRAIN-V2"))
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else (
            self.ida_train_root / "artifacts" / "tokenizers" / "omni_prism_bpe_256k")
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._inference: Any = None
        self._tokenizer: Any = None
        self._artifact_identity: dict[str, Any] = {}

    @property
    def artifact_identity(self) -> dict[str, Any]:
        return dict(self._artifact_identity)

    def _ensure_on_path(self, root: Path) -> None:
        root_s = str(root.resolve())
        if root_s not in sys.path:
            sys.path.insert(0, root_s)

    def _load(self) -> None:
        from transformers import AutoTokenizer  # lazy: heavy dep, opt-in backend

        self._ensure_on_path(self.ask_ida_cli_root)
        self._ensure_on_path(self.ida_train_root / "src")
        from ask_ida_cli.omni_inference import (  # noqa: PLC0415
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
