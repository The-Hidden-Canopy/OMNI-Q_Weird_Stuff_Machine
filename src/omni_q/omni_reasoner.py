"""Reasoner backends — the bridge between an optional model runtime and planning.

A *reasoner* turns a rendered prompt into text advice. Two backends:

- :class:`MockReasoner` — deterministic stand-in so the whole Omni Q loop is
  testable while the IDA Omni body is still in pretrain.
- :class:`OmniReferenceReasoner` — identity-gated loader around an optional
  operator-supplied ``omni_reference`` runtime. The runtime is deliberately
  not shipped in this public repository.

Truth-in-labeling: an unavailable backend raises
:class:`ReasonerUnavailable`; it never silently degrades to something else.
The planner layer records which backend produced every decision.

The public package owns the adapter contract and fail-closed admission. A
private deployment may provide the runtime through the documented local binding
without adding that runtime to this repository.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class ReasonerUnavailable(RuntimeError):
    """The configured reasoner backend cannot serve requests right now."""


def _last_position_head(weight):
    """Vocabulary projection restricted to the position generation actually reads.

    The reference model projects its whole hidden sequence through the
    256k-row tied head on *every* decode step: ``lm_head(forecast_hidden[:,
    -end:])`` is ``[1, evidence_len, 256000]`` -- 341 MB in FP32 at a
    333-token planner prompt -- and each subsequent step repeats it one
    token longer. ``OmniInference.generate`` reads exactly one row of that
    tensor (``output.logits[0, -1]``); every other row is computed,
    allocated, and discarded.

    On a 6 GiB card the cost is not the 341 MB itself but the allocator
    fragmentation behind it: measured on an RTX 4050, generation reserved
    **12.29 GiB** (against 6.14 GiB of VRAM) and died of OOM after ~16
    tokens, because each step requests a differently-shaped multi-hundred-MB
    block that no previous block can satisfy. Projecting only the final
    position instead: **1.11 GiB reserved, 1.10 GiB peak, 48 tokens clean,
    and 0.76 s/token versus 1.08 s/token** -- the discarded rows were also
    pure compute.

    Substituted onto the loaded model rather than patched into the operator
    runtime. The
    contract that matters is preserved exactly: ``logits[0, -1]`` is the same
    vector it always was, because only rows ``generate`` never indexes are
    the ones now missing. Positions other than the last are NOT available
    through this head -- training reads many positions and must not use it
    (the training-side path keeps its own head).

    This is the same defect class as the native trainer's ``[T, vocab]``
    logits buffer from the reference runtime: a full-vocabulary surface
    materialized where a single row was wanted.

    Built here rather than at module scope because ``torch`` is a lazy
    dependency of this backend, and it must be a real ``nn.Module``: the
    reference model registers ``lm_head`` as a submodule, so plain objects
    are refused by ``nn.Module.__setattr__``. Keeping the tied weight as this
    module's own parameter mirrors the tying the model already performs.
    """
    import torch.nn as nn  # lazy: torch is an opt-in dependency
    import torch.nn.functional as F

    class _LastPositionHead(nn.Module):
        def __init__(self, tied_weight) -> None:
            super().__init__()
            self.weight = tied_weight

        def forward(self, hidden):
            return F.linear(hidden[:, -1:], self.weight)

    return _LastPositionHead(weight)


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


RUNTIME_ROOT_ENV_VAR = "OMNIQ_REASONER_RUNTIME_ROOT"
TOKENIZER_ENV_VAR = "OMNIQ_REASONER_TOKENIZER"


class OmniReferenceReasoner:
    """Identity-gated inference through an operator-supplied runtime.

    The runtime must be supplied through ``runtime_root`` or
    ``OMNIQ_REASONER_RUNTIME_ROOT``. It must expose the bounded
    ``omni_reference.omni_inference`` loader and its own artifact checks. A
    missing or malformed runtime is rejected; the public package never falls
    back to a model, checkpoint, tokenizer, or deployment path of its own.

    Dependencies (``torch``, ``transformers``) are imported lazily so Omni Q
    stays dependency-free unless this backend is actually configured. The
    ``tokenizer_path`` is explicit or resolves to ``<runtime_root>/tokenizer``;
    it is never inferred from a public checkout or a receipt path.
    """

    backend = "omni_torch_reference"

    def __init__(
        self,
        checkpoint: "str | Path",
        receipt: "str | Path",
        *,
        runtime_root: "str | Path | None" = None,
        tokenizer_path: "str | Path | None" = None,
        device: str = "cpu",
        max_new_tokens: int = 64,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.receipt = Path(receipt)
        configured_root = runtime_root or os.environ.get(RUNTIME_ROOT_ENV_VAR)
        self.runtime_root = Path(configured_root).expanduser() if configured_root else None
        configured_tokenizer = tokenizer_path or os.environ.get(TOKENIZER_ENV_VAR)
        self.tokenizer_path = Path(configured_tokenizer).expanduser() if configured_tokenizer else (
            self.runtime_root / "tokenizer" if self.runtime_root else None
        )
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
        if self.runtime_root is None:
            raise ReasonerUnavailable(
                f"operator reasoner runtime is not configured; set {RUNTIME_ROOT_ENV_VAR}"
            )
        if not self.runtime_root.is_dir():
            raise ReasonerUnavailable(f"reasoner runtime root is not a directory: {self.runtime_root}")
        if self.tokenizer_path is None or not self.tokenizer_path.exists():
            raise ReasonerUnavailable(
                f"reasoner tokenizer is not configured or does not exist: {self.tokenizer_path}"
            )

        from transformers import AutoTokenizer  # lazy: heavy dep, opt-in backend

        self._ensure_on_path(self.runtime_root)
        from omni_reference.omni_inference import (  # noqa: PLC0415
            OmniInference,
            load_omni_reference,
        )

        model = load_omni_reference(self.checkpoint, self.receipt,
                                    device=self.device)
        # Identity/digest verification above is complete before this point;
        # swapping the head afterwards cannot affect what was validated or
        # loaded (see _LastPositionHead for the measurement that motivates it).
        model.lm_head = _last_position_head(model.get_input_embeddings().weight)
        self._inference = OmniInference(model)
        self._artifact_identity = dict(getattr(model, "ida_artifact_identity", {}))
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))

    #: OmniPlanner passes the world's legal ids when this is true (see
    #: OmniPlanner._reason); the fence is on unless OMNIQ_OMNI_FENCED=0.
    accepts_vocabulary = True

    def _generate_fenced(self, session_id: str, ids: list[int], max_new: int, vocabulary, eos_id: int) -> list[int]:
        """Greedy decode with identifier slots fenced to ``vocabulary``
        (omni_q.plan_grammar). Same forward call as the runtime generate,
        same head; the only difference is the mask over the next token.
        Measured on r1 (reasoner/README.md): acceptance 0.12 -> 0.46,
        usable plans 0.33 -> 0.67, with no retraining."""
        import torch  # lazy
        from .plan_grammar import PlanGrammarConstraint

        model = self._inference.model
        device = model.get_input_embeddings().weight.device
        fence = PlanGrammarConstraint(vocabulary, lambda s: self._tokenizer(s, add_special_tokens=False)["input_ids"])
        sequence = list(ids)
        out: list[int] = []
        with torch.inference_mode():
            for _ in range(max_new):
                output = model(
                    input_ids=torch.tensor([sequence], device=device),
                    omni_prediction_boundary=torch.tensor([len(ids)], device=device),
                    omni_stream_ids=(session_id,), omni_step_index=torch.tensor([0], device=device),
                )
                logits = output.logits[0, -1].float()
                allowed = fence.allowed_tokens()
                if allowed:
                    mask = torch.full_like(logits, float("-inf"))
                    index = torch.tensor(sorted(allowed), device=logits.device)
                    mask[index] = logits[index]
                    logits = mask
                token = int(logits.argmax())
                out.append(token)
                sequence.append(token)
                fence.accept(token, self._tokenizer.decode([token]))
                if token == eos_id:
                    break
        return out

    def reason(self, session_id: str, prompt: str, *,
               max_new_tokens: int | None = None, vocabulary=None) -> ReasonerResult:
        if self._inference is None:
            try:
                self._load()
            except Exception as exc:  # deps missing, identity mismatch, ...
                raise ReasonerUnavailable(
                    f"external reasoner unavailable: {exc}") from exc
        ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ReasonerUnavailable("prompt produced no evidence tokens")
        fenced = vocabulary is not None and os.environ.get("OMNIQ_OMNI_FENCED", "1") not in {"", "0", "false", "no"}
        if fenced:
            # the tokenizer carries no eos id; training appended EOS_ID = 2
            eos_id = int(self._tokenizer.eos_token_id) if self._tokenizer.eos_token_id is not None else 2
            token_ids = self._generate_fenced(session_id, [int(t) for t in ids], max_new_tokens or self.max_new_tokens,
                                              vocabulary, eos_id)
            backend = self.backend + "+fenced"
        else:
            reply = self._inference.generate(
                session_id, [int(t) for t in ids],
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                eos_token_id=self._tokenizer.eos_token_id,
            )
            token_ids = list(reply.token_ids)
            backend = self.backend
        text = self._tokenizer.decode(list(token_ids), skip_special_tokens=True)
        return ReasonerResult(
            text=text,
            token_ids=tuple(int(t) for t in token_ids),
            backend=backend,
            prompt_tokens=len(ids),
            artifact_identity=dict(self._artifact_identity),
        )
