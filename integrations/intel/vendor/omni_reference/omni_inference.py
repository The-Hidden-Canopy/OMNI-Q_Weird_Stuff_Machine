# Portions derived from *The Hidden Canopy LLC* —
# [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2) /
# [`Ask_IDA_CLI`](https://github.com/The-Hidden-Canopy/Ask_IDA_CLI). Used with permission.
"""Opt-in Omni reference inference; no generic HF generation or MoE wrapping.

Vendored from `Ask_IDA_CLI/ask_ida_cli/omni_inference.py`. The model
implementation is owned by The Hidden Canopy LLC's IDA-TRAIN-V2 and is
vendored next to this module under `ida_lattice/` (see SOURCE.md). This
adapter publishes evidence history only after an entire turn succeeds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any


class OmniHarnessError(RuntimeError):
    """Vendored replacement for ``ask_ida_cli.errors.CLIError``.

    The source harness raised ``CLIError(message, code="...")``; this vendored
    twin maps the ``code=`` keyword to an attribute of the same name.
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class OmniReply:
    token_ids: tuple[int, ...]
    evidence_step: int
    backend: str = "torch_reference"
    decoding: str = "fixed_boundary_recompute"


def load_omni_reference(checkpoint: Path, receipt: Path, *, device: str = "cpu") -> Any:
    """Load a local numerical checkpoint with its recorded architecture identity.

    FP32 is explicit for this initial correctness path. Neither the source
    checkpoint nor the native admission contract is modified.
    """
    import torch
    from .ida_lattice.config import IDALatticeConfig
    from .ida_lattice.omni_state_model import (
        OMNI_EXECUTION_REVISION,
        OmniMorphableForCausalLM,
    )

    metadata = json.loads(Path(receipt).read_text(encoding="utf-8"))
    architecture = metadata["config"]["architecture"]
    if architecture.get("omni_architecture") != "state_coupled_v1":
        raise OmniHarnessError("Expected state_coupled_v1", code="omni_identity_mismatch")
    with Path(checkpoint).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != metadata.get("checkpoint_sha256"):
            raise OmniHarnessError("Omni checkpoint digest mismatch", code="omni_identity_mismatch")
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
    identity = saved["identity"]
    architecture_hash = hashlib.sha256(
        json.dumps(architecture, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (identity.get("architecture_hash") != architecture_hash
            or identity.get("execution_revision") != OMNI_EXECUTION_REVISION
            or identity.get("precision") != "torch_reference_fp32"
            or identity.get("checkpoint_revision") != 2
            or saved.get("state_boundary") != "completed_example"
            or saved.get("carried_history") is not None):
        raise OmniHarnessError("Omni execution/checkpoint identity mismatch", code="omni_identity_mismatch")
    # A reference scan is an explicit execution choice, separate from the
    # artifact's original scan backend, and is reported on the returned model.
    config = IDALatticeConfig(**{**architecture, "omni_scan_backend": "reference"})
    model = OmniMorphableForCausalLM(config)
    model.load_state_dict(saved["model"], strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    model.requires_grad_(False)
    model.ida_artifact_identity = {**identity, "checkpoint_sha256": digest,
                                   "inference_scan_backend": "reference"}
    return model


class OmniInference:
    """One model owner, isolated conversation histories, serial turn publication."""

    def __init__(self, model: Any) -> None:
        if model.config.omni_architecture != "state_coupled_v1":
            raise OmniHarnessError("Expected Omni coupled model", code="omni_identity_mismatch")
        self.model = model.eval()
        self._sessions: dict[str, tuple[int, tuple[Any, ...]]] = {}
        self._lock = RLock()

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def generate(self, session_id: str, evidence_ids: list[int], *,
                 max_new_tokens: int = 32, eos_token_id: int | None = None,
                 cancelled=None) -> OmniReply:
        """Greedy token harness. Call with NEW evidence, never transcript replay.

        Recompute from the same pre-turn snapshot with a fixed evidence boundary
        on every decode step. V2 handles answer tokens as private continuation.
        This is a correctness baseline; an incremental adapter comes later.
        """
        import torch

        if not session_id.strip() or not evidence_ids or max_new_tokens < 1:
            raise OmniHarnessError("Nonempty session/evidence and a positive token limit are required",
                                   code="omni_invalid_request")
        if len(evidence_ids) + max_new_tokens > self.model.config.max_position_embeddings:
            raise OmniHarnessError("Omni turn exceeds position capacity", code="omni_context_full")
        if any(type(t) is not int or not 0 <= t < self.model.config.vocab_size for t in evidence_ids):
            raise OmniHarnessError("Invalid evidence token", code="omni_invalid_request")
        with self._lock, torch.inference_mode():
            step, previous = self._sessions.get(session_id, (0, None))
            # Copy the snapshot so even an accidental in-place backend write
            # cannot corrupt already published history on a failed turn.
            def snapshot(states):
                return tuple(replace(s, **{k: v.detach().clone() for k, v in vars(s).items()
                                           if isinstance(v, torch.Tensor)}) for s in states)

            device = self.model.get_input_embeddings().weight.device
            sequence = list(evidence_ids)
            generated = []
            candidate = None
            for _ in range(max_new_tokens):
                if cancelled is not None and cancelled():
                    raise OmniHarnessError("Omni generation cancelled", code="generation_cancelled")
                output = self.model(
                    input_ids=torch.tensor([sequence], device=device),
                    omni_prediction_boundary=torch.tensor([len(evidence_ids)], device=device),
                    omni_stream_ids=(session_id,), omni_step_index=torch.tensor([step], device=device),
                    omni_layer_states=None if previous is None else snapshot(previous),
                )
                logits = output.logits[0, -1].float()
                if not bool(torch.isfinite(logits).all()):
                    raise OmniHarnessError("Non-finite Omni logits", code="omni_invalid_output")
                candidate = output.omni_layer_states
                for state in candidate:
                    if (state.stream_ids != (session_id,)
                            or not bool((state.committed_position == step).all())
                            or any(not bool(torch.isfinite(v).all()) for v in vars(state).values()
                                   if isinstance(v, torch.Tensor) and v.is_floating_point())):
                        raise OmniHarnessError("Invalid Omni state candidate", code="omni_invalid_output")
                if len(candidate) != self.model.config.num_hidden_layers:
                    raise OmniHarnessError("Incomplete Omni state candidate", code="omni_invalid_output")
                token = int(logits.argmax())
                generated.append(token)
                sequence.append(token)
                if token == eos_token_id:
                    break
            if cancelled is not None and cancelled():
                raise OmniHarnessError("Omni generation cancelled", code="generation_cancelled")
            self._sessions[session_id] = (step + 1, snapshot(candidate))
            return OmniReply(tuple(generated), step)
