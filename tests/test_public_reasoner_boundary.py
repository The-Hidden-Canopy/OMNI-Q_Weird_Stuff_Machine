"""Public boundary tests for the optional reasoner adapter."""

from __future__ import annotations

import pytest

from omni_q.omni_reasoner import OmniReferenceReasoner, ReasonerUnavailable


def test_reasoner_requires_explicit_operator_runtime(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNIQ_REASONER_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("OMNIQ_REASONER_TOKENIZER", raising=False)
    reasoner = OmniReferenceReasoner(tmp_path / "model.pt", tmp_path / "receipt.json")

    with pytest.raises(ReasonerUnavailable, match="OMNIQ_REASONER_RUNTIME_ROOT"):
        reasoner.reason("boundary-test", "hello")


def test_reasoner_rejects_missing_explicit_runtime(tmp_path):
    reasoner = OmniReferenceReasoner(
        tmp_path / "model.pt",
        tmp_path / "receipt.json",
        runtime_root=tmp_path / "not-a-runtime",
        tokenizer_path=tmp_path / "missing-tokenizer",
    )

    with pytest.raises(ReasonerUnavailable, match="not a directory"):
        reasoner.reason("boundary-test", "hello")
