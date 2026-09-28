from __future__ import annotations

from omni_q.behavior_api import BehaviorAdvice, BehaviorEvidence, EvidenceSource


def test_evidence_serializes_bounded_public_contract() -> None:
    evidence = BehaviorEvidence(
        sources=(EvidenceSource("camera.front", 0.9, freshness=0.8, continuity=0.7, provenance=1.0),),
        contact_confidence=0.85,
        target_stable=True,
        receiver_ready=True,
    )
    payload = evidence.as_dict()
    assert payload["sources"][0]["source_id"] == "camera.front"
    assert payload["contact_confidence"] == 0.85
    assert payload["target_stable"] is True
    assert payload["receiver_ready"] is True


def test_advice_parses_only_public_disposition_and_hints() -> None:
    advice = BehaviorAdvice.from_mapping(
        {
            "request_id": "r1",
            "disposition": "HOLD_GIVER",
            "reason": "receiver not ready",
            "hints": {
                "retry": True,
                "reobserve": False,
                "switch_provider": False,
                "preserve_grip": True,
            },
            "confidence": {"aggregate": 0.81, "disagreement": 0.04},
        }
    )
    assert advice.request_id == "r1"
    assert advice.disposition == "HOLD_GIVER"
    assert advice.retry is True
    assert advice.preserve_grip is True
    assert advice.confidence["aggregate"] == 0.81
