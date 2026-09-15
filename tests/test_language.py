"""Adversarial tests for the bounded multilingual voice seam."""

from __future__ import annotations

from omni_q.language import canonicalize


def test_spanish_arm_constraint_keeps_original_and_canonical_evidence() -> None:
    result = canonicalize("No uses más el brazo izquierdo", "es-MX")

    assert result.original_text == "No uses más el brazo izquierdo"
    assert result.language == "es-mx"
    assert result.canonical_text == "don't use the left arm anymore"
    assert result.confidence == 1.0
    assert result.method == "bounded_lexicon:es"
    assert result.phrase_id == "disable_izquierdo_arm"


def test_french_speed_and_addressed_phrase_are_bounded() -> None:
    result = canonicalize("Omni, plus vite!", "fr")

    assert result.canonical_text == "Omni, speed up!"
    assert result.phrase_id == "speed_up"


def test_unknown_language_does_not_invent_translation() -> None:
    result = canonicalize("Usa el brazo izquierdo", "de")

    assert result.canonical_text == result.original_text
    assert result.method == "passthrough"
    assert result.confidence == 0.0


def test_known_language_unknown_phrase_is_preserved() -> None:
    result = canonicalize("¿Dónde está la taza?", "es")

    assert result.canonical_text == result.original_text
    assert result.method == "passthrough"
    assert result.phrase_id is None

