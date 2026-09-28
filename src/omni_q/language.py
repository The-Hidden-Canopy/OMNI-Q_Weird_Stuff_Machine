"""Bounded multilingual canonicalization for the OMNI-Q voice boundary.

This is deliberately a small semantic lexicon, not a general translation
system.  It recognizes only operational phrases that already have a governed
English path in :mod:`omni_q.nlu`.  Unknown languages and unknown phrases are
returned unchanged so the caller can retain the provider evidence without
inventing an intent.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

__all__ = [
    "CanonicalUtterance",
    "SUPPORTED_LANGUAGES",
    "canonicalize",
    "normalize_language",
]

SUPPORTED_LANGUAGES = frozenset({"en", "es", "fr"})
_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_language(language: str) -> str:
    """Return a stable provider language tag without claiming translation."""
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be a non-empty string")
    return language.strip().lower().replace("_", "-")


def _base_language(language: str) -> str:
    return normalize_language(language).split("-", 1)[0]


def _normalize_text(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(
        character for character in folded
        if not unicodedata.combining(character)
    )
    # Treat apostrophes as word separators for matching French and English
    # contractions ("n'utilise" / "don't") while leaving the original text
    # untouched in the returned evidence.
    return _SPACE_RE.sub(" ", _WORD_RE.sub(" ", without_marks)).strip()


def _strip_omni_address(normalized: str) -> tuple[str, bool]:
    if normalized == "omni":
        return "", True
    if normalized.startswith("omni "):
        return normalized[5:].strip(), True
    return normalized, False


@dataclass(frozen=True)
class CanonicalUtterance:
    """Original speech plus the bounded canonical form used by OMNI-Q."""

    original_text: str
    language: str
    canonical_text: str
    confidence: float
    method: str
    phrase_id: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "original_text": self.original_text,
            "language": self.language,
            "canonical_text": self.canonical_text,
            "confidence": self.confidence,
            "method": self.method,
            "phrase_id": self.phrase_id,
        }


@dataclass(frozen=True)
class _Phrase:
    phrase_id: str
    language: str
    source: str
    canonical: str


def _phrases() -> tuple[_Phrase, ...]:
    rows: list[_Phrase] = []

    def add(phrase_id: str, language: str, canonical: str,
            *sources: str) -> None:
        rows.extend(
            _Phrase(phrase_id, language, source, canonical)
            for source in sources
        )

    add("set_table", "en", "set the table",
        "set the table", "set table", "lay the table", "arrange the table")
    add("set_table", "es", "set the table",
        "pon la mesa", "poner la mesa", "prepara la mesa",
        "preparar la mesa", "organiza la mesa")
    add("set_table", "fr", "set the table",
        "mets la table", "mettre la table", "dresse la table",
        "dresser la table", "prepare la table", "preparer la table")

    for language, side_word in (("en", ("left", "right")),
                                ("es", ("izquierdo", "derecho")),
                                ("fr", ("gauche", "droit"))):
        for side in side_word:
            english_side = {
                "left": "left", "right": "right",
                "izquierdo": "left", "derecho": "right",
                "gauche": "left", "droit": "right",
            }[side]
            if language == "en":
                add(f"disable_{side}_arm", language,
                    f"don't use the {side} arm anymore",
                    f"don't use the {side} arm",
                    f"do not use the {side} arm",
                    f"don't use your {side} arm",
                    f"do not use your {side} arm",
                    f"stop using the {side} arm",
                    f"stop using your {side} arm",
                    f"never use the {side} arm")
                add(f"prefer_{side}_arm", language,
                    f"use the {english_side} arm",
                    f"use the {side} arm",
                    f"use your {side} arm",
                    f"prefer the {side} arm",
                    f"{side} arm only")
            elif language == "es":
                add(f"disable_{side}_arm", language,
                    f"no uses el brazo {side}",
                    f"no uses mas el brazo {side}",
                    f"deja de usar el brazo {side}",
                    f"no utilices el brazo {side}")
                add(f"prefer_{side}_arm", language,
                    f"use the {english_side} arm",
                    f"utiliza el brazo {side}",
                    f"prefiere el brazo {side}",
                    f"brazo {side} solamente")
            else:
                add(f"disable_{side}_arm", language,
                    f"n utilise pas le bras {side}",
                    f"n utilise plus le bras {side}",
                    f"cesse d utiliser le bras {side}",
                    f"arrete d utiliser le bras {side}")
                add(f"prefer_{side}_arm", language,
                    f"use the {english_side} arm",
                    f"prefere le bras {side}",
                    f"bras {side} uniquement")

    # The generated arm rows above intentionally encode the spoken side in the
    # phrase id.  Canonical text is corrected to English in
    # _canonical_for_phrase; the source phrase itself remains the auditable
    # provider-language evidence.
    add("stop", "en", "stop", "stop", "stop now", "stop moving",
        "emergency stop", "freeze")
    add("stop", "es", "stop", "stop", "detente", "para", "alto", "emergency stop",
        "emergencia", "congela")
    add("stop", "fr", "stop", "arrete", "arretez", "stop", "urgence",
        "arret", "gele")

    add("continue", "en", "continue", "continue", "resume", "carry on",
        "keep going")
    add("continue", "es", "continue", "continue", "continua", "seguir", "reanuda",
        "sigue")
    add("continue", "fr", "continue", "continue", "reprends", "poursuis",
        "reprendre")

    add("keep_local", "en", "keep everything local", "keep everything local",
        "keep inference local", "stay offline", "no cloud", "fully local")
    add("keep_local", "es", "keep everything local", "manten todo local",
        "mantén todo local", "mantén la inferencia local", "sin nube",
        "mantente sin conexion")
    add("keep_local", "fr", "keep everything local", "garde tout en local",
        "garde l inference en local", "hors ligne", "sans cloud")

    add("slow_down", "en", "slow down", "slow down", "slower", "take your time")
    add("slow_down", "es", "slow down", "mas despacio", "reduce la velocidad",
        "ve mas lento", "mas lento")
    add("slow_down", "fr", "slow down", "ralentis", "plus lentement",
        "ralentissez", "moins vite")
    add("speed_up", "en", "speed up", "speed up", "faster", "hurry")
    add("speed_up", "es", "speed up", "mas rapido", "aumenta la velocidad",
        "ve mas rapido", "mas deprisa")
    add("speed_up", "fr", "speed up", "accelere", "plus vite", "accelerez")

    # A small object/direction slice keeps the feature useful for the table
    # demo without pretending to translate arbitrary noun phrases.
    for language, rows_for_language in {
        "en": (
            ("move the plate left", "move the plate left"),
            ("move the plate right", "move the plate right"),
            ("nudge the cup left", "nudge the cup left"),
            ("nudge the cup right", "nudge the cup right"),
        ),
        "es": (
            ("mueve el plato a la izquierda", "move the plate left"),
            ("mueve el plato a la derecha", "move the plate right"),
            ("mueve la taza a la izquierda", "nudge the cup left"),
            ("mueve la taza a la derecha", "nudge the cup right"),
        ),
        "fr": (
            ("deplace l assiette a gauche", "move the plate left"),
            ("deplace l assiette a droite", "move the plate right"),
            ("pousse la tasse a gauche", "nudge the cup left"),
            ("pousse la tasse a droite", "nudge the cup right"),
        ),
    }.items():
        for source, canonical in rows_for_language:
            add("move_object", language, canonical, source)

    return tuple(rows)


_PHRASE_INDEX: dict[tuple[str, str], _Phrase] = {}
for _phrase in _phrases():
    _PHRASE_INDEX.setdefault(
        (_phrase.language, _normalize_text(_phrase.source)), _phrase
    )


def _canonical_for_phrase(phrase: _Phrase) -> str:
    side_token = phrase.phrase_id.removeprefix("disable_").removeprefix("prefer_")
    side_token = side_token.removesuffix("_arm")
    english_side = {
        "left": "left", "right": "right",
        "izquierdo": "left", "derecho": "right",
        "gauche": "left", "droit": "right",
    }.get(side_token)
    if phrase.phrase_id.startswith("prefer_") and english_side:
        return f"use the {english_side} arm"
    if not phrase.phrase_id.startswith("disable_") or not english_side:
        return phrase.canonical
    # Disable means prefer the surviving arm.  The canonical English NLU
    # understands that consequence rather than a separate disable operation.
    return f"don't use the {english_side} arm anymore"


def canonicalize(text: str, language: str = "en") -> CanonicalUtterance:
    """Canonicalize one bounded operational utterance.

    The returned ``canonical_text`` is suitable for the existing English NLU.
    A miss is explicit: the original text is retained and ``method`` is
    ``identity`` for English or ``passthrough`` for another language.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    normalized_language = normalize_language(language)
    base = _base_language(normalized_language)
    original = text.strip()
    normalized = _normalize_text(original)
    core, addressed = _strip_omni_address(normalized)
    phrase = _PHRASE_INDEX.get((base, core))

    if phrase is None:
        return CanonicalUtterance(
            original_text=original,
            language=normalized_language,
            canonical_text=original,
            confidence=1.0 if base == "en" else 0.0,
            method="identity" if base == "en" else "passthrough",
        )

    canonical = _canonical_for_phrase(phrase)
    if addressed:
        canonical = f"Omni, {canonical}"
    # Retain sentence-final punctuation so IntentAccumulator's existing
    # command-completeness rule still sees a provider-finalized sentence.
    punctuation = next(
        (character for character in reversed(original) if character in ".!?"),
        "",
    )
    if punctuation and not canonical.endswith(punctuation):
        canonical += punctuation
    return CanonicalUtterance(
        original_text=original,
        language=normalized_language,
        canonical_text=canonical,
        confidence=1.0,
        method=f"bounded_lexicon:{base}",
        phrase_id=phrase.phrase_id,
    )
