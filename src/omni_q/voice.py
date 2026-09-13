"""Provider-neutral multi-speaker voice boundary.

Speechmatics is intentionally reduced to an attributed, timestamped source of
speech observations.  This module owns the session-scoped state around those
observations: speaker profiles, adaptive endpointing, conversation arbitration,
reference claims, and authority decisions.

The boundary is deliberately conservative:

* partial speech can prepare context, but cannot execute or mutate;
* final speech becomes a claim, not an authoritative fact;
* identity and physical grounding remain claims with explicit confidence;
* only an explicitly authorized final intent may call ``RuntimeMutator`` or an
  injected OMNI handler;
* interruption detection requests a local reflex, but does not mutate the
  authoritative world itself.

No Speechmatics SDK is imported here.  ``SpeechmaticsRealtimeAdapter`` is a
small payload translator so the transport can be replaced without making the
rest of OMNI-Q depend on one vendor.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from .contracts import WorldState
from .events import EventBus
from . import nlu

__all__ = [
    "AuthorityDecision",
    "AuthorityResolver",
    "ClaimStatus",
    "ConversationArbiter",
    "ConversationMode",
    "InterruptionDecision",
    "IntentAccumulator",
    "InterruptionGate",
    "IntentCandidate",
    "ReferenceClaim",
    "ReferenceResolver",
    "SpeakerRegistry",
    "SpeakerState",
    "SpeechClaim",
    "SpeechFinal",
    "SpeechPartial",
    "SpeechmaticsRealtimeAdapter",
    "TurnDecision",
    "VoiceCapability",
    "VoiceDispatchResult",
    "VoiceError",
    "VoiceScopeError",
    "VoiceRuntime",
]


class VoiceError(ValueError):
    """Invalid or unsafe voice-boundary input."""


class VoiceScopeError(VoiceError):
    """A speech event targeted a different session or organization."""


class ClaimStatus(str, Enum):
    OBSERVED = "OBSERVED"
    CANDIDATE = "CANDIDATE"
    AUTHORIZED = "AUTHORIZED"
    COMMITTED = "COMMITTED"
    DENIED = "DENIED"
    UNRESOLVED = "UNRESOLVED"


class VoiceCapability(str, Enum):
    """Capability names used by the voice authority boundary."""

    GRAPH_MUTATION = "graph_mutation"
    COMMAND = "command"
    EXPRESSIVE = "expressive_motion"
    INTERRUPT = "execution_interrupt"
    REFERENCE = "reference_resolution"


class ConversationMode(str, Enum):
    OPEN = "OPEN"
    SPEAKER_ACTIVE = "SPEAKER_ACTIVE"
    OVERLAPPING = "OVERLAPPING"
    INTERRUPTED = "INTERRUPTED"
    WAITING_FOR_CLARIFICATION = "WAITING_FOR_CLARIFICATION"
    EXECUTING = "EXECUTING"


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VoiceError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_probability(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VoiceError(f"{name} must be between 0 and 1") from exc
    if not 0.0 <= result <= 1.0:
        raise VoiceError(f"{name} must be between 0 and 1")
    return result


def _timestamp(value: Any, name: str) -> int:
    # Integer nanoseconds make clock units explicit and reject naive datetime
    # values at this provider boundary.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VoiceError(f"{name} must be a non-negative integer nanosecond timestamp")
    return value


def _scope(session_id: str, org_id: str, expected_session: str, expected_org: str) -> None:
    if session_id != expected_session or org_id != expected_org:
        raise VoiceScopeError(
            f"voice scope mismatch: event={org_id}/{session_id}, "
            f"runtime={expected_org}/{expected_session}"
        )


@dataclass(frozen=True)
class SpeechPartial:
    session_id: str
    org_id: str
    speaker_id: str
    text: str
    t_start_ns: int
    t_end_ns: int
    sequence: int
    source: str = "speechmatics"
    clock_source: str = "monotonic"
    latency_ms: float = 0.0
    confidence: float = 1.0
    pause_ms: float | None = None

    def __post_init__(self) -> None:
        _text(self.session_id, "session_id")
        _text(self.org_id, "org_id")
        _text(self.speaker_id, "speaker_id")
        _timestamp(self.t_start_ns, "t_start_ns")
        _timestamp(self.t_end_ns, "t_end_ns")
        if self.t_end_ns <= self.t_start_ns:
            raise VoiceError("t_end_ns must be greater than t_start_ns")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence <= 0:
            raise VoiceError("sequence must be a positive integer")
        _text(self.source, "source")
        _text(self.clock_source, "clock_source")
        if self.latency_ms < 0 or self.pause_ms is not None and self.pause_ms < 0:
            raise VoiceError("latency_ms and pause_ms must be non-negative")
        _finite_probability(self.confidence, "confidence")

    @property
    def duration_ms(self) -> float:
        return (self.t_end_ns - self.t_start_ns) / 1_000_000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "org_id": self.org_id,
            "speaker_id": self.speaker_id,
            "text": self.text,
            "t_start_ns": self.t_start_ns,
            "t_end_ns": self.t_end_ns,
            "sequence": self.sequence,
            "source": self.source,
            "clock_source": self.clock_source,
            "latency_ms": self.latency_ms,
            "confidence": self.confidence,
            "pause_ms": self.pause_ms,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class SpeechFinal(SpeechPartial):
    """Provider-finalized speech; it still remains an observed claim."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _text(self.text, "text")


@dataclass(frozen=True)
class SpeechClaim:
    claim_id: str
    session_id: str
    org_id: str
    speaker_id: str
    text: str
    t_start_ns: int
    t_end_ns: int
    source: str
    confidence: float
    status: ClaimStatus = ClaimStatus.OBSERVED
    identity_confidence: float = 0.0
    world_entity_id: str | None = None
    #: World revision this claim was acted against, and the revision the
    #: speaker was looking at when they started talking. Speech takes real
    #: time -- measured 1.0-3.7 s from last word to dispatch on this hardware,
    #: part of it a deliberate hold -- so the world can advance mid-sentence.
    #: Recording both makes that lag auditable; see
    #: docs/evidence-lag-audit-2026-09-13.md.
    world_revision: int | None = None
    observed_revision: int | None = None

    @property
    def revision_lag(self) -> int | None:
        """How many world revisions passed while this was being said."""
        if self.world_revision is None or self.observed_revision is None:
            return None
        return self.world_revision - self.observed_revision

    @classmethod
    def from_final(cls, event: SpeechFinal, speaker: "SpeakerState", *,
                   world_revision: int | None = None,
                   observed_revision: int | None = None) -> "SpeechClaim":
        digest = hashlib.sha256(
            f"{event.session_id}|{event.org_id}|{event.sequence}|{event.speaker_id}|"
            f"{event.t_start_ns}|{event.t_end_ns}|{event.text}".encode()
        ).hexdigest()[:20]
        return cls(
            claim_id=f"speech_{digest}",
            session_id=event.session_id,
            org_id=event.org_id,
            speaker_id=event.speaker_id,
            text=event.text,
            t_start_ns=event.t_start_ns,
            t_end_ns=event.t_end_ns,
            source=event.source,
            confidence=event.confidence,
            identity_confidence=speaker.identity_confidence,
            world_entity_id=speaker.world_entity_id,
            world_revision=world_revision,
            observed_revision=observed_revision,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "session_id": self.session_id,
            "org_id": self.org_id,
            "speaker_id": self.speaker_id,
            "text": self.text,
            "t_start_ns": self.t_start_ns,
            "t_end_ns": self.t_end_ns,
            "source": self.source,
            "confidence": self.confidence,
            "status": self.status.value,
            "identity_confidence": self.identity_confidence,
            "world_entity_id": self.world_entity_id,
            "world_revision": self.world_revision,
            "observed_revision": self.observed_revision,
            "revision_lag": self.revision_lag,
        }


@dataclass
class SpeakerState:
    speaker_id: str
    session_id: str
    org_id: str
    display_name: str | None = None
    identity_confidence: float = 0.0
    avg_words_per_sec: float = 2.5
    avg_pause_ms: float = 650.0
    endpoint_ms: int = 700
    active_turn: bool = False
    last_spoke_ns: int = 0
    interruption_count: int = 0
    authority_role: str = "unknown"
    allowed_capabilities: set[str] = field(default_factory=set)
    world_entity_id: str | None = None
    visible_from: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "session_id": self.session_id,
            "org_id": self.org_id,
            "display_name": self.display_name,
            "identity_confidence": self.identity_confidence,
            "avg_words_per_sec": self.avg_words_per_sec,
            "avg_pause_ms": self.avg_pause_ms,
            "endpoint_ms": self.endpoint_ms,
            "active_turn": self.active_turn,
            "last_spoke_ns": self.last_spoke_ns,
            "interruption_count": self.interruption_count,
            "authority_role": self.authority_role,
            "allowed_capabilities": sorted(self.allowed_capabilities),
            "world_entity_id": self.world_entity_id,
            "visible_from": list(self.visible_from),
        }


class SpeakerRegistry:
    """Session-local diarization and physical-association state."""

    def __init__(self, session_id: str, org_id: str) -> None:
        self.session_id = _text(session_id, "session_id")
        self.org_id = _text(org_id, "org_id")
        self._speakers: dict[str, SpeakerState] = {}

    def observe(self, event: SpeechPartial) -> SpeakerState:
        _scope(event.session_id, event.org_id, self.session_id, self.org_id)
        speaker = self._speakers.get(event.speaker_id)
        if speaker is None:
            speaker = SpeakerState(event.speaker_id, self.session_id, self.org_id)
            self._speakers[event.speaker_id] = speaker
        speaker.active_turn = True
        speaker.last_spoke_ns = event.t_end_ns
        return speaker

    def get(self, speaker_id: str) -> SpeakerState:
        try:
            return self._speakers[speaker_id]
        except KeyError as exc:
            raise VoiceError(f"unknown speaker: {speaker_id}") from exc

    def bind_identity(self, speaker_id: str, world_entity_id: str, *,
                      display_name: str | None = None,
                      confidence: float = 0.0,
                      visible_from: Sequence[str] = ()) -> SpeakerState:
        speaker = self.get(speaker_id)
        speaker.world_entity_id = _text(world_entity_id, "world_entity_id")
        speaker.display_name = display_name
        speaker.identity_confidence = _finite_probability(confidence, "confidence")
        speaker.visible_from = list(dict.fromkeys(_text(c, "camera") for c in visible_from))
        return speaker

    def set_authority(self, speaker_id: str, role: str, capabilities: Sequence[str]) -> SpeakerState:
        speaker = self.get(speaker_id)
        speaker.authority_role = _text(role, "role")
        speaker.allowed_capabilities = {_text(c, "capability") for c in capabilities}
        return speaker

    def end_turn(self, speaker_id: str) -> None:
        self.get(speaker_id).active_turn = False

    def snapshot(self) -> tuple[SpeakerState, ...]:
        return tuple(self._speakers.values())


@dataclass(frozen=True)
class TurnDecision:
    speaker_id: str
    endpoint_ms: int
    acoustically_finished: bool
    semantically_complete: bool
    should_wait: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "endpoint_ms": self.endpoint_ms,
            "acoustically_finished": self.acoustically_finished,
            "semantically_complete": self.semantically_complete,
            "should_wait": self.should_wait,
            "reason": self.reason,
        }


class AdaptiveTurnManager:
    """Per-speaker endpointing with a semantic-completeness guard."""

    _INCOMPLETE_END = re.compile(
        r"(?:\.{3}|\u2026|\b(?:a|an|and|at|but|for|from|give|in|on|or|pick|place|put|set|take|that|the|this|to|with)\s*)$",
        re.IGNORECASE,
    )
    _IMMEDIATE = re.compile(r"^\s*(stop|hold|freeze|emergency stop|back ?off)\s*[.!?]*\s*$", re.I)

    def __init__(self, registry: SpeakerRegistry) -> None:
        self.registry = registry

    @classmethod
    def semantic_completeness(cls, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        if cls._IMMEDIATE.match(text):
            return True
        return cls._INCOMPLETE_END.search(text) is None

    def update(self, event: SpeechPartial) -> TurnDecision:
        speaker = self.registry.observe(event)
        words = len(event.text.split())
        duration_s = max(event.duration_ms / 1000.0, 0.05)
        observed_wps = words / duration_s if words else speaker.avg_words_per_sec
        speaker.avg_words_per_sec = 0.8 * speaker.avg_words_per_sec + 0.2 * observed_wps
        if event.pause_ms is not None:
            speaker.avg_pause_ms = 0.8 * speaker.avg_pause_ms + 0.2 * event.pause_ms
        endpoint = int(max(350, min(1400, 450 + speaker.avg_pause_ms * 0.45)))
        complete = self.semantic_completeness(event.text)
        if not complete:
            endpoint = min(1400, endpoint + 250)
        speaker.endpoint_ms = endpoint
        acoustic = event.pause_ms is None or event.pause_ms >= endpoint
        wait = not complete or not acoustic
        reason = "semantic continuation likely" if not complete else (
            "speaker-specific pause budget not reached" if not acoustic else "turn complete"
        )
        return TurnDecision(event.speaker_id, endpoint, acoustic, complete, wait, reason)


@dataclass(frozen=True)
class IntentCandidate:
    intent_id: str
    claim_id: str
    session_id: str
    org_id: str
    speaker_id: str
    capability: str
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    status: ClaimStatus = ClaimStatus.CANDIDATE
    correction_of: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "claim_id": self.claim_id,
            "session_id": self.session_id,
            "org_id": self.org_id,
            "speaker_id": self.speaker_id,
            "capability": self.capability,
            "action": self.action,
            "args": self.args,
            "confidence": self.confidence,
            "status": self.status.value,
            "correction_of": self.correction_of,
        }


class ConversationArbiter:
    """Tracks overlap and explicit corrections without last-speaker-wins."""

    _CORRECTION = re.compile(r"^\s*(no|actually|wait|correction|instead)\b", re.I)

    def __init__(self) -> None:
        self.mode = ConversationMode.OPEN
        self.active_speaker: str | None = None
        self.overlapping_speakers: set[str] = set()
        self._last_by_speaker: dict[str, IntentCandidate] = {}

    def partial(self, speaker_id: str) -> None:
        if self.active_speaker and self.active_speaker != speaker_id:
            self.overlapping_speakers.update({self.active_speaker, speaker_id})
            self.mode = ConversationMode.OVERLAPPING
        else:
            self.active_speaker = speaker_id
            self.mode = ConversationMode.SPEAKER_ACTIVE

    def final(self, claim: SpeechClaim) -> None:
        self.active_speaker = claim.speaker_id
        self.overlapping_speakers.discard(claim.speaker_id)
        self.mode = ConversationMode.SPEAKER_ACTIVE

    def submit(self, candidate: IntentCandidate, text: str) -> IntentCandidate:
        previous = self._last_by_speaker.get(candidate.speaker_id)
        if previous and self._CORRECTION.match(text):
            candidate = replace(candidate, correction_of=previous.intent_id)
        self._last_by_speaker[candidate.speaker_id] = candidate
        return candidate

    def interrupt(self) -> None:
        self.mode = ConversationMode.INTERRUPTED


@dataclass(frozen=True)
class ReferenceClaim:
    claim_id: str
    phrase: str
    subject: str
    resolved: str | None
    confidence: float
    status: ClaimStatus
    evidence: tuple[str, ...] = ()
    recipient: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "phrase": self.phrase,
            "subject": self.subject,
            "resolved": self.resolved,
            "confidence": self.confidence,
            "status": self.status.value,
            "evidence": list(self.evidence),
            "recipient": self.recipient,
        }


class ReferenceResolver:
    """Conservative language-to-world grounding hook.

    A production vision stack can supply ``pointed_at`` and ``recipient``.
    Without a unique world candidate, demonstratives remain unresolved instead
    of being guessed from transcript order.
    """

    _DEMONSTRATIVE = re.compile(r"\b(that|this|one|it)\b", re.I)
    _RECIPIENT = re.compile(r"\b(him|her|them|that person)\b", re.I)

    def resolve(
        self,
        text: str,
        *,
        claim_id: str,
        world: WorldState | None = None,
        pointed_at: str | None = None,
        recipient: str | None = None,
        object_candidates: Sequence[str] = (),
    ) -> tuple[ReferenceClaim, ...]:
        objects = set(world.objects) if world is not None else None
        candidates = [c for c in object_candidates if objects is None or c in objects]
        subject = "object" if self._DEMONSTRATIVE.search(text) else "none"
        resolved: str | None = None
        evidence: list[str] = []
        if subject == "object":
            if pointed_at and (objects is None or pointed_at in objects):
                resolved, evidence = pointed_at, ["vision.pointing"]
            elif len(candidates) == 1:
                resolved, evidence = candidates[0], ["world.unique_candidate"]
        object_claim = ReferenceClaim(
            claim_id=f"{claim_id}:object",
            phrase=text,
            subject=subject,
            resolved=resolved,
            confidence=0.95 if resolved and pointed_at else (0.65 if resolved else 0.0),
            status=ClaimStatus.CANDIDATE if resolved else ClaimStatus.UNRESOLVED,
            evidence=tuple(evidence),
        )
        if self._RECIPIENT.search(text):
            person_claim = ReferenceClaim(
                claim_id=f"{claim_id}:recipient",
                phrase=text,
                subject="recipient",
                resolved=recipient,
                confidence=0.9 if recipient else 0.0,
                status=ClaimStatus.CANDIDATE if recipient else ClaimStatus.UNRESOLVED,
                evidence=("vision.person_association",) if recipient else (),
                recipient=recipient,
            )
            return object_claim, person_claim
        return (object_claim,) if subject != "none" else ()


@dataclass(frozen=True)
class AuthorityDecision:
    speaker_id: str
    capability: str
    allowed: bool
    scope: tuple[str, ...] = ()
    constraints: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "capability": self.capability,
            "allowed": self.allowed,
            "scope": list(self.scope),
            "constraints": self.constraints,
            "reason": self.reason,
        }


class AuthorityResolver:
    """Role/capability gate that never infers authority from a name."""

    def __init__(self, role_capabilities: Mapping[str, Sequence[str]] | None = None,
                 *, interrupt_roles: Sequence[str] = ("operator", "supervisor")) -> None:
        self.role_capabilities = {
            role: frozenset(c for c in caps)
            for role, caps in (role_capabilities or {}).items()
        }
        self.interrupt_roles = frozenset(interrupt_roles)

    def resolve(self, speaker: SpeakerState, capability: str, *,
                world: WorldState | None = None) -> AuthorityDecision:
        capability = _text(capability, "capability")
        if world is not None and world.org_id != speaker.org_id:
            return AuthorityDecision(speaker.speaker_id, capability, False,
                                     reason="speaker organization does not match world")
        granted = set(speaker.allowed_capabilities)
        granted.update(self.role_capabilities.get(speaker.authority_role, ()))
        if capability == VoiceCapability.INTERRUPT.value and speaker.authority_role in self.interrupt_roles:
            granted.add(capability)
        allowed = capability in granted
        return AuthorityDecision(
            speaker_id=speaker.speaker_id,
            capability=capability,
            allowed=allowed,
            scope=tuple(sorted(granted)) if allowed else (),
            reason=("capability granted by explicit role/speaker registration"
                    if allowed else "capability is not authorized for this speaker"),
        )


@dataclass(frozen=True)
class InterruptionDecision:
    detected: bool
    action: str | None
    priority: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "action": self.action,
            "priority": self.priority,
            "reason": self.reason,
        }


class InterruptionGate:
    # "stop" is an emergency reflex, but "stop using the left arm" is an
    # ordinary preference constraint that ``nlu._rule_stop_using_arm`` exists to
    # parse. Without the lookahead the gate claimed the whole sentence, the
    # parser never saw it, and — with no interrupt handler registered — it died
    # as ``interrupt_denied``: the operator's instruction neither stopped
    # anything nor changed the graph. Observed live 2026-09-13.
    # Narrow on purpose: only "stop use/using ..." is excluded. "stop",
    # "stop now", "stop moving", "emergency stop" and "freeze" all still fire,
    # and "stop using the left arm" still stops using it — via prefer_arm.
    _PATTERNS = (
        ("STOP", re.compile(r"\b(emergency\s+stop|stop(?!\s+us(?:e|ing)\b)|freeze)\b",
                            re.I), 100),
        ("HOLD", re.compile(r"\bhold\s+(it|still|position)\b", re.I), 90),
        ("BACKOFF", re.compile(r"\b(back\s+off|back\s+away|retreat)\b", re.I), 80),
    )

    def inspect(self, text: str) -> InterruptionDecision:
        for action, pattern, priority in self._PATTERNS:
            if pattern.search(text):
                return InterruptionDecision(True, action, priority,
                                            f"lexical safety gate matched {action}")
        return InterruptionDecision(False, None, 0, "no interruption phrase")


@dataclass(frozen=True)
class VoiceDispatchResult:
    claim: SpeechClaim
    candidate: IntentCandidate | None = None
    authority: AuthorityDecision | None = None
    references: tuple[ReferenceClaim, ...] = ()
    interruption: InterruptionDecision | None = None
    mutation: dict[str, Any] | None = None
    status: str = "observed"
    response_text: str | None = None
    response_backend: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.as_dict(),
            "candidate": self.candidate.as_dict() if self.candidate else None,
            "authority": self.authority.as_dict() if self.authority else None,
            "references": [r.as_dict() for r in self.references],
            "interruption": self.interruption.as_dict() if self.interruption else None,
            "mutation": self.mutation,
            "response_text": self.response_text,
            "response_backend": self.response_backend,
            "status": self.status,
        }


def intent_capability(parsed: Any, text: str) -> str:
    """Capability a parsed utterance asks for, or ``""`` if it asks for nothing.

    ``""`` is the "not yet an instruction" predicate: the text is an
    observation, either because it genuinely is one ("the plate is blue") or
    because it is only *part* of an instruction ("use your").
    ``IntentAccumulator`` uses exactly this function to decide whether to hold
    an utterance back, so "semantically complete" can never drift apart from
    what ``ingest_final`` actually acts on.
    """
    if parsed.constraints or parsed.mutations:
        return VoiceCapability.GRAPH_MUTATION.value
    return VoiceCapability.COMMAND.value if parsed.goal != text else ""


#: Words an English instruction does not end on. Demonstratives ("that",
#: "this") are deliberately absent -- "don't touch that" is a whole
#: instruction. Used to spot a sentence the provider cut short: live on
#: 2026-09-13 "Omni. Don't use your." arrived punctuated as a finished
#: sentence, a full second before "Left arm."
_DANGLING_WORDS = frozenset({
    "the", "a", "an", "your", "my", "its", "his", "her", "their", "our",
    "and", "or", "to", "of", "with", "for", "on", "in", "at", "from",
    "you", "are", "is", "am", "can", "could", "would", "will", "do", "did",
    "have", "has", "who", "what", "when", "where", "why", "how",
})


def _ends_mid_phrase(text: str) -> bool:
    words = re.findall(r"[\w']+", text.lower())
    return bool(words) and words[-1] in _DANGLING_WORDS


_OMNI_ADDRESS = re.compile(r"^\s*omni(?=\s|[,.!?;:])", re.I)
_DIALOGUE_QUESTION = re.compile(
    r"^(?:who|what|when|where|why|how|can|could|would|will|are|is|do|did|have|has)\b",
    re.I,
)


def _is_addressed_dialogue(text: str) -> bool:
    """Return true only for speech explicitly addressed to OMNI.

    Requiring the wake/address word keeps nearby human conversation from
    becoming model dialogue.  Actionable speech is still selected by
    ``intent_capability`` before this predicate is consulted.
    """
    match = _OMNI_ADDRESS.match(text)
    if match is None:
        return False
    remainder = text[match.end():].lstrip(" \t,.;:!?")
    return bool(re.search(r"[\w']", remainder))


def _is_addressed_question(text: str) -> bool:
    """Recognize a question without weakening explicit action parsing."""
    match = _OMNI_ADDRESS.match(text)
    if match is None:
        return False
    remainder = text[match.end():].lstrip(" \t,.;:!?")
    return bool(_DIALOGUE_QUESTION.match(remainder) or
                remainder.rstrip().endswith("?"))


Hook = Callable[..., None]


class VoiceRuntime:
    """Owns the voice observation-to-intent boundary for one session."""

    def __init__(
        self,
        session_id: str,
        org_id: str,
        *,
        bus: EventBus | None = None,
        registry: SpeakerRegistry | None = None,
        authority: AuthorityResolver | None = None,
        reference_resolver: ReferenceResolver | None = None,
        interruption_gate: InterruptionGate | None = None,
        mutator: Any | None = None,
        intent_handler: Callable[[IntentCandidate, WorldState | None], Any] | None = None,
        interrupt_handler: Callable[[str, str], Any] | None = None,
        dialogue_handler: Callable[..., Any] | None = None,
        min_action_confidence: float = 0.7,
    ) -> None:
        self.session_id = _text(session_id, "session_id")
        self.org_id = _text(org_id, "org_id")
        self.bus = bus or EventBus()
        self.registry = registry or SpeakerRegistry(self.session_id, self.org_id)
        self.turns = AdaptiveTurnManager(self.registry)
        self.arbiter = ConversationArbiter()
        self.authority = authority or AuthorityResolver()
        self.reference_resolver = reference_resolver or ReferenceResolver()
        self.interruption_gate = interruption_gate or InterruptionGate()
        self.mutator = mutator
        self.intent_handler = intent_handler
        self.interrupt_handler = interrupt_handler
        self.dialogue_handler = dialogue_handler
        # Below this transcription confidence a claim is recorded but never
        # becomes an actionable intent. Measured 2026-09-13: room noise came
        # back as "Praise the tag" (0.62) and was authorized as a COMMAND,
        # while every real instruction in the same session scored 0.81-1.0.
        # The recognizer knew it was unsure; nothing was asking.
        # Interruptions are deliberately exempt -- a safety reflex must not be
        # suppressed for being mumbled.
        self.min_action_confidence = _finite_probability(
            min_action_confidence, "min_action_confidence")
        self._hooks: dict[str, list[Hook]] = {}
        self._last_sequence = 0

    def on(self, hook: str, callback: Hook) -> Callable[[], None]:
        if not callable(callback):
            raise VoiceError("voice hook callback must be callable")
        self._hooks.setdefault(_text(hook, "hook"), []).append(callback)

        def remove() -> None:
            if callback in self._hooks.get(hook, []):
                self._hooks[hook].remove(callback)
        return remove

    def _emit_hook(self, hook: str, **payload: Any) -> None:
        for callback in list(self._hooks.get(hook, [])):
            callback(**payload)

    def _check_event(self, event: SpeechPartial) -> None:
        _scope(event.session_id, event.org_id, self.session_id, self.org_id)
        if event.sequence <= self._last_sequence:
            raise VoiceError("speech sequence is not strictly increasing")
        self._last_sequence = event.sequence

    def ingest_partial(self, event: SpeechPartial) -> TurnDecision:
        self._check_event(event)
        known = {speaker.speaker_id for speaker in self.registry.snapshot()}
        previous = self.arbiter.active_speaker
        was_overlapping = self.arbiter.mode is ConversationMode.OVERLAPPING
        turn = self.turns.update(event)
        self.arbiter.partial(event.speaker_id)
        self._emit_speaker_lifecycle(event.speaker_id, known, previous, was_overlapping)
        payload = {"event": event.as_dict(), "turn": turn.as_dict(),
                   "conversation_mode": self.arbiter.mode.value}
        self.bus.publish("voice.partial", source=event.source, **payload)
        self._emit_hook("on_partial_speech", event=event, turn=turn)
        return turn

    def ingest_final(
        self,
        event: SpeechFinal,
        *,
        world: WorldState | None = None,
        pointed_at: str | None = None,
        recipient: str | None = None,
        object_candidates: Sequence[str] = (),
        observed_revision: int | None = None,
    ) -> VoiceDispatchResult:
        """Ingest finalized speech as a claim.

        ``observed_revision`` is the world revision the speaker was looking at
        when they began the utterance. Supplying it turns on evidence-lag
        handling for *references*: speech takes 1.0-3.7 s to arrive here on
        this hardware (measured 2026-09-13, partly a deliberate hold), and a
        demonstrative like "that one" resolved at commit time binds against a
        world the speaker may never have seen. Constraints are unaffected --
        "don't use the left arm" is still valid three seconds later, and
        rejecting on lag would break the feature while looking like rigour.
        Omit it and behaviour is exactly as before.
        """
        self._check_event(event)
        known = {speaker_state.speaker_id for speaker_state in self.registry.snapshot()}
        previous = self.arbiter.active_speaker
        was_overlapping = self.arbiter.mode is ConversationMode.OVERLAPPING
        speaker = self.registry.observe(event)
        turn = self.turns.update(event)
        world_revision = getattr(world, "revision", None) if world is not None else None
        claim = SpeechClaim.from_final(event, speaker, world_revision=world_revision,
                                       observed_revision=observed_revision)
        self.arbiter.final(claim)
        self._emit_speaker_lifecycle(event.speaker_id, known, previous, was_overlapping)
        if was_overlapping:
            self.bus.publish("voice.overlap.resolved", source="voice",
                             speaker_id=event.speaker_id)
            self._emit_hook("on_overlap_resolved", speaker_id=event.speaker_id)
        self.bus.publish("voice.claim.observed", source=event.source,
                         claim=claim.as_dict(), turn=turn.as_dict(),
                         conversation_mode=self.arbiter.mode.value)
        self._emit_hook("on_final_speech", event=event, claim=claim, turn=turn)

        interruption = self.interruption_gate.inspect(claim.text)
        if interruption.detected:
            self.arbiter.interrupt()
            self.bus.publish("voice.interrupt.detected", source="voice",
                             claim=claim.as_dict(), interruption=interruption.as_dict())
            self._emit_hook("on_interrupt", claim=claim, interruption=interruption)
            decision = self.authority.resolve(speaker, VoiceCapability.INTERRUPT.value, world=world)
            self.bus.publish("voice.authority.resolved", source="voice",
                             claim_id=claim.claim_id, authorization=decision.as_dict())
            if decision.allowed and self.interrupt_handler is not None:
                self.interrupt_handler(interruption.action or "STOP", claim.claim_id)
                self.bus.publish("execution.interrupt.requested", source="voice",
                                 action=interruption.action, claim_id=claim.claim_id)
                self._emit_hook("on_execution_interrupt", claim=claim,
                                decision=decision, interruption=interruption)
                return VoiceDispatchResult(claim, authority=decision,
                                           interruption=interruption, status="interrupted")
            return VoiceDispatchResult(claim, authority=decision,
                                       interruption=interruption, status="interrupt_denied")

        parsed = nlu.parse(claim.text)
        references = self.reference_resolver.resolve(
            claim.text, claim_id=claim.claim_id, world=world,
            pointed_at=pointed_at, recipient=recipient,
            object_candidates=object_candidates,
        )
        lag = claim.revision_lag
        if lag:
            # The world advanced while this was being said, so a demonstrative
            # was resolved against a scene the speaker was not looking at.
            # Unresolve rather than guess -- the same conservatism the resolver
            # already applies to an ambiguous phrase.
            stale = tuple(
                replace(reference, resolved=None, status=ClaimStatus.UNRESOLVED,
                        evidence=reference.evidence + (f"stale_by_{lag}_revisions",))
                if reference.resolved else reference
                for reference in references
            )
            if any(r.resolved for r in references):
                self.bus.publish("voice.reference.stale", source=event.source,
                                 claim_id=claim.claim_id, revision_lag=lag,
                                 observed_revision=claim.observed_revision,
                                 world_revision=claim.world_revision)
                self._emit_hook("on_reference_stale", claim=claim, revision_lag=lag)
            references = stale
        for reference in references:
            self.bus.publish("voice.reference.candidate", source="voice",
                             claim_id=claim.claim_id, reference=reference.as_dict())
            self._emit_hook("on_reference_candidate", claim=claim, reference=reference)
            if reference.resolved or reference.recipient:
                self.bus.publish("voice.reference.resolved", source="voice",
                                 claim_id=claim.claim_id, reference=reference.as_dict())
                self._emit_hook("on_reference_resolved", claim=claim, reference=reference)
        addressed_question = _is_addressed_question(claim.text)
        capability = intent_capability(parsed, claim.text)
        # Preserve the existing conservative authority behavior for ordinary
        # capitalized sentences, but let an explicit question addressed to
        # OMNI use the read-only lane when NLU only produced a bare COMMAND.
        # Explicit graph constraints/mutations are left on the action lane.
        if addressed_question and capability == VoiceCapability.COMMAND.value:
            capability = ""
        low_confidence_action = False
        if capability and claim.confidence < self.min_action_confidence:
            low_confidence_action = True
            self.bus.publish("voice.intent.low_confidence", source=event.source,
                             claim=claim.as_dict(), capability=capability,
                             floor=self.min_action_confidence)
            self._emit_hook("on_low_confidence", claim=claim, capability=capability)
            capability = ""  # recorded as an observation, never authorized
        action = "GRAPH_MUTATION" if capability == VoiceCapability.GRAPH_MUTATION.value else (
            "COMMAND" if capability else "OBSERVATION"
        )
        args = parsed.as_dict()
        if references:
            args["references"] = [r.as_dict() for r in references]
        candidate = IntentCandidate(
            intent_id=f"intent_{claim.claim_id.removeprefix('speech_')}",
            claim_id=claim.claim_id,
            session_id=self.session_id,
            org_id=self.org_id,
            speaker_id=claim.speaker_id,
            capability=capability or "observation",
            action=action,
            args=args,
            confidence=claim.confidence,
        )
        candidate = self.arbiter.submit(candidate, claim.text)
        self.bus.publish("voice.intent.candidate", source="voice",
                         intent=candidate.as_dict(), references=[r.as_dict() for r in references])
        self._emit_hook("on_intent_candidate", claim=claim, intent=candidate,
                        references=references)

        if not capability:
            if (not low_confidence_action and self.dialogue_handler is not None
                    and _is_addressed_dialogue(claim.text)):
                return self._answer_dialogue(
                    claim, candidate, references, world,
                )
            return VoiceDispatchResult(claim, candidate=candidate, references=references,
                                       status="observed")
        decision = self.authority.resolve(speaker, capability, world=world)
        self.bus.publish("voice.authority.resolved", source="voice",
                         claim_id=claim.claim_id, authorization=decision.as_dict())
        self._emit_hook("on_authority_resolved", claim=claim, intent=candidate,
                        decision=decision)
        if not decision.allowed:
            denied = replace(candidate, status=ClaimStatus.DENIED)
            self.bus.publish("voice.intent.denied", source="voice",
                             intent=denied.as_dict(), authorization=decision.as_dict())
            return VoiceDispatchResult(claim, denied, decision, references,
                                       status="denied")

        mutation: dict[str, Any] | None = None
        committed = False
        if capability == VoiceCapability.GRAPH_MUTATION.value and self.mutator is not None:
            result = self.mutator.apply(claim.text)
            mutation = result.as_dict() if hasattr(result, "as_dict") else dict(result)
            committed = bool(getattr(result, "ok", mutation.get("ok", False)))
        if self.intent_handler is not None:
            handled = self.intent_handler(candidate, world)
            if handled is not None:
                mutation = handled.as_dict() if hasattr(handled, "as_dict") else handled
                handled_ok = handled if isinstance(handled, bool) else (
                    handled.get("ok", True) if isinstance(handled, dict) else
                    getattr(handled, "ok", True)
                )
                committed = bool(handled_ok)
        authorized = replace(candidate, status=ClaimStatus.AUTHORIZED)
        self.bus.publish("voice.intent.authorized", source="voice",
                         intent=authorized.as_dict(), mutation=mutation)
        if not committed:
            self.bus.publish("voice.intent.not_committed", source="voice",
                             intent=authorized.as_dict(), mutation=mutation,
                             reason="no downstream handler confirmed commitment")
            return VoiceDispatchResult(claim, authorized, decision, references,
                                       mutation=mutation, status="authorized")
        committed_intent = replace(candidate, status=ClaimStatus.COMMITTED)
        self.bus.publish("voice.intent.committed", source="voice",
                         intent=committed_intent.as_dict(), mutation=mutation)
        self._emit_hook("on_intent_committed", claim=claim, intent=committed_intent,
                        decision=decision, mutation=mutation)
        return VoiceDispatchResult(claim, committed_intent, decision, references,
                                   mutation=mutation, status="committed")

    def _answer_dialogue(
        self,
        claim: SpeechClaim,
        candidate: IntentCandidate,
        references: tuple[ReferenceClaim, ...],
        world: WorldState | None,
    ) -> VoiceDispatchResult:
        """Run the read-only dialogue lane after action classification.

        The callback receives no mutator or authority object.  A world from a
        different organization is rejected before the callback is invoked,
        and backend failures become an explicit unavailable response rather
        than a fabricated observation or an action retry.
        """
        self.bus.publish(
            "voice.dialogue.requested",
            source="voice",
            claim_id=claim.claim_id,
            session_id=self.session_id,
            org_id=self.org_id,
            references=[reference.as_dict() for reference in references],
        )
        try:
            if world is not None and world.org_id != self.org_id:
                raise VoiceScopeError(
                    "dialogue world organization does not match runtime organization"
                )
            response = self.dialogue_handler(
                claim=claim,
                world=world,
                references=references,
            )
            if isinstance(response, Mapping):
                text = response.get("text")
                backend = response.get("backend", "dialogue")
            else:
                text = getattr(response, "text", response if isinstance(response, str) else None)
                backend = getattr(response, "backend", "dialogue")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("dialogue handler returned empty text")
            if not isinstance(backend, str) or not backend.strip():
                backend = "dialogue"
        except Exception as exc:  # noqa: BLE001 - read-only provider boundary
            reason = "scope_mismatch" if isinstance(exc, VoiceScopeError) else "backend_error"
            self.bus.publish(
                "voice.dialogue.failed",
                source="voice",
                claim_id=claim.claim_id,
                reason=reason,
                error_type=type(exc).__name__,
            )
            return VoiceDispatchResult(
                claim,
                candidate=candidate,
                references=references,
                response_text=(
                    "I can't answer that from the current workspace."
                    if reason == "scope_mismatch"
                    else "I can't answer that from my current state."
                ),
                response_backend="unavailable",
                status="dialogue_failed",
            )

        self.bus.publish(
            "voice.dialogue.answered",
            source="voice",
            claim_id=claim.claim_id,
            backend=backend.strip(),
        )
        return VoiceDispatchResult(
            claim,
            candidate=candidate,
            references=references,
            response_text=text.strip(),
            response_backend=backend.strip(),
            status="answered",
        )

    def finish_turn(self, speaker_id: str) -> None:
        """Close a speaker turn and expose the idle boundary to subscribers."""
        self.registry.end_turn(speaker_id)
        if self.arbiter.active_speaker == speaker_id:
            self.arbiter.active_speaker = None
            self.arbiter.mode = ConversationMode.OPEN
        self.bus.publish("voice.speaker.stopped", source="voice", speaker_id=speaker_id)
        self._emit_hook("on_speaker_stopped", speaker_id=speaker_id)
        if self.arbiter.active_speaker is None:
            self.bus.publish("voice.conversation.idle", source="voice")
            self._emit_hook("on_conversation_idle")

    def _emit_speaker_lifecycle(self, speaker_id: str, known: set[str],
                                previous: str | None, was_overlapping: bool) -> None:
        if speaker_id not in known:
            self.bus.publish("voice.speaker.started", source="voice", speaker_id=speaker_id)
            self._emit_hook("on_speaker_started", speaker_id=speaker_id)
        if previous and previous != speaker_id:
            self.bus.publish("voice.speaker.changed", source="voice",
                             old_speaker_id=previous, new_speaker_id=speaker_id)
            self._emit_hook("on_speaker_changed", old_speaker_id=previous,
                            new_speaker_id=speaker_id)
        if self.arbiter.mode is ConversationMode.OVERLAPPING and not was_overlapping:
            speakers = tuple(sorted(self.arbiter.overlapping_speakers))
            self.bus.publish("voice.overlap.started", source="voice", speakers=speakers)
            self._emit_hook("on_overlap_started", speakers=speakers)


class IntentAccumulator:
    """Hold utterances that are not yet an instruction, and join them.

    ``VoiceRuntime`` acts on one final at a time, which assumes each final is a
    whole thought.  Speech does not work that way: a speaker pauses mid-clause,
    and a transport that segments on silence alone then delivers "don't use"
    and "your left arm anymore" as two claims — each individually meaningless,
    so the instruction is recorded as two observations and never executed.
    (Observed for real on 2026-09-13; see
    ``integrations/speechmatics/README.md``.)

    This sits between utterance segmentation and the runtime and is shaped like
    a ``VoiceRuntime``, so anything that takes a runtime — the Speechmatics
    adapter included — takes one of these instead, unmodified::

        adapter = SpeechmaticsRealtimeAdapter(IntentAccumulator(runtime))

    Rules, each of which exists to prevent a specific failure:

    * an utterance whose parse asks for no capability (``intent_capability``
      returns ``""``) is **held**, not dispatched;
    * the next utterance from the same speaker is appended and the joined text
      re-parsed — so completion is decided by the same predicate the runtime
      acts on, never by a second opinion;
    * holding is bounded by ``window_ms`` (the gap between consecutive
      utterances) and ``max_utterances``.  Without bounds, two unrelated
      remarks minutes apart would concatenate into a command nobody uttered.
      ``window_ms`` is the meaningful bound; the count is a backstop, and it is
      deliberately generous — halting speech produced **six** fragments for one
      instruction on 2026-09-13, and a limit of 3 cut the instruction in half;
    * a hold that expires is **dispatched anyway** as an ordinary observation.
      Speech is never silently discarded — an unexecuted instruction must still
      be visible in the claim record;
    * authority is not consulted here.  It stays inside ``ingest_final``, so
      unauthorized speakers still produce claims and explicit denials rather
      than vanishing before the boundary sees them.
    """

    def __init__(self, runtime: VoiceRuntime, *, window_ms: float = 4000.0,
                 max_utterances: int = 8, min_command_words: int = 2,
                 sequence_source: Callable[[], int] | None = None,
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        self.runtime = runtime
        self.clock = clock
        self._held_since_ns: int | None = None
        self.window_ms = float(window_ms)
        self.max_utterances = int(max_utterances)
        self.min_command_words = int(min_command_words)
        self.sequence_source = sequence_source
        self._held: list[SpeechFinal] = []
        self._held_kwargs: dict[str, Any] = {}
        self.held_count = 0
        self.joined_count = 0
        self.expired_count = 0
        self.semantic_turn_count = 0
        self.semantic_turn_ignored_count = 0

    # Runtime-shaped surface -------------------------------------------
    @property
    def session_id(self) -> str:
        return self.runtime.session_id

    @property
    def org_id(self) -> str:
        return self.runtime.org_id

    @property
    def registry(self) -> SpeakerRegistry:
        return self.runtime.registry

    @property
    def bus(self) -> EventBus:
        return self.runtime.bus

    def on(self, hook: str, callback: Hook) -> Callable[[], None]:
        return self.runtime.on(hook, callback)

    def ingest_partial(self, event: SpeechPartial) -> TurnDecision:
        return self.runtime.ingest_partial(event)

    def ingest_final(self, event: SpeechFinal, **kwargs: Any):
        """Dispatch, or hold and return ``None``.

        ``None`` means "not an instruction yet, kept for the next utterance" —
        it is not a failure, and callers should report it as held rather than
        as a dropped final.
        """
        if self._held and self._expired(event):
            self._release(**kwargs)
        candidate = self._joined_text(event)
        if self._complete(candidate):
            joined = self._merge(event, candidate)
            dispatch_kwargs = dict(self._held_kwargs or kwargs)
            self._held = []
            self._held_since_ns = None
            self._held_kwargs = {}
            if joined is not event:
                self.joined_count += 1
            return self.runtime.ingest_final(joined, **dispatch_kwargs)
        self._held.append(event)
        self.held_count += 1
        if self._held_since_ns is None:
            self._held_since_ns = int(self.clock())
            self._held_kwargs = dict(kwargs)
        self.bus.publish("voice.intent.incomplete", source=event.source,
                         speaker_id=event.speaker_id, text=candidate,
                         held_utterances=len(self._held))
        if len(self._held) >= self.max_utterances:
            return self._release(**kwargs)
        return None

    def flush(self, **kwargs: Any):
        """Release anything still held, e.g. at end of stream."""
        return self._release(**kwargs) if self._held else None

    def tick(self, now_ns: int | None = None, **kwargs: Any):
        """Release a hold that has waited out ``window_ms`` on the wall clock.

        Needed because holds are otherwise only re-examined when the *next*
        utterance arrives: an operator who says "set the table" and then stops
        talking would wait forever, since a bare goal waits for a sentence end
        that may never be transcribed.
        """
        if not self._held or self._held_since_ns is None:
            return None
        now = int(self.clock() if now_ns is None else now_ns)
        if (now - self._held_since_ns) / 1_000_000.0 >= self.window_ms:
            return self._release(**kwargs)
        return None

    def on_turn_signal(self, payload: Mapping[str, Any], **kwargs: Any):
        """Close a hold when the provider declares a semantic turn end.

        Provider predictions are deliberately not authority.  A predicted
        wait or an unrecognised Smart Turn payload is telemetry only; the
        accumulator keeps its existing completion and timeout fallbacks.
        """
        if not isinstance(payload, Mapping):
            raise VoiceError("provider turn signal must be a mapping")

        kind = _provider_event_kind(payload)
        if kind == "endofturnprediction":
            self.bus.publish(
                "voice.intent.turn_prediction",
                source="speechmatics",
                predicted_wait=payload.get("predicted_wait"),
            )
            return None

        if not self._turn_signal_complete(payload, kind):
            return None
        if not self._held:
            return None

        signal_speaker = _provider_speaker(payload)
        held_speaker = self._held[-1].speaker_id
        if signal_speaker is not None and signal_speaker != held_speaker:
            self.semantic_turn_ignored_count += 1
            self.bus.publish(
                "voice.intent.turn_ignored",
                source="speechmatics",
                reason="speaker_mismatch",
                signal_speaker_id=signal_speaker,
                held_speaker_id=held_speaker,
            )
            return None

        self.semantic_turn_count += 1
        self.bus.publish(
            "voice.intent.semantic_turn",
            source="speechmatics",
            provider_event=kind,
            speaker_id=held_speaker,
            held_utterances=len(self._held),
        )
        return self._release(**kwargs)

    @staticmethod
    def _turn_signal_complete(payload: Mapping[str, Any], kind: str) -> bool:
        if kind == "endofturn":
            return True
        if kind != "smartturnresult":
            return False
        for key in (
            "is_end_of_turn",
            "end_of_turn",
            "turn_complete",
            "is_complete",
            "complete",
        ):
            value = payload.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "complete", "completed", "end_of_turn"}:
                    return True
                if normalized in {"false", "incomplete", "not_end_of_turn"}:
                    return False
        result = payload.get("result")
        if isinstance(result, Mapping):
            return IntentAccumulator._turn_signal_complete(result, "smartturnresult")
        return False

    # Internals ---------------------------------------------------------
    def _complete(self, text: str) -> bool:
        """Is this text something the runtime could actually act on?

        Safety first: anything the interruption gate recognizes (STOP / HOLD /
        BACKOFF) is complete by definition and is never held.  Delaying a
        safety word to wait for more speech would be the worst bug in this
        file.
        """
        if self.runtime.interruption_gate.inspect(text).detected:
            return True
        if _is_addressed_dialogue(text):
            # Questions are a complete read-only lane of their own. Without
            # this branch, ``Omni, why are you...`` has no action capability
            # and waits for the accumulation timeout before OMNI can answer.
            return not _ends_mid_phrase(text)
        capability = intent_capability(nlu.parse(text), text)
        if not capability:
            return False
        if _ends_mid_phrase(text):
            # The provider punctuates aggressively; a period after "your" is
            # not a sentence end, and dispatching there loses the object of
            # the instruction entirely.
            return False
        if capability == VoiceCapability.GRAPH_MUTATION.value:
            # A constraint or mutation is unambiguous: act at once.
            return True
        # A bare COMMAND is the weak signal -- it means only that goal
        # classification reworded the text, which fires on fragments that are
        # not instructions at all. Live on 2026-09-13, "Keep" dispatched as an
        # authorized command a second before "everything local" arrived, and
        # "They don't use" dispatched before "your left arm anymore". So a bare
        # goal waits for the sentence to actually end; the provider sends
        # terminal punctuation as its own fragment moments later. If it never
        # comes, the window backstop releases the text anyway.
        if len(text.split()) < self.min_command_words:
            return False
        return text.rstrip().endswith((".", "!", "?"))

    def _expired(self, event: SpeechFinal) -> bool:
        last = self._held[-1]
        if event.speaker_id != last.speaker_id:
            return True
        gap_ms = (event.t_start_ns - last.t_end_ns) / 1_000_000.0
        return gap_ms > self.window_ms

    def _joined_text(self, event: SpeechFinal | None = None) -> str:
        # Held text was judged incomplete, so any sentence-final punctuation
        # the provider attached to it was wrong. Keeping it breaks the join:
        # "don't use your." + "Left arm." reads as "don't use your. Left arm.",
        # and the parser cannot see across the period -- the instruction stays
        # unrecognized even though both halves arrived. Observed 2026-09-13.
        parts = [held.text.rstrip().rstrip(".!?") for held in self._held]
        if event is not None:
            parts.append(event.text)
        joined = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if joined and part[0] not in ",.!?;:":
                joined += " "
            joined += part
        return joined

    def _merge(self, event: SpeechFinal, text: str) -> SpeechFinal:
        changes: dict[str, Any] = {}
        if self._held:
            changes["text"] = text
            changes["t_start_ns"] = self._held[0].t_start_ns
        if self.sequence_source is None:
            return replace(event, **changes) if changes else event
        # Always take a fresh number, even when nothing was held. Holding
        # reorders dispatch relative to arrival: an expiring hold is released
        # *during* the handling of a newer event, takes a sequence, and the
        # event that triggered the release would then follow carrying an older
        # one -- which VoiceRuntime rejects, dropping real speech. Observed
        # live 2026-09-13 (one "sequence is not strictly increasing" rejection
        # in a 63 s session). Allocating at dispatch keeps arrival order and
        # dispatch order from ever disagreeing.
        changes["sequence"] = self.sequence_source()
        return replace(event, **changes)

    def _release(self, **kwargs: Any):
        """Dispatch held speech as-is; it never became an instruction."""
        held, self._held = self._held, []
        self._held_since_ns = None
        dispatch_kwargs = dict(self._held_kwargs or kwargs)
        self._held_kwargs = {}
        text = ""
        for part in (h.text for h in held):
            part = part.strip()
            if not part:
                continue
            if text and part[0] not in ",.!?;:":
                text += " "
            text += part
        if not text:
            return None
        self.expired_count += 1
        changes: dict[str, Any] = {"text": text, "t_start_ns": held[0].t_start_ns,
                                   "t_end_ns": held[-1].t_end_ns}
        if self.sequence_source is not None:
            changes["sequence"] = self.sequence_source()
        return self.runtime.ingest_final(
            replace(held[-1], **changes), **dispatch_kwargs
        )

    def stats(self) -> dict[str, Any]:
        return {
            "window_ms": self.window_ms,
            "max_utterances": self.max_utterances,
            "utterances_held": self.held_count,
            "instructions_completed_by_joining": self.joined_count,
            "holds_released_unexecuted": self.expired_count,
            "semantic_turns": self.semantic_turn_count,
            "semantic_turns_ignored": self.semantic_turn_ignored_count,
        }


class SpeechmaticsRealtimeAdapter:
    """Translate Speechmatics-like payloads without importing its SDK."""

    def __init__(self, runtime: VoiceRuntime) -> None:
        self.runtime = runtime

    def _event(self, payload: Mapping[str, Any], *, final: bool) -> SpeechPartial:
        if not isinstance(payload, Mapping):
            raise VoiceError("Speechmatics payload must be a mapping")
        cls = SpeechFinal if final else SpeechPartial
        return cls(
            session_id=payload.get("session_id", self.runtime.session_id),
            org_id=payload.get("org_id", self.runtime.org_id),
            speaker_id=payload.get("speaker_id", payload.get("speaker")),
            text=payload.get("text", ""),
            t_start_ns=payload.get("t_start_ns", payload.get("start_ns")),
            t_end_ns=payload.get("t_end_ns", payload.get("end_ns")),
            sequence=payload.get("sequence"),
            source="speechmatics",
            clock_source=payload.get("clock_source", "monotonic"),
            latency_ms=payload.get("latency_ms", 0.0),
            confidence=payload.get("confidence", 1.0),
            pause_ms=payload.get("pause_ms"),
        )

    def on_partial(self, payload: Mapping[str, Any]) -> TurnDecision:
        return self.runtime.ingest_partial(self._event(payload, final=False))

    def on_final(self, payload: Mapping[str, Any], **kwargs: Any) -> VoiceDispatchResult:
        return self.runtime.ingest_final(self._event(payload, final=True), **kwargs)

    def on_provider_event(self, payload: Mapping[str, Any], **kwargs: Any):
        """Pass provider control events to an accumulation layer if present."""
        handler = getattr(self.runtime, "on_turn_signal", None)
        if handler is None:
            return None
        return handler(payload, **kwargs)


def _provider_event_kind(payload: Mapping[str, Any]) -> str:
    value = payload.get("message", payload.get("type", ""))
    return str(value).strip().lower().replace("_", "")


def _provider_speaker(payload: Mapping[str, Any]) -> str | None:
    for key in ("speaker_id", "speaker", "speaker_label"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None
