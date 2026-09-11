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

    @classmethod
    def from_final(cls, event: SpeechFinal, speaker: "SpeakerState") -> "SpeechClaim":
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
    _PATTERNS = (
        ("STOP", re.compile(r"\b(emergency\s+stop|stop|freeze)\b", re.I), 100),
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.as_dict(),
            "candidate": self.candidate.as_dict() if self.candidate else None,
            "authority": self.authority.as_dict() if self.authority else None,
            "references": [r.as_dict() for r in self.references],
            "interruption": self.interruption.as_dict() if self.interruption else None,
            "mutation": self.mutation,
            "status": self.status,
        }


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
    ) -> VoiceDispatchResult:
        self._check_event(event)
        known = {speaker_state.speaker_id for speaker_state in self.registry.snapshot()}
        previous = self.arbiter.active_speaker
        was_overlapping = self.arbiter.mode is ConversationMode.OVERLAPPING
        speaker = self.registry.observe(event)
        turn = self.turns.update(event)
        claim = SpeechClaim.from_final(event, speaker)
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
        for reference in references:
            self.bus.publish("voice.reference.candidate", source="voice",
                             claim_id=claim.claim_id, reference=reference.as_dict())
            self._emit_hook("on_reference_candidate", claim=claim, reference=reference)
            if reference.resolved or reference.recipient:
                self.bus.publish("voice.reference.resolved", source="voice",
                                 claim_id=claim.claim_id, reference=reference.as_dict())
                self._emit_hook("on_reference_resolved", claim=claim, reference=reference)
        capability = VoiceCapability.GRAPH_MUTATION.value if (
            parsed.constraints or parsed.mutations
        ) else VoiceCapability.COMMAND.value if parsed.goal != claim.text else ""
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
