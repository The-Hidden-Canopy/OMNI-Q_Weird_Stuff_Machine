"""Speechmatics realtime message -> OMNI-Q voice payload.

Pure translation: no sockets, no SDK, no clock of its own beyond an injected
one.  Everything the transport needs to decide is decided here, so the mapping
rules can be tested against recorded provider messages with no network and no
API key.

What the provider gives us (``AddPartialTranscript`` / ``AddTranscript``)::

    {"message": "AddTranscript",
     "metadata": {"start_time": 1.2, "end_time": 2.4, "transcript": "hello"},
     "results": [{"type": "word", "start_time": 1.2, "end_time": 1.5,
                  "alternatives": [{"content": "hello", "confidence": 0.95,
                                    "speaker": "S1"}]}]}

What ``omni_q.voice.SpeechmaticsRealtimeAdapter`` requires (see
``SpeechPartial.__post_init__`` -- these are hard validation errors, not
conventions):

* ``speaker_id`` non-empty string -- Speechmatics only labels speakers when
  diarization is on, so an explicit fallback id is configured rather than
  invented per message;
* ``t_start_ns`` / ``t_end_ns`` non-negative **integer nanoseconds** with
  ``t_end_ns > t_start_ns`` strictly -- provider times are float seconds
  relative to stream start, and a zero-length partial is legal there;
* ``sequence`` a positive integer, strictly increasing across partials *and*
  finals together (``VoiceRuntime._check_event`` shares one counter);
* ``confidence`` in ``[0, 1]``.

Two deliberate honesty choices:

* ``confidence`` is the mean of the word-level alternative confidences that the
  provider actually sent.  When a message carries no confidence at all we mark
  ``confidence_measured=False`` on the mapped record rather than silently
  reporting the 1.0 that the dataclass defaults to.
* ``latency_ms`` is measured, not declared: wall-clock now minus the stream
  position the message claims to describe, on the same monotonic clock that
  timestamps the audio.  It is the "how far behind live are we" number that the
  latency receipt reports, and it is clamped at zero rather than allowed to go
  negative when the provider's stream clock runs slightly ahead of ours.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

__all__ = [
    "MappedTranscript",
    "TranscriptMapper",
    "PARTIAL_MESSAGE",
    "FINAL_MESSAGE",
]

PARTIAL_MESSAGE = "AddPartialTranscript"
FINAL_MESSAGE = "AddTranscript"

_NS_PER_SEC = 1_000_000_000


@dataclass(frozen=True)
class MappedTranscript:
    """One provider transcript message translated for the voice adapter."""

    payload: dict[str, Any]
    final: bool
    confidence_measured: bool
    speaker_labelled: bool

    @property
    def text(self) -> str:
        return str(self.payload.get("text", ""))

    @property
    def latency_ms(self) -> float:
        return float(self.payload.get("latency_ms", 0.0))


@dataclass
class TranscriptMapper:
    """Translate one Speechmatics realtime session into adapter payloads.

    One mapper per session: it owns the sequence counter, the stream epoch, and
    the previous-final boundary used for ``pause_ms``.
    """

    session_id: str
    org_id: str
    default_speaker_id: str = "S1"
    clock: Callable[[], int] = time.monotonic_ns
    clock_source: str = "monotonic"
    #: Minimum span forced onto a transcript whose provider start == end.
    min_span_ns: int = 1_000_000  # 1 ms
    language: str = "en"

    _sequence: int = field(default=0, init=False)
    _epoch_ns: int | None = field(default=None, init=False)
    _last_final_end_ns: int | None = field(default=None, init=False)
    dropped_empty: int = field(default=0, init=False)
    #: Stream position of the most recently dropped empty transcript.  Silence
    #: arrives as empty ``AddTranscript`` messages, and that is real evidence
    #: the speaker stopped -- the aggregator needs it to close an utterance on
    #: the provider's own clock instead of waiting for the socket to close.
    last_empty_end_ns: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("language must be a non-empty string")
        self.language = self.language.strip().lower().replace("_", "-")

    # -- lifecycle -------------------------------------------------------

    def start_stream(self, epoch_ns: int | None = None) -> int:
        """Anchor provider-relative times to our clock.

        Call this when ``RecognitionStarted`` arrives -- not at connect time --
        so network/handshake time is not charged to the audio timeline.
        """
        self._epoch_ns = int(self.clock() if epoch_ns is None else epoch_ns)
        return self._epoch_ns

    @property
    def epoch_ns(self) -> int | None:
        return self._epoch_ns

    @property
    def sequence(self) -> int:
        """Last sequence number actually emitted."""
        return self._sequence

    def next_sequence(self) -> int:
        """Claim the next sequence number.

        The aggregator uses this when it flushes a reassembled utterance: the
        number has to come from this one counter, because partials emitted
        while the utterance was buffering already consumed numbers and
        ``VoiceRuntime`` requires strict increase across partials and finals
        alike.  Reusing the buffered fragment's own number would go backwards
        and be rejected.
        """
        self._sequence += 1
        return self._sequence

    # -- translation -----------------------------------------------------

    def map_message(self, message: Mapping[str, Any]) -> MappedTranscript | None:
        """Return a mapped transcript, or ``None`` for anything not one.

        ``None`` covers both non-transcript control messages (``AudioAdded``,
        ``Info``, ...) and transcripts that carry no usable text -- an empty
        final would fail ``SpeechFinal``'s own text validation, so it is
        counted and dropped here instead of raising inside the runtime.
        """
        if not isinstance(message, Mapping):
            raise TypeError("Speechmatics message must be a mapping")
        kind = message.get("message")
        if kind not in (PARTIAL_MESSAGE, FINAL_MESSAGE):
            return None
        final = kind == FINAL_MESSAGE

        text = self._transcript_text(message)
        if not text:
            self.dropped_empty += 1
            if self._epoch_ns is not None:
                end_s = _first_float((message.get("metadata") or {}).get("end_time"),
                                     default=-1.0)
                if end_s >= 0.0:
                    self.last_empty_end_ns = (
                        self._epoch_ns + int(round(end_s * _NS_PER_SEC))
                    )
            return None

        if self._epoch_ns is None:
            # Defensive: a transcript before RecognitionStarted still needs an
            # anchor, and anchoring late is better than emitting garbage times.
            self.start_stream()
        epoch = int(self._epoch_ns or 0)

        metadata = message.get("metadata") or {}
        results = [r for r in (message.get("results") or []) if isinstance(r, Mapping)]
        # Word timings win over the message envelope. A partial's metadata
        # end_time is the current *audio position*, not where the recognized
        # speech ends -- measured 2026-09-13, the partial "Omni" arrived
        # stamped 0.00-1.64. Using the envelope made every partial look like it
        # had arrived before the audio it describes, so the reported latency
        # clamped to a meaningless 0.0 ms. The last word's end is when the
        # speech actually stopped, which is also the honest basis for
        # duration_ms and pause_ms.
        start_s = _first_float(
            *(r.get("start_time") for r in results),
            metadata.get("start_time"),
            default=0.0,
        )
        end_s = _first_float(
            *(r.get("end_time") for r in reversed(results)),
            metadata.get("end_time"),
            default=start_s,
        )
        start_ns = epoch + max(0, int(round(start_s * _NS_PER_SEC)))
        end_ns = epoch + max(0, int(round(end_s * _NS_PER_SEC)))
        if end_ns <= start_ns:
            end_ns = start_ns + self.min_span_ns

        confidence, measured = self._confidence(results)
        speaker, labelled = self._speaker(results)
        language = self._language(message, results)

        now_ns = int(self.clock())
        latency_ms = max(0.0, (now_ns - end_ns) / 1_000_000.0)

        pause_ms: float | None = None
        if self._last_final_end_ns is not None:
            pause_ms = max(0.0, (start_ns - self._last_final_end_ns) / 1_000_000.0)

        self._sequence += 1
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "org_id": self.org_id,
            "speaker_id": speaker,
            "text": text,
            "language": language,
            "t_start_ns": start_ns,
            "t_end_ns": end_ns,
            "sequence": self._sequence,
            "clock_source": self.clock_source,
            "latency_ms": latency_ms,
            "confidence": confidence,
            "pause_ms": pause_ms,
        }
        if final:
            self._last_final_end_ns = end_ns
        return MappedTranscript(
            payload=payload,
            final=final,
            confidence_measured=measured,
            speaker_labelled=labelled,
        )

    # -- internals -------------------------------------------------------

    @staticmethod
    def _transcript_text(message: Mapping[str, Any]) -> str:
        metadata = message.get("metadata") or {}
        text = metadata.get("transcript")
        if isinstance(text, str) and text.strip():
            return text.strip()
        # Fall back to reassembling words; punctuation results must not get a
        # leading space, which is why this is not a plain " ".join.
        parts: list[str] = []
        for result in message.get("results") or []:
            if not isinstance(result, Mapping):
                continue
            alternatives = result.get("alternatives") or []
            if not alternatives or not isinstance(alternatives[0], Mapping):
                continue
            content = alternatives[0].get("content")
            if not isinstance(content, str) or not content:
                continue
            if parts and result.get("type") != "punctuation":
                parts.append(" ")
            parts.append(content)
        return "".join(parts).strip()

    @staticmethod
    def _confidence(results: list[Mapping[str, Any]]) -> tuple[float, bool]:
        scores: list[float] = []
        for result in results:
            if result.get("type") == "punctuation":
                continue
            alternatives = result.get("alternatives") or []
            if not alternatives or not isinstance(alternatives[0], Mapping):
                continue
            raw = alternatives[0].get("confidence")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            scores.append(min(1.0, max(0.0, float(raw))))
        if not scores:
            return 1.0, False
        return sum(scores) / len(scores), True

    def _speaker(self, results: list[Mapping[str, Any]]) -> tuple[str, bool]:
        labels: Counter[str] = Counter()
        for result in results:
            alternatives = result.get("alternatives") or []
            if not alternatives or not isinstance(alternatives[0], Mapping):
                continue
            speaker = alternatives[0].get("speaker")
            # "UU" is Speechmatics' explicit "unknown speaker" label; treating
            # it as an identity would create a phantom speaker in the registry.
            if isinstance(speaker, str) and speaker.strip() and speaker != "UU":
                labels[speaker.strip()] += 1
        if not labels:
            return self.default_speaker_id, False
        return labels.most_common(1)[0][0], True

    def _language(self, message: Mapping[str, Any],
                  results: list[Mapping[str, Any]]) -> str:
        metadata = message.get("metadata") or {}
        values: list[Any] = [
            message.get("language"),
            metadata.get("language"),
        ]
        for result in results:
            alternatives = result.get("alternatives") or []
            if alternatives and isinstance(alternatives[0], Mapping):
                values.append(alternatives[0].get("language"))
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip().lower().replace("_", "-")
        return self.language


def _first_float(*values: Any, default: float) -> float:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return float(value)
    return default
