"""Optional Speechmatics Voice SDK bridge for the OMNI-Q voice boundary.

The existing :mod:`transport` module intentionally stays on the raw
Speechmatics Realtime API.  This module is the opt-in higher-level path for
``speechmatics-voice``.  It translates Voice SDK segment events into the same
``VoiceSink`` and provider-turn seam, so authority, accumulation, response
speech, and receipts do not fork between providers.

The SDK is imported lazily.  Offline replay and the raw transport therefore
remain dependency-free, while a live Voice SDK run fails with an actionable
installation message instead of silently falling back to a different protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Mapping

from .mapper import FINAL_MESSAGE, PARTIAL_MESSAGE, TranscriptMapper
from .transport import AudioStream, SpeechmaticsError, VoiceSink

__all__ = [
    "SpeechmaticsVoiceBridge",
    "SpeechmaticsVoiceTransport",
    "SpeakerFocusRequest",
    "VoiceIdentityBinding",
    "apply_speaker_focus",
]


_SEGMENT_EVENTS = {
    "addsegment": True,
    "addpartialsegment": False,
    # Accept the lower-level names too; this keeps recorded event streams
    # portable between the Voice SDK and the existing raw transport.
    "addtranscript": True,
    "addpartialtranscript": False,
}
_CONTROL_EVENTS = {
    "recognitionstarted",
    "endofturn",
    "endofturnprediction",
    "smartturnresult",
    "error",
}


def _kind(message: Mapping[str, Any]) -> str:
    value = message.get("message", message.get("type", ""))
    return str(value).strip().lower().replace("_", "")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class SpeakerFocusRequest:
    """An explicit focus command; it is never inferred from speech."""

    speaker_ids: tuple[str, ...]
    mode: str = "retain"

    def __post_init__(self) -> None:
        if self.mode not in {"retain", "ignore"}:
            raise ValueError("speaker focus mode must be 'retain' or 'ignore'")
        if not self.speaker_ids:
            raise ValueError("speaker focus requires at least one speaker id")
        if any(not isinstance(speaker, str) or not speaker.strip()
               for speaker in self.speaker_ids):
            raise ValueError("speaker focus ids must be non-empty strings")


@dataclass(frozen=True)
class VoiceIdentityBinding:
    """Provider voiceprint binding, deliberately separate from authority."""

    label: str
    identifiers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("voice identity label must be a non-empty string")
        if not self.identifiers:
            raise ValueError("voice identity requires at least one identifier")
        if any(not isinstance(identifier, str) or not identifier.strip()
               for identifier in self.identifiers):
            raise ValueError("voice identity identifiers must be non-empty strings")


def apply_speaker_focus(client: Any, request: SpeakerFocusRequest) -> Any:
    """Apply an explicitly governed focus request to a Voice SDK client.

    This function does not grant OMNI authority to a speaker.  It only changes
    which labelled audio the provider emits; identity and authorization remain
    in ``VoiceRuntime``.
    """
    try:
        from speechmatics.voice import SpeakerFocusConfig, SpeakerFocusMode
    except ImportError as exc:  # pragma: no cover - exercised with the extra installed
        raise SpeechmaticsError(
            "speaker focus needs the optional 'speechmatics-voice' extra: "
            "pip install -e .[speechmatics-voice]"
        ) from exc

    if request.mode == "retain":
        config = SpeakerFocusConfig(
            focus_speakers=list(request.speaker_ids),
            focus_mode=SpeakerFocusMode.RETAIN,
        )
    else:
        config = SpeakerFocusConfig(
            ignore_speakers=list(request.speaker_ids),
            focus_mode=SpeakerFocusMode.IGNORE,
        )
    return client.update_diarization_config(config)


class SpeechmaticsVoiceBridge:
    """Translate Voice SDK callback payloads into the existing voice seam."""

    def __init__(self, sink: VoiceSink, mapper: TranscriptMapper) -> None:
        self.sink = sink
        self.mapper = mapper

    def handle_message(self, message: Mapping[str, Any]) -> Any:
        if not isinstance(message, Mapping):
            raise SpeechmaticsError("Speechmatics Voice SDK message must be a mapping")

        kind = _kind(message)
        if kind == "recognitionstarted":
            self.mapper.start_stream()
            return None
        if kind == "error":
            raise SpeechmaticsError(
                f"Speechmatics Voice SDK error: {message.get('reason', message)}"
            )
        if kind in {"endofturn", "endofturnprediction", "smartturnresult"}:
            return self.sink.on_provider_event(message)

        if kind in _SEGMENT_EVENTS:
            final = _SEGMENT_EVENTS[kind]
            if kind in {"addtranscript", "addpartialtranscript"}:
                mapped = self.mapper.map_message(message)
                return None if mapped is None else self.sink.deliver(mapped)

            segments = message.get("segments")
            if not isinstance(segments, list):
                segments = [message]
            result = None
            for segment in segments:
                if not isinstance(segment, Mapping):
                    continue
                mapped = self.mapper.map_message(
                    self._as_realtime_message(message, segment, final=final)
                )
                if mapped is not None:
                    result = self.sink.deliver(mapped)
            return result

        # Speaker lifecycle, diagnostics, and metrics are useful to a caller
        # but do not belong on the transcript/authority path.
        return None

    @staticmethod
    def _as_realtime_message(
        envelope: Mapping[str, Any],
        segment: Mapping[str, Any],
        *,
        final: bool,
    ) -> dict[str, Any]:
        envelope_metadata = _as_mapping(envelope.get("metadata"))
        segment_metadata = _as_mapping(segment.get("metadata"))
        start = segment_metadata.get(
            "start_time", envelope_metadata.get("start_time", 0.0)
        )
        end = segment_metadata.get(
            "end_time", envelope_metadata.get("end_time", start)
        )
        text = str(segment.get("text", "")).strip()
        alternative: dict[str, Any] = {"content": text}
        language = segment.get("language", envelope.get("language", "en"))
        if isinstance(language, str) and language.strip():
            alternative["language"] = language
        speaker = segment.get("speaker_id", segment.get("speaker"))
        if speaker is not None:
            alternative["speaker"] = speaker
        if "confidence" in segment:
            alternative["confidence"] = segment["confidence"]
        return {
            "message": FINAL_MESSAGE if final else PARTIAL_MESSAGE,
            "metadata": {
                "start_time": start,
                "end_time": end,
                "transcript": text,
                "language": language,
            },
            "results": [{
                "type": "word",
                "start_time": start,
                "end_time": end,
                "alternatives": [alternative],
            }],
        }

    def register(self, client: Any,
                 event_types: Mapping[str, Any] | None = None) -> tuple[Any, ...]:
        """Register callbacks on a Voice SDK client.

        ``event_types`` is injectable for tests and for SDK-compatible wrappers.
        With the real SDK it is obtained from ``AgentServerMessageType``.
        """
        if event_types is None:
            try:
                from speechmatics.voice import AgentServerMessageType
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise SpeechmaticsError(
                    "Voice SDK mode needs 'speechmatics-voice': "
                    "pip install -e .[speechmatics-voice]"
                ) from exc
            event_types = {
                name: getattr(AgentServerMessageType, name, None)
                for name in (
                    "RECOGNITION_STARTED",
                    "ADD_PARTIAL_SEGMENT",
                    "ADD_SEGMENT",
                    "END_OF_TURN",
                    "END_OF_TURN_PREDICTION",
                    "SMART_TURN_RESULT",
                    "ERROR",
                )
            }

        registered: list[Any] = []
        for event in event_types.values():
            if event is None:
                continue
            decorator = client.on(event)
            if not callable(decorator):
                raise SpeechmaticsError(
                    "Speechmatics Voice SDK client.on() did not return a decorator"
                )
            decorator(self.handle_message)
            registered.append(event)
        if not registered:
            raise SpeechmaticsError("no compatible Speechmatics Voice SDK events found")
        return tuple(registered)


class SpeechmaticsVoiceTransport:
    """Run one ``speechmatics-voice`` client over an existing AudioStream."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        preset: str = "smart_turn",
        client_factory: Callable[[], Any] | None = None,
        event_types: Mapping[str, Any] | None = None,
        focus_request: SpeakerFocusRequest | None = None,
        known_speakers: tuple[VoiceIdentityBinding, ...] = (),
    ) -> None:
        if not preset.strip():
            raise ValueError("preset must be non-empty")
        self.api_key = api_key
        self.preset = preset
        self.client_factory = client_factory
        self.event_types = event_types
        self.focus_request = focus_request
        self.known_speakers = tuple(known_speakers)

    def _client(self) -> Any:
        if self.client_factory is not None:
            return self.client_factory()
        try:
            from speechmatics.voice import VoiceAgentClient
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SpeechmaticsError(
                "Voice API mode needs 'speechmatics-voice[smart]': "
                "pip install -e .[speechmatics-voice]"
            ) from exc
        if not self.known_speakers:
            return VoiceAgentClient(api_key=self.api_key, preset=self.preset)

        try:
            from speechmatics.voice import (
                SpeakerIdentifier,
                VoiceAgentConfigPreset,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SpeechmaticsError(
                "known speaker bindings need the optional 'speechmatics-voice' extra"
            ) from exc

        config = VoiceAgentConfigPreset.load(self.preset)
        config.enable_diarization = True
        config.known_speakers = [
            SpeakerIdentifier(
                label=binding.label,
                speaker_identifiers=list(binding.identifiers),
            )
            for binding in self.known_speakers
        ]
        return VoiceAgentClient(api_key=self.api_key, config=config)

    async def run(
        self,
        audio: AudioStream,
        sink: VoiceSink,
        mapper: TranscriptMapper,
    ) -> dict[str, Any]:
        client = self._client()
        bridge = SpeechmaticsVoiceBridge(sink, mapper)
        registered = bridge.register(client, self.event_types)
        sent = 0
        await client.connect()
        try:
            if self.focus_request is not None:
                apply_speaker_focus(client, self.focus_request)
            async for chunk in audio.chunks():
                await client.send_audio(chunk)
                sent += 1
        finally:
            await client.disconnect()
        return {
            "provider": "speechmatics-voice",
            "preset": self.preset,
            "speaker_focus": (
                None if self.focus_request is None else {
                    "speaker_ids": list(self.focus_request.speaker_ids),
                    "mode": self.focus_request.mode,
                }
            ),
            # Labels are safe operational metadata; voiceprint identifiers are
            # intentionally never written into a receipt.
            "known_speakers": [
                binding.label for binding in self.known_speakers
            ],
            "events_registered": [str(event) for event in registered],
            "chunks_sent": sent,
        }
