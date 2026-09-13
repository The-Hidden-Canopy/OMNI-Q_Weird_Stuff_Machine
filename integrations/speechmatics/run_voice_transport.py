"""Speechmatics voice transport glue for the existing OMNI-Q voice boundary.

The transport owns provider I/O only. ``VoiceRuntime`` remains the source of
truth for authority, interruption, mutation, and execution outcomes. A result
is rendered after ``SpeechmaticsRealtimeAdapter.on_final`` returns; no response
is inferred from partial speech or from the transcript alone.

The TTS client uses Speechmatics' preview HTTP API and deliberately accepts an
injected audio player. That keeps microphone/audio-device dependencies out of
the governed core while making the real outbound response path testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import sys
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from omni_q.voice import SpeechmaticsRealtimeAdapter, VoiceDispatchResult, VoiceRuntime


class VoiceTransportError(RuntimeError):
    """A response could not be rendered or delivered."""


class AudioPlayer(Protocol):
    """Play or forward one complete audio payload."""

    def play(self, audio: bytes, receipt: "SpeechOutputReceipt") -> None:
        ...


class SystemAudioPlayer:
    """Play Speechmatics' default WAV response on the local machine.

    Windows uses the standard-library ``winsound`` memory playback path, so no
    temporary audio file or third-party dependency is required. Other
    platforms fail with an actionable message rather than silently discarding
    synthesized audio; callers can inject a platform-specific player.
    """

    def play(self, audio: bytes, receipt: "SpeechOutputReceipt") -> None:
        if receipt.output_format != "wav_16000":
            raise VoiceTransportError(
                "SystemAudioPlayer requires Speechmatics output_format=wav_16000"
            )
        if not isinstance(audio, bytes) or not audio.startswith(b"RIFF"):
            raise VoiceTransportError("SystemAudioPlayer requires a WAV payload")
        if sys.platform != "win32":
            raise VoiceTransportError(
                "SystemAudioPlayer is Windows-only; inject an AudioPlayer on this platform"
            )
        try:
            import winsound
            winsound.PlaySound(audio, winsound.SND_MEMORY | winsound.SND_SYNC)
        except Exception as exc:  # noqa: BLE001 - report device/API failure
            raise VoiceTransportError(f"system audio playback failed: {exc}") from exc


@dataclass(frozen=True)
class SpeechResponse:
    """The response text derived from an actual OMNI result."""

    response_id: str
    session_id: str
    org_id: str
    claim_id: str
    source_status: str
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "session_id": self.session_id,
            "org_id": self.org_id,
            "claim_id": self.claim_id,
            "source_status": self.source_status,
            "text": self.text,
        }


@dataclass(frozen=True)
class SpeechOutputReceipt:
    """Evidence that a rendered response was synthesized and forwarded."""

    response_id: str
    session_id: str
    org_id: str
    voice: str
    output_format: str
    text_sha256: str
    audio_bytes: int
    status: str = "sent"

    def as_dict(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "session_id": self.session_id,
            "org_id": self.org_id,
            "voice": self.voice,
            "output_format": self.output_format,
            "text_sha256": self.text_sha256,
            "audio_bytes": self.audio_bytes,
            "status": self.status,
        }


class VoiceResponseRenderer:
    """Render only states present in a VoiceRuntime or OMNI result."""

    _SPACE = re.compile(r"\s+")

    def __init__(self, *, max_chars: int = 320) -> None:
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        self.max_chars = max_chars

    def render(self, result: Any) -> SpeechResponse | None:
        if isinstance(result, VoiceDispatchResult):
            claim = result.claim
            status = result.status
            text = self._voice_text(result)
            if text is None:
                return None
            return SpeechResponse(
                response_id=self._id(claim.claim_id, status, text),
                session_id=claim.session_id,
                org_id=claim.org_id,
                claim_id=claim.claim_id,
                source_status=status,
                text=self._bounded(text),
            )

        payload = self._mapping(result)
        if payload is None:
            raise VoiceTransportError("OMNI result must expose as_dict() or be a mapping")
        text = self._result_text(payload)
        if text is None:
            return None
        session_id = self._required_text(payload.get("session_id"), "session_id")
        org_id = self._required_text(payload.get("org_id"), "org_id")
        claim_id = self._required_text(
            payload.get("claim_id", payload.get("receipt_id", "omni-result")),
            "claim_id",
        )
        status = self._required_text(payload.get("status", "result"), "status")
        return SpeechResponse(
            response_id=self._id(claim_id, status, text),
            session_id=session_id,
            org_id=org_id,
            claim_id=claim_id,
            source_status=status,
            text=self._bounded(text),
        )

    def _voice_text(self, result: VoiceDispatchResult) -> str | None:
        status = result.status
        if status == "interrupted":
            action = result.interruption.action if result.interruption else "STOP"
            return f"{action.title()} requested."
        if status == "interrupt_denied":
            return "The stop request was not authorized."
        if status == "denied":
            return "That request was not authorized."
        if status == "authorized":
            return "The request was authorized but was not committed."
        if status == "committed":
            mutation = self._mapping(result.mutation)
            if mutation:
                explicit = mutation.get("message") or mutation.get("summary")
                detail = self._mapping(mutation.get("detail"))
                explicit = explicit or (detail.get("message") if detail else None)
                if isinstance(explicit, str) and explicit.strip():
                    return self._safe_fragment(explicit)
            if mutation and isinstance(mutation.get("applied"), list):
                count = len(mutation["applied"])
                return f"Committed {count} change{'s' if count != 1 else ''}."
            return "The request was committed."
        # Observations and candidate intents do not warrant an audio claim of
        # execution. The caller may still inspect the returned result.
        return None

    def _result_text(self, payload: Mapping[str, Any]) -> str | None:
        status = str(payload.get("status", "result"))
        if status in {"denied", "rejected", "failed", "error"} or payload.get("ok") is False:
            reason = payload.get("reason") or payload.get("error")
            return f"The request failed{': ' + self._safe_fragment(reason) if reason else '.'}"
        if status in {"committed", "completed", "success"} or payload.get("ok") is True:
            detail = self._mapping(payload.get("detail"))
            message = payload.get("message") or payload.get("summary")
            message = message or (detail.get("message") if detail else None)
            return self._safe_fragment(message) if message else "The request completed."
        if status in {"authorized", "pending"}:
            return "The request was authorized but has not completed."
        return None

    @staticmethod
    def _mapping(value: Any) -> Mapping[str, Any] | None:
        if isinstance(value, Mapping):
            return value
        if value is not None and hasattr(value, "as_dict"):
            converted = value.as_dict()
            return converted if isinstance(converted, Mapping) else None
        return None

    @staticmethod
    def _required_text(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise VoiceTransportError(f"OMNI result requires {name}")
        return value.strip()

    @classmethod
    def _safe_fragment(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return ""
        return cls._SPACE.sub(" ", value).strip(" .")[:160]

    @staticmethod
    def _id(claim_id: str, status: str, text: str) -> str:
        digest = hashlib.sha256(f"{claim_id}|{status}|{text}".encode()).hexdigest()[:20]
        return f"voice_response_{digest}"

    def _bounded(self, text: str) -> str:
        normalized = self._SPACE.sub(" ", text).strip()
        if not normalized:
            raise VoiceTransportError("rendered response is empty")
        return normalized[: self.max_chars]


class SpeechmaticsTTS:
    """Speechmatics text-to-speech output with an injected playback sink."""

    def __init__(
        self,
        api_key: str,
        *,
        voice: str = "sarah",
        output_format: str = "wav_16000",
        endpoint: str = "https://preview.tts.speechmatics.com/generate",
        timeout_s: float = 20.0,
        player: AudioPlayer | None = None,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be non-empty")
        if not isinstance(voice, str) or not voice.strip() or "/" in voice:
            raise ValueError("voice must be a safe voice id")
        if output_format not in {"wav_16000", "pcm_16000"}:
            raise ValueError("unsupported Speechmatics output format")
        if not isinstance(endpoint, str) or not endpoint.startswith("https://"):
            raise ValueError("endpoint must use https")
        if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not callable(opener):
            raise ValueError("opener must be callable")
        self._api_key = api_key
        self.voice = voice.strip()
        self.output_format = output_format
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.player = player if player is not None else SystemAudioPlayer()
        self.opener = opener

    def speak(self, response: SpeechResponse) -> SpeechOutputReceipt:
        if not isinstance(response, SpeechResponse):
            raise TypeError("speak expects SpeechResponse")
        url = f"{self.endpoint}/{self.voice}?{urlencode({'output_format': self.output_format})}"
        request = Request(
            url,
            data=json.dumps({"text": response.text}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self.opener(request, timeout=self.timeout_s) as result:
                audio = result.read()
                status = getattr(result, "status", 200)
        except Exception as exc:  # noqa: BLE001 - transport boundary reports it
            raise VoiceTransportError(f"Speechmatics TTS request failed: {exc}") from exc
        if status < 200 or status >= 300 or not isinstance(audio, bytes) or not audio:
            raise VoiceTransportError(f"Speechmatics TTS returned invalid response status={status}")
        receipt = SpeechOutputReceipt(
            response_id=response.response_id,
            session_id=response.session_id,
            org_id=response.org_id,
            voice=self.voice,
            output_format=self.output_format,
            text_sha256=hashlib.sha256(response.text.encode()).hexdigest(),
            audio_bytes=len(audio),
        )
        if self.player is not None:
            self.player.play(audio, receipt)
        return receipt


class VoiceSink:
    """Close the existing adapter/result boundary with optional outbound TTS."""

    def __init__(
        self,
        adapter: SpeechmaticsRealtimeAdapter,
        *,
        on_result: Callable[[VoiceDispatchResult], Any] | None = None,
        response_renderer: VoiceResponseRenderer | None = None,
        speech_output: SpeechmaticsTTS | None = None,
    ) -> None:
        if not isinstance(adapter, SpeechmaticsRealtimeAdapter):
            raise TypeError("adapter must be SpeechmaticsRealtimeAdapter")
        if on_result is not None and not callable(on_result):
            raise TypeError("on_result must be callable")
        self.adapter = adapter
        self.on_result = on_result
        self.response_renderer = response_renderer or VoiceResponseRenderer()
        self.speech_output = speech_output
        self.outputs: list[SpeechOutputReceipt] = []

    @property
    def runtime(self) -> VoiceRuntime:
        return self.adapter.runtime

    def on_partial(self, payload: Mapping[str, Any]):
        return self.adapter.on_partial(payload)

    def on_final(self, payload: Mapping[str, Any], **kwargs: Any) -> VoiceDispatchResult:
        result = self.adapter.on_final(payload, **kwargs)
        if self.on_result is not None:
            self.on_result(result)
        response = self.response_renderer.render(result)
        if response is None or self.speech_output is None:
            return result
        try:
            receipt = self.speech_output.speak(response)
        except VoiceTransportError as exc:
            self.runtime.bus.publish(
                "voice.response.failed", source="speechmatics",
                response=response.as_dict(), error=str(exc),
            )
            raise
        self.outputs.append(receipt)
        self.runtime.bus.publish(
            "voice.response.sent", source="speechmatics",
            response=response.as_dict(), output=receipt.as_dict(),
        )
        return result


__all__ = [
    "AudioPlayer",
    "SpeechOutputReceipt",
    "SpeechResponse",
    "SpeechmaticsTTS",
    "SystemAudioPlayer",
    "VoiceResponseRenderer",
    "VoiceSink",
    "VoiceTransportError",
]


def speak_smoke(text: str, *, api_key: str | None = None) -> SpeechOutputReceipt:
    """Synthesize and play one explicit transport smoke phrase.

    This is a transport check, not an OMNI authorization path. Production
    responses must enter through ``VoiceSink.on_final``.
    """
    if not isinstance(text, str) or not text.strip():
        raise VoiceTransportError("smoke text must be non-empty")
    key = api_key or os.environ.get("SPEECHMATICS_API_KEY")
    if not key:
        raise VoiceTransportError(
            "SPEECHMATICS_API_KEY is required; set it in the process environment"
        )
    response = SpeechResponse(
        response_id="transport_smoke",
        session_id="transport_smoke",
        org_id="transport_smoke",
        claim_id="transport_smoke",
        source_status="smoke",
        text=VoiceResponseRenderer()._bounded(text),
    )
    return SpeechmaticsTTS(key).speak(response)


if __name__ == "__main__":  # pragma: no cover - live API/device path
    import argparse

    parser = argparse.ArgumentParser(description="speak one Speechmatics TTS smoke phrase")
    parser.add_argument("--text", default="OMNI response audio is working.")
    options = parser.parse_args()
    print(speak_smoke(options.text).as_dict())
