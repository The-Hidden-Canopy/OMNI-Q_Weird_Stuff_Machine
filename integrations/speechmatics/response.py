"""Result-driven Speechmatics TTS output for the existing transport sink."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import sys
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from omni_q.voice import VoiceDispatchResult


class VoiceTransportError(RuntimeError):
    """A response could not be rendered or delivered."""


class AudioPlayer(Protocol):
    def play(self, audio: bytes, receipt: "SpeechOutputReceipt") -> None:
        ...


class SystemAudioPlayer:
    """Play Speechmatics' default WAV response on the local Windows device."""

    def play(self, audio: bytes, receipt: "SpeechOutputReceipt") -> None:
        if receipt.output_format != "wav_16000":
            raise VoiceTransportError("SystemAudioPlayer requires wav_16000")
        if not isinstance(audio, bytes) or not audio.startswith(b"RIFF"):
            raise VoiceTransportError("SystemAudioPlayer requires a WAV payload")
        if sys.platform != "win32":
            raise VoiceTransportError(
                "SystemAudioPlayer is Windows-only; inject an AudioPlayer"
            )
        try:
            import winsound
            winsound.PlaySound(audio, winsound.SND_MEMORY | winsound.SND_SYNC)
        except Exception as exc:  # noqa: BLE001 - report device/API failure
            raise VoiceTransportError(f"system audio playback failed: {exc}") from exc


@dataclass(frozen=True)
class SpeechResponse:
    response_id: str
    session_id: str
    org_id: str
    claim_id: str
    source_status: str
    text: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class SpeechOutputReceipt:
    response_id: str
    session_id: str
    org_id: str
    voice: str
    output_format: str
    text_sha256: str
    audio_bytes: int
    status: str = "sent"

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class SpeechResponseRenderer:
    """Speak only state present in the real VoiceRuntime/OMNI result."""

    _SPACE = re.compile(r"\s+")

    def __init__(self, *, max_chars: int = 320) -> None:
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        self.max_chars = max_chars

    def render(self, result: Any) -> SpeechResponse | None:
        if isinstance(result, VoiceDispatchResult):
            text = self._voice_text(result)
            if text is None:
                return None
            claim = result.claim
            return self._response(claim.session_id, claim.org_id, claim.claim_id,
                                  result.status, text)
        payload = self._mapping(result)
        if payload is None:
            raise VoiceTransportError("OMNI result must expose as_dict() or be a mapping")
        text = self._mapping_text(payload)
        if text is None:
            return None
        session = self._text(payload.get("session_id"), "session_id")
        org = self._text(payload.get("org_id"), "org_id")
        claim = self._text(payload.get("claim_id", payload.get("receipt_id", "result")), "claim_id")
        return self._response(session, org, claim, str(payload.get("status", "result")), text)

    def _voice_text(self, result: VoiceDispatchResult) -> str | None:
        if result.status == "interrupted":
            return f"{(result.interruption.action if result.interruption else 'STOP').title()} requested."
        if result.status == "interrupt_denied":
            return "The stop request was not authorized."
        if result.status == "denied":
            return "That request was not authorized."
        if result.status == "authorized":
            return "The request was authorized but was not committed."
        if result.status != "committed":
            return None
        mutation = self._mapping(result.mutation)
        if mutation:
            explicit = mutation.get("message") or mutation.get("summary")
            detail = self._mapping(mutation.get("detail"))
            explicit = explicit or (detail.get("message") if detail else None)
            if isinstance(explicit, str) and explicit.strip():
                return self._fragment(explicit)
            if isinstance(mutation.get("applied"), list):
                count = len(mutation["applied"])
                return f"Committed {count} change{'s' if count != 1 else ''}."
        return "The request was committed."

    def _mapping_text(self, payload: Mapping[str, Any]) -> str | None:
        status = str(payload.get("status", "result"))
        if status in {"denied", "rejected", "failed", "error"} or payload.get("ok") is False:
            reason = payload.get("reason") or payload.get("error")
            return f"The request failed{': ' + self._fragment(reason) if reason else '.'}"
        if status in {"committed", "completed", "success"} or payload.get("ok") is True:
            detail = self._mapping(payload.get("detail"))
            message = payload.get("message") or payload.get("summary")
            message = message or (detail.get("message") if detail else None)
            return self._fragment(message) if message else "The request completed."
        if status in {"authorized", "pending"}:
            return "The request was authorized but has not completed."
        return None

    def _response(self, session: str, org: str, claim: str, status: str, text: str) -> SpeechResponse:
        text = self._bounded(text)
        digest = hashlib.sha256(f"{claim}|{status}|{text}".encode()).hexdigest()[:20]
        return SpeechResponse(f"voice_response_{digest}", session, org, claim, status, text)

    def _bounded(self, text: Any) -> str:
        if not isinstance(text, str) or not text.strip():
            raise VoiceTransportError("rendered response is empty")
        return self._SPACE.sub(" ", text).strip()[: self.max_chars]

    @classmethod
    def _fragment(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return ""
        return cls._SPACE.sub(" ", value).strip(" .")[:160]

    @staticmethod
    def _mapping(value: Any) -> Mapping[str, Any] | None:
        if isinstance(value, Mapping):
            return value
        if value is not None and hasattr(value, "as_dict"):
            converted = value.as_dict()
            return converted if isinstance(converted, Mapping) else None
        return None

    @staticmethod
    def _text(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise VoiceTransportError(f"OMNI result requires {name}")
        return value.strip()


class SpeechmaticsTTS:
    """Speechmatics preview TTS with an injected or native audio player."""

    def __init__(self, api_key: str, *, voice: str = "sarah",
                 output_format: str = "wav_16000",
                 endpoint: str = "https://preview.tts.speechmatics.com/generate",
                 timeout_s: float = 20.0, player: AudioPlayer | None = None,
                 opener: Callable[..., Any] = urlopen) -> None:
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
        self._api_key = api_key
        self.voice, self.output_format = voice.strip(), output_format
        self.endpoint, self.timeout_s = endpoint.rstrip("/"), float(timeout_s)
        self.player, self.opener = player or SystemAudioPlayer(), opener

    def speak(self, response: SpeechResponse) -> SpeechOutputReceipt:
        if not isinstance(response, SpeechResponse):
            raise TypeError("speak expects SpeechResponse")
        url = f"{self.endpoint}/{self.voice}?{urlencode({'output_format': self.output_format})}"
        request = Request(url, data=json.dumps({"text": response.text}).encode(),
                          headers={"Authorization": f"Bearer {self._api_key}",
                                   "Content-Type": "application/json"}, method="POST")
        try:
            with self.opener(request, timeout=self.timeout_s) as result:
                audio, status = result.read(), getattr(result, "status", 200)
        except Exception as exc:  # noqa: BLE001 - transport boundary
            raise VoiceTransportError(f"Speechmatics TTS request failed: {exc}") from exc
        if status < 200 or status >= 300 or not isinstance(audio, bytes) or not audio:
            raise VoiceTransportError(f"Speechmatics TTS returned invalid response status={status}")
        receipt = SpeechOutputReceipt(
            response.response_id, response.session_id, response.org_id, self.voice,
            self.output_format, hashlib.sha256(response.text.encode()).hexdigest(), len(audio),
        )
        try:
            self.player.play(audio, receipt)
        except VoiceTransportError:
            raise
        except Exception as exc:  # noqa: BLE001 - audio device boundary
            raise VoiceTransportError(f"audio playback failed: {exc}") from exc
        return receipt


__all__ = [
    "AudioPlayer", "SpeechOutputReceipt", "SpeechResponse", "SpeechResponseRenderer",
    "SpeechmaticsTTS", "SystemAudioPlayer", "VoiceTransportError",
]
