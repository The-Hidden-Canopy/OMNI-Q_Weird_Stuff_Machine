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

from omni_q.language import normalize_language
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
    language: str = "en"

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
    language: str = "en"

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class SpeechResponseRenderer:
    """Speak only state present in the real VoiceRuntime/OMNI result."""

    _SPACE = re.compile(r"\s+")

    _MESSAGES = {
        "en": {
            "stop": "Stopping.",
            "hold": "Holding.",
            "backoff": "Backing off.",
            "interrupt_denied": "The stop request was not authorized.",
            "denied": "That request was not authorized.",
            "failed": "The request failed.",
            "authorized": "The request was authorized but was not committed.",
            "committed": "The request was committed.",
            "dialogue_unavailable": "I can't answer that from my current state.",
            "scope_unavailable": "I can't answer that from the current workspace.",
            "left_arm_removed": "Left arm unavailable. Replanning with the right arm.",
            "right_arm_removed": "Right arm unavailable. Replanning with the left arm.",
            "set_table": "The table-setting request was committed.",
            "keep_local": "Inference will stay on-device.",
            "slow_down": "Slowing down.",
            "speed_up": "Speeding up.",
        },
        "es": {
            "stop": "Deteniendo.",
            "hold": "Manteniendo la posición.",
            "backoff": "Retrocediendo.",
            "interrupt_denied": "La solicitud de parada no fue autorizada.",
            "denied": "La solicitud no fue autorizada.",
            "failed": "La solicitud falló.",
            "authorized": "La solicitud fue autorizada pero no se aplicó.",
            "committed": "La solicitud fue aplicada.",
            "dialogue_unavailable": "No puedo responder con mi estado actual.",
            "scope_unavailable": "No puedo responder desde este espacio de trabajo.",
            "left_arm_removed": "Brazo izquierdo no disponible. Replanificando con el derecho.",
            "right_arm_removed": "Brazo derecho no disponible. Replanificando con el izquierdo.",
            "set_table": "La solicitud de preparar la mesa fue aplicada.",
            "keep_local": "La inferencia permanecerá en el dispositivo.",
            "slow_down": "Reduciendo la velocidad.",
            "speed_up": "Aumentando la velocidad.",
        },
        "fr": {
            "stop": "Arrêt.",
            "hold": "Maintien de la position.",
            "backoff": "Recul.",
            "interrupt_denied": "La demande d'arrêt n'a pas été autorisée.",
            "denied": "La demande n'a pas été autorisée.",
            "failed": "La demande a échoué.",
            "authorized": "La demande a été autorisée mais n'a pas été appliquée.",
            "committed": "La demande a été appliquée.",
            "dialogue_unavailable": "Je ne peux pas répondre avec mon état actuel.",
            "scope_unavailable": "Je ne peux pas répondre depuis cet espace de travail.",
            "left_arm_removed": "Bras gauche indisponible. Replanification avec le bras droit.",
            "right_arm_removed": "Bras droit indisponible. Replanification avec le bras gauche.",
            "set_table": "La demande de dresser la table a été appliquée.",
            "keep_local": "L'inférence restera sur l'appareil.",
            "slow_down": "Je ralentis.",
            "speed_up": "J'accélère.",
        },
    }

    def __init__(self, *, max_chars: int = 320) -> None:
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        self.max_chars = max_chars

    def render(self, result: Any, *, language: str | None = None) -> SpeechResponse | None:
        if isinstance(result, VoiceDispatchResult):
            response_language = self._language(language or result.claim.language)
            text = self._voice_text(result, response_language)
            if text is None:
                return None
            claim = result.claim
            return self._response(claim.session_id, claim.org_id, claim.claim_id,
                                  result.status, text, response_language)
        payload = self._mapping(result)
        if payload is None:
            raise VoiceTransportError("OMNI result must expose as_dict() or be a mapping")
        response_language = self._language(
            language or payload.get("language", "en")
        )
        text = self._mapping_text(payload, response_language)
        if text is None:
            return None
        session = self._text(payload.get("session_id"), "session_id")
        org = self._text(payload.get("org_id"), "org_id")
        claim = self._text(payload.get("claim_id", payload.get("receipt_id", "result")), "claim_id")
        return self._response(session, org, claim, str(payload.get("status", "result")),
                              text, response_language)

    def _voice_text(self, result: VoiceDispatchResult, language: str) -> str | None:
        if result.status == "answered":
            return self._fragment(result.response_text)
        if result.status == "dialogue_failed":
            scope_failure = (
                result.response_backend == "unavailable"
                and result.response_text
                and "workspace" in result.response_text.lower()
            )
            key = "scope_unavailable" if scope_failure else "dialogue_unavailable"
            return self._message(key, language)
        if result.status == "interrupted":
            action = (result.interruption.action if result.interruption else "STOP").lower()
            return self._message(action.lower(), language)
        if result.status == "interrupt_denied":
            return self._message("interrupt_denied", language)
        if result.status == "denied":
            return self._message("denied", language)
        if result.status == "authorized":
            return self._message("authorized", language)
        if result.status != "committed":
            return None
        key = self._commit_message_key(result)
        if key is not None:
            return self._message(key, language)
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
        return self._message("committed", language)

    def _mapping_text(self, payload: Mapping[str, Any], language: str) -> str | None:
        status = str(payload.get("status", "result"))
        if status in {"denied", "rejected", "failed", "error"} or payload.get("ok") is False:
            reason = payload.get("reason") or payload.get("error")
            return f"{self._message('failed', language)}{': ' + self._fragment(reason) if reason else ''}"
        if status in {"committed", "completed", "success"} or payload.get("ok") is True:
            detail = self._mapping(payload.get("detail"))
            message = payload.get("message") or payload.get("summary")
            message = message or (detail.get("message") if detail else None)
            return self._fragment(message) if message else self._message("committed", language)
        if status in {"authorized", "pending"}:
            return self._message("authorized", language)
        return None

    def _response(self, session: str, org: str, claim: str, status: str,
                  text: str, language: str) -> SpeechResponse:
        text = self._bounded(text)
        digest = hashlib.sha256(
            f"{claim}|{status}|{language}|{text}".encode()
        ).hexdigest()[:20]
        return SpeechResponse(
            f"voice_response_{digest}", session, org, claim, status, text, language
        )

    @classmethod
    def _language(cls, value: Any) -> str:
        try:
            return normalize_language(value)
        except ValueError:
            return "en"

    @classmethod
    def _message(cls, key: str, language: str) -> str:
        base = language.split("-", 1)[0]
        return cls._MESSAGES.get(base, cls._MESSAGES["en"]).get(
            key, cls._MESSAGES["en"].get(key, "The request completed.")
        )

    @classmethod
    def _commit_message_key(cls, result: VoiceDispatchResult) -> str | None:
        candidate = result.candidate
        args = candidate.args if candidate is not None else {}
        constraints = args.get("constraints", []) if isinstance(args, Mapping) else []
        for constraint in constraints:
            if not isinstance(constraint, (list, tuple)) or len(constraint) != 2:
                continue
            kind, value = constraint
            if kind == "prefer_arm" and value == "right":
                return "left_arm_removed"
            if kind == "prefer_arm" and value == "left":
                return "right_arm_removed"
            if kind == "keep_local":
                return "keep_local"
            if kind == "style" and value == "slow":
                return "slow_down"
            if kind == "style" and value == "fast":
                return "speed_up"
        if isinstance(args, Mapping):
            if args.get("goal") == "set the table":
                return "set_table"
        return None

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
            language=response.language,
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
