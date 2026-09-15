from __future__ import annotations

import json

import pytest

from omni_q import AuthorityResolver, VoiceCapability, VoiceRuntime
from omni_q.mutation import RuntimeMutator
from integrations.speechmatics.mapper import MappedTranscript
from integrations.speechmatics.response import (
    SpeechResponse,
    SpeechResponseRenderer,
    SpeechmaticsTTS,
    SystemAudioPlayer,
    VoiceTransportError,
)
from integrations.speechmatics.transport import VoiceSink


def mapped(seq: int, text: str, *, final: bool = True,
           language: str = "en") -> MappedTranscript:
    return MappedTranscript(
        payload={
            "session_id": "session_01",
            "org_id": "org_a",
            "speaker_id": "S1",
            "text": text,
            "language": language,
            "t_start_ns": seq * 1_000_000_000,
            "t_end_ns": seq * 1_000_000_000 + 500_000_000,
            "sequence": seq,
            "confidence": 0.94,
            "latency_ms": 12.0,
        },
        final=final,
        confidence_measured=True,
        speaker_labelled=True,
    )


class Response:
    status = 200

    def __init__(self, audio: bytes = b"RIFF-test-audio") -> None:
        self.audio = audio

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def read(self) -> bytes:
        return self.audio


class Player:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, object]] = []

    def play(self, audio: bytes, receipt: object) -> None:
        self.calls.append((audio, receipt))


def tts(player: Player, expected_text: str = "Committed 1 change.",
        response: Response | None = None) -> SpeechmaticsTTS:
    response = response or Response()

    def opener(request, timeout):
        assert request.full_url.endswith("sarah?output_format=wav_16000")
        assert request.get_header("Authorization") == "Bearer test-key"
        assert json.loads(request.data) == {"text": expected_text}
        assert timeout == 3.0
        return response

    return SpeechmaticsTTS("test-key", player=player, timeout_s=3,
                           opener=opener)


def operator_runtime() -> VoiceRuntime:
    from omni_q import build_mock_engine

    runtime = VoiceRuntime(
        "session_01",
        "org_a",
        authority=AuthorityResolver({
            "operator": [
                VoiceCapability.GRAPH_MUTATION.value,
                VoiceCapability.COMMAND.value,
            ]
        }),
        mutator=RuntimeMutator(build_mock_engine()),
    )

    def grant(*, event, **_ignored) -> None:
        runtime.registry.set_authority(
            event.speaker_id,
            "operator",
            [VoiceCapability.GRAPH_MUTATION.value,
             VoiceCapability.COMMAND.value],
        )

    runtime.on("on_final_speech", grant)
    return runtime


def test_sink_speaks_only_from_committed_runtime_result() -> None:
    from omni_q import SpeechmaticsRealtimeAdapter

    runtime = operator_runtime()
    player = Player()
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime),
                     speech_output=tts(player))

    result = sink.deliver(mapped(1, "don't touch the red connector"))

    assert result.status == "committed"
    assert len(player.calls) == 1
    assert sink.outputs[0].session_id == "session_01"
    assert sink.outputs[0].audio_bytes == len(b"RIFF-test-audio")
    assert [event.kind for event in runtime.bus.log
            if event.kind.startswith("voice.response")] == [
                "voice.response.started",
                "voice.response.sent",
                "voice.response.stopped",
            ]


def test_addressed_dialogue_is_spoken_without_a_mutation_result() -> None:
    from omni_q import SpeechmaticsRealtimeAdapter

    runtime = VoiceRuntime(
        "session_01",
        "org_a",
        dialogue_handler=lambda **_kwargs: {
            "text": "The left arm is excluded by the observed state.",
            "backend": "test-dialogue",
        },
    )
    player = Player()
    sink = VoiceSink(
        SpeechmaticsRealtimeAdapter(runtime),
        speech_output=tts(player, "The left arm is excluded by the observed state"),
    )

    result = sink.deliver(mapped(1, "Omni, why are you using only one arm?"))

    assert result.status == "answered"
    assert result.mutation is None
    assert len(player.calls) == 1
    assert sink.outputs[0].status == "sent"
    assert [event.kind for event in runtime.bus.log
            if event.kind.startswith("voice.response")] == [
                "voice.response.started",
                "voice.response.sent",
                "voice.response.stopped",
            ]


def test_committed_multilingual_constraint_gets_bounded_localized_response() -> None:
    from omni_q import SpeechmaticsRealtimeAdapter

    runtime = operator_runtime()
    player = Player()
    sink = VoiceSink(
        SpeechmaticsRealtimeAdapter(runtime),
        speech_output=tts(
            player,
            "Brazo izquierdo no disponible. Replanificando con el derecho.",
        ),
    )

    result = sink.deliver(
        mapped(1, "No uses más el brazo izquierdo", language="es")
    )

    assert result.status == "committed"
    assert result.claim.original_text == "No uses más el brazo izquierdo"
    assert result.claim.canonical_text == "don't use the left arm anymore"
    assert result.claim.language == "es"
    assert sink.outputs[0].language == "es"


def test_partial_and_observation_do_not_speak_or_invent_execution() -> None:
    from omni_q import SpeechmaticsRealtimeAdapter

    runtime = VoiceRuntime("session_01", "org_a")
    player = Player()
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime),
                     speech_output=tts(player))

    partial = sink.deliver(mapped(1, "set the", final=False))
    observation = sink.deliver(mapped(2, "thanks"))

    assert not hasattr(partial, "status")
    assert observation.status == "observed"
    assert player.calls == []
    assert not any(event.kind == "voice.response.sent"
                   for event in runtime.bus.log)


def test_denied_and_interrupt_responses_are_derived_from_authority_result() -> None:
    from omni_q import SpeechmaticsRealtimeAdapter

    runtime = VoiceRuntime("session_01", "org_a")
    renderer = SpeechResponseRenderer()
    adapter = SpeechmaticsRealtimeAdapter(runtime)
    denied = adapter.on_final(mapped(1, "keep everything local").payload)
    denied_response = renderer.render(denied)
    assert denied.status == "denied"
    assert denied_response and "not authorized" in denied_response.text

    interrupt = adapter.on_final(mapped(2, "stop").payload)
    interrupt_response = renderer.render(interrupt)
    assert interrupt.status == "interrupt_denied"
    assert interrupt_response and "not authorized" in interrupt_response.text


def test_tts_transport_rejects_bad_response_without_playback() -> None:
    player = Player()
    client = SpeechmaticsTTS("test-key", player=player,
                             opener=lambda *_args, **_kwargs: Response(b""))
    response = SpeechResponse("response_1", "session_01", "org_a", "claim_1",
                              "committed", "The request was committed.")
    with pytest.raises(VoiceTransportError, match="invalid response"):
        client.speak(response)
    assert player.calls == []


def test_default_tts_player_is_a_real_system_player() -> None:
    client = SpeechmaticsTTS("test-key",
                             opener=lambda *_args, **_kwargs: Response())
    assert isinstance(client.player, SystemAudioPlayer)


def test_system_player_rejects_raw_pcm_without_speaking() -> None:
    player = SystemAudioPlayer()
    receipt = type("Receipt", (), {"output_format": "pcm_16000"})()
    with pytest.raises(VoiceTransportError, match="wav_16000"):
        player.play(b"audio", receipt)


def test_renderer_rejects_unscoped_omni_result() -> None:
    with pytest.raises(VoiceTransportError, match="session_id"):
        SpeechResponseRenderer().render({"ok": True, "status": "completed"})


def test_renderer_preserves_explicit_downstream_receipt_message() -> None:
    result = {
        "session_id": "session_01",
        "org_id": "org_a",
        "claim_id": "claim_1",
        "status": "completed",
        "ok": True,
        "detail": {"message": "Placed the cup in table.N."},
    }
    response = SpeechResponseRenderer().render(result)
    assert response and response.text == "Placed the cup in table.N"


def test_response_failure_is_recorded_without_replaying_mutation() -> None:
    from omni_q import SpeechmaticsRealtimeAdapter

    runtime = operator_runtime()
    sink = VoiceSink(
        SpeechmaticsRealtimeAdapter(runtime),
        speech_output=SpeechmaticsTTS(
            "test-key",
            player=Player(),
            opener=lambda *_args, **_kwargs: Response(b""),
        ),
    )

    result = sink.deliver(mapped(1, "don't touch the red connector"))

    assert result.status == "committed"
    assert len(sink.response_errors) == 1
    assert sink.stats()["responses_sent"] == 0
    assert sink.stats()["response_failures"] == 1
    assert [event.kind for event in runtime.bus.log
            if event.kind.startswith("voice.response")] == [
                "voice.response.started",
                "voice.response.failed",
                "voice.response.stopped",
            ]
