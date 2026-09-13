from __future__ import annotations

import json

import pytest

from omni_q import AuthorityResolver, SpeechFinal, VoiceCapability, VoiceRuntime
from omni_q.mutation import RuntimeMutator
from integrations.speechmatics.run_voice_transport import (
    SpeechResponse,
    SpeechmaticsTTS,
    SystemAudioPlayer,
    VoiceResponseRenderer,
    VoiceSink,
    VoiceTransportError,
)


def final(seq: int, text: str) -> SpeechFinal:
    start = seq * 1_000_000_000
    return SpeechFinal("session_01", "org_a", "speaker_01", text,
                       start, start + 500_000_000, seq, confidence=.94)


def operator(runtime: VoiceRuntime, event: SpeechFinal) -> None:
    runtime.registry.observe(event)
    runtime.registry.set_authority(
        event.speaker_id, "operator",
        [VoiceCapability.GRAPH_MUTATION.value, VoiceCapability.COMMAND.value,
         VoiceCapability.INTERRUPT.value],
    )


class Response:
    status = 200

    def __init__(self, audio=b"RIFF-test-audio"):
        self.audio = audio

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.audio


class Player:
    def __init__(self):
        self.calls = []

    def play(self, audio, receipt):
        self.calls.append((audio, receipt))


def tts(player, expected_text="Committed 1 change.", response=Response()):
    def opener(request, timeout):
        assert request.full_url.endswith("sarah?output_format=wav_16000")
        assert request.get_header("Authorization") == "Bearer test-key"
        assert json.loads(request.data) == {"text": expected_text}
        assert timeout == 3.0
        return response
    return SpeechmaticsTTS("test-key", player=player, timeout_s=3, opener=opener)


def test_sink_speaks_only_from_committed_runtime_result():
    from omni_q import build_mock_engine
    engine = build_mock_engine()
    runtime = VoiceRuntime(
        "session_01", "org_a",
        authority=AuthorityResolver({"operator": [VoiceCapability.GRAPH_MUTATION.value]}),
        mutator=RuntimeMutator(engine),
    )
    adapter = __import__("omni_q").SpeechmaticsRealtimeAdapter(runtime)
    player = Player()
    sink = VoiceSink(adapter, speech_output=tts(player))
    event = final(1, "don't touch the red connector")
    operator(runtime, event)

    result = sink.on_final(event.as_dict())

    assert result.status == "committed"
    assert len(player.calls) == 1
    assert player.calls[0][1].session_id == "session_01"
    assert sink.outputs[0].audio_bytes == len(b"RIFF-test-audio")
    assert [event.kind for event in runtime.bus.log if event.kind.startswith("voice.response")] == [
        "voice.response.sent"
    ]


def test_partial_and_observation_do_not_speak_or_invent_execution():
    runtime = VoiceRuntime("session_01", "org_a")
    adapter = __import__("omni_q").SpeechmaticsRealtimeAdapter(runtime)
    player = Player()
    sink = VoiceSink(adapter, speech_output=tts(player))
    sink.on_partial({
        "speaker": "speaker_01", "text": "set the", "start_ns": 1,
        "end_ns": 2_000_000, "sequence": 1,
    })
    result = sink.on_final({
        "speaker": "speaker_01", "text": "thanks", "start_ns": 3_000_000_000,
        "end_ns": 3_500_000_000, "sequence": 2,
    })
    assert result.status == "observed"
    assert player.calls == []
    assert not any(event.kind == "voice.response.sent" for event in runtime.bus.log)


def test_denied_and_interrupt_responses_are_derived_from_authority_result():
    runtime = VoiceRuntime("session_01", "org_a")
    adapter = __import__("omni_q").SpeechmaticsRealtimeAdapter(runtime)
    renderer = VoiceResponseRenderer()
    denied = adapter.on_final({
        "speaker": "speaker_01", "text": "keep everything local", "start_ns": 1,
        "end_ns": 2_000_000, "sequence": 1,
    })
    denied_response = renderer.render(denied)
    assert denied.status == "denied"
    assert denied_response and "not authorized" in denied_response.text
    interrupt = adapter.on_final({
        "speaker": "speaker_01", "text": "stop", "start_ns": 3_000_000_000,
        "end_ns": 3_500_000_000, "sequence": 2,
    })
    interrupt_response = renderer.render(interrupt)
    assert interrupt.status == "interrupt_denied"
    assert interrupt_response and "not authorized" in interrupt_response.text


def test_tts_transport_rejects_bad_response_and_renderer_does_not_require_fake_receipt():
    player = Player()
    bad = Response(audio=b"")
    def opener(_request, timeout):
        return bad
    client = SpeechmaticsTTS("test-key", player=player, opener=opener)
    response = SpeechResponse("response_1", "session_01", "org_a", "claim_1",
                              "committed", "The request was committed.")
    with pytest.raises(VoiceTransportError, match="invalid response"):
        client.speak(response)
    assert player.calls == []


def test_default_tts_player_is_a_real_system_player():
    client = SpeechmaticsTTS("test-key", opener=lambda *_args, **_kwargs: Response())
    assert isinstance(client.player, SystemAudioPlayer)


def test_system_player_rejects_raw_pcm_without_speaking():
    player = SystemAudioPlayer()
    receipt = type("Receipt", (), {"output_format": "pcm_16000"})()
    with pytest.raises(VoiceTransportError, match="wav_16000"):
        player.play(b"audio", receipt)


def test_renderer_rejects_unscoped_omni_result():
    with pytest.raises(VoiceTransportError, match="session_id"):
        VoiceResponseRenderer().render({"ok": True, "status": "completed"})


def test_renderer_preserves_explicit_downstream_receipt_message():
    result = {
        "session_id": "session_01", "org_id": "org_a", "claim_id": "claim_1",
        "status": "completed", "ok": True,
        "detail": {"message": "Placed the cup in table.N."},
    }
    response = VoiceResponseRenderer().render(result)
    assert response and response.text == "Placed the cup in table.N"
