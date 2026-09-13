from __future__ import annotations

import asyncio

from integrations.speechmatics.mapper import TranscriptMapper
from integrations.speechmatics.transport import AudioStream, VoiceSink
from integrations.speechmatics.voice_sdk import (
    SpeechmaticsVoiceBridge,
    SpeechmaticsVoiceTransport,
    SpeakerFocusRequest,
    VoiceIdentityBinding,
)
from omni_q.voice import (
    IntentAccumulator,
    SpeechmaticsRealtimeAdapter,
    VoiceCapability,
    VoiceRuntime,
)


def _runtime() -> VoiceRuntime:
    runtime = VoiceRuntime("session_01", "org_a")

    def grant(*, event, **_ignored):
        runtime.registry.set_authority(
            event.speaker_id,
            "operator",
            [VoiceCapability.COMMAND.value, VoiceCapability.GRAPH_MUTATION.value],
        )

    runtime.on("on_final_speech", grant)
    return runtime


def test_voice_sdk_segment_and_end_of_turn_share_the_existing_boundary():
    runtime = _runtime()
    mapper = TranscriptMapper("session_01", "org_a")
    accumulator = IntentAccumulator(runtime, sequence_source=mapper.next_sequence)
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(accumulator))
    bridge = SpeechmaticsVoiceBridge(sink, mapper)

    bridge.handle_message({"message": "RecognitionStarted", "id": "s"})
    assert bridge.handle_message({
        "message": "AddSegment",
        "metadata": {"start_time": 0.0, "end_time": 0.4},
        "segments": [{
            "speaker_id": "S1",
            "text": "keep",
            "metadata": {"start_time": 0.0, "end_time": 0.4},
        }],
    }) is None

    result = bridge.handle_message({
        "message": "EndOfTurn",
        "turn_id": 1,
    })

    assert result is not None
    assert result.claim.text == "keep"
    assert accumulator.stats()["semantic_turns"] == 1


def test_voice_sdk_bridge_does_not_treat_partial_segment_as_final():
    runtime = _runtime()
    mapper = TranscriptMapper("session_01", "org_a")
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    bridge = SpeechmaticsVoiceBridge(sink, mapper)

    partial = bridge.handle_message({
        "message": "AddPartialSegment",
        "segments": [{"speaker_id": "S1", "text": "keep"}],
    })

    assert not hasattr(partial, "status")
    assert sink.finals == 0
    assert sink.partials == 1


class _FakeVoiceClient:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}
        self.connected = False
        self.sent: list[bytes] = []

    def on(self, event):
        def register(callback):
            self.handlers[event] = callback
            return callback
        return register

    async def connect(self):
        self.connected = True

    async def send_audio(self, chunk: bytes):
        self.sent.append(chunk)

    async def disconnect(self):
        self.connected = False


def test_voice_sdk_transport_registers_events_and_streams_audio_without_network():
    client = _FakeVoiceClient()

    async def chunks():
        yield b"one"
        yield b"two"

    audio = AudioStream(
        sample_rate=16000,
        encoding="pcm_s16le",
        chunks=chunks,
        description="test",
    )
    runtime = _runtime()
    mapper = TranscriptMapper("session_01", "org_a")
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    transport = SpeechmaticsVoiceTransport(
        preset="smart_turn",
        client_factory=lambda: client,
        event_types={
            "started": "RecognitionStarted",
            "partial": "AddPartialSegment",
            "final": "AddSegment",
            "turn": "EndOfTurn",
        },
    )

    receipt = asyncio.run(transport.run(audio, sink, mapper))

    assert client.sent == [b"one", b"two"]
    assert client.connected is False
    assert receipt["provider"] == "speechmatics-voice"
    assert len(client.handlers) == 4


def test_focus_requests_are_explicit_and_fail_closed_on_invalid_inputs():
    assert SpeakerFocusRequest(("S1",), mode="retain").speaker_ids == ("S1",)
    for request in (
        lambda: SpeakerFocusRequest((), mode="retain"),
        lambda: SpeakerFocusRequest(("S1",), mode="automatic"),
        lambda: SpeakerFocusRequest(("",), mode="ignore"),
    ):
        try:
            request()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid focus request was accepted")


def test_voiceprint_bindings_validate_without_granting_runtime_authority():
    binding = VoiceIdentityBinding(
        label="Gerron",
        identifiers=("voiceprint:abc123",),
    )
    assert binding.label == "Gerron"

    for request in (
        lambda: VoiceIdentityBinding(label="", identifiers=("x",)),
        lambda: VoiceIdentityBinding(label="Gerron", identifiers=()),
        lambda: VoiceIdentityBinding(label="Gerron", identifiers=("",)),
    ):
        try:
            request()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid voiceprint binding was accepted")

    transport = SpeechmaticsVoiceTransport(
        client_factory=lambda: _FakeVoiceClient(),
        known_speakers=(binding,),
    )
    assert [item.label for item in transport.known_speakers] == ["Gerron"]
