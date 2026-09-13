"""Speechmatics transport tests -- mapping rules, offline replay, credentials.

No network, no API key, no audio device.  What is actually being checked is
that the vendor transport cannot hand ``omni_q.voice`` anything the voice
boundary would reject, and that the boundary's own rules (unknown speakers are
unauthorized, partials cannot commit) still hold when the input arrives through
the provider path rather than through hand-built dataclasses.
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
import wave
from pathlib import Path

import pytest

OMNIQ_ROOT = Path(__file__).resolve().parents[1]
if str(OMNIQ_ROOT) not in sys.path:
    sys.path.insert(0, str(OMNIQ_ROOT))

from integrations.speechmatics.mapper import TranscriptMapper  # noqa: E402
from integrations.speechmatics.transport import (  # noqa: E402
    API_KEY_ENV,
    AudioStream,
    SpeechmaticsConfig,
    SpeechmaticsError,
    SpeechmaticsTransport,
    UtteranceAggregator,
    VoiceSink,
    api_key_from_env,
    raw_audio_stream,
    read_jsonl,
    replay_messages,
    wav_audio_stream,
)
from omni_q.voice import SpeechmaticsRealtimeAdapter, VoiceCapability, VoiceRuntime  # noqa: E402

SAMPLE = OMNIQ_ROOT / "integrations" / "speechmatics" / "samples" / "session_replay.jsonl"


def _word(content: str, start: float, end: float, *, speaker: str | None = "S1",
          confidence: float | None = 0.9, kind: str = "word") -> dict:
    alternative: dict = {"content": content}
    if confidence is not None:
        alternative["confidence"] = confidence
    if speaker is not None:
        alternative["speaker"] = speaker
    return {"type": kind, "start_time": start, "end_time": end,
            "alternatives": [alternative]}


def _message(kind: str, transcript: str, start: float, end: float,
             results: list[dict] | None = None) -> dict:
    return {
        "message": kind,
        "metadata": {"start_time": start, "end_time": end, "transcript": transcript},
        "results": results if results is not None else [_word(transcript, start, end)],
    }


def _mapper(**kwargs) -> TranscriptMapper:
    clock = kwargs.pop("clock", None)
    mapper = TranscriptMapper(session_id="session_01", org_id="org_a",
                              **({"clock": clock} if clock else {}), **kwargs)
    mapper.start_stream(epoch_ns=0)
    return mapper


# -- mapping rules -----------------------------------------------------------

def test_sequence_is_shared_and_strictly_increasing_across_partials_and_finals():
    mapper = _mapper()
    seen = []
    for kind in ("AddPartialTranscript", "AddPartialTranscript", "AddTranscript",
                 "AddPartialTranscript", "AddTranscript"):
        mapped = mapper.map_message(_message(kind, "hello there", 0.5, 1.2))
        seen.append(mapped.payload["sequence"])
    assert seen == [1, 2, 3, 4, 5]  # VoiceRuntime shares one counter for both


def test_zero_length_transcript_still_satisfies_the_end_after_start_rule():
    mapper = _mapper()
    mapped = mapper.map_message(_message("AddTranscript", "go", 2.0, 2.0))
    payload = mapped.payload
    assert payload["t_end_ns"] > payload["t_start_ns"]
    assert isinstance(payload["t_start_ns"], int)


def test_empty_transcripts_are_dropped_not_forwarded():
    mapper = _mapper()
    assert mapper.map_message(_message("AddTranscript", "   ", 1.0, 1.0, [])) is None
    assert mapper.dropped_empty == 1
    assert mapper.sequence == 0  # a dropped message must not burn a sequence


def test_non_transcript_messages_map_to_none():
    mapper = _mapper()
    assert mapper.map_message({"message": "AudioAdded", "seq_no": 3}) is None
    assert mapper.map_message({"message": "Info", "reason": "x"}) is None


def test_confidence_is_the_measured_word_mean_and_ignores_punctuation():
    mapper = _mapper()
    results = [
        _word("stop", 0.0, 0.3, confidence=0.8),
        _word("now", 0.4, 0.6, confidence=0.6),
        _word(".", 0.6, 0.6, confidence=1.0, kind="punctuation"),
    ]
    mapped = mapper.map_message(_message("AddTranscript", "stop now.", 0.0, 0.6, results))
    assert mapped.payload["confidence"] == pytest.approx(0.7)
    assert mapped.confidence_measured is True


def test_missing_confidence_is_flagged_rather_than_silently_reported_as_one():
    mapper = _mapper()
    results = [_word("stop", 0.0, 0.3, confidence=None)]
    mapped = mapper.map_message(_message("AddTranscript", "stop", 0.0, 0.3, results))
    assert mapped.payload["confidence"] == 1.0
    assert mapped.confidence_measured is False


def test_speaker_falls_back_when_unlabelled_and_ignores_the_unknown_label():
    mapper = _mapper(default_speaker_id="unattributed")
    results = [_word("hello", 0.0, 0.3, speaker="UU")]
    mapped = mapper.map_message(_message("AddTranscript", "hello", 0.0, 0.3, results))
    assert mapped.payload["speaker_id"] == "unattributed"
    assert mapped.speaker_labelled is False


def test_speaker_is_the_majority_label_in_the_transcript():
    mapper = _mapper()
    results = [
        _word("a", 0.0, 0.1, speaker="S2"),
        _word("b", 0.2, 0.3, speaker="S1"),
        _word("c", 0.4, 0.5, speaker="S2"),
    ]
    mapped = mapper.map_message(_message("AddTranscript", "a b c", 0.0, 0.5, results))
    assert mapped.payload["speaker_id"] == "S2"
    assert mapped.speaker_labelled is True


def test_pause_ms_is_measured_from_the_previous_final_only():
    mapper = _mapper()
    first = mapper.map_message(_message("AddTranscript", "one", 0.0, 1.0))
    assert first.payload["pause_ms"] is None  # nothing to measure against yet
    second = mapper.map_message(_message("AddTranscript", "two", 1.6, 2.0))
    assert second.payload["pause_ms"] == pytest.approx(600.0, abs=1.0)


def test_latency_is_never_negative_when_the_stream_clock_runs_ahead():
    ticks = iter([0, 0])  # transcript claims 5 s of audio at t=0
    mapper = TranscriptMapper(session_id="s", org_id="o", clock=lambda: next(ticks))
    mapper.start_stream()
    mapped = mapper.map_message(_message("AddTranscript", "ahead", 4.0, 5.0))
    assert mapped.payload["latency_ms"] == 0.0


def test_latency_is_the_lag_behind_the_stream_position():
    ticks = iter([0, 3_000_000_000])  # 3 s wall for a transcript ending at 1 s
    mapper = TranscriptMapper(session_id="s", org_id="o", clock=lambda: next(ticks))
    mapper.start_stream()
    mapped = mapper.map_message(_message("AddTranscript", "late", 0.0, 1.0))
    assert mapped.payload["latency_ms"] == pytest.approx(2000.0)


def test_text_is_rebuilt_from_words_when_metadata_transcript_is_absent():
    mapper = _mapper()
    message = {"message": "AddTranscript",
               "results": [_word("stop", 0.0, 0.3),
                           _word("now", 0.4, 0.6),
                           _word(".", 0.6, 0.6, kind="punctuation")]}
    mapped = mapper.map_message(message)
    assert mapped.text == "stop now."  # punctuation gets no leading space


# -- the boundary, driven through the real adapter ---------------------------

def _runtime_with_operator(speaker_id: str) -> tuple[VoiceRuntime, list]:
    runtime = VoiceRuntime("session_01", "org_a")
    granted: list[str] = []

    def _grant(*, event, **_):
        if event.speaker_id == speaker_id and speaker_id not in granted:
            runtime.registry.set_authority(
                speaker_id, "operator",
                [VoiceCapability.COMMAND.value, VoiceCapability.GRAPH_MUTATION.value])
            granted.append(speaker_id)

    runtime.on("on_final_speech", _grant)
    return runtime, granted


def test_sample_replay_authorizes_the_operator_and_denies_the_unknown_speaker():
    assert SAMPLE.exists(), "regenerate with samples/make_sample_replay.py"
    runtime, _ = _runtime_with_operator("S1")
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    results: list = []
    sink.on_result = lambda mapped, result: results.append((mapped, result))
    mapper = TranscriptMapper(session_id="session_01", org_id="org_a")

    stats = replay_messages(read_jsonl(SAMPLE), sink, mapper)

    assert stats["messages_mapped"] == sink.partials + sink.finals
    assert sink.rejected == []
    finals = [r for mapped, r in results if mapped.final]
    assert [getattr(f, "status", None) for f in finals] == [
        "authorized", "authorized", "denied",
    ]
    # The denied one is the speaker nobody registered.
    assert finals[-1].claim.speaker_id == "S2"


def test_partial_speech_through_the_transport_never_produces_an_intent():
    runtime, _ = _runtime_with_operator("S1")
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    mapper = _mapper()
    mapped = mapper.map_message(
        _message("AddPartialTranscript", "don't use the left arm", 0.0, 1.0))
    decision = sink.deliver(mapped)
    assert sink.finals == 0
    assert not hasattr(decision, "candidate")  # a TurnDecision, not a dispatch


def test_adapter_rejection_is_recorded_and_does_not_stop_the_stream():
    runtime, _ = _runtime_with_operator("S1")
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    mapper = _mapper()
    good = mapper.map_message(_message("AddTranscript", "hello", 0.0, 1.0))
    sink.deliver(good)
    # Replay the same sequence number: VoiceRuntime requires strict increase.
    sink.deliver(good)
    later = mapper.map_message(_message("AddTranscript", "still here", 2.0, 3.0))
    sink.deliver(later)

    assert sink.finals == 2
    assert len(sink.rejected) == 1
    assert "sequence" in sink.rejected[0]["error"]


def test_cross_org_payload_is_rejected_at_the_boundary():
    runtime, _ = _runtime_with_operator("S1")
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    mapper = TranscriptMapper(session_id="session_01", org_id="org_b")
    mapper.start_stream(epoch_ns=0)
    sink.deliver(mapper.map_message(_message("AddTranscript", "hello", 0.0, 1.0)))
    assert sink.finals == 0
    assert "VoiceScopeError" in sink.rejected[0]["error"]


def test_provider_error_message_stops_the_replay():
    runtime, _ = _runtime_with_operator("S1")
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    messages = [{"message": "RecognitionStarted", "id": "x"},
                {"message": "Error", "type": "quota_exceeded",
                 "reason": "too many", "code": 4005}]
    with pytest.raises(SpeechmaticsError, match="quota_exceeded"):
        replay_messages(messages, sink, TranscriptMapper(session_id="s", org_id="o"))


# -- utterance segmentation --------------------------------------------------
#
# Regression fixture: exactly what Speechmatics returned for one spoken
# sentence on 2026-09-13 (tmp/msgs.jsonl). Five finals, no gap between them.

LIVE_FRAGMENTS = [("Omni.", 0.0, 1.08), ("Don't", 1.08, 1.32),
                  ("use your", 1.32, 1.68), ("left arm", 1.68, 2.16),
                  ("anymore.", 2.16, 2.84)]


def _fragment_messages(fragments=LIVE_FRAGMENTS, speaker: str = "S1") -> list[dict]:
    return [_message("AddTranscript", text, start, end,
                     [_word(w, start, end, speaker=speaker) for w in text.split()])
            for text, start, end in fragments]


def _aggregating_sink(silence_ms: float = 800.0):
    runtime, _ = _runtime_with_operator("S1")
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    delivered: list = []
    sink.on_result = lambda mapped, result: delivered.append((mapped, result))
    mapper = _mapper()
    aggregator = UtteranceAggregator(sink, mapper, silence_ms=silence_ms)
    return aggregator, mapper, sink, delivered


def test_provider_fragments_of_one_sentence_become_one_utterance():
    aggregator, mapper, sink, delivered = _aggregating_sink()
    for message in _fragment_messages():
        aggregator.offer(mapper.map_message(message))
    aggregator.close()

    assert sink.finals == 1  # not 5
    assert delivered[0][0].text == "Omni. Don't use your left arm anymore."
    assert aggregator.stats()["provider_final_fragments"] == 5


def test_the_reassembled_utterance_is_what_makes_the_instruction_actionable():
    """The whole point: fragments parse as nothing, the utterance parses."""
    from omni_q import nlu

    for text, _, _ in LIVE_FRAGMENTS:
        assert nlu.parse(text).constraints == []
    joined = "Omni. Don't use your left arm anymore."
    assert ("prefer_arm", "right") in nlu.parse(joined).constraints


def test_a_silence_gap_separates_two_utterances():
    aggregator, mapper, sink, delivered = _aggregating_sink(silence_ms=800)
    for message in _fragment_messages([("stop", 0.0, 0.4), ("that", 0.5, 0.9)]):
        aggregator.offer(mapper.map_message(message))
    # 1.5 s later: a new utterance, not a continuation.
    aggregator.offer(mapper.map_message(
        _message("AddTranscript", "keep everything local", 2.4, 3.2)))
    aggregator.close()

    assert sink.finals == 2
    assert [m.text for m, _ in delivered] == ["stop that", "keep everything local"]


def test_a_speaker_change_closes_the_current_utterance():
    aggregator, mapper, sink, delivered = _aggregating_sink()
    aggregator.offer(mapper.map_message(
        _message("AddTranscript", "use the left", 0.0, 0.5,
                 [_word("use", 0.0, 0.5, speaker="S1")])))
    aggregator.offer(mapper.map_message(
        _message("AddTranscript", "no stop", 0.5, 0.9,
                 [_word("no", 0.5, 0.9, speaker="S2")])))
    aggregator.close()

    assert sink.finals == 2
    assert [m.payload["speaker_id"] for m, _ in delivered] == ["S1", "S2"]


def test_flushed_utterance_takes_a_fresh_sequence_above_interleaved_partials():
    """A buffered fragment's own number is already stale by flush time."""
    aggregator, mapper, sink, delivered = _aggregating_sink()
    aggregator.offer(mapper.map_message(_message("AddTranscript", "don't", 0.0, 0.4)))
    # A partial arrives while the utterance is still buffering.
    aggregator.offer(mapper.map_message(
        _message("AddPartialTranscript", "don't use", 0.0, 0.6)))
    aggregator.offer(mapper.map_message(_message("AddTranscript", "use it", 0.4, 0.9)))
    aggregator.close()

    partial_seq = [m.payload["sequence"] for m, _ in delivered if not m.final]
    final_seq = [m.payload["sequence"] for m, _ in delivered if m.final]
    assert final_seq[0] > max(partial_seq)
    assert sink.rejected == []  # which is what VoiceRuntime would have done


def test_empty_finals_are_silence_evidence_not_just_noise():
    """Without this the whole session merges into a single utterance."""
    aggregator, mapper, sink, delivered = _aggregating_sink(silence_ms=800)
    aggregator.offer(mapper.map_message(_message("AddTranscript", "stop", 0.0, 0.4)))
    assert sink.finals == 0  # still buffering
    mapper.map_message(_message("AddTranscript", "", 0.4, 2.0, []))  # silence
    aggregator.mark_silence(mapper.last_empty_end_ns)
    assert sink.finals == 1
    assert delivered[0][0].text == "stop"


def test_an_unbroken_monologue_is_capped_rather_than_buffered_forever():
    aggregator, mapper, sink, _ = _aggregating_sink()
    aggregator.max_utterance_ms = 1000.0
    for index in range(6):
        start = index * 0.3
        aggregator.offer(mapper.map_message(
            _message("AddTranscript", f"word{index}", start, start + 0.25)))
    assert sink.finals >= 1


def test_utterance_confidence_is_weighted_by_words_not_by_fragment_count():
    aggregator, mapper, _, delivered = _aggregating_sink()
    aggregator.offer(mapper.map_message(_message(
        "AddTranscript", "one two three four", 0.0, 0.8,
        [_word(w, 0.0, 0.8, confidence=1.0) for w in "one two three four".split()])))
    aggregator.offer(mapper.map_message(_message(
        "AddTranscript", "five", 0.8, 1.0, [_word("five", 0.8, 1.0, confidence=0.5)])))
    aggregator.close()
    # 4 words at 1.0 and 1 at 0.5 -> 0.9, not the 0.75 a flat mean would give.
    assert delivered[-1][0].payload["confidence"] == pytest.approx(0.9)


def test_utterance_latency_includes_the_hold_and_is_reported_honestly():
    # clock() is read at start_stream, per mapped message, on buffering
    # (arrival stamp), and at flush
    ticks = iter([0, 0, 0, 2_000_000_000])
    mapper = TranscriptMapper(session_id="session_01", org_id="org_a",
                              clock=lambda: next(ticks))
    mapper.start_stream()
    runtime, _ = _runtime_with_operator("S1")
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    delivered: list = []
    sink.on_result = lambda mapped, result: delivered.append(mapped)
    aggregator = UtteranceAggregator(sink, mapper)
    aggregator.offer(mapper.map_message(_message("AddTranscript", "hello", 0.0, 1.0)))
    aggregator.close()
    # 2 s wall against a transcript ending at 1 s of stream time.
    assert delivered[-1].payload["latency_ms"] == pytest.approx(1000.0)


def test_provider_latency_alone_must_not_look_like_silence():
    """Regression: the live receipt showed 10 fragments -> 10 utterances.

    ``tick`` compared wall-now against the transcript's *stream* end, and a
    transcript arrives ~1.1 s after the audio it describes, so an 800 ms
    threshold read as idle the instant every fragment landed.
    """
    wall = [0]

    def clock() -> int:
        return wall[0]

    mapper = TranscriptMapper(session_id="session_01", org_id="org_a", clock=clock)
    mapper.start_stream()
    runtime, _ = _runtime_with_operator("S1")
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    aggregator = UtteranceAggregator(sink, mapper, silence_ms=800)

    # Two contiguous fragments, each arriving 1.1 s behind its own audio.
    for text, start, end in [("don't use", 0.0, 1.0), ("your left arm", 1.0, 2.0)]:
        wall[0] = int((end + 1.1) * 1_000_000_000)
        aggregator.offer(mapper.map_message(_message("AddTranscript", text, start, end)))
        aggregator.tick()

    assert sink.finals == 0, "provider lag was mistaken for the speaker stopping"
    aggregator.close()
    assert sink.finals == 1


def test_the_wall_clock_safety_net_still_closes_a_trailing_utterance():
    wall = [0]
    mapper = TranscriptMapper(session_id="session_01", org_id="org_a",
                              clock=lambda: wall[0])
    mapper.start_stream()
    runtime, _ = _runtime_with_operator("S1")
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    aggregator = UtteranceAggregator(sink, mapper, silence_ms=800,
                                     idle_grace_ms=1400)
    wall[0] = 2_100_000_000
    aggregator.offer(mapper.map_message(_message("AddTranscript", "stop that", 0.0, 1.0)))
    aggregator.tick()
    assert sink.finals == 0
    wall[0] += 2_300_000_000  # nothing has arrived for 2.3 s
    aggregator.tick()
    assert sink.finals == 1


LIVE_CAPTURE = (OMNIQ_ROOT / "integrations" / "speechmatics" / "samples"
                / "live_session_2026-09-13.jsonl")
HALTING_CAPTURE = (OMNIQ_ROOT / "integrations" / "speechmatics" / "samples"
                   / "live_halting_2026-09-13.jsonl")


def test_the_real_recorded_session_commits_the_instruction_end_to_end():
    """Against actual Speechmatics output, not a hand-written fixture.

    This is the session that failed before segmentation existed: five provider
    finals, one spoken instruction, and a mutation that never fired.
    """
    pytest.importorskip("omni_q.mutation")
    from omni_q import build_mock_engine
    from omni_q.mutation import RuntimeMutator

    runtime = VoiceRuntime("session_01", "org_a",
                           mutator=RuntimeMutator(build_mock_engine()))

    def _grant(*, event, **_):
        if event.speaker_id == "S1":
            runtime.registry.set_authority(
                "S1", "operator", [VoiceCapability.COMMAND.value,
                                   VoiceCapability.GRAPH_MUTATION.value])

    runtime.on("on_final_speech", _grant)
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    delivered: list = []
    sink.on_result = lambda mapped, result: delivered.append((mapped, result))
    mapper = TranscriptMapper(session_id="session_01", org_id="org_a")
    aggregator = UtteranceAggregator(sink, mapper)

    replay_messages(read_jsonl(LIVE_CAPTURE), sink, mapper, aggregator)

    finals = [(m, r) for m, r in delivered if m.final]
    assert len(finals) == 1, "five provider fragments must arrive as one claim"
    mapped, result = finals[0]
    assert mapped.text == "Omni. Don't use your left arm anymore."
    assert result.status == "committed"
    assert ["prefer_arm", "right"] in result.mutation["applied"]


def test_halting_speech_needs_the_intent_accumulator_not_just_silence():
    """The 2026-09-13 halting session: real pauses mid-instruction.

    Acoustic segmentation is *correct* here and still not enough -- the
    speaker genuinely stopped between "don't use" and "your left arm", so no
    silence-based rule can join them. Only accumulating until the parse is
    actionable recovers the instruction.
    """
    from omni_q import build_mock_engine
    from omni_q.mutation import RuntimeMutator
    from omni_q.voice import IntentAccumulator

    runtime = VoiceRuntime("session_01", "org_a",
                           mutator=RuntimeMutator(build_mock_engine()))
    runtime.on("on_final_speech", lambda *, event, **_: runtime.registry.set_authority(
        event.speaker_id, "operator",
        [VoiceCapability.COMMAND.value, VoiceCapability.GRAPH_MUTATION.value]))
    mapper = TranscriptMapper(session_id="session_01", org_id="org_a")
    accumulator = IntentAccumulator(runtime, sequence_source=mapper.next_sequence)
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(accumulator))
    committed: list = []
    sink.on_result = lambda mapped, result: (
        committed.append(result) if result is not None and
        getattr(result, "status", None) == "committed" else None)

    # No UtteranceAggregator: every provider fragment arrives on its own,
    # exactly as the live receipt recorded it.
    replay_messages(read_jsonl(HALTING_CAPTURE), sink, mapper)
    accumulator.flush()

    applied = [pair for result in committed for pair in result.mutation["applied"]]
    assert ["prefer_arm", "right"] in applied
    assert ["keep_local", None] in applied
    assert sink.held > 0, "fragments must have been held, not dispatched"


def test_a_single_word_command_waits_for_the_rest_of_the_instruction():
    """Live on 2026-09-13: "Keep" dispatched before "everything local"."""
    from omni_q.voice import IntentAccumulator

    runtime, _ = _runtime_with_operator("S1")
    accumulator = IntentAccumulator(runtime)
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(accumulator))
    mapper = _mapper()
    assert sink.deliver(mapper.map_message(
        _message("AddTranscript", "Keep", 0.0, 0.4))) is None
    result = sink.deliver(mapper.map_message(
        _message("AddTranscript", "everything local", 0.5, 1.2)))
    assert result.claim.text == "Keep everything local"
    assert result.candidate.action == "GRAPH_MUTATION"


def test_a_safety_word_is_never_held_waiting_for_more_speech():
    from omni_q.voice import IntentAccumulator

    runtime, _ = _runtime_with_operator("S1")
    accumulator = IntentAccumulator(runtime)
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(accumulator))
    mapper = _mapper()
    result = sink.deliver(mapper.map_message(_message("AddTranscript", "stop", 0.0, 0.3)))
    assert result is not None
    assert result.interruption is not None and result.interruption.detected


def test_speech_that_never_becomes_an_instruction_is_still_recorded():
    from omni_q.voice import IntentAccumulator

    runtime, _ = _runtime_with_operator("S1")
    accumulator = IntentAccumulator(runtime, max_utterances=2)
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(accumulator))
    mapper = _mapper()
    sink.deliver(mapper.map_message(_message("AddTranscript", "left arm", 0.0, 0.4)))
    sink.deliver(mapper.map_message(_message("AddTranscript", "over there", 0.5, 0.9)))
    assert sink.finals == 1  # released as an observation, not discarded
    assert accumulator.stats()["holds_released_unexecuted"] == 1


STOP_USING_CAPTURE = (OMNIQ_ROOT / "integrations" / "speechmatics" / "samples"
                      / "live_stop_using_2026-09-13.jsonl")


def test_the_recorded_stop_using_session_reaches_the_graph():
    """Third live session: the safety gate had claimed this whole sentence."""
    from omni_q import build_mock_engine
    from omni_q.mutation import RuntimeMutator
    from omni_q.voice import IntentAccumulator

    runtime = VoiceRuntime("session_01", "org_a",
                           mutator=RuntimeMutator(build_mock_engine()))
    runtime.on("on_final_speech", lambda *, event, **_: runtime.registry.set_authority(
        event.speaker_id, "operator",
        [VoiceCapability.COMMAND.value, VoiceCapability.GRAPH_MUTATION.value]))
    mapper = TranscriptMapper(session_id="session_01", org_id="org_a")
    accumulator = IntentAccumulator(runtime, sequence_source=mapper.next_sequence)
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(accumulator))
    results: list = []
    sink.on_result = lambda mapped, result: results.append(result)
    aggregator = UtteranceAggregator(sink, mapper)

    replay_messages(read_jsonl(STOP_USING_CAPTURE), sink, mapper, aggregator)
    accumulator.flush()

    finals = [r for r in results if r is not None and hasattr(r, "status")]
    assert finals and finals[-1].status == "committed"
    assert ["prefer_arm", "right"] in finals[-1].mutation["applied"]


def test_disabling_aggregation_reproduces_the_fragmented_behaviour():
    """Documents the defect the aggregator exists to fix."""
    runtime, _ = _runtime_with_operator("S1")
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    results: list = []
    sink.on_result = lambda mapped, result: results.append(result)
    mapper = _mapper()
    for message in _fragment_messages():
        sink.deliver(mapper.map_message(message))
    assert sink.finals == 5
    assert [getattr(r, "action", None) or getattr(r.candidate, "action", None)
            for r in results].count("OBSERVATION") >= 3


# -- configuration and credentials ------------------------------------------

def test_start_recognition_matches_the_documented_shape():
    config = SpeechmaticsConfig(sample_rate=44100, max_speakers=3)
    start = config.start_recognition()
    assert start["message"] == "StartRecognition"
    assert start["audio_format"] == {"type": "raw", "encoding": "pcm_s16le",
                                     "sample_rate": 44100}
    transcription = start["transcription_config"]
    assert transcription["enable_partials"] is True
    assert transcription["diarization"] == "speaker"
    assert transcription["speaker_diarization_config"] == {"max_speakers": 3}


def test_diarization_can_be_disabled_entirely():
    start = SpeechmaticsConfig(diarization=None).start_recognition()
    assert "diarization" not in start["transcription_config"]


def test_api_key_is_read_from_the_environment_and_absence_is_explained():
    assert api_key_from_env({API_KEY_ENV: " secret-key "}) == "secret-key"
    with pytest.raises(SpeechmaticsError, match=API_KEY_ENV):
        api_key_from_env({})


def test_receipt_carries_no_credential():
    config = SpeechmaticsConfig()
    transport = SpeechmaticsTransport(config=config, api_key="secret-key")
    runtime, _ = _runtime_with_operator("S1")
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    mapper = _mapper()
    receipt = transport.receipt(
        sink, mapper,
        AudioStream(16000, "pcm_s16le", lambda: iter(()), "test"),
        chunks_sent=0, wall_s=1.0, started_wall=0.0, mode="replay")
    assert "secret-key" not in json.dumps(receipt)
    assert receipt["provider"] == "speechmatics"


# -- audio sources -----------------------------------------------------------

def _write_wav(path: Path, *, rate: int, channels: int, width: int,
               seconds: float = 0.5) -> Path:
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 0) * frames * channels)
    return path


def test_wav_stream_reports_the_files_own_rate_and_chunks_it(tmp_path):
    path = _write_wav(tmp_path / "a.wav", rate=8000, channels=1, width=2)
    stream = wav_audio_stream(path, chunk_ms=100, realtime=False)
    assert stream.sample_rate == 8000  # not silently resampled to 16 kHz
    assert stream.duration_s == pytest.approx(0.5)

    async def _collect():
        return [chunk async for chunk in stream.chunks()]

    chunks = asyncio.run(_collect())
    assert len(chunks) == 5
    assert sum(len(c) for c in chunks) == 8000 * 2 // 2


def test_stereo_and_8bit_wavs_are_refused_rather_than_downmixed(tmp_path):
    stereo = _write_wav(tmp_path / "s.wav", rate=16000, channels=2, width=2)
    with pytest.raises(SpeechmaticsError, match="mono"):
        wav_audio_stream(stereo)
    eight_bit = _write_wav(tmp_path / "e.wav", rate=16000, channels=1, width=1)
    with pytest.raises(SpeechmaticsError, match="16-bit"):
        wav_audio_stream(eight_bit)


def test_raw_stream_chunks_at_the_declared_rate(tmp_path):
    path = tmp_path / "a.pcm"
    path.write_bytes(b"\x00\x00" * 16000)  # 1 s of 16 kHz silence
    stream = raw_audio_stream(path, sample_rate=16000, chunk_ms=250, realtime=False)
    assert stream.duration_s == pytest.approx(1.0)

    async def _collect():
        return [chunk async for chunk in stream.chunks()]

    assert len(asyncio.run(_collect())) == 4


def test_terminal_punctuation_flushes_without_waiting_out_silence():
    """The 3.3 s an operator measured was mostly an unnecessary wait.

    Speechmatics sends sentence-final punctuation as its own fragment; once it
    arrives the utterance is over and there is nothing left to wait for.
    """
    aggregator, mapper, sink, delivered = _aggregating_sink()
    aggregator.offer(mapper.map_message(
        _message("AddTranscript", "Stop using the left arm", 0.0, 1.4)))
    assert sink.finals == 0
    aggregator.offer(mapper.map_message(
        _message("AddTranscript", ".", 1.4, 1.4,
                 [_word(".", 1.4, 1.4, kind="punctuation")])))
    assert sink.finals == 1  # no tick(), no silence gap, no close()
    assert delivered[-1][0].text == "Stop using the left arm."


def test_punctuation_flush_can_be_disabled():
    runtime, _ = _runtime_with_operator("S1")
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    mapper = _mapper()
    aggregator = UtteranceAggregator(sink, mapper,
                                     flush_on_terminal_punctuation=False)
    aggregator.offer(mapper.map_message(_message("AddTranscript", "done.", 0.0, 0.5)))
    assert sink.finals == 0
    aggregator.close()
    assert sink.finals == 1


def test_punctuation_only_fragments_do_not_inflate_utterance_confidence():
    """A "." carries no measured word confidence; it must not count as 1.0."""
    aggregator, mapper, _, delivered = _aggregating_sink()
    aggregator.offer(mapper.map_message(_message(
        "AddTranscript", "left arm", 0.0, 0.6,
        [_word(w, 0.0, 0.6, confidence=0.5) for w in ("left", "arm")])))
    aggregator.offer(mapper.map_message(_message(
        "AddTranscript", ".", 0.6, 0.6,
        [_word(".", 0.6, 0.6, kind="punctuation", confidence=None)])))
    aggregator.close()  # too short to trigger the punctuation flush
    assert delivered[-1][0].payload["confidence"] == pytest.approx(0.5)


def test_a_vocative_period_does_not_split_the_instruction():
    """The real capture's first fragment is "Omni." -- punctuated, not a
    sentence. Flushing there separates the address from the command."""
    aggregator, mapper, sink, delivered = _aggregating_sink()
    for message in _fragment_messages():
        aggregator.offer(mapper.map_message(message))
    assert sink.finals == 1, "should flush once, at 'anymore.', not at 'Omni.'"
    assert delivered[-1][0].text == "Omni. Don't use your left arm anymore."


def test_a_bare_safety_word_is_flushed_at_once_not_after_the_idle_timeout():
    """"Stop!" is two characters short of the punctuation rule and is the one
    utterance that must never sit in a buffer."""
    from omni_q.voice import InterruptionGate

    runtime, _ = _runtime_with_operator("S1")
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(runtime))
    mapper = _mapper()
    gate = InterruptionGate()
    aggregator = UtteranceAggregator(
        sink, mapper, urgent=lambda text: gate.inspect(text).detected)
    aggregator.offer(mapper.map_message(_message("AddTranscript", "Stop!", 0.0, 0.4)))
    assert sink.finals == 1  # no tick(), no close()


SEVERED_CAPTURE = (OMNIQ_ROOT / "integrations" / "speechmatics" / "samples"
                   / "live_stop_severed_2026-09-13.jsonl")


def _gate_urgent():
    from omni_q.voice import InterruptionGate
    gate = InterruptionGate()
    return lambda text: gate.inspect(text).detected


def test_urgent_flush_does_not_sever_stop_using_from_its_object():
    """Live 2026-09-13: the operator said "stop using the left arm" and the
    graph committed *use the left arm*.

    The fragment "Stop" alone looks like an emergency. Flushing there left
    "using the left arm" to parse as a preference for the left arm -- the
    inversion, reintroduced by the safety optimization itself.
    """
    aggregator, mapper, sink, delivered = _aggregating_sink()
    aggregator.urgent = _gate_urgent()
    # The partial leads the final: by the time "Stop" is finalized the
    # provider has already transcribed "Stop using".
    aggregator.offer(mapper.map_message(
        _message("AddPartialTranscript", "Stop using", 0.0, 0.8)))
    aggregator.offer(mapper.map_message(_message("AddTranscript", "Stop", 0.0, 0.5)))
    assert sink.finals == 0, "an in-flight 'stop using' is not an emergency"

    aggregator.offer(mapper.map_message(
        _message("AddTranscript", "using the left arm.", 0.5, 1.6)))
    aggregator.close()
    assert delivered[-1][0].text == "Stop using the left arm."


def test_a_bare_stop_is_still_urgent_when_nothing_follows_it():
    aggregator, mapper, sink, _ = _aggregating_sink()
    aggregator.urgent = _gate_urgent()
    aggregator.offer(mapper.map_message(
        _message("AddPartialTranscript", "Stop", 0.0, 0.4)))
    aggregator.offer(mapper.map_message(_message("AddTranscript", "Stop", 0.0, 0.4)))
    assert sink.finals == 1  # dispatched at once, no timeout


def test_the_severed_session_replays_to_the_correct_constraint():
    from omni_q import build_mock_engine
    from omni_q.mutation import RuntimeMutator
    from omni_q.voice import IntentAccumulator

    runtime = VoiceRuntime("session_01", "org_a",
                           mutator=RuntimeMutator(build_mock_engine()))
    runtime.on("on_final_speech", lambda *, event, **_: runtime.registry.set_authority(
        event.speaker_id, "operator",
        [VoiceCapability.COMMAND.value, VoiceCapability.GRAPH_MUTATION.value]))
    mapper = TranscriptMapper(session_id="session_01", org_id="org_a")
    accumulator = IntentAccumulator(runtime, sequence_source=mapper.next_sequence)
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(accumulator))
    committed: list = []
    sink.on_result = lambda mapped, result: (
        committed.append(result)
        if result is not None and getattr(result, "status", None) == "committed"
        else None)
    aggregator = UtteranceAggregator(sink, mapper, urgent=_gate_urgent())

    replay_messages(read_jsonl(SEVERED_CAPTURE), sink, mapper, aggregator)
    accumulator.flush()

    applied = [pair for r in committed for pair in r.mutation["applied"]]
    assert ["prefer_arm", "right"] in applied
    assert ["prefer_arm", "left"] not in applied  # the inversion


def test_partial_latency_is_measured_against_speech_not_the_audio_window():
    """A partial's metadata end_time is the current audio position.

    Live 2026-09-13 the partial "Omni" arrived stamped 0.00-1.64; measuring
    against that envelope made every partial look like it had arrived before
    its own audio, and the receipt reported a meaningless 0.0 ms.
    """
    ticks = iter([0, 2_000_000_000])
    mapper = TranscriptMapper(session_id="s", org_id="o", clock=lambda: next(ticks))
    mapper.start_stream()
    message = {
        "message": "AddPartialTranscript",
        "metadata": {"start_time": 0.0, "end_time": 1.64, "transcript": "Omni"},
        "results": [_word("Omni", 0.0, 0.5)],
    }
    mapped = mapper.map_message(message)
    # 2 s wall against speech that ended at 0.5 s, not the 1.64 s window.
    assert mapped.payload["latency_ms"] == pytest.approx(1500.0)


def test_an_instruction_cut_at_a_false_period_is_rejoined():
    """Live 2026-09-13: "Omni. Don't use your." arrived punctuated as a
    finished sentence a second before "Left arm."."""
    from omni_q import build_mock_engine
    from omni_q.mutation import RuntimeMutator
    from omni_q.voice import IntentAccumulator

    runtime = VoiceRuntime("session_01", "org_a",
                           mutator=RuntimeMutator(build_mock_engine()))
    runtime.on("on_final_speech", lambda *, event, **_: runtime.registry.set_authority(
        event.speaker_id, "operator",
        [VoiceCapability.COMMAND.value, VoiceCapability.GRAPH_MUTATION.value]))
    mapper = _mapper()
    accumulator = IntentAccumulator(runtime, sequence_source=mapper.next_sequence)
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(accumulator))

    held = sink.deliver(mapper.map_message(
        _message("AddTranscript", "Omni. Don't use your.", 0.0, 1.4)))
    assert held is None, "an instruction ending on 'your' is not finished"
    result = sink.deliver(mapper.map_message(
        _message("AddTranscript", "Left arm.", 1.5, 2.2)))
    assert result.status == "committed"
    assert ["prefer_arm", "right"] in result.mutation["applied"]


def test_an_expiring_hold_does_not_strand_the_event_that_expired_it():
    """Live 2026-09-13: one final was dropped with "speech sequence is not
    strictly increasing".

    Holding reorders dispatch relative to arrival. A stale hold is released
    *during* the handling of a newer event and takes a sequence number; the
    event that triggered the release then followed carrying an older one, and
    VoiceRuntime correctly rejected it -- losing real speech.
    """
    from omni_q.voice import IntentAccumulator

    runtime, _ = _runtime_with_operator("S1")
    mapper = _mapper()
    accumulator = IntentAccumulator(runtime, window_ms=500,
                                    sequence_source=mapper.next_sequence)
    sink = VoiceSink(SpeechmaticsRealtimeAdapter(accumulator))

    # Held: not an instruction on its own.
    assert sink.deliver(mapper.map_message(
        _message("AddTranscript", "left arm", 0.0, 0.5))) is None
    # Partials in between move the shared counter along.
    for index in range(3):
        sink.deliver(mapper.map_message(
            _message("AddPartialTranscript", f"noise {index}", 1.0, 1.4)))
    # Far enough later that the hold expires, releasing it first.
    result = sink.deliver(mapper.map_message(
        _message("AddTranscript", "don't use the left arm.", 9.0, 10.0)))

    assert sink.rejected == [], sink.rejected
    assert result is not None and result.status != "observed"
