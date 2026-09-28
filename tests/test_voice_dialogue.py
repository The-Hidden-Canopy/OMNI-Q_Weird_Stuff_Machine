from __future__ import annotations

from omni_q import (
    DialogueResponse,
    IntentAccumulator,
    OmniDialogue,
    ReasonerResult,
    SpeechFinal,
    VoiceRuntime,
    WorldState,
)


def _final(seq: int, text: str, *, confidence: float = 0.94) -> SpeechFinal:
    start = seq * 1_000_000_000
    return SpeechFinal(
        session_id="session_01",
        org_id="org_a",
        speaker_id="speaker_01",
        text=text,
        t_start_ns=start,
        t_end_ns=start + 600_000_000,
        sequence=seq,
        confidence=confidence,
    )


class RecordingReasoner:
    backend = "recording_reasoner"

    def __init__(self, text: str = "The left arm is excluded by the current state.") -> None:
        self.text = text
        self.calls: list[tuple[str, str, int | None]] = []

    def reason(self, session_id: str, prompt: str, *, max_new_tokens: int | None = None):
        self.calls.append((session_id, prompt, max_new_tokens))
        return ReasonerResult(
            text=self.text,
            token_ids=(),
            backend=self.backend,
            prompt_tokens=len(prompt.split()),
        )


def test_addressed_dialogue_is_read_only_and_uses_the_scoped_world() -> None:
    reasoner = RecordingReasoner()
    mutator_calls: list[str] = []

    class Mutator:
        def apply(self, text: str):
            mutator_calls.append(text)
            raise AssertionError("dialogue must never reach the mutator")

    runtime = VoiceRuntime(
        "session_01",
        "org_a",
        mutator=Mutator(),
        dialogue_handler=OmniDialogue(reasoner),
    )
    world = WorldState(frame=7, objects={}, org_id="org_a")

    result = runtime.ingest_final(
        _final(1, "Omni, why are you using only one arm?"),
        world=world,
    )

    assert result.status == "answered"
    assert result.response_text == reasoner.text
    assert result.response_backend == "recording_reasoner"
    assert mutator_calls == []
    assert len(reasoner.calls) == 1
    session_id, prompt, max_new_tokens = reasoner.calls[0]
    assert session_id == "session_01"
    assert max_new_tokens == 48
    assert "read-only dialogue" in prompt
    assert '"frame": 7' in prompt
    assert [event.kind for event in runtime.bus.log
            if event.kind.startswith("voice.dialogue")] == [
                "voice.dialogue.requested",
                "voice.dialogue.answered",
            ]


def test_unaddressed_observation_does_not_invoke_dialogue() -> None:
    reasoner = RecordingReasoner()
    runtime = VoiceRuntime(
        "session_01",
        "org_a",
        dialogue_handler=OmniDialogue(reasoner),
    )

    result = runtime.ingest_final(_final(1, "the workspace is quiet"))

    assert result.status == "observed"
    assert result.response_text is None
    assert reasoner.calls == []
    assert not any(event.kind == "voice.dialogue.requested"
                   for event in runtime.bus.log)


def test_addressed_action_stays_on_the_authority_lane() -> None:
    reasoner = RecordingReasoner()
    runtime = VoiceRuntime(
        "session_01",
        "org_a",
        dialogue_handler=OmniDialogue(reasoner),
    )

    result = runtime.ingest_final(_final(1, "Omni, don't use the left arm"))

    assert result.status == "denied"
    assert result.candidate is not None
    assert result.candidate.action == "GRAPH_MUTATION"
    assert reasoner.calls == []
    assert not any(event.kind == "voice.dialogue.requested"
                   for event in runtime.bus.log)


def test_dialogue_rejects_a_cross_organization_world_before_model_call() -> None:
    reasoner = RecordingReasoner()
    runtime = VoiceRuntime(
        "session_01",
        "org_a",
        dialogue_handler=OmniDialogue(reasoner),
    )

    result = runtime.ingest_final(
        _final(1, "Omni, what is in the workspace?"),
        world=WorldState(frame=1, objects={}, org_id="org_b"),
    )

    assert result.status == "dialogue_failed"
    assert result.response_backend == "unavailable"
    assert "workspace" in (result.response_text or "")
    assert reasoner.calls == []
    failed = [event for event in runtime.bus.log
              if event.kind == "voice.dialogue.failed"]
    assert failed and failed[-1].data["reason"] == "scope_mismatch"


def test_accumulator_releases_an_addressed_question_without_timeout() -> None:
    calls: list[str] = []

    def dialogue_handler(*, claim, world, references):
        calls.append(claim.text)
        return DialogueResponse("I am checking the observed state.", "test")

    runtime = VoiceRuntime(
        "session_01",
        "org_a",
        dialogue_handler=dialogue_handler,
    )
    accumulator = IntentAccumulator(runtime)

    assert accumulator.ingest_final(_final(1, "Omni, why are you")) is None
    result = accumulator.ingest_final(_final(2, "doing that?"))

    assert result is not None
    assert result.status == "answered"
    assert result.claim.text == "Omni, why are you doing that?"
    assert calls == [result.claim.text]
    assert accumulator.held_count == 1
    assert accumulator.joined_count == 1


def test_low_confidence_action_is_not_reinterpreted_as_dialogue() -> None:
    reasoner = RecordingReasoner()
    runtime = VoiceRuntime(
        "session_01",
        "org_a",
        dialogue_handler=OmniDialogue(reasoner),
    )

    result = runtime.ingest_final(
        _final(1, "Omni, don't use the left arm", confidence=0.2)
    )

    assert result.status == "observed"
    assert result.candidate is not None
    assert result.candidate.action == "OBSERVATION"
    assert reasoner.calls == []
