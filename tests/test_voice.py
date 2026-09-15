"""Adversarial tests for the provider-neutral multi-speaker voice boundary."""

from __future__ import annotations

import pytest

from omni_q import (
    AuthorityResolver,
    ConversationArbiter,
    IntentCandidate,
    ReferenceResolver,
    SpeechFinal,
    SpeechPartial,
    SpeechmaticsRealtimeAdapter,
    VoiceCapability,
    VoiceError,
    VoiceRuntime,
    VoiceScopeError,
    WorldState,
)
from omni_q.contracts import Detection
from omni_q.voice import ClaimStatus
from omni_q.mutation import RuntimeMutator


def _final(seq: int, text: str, *, speaker: str = "speaker_01",
           session: str = "session_01", org: str = "org_a") -> SpeechFinal:
    start = seq * 1_000_000_000
    return SpeechFinal(
        session_id=session,
        org_id=org,
        speaker_id=speaker,
        text=text,
        t_start_ns=start,
        t_end_ns=start + 600_000_000,
        sequence=seq,
        confidence=0.94,
    )


def _partial(seq: int, text: str, *, pause_ms: float | None = None) -> SpeechPartial:
    return SpeechPartial(
        session_id="session_01",
        org_id="org_a",
        speaker_id="speaker_01",
        text=text,
        t_start_ns=seq * 1_000_000_000,
        t_end_ns=seq * 1_000_000_000 + 500_000_000,
        sequence=seq,
        pause_ms=pause_ms,
    )


def _operator(runtime: VoiceRuntime, event: SpeechFinal) -> None:
    runtime.registry.observe(event)
    runtime.registry.set_authority(
        event.speaker_id,
        "operator",
        [VoiceCapability.GRAPH_MUTATION.value, VoiceCapability.COMMAND.value],
    )


def test_partial_speech_is_observation_only_and_never_commits():
    calls: list[str] = []
    runtime = VoiceRuntime(
        "session_01", "org_a",
        intent_handler=lambda *_: calls.append("handler"),
    )

    turn = runtime.ingest_partial(_partial(1, "pick up the"))

    assert turn.should_wait is True
    assert calls == []
    assert [e.kind for e in runtime.bus.log] == ["voice.speaker.started", "voice.partial"]


def test_final_claim_is_attributed_but_unknown_speaker_cannot_mutate():
    calls: list[str] = []
    runtime = VoiceRuntime(
        "session_01", "org_a",
        intent_handler=lambda *_: calls.append("handler"),
    )

    result = runtime.ingest_final(_final(1, "keep everything local"))

    assert result.status == "denied"
    assert result.claim.status.value == "OBSERVED"
    assert result.authority and result.authority.allowed is False
    assert calls == []
    assert any(e.kind == "voice.intent.denied" for e in runtime.bus.log)


def test_explicit_operator_authority_uses_existing_governed_mutator():
    from omni_q import build_mock_engine

    engine = build_mock_engine()
    runtime = VoiceRuntime(
        "session_01", "local-demo",
        authority=AuthorityResolver({"operator": [VoiceCapability.GRAPH_MUTATION.value]}),
        mutator=RuntimeMutator(engine),
    )
    event = _final(1, "don't touch the red connector", org="local-demo")
    _operator(runtime, event)

    result = runtime.ingest_final(event, world=engine.world.state())

    assert result.status == "committed"
    assert result.mutation and result.mutation["applied"] == [["forbid_object", "connector_2"]]
    assert engine._pending_constraints[0].value == "connector_2"
    assert any(e.kind == "voice.intent.committed" for e in runtime.bus.log)


def test_authorized_but_deferred_mutation_is_not_falsely_committed():
    class Deferred:
        ok = False

        def as_dict(self):
            return {"ok": False, "deferred": [["spin", {"object": "plate_1"}, "not ready"]]}

    class Mutator:
        def apply(self, _text):
            return Deferred()

    runtime = VoiceRuntime(
        "session_01", "org_a",
        authority=AuthorityResolver({"operator": [VoiceCapability.GRAPH_MUTATION.value]}),
        mutator=Mutator(),
    )
    event = _final(1, "spin the plate")
    _operator(runtime, event)

    result = runtime.ingest_final(event)

    assert result.status == "authorized"
    assert result.candidate and result.candidate.status.value == "AUTHORIZED"
    assert not any(e.kind == "voice.intent.committed" for e in runtime.bus.log)
    assert any(e.kind == "voice.intent.not_committed" for e in runtime.bus.log)


def test_cross_org_and_out_of_order_events_fail_closed():
    runtime = VoiceRuntime("session_01", "org_a")

    with pytest.raises(VoiceScopeError):
        runtime.ingest_final(_final(1, "set the table", org="org_b"))

    runtime.ingest_partial(_partial(2, "set the table"))
    with pytest.raises(VoiceError, match="strictly increasing"):
        runtime.ingest_partial(_partial(2, "set the table"))


def test_adaptive_turn_keeps_incomplete_phrases_open_but_stop_is_complete():
    runtime = VoiceRuntime("session_01", "org_a")

    incomplete = runtime.ingest_partial(_partial(1, "Pick up the"))
    complete = runtime.ingest_final(_final(2, "Stop."))

    assert incomplete.semantically_complete is False
    assert incomplete.should_wait is True
    # The final stop is routed through the dedicated interruption gate; it is
    # complete semantically even though no general intent authority is granted.
    assert complete.interruption and complete.interruption.detected
    assert complete.interruption.action == "STOP"


def test_reference_resolution_uses_pointing_and_preserves_ambiguity():
    world = WorldState(
        frame=10,
        objects={
            "object_12": Detection("object_12", "bearing", "table", "table"),
            "object_17": Detection("object_17", "bearing", "table", "table"),
        },
        org_id="org_a",
    )
    resolver = ReferenceResolver()

    grounded = resolver.resolve(
        "Give him that one", claim_id="claim_1", world=world,
        pointed_at="object_17", recipient="person_bryan",
    )
    ambiguous = resolver.resolve(
        "Move that one", claim_id="claim_2", world=world,
        object_candidates=("object_12", "object_17"),
    )

    assert grounded[0].resolved == "object_17"
    assert grounded[1].recipient == "person_bryan"
    assert ambiguous[0].resolved is None
    assert ambiguous[0].status.value == "UNRESOLVED"


def test_correction_is_linked_instead_of_last_transcript_wins():
    arbiter = ConversationArbiter()
    first = IntentCandidate("intent_1", "claim_1", "s", "o", "speaker_01",
                            "command", "PLACE", {"to": "zone_a"}, 0.9)
    second = IntentCandidate("intent_2", "claim_2", "s", "o", "speaker_01",
                             "command", "PLACE", {"to": "zone_b"}, 0.9)

    arbiter.submit(first, "Put the plate there")
    corrected = arbiter.submit(second, "No, put it over there")

    assert corrected.correction_of == "intent_1"
    assert corrected.intent_id != corrected.correction_of


def test_authorized_stop_calls_reflex_handler_and_unknown_stop_is_not_executed():
    actions: list[tuple[str, str]] = []
    runtime = VoiceRuntime(
        "session_01", "org_a",
        authority=AuthorityResolver(interrupt_roles=("operator",)),
        interrupt_handler=lambda action, claim_id: actions.append((action, claim_id)),
    )
    event = _final(1, "stop", org="org_a")
    _operator(runtime, event)
    result = runtime.ingest_final(event)

    assert result.status == "interrupted"
    assert actions == [("STOP", result.claim.claim_id)]

    denied_runtime = VoiceRuntime(
        "session_02", "org_a",
        interrupt_handler=lambda action, claim_id: actions.append((action, claim_id)),
    )
    denied = denied_runtime.ingest_final(_final(1, "stop", session="session_02"))
    assert denied.status == "interrupt_denied"
    assert len(actions) == 1


def test_speechmatics_adapter_is_only_a_payload_translation_boundary():
    runtime = VoiceRuntime("session_01", "org_a")
    adapter = SpeechmaticsRealtimeAdapter(runtime)

    turn = adapter.on_partial({
        "speaker": "speaker_02",
        "text": "Actually,",
        "start_ns": 1_000_000_000,
        "end_ns": 1_400_000_000,
        "sequence": 1,
    })

    assert turn.speaker_id == "speaker_02"
    assert runtime.registry.get("speaker_02").authority_role == "unknown"
    assert runtime.bus.log[-1].source == "speechmatics"


def test_invalid_provider_timestamp_and_missing_final_text_are_rejected():
    with pytest.raises(VoiceError, match="nanosecond timestamp"):
        SpeechPartial("s", "o", "speaker", "hello", 1.5, 2, 1)  # type: ignore[arg-type]
    with pytest.raises(VoiceError, match="text"):
        SpeechFinal("s", "o", "speaker", " ", 1, 2, 1)


def test_stop_using_an_arm_is_a_constraint_not_an_emergency_stop():
    """Observed live 2026-09-13, spoken into the real Speechmatics transport.

    "Stop using the left arm." matched the bare-``stop`` safety pattern, so the
    gate claimed the whole sentence, ``nlu`` never saw it, and with no
    interrupt handler registered the result was ``interrupt_denied`` -- the
    instruction neither stopped anything nor changed the graph.
    """
    from omni_q.voice import InterruptionGate
    from omni_q import nlu

    gate = InterruptionGate()
    assert gate.inspect("Stop using the left arm.").detected is False
    assert ("prefer_arm", "right") in nlu.parse("Stop using the left arm.").constraints

    runtime = VoiceRuntime("session_01", "org_a")
    event = _final(1, "Stop using the left arm.")
    _operator(runtime, event)
    result = runtime.ingest_final(event)
    assert result.status != "interrupt_denied"
    assert result.candidate.action == "GRAPH_MUTATION"


def test_the_safety_gate_still_fires_on_every_real_stop_phrasing():
    """The narrowing above must not cost a single emergency phrase."""
    from omni_q.voice import InterruptionGate

    gate = InterruptionGate()
    for phrase in ("stop", "Stop!", "stop now", "emergency stop", "stop moving",
                   "freeze", "hold still", "back off", "retreat"):
        assert gate.inspect(phrase).detected is True, phrase


def test_a_safety_stop_is_still_dispatched_immediately_by_the_accumulator():
    from omni_q.voice import IntentAccumulator

    calls: list[str] = []
    runtime = VoiceRuntime("session_01", "org_a",
                           interrupt_handler=lambda action, claim_id: calls.append(action))
    event = _final(1, "stop")
    runtime.registry.observe(event)
    runtime.registry.set_authority("speaker_01", "operator",
                                   [VoiceCapability.INTERRUPT.value])
    result = IntentAccumulator(runtime).ingest_final(event)
    assert result is not None and calls == ["STOP"]


def test_low_confidence_speech_is_recorded_but_never_authorized():
    """Live 2026-09-13: room noise transcribed as "Praise the tag" at 0.62
    confidence and was authorized as a COMMAND. Every real instruction in the
    same session scored 0.81-1.0 -- the recognizer knew; nothing asked."""
    runtime = VoiceRuntime("session_01", "org_a", min_action_confidence=0.7)
    event = SpeechFinal(
        session_id="session_01", org_id="org_a", speaker_id="speaker_01",
        text="don't use the left arm", t_start_ns=1_000_000_000,
        t_end_ns=1_600_000_000, sequence=1, confidence=0.62,
    )
    _operator(runtime, event)
    result = runtime.ingest_final(event)

    assert result.status == "observed"
    assert result.candidate.action == "OBSERVATION"
    assert any(e.kind == "voice.intent.low_confidence" for e in runtime.bus.log)


def test_the_same_words_act_normally_above_the_floor():
    runtime = VoiceRuntime("session_01", "org_a", min_action_confidence=0.7)
    event = SpeechFinal(
        session_id="session_01", org_id="org_a", speaker_id="speaker_01",
        text="don't use the left arm", t_start_ns=1_000_000_000,
        t_end_ns=1_600_000_000, sequence=1, confidence=0.81,
    )
    _operator(runtime, event)
    assert runtime.ingest_final(event).candidate.action == "GRAPH_MUTATION"


def test_multilingual_constraint_uses_one_existing_authority_path():
    from omni_q import build_mock_engine

    engine = build_mock_engine()
    runtime = VoiceRuntime(
        "session_01", "local-demo",
        authority=AuthorityResolver({
            "operator": [VoiceCapability.GRAPH_MUTATION.value],
        }),
        mutator=RuntimeMutator(engine),
    )
    event = SpeechFinal(
        session_id="session_01", org_id="local-demo", speaker_id="speaker_01",
        text="No uses más el brazo izquierdo", t_start_ns=1_000_000_000,
        t_end_ns=1_700_000_000, sequence=1, confidence=0.94, language="es",
    )
    runtime.registry.observe(event)
    runtime.registry.set_authority(
        event.speaker_id, "operator", [VoiceCapability.GRAPH_MUTATION.value]
    )

    result = runtime.ingest_final(event, world=engine.world.state())

    assert result.status == "committed"
    assert result.claim.text == "No uses más el brazo izquierdo"
    assert result.claim.original_text == result.claim.text
    assert result.claim.canonical_text == "don't use the left arm anymore"
    assert result.candidate and result.candidate.args["constraints"] == [
        ["prefer_arm", "right"]
    ]
    assert engine._pending_constraints[0].value == "right"


def test_multilingual_fragments_complete_through_intent_accumulator():
    from omni_q import build_mock_engine
    from omni_q.voice import IntentAccumulator

    engine = build_mock_engine()
    runtime = VoiceRuntime(
        "session_01", "local-demo",
        authority=AuthorityResolver({
            "operator": [VoiceCapability.GRAPH_MUTATION.value],
        }),
        mutator=RuntimeMutator(engine),
    )
    first = SpeechFinal(
        session_id="session_01", org_id="local-demo", speaker_id="speaker_01",
        text="No uses", t_start_ns=1_000_000_000,
        t_end_ns=1_300_000_000, sequence=1, confidence=0.94, language="es",
    )
    second = SpeechFinal(
        session_id="session_01", org_id="local-demo", speaker_id="speaker_01",
        text="más el brazo izquierdo", t_start_ns=1_400_000_000,
        t_end_ns=2_000_000_000, sequence=2, confidence=0.94, language="es",
    )
    runtime.registry.observe(first)
    runtime.registry.set_authority(
        first.speaker_id, "operator", [VoiceCapability.GRAPH_MUTATION.value]
    )
    accumulator = IntentAccumulator(runtime)

    assert accumulator.ingest_final(first) is None
    result = accumulator.ingest_final(second)

    assert result is not None and result.status == "committed"
    assert result.claim.original_text == "No uses más el brazo izquierdo"
    assert result.claim.canonical_text == "don't use the left arm anymore"


def test_speaker_language_is_observed_without_granting_authority():
    runtime = VoiceRuntime("session_01", "org_a")
    event = _final(1, "Hola", org="org_a")
    event = SpeechFinal(**{**event.__dict__, "language": "es"})

    runtime.ingest_final(event)

    speaker = runtime.registry.get(event.speaker_id)
    assert speaker.preferred_language == "es"
    assert speaker.authority_role == "unknown"


def test_a_mumbled_safety_word_is_still_an_interruption():
    """Confidence gating must never suppress a safety reflex."""
    calls: list[str] = []
    runtime = VoiceRuntime("session_01", "org_a", min_action_confidence=0.9,
                           interrupt_handler=lambda a, c: calls.append(a))
    event = SpeechFinal(
        session_id="session_01", org_id="org_a", speaker_id="speaker_01",
        text="stop", t_start_ns=1_000_000_000, t_end_ns=1_400_000_000,
        sequence=1, confidence=0.35,
    )
    runtime.registry.observe(event)
    runtime.registry.set_authority("speaker_01", "operator",
                                   [VoiceCapability.INTERRUPT.value])
    runtime.ingest_final(event)
    assert calls == ["STOP"]


def _world_with(objects, revision):
    from omni_q.contracts import WorldState
    return WorldState(frame=0, objects=objects, goal="set the table",
                      revision=revision, org_id="org_a")


def test_a_claim_records_the_world_revision_it_was_acted_against():
    """Speech takes 1.0-3.7 s to arrive; the lag must be auditable."""
    runtime = VoiceRuntime("session_01", "org_a")
    event = _final(1, "don't use the left arm")
    _operator(runtime, event)
    result = runtime.ingest_final(event, world=_world_with({}, 7),
                                  observed_revision=5)
    assert result.claim.world_revision == 7
    assert result.claim.observed_revision == 5
    assert result.claim.revision_lag == 2
    assert result.claim.as_dict()["revision_lag"] == 2


def test_a_reference_resolved_against_a_moved_world_is_unresolved():
    """The speaker pointed at a scene that no longer exists by commit time."""
    runtime = VoiceRuntime("session_01", "org_a")
    event = _final(1, "don't move that")
    _operator(runtime, event)
    world = _world_with({"cup_1": Detection("cup_1", "cup", "left_table", "left_table")}, 9)

    fresh = runtime.ingest_final(event, world=world, pointed_at="cup_1",
                                 observed_revision=9)
    assert any(r.resolved for r in fresh.references), "baseline: it resolves"

    runtime2 = VoiceRuntime("session_01", "org_a")
    event2 = _final(1, "don't move that")
    _operator(runtime2, event2)
    stale = runtime2.ingest_final(event2, world=world, pointed_at="cup_1",
                                  observed_revision=6)
    assert not any(r.resolved for r in stale.references)
    assert all(r.status is ClaimStatus.UNRESOLVED
               for r in stale.references if "stale_by_3_revisions" in r.evidence)
    assert any(e.kind == "voice.reference.stale" for e in runtime2.bus.log)


def test_lag_does_not_suppress_a_constraint():
    """"Don't use the left arm" is still valid three seconds later."""
    runtime = VoiceRuntime("session_01", "org_a")
    event = _final(1, "don't use the left arm")
    _operator(runtime, event)
    result = runtime.ingest_final(event, world=_world_with({}, 40),
                                  observed_revision=1)
    assert result.candidate.action == "GRAPH_MUTATION"
    assert result.claim.revision_lag == 39


def test_omitting_observed_revision_keeps_the_previous_behaviour():
    runtime = VoiceRuntime("session_01", "org_a")
    event = _final(1, "don't move that")
    _operator(runtime, event)
    world = _world_with({"cup_1": Detection("cup_1", "cup", "left_table", "left_table")}, 9)
    result = runtime.ingest_final(event, world=world, pointed_at="cup_1")
    assert result.claim.revision_lag is None
    assert any(r.resolved for r in result.references)
