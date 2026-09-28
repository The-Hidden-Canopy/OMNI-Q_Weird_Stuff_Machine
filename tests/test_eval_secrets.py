"""Evaluator-only secrets + deterministic seeding (ported from open_world_model_harness)."""

from __future__ import annotations

import json

from omni_q.eval_secrets import SealedSnapshot, SensoryCue, stable_unit_float


def test_stable_unit_float_is_deterministic_across_calls():
    first = stable_unit_float("aptitude", "session-1", "player-1", "skill", "node")
    second = stable_unit_float("aptitude", "session-1", "player-1", "skill", "node")
    assert first == second
    # Closed interval, not half-open: byte 255 (possible, 1/256 of inputs)
    # lands exactly on 1.35 -- see stable_unit_float's docstring.
    assert 0.65 <= first <= 1.35


def test_stable_unit_float_distinct_keys_differ():
    values = {
        stable_unit_float("aptitude", "player-1", "skill", f"node-{i}")
        for i in range(8)
    }
    assert len(values) > 1, "distinct entity keys should not collapse to one value"


def test_stable_unit_float_domain_separates_namespaces():
    a = stable_unit_float("competency", "player-1")
    b = stable_unit_float("readiness", "player-1")
    assert a != b


def test_sealed_snapshot_public_view_excludes_secret_fields(tmp_path):
    destination = tmp_path / "final_evaluator_snapshot.json"
    sealed = SealedSnapshot(
        payload={
            "session_id": "s-1",
            "hidden_ground_truth": {"object_3": "hot_glue"},
            "scores": {"team_a": 0.9},
        },
        destination=destination,
        public_fields=("session_id", "scores"),
    )
    view = sealed.public_view()
    assert "session_id" in view.fields
    assert "scores" in view.fields
    assert "hidden_ground_truth" not in view.fields
    assert "hidden_ground_truth" not in view.to_dict()
    assert destination.exists(), "sealed payload written to its own file"


def test_sealed_snapshot_round_trips_through_own_writer(tmp_path):
    destination = tmp_path / "final_evaluator_snapshot.json"
    payload = {"session_id": "s-2", "latent": {"x": 1.5}, "tick": 41}
    sealed = SealedSnapshot(payload, destination, public_fields=("session_id", "tick"))
    loaded = SealedSnapshot.load(destination, public_fields=("session_id",))
    assert loaded.payload == payload
    assert loaded.public_view().fields == {"session_id": "s-2"}


def test_mutating_public_view_cannot_leak_into_sealed_payload(tmp_path):
    destination = tmp_path / "final_evaluator_snapshot.json"
    sealed = SealedSnapshot(
        payload={"session_id": "s-3", "scores": {"team_a": 0.9}},
        destination=destination,
        public_fields=("scores",),
    )
    view = sealed.public_view()
    view.fields["scores"]["team_a"] = 0.0
    view.fields["injected"] = "forged"

    fresh = sealed.public_view()
    assert fresh.fields == {"scores": {"team_a": 0.9}}
    on_disk = json.loads(destination.read_text(encoding="utf-8"))
    assert on_disk == {"session_id": "s-3", "scores": {"team_a": 0.9}}


def test_sensory_cue_defaults_to_reliable():
    cue = SensoryCue(cue_id="cue:1", text="The workspace is clear.")
    assert cue.reliable is True
    assert cue.source == "game_sensory"
    assert cue.to_dict()["reliable"] is True


def test_sensory_cue_can_flag_hallucination_prone_channel():
    cue = SensoryCue(
        cue_id="cue:2",
        text="Something moved at the edge of perception.",
        reliable=False,
    )
    assert cue.reliable is False
    assert cue.to_dict()["reliable"] is False
