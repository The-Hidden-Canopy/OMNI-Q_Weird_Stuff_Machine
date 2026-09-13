from __future__ import annotations

import json

import pytest

from omni_q.contracts import Detection, TransitionRejected, TransitionRequest
from omni_q.world import MockWorld

from integrations.intel.demonstrations import (
    DemonstrationRecorder,
    SCHEMA_VERSION,
)


def request(world, *, step_id="pick-cup", org_id=None, expected=None):
    return TransitionRequest(
        step_id=step_id,
        op="PICK",
        args={"object": "plate_1"},
        expected_revision=world.revision if expected is None else expected,
        actor="left_arm",
        org_id=world.org_id if org_id is None else org_id,
    )


def make_world() -> MockWorld:
    return MockWorld([
        Detection("plate_1", "plate", "tray", "table"),
    ])


def test_recorder_uses_governed_transition_and_preserves_receipt() -> None:
    world = make_world()
    recorder = DemonstrationRecorder(world, episode_id="ep-1", goal="pick the plate")

    frame = recorder.record(request(world))

    assert frame.accepted
    assert frame.result["ok"] is True
    assert frame.before["revision"] == 0
    assert frame.after["revision"] == 1
    assert frame.request["actor"] == "left_arm"
    assert len(recorder.frames) == 1


def test_cross_org_request_is_blocked_before_world_mutation() -> None:
    world = make_world()
    recorder = DemonstrationRecorder(world, episode_id="ep-2", goal="stay scoped")

    with pytest.raises(TransitionRejected, match="does not match world org"):
        recorder.record(request(world, org_id="other-org"))

    assert world.revision == 0
    assert recorder.frames[0].accepted is False
    assert recorder.frames[0].rejection["error_type"] == "TransitionRejected"


def test_stale_request_is_recorded_as_rejection_and_re_raised() -> None:
    world = make_world()
    recorder = DemonstrationRecorder(world, episode_id="ep-3", goal="preserve failure")
    world.revision = 4

    with pytest.raises(TransitionRejected, match="current revision is 4"):
        recorder.record(request(world, expected=3))

    frame = recorder.frames[0]
    assert frame.accepted is False
    assert frame.rejection["error_type"] == "TransitionRejected"
    assert frame.before["revision"] == 4
    assert frame.after["revision"] == 4


def test_jsonl_is_exclusive_and_payload_is_hashed(tmp_path) -> None:
    world = make_world()
    recorder = DemonstrationRecorder(world, episode_id="ep-4", goal="write evidence")
    recorder.record(request(world))
    target = tmp_path / "episode.jsonl"

    digest = recorder.write_jsonl(target)

    lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["schema_version"] == SCHEMA_VERSION
    assert lines[0]["content_hash"] == digest
    assert lines[1]["accepted"] is True
    with pytest.raises(FileExistsError):
        recorder.write_jsonl(target)
