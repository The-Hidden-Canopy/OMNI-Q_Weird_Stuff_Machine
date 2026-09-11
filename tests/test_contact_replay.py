"""Independent, non-controller replay checks for retained contact receipts."""

from __future__ import annotations

import json

import pytest

from omni_q.contact_replay import (
    ContactReplayRejected,
    _canonical_hash,
    replay_contact_handoff_directory,
    replay_contact_handoff_receipt,
)
from omni_q.intel_sim import ContactHandoffConfig, run_contact_handoff


def test_independent_replay_accepts_a_valid_receipt():
    receipt = run_contact_handoff(ContactHandoffConfig(seed=19))

    replay = replay_contact_handoff_receipt(receipt.as_dict())

    assert replay["ok"] is True
    assert replay["success"] is True
    assert replay["checks"]["phase_order"] is True
    assert replay["checks"]["final_state"] is True


def test_independent_replay_rejects_tampered_state_sample():
    receipt = run_contact_handoff(ContactHandoffConfig(seed=19)).as_dict()
    receipt["state_samples"][-1]["cup_on_table"] = False

    with pytest.raises(ContactReplayRejected, match="content hash mismatch"):
        replay_contact_handoff_receipt(receipt)


def test_independent_replay_rejects_wrong_model_identity():
    receipt = run_contact_handoff(ContactHandoffConfig(seed=19)).as_dict()

    with pytest.raises(ContactReplayRejected, match="MJCF hash"):
        replay_contact_handoff_receipt(receipt, expected_mjcf_sha256="sha256:not-the-model")


def test_directory_replay_retains_each_receipt_and_rejection(tmp_path):
    receipt = run_contact_handoff(ContactHandoffConfig(seed=19)).as_dict()
    (tmp_path / "trial-00.json").write_text(
        json.dumps(receipt), encoding="utf-8",
    )
    (tmp_path / "trial-01.json").write_text(
        json.dumps(receipt), encoding="utf-8",
    )
    tampered = dict(receipt)
    tampered["content_hash"] = "sha256:tampered"
    (tmp_path / "trial-02.json").write_text(
        json.dumps(tampered), encoding="utf-8",
    )

    report = replay_contact_handoff_directory(
        tmp_path, output_path=tmp_path / "replay-report.json",
    )

    assert report["receipts_seen"] == 3
    assert report["outcomes"] == {
        "accepted": 2, "rejected": 1, "successful": 2, "failed": 0,
    }
    assert sum(item["accepted"] is False for item in report["receipts"]) == 1
    assert (tmp_path / "replay-report.json").exists()
    persisted = json.loads((tmp_path / "replay-report.json").read_text(encoding="utf-8"))
    assert persisted["outcomes"] == report["outcomes"]
    with pytest.raises(ContactReplayRejected, match="overwrite"):
        replay_contact_handoff_directory(tmp_path, output_path=tmp_path / "replay-report.json")


def test_replay_rejects_malformed_json_and_missing_numeric_samples(tmp_path):
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(ContactReplayRejected, match="not an object"):
        replay_contact_handoff_receipt(malformed)

    receipt = run_contact_handoff(ContactHandoffConfig(seed=19)).as_dict()
    receipt["state_samples"][-1].pop("max_gripper_force_n")
    receipt["content_hash"] = _canonical_hash(receipt)
    with pytest.raises(ContactReplayRejected, match="missing max_gripper_force_n"):
        replay_contact_handoff_receipt(receipt)
