"""OQ-037 acceptance: durable evidence/receipt collection.

Covers the additions on top of the base EvidenceLedger (Gerron/Claude):
model-ref provenance, cross-instance hash-chain continuity, tamper
detection via verify_ledger, and Receipt protocol conformance.
"""

from __future__ import annotations

import json

from omni_q import build_mock_engine
from omni_q.contracts import Receipt, content_hash_of, verify_chain
from omni_q.evidence import EvidenceLedger, extract_model_refs, verify_ledger
from omni_q.fakes import FakeRecorder


def test_conforms_to_receipt_protocol(tmp_path):
    assert isinstance(EvidenceLedger(tmp_path), Receipt)


def test_run_records_inputs_plan_actions_metrics_hashes(tmp_path):
    engine = build_mock_engine(recorder=EvidenceLedger(tmp_path))
    receipt = engine.run("inspect and correct the workspace")
    assert receipt.plan["steps"], "task graph recorded"
    assert receipt.actions, "actions recorded"
    assert "steps_executed" in receipt.metrics
    assert set(receipt.hashes) == {"inputs", "plan", "actions"}
    run_dir = tmp_path / "runs" / receipt.run_id
    persisted = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    assert persisted["run_id"] == receipt.run_id
    assert (run_dir / "authorizations.jsonl").exists(), "authorizations persisted before/with run"
    assert content_hash_of(persisted) == receipt.content_hash


def test_chain_continues_across_instances(tmp_path):
    """Two ledger objects on one root simulate two process runs."""
    first = EvidenceLedger(tmp_path)
    r1 = build_mock_engine(recorder=first).run("inspect and correct the workspace")
    second = EvidenceLedger(tmp_path)
    engine2 = build_mock_engine(recorder=second)
    engine2.add_constraint("keep_local", justification="second run differs by constraint")
    r2 = engine2.run("inspect and correct the workspace")
    assert r1.run_id != r2.run_id
    assert r1.parent_hash == "GENESIS"
    assert r2.parent_hash == r1.content_hash
    assert second.records == [r2]  # it wrote only this process's new record
    assert r1 not in second.records  # prior records remain on disk only
    verify_chain([r1, r2])


def test_provenance_captures_code_and_model_versions(tmp_path):
    engine = build_mock_engine(recorder=EvidenceLedger(tmp_path))
    receipt = engine.run("inspect and correct the workspace")
    prov = receipt.provenance
    assert "git_commit" in prov, "code version"
    assert prov["python"]
    assert "model_refs" not in prov  # mock run carries no model id

    led2 = EvidenceLedger(tmp_path / "with-model")
    engine2 = build_mock_engine(recorder=led2)
    original = engine2.recorder.record

    def inject_model(**kwargs):
        kwargs["inputs"] = {**kwargs["inputs"], "model_ref": "KissTheHabit/yolov8n-hituav-thermal-finetune"}
        return original(**kwargs)

    engine2.recorder.record = inject_model  # type: ignore[method-assign]
    receipt2 = engine2.run("inspect and correct the workspace")
    assert receipt2.provenance["model_refs"] == ["KissTheHabit/yolov8n-hituav-thermal-finetune"]


def test_extract_model_refs_is_conservative():
    payload = {
        "model": "org/model-a",
        "nested": {"checkpoint": "org/model-b", "note": "not/a/model"},
        "items": [{"weights": "org/model-c"}, {"model_id": 42}],  # non-str ignored
    }
    assert extract_model_refs(payload) == ["org/model-a", "org/model-b", "org/model-c"]


def test_verify_ledger_clean_and_tampered(tmp_path):
    engine = build_mock_engine(recorder=EvidenceLedger(tmp_path))
    receipt = engine.run("inspect and correct the workspace")
    assert verify_ledger(tmp_path) == []

    receipt_path = tmp_path / "runs" / receipt.run_id / "receipt.json"
    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["metrics"]["steps_executed"] = 999
    receipt_path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    problems = verify_ledger(tmp_path)
    assert any("content hash mismatch" in p for p in problems)


def test_verify_ledger_detects_broken_parent_link(tmp_path):
    engine = build_mock_engine(recorder=EvidenceLedger(tmp_path))
    engine.run("inspect and correct the workspace")
    manifest = tmp_path / "manifest.jsonl"
    entry = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    entry["parent_hash"] = "sha256:ffff"
    manifest.write_text(json.dumps(entry, sort_keys=True) + "\n", encoding="utf-8")
    assert any("parent" in p for p in verify_ledger(tmp_path))


def test_duplicate_deterministic_run_id_is_rejected(tmp_path):
    ledger = EvidenceLedger(tmp_path)
    engine = build_mock_engine(recorder=ledger)
    engine.run("inspect and correct the workspace")
    try:
        build_mock_engine(recorder=ledger).run("inspect and correct the workspace")
    except FileExistsError:
        return
    raise AssertionError("expected FileExistsError for a repeated deterministic run id")


def test_default_demo_path_unchanged():
    engine = build_mock_engine()
    assert isinstance(engine.recorder, FakeRecorder)
