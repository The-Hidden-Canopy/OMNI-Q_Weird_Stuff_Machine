"""Re-verifiable evidence bundles (ported from open_world_model_harness)."""

from __future__ import annotations

import json

import pytest

from omni_q.evidence_bundle import (
    EvidenceBundleWriter,
    validate_evidence_bundle,
)


def _write_basic_bundle(root, run_id="run-001"):
    writer = EvidenceBundleWriter(root)
    return writer.write_run(
        run_id=run_id,
        metadata={"scenario": "table-setting", "seed": 7},
        counts={"detections": 2},
        artifacts={
            "detections.jsonl": b'{"object": "plate"}\n{"object": "cup"}\n',
            "frame.png": b"\x89PNG\r\n\x1a\n fake-bytes",
        },
    )


def test_write_validate_round_trip(tmp_path):
    run_dir = _write_basic_bundle(tmp_path)
    ok, errors = validate_evidence_bundle(run_dir)
    assert ok, errors

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run-001"
    assert manifest["metadata"]["scenario"] == "table-setting"
    assert manifest["counts"] == {"detections": 2}
    assert set(manifest["artifact_files"]) == {
        "manifest.json", "detections.jsonl", "frame.png", "checksums.json",
    }
    checksums = json.loads((run_dir / "checksums.json").read_text(encoding="utf-8"))
    assert "checksums.json" not in checksums


def test_overwrite_is_refused(tmp_path):
    _write_basic_bundle(tmp_path)
    with pytest.raises(FileExistsError):
        _write_basic_bundle(tmp_path)


def test_traversal_artifact_name_is_rejected(tmp_path):
    writer = EvidenceBundleWriter(tmp_path)
    for bad in ("../escape.txt", "sub/../../escape.txt", "/abs/path.txt", "a/./b.txt"):
        with pytest.raises(ValueError):
            writer.write_run(run_id="run-x", artifacts={bad: b"payload"})


def test_reserved_artifact_names_are_rejected(tmp_path):
    writer = EvidenceBundleWriter(tmp_path)
    with pytest.raises(ValueError):
        writer.write_run(run_id="run-x", artifacts={"manifest.json": b"{}"})


def test_tampered_artifact_fails_validation(tmp_path):
    run_dir = _write_basic_bundle(tmp_path)
    (run_dir / "detections.jsonl").write_bytes(b'{"object": "TAMPERED"}\n')
    ok, errors = validate_evidence_bundle(run_dir)
    assert not ok
    assert any("checksum mismatch: detections.jsonl" in e for e in errors)


def test_extra_file_fails_validation(tmp_path):
    run_dir = _write_basic_bundle(tmp_path)
    (run_dir / "stray.txt").write_text("not listed\n", encoding="utf-8")
    ok, errors = validate_evidence_bundle(run_dir)
    assert not ok
    assert any("unlisted artifact files" in e for e in errors)


def test_missing_file_fails_validation(tmp_path):
    run_dir = _write_basic_bundle(tmp_path)
    (run_dir / "frame.png").unlink()
    ok, errors = validate_evidence_bundle(run_dir)
    assert not ok
    assert any("missing artifact: frame.png" in e for e in errors)


def test_non_finite_float_rejected_by_serialize(tmp_path):
    writer = EvidenceBundleWriter(tmp_path)
    for index, bad in enumerate((float("nan"), float("inf"), float("-inf"))):
        try:
            writer.write_run(run_id=f"run-bad-{index}", metadata={"score": bad})
        except ValueError:
            continue
        raise AssertionError("expected ValueError for a non-finite artifact number")


def test_count_cross_check_detects_swapped_jsonl(tmp_path):
    run_dir = _write_basic_bundle(tmp_path)
    (run_dir / "detections.jsonl").write_text(
        '{"object": "plate"}\n{"object": "cup"}\n{"object": "fork"}\n',
        encoding="utf-8",
    )
    # keep the checksum consistent so only the count cross-check can fire
    import hashlib
    checksums = json.loads((run_dir / "checksums.json").read_text(encoding="utf-8"))
    checksums["detections.jsonl"] = hashlib.sha256(
        (run_dir / "detections.jsonl").read_bytes()
    ).hexdigest()
    (run_dir / "checksums.json").write_text(
        json.dumps(checksums, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    ok, errors = validate_evidence_bundle(run_dir)
    assert not ok
    assert any("detections count does not match manifest" in e for e in errors)


def test_bytes_and_path_artifacts_both_written(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"from-a-file")
    run_dir = EvidenceBundleWriter(tmp_path / "out").write_run(
        run_id="run-mixed",
        artifacts={"inline.bin": b"inline-bytes", "copied.bin": source},
    )
    assert (run_dir / "inline.bin").read_bytes() == b"inline-bytes"
    assert (run_dir / "copied.bin").read_bytes() == b"from-a-file"
    ok, errors = validate_evidence_bundle(run_dir)
    assert ok, errors


def test_safe_run_id_suffixes_directory(tmp_path):
    run_dir = _write_basic_bundle(tmp_path, run_id="run/with slashes")
    assert run_dir.name.startswith("run_with_slashes-")
    assert run_dir.parent == tmp_path
