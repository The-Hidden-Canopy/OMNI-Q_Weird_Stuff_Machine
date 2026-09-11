# Portions derived from *The Hidden Canopy LLC* — [`open_world_model_harness`](https://github.com/The-Hidden-Canopy/open_world_model_harness). Used with permission.
"""Re-verifiable evidence bundles.

A bundle is one immutable run directory under a caller-specified root
(mirroring the ``evidence/`` layout): ``manifest.json`` describing the run,
the caller's artifact files, and ``checksums.json`` written *last* so a
complete directory is always internally consistent. Validation is
fail-closed: it re-checks checksum set equality, missing *and* extra files,
per-file SHA-256 digests, and any line-count cross-references the caller
recorded in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

EVIDENCE_SCHEMA_VERSION = "1.0.0"

RESERVED_ARTIFACT_NAMES = frozenset({"manifest.json", "checksums.json"})


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    run_id: str
    created_at: str

    python_version: str
    platform: str

    metadata: Mapping[str, Any]
    counts: Mapping[str, int]
    artifact_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "python_version": self.python_version,
            "platform": self.platform,
            "metadata": dict(self.metadata),
            "counts": dict(self.counts),
            "artifact_files": list(self.artifact_files),
        }


class EvidenceBundleWriter:
    """Write one immutable evidence bundle directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write_run(
        self,
        *,
        run_id: str,
        metadata: Mapping[str, Any] | None = None,
        counts: Mapping[str, int] | None = None,
        artifacts: Mapping[str, "bytes | str | Path"] | None = None,
    ) -> Path:
        """Create ``<root>/<safe run_id>/`` with manifest, artifacts, checksums.

        ``artifacts`` maps a bundle-relative file name to its content: ``bytes``
        are written verbatim, ``str``/``Path`` values are read as source files.
        Refuses to overwrite an existing run directory and rejects artifact
        names that could escape it (path traversal) or shadow the manifest or
        checksum files.
        """
        run_dir = self.root / _safe_name(run_id)
        if run_dir.exists():
            raise FileExistsError(f"evidence bundle already exists: {run_dir}")

        artifact_map = dict(artifacts or {})
        for name in artifact_map:
            if not _is_safe_artifact_name(name):
                raise ValueError(f"invalid artifact path: {name}")
            if name in RESERVED_ARTIFACT_NAMES:
                raise ValueError(f"artifact name is reserved: {name}")
        artifact_names = tuple(artifact_map)

        manifest = RunManifest(
            schema_version=EVIDENCE_SCHEMA_VERSION,
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            metadata=dict(metadata or {}),
            counts=dict(counts or {}),
            artifact_files=("manifest.json",) + artifact_names + ("checksums.json",),
        )

        run_dir.mkdir(parents=True, exist_ok=False)

        _write_json(run_dir / "manifest.json", manifest.to_dict())
        for name, content in artifact_map.items():
            artifact_path = run_dir / name
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                artifact_path.write_bytes(content)
            else:
                artifact_path.write_bytes(Path(content).read_bytes())

        checksums = {
            name: _sha256(run_dir / name)
            for name in manifest.artifact_files
            if name != "checksums.json"
        }
        _write_json(run_dir / "checksums.json", checksums)
        return run_dir


def validate_evidence_bundle(
    run_dir: str | Path,
) -> tuple[bool, tuple[str, ...]]:
    """Fail-closed re-verification of a bundle directory.

    Returns ``(ok, errors)``; any deviation — tampered bytes, missing or
    unlisted files, a checksum set that disagrees with the manifest, or a
    line count that disagrees with a manifest ``counts`` entry — makes ``ok``
    false rather than being waved through.
    """
    path = Path(run_dir)
    errors: list[str] = []
    manifest_path = path / "manifest.json"
    checksums_path = path / "checksums.json"

    if not manifest_path.exists():
        errors.append("manifest.json missing")
    if not checksums_path.exists():
        errors.append("checksums.json missing")
    if errors:
        return False, tuple(errors)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, (f"invalid manifest or checksums JSON: {exc}",)
    if not isinstance(manifest, dict) or not isinstance(checksums, dict):
        return False, ("manifest and checksums must be JSON objects",)

    if manifest.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        errors.append("unsupported evidence schema version")

    artifact_files = manifest.get("artifact_files", [])
    if (
        not isinstance(artifact_files, list)
        or any(not isinstance(item, str) for item in artifact_files)
        or len(set(artifact_files)) != len(artifact_files)
    ):
        errors.append("manifest artifact file list must contain unique paths")
        artifact_files = []
    listed_files = set(artifact_files)
    checksum_files = set(checksums)
    if checksum_files != listed_files - {"checksums.json"}:
        errors.append("checksum file set does not match manifest artifact file set")
    for filename in artifact_files:
        if not _is_safe_artifact_name(filename):
            errors.append(f"invalid artifact path: {filename}")
            continue
        if not (path / filename).exists():
            errors.append(f"missing artifact: {filename}")

    actual_files = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file()
    }
    if actual_files != listed_files:
        extras = sorted(actual_files - listed_files)
        missing = sorted(listed_files - actual_files)
        if extras:
            errors.append(f"unlisted artifact files: {extras}")
        if missing:
            errors.append(f"manifest files missing on disk: {missing}")

    for filename, expected_hash in checksums.items():
        if not _is_safe_artifact_name(filename):
            errors.append(f"invalid checksum path: {filename}")
            continue
        file_path = path / filename
        if not file_path.exists():
            continue
        if _sha256(file_path) != expected_hash:
            errors.append(f"checksum mismatch: {filename}")

    if not errors:
        counts = manifest.get("counts", {})
        if not isinstance(counts, dict):
            errors.append("manifest counts must be a JSON object")
            counts = {}
        for count_name, expected in counts.items():
            jsonl_path = path / f"{count_name}.jsonl"
            if not jsonl_path.exists():
                continue
            try:
                actual = len(load_jsonl(jsonl_path))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"invalid count file {jsonl_path.name}: {exc}")
                continue
            if actual != int(expected):
                errors.append(f"{count_name} count does not match manifest")

    return not errors, tuple(errors)


def load_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(value)
    return tuple(rows)


def _write_json(path: Path, value: Any) -> None:
    serialized = json.dumps(
        _serialize(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    path.write_text(serialized + "\n", encoding="utf-8")


def _serialize(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _serialize(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_serialize(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("artifact numbers must be finite")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("run id must be a string")
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789-_"
    )
    sanitized = "".join(
        character if character in allowed else "_"
        for character in value
    )
    if not sanitized:
        raise ValueError("run id cannot become an empty path")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{sanitized}-{digest}"


def _is_safe_artifact_name(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    return not path.is_absolute() and all(
        part not in {"", ".", ".."} for part in normalized.split("/")
    )
