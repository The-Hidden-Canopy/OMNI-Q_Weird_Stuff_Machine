"""Durable, append-only Omni Q evidence ledger.

The in-memory recorder is useful for contract tests.  This recorder is the
demo-facing boundary: it fsyncs an authorization before an action is allowed
to execute, stores the final run receipt, and links each run in a manifest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .contracts import ActionAuthorization, ReceiptRecord, PlanGraph, content_hash_of, sha256_of
from .provenance import build_provenance

MODEL_REF_KEYS = frozenset({"model", "model_id", "model_ref", "checkpoint", "weights"})


def extract_model_refs(obj: Any) -> list[str]:
    """Best-effort model identifiers (e.g. Hugging Face ids) found in a payload.

    Conservative on purpose: only explicit key names count, so arbitrary
    strings with slashes don't pollute provenance.
    """
    refs: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in MODEL_REF_KEYS and isinstance(value, str):
                refs.add(value)
            else:
                refs.update(extract_model_refs(value))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            refs.update(extract_model_refs(item))
    return sorted(refs)


def verify_ledger(root: str | Path) -> list[str]:
    """Check every receipt listed in ``manifest.jsonl``: content hash recomputes
    and the parent chain is unbroken. Returns human-readable problems (empty ==
    clean) so acceptance tooling (OQ-038) can assert on it."""
    root = Path(root)
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        return [f"no manifest at {manifest}"]
    problems: list[str] = []
    prev_hash: str | None = None
    for index, line in enumerate(manifest.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        entry = json.loads(line)
        receipt_path = root / entry["path"]
        if not receipt_path.exists():
            problems.append(f"entry {index} ({entry['run_id']}): missing {entry['path']}")
            continue
        record = json.loads(receipt_path.read_text(encoding="utf-8"))
        if content_hash_of(record) != entry["content_hash"]:
            problems.append(f"entry {index} ({entry['run_id']}): content hash mismatch (tampered or moved)")
        if prev_hash is None and entry["parent_hash"] != "GENESIS":
            problems.append(f"entry {index} ({entry['run_id']}): genesis entry claims a parent")
        expected_parent = "GENESIS" if prev_hash is None else prev_hash
        if entry["parent_hash"] != expected_parent:
            problems.append(f"entry {index} ({entry['run_id']}): parent link broken")
        prev_hash = entry["content_hash"]
    return problems


class EvidenceLedger:
    """Local parent-chained receipts with fail-closed authorization writes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.runs_root = self.root / "runs"
        self.manifest_path = self.root / "manifest.jsonl"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.records: list[ReceiptRecord] = []
        self.authorizations: list[ActionAuthorization] = []
        self._last_hash = self._load_last_hash()

    def _load_last_hash(self) -> str:
        if not self.manifest_path.exists():
            return "GENESIS"
        last = "GENESIS"
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            last = entry["content_hash"]
        return last

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str) + "\n"
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def authorize(self, authorization: ActionAuthorization) -> ActionAuthorization:
        """Persist and finalize an action decision before a driver can run."""
        base = authorization.as_dict()
        finalized = ActionAuthorization(
            run_id=authorization.run_id,
            step_id=authorization.step_id,
            op=authorization.op,
            verdict=authorization.verdict,
            reason=authorization.reason,
            state_revision=authorization.state_revision,
            envelope_digest=authorization.envelope_digest,
            content_hash=content_hash_of(base),
        )
        self._append_jsonl(
            self.runs_root / finalized.run_id / "authorizations.jsonl",
            finalized.as_dict(),
        )
        self.authorizations.append(finalized)
        return finalized

    def record(
        self,
        run_id: str,
        goal: str,
        inputs: dict[str, Any],
        plan: PlanGraph,
        actions: list[dict[str, Any]],
        metrics: dict[str, Any],
        decisions: list[dict[str, Any]] | None = None,
        rejected: list[dict[str, Any]] | None = None,
    ) -> ReceiptRecord:
        run_dir = self.runs_root / run_id
        receipt_path = run_dir / "receipt.json"
        if receipt_path.exists():
            raise FileExistsError(f"evidence already exists for deterministic run {run_id}")

        plan_d = plan.as_dict()
        provenance = build_provenance("omni_q.evidence")
        model_refs = extract_model_refs(inputs)
        if model_refs:
            provenance["model_refs"] = model_refs
        base = ReceiptRecord(
            run_id=run_id,
            goal=goal,
            inputs=inputs,
            plan=plan_d,
            actions=tuple(actions),
            metrics=metrics,
            hashes={
                "inputs": sha256_of(inputs),
                "plan": sha256_of(plan_d),
                "actions": sha256_of(actions),
            },
            decisions=tuple(decisions or ()),
            rejected=tuple(rejected or ()),
            provenance=provenance,
            parent_hash=self._last_hash,
        )
        receipt = ReceiptRecord(
            **{**base.as_dict(), "content_hash": content_hash_of(base.as_dict())}
        )
        self._write_json(receipt_path, receipt.as_dict())
        self._append_jsonl(
            self.manifest_path,
            {
                "run_id": receipt.run_id,
                "path": str(receipt_path.relative_to(self.root)).replace("\\", "/"),
                "parent_hash": receipt.parent_hash,
                "content_hash": receipt.content_hash,
            },
        )
        self.records.append(receipt)
        self._last_hash = receipt.content_hash
        return receipt
