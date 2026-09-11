"""Independent structural replay checks for contact-handoff receipts.

This module intentionally does not import the MuJoCo controller or the
authoritative contact world.  It consumes only a serialized receipt and
recomputes the receipt hash plus the evidence invariants needed to decide
whether a recorded handoff is internally replayable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


CONTACT_HANDOFF_MODE = "simulation-contact-handoff"
CONTACT_HANDOFF_SCHEMA_VERSION = 1
_REQUIRED_PHASES = (
    "initial_settle",
    "left_approach",
    "left_descend",
    "left_grasp",
    "left_lift_transfer",
    "right_approach",
    "right_descend",
    "right_grasp",
    "left_release",
    "right_retreat",
    "right_place",
    "right_release",
    "final_settle",
)


class ContactReplayRejected(ValueError):
    """A serialized receipt failed an independent replay invariant."""


def _canonical_hash(payload: dict[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "content_hash"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_payload(receipt: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(receipt, (str, Path)):
        try:
            payload = json.loads(Path(receipt).read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as exc:
            raise ContactReplayRejected("receipt is not valid JSON") from exc
    else:
        try:
            payload = dict(receipt)
        except (TypeError, ValueError) as exc:
            raise ContactReplayRejected("receipt is not an object") from exc
    if not isinstance(payload, dict):
        raise ContactReplayRejected("receipt is not an object")
    return payload


def replay_contact_handoff_receipt(
    receipt: dict[str, Any] | str | Path,
    *,
    expected_mjcf_sha256: str | None = None,
) -> dict[str, Any]:
    """Recompute and independently validate one serialized handoff receipt.

    The return value is a compact verification record suitable for a report;
    it does not grant authority or mutate any world state.
    """
    payload = _load_payload(receipt)
    if payload.get("schema_version") != CONTACT_HANDOFF_SCHEMA_VERSION:
        raise ContactReplayRejected("unsupported contact-handoff receipt schema")
    if payload.get("mode") != CONTACT_HANDOFF_MODE:
        raise ContactReplayRejected("receipt is not contact-handoff evidence")
    mjcf_hash = payload.get("mjcf_sha256")
    if not isinstance(mjcf_hash, str) or not mjcf_hash:
        raise ContactReplayRejected("receipt is missing an MJCF hash")
    if expected_mjcf_sha256 is not None and mjcf_hash != expected_mjcf_sha256:
        raise ContactReplayRejected("receipt MJCF hash does not match the replay input")
    expected_hash = _canonical_hash(payload)
    if payload.get("content_hash") != expected_hash:
        raise ContactReplayRejected("receipt content hash mismatch")

    controller = payload.get("controller")
    transitions = payload.get("contact_transitions")
    phase_timings = payload.get("phase_timings")
    samples = payload.get("state_samples")
    final_pose = payload.get("final_cup_pose")
    if not isinstance(controller, dict) or not isinstance(transitions, list):
        raise ContactReplayRejected("receipt is missing controller/contact evidence")
    if not isinstance(phase_timings, list) or not isinstance(samples, list):
        raise ContactReplayRejected("receipt is missing timing/state samples")
    if not isinstance(final_pose, dict):
        raise ContactReplayRejected("receipt is missing the final cup pose")
    if not samples or not transitions:
        raise ContactReplayRejected("receipt has no replay samples")
    if any(not isinstance(item, dict) for item in phase_timings + transitions):
        raise ContactReplayRejected("receipt contains a non-object phase or contact record")

    for field, width in (("position", 3), ("quaternion", 4), ("linear_velocity", 3)):
        values = final_pose.get(field)
        if not isinstance(values, list) or len(values) != width:
            raise ContactReplayRejected(f"final cup pose is missing {field}")
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
            raise ContactReplayRejected(f"final cup pose contains an invalid {field}")

    phases = tuple(item.get("phase") for item in phase_timings)
    missing_phases = [phase for phase in _REQUIRED_PHASES if phase not in phases]
    if missing_phases:
        raise ContactReplayRejected(f"receipt is missing phases: {','.join(missing_phases)}")
    if tuple(phases.index(phase) for phase in _REQUIRED_PHASES) != tuple(sorted(phases.index(phase) for phase in _REQUIRED_PHASES)):
        raise ContactReplayRejected("receipt phases are out of order")

    transition_phases = {str(item.get("phase")) for item in transitions}
    required_contact_phases = {"left_grasp", "right_grasp", "left_release", "right_release"}
    missing = [
        phase for phase in required_contact_phases
        if not any(
            observed == phase
            or (phase == "right_release" and observed.startswith("right_release"))
            for observed in transition_phases
        )
    ]
    if missing:
        missing = sorted(missing)
        raise ContactReplayRejected(f"receipt is missing contact transitions: {','.join(missing)}")

    max_force = 0.0
    min_joint_margin = float("inf")
    for sample in samples:
        if not isinstance(sample, dict):
            raise ContactReplayRejected("state sample is not an object")
        for field in ("max_gripper_force_n", "min_joint_margin_rad"):
            if field not in sample:
                raise ContactReplayRejected(f"state sample is missing {field}")
            try:
                value = float(sample[field])
            except (TypeError, ValueError) as exc:
                raise ContactReplayRejected(f"state sample contains invalid {field}") from exc
            if not math.isfinite(value):
                raise ContactReplayRejected(f"state sample contains invalid {field}")
            if field == "max_gripper_force_n":
                max_force = max(max_force, value)
            else:
                min_joint_margin = min(min_joint_margin, value)
    try:
        configured_force = float(controller.get("max_contact_force_n", 0.0))
        configured_margin = float(controller.get("joint_limit_margin_rad", 0.0))
    except (TypeError, ValueError) as exc:
        raise ContactReplayRejected("controller bounds are not numeric") from exc
    if not math.isfinite(configured_force) or not math.isfinite(configured_margin):
        raise ContactReplayRejected("controller bounds are not finite")
    if configured_force <= 0.0 or max_force > configured_force:
        raise ContactReplayRejected("receipt exceeds its configured contact-force bound")
    if configured_margin <= 0.0 or min_joint_margin < 0.0:
        raise ContactReplayRejected("receipt contains an invalid joint-limit margin")

    final_sample = samples[-1]
    if payload.get("success"):
        if payload.get("final_owner") is not None or not payload.get("final_stable"):
            raise ContactReplayRejected("successful receipt has non-final ownership/stability")
        if final_sample.get("ownership") is not None or not final_sample.get("cup_on_table"):
            raise ContactReplayRejected("successful receipt final sample is not released/stable")
    return {
        "ok": True,
        "mode": payload["mode"],
        "seed": payload.get("seed"),
        "success": bool(payload.get("success")),
        "content_hash": payload["content_hash"],
        "checks": {
            "content_hash": True,
            "phase_order": True,
            "contact_phases": True,
            "force_bound": True,
            "joint_margin": True,
            "final_state": bool(payload.get("success")),
        },
    }


def replay_contact_handoff_directory(
    root: str | Path,
    *,
    expected_mjcf_sha256: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Replay every retained JSON receipt in ``root`` without controller code.

    The report preserves rejected files instead of dropping them.  When
    ``output_path`` is supplied it is written once, atomically, and an
    existing report is never overwritten.  This is a structural replay
    report, not a second authority or a promotion decision.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise ContactReplayRejected(f"receipt directory does not exist: {root_path}")
    paths = sorted(
        path for path in root_path.glob("*.json")
        if path.name != "report.json"
    )
    entries: list[dict[str, Any]] = []
    outcomes = {"accepted": 0, "rejected": 0, "successful": 0, "failed": 0}
    for path in paths:
        try:
            replay = replay_contact_handoff_receipt(
                path, expected_mjcf_sha256=expected_mjcf_sha256,
            )
        except (ContactReplayRejected, OSError, ValueError) as exc:
            outcomes["rejected"] += 1
            entries.append({"receipt": path.name, "accepted": False, "reason": str(exc)})
            continue
        outcomes["accepted"] += 1
        outcomes["successful" if replay["success"] else "failed"] += 1
        entries.append({"receipt": path.name, "accepted": True, **replay})

    report = {
        "schema_version": CONTACT_HANDOFF_SCHEMA_VERSION,
        "mode": CONTACT_HANDOFF_MODE,
        "kind": "independent contact-handoff structural replay; not a promotion claim",
        "root": str(root_path),
        "receipts_seen": len(paths),
        "outcomes": outcomes,
        "receipts": entries,
    }
    if output_path is not None:
        target = Path(output_path)
        if target.exists():
            raise ContactReplayRejected("refusing to overwrite an existing replay report")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    return report
