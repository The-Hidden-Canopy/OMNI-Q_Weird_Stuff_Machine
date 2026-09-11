# Portions derived from *The Hidden Canopy LLC* — [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
"""Born-compressed residency: OMNI is never resident in FP32/BF16.

The owner's thesis: the model is **born compressed**. FP32/BF16 is never a
residency state — BF16 appears only as implementation plumbing (accumulators,
norms) inside kernels, never as "the model lives in BF16." A single
``MXFP8 → MXFP4 → MXFP2 → CORE_ONLY`` slider is selected by available memory:

* **MXFP8** — the high-fidelity compressed master (~1 byte/weight).
* **MXFP4** — the same body recompiled lower (~0.5 byte/weight).
* **MXFP2** — extreme residency (~0.25 byte/weight).

Through all three MX tiers the *same* body/capability stays resident; only
the format changes. Below the MXFP2 envelope the neural reasoner is evicted:
OMNI-Q core remains (``CORE_ONLY``), autonomy is ``DEGRADED``, and a novel
task at that level produces ``HOLD`` (queue it — do not attempt degraded
execution) instead of a pretend run.

Residency accounting is deliberately honest:

* Body-resident bytes per tier = ``numel * code_bits / 8`` payload
  + ``ceil(numel / block_size)`` block-scale bytes, per the lowbit codec.
* A small runtime constant (``RUNTIME_BYTES``, 256 MiB) rides along in every
  tier; because it is identical across tiers it never tips tier selection,
  so the envelope fit compares body residency only. It is reported by
  :class:`BodySize` for anyone who wants the full resident picture.
* The real-weight path (:func:`pack_body` / :func:`resident_footprint`) is
  the "recompile to MXFPn" step made concrete through
  ``integrations.qualcomm.lowbit`` — software-dequantize semantics only, the
  same honesty label as the Qualcomm evals: no native low-precision hardware
  is exercised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

GB = 1024 ** 3
"""Envelope units used throughout: 1 GB = 1024³ bytes (GiB)."""

RUNTIME_BYTES = 256 * 1024 ** 2
"""Runtime/plumbing constant, identical across tiers — never tips selection."""

DEFAULT_BODY_PARAMS = 6_000_000_000
"""Default body size: a 6B-parameter neural reasoner."""

RESIDENCY_SCHEMA_VERSION = "1.0.0"

# Owner's constants for formats the codec does not (yet) advertise; the codec
# dicts win when present so residency tracks the codec of record.
_FALLBACK_CODE_BITS = {"mxfp8": 8, "mxfp4": 4, "mxfp2": 2}
_FALLBACK_BLOCK_SIZES = {"mxfp8": 32, "mxfp4": 32, "mxfp2": 32}


def _codec_table(name: str) -> Mapping[str, int]:
    try:
        from integrations.qualcomm.lowbit import lowbit_formats

        return getattr(lowbit_formats, name)
    except Exception:
        return {}


def code_bits(fmt: str) -> int:
    """Payload code width for a residency format (8/4/2)."""
    table = _codec_table("CODE_BITS")
    if fmt in table:
        return int(table[fmt])
    if fmt in _FALLBACK_CODE_BITS:
        return _FALLBACK_CODE_BITS[fmt]
    raise ValueError(f"unknown residency format {fmt!r}")


def block_size(fmt: str) -> int:
    """Block size for a residency format's shared exponents."""
    table = _codec_table("BLOCK_SIZES")
    if fmt in table:
        return int(table[fmt])
    if fmt in _FALLBACK_BLOCK_SIZES:
        return _FALLBACK_BLOCK_SIZES[fmt]
    raise ValueError(f"unknown residency format {fmt!r}")


class ResidencyState(Enum):
    """The residency slider, highest fidelity first.

    Declaration order is fidelity order: MXFP8 > MXFP4 > MXFP2 > CORE_ONLY.
    """

    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    MXFP2 = "mxfp2"
    CORE_ONLY = "core_only"

    @property
    def format(self) -> "str | None":
        """Low-bit format string, or ``None`` at CORE_ONLY (no body resident)."""
        return None if self is ResidencyState.CORE_ONLY else self.value

    @property
    def is_mx(self) -> bool:
        """True while the compressed neural body is resident."""
        return self is not ResidencyState.CORE_ONLY


class AutonomyStatus(Enum):
    """NOMINAL on any MX tier; DEGRADED once the reasoner is evicted."""

    NOMINAL = "NOMINAL"
    DEGRADED = "DEGRADED"


@dataclass(frozen=True)
class ResidencyDecision:
    """One point on the slider: what is resident and why."""

    state: ResidencyState
    autonomy: AutonomyStatus
    format: "str | None"
    resident_bytes: int
    body_params: int
    capability_retained: bool = True
    placement: str = "LOCAL"
    graph_revision: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.name,
            "autonomy": self.autonomy.name,
            "format": self.format,
            "resident_bytes": self.resident_bytes,
            "body_params": self.body_params,
            "capability_retained": self.capability_retained,
            "placement": self.placement,
            "graph_revision": self.graph_revision,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BodySize:
    """Per-tier resident bytes for one body parameter count.

    ``tier_bytes`` maps each MX format to its body residency
    (payload + block scales). ``resident_bytes(fmt)`` adds the runtime
    constant for the full on-device picture.
    """

    body_params: int
    tier_bytes: Mapping[str, int]
    runtime_bytes: int = RUNTIME_BYTES

    def resident_bytes(self, fmt: str) -> int:
        return self.tier_bytes[fmt] + self.runtime_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_params": self.body_params,
            "tier_bytes": dict(self.tier_bytes),
            "runtime_bytes": self.runtime_bytes,
        }


def body_size(body_params: int = DEFAULT_BODY_PARAMS) -> BodySize:
    """Resident bytes per MX tier for ``body_params`` weights.

    For the default 6B body this yields the owner's envelope table
    (body residency, before the tier-constant runtime adder):
    MXFP8 ≈ 6.0 GB, MXFP4 ≈ 3.0 GB, MXFP2 ≈ 1.5 GB.
    """
    if body_params <= 0:
        raise ValueError("body_params must be positive")
    tiers: dict[str, int] = {}
    for state in (ResidencyState.MXFP8, ResidencyState.MXFP4, ResidencyState.MXFP2):
        fmt = state.format
        payload = body_params * code_bits(fmt) // 8
        scales = -(-body_params // block_size(fmt))
        tiers[fmt] = payload + scales
    return BodySize(body_params=body_params, tier_bytes=tiers)


def select_residency(
    available_bytes: int,
    body_params: int = DEFAULT_BODY_PARAMS,
    *,
    hysteresis_bytes: int = 0,
    current: "ResidencyState | None" = None,
) -> ResidencyDecision:
    """Pure slider selection: the HIGHEST-fidelity tier whose body residency
    fits ``available_bytes``, else CORE_ONLY.

    ``hysteresis_bytes`` damps flapping: when ``current`` is an MX tier and
    the envelope has shrunk past a lower tier's bytes but not more than
    ``hysteresis_bytes`` below the *current* tier's bytes, the current tier
    is kept (recompiling a body is expensive; a trickle of memory pressure
    shouldn't bounce the graph). Upward moves (more memory appeared) are
    always taken immediately. Default 0 = no hysteresis, pure envelope fit.
    """
    if available_bytes < 0:
        raise ValueError("available_bytes must be non-negative")
    sizes = body_size(body_params)

    chosen = ResidencyState.CORE_ONLY
    for state in (ResidencyState.MXFP8, ResidencyState.MXFP4, ResidencyState.MXFP2):
        if sizes.tier_bytes[state.format] <= available_bytes:
            chosen = state
            break

    if (
        current is not None
        and current.is_mx
        and _fidelity(chosen) > _fidelity(current)
        and sizes.tier_bytes[current.format] - hysteresis_bytes <= available_bytes
    ):
        chosen = current

    if chosen is ResidencyState.CORE_ONLY:
        return ResidencyDecision(
            state=chosen,
            autonomy=AutonomyStatus.DEGRADED,
            format=None,
            resident_bytes=0,
            body_params=body_params,
            reason=(
                f"envelope {available_bytes} B below MXFP2 body residency "
                f"({sizes.tier_bytes['mxfp2']} B): neural reasoner evicted, "
                "OMNI-Q core remains; novel tasks HOLD"
            ),
        )

    reason = (
        f"envelope {available_bytes} B fits {chosen.format} body residency "
        f"({sizes.tier_bytes[chosen.format]} B): body recompiled to {chosen.format}"
    )
    if (
        current is not None
        and current.is_mx
        and chosen is current
        and sizes.tier_bytes[current.format] > available_bytes
    ):
        reason = (
            f"envelope {available_bytes} B is below {current.format} body residency "
            f"({sizes.tier_bytes[current.format]} B) but within hysteresis "
            f"({hysteresis_bytes} B): staying at {current.format}, no recompile"
        )
    return ResidencyDecision(
        state=chosen,
        autonomy=AutonomyStatus.NOMINAL,
        format=chosen.format,
        resident_bytes=sizes.tier_bytes[chosen.format],
        body_params=body_params,
        reason=reason,
    )


def _fidelity(state: ResidencyState) -> int:
    order = (
        ResidencyState.MXFP8,
        ResidencyState.MXFP4,
        ResidencyState.MXFP2,
        ResidencyState.CORE_ONLY,
    )
    return order.index(state)


@dataclass(frozen=True)
class ResidencyTransition:
    """Append-only log entry. ``trigger`` is ``memory_change`` |
    ``novel_task`` | ``restore``."""

    from_state: ResidencyState
    to_state: ResidencyState
    trigger: str
    timestamp: str
    envelope_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_state": self.from_state.name,
            "to_state": self.to_state.name,
            "trigger": self.trigger,
            "timestamp": self.timestamp,
            "envelope_bytes": self.envelope_bytes,
        }


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResidencyManager:
    """Stateful slider: current decision, graph-revision lineage, and an
    append-only transition log.

    Every tier change appends a :class:`ResidencyTransition` and bumps
    ``graph_revision`` — the owner's "GRAPH REVISION 18 → 19" lineage: a
    recompiled body is a new graph. Non-tier events (a HOLD at CORE_ONLY)
    are logged but do not bump the revision.
    """

    def __init__(
        self,
        available_bytes: "int | None" = None,
        *,
        body_params: int = DEFAULT_BODY_PARAMS,
        hysteresis_bytes: int = 0,
    ) -> None:
        self.body_params = body_params
        self.hysteresis_bytes = hysteresis_bytes
        self._sizes = body_size(body_params)
        self._transitions: list[ResidencyTransition] = []
        self._revision = 0
        if available_bytes is None:
            available_bytes = self._sizes.tier_bytes["mxfp8"]
            self._current = replace(
                select_residency(available_bytes, body_params),
                reason="initial residency: full envelope assumed; body compiled to MXFP8",
            )
        else:
            self._current = select_residency(available_bytes, body_params)
        self._available = int(available_bytes)

    @property
    def current(self) -> ResidencyDecision:
        return self._current

    @property
    def autonomy(self) -> AutonomyStatus:
        return self._current.autonomy

    @property
    def graph_revision(self) -> int:
        return self._revision

    @property
    def transitions(self) -> tuple[ResidencyTransition, ...]:
        return tuple(self._transitions)

    @property
    def available_bytes(self) -> int:
        return self._available

    def update_envelope(self, available_bytes: int) -> ResidencyDecision:
        """Recompute the slider under a new memory envelope."""
        if available_bytes < 0:
            raise ValueError("available_bytes must be non-negative")
        self._available = int(available_bytes)
        return self._apply(
            select_residency(
                self._available,
                self.body_params,
                hysteresis_bytes=self.hysteresis_bytes,
                current=self._current.state,
            ),
            trigger="memory_change",
        )

    def admit_task(self, novel: bool) -> str:
        """``"EXECUTE"`` or ``"HOLD"``.

        Any MX tier executes. At CORE_ONLY a *novel* task is HOLDed (queued,
        never degraded-executed) and the decision is logged; a *routine* task
        still executes — the core handles it under DEGRADED autonomy.
        Neither changes the tier nor bumps the graph revision.
        """
        if self._current.state.is_mx:
            return "EXECUTE"
        if novel:
            self._transitions.append(
                ResidencyTransition(
                    from_state=ResidencyState.CORE_ONLY,
                    to_state=ResidencyState.CORE_ONLY,
                    trigger="novel_task",
                    timestamp=_utcnow(),
                    envelope_bytes=self._available,
                )
            )
            return "HOLD"
        return "EXECUTE"

    def evict_reasoner(self) -> ResidencyDecision:
        """Force the slider to CORE_ONLY (reasoner eviction)."""
        if not self._current.state.is_mx:
            raise ValueError("illegal transition: reasoner already evicted (CORE_ONLY)")
        decision = select_residency(0, self.body_params)
        return self._apply(decision, trigger="memory_change")

    def restore_reasoner(self, available_bytes: int) -> ResidencyDecision:
        """Bring the neural body back at the highest tier the envelope fits.

        Illegal unless currently CORE_ONLY and the envelope can hold at
        least MXFP2 body residency.
        """
        if self._current.state.is_mx:
            raise ValueError(
                f"illegal transition: reasoner already resident at "
                f"{self._current.state.name}"
            )
        target = select_residency(available_bytes, self.body_params)
        if not target.state.is_mx:
            raise ValueError(
                f"illegal transition: envelope {available_bytes} B cannot restore "
                f"reasoner — need at least MXFP2 body residency "
                f"({self._sizes.tier_bytes['mxfp2']} B)"
            )
        self._available = int(available_bytes)
        return self._apply(target, trigger="restore")

    def _apply(self, decision: ResidencyDecision, *, trigger: str) -> ResidencyDecision:
        if decision.state is not self._current.state:
            self._transitions.append(
                ResidencyTransition(
                    from_state=self._current.state,
                    to_state=decision.state,
                    trigger=trigger,
                    timestamp=_utcnow(),
                    envelope_bytes=self._available,
                )
            )
            self._revision += 1
        self._current = replace(decision, graph_revision=self._revision)
        return self._current

    # ------------------------------------------------------------------ #
    # persistence                                                         #
    # ------------------------------------------------------------------ #
    def to_receipt(self) -> dict[str, Any]:
        """JSON-serializable residency receipt: final decision + lineage."""
        return {
            "schema_version": RESIDENCY_SCHEMA_VERSION,
            "body_params": self.body_params,
            "available_bytes": self._available,
            "graph_revision": self._revision,
            "decision": self._current.to_dict(),
            "transitions": [t.to_dict() for t in self._transitions],
            "labels": {
                "born_compressed": "the model never resides in FP32/BF16; BF16 is plumbing only (accumulators, norms)",
                "capability_retained": "the same body/capability stays resident across all MX tiers; only the format changes",
                "core_only_semantics": "below the MXFP2 envelope the neural reasoner is evicted; OMNI-Q core remains and novel tasks HOLD",
                "software_dequantize_only": "no native low-precision hardware exercised",
            },
        }

    def write_receipt(
        self,
        bundle_root: "str | Path",
        *,
        run_id: "str | None" = None,
    ) -> Path:
        """Write the transition log + final decision into a validated
        evidence bundle under ``bundle_root``. Imports
        :mod:`omni_q.evidence_bundle` lazily."""
        from .evidence_bundle import EvidenceBundleWriter

        receipt = self.to_receipt()
        transitions_jsonl = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in receipt["transitions"]
        )
        if run_id is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"residency_{stamp}_{self._revision:04d}"
        writer = EvidenceBundleWriter(bundle_root)
        return writer.write_run(
            run_id=run_id,
            metadata={
                "body_params": self.body_params,
                "graph_revision": self._revision,
                "final_state": receipt["decision"]["state"],
                "autonomy": receipt["decision"]["autonomy"],
            },
            counts={"transitions": len(self._transitions)},
            artifacts={
                "decision.json": (
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
                ).encode("utf-8"),
                "transitions.jsonl": transitions_jsonl.encode("utf-8"),
            },
        )


# --------------------------------------------------------------------------- #
# Real-weight path ("recompile to MXFPn" made concrete)                        #
# --------------------------------------------------------------------------- #
def pack_body(
    weights: Mapping[str, np.ndarray],
    fmt: str,
) -> "dict[str, Any]":
    """Compress a real weight dict to low-bit masters via the lowbit codec.

    This is the concrete form of "recompile the body to MXFPn": each tensor
    becomes a :class:`integrations.qualcomm.lowbit.PackedTensor`. Software-
    dequantize semantics only — no native low-precision hardware exercised,
    the same honesty label as the Qualcomm evals.
    """
    from integrations.qualcomm.lowbit import quantize_tensor

    return {
        name: quantize_tensor(np.ascontiguousarray(array, dtype=np.float32), fmt)
        for name, array in weights.items()
    }


def resident_footprint(packed: Mapping[str, Any]) -> int:
    """Total resident bytes of a packed body (payloads + block scales)."""
    return sum(int(tensor.bytes_total) for tensor in packed.values())
