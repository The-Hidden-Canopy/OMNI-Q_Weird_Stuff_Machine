"""Born-compressed residency slider — envelope selection, hysteresis, lineage.

Covers tier-boundary selection (exactly at / 1 byte under / between / way
below), the owner's 6B envelope table (8 GB → MXFP8, 3 GB → MXFP4,
1.8 GB → MXFP2, 1.2 GB → CORE_ONLY/DEGRADED), hysteresis anti-flapping,
graph-revision bumps on tier change only, HOLD-vs-EXECUTE task admission at
CORE_ONLY, illegal-transition guards, the optional real-weight pack path
via the lowbit codec (skipped while ``mxfp8`` is absent from the codec's
FORMATS — a concurrent agent owns that file), and receipt JSON round-trip.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from omni_q.residency import (
    DEFAULT_BODY_PARAMS,
    GB,
    RUNTIME_BYTES,
    AutonomyStatus,
    ResidencyManager,
    ResidencyState,
    body_size,
    pack_body,
    resident_footprint,
    select_residency,
)

SMALL = 32  # body_params for boundary tests — small enough to eyeball

FORMATS = ()
try:
    from integrations.qualcomm.lowbit import FORMATS as _CODEC_FORMATS

    FORMATS = tuple(_CODEC_FORMATS)
except Exception:  # codec not importable at all yet — pack tests skip
    pass


# --------------------------------------------------------------------------- #
# envelope selection boundaries                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "state",
    [ResidencyState.MXFP8, ResidencyState.MXFP4, ResidencyState.MXFP2],
)
def test_exactly_at_tier_bytes_selects_that_tier(state):
    sizes = body_size(SMALL)
    decision = select_residency(sizes.tier_bytes[state.format], SMALL)
    assert decision.state is state
    assert decision.format == state.format
    assert decision.autonomy is AutonomyStatus.NOMINAL
    assert decision.resident_bytes == sizes.tier_bytes[state.format]


@pytest.mark.parametrize(
    "state, lower",
    [
        (ResidencyState.MXFP8, ResidencyState.MXFP4),
        (ResidencyState.MXFP4, ResidencyState.MXFP2),
        (ResidencyState.MXFP2, ResidencyState.CORE_ONLY),
    ],
)
def test_one_byte_under_tier_drops_to_next(state, lower):
    sizes = body_size(SMALL)
    decision = select_residency(sizes.tier_bytes[state.format] - 1, SMALL)
    assert decision.state is lower


def test_between_tiers_selects_lower_tier():
    sizes = body_size(SMALL)
    between = (sizes.tier_bytes["mxfp4"] + sizes.tier_bytes["mxfp2"]) // 2
    decision = select_residency(between, SMALL)
    assert decision.state is ResidencyState.MXFP2


def test_way_below_envelope_is_core_only():
    sizes = body_size(SMALL)
    decision = select_residency(sizes.tier_bytes["mxfp2"] - 1, SMALL)
    assert decision.state is ResidencyState.CORE_ONLY
    assert decision.format is None
    assert decision.autonomy is AutonomyStatus.DEGRADED
    assert decision.resident_bytes == 0
    # the core remains: capability/body claims stay truthful
    assert decision.capability_retained is True
    assert decision.placement == "LOCAL"


def test_negative_envelope_rejected():
    with pytest.raises(ValueError):
        select_residency(-1, SMALL)


# --------------------------------------------------------------------------- #
# owner's 6B envelope table                                                    #
# --------------------------------------------------------------------------- #
def test_owner_table_6b_defaults():
    assert select_residency(8 * GB).state is ResidencyState.MXFP8
    assert select_residency(3 * GB).state is ResidencyState.MXFP4
    assert select_residency(int(1.8 * GB)).state is ResidencyState.MXFP2
    low = select_residency(int(1.2 * GB))
    assert low.state is ResidencyState.CORE_ONLY
    assert low.autonomy is AutonomyStatus.DEGRADED


def test_6b_tier_bytes_match_owner_envelope_numbers():
    sizes = body_size(DEFAULT_BODY_PARAMS)
    # payload + block scales ≈ owner's MXFP8≈6.0GB / MXFP4≈3.0GB / MXFP2≈1.5GB
    assert sizes.tier_bytes["mxfp8"] == 6_000_000_000 * 1 + 6_000_000_000 // 32
    assert sizes.tier_bytes["mxfp4"] == 6_000_000_000 // 2 + 6_000_000_000 // 32
    assert sizes.tier_bytes["mxfp2"] == 6_000_000_000 // 4 + 6_000_000_000 // 32
    assert sizes.runtime_bytes == RUNTIME_BYTES
    # runtime constant is identical across tiers — it never tips selection
    for fmt in ("mxfp8", "mxfp4", "mxfp2"):
        assert sizes.resident_bytes(fmt) == sizes.tier_bytes[fmt] + RUNTIME_BYTES


# --------------------------------------------------------------------------- #
# hysteresis                                                                   #
# --------------------------------------------------------------------------- #
def test_hysteresis_prevents_flap_on_memory_trickle():
    sizes = body_size(1024)  # big enough that tier_bytes >> hysteresis
    mxfp8_bytes = sizes.tier_bytes["mxfp8"]
    hysteresis = 64
    # envelope slips 1 byte under MXFP8 — without hysteresis it would drop
    dropped = select_residency(mxfp8_bytes - 1, 1024, current=None)
    assert dropped.state is ResidencyState.MXFP4
    # with hysteresis and current=MXFP8 it stays put...
    held = select_residency(
        mxfp8_bytes - 1, 1024, hysteresis_bytes=hysteresis, current=ResidencyState.MXFP8
    )
    assert held.state is ResidencyState.MXFP8
    assert "hysteresis" in held.reason
    # ...until the envelope falls more than hysteresis below MXFP8 residency
    slipped = select_residency(
        mxfp8_bytes - hysteresis - 1,
        1024,
        hysteresis_bytes=hysteresis,
        current=ResidencyState.MXFP8,
    )
    assert slipped.state is ResidencyState.MXFP4


def test_upward_move_taken_immediately_despite_hysteresis():
    sizes = body_size(1024)
    grown = select_residency(
        sizes.tier_bytes["mxfp8"],
        1024,
        hysteresis_bytes=1 << 20,
        current=ResidencyState.MXFP2,
    )
    assert grown.state is ResidencyState.MXFP8


# --------------------------------------------------------------------------- #
# manager: lineage, task admission, illegal transitions                        #
# --------------------------------------------------------------------------- #
def test_graph_revision_bumps_only_on_tier_change():
    manager = ResidencyManager(8 * GB)
    assert manager.graph_revision == 0
    assert manager.current.state is ResidencyState.MXFP8
    # same-tier envelope wiggle: no bump, no transition
    manager.update_envelope(7 * GB)
    assert manager.graph_revision == 0
    assert manager.transitions == ()
    # tier change: bump + logged
    manager.update_envelope(3 * GB)
    assert manager.graph_revision == 1
    assert len(manager.transitions) == 1
    assert manager.transitions[0].from_state is ResidencyState.MXFP8
    assert manager.transitions[0].to_state is ResidencyState.MXFP4
    assert manager.transitions[0].trigger == "memory_change"
    manager.update_envelope(int(1.8 * GB))
    manager.update_envelope(int(1.2 * GB))
    assert manager.graph_revision == 3
    assert manager.current.state is ResidencyState.CORE_ONLY


def test_novel_task_at_core_only_holds_without_tier_change():
    manager = ResidencyManager(int(1.2 * GB))
    assert manager.current.state is ResidencyState.CORE_ONLY
    revision_before = manager.graph_revision
    assert manager.admit_task(novel=True) == "HOLD"
    assert manager.graph_revision == revision_before
    assert manager.current.state is ResidencyState.CORE_ONLY
    # the HOLD is still logged as an event
    assert manager.transitions[-1].trigger == "novel_task"
    assert manager.transitions[-1].from_state is ResidencyState.CORE_ONLY
    assert manager.transitions[-1].to_state is ResidencyState.CORE_ONLY


def test_routine_task_at_core_only_executes_degraded():
    manager = ResidencyManager(int(1.2 * GB))
    assert manager.admit_task(novel=False) == "EXECUTE"
    assert manager.autonomy is AutonomyStatus.DEGRADED


def test_any_mx_tier_executes_nominal():
    for envelope in (8 * GB, 3 * GB, int(1.8 * GB)):
        manager = ResidencyManager(envelope)
        assert manager.current.state.is_mx
        assert manager.admit_task(novel=True) == "EXECUTE"
        assert manager.autonomy is AutonomyStatus.NOMINAL


def test_evict_and_restore_lineage():
    manager = ResidencyManager(8 * GB)
    manager.evict_reasoner()
    assert manager.current.state is ResidencyState.CORE_ONLY
    assert manager.autonomy is AutonomyStatus.DEGRADED
    assert manager.transitions[-1].trigger == "memory_change"
    with pytest.raises(ValueError):
        manager.evict_reasoner()  # already evicted — illegal
    with pytest.raises(ValueError):
        manager.restore_reasoner(1024)  # envelope cannot hold even MXFP2
    manager.restore_reasoner(3 * GB)
    assert manager.current.state is ResidencyState.MXFP4
    assert manager.transitions[-1].trigger == "restore"
    with pytest.raises(ValueError):
        manager.restore_reasoner(8 * GB)  # already resident — illegal


def test_evicted_reasoner_reports_zero_available_bytes_not_stale_pre_eviction_value():
    """A found bug, not a hypothetical: evict_reasoner() computes its
    decision by pretending the envelope is 0 (select_residency(0, ...)),
    but used to leave self._available at whatever it was before eviction
    -- e.g. still 8 GB. Since available_bytes flows straight into
    to_receipt()'s JSON alongside the CORE_ONLY/DEGRADED decision, a
    receipt taken right after a forced eviction would show a
    self-contradictory record: "evicted, degraded" next to "8 GB
    available" -- exactly what "residency accounting is deliberately
    honest" (this module's own stated design goal) says shouldn't
    happen. Fixed to zero available_bytes to match the decision it
    actually computed."""
    manager = ResidencyManager(8 * GB)
    manager.evict_reasoner()

    assert manager.available_bytes == 0
    receipt = manager.to_receipt()
    assert receipt["available_bytes"] == 0
    assert receipt["decision"]["state"] == "CORE_ONLY"


def test_manager_hysteresis_damps_updates():
    mxfp8_bytes = body_size(1024).tier_bytes["mxfp8"]
    manager = ResidencyManager(mxfp8_bytes, body_params=1024, hysteresis_bytes=64)
    manager.update_envelope(mxfp8_bytes - 1)
    assert manager.current.state is ResidencyState.MXFP8
    assert manager.graph_revision == 0
    manager.update_envelope(mxfp8_bytes - 65)
    assert manager.current.state is ResidencyState.MXFP4
    assert manager.graph_revision == 1


# --------------------------------------------------------------------------- #
# real-weight pack path (skips until the concurrent codec agent lands mxfp8)   #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif("mxfp8" not in FORMATS, reason="codec FORMATS lacks mxfp8 (concurrent work)")
def test_pack_body_roundtrip_and_footprint():
    from integrations.qualcomm.lowbit import dequantize_tensor

    rng = np.random.default_rng(7)
    weights = {
        "body.attn.q": rng.standard_normal((64, 32)).astype(np.float32),
        "body.mlp.up": rng.standard_normal((48,)).astype(np.float32),
    }
    packed = pack_body(weights, "mxfp8")
    assert set(packed) == set(weights)
    for name, tensor in packed.items():
        assert tensor.fmt == "mxfp8"
        restored = dequantize_tensor(tensor)
        assert restored.shape == weights[name].shape
        flat_w = weights[name].reshape(-1)
        cos = float(
            np.dot(restored.reshape(-1), flat_w)
            / (np.linalg.norm(restored) * np.linalg.norm(flat_w))
        )
        assert cos > 0.99
    footprint = resident_footprint(packed)
    assert footprint == sum(t.bytes_total for t in packed.values())
    assert 0 < footprint < sum(w.nbytes for w in weights.values())


@pytest.mark.skipif("mxfp8" not in FORMATS, reason="codec FORMATS lacks mxfp8 (concurrent work)")
def test_pack_body_rejects_unknown_format():
    with pytest.raises(ValueError):
        pack_body({"w": np.zeros((4, 4), dtype=np.float32)}, "fp32-master")


# --------------------------------------------------------------------------- #
# receipts                                                                     #
# --------------------------------------------------------------------------- #
def test_receipt_dict_json_round_trips():
    manager = ResidencyManager(8 * GB)
    manager.update_envelope(3 * GB)
    manager.update_envelope(int(1.2 * GB))
    manager.admit_task(novel=True)
    receipt = manager.to_receipt()
    encoded = json.dumps(receipt, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["graph_revision"] == 2
    assert decoded["decision"]["state"] == "CORE_ONLY"
    assert decoded["decision"]["autonomy"] == "DEGRADED"
    assert decoded["decision"]["format"] is None
    assert len(decoded["transitions"]) == 3  # two tier changes + the HOLD event
    assert decoded["transitions"][1]["trigger"] == "memory_change"
    assert decoded["transitions"][2]["trigger"] == "novel_task"


def test_write_receipt_into_validated_bundle(tmp_path):
    from omni_q.evidence_bundle import validate_evidence_bundle

    manager = ResidencyManager(8 * GB)
    manager.update_envelope(int(1.2 * GB))
    manager.admit_task(novel=True)
    run_dir = manager.write_receipt(tmp_path, run_id="residency-test-run")
    ok, errors = validate_evidence_bundle(run_dir)
    assert ok, errors
    decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    assert decision["graph_revision"] == 1
    assert decision["decision"]["state"] == "CORE_ONLY"
    lines = (run_dir / "transitions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # tier change + the HOLD event
