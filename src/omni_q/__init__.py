"""Omni Q core: objective -> capability graph -> placement -> execution -> verification.

The package is dependency-free. Six contracts (:mod:`omni_q.contracts`) are the
seams every provider plugs into:

    Observe   Plan   Manipulate   Verify   Device   Receipt

:mod:`omni_q.fakes` and :mod:`omni_q.devices` supply mock providers so the whole
:class:`omni_q.engine.OmniQ` loop runs on a laptop with no sponsor hardware.
"""

from __future__ import annotations

import os

from .contracts import (
    CONTRACTS,
    ActionAuthorization,
    AuthorizationVerdict,
    AutonomyMode,
    Constraint,
    ConstraintValidationError,
    DataStatus,
    Detection,
    Device,
    DeviceSpec,
    Manipulate,
    ManipResult,
    MissionEnvelope,
    Observation,
    Observe,
    Plan,
    PlanDecision,
    PlanGraph,
    Pose,
    Receipt,
    ReceiptRecord,
    Step,
    TransitionRejected,
    TransitionRequest,
    TransitionResult,
    Verify,
    VerifyResult,
    WorldState,
    World,
    content_hash_of,
    merge_mode,
    validate_operator_constraint,
    verify_chain,
)
from .devices import DeviceRouter, default_devices
from .engine import OmniQ
from .evidence import EvidenceLedger
from .events import Event, EventBus, EventValidationError, validate_event_chain
from .omni_planner import OmniPlanner
from .omni_reasoner import (
    MockReasoner,
    OmniReferenceReasoner,
    ReasonerResult,
    ReasonerUnavailable,
)
from .fakes import (
    FakeManipulator,
    FakeObserver,
    FakeRecorder,
    FakeVerifier,
    RulePlanner,
)
from .world import MockWorld

__all__ = [
    "CONTRACTS",
    "ActionAuthorization",
    "AuthorizationVerdict",
    "AutonomyMode",
    "Constraint",
    "ConstraintValidationError",
    "DataStatus",
    "Detection",
    "Device",
    "DeviceSpec",
    "Manipulate",
    "ManipResult",
    "MissionEnvelope",
    "Observation",
    "Observe",
    "Plan",
    "PlanDecision",
    "PlanGraph",
    "Pose",
    "Receipt",
    "ReceiptRecord",
    "Step",
    "TransitionRejected",
    "TransitionRequest",
    "TransitionResult",
    "Verify",
    "VerifyResult",
    "WorldState",
    "World",
    "content_hash_of",
    "merge_mode",
    "validate_operator_constraint",
    "verify_chain",
    "DeviceRouter",
    "default_devices",
    "OmniQ",
    "Event",
    "EventBus",
    "EventValidationError",
    "validate_event_chain",
    "EvidenceLedger",
    "OmniPlanner",
    "MockReasoner",
    "OmniReferenceReasoner",
    "ReasonerResult",
    "ReasonerUnavailable",
    "FakeManipulator",
    "FakeObserver",
    "FakeRecorder",
    "FakeVerifier",
    "RulePlanner",
    "MockWorld",
    "build_mock_engine",
]

__version__ = "0.1.0"


def build_mock_engine(bus: "EventBus | None" = None, recorder: "object | None" = None,
                      planner: "object | None" = None) -> "OmniQ":
    """A fully wired OmniQ with mock providers — the OQ-001 end-to-end path.

    ``recorder`` overrides the in-memory FakeRecorder, e.g. with an
    :class:`omni_q.evidence.EvidenceLedger` for durable receipts (OQ-037).
    ``planner`` overrides the default RulePlanner, e.g. with an
    :class:`omni_q.omni_planner.OmniPlanner` (see ``_planner_from_env``).
    """
    world = MockWorld.sample()
    return OmniQ(
        world=world,
        observer=FakeObserver(),
        planner=planner if planner is not None else _planner_from_env(),
        manipulator=FakeManipulator(world),
        verifier=FakeVerifier(),
        device=default_devices(),
        recorder=recorder if recorder is not None else FakeRecorder(),
        bus=bus,
    )


def _planner_from_env() -> "object":
    """Env-gated planner selection.

    ``OMNIQ_OMNI_REASONER`` = ``off``/unset (default RulePlanner) |
    ``mock`` (OmniPlanner over MockReasoner) | ``omni`` (OmniPlanner over the
    identity-gated IDA Omni reference reasoner; requires
    ``OMNIQ_OMNI_CHECKPOINT`` + ``OMNIQ_OMNI_RECEIPT``, optional
    ``OMNIQ_OMNI_DEVICE`` defaulting to cpu).
    """
    mode = os.environ.get("OMNIQ_OMNI_REASONER", "").strip().lower()
    if mode in {"", "0", "off", "rule"}:
        return RulePlanner()
    if mode == "mock":
        return OmniPlanner(MockReasoner())
    if mode == "omni":
        checkpoint = os.environ.get("OMNIQ_OMNI_CHECKPOINT")
        receipt = os.environ.get("OMNIQ_OMNI_RECEIPT")
        if not checkpoint or not receipt:
            raise ValueError(
                "OMNIQ_OMNI_REASONER=omni requires OMNIQ_OMNI_CHECKPOINT "
                "and OMNIQ_OMNI_RECEIPT")
        reasoner = OmniReferenceReasoner(
            checkpoint, receipt,
            device=os.environ.get("OMNIQ_OMNI_DEVICE", "cpu"))
        return OmniPlanner(reasoner)
    raise ValueError(f"unknown OMNIQ_OMNI_REASONER mode: {mode!r}")
