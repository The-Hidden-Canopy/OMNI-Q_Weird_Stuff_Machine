"""Omni Q core: objective -> capability graph -> placement -> execution -> verification.

The package is dependency-free. Six contracts (:mod:`omni_q.contracts`) are the
seams every provider plugs into:

    Observe   Plan   Manipulate   Verify   Device   Receipt

:mod:`omni_q.fakes` and :mod:`omni_q.devices` supply mock providers so the whole
:class:`omni_q.engine.OmniQ` loop runs on a laptop with no sponsor hardware.
"""

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
    "FakeManipulator",
    "FakeObserver",
    "FakeRecorder",
    "FakeVerifier",
    "RulePlanner",
    "MockWorld",
    "build_mock_engine",
]

__version__ = "0.1.0"


def build_mock_engine(bus: "EventBus | None" = None, recorder: "object | None" = None) -> "OmniQ":
    """A fully wired OmniQ with mock providers — the OQ-001 end-to-end path.

    ``recorder`` overrides the in-memory FakeRecorder, e.g. with an
    :class:`omni_q.evidence.EvidenceLedger` for durable receipts (OQ-037).
    """
    world = MockWorld.sample()
    return OmniQ(
        world=world,
        observer=FakeObserver(),
        planner=RulePlanner(),
        manipulator=FakeManipulator(world),
        verifier=FakeVerifier(),
        device=default_devices(),
        recorder=recorder if recorder is not None else FakeRecorder(),
        bus=bus,
    )
