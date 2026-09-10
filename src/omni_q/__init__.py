"""Omni Q core: objective -> capability graph -> placement -> execution -> verification.

The package is dependency-free. Six contracts (:mod:`omni_q.contracts`) are the
seams every provider plugs into:

    Observe   Plan   Manipulate   Verify   Device   Receipt

:mod:`omni_q.fakes` and :mod:`omni_q.devices` supply mock providers so the whole
:class:`omni_q.engine.OmniQ` loop runs on a laptop with no sponsor hardware.
"""

from .contracts import (
    CONTRACTS,
    Constraint,
    Detection,
    Device,
    DeviceSpec,
    Manipulate,
    ManipResult,
    Observation,
    Observe,
    Plan,
    PlanGraph,
    Receipt,
    ReceiptRecord,
    Step,
    Verify,
    VerifyResult,
    WorldState,
)
from .devices import DeviceRouter, default_devices
from .engine import OmniQ
from .events import Event, EventBus
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
    "Constraint",
    "Detection",
    "Device",
    "DeviceSpec",
    "Manipulate",
    "ManipResult",
    "Observation",
    "Observe",
    "Plan",
    "PlanGraph",
    "Receipt",
    "ReceiptRecord",
    "Step",
    "Verify",
    "VerifyResult",
    "WorldState",
    "DeviceRouter",
    "default_devices",
    "OmniQ",
    "Event",
    "EventBus",
    "FakeManipulator",
    "FakeObserver",
    "FakeRecorder",
    "FakeVerifier",
    "RulePlanner",
    "MockWorld",
    "build_mock_engine",
]

__version__ = "0.1.0"


def build_mock_engine(bus: "EventBus | None" = None) -> "OmniQ":
    """A fully wired OmniQ with mock providers — the OQ-001 end-to-end path."""
    world = MockWorld.sample()
    return OmniQ(
        world=world,
        observer=FakeObserver(),
        planner=RulePlanner(),
        manipulator=FakeManipulator(world),
        verifier=FakeVerifier(),
        device=default_devices(),
        recorder=FakeRecorder(),
        bus=bus,
    )
