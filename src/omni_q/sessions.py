"""Session-scoped execution management for the SSE/UI boundary."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable
from uuid import uuid4

from .contracts import ReceiptRecord
from .engine import OmniQ
from .events import EventBus


class SessionError(ValueError):
    """A caller targeted a missing or terminal Omni Q session."""


@dataclass
class RunSession:
    session_id: str
    engine: OmniQ
    status: str = "queued"  # queued | running | finished | failed
    receipt: ReceiptRecord | None = None
    error: str | None = None
    done: threading.Event = field(default_factory=threading.Event)


EngineFactory = Callable[[EventBus], OmniQ]


class SessionManager:
    """Owns one engine per UI run; constraints target exactly one session."""

    def __init__(self, factory: EngineFactory) -> None:
        self._factory = factory
        self._sessions: dict[str, RunSession] = {}
        self._lock = threading.RLock()

    def create(self) -> RunSession:
        session_id = uuid4().hex[:12]
        bus = EventBus()
        session = RunSession(session_id=session_id, engine=self._factory(bus))
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> RunSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionError("unknown session")
        return session

    def start(self, session_id: str, goal: str) -> RunSession:
        session = self.get(session_id)
        with self._lock:
            if session.status != "queued":
                raise SessionError(f"session is already {session.status}")
            session.status = "running"
        threading.Thread(target=self._run, args=(session, goal), daemon=True).start()
        return session

    def create_and_start(self, goal: str) -> RunSession:
        session = self.create()
        return self.start(session.session_id, goal)

    def add_constraint(
        self,
        session_id: str,
        kind: str,
        value,
        *,
        justification: str,
    ) -> None:
        session = self.get(session_id)
        with self._lock:
            if session.status not in {"queued", "running"}:
                raise SessionError(f"session is {session.status}; constraints are closed")
            session.engine.add_constraint(
                kind,
                value,
                source="operator",
                justification=justification,
            )

    def events_after(self, session_id: str, cursor: int = 0):
        session = self.get(session_id)
        return [event for event in session.engine.bus.log if event.seq > cursor]

    def summary(self, session_id: str) -> dict[str, object]:
        session = self.get(session_id)
        receipt = session.receipt
        return {
            "session_id": session.session_id,
            "status": session.status,
            "mode": getattr(session.engine.world, "mode", "mock"),
            "run_id": receipt.run_id if receipt else None,
            "content_hash": receipt.content_hash if receipt else None,
            "error": session.error,
            "runtime": runtime_metadata(session.engine),
            "receipt_ready": receipt is not None,
        }

    @staticmethod
    def _run(session: RunSession, goal: str) -> None:
        try:
            session.receipt = session.engine.run(goal)
        except Exception as exc:  # surfaced truthfully to the UI/session owner
            session.status = "failed"
            session.error = str(exc)
        else:
            session.status = "finished"
        finally:
            session.done.set()


def _device_label(name: str) -> tuple[str, str | None]:
    """Turn an internal placement name into a judge-facing capability label."""
    lowered = name.lower()
    if "left" in lowered:
        return "Left SO-101", "left"
    if "right" in lowered:
        return "Right SO-101", "right"
    return name.replace(".", " / "), None


def runtime_metadata(engine: OmniQ) -> dict[str, object]:
    """Expose truthful, provider-neutral metadata to the judge UI.

    This is deliberately introspection-only. It does not initialize hardware or
    claim that an optional provider is live; the UI can therefore show the
    configured seams without inventing a route that the session did not use.
    """
    world_mode = str(getattr(engine.world, "mode", "mock"))
    simulated = world_mode == "mock"
    observer = getattr(engine, "observer", None)
    planner = getattr(engine, "planner", None)
    reasoner = getattr(planner, "reasoner", None)
    reasoner_name = getattr(reasoner, "backend", None)
    planner_name = reasoner_name or type(planner).__name__
    device_router = getattr(engine, "device", None)
    specs = list(device_router.devices()) if hasattr(device_router, "devices") else []

    capabilities: list[dict[str, object]] = [
        {
            "id": "perception",
            "kind": "perception",
            "name": "Perception",
            "detail": f"{type(observer).__name__} · structured state",
            "available": True,
            "local": True,
        },
        {
            "id": "reasoning",
            "kind": "reasoning",
            "name": "OMNI reasoning",
            "detail": f"{planner_name} · local process",
            "available": True,
            "local": True,
        },
    ]
    for spec in specs:
        if "arm" not in spec.kinds:
            continue
        name, arm = _device_label(spec.name)
        capabilities.append({
            "id": spec.name,
            "device": spec.name,
            "kind": "arm",
            "arm": arm,
            "name": name,
            "detail": f"{spec.name} · {'local' if spec.local else 'remote'} placement",
            "available": spec.online,
            "online": spec.online,
            "local": spec.local,
        })

    placement_devices = [
        {
            "name": spec.name,
            "kinds": list(spec.kinds),
            "available": spec.online,
            "online": spec.online,
            "local": spec.local,
        }
        for spec in specs
    ]
    return {
        "mode": world_mode,
        "environment": "mock session · no hardware" if simulated else world_mode,
        "execution": "simulated" if simulated else "configured",
        "observer": type(observer).__name__,
        "planner": type(planner).__name__,
        "reasoner": reasoner_name,
        "safety": "Mission envelope ready",
        "capabilities": capabilities,
        "placement_devices": placement_devices,
        "voice": {
            "status": "standby",
            "provider": "Speechmatics adapter",
            "wired": False,
        },
    }
