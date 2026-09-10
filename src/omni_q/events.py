"""A small causal pub/sub bus.

Every interesting thing the engine does is published as an :class:`Event`. The
CLI demo subscribes and prints; the SSE server subscribes and forwards to the
browser. Nothing here is Omni-Q-specific.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class Event:
    kind: str
    """Dotted name, e.g. ``node.started``, ``graph.compiled``, ``constraint.added``."""

    data: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    event_id: str = ""
    parent_id: str | None = None
    run_id: str | None = None
    state_revision: int | None = None
    graph_revision: int | None = None
    source: str = "engine"
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "seq": self.seq,
            "event_id": self.event_id,
            "parent_id": self.parent_id,
            "run_id": self.run_id,
            "state_revision": self.state_revision,
            "graph_revision": self.graph_revision,
            "source": self.source,
            "ts": self.ts,
            "data": self.data,
        }


Subscriber = Callable[[Event], None]


class EventValidationError(ValueError):
    """A consumer received an invalid or causally impossible event stream."""


def validate_event_chain(events: Sequence[Event]) -> None:
    """Reject duplicate/out-of-order sequences and broken causal links."""
    last_seq = 0
    last_by_run: dict[str | None, str] = {}
    for event in events:
        if event.seq <= last_seq:
            raise EventValidationError(f"event {event.event_id}: non-monotonic sequence")
        last_seq = event.seq
        expected_parent = last_by_run.get(event.run_id)
        if event.parent_id != expected_parent:
            raise EventValidationError(f"event {event.event_id}: invalid parent link")
        if not event.event_id:
            raise EventValidationError("event is missing event_id")
        last_by_run[event.run_id] = event.event_id


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self.log: list[Event] = []
        self._seq = 0
        self._run_id: str | None = None
        self._last_by_run: dict[str | None, str] = {}

    def begin_run(self, run_id: str) -> None:
        self._run_id = run_id

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        self._subscribers.append(fn)

        def unsubscribe() -> None:
            if fn in self._subscribers:
                self._subscribers.remove(fn)

        return unsubscribe

    def publish(
        self,
        kind: str,
        /,
        *,
        run_id: str | None = None,
        parent_id: str | None = None,
        state_revision: int | None = None,
        graph_revision: int | None = None,
        source: str = "engine",
        **data: Any,
    ) -> Event:
        # ``kind`` is positional-only so an event may carry a ``kind`` field.
        effective_run = self._run_id if run_id is None else run_id
        self._seq += 1
        event_id = f"{effective_run or 'global'}:{self._seq}"
        event = Event(
            kind=kind,
            data=data,
            seq=self._seq,
            event_id=event_id,
            parent_id=(self._last_by_run.get(effective_run) if parent_id is None else parent_id),
            run_id=effective_run,
            state_revision=state_revision,
            graph_revision=graph_revision,
            source=source,
        )
        self.log.append(event)
        self._last_by_run[effective_run] = event.event_id
        for fn in list(self._subscribers):
            fn(event)
        return event
