"""A tiny synchronous pub/sub bus.

Every interesting thing the engine does is published as an :class:`Event`. The
CLI demo subscribes and prints; the SSE server subscribes and forwards to the
browser. Nothing here is Omni-Q-specific.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Event:
    kind: str
    """Dotted name, e.g. ``node.started``, ``graph.compiled``, ``constraint.added``."""

    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ts": self.ts, "data": self.data}


Subscriber = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self.log: list[Event] = []

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        self._subscribers.append(fn)

        def unsubscribe() -> None:
            if fn in self._subscribers:
                self._subscribers.remove(fn)

        return unsubscribe

    def publish(self, kind: str, /, **data: Any) -> Event:
        # ``kind`` is positional-only so an event may carry a ``kind`` field.
        event = Event(kind=kind, data=data)
        self.log.append(event)
        for fn in list(self._subscribers):
            fn(event)
        return event
