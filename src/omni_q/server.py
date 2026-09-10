"""Minimal SSE bridge from the EventBus to a browser (stdlib only).

Not the UI — that's OQ-005 (Bryan/Codex). This just exposes the engine's event
stream so a thin TS front end has something to consume:

    GET  /events        text/event-stream of engine events (JSON per line)
    POST /run           body {"goal": "..."} -> starts a mock run
    POST /constraint    body {"kind": "...", "value": ...} -> queues a constraint
    GET  /health        {"ok": true}

    python -m omni_q.server  [--port 8770]
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import build_mock_engine
from .events import Event, EventBus

_bus = EventBus()
_subscribers: list[queue.Queue] = []
_lock = threading.Lock()


def _fanout(event: Event) -> None:
    with _lock:
        targets = list(_subscribers)
    for q in targets:
        q.put(event.as_dict())


_bus.subscribe(_fanout)


def _run_async(goal: str) -> None:
    engine = build_mock_engine(_bus)
    threading.Thread(target=engine.run, args=(goal,), daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):  # quiet
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._json(200, {"ok": True})
        if self.path != "/events":
            return self._json(404, {"error": "not found"})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q: queue.Queue = queue.Queue()
        with _lock:
            _subscribers.append(q)
        try:
            while True:
                event = q.get()
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with _lock:
                if q in _subscribers:
                    _subscribers.remove(q)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad json"})

        if self.path == "/run":
            _run_async(body.get("goal", "inspect and correct the workspace"))
            return self._json(202, {"started": True})
        if self.path == "/constraint":
            _bus.publish("constraint.queued", kind=body.get("kind"),
                         value=body.get("value"))
            return self._json(202, {"queued": True})
        return self._json(404, {"error": "not found"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"omni-q SSE bridge on http://127.0.0.1:{args.port}  (GET /events, POST /run)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
