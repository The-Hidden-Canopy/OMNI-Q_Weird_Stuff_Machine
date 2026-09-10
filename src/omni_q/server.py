"""Session-scoped SSE bridge for the judge-facing Omni Q UI.

``POST /sessions`` starts one explicit mock session. Constraints are scoped to
that session and require an operator justification. ``GET /sessions/<id>/events``
replays causal events after a numeric cursor so a browser can reconnect without
silently dropping or duplicating state.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import build_mock_engine
from .sessions import SessionError, SessionManager


_sessions = SessionManager(build_mock_engine)


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

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            raise ValueError("bad json") from exc
        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object")
        return body

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            return self._json(200, {"ok": True, "mode": "mock"})

        segments = [part for part in parsed.path.split("/") if part]
        if len(segments) == 2 and segments[0] == "sessions":
            try:
                return self._json(200, _sessions.summary(segments[1]))
            except SessionError as exc:
                return self._json(404, {"error": str(exc)})
        if len(segments) == 3 and segments[0] == "sessions" and segments[2] == "events":
            query = parse_qs(parsed.query)
            try:
                cursor = int(query.get("cursor", ["0"])[0])
            except ValueError:
                return self._json(400, {"error": "cursor must be an integer"})
            return self._events(segments[1], cursor)
        return self._json(404, {"error": "not found"})

    def _events(self, session_id: str, cursor: int) -> None:
        try:
            _sessions.get(session_id)
        except SessionError as exc:
            return self._json(404, {"error": str(exc)})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            while True:
                for event in _sessions.events_after(session_id, cursor):
                    cursor = event.seq
                    self.wfile.write(f"id: {event.seq}\n".encode())
                    self.wfile.write(f"data: {json.dumps(event.as_dict())}\n\n".encode())
                    self.wfile.flush()
                summary = _sessions.summary(session_id)
                if summary["status"] in {"finished", "failed"}:
                    self.wfile.write(f"event: terminal\ndata: {json.dumps(summary)}\n\n".encode())
                    self.wfile.flush()
                    return
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self) -> None:
        try:
            body = self._body()
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})

        parsed = urlparse(self.path)
        if parsed.path in {"/sessions", "/run"}:
            return self._create_session(body)

        segments = [part for part in parsed.path.split("/") if part]
        if len(segments) == 3 and segments[0] == "sessions" and segments[2] == "constraints":
            return self._add_constraint(segments[1], body)
        if parsed.path == "/constraint":  # compatibility path, still session-scoped
            session_id = body.get("session_id")
            if not isinstance(session_id, str):
                return self._json(400, {"error": "session_id is required"})
            return self._add_constraint(session_id, body)
        return self._json(404, {"error": "not found"})

    def _create_session(self, body: dict) -> None:
        goal = body.get("goal", "set the table")
        if not isinstance(goal, str) or not goal.strip():
            return self._json(400, {"error": "goal must be a non-empty string"})
        session = _sessions.create()
        try:
            for constraint in body.get("constraints", []):
                if not isinstance(constraint, dict):
                    raise ValueError("constraints must contain objects")
                _sessions.add_constraint(
                    session.session_id,
                    constraint.get("kind"),
                    constraint.get("value"),
                    justification=constraint.get("justification"),
                )
            _sessions.start(session.session_id, goal)
        except (SessionError, ValueError) as exc:
            return self._json(400, {"error": str(exc), "session_id": session.session_id})
        return self._json(202, _sessions.summary(session.session_id))

    def _add_constraint(self, session_id: str, body: dict) -> None:
        try:
            _sessions.add_constraint(
                session_id,
                body.get("kind"),
                body.get("value"),
                justification=body.get("justification"),
            )
        except (SessionError, ValueError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(202, _sessions.summary(session_id))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"omni-q SSE bridge on http://127.0.0.1:{args.port}  (mock mode)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
