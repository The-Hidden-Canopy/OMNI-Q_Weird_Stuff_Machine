"""Session-scoped SSE bridge for the judge-facing Omni Q UI.

``POST /sessions`` starts one explicit session. The safe default is the mock
runtime; ``OMNIQ_UI_RUNTIME=intel`` selects the Intel MuJoCo runtime, and the
existing ``OMNIQ_OMNI_REASONER`` setting selects rule, mock-reasoner, or the
identity-gated OMNI reasoner within that runtime. Constraints are scoped to
the session and require an operator justification. ``GET
/sessions/<id>/events`` replays causal events after a numeric cursor so a
browser can reconnect without silently dropping or duplicating state.
``GET /sessions/<id>/receipt`` returns the full parent-chained receipt once the
run is terminal. ``GET /profiles`` exposes the checked-in venue/event profile
catalog used by the product UI; a selected profile is passed in the
``POST /sessions`` body as ``profile``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import build_mock_engine
from .nlu import parse
from .sessions import SessionError, SessionManager


def _configured_runtime() -> str:
    value = os.environ.get("OMNIQ_UI_RUNTIME", "mock").strip().lower()
    aliases = {
        "": "mock",
        "mock": "mock",
        "intel": "intel",
        "intel-sim": "intel",
        "intel_sim": "intel",
    }
    return aliases.get(value, value)


def _intel_engine_factory(bus):
    """Build the opt-in Intel session without importing MuJoCo by default."""
    from .intel_sim import IntelSimulationUnavailable, build_intel_sim_engine

    try:
        engine = build_intel_sim_engine(bus=bus)
    except IntelSimulationUnavailable as exc:
        raise ValueError(f"Intel MuJoCo runtime unavailable: {exc}") from exc

    from .scheduler import ScheduledPlanner

    fallback = ScheduledPlanner(engine.planner)
    reasoner_mode = os.environ.get("OMNIQ_OMNI_REASONER", "off").strip().lower()
    if reasoner_mode in {"", "0", "off", "rule"}:
        engine.planner = fallback
        return engine

    from .omni_planner import OmniPlanner

    if reasoner_mode == "mock":
        from .omni_reasoner import MockReasoner

        reasoner = MockReasoner()
    elif reasoner_mode == "omni":
        from .omni_reasoner import OmniReferenceReasoner

        checkpoint = os.environ.get("OMNIQ_OMNI_CHECKPOINT")
        receipt = os.environ.get("OMNIQ_OMNI_RECEIPT")
        if not checkpoint or not receipt:
            raise ValueError(
                "OMNIQ_OMNI_REASONER=omni requires OMNIQ_OMNI_CHECKPOINT "
                "and OMNIQ_OMNI_RECEIPT"
            )
        reasoner = OmniReferenceReasoner(
            checkpoint,
            receipt,
            device=os.environ.get("OMNIQ_OMNI_DEVICE", "cpu"),
        )
    else:
        raise ValueError(f"unknown OMNIQ_OMNI_REASONER mode: {reasoner_mode!r}")

    engine.planner = OmniPlanner(reasoner, fallback=fallback)
    return engine


def _ui_engine_factory(bus, *, profile=None):
    """Build the selected runtime, optionally wrapped in a TableOps profile.

    The profile wrapper adds desired-state diff and report evidence around the
    existing OMNI-Q engine.  It does not replace the planner, authorization,
    transition, verification, or receipt seams.
    """
    runtime = _configured_runtime()
    if profile is None:
        if runtime == "mock":
            return build_mock_engine(bus)
        if runtime == "intel":
            return _intel_engine_factory(bus)
        raise ValueError(
            f"unknown OMNIQ_UI_RUNTIME={runtime!r}; expected 'mock' or 'intel'"
        )

    from .profiles import attach_profile, build_profile_engine, load_profile

    reasoner_mode = os.environ.get("OMNIQ_OMNI_REASONER", "off").strip().lower()
    checkpoint = os.environ.get("OMNIQ_OMNI_CHECKPOINT")
    receipt = os.environ.get("OMNIQ_OMNI_RECEIPT")
    device = os.environ.get("OMNIQ_OMNI_DEVICE", "cpu")
    if runtime == "mock":
        return build_profile_engine(
            profile,
            bus,
            reasoner_mode=reasoner_mode,
            checkpoint=checkpoint,
            receipt=receipt,
            device=device,
        )
    if runtime == "intel":
        # Intel profile execution is intentionally an evidence wrapper around
        # the existing simulation world.  Any profile items absent from that
        # world remain unresolved in the receipt; we do not synthesize them.
        return attach_profile(_intel_engine_factory(bus), load_profile(profile))
    raise ValueError(
        f"unknown OMNIQ_UI_RUNTIME={runtime!r}; expected 'mock' or 'intel'"
    )


def _health_payload() -> dict[str, object]:
    runtime = _configured_runtime()
    if runtime == "mock":
        environment = "Mock session · no hardware"
    elif runtime == "intel":
        environment = "Intel MuJoCo simulation · no hardware"
    else:
        environment = f"Unrecognized runtime · {runtime}"
    return {
        "ok": runtime in {"mock", "intel"},
        "runtime": runtime,
        "environment": environment,
        "execution": "simulated",
        "hardware": False,
        "reasoner": os.environ.get("OMNIQ_OMNI_REASONER", "off").strip().lower() or "off",
    }


_sessions = SessionManager(_ui_engine_factory)
_UI_ROOT = Path(__file__).resolve().parents[2] / "ui"
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/ui/index.html": ("index.html", "text/html; charset=utf-8"),
    "/ui/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/ui/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


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
        if self._static(parsed.path):
            return
        if parsed.path == "/health":
            payload = _health_payload()
            return self._json(200 if payload["ok"] else 503, {
                **payload,
                "mode": payload["runtime"],
            })

        if parsed.path == "/profiles":
            try:
                from .profiles import list_profiles

                profiles = [profile.as_dict() for profile in list_profiles()]
            except ValueError as exc:
                return self._json(500, {"error": str(exc)})
            return self._json(200, {"profiles": profiles})

        segments = [part for part in parsed.path.split("/") if part]
        if len(segments) == 2 and segments[0] == "sessions":
            try:
                return self._json(200, _sessions.summary(segments[1]))
            except SessionError as exc:
                return self._json(404, {"error": str(exc)})
        if len(segments) == 3 and segments[0] == "sessions" and segments[2] == "receipt":
            try:
                session = _sessions.get(segments[1])
            except SessionError as exc:
                return self._json(404, {"error": str(exc)})
            if session.receipt is None:
                return self._json(409, {"error": "receipt is not ready"})
            return self._json(200, session.receipt.as_dict())
        if len(segments) == 3 and segments[0] == "sessions" and segments[2] == "events":
            query = parse_qs(parsed.query)
            try:
                cursor = int(query.get("cursor", ["0"])[0])
            except ValueError:
                return self._json(400, {"error": "cursor must be an integer"})
            last_event_id = self.headers.get("Last-Event-ID")
            if last_event_id and last_event_id.isdigit():
                cursor = max(cursor, int(last_event_id))
            return self._events(segments[1], cursor)
        return self._json(404, {"error": "not found"})

    def _static(self, path: str) -> bool:
        item = _STATIC_FILES.get(path)
        if item is None:
            return False
        filename, content_type = item
        body = (_UI_ROOT / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return True

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
        profile = body.get("profile")
        if profile == "":
            profile = None
        if profile is not None and not isinstance(profile, str):
            return self._json(400, {"error": "profile must be a string or null"})
        instruction = parse(goal)
        session = None
        try:
            session = _sessions.create(profile=profile)
            for kind, value in instruction.constraints:
                _sessions.add_constraint(
                    session.session_id,
                    kind,
                    value,
                    justification=f"parsed from operator instruction: {goal!r}",
                )
            for constraint in body.get("constraints", []):
                if not isinstance(constraint, dict):
                    raise ValueError("constraints must contain objects")
                _sessions.add_constraint(
                    session.session_id,
                    constraint.get("kind"),
                    constraint.get("value"),
                    justification=constraint.get("justification"),
                )
            _sessions.start(session.session_id, instruction.goal)
        except (SessionError, ValueError) as exc:
            payload = {"error": str(exc)}
            if session is not None:
                payload["session_id"] = session.session_id
            return self._json(400, payload)
        return self._json(202, {
            **_sessions.summary(session.session_id),
            "instruction": instruction.as_dict(),
        })

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
    config = _health_payload()
    print(
        f"omni-q SSE bridge on http://127.0.0.1:{args.port}  "
        f"({config['runtime']} runtime; {config['reasoner']} reasoner)"
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
