"""The stdlib judge surface serves explicit, truthfully labelled sessions."""

from __future__ import annotations

import json
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from omni_q import server as server_module
from omni_q.events import EventBus
from omni_q.server import Handler


def _request(url: str, *, payload: dict | None = None):
    request = Request(url, method="POST" if payload is not None else "GET")
    if payload is not None:
        request.add_header("Content-Type", "application/json")
        request.data = json.dumps(payload).encode()
    with urlopen(request, timeout=2) as response:
        return response.status, response.read(), dict(response.headers)


def test_server_serves_the_mock_labelled_ui_and_scoped_session_api():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, body, headers = _request(base + "/")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert b"MOCK / NO HARDWARE" in body

        status, body, _ = _request(base + "/profiles")
        assert status == 200
        profiles = json.loads(body)["profiles"]
        assert any(profile["profile"] == "formal_dinner_v3" for profile in profiles)

        status, body, _ = _request(base + "/sessions", payload={
            "goal": "set the table",
            "profile": "formal_dinner_v3",
            "constraints": [{
                "kind": "keep_local",
                "value": None,
                "justification": "judge demo stays local",
            }],
        })
        assert status == 202
        session = json.loads(body)
        assert session["session_id"]
        assert session["mode"] == "tableops-profile"
        assert session["profile"]["profile"] == "formal_dinner_v3"
        assert session["instruction"]["goal"] == "set the table"
        assert session["runtime"]["execution"] == "simulated"
        assert session["runtime"]["capabilities"]

        status, body, _ = _request(base + f"/sessions/{session['session_id']}")
        assert status == 200
        summary = json.loads(body)
        assert summary["mode"] == "tableops-profile"
        assert summary["receipt_ready"] is False

        deadline = time.time() + 2
        while time.time() < deadline:
            _, body, _ = _request(base + f"/sessions/{session['session_id']}")
            if json.loads(body)["receipt_ready"]:
                break
            time.sleep(0.02)
        status, body, _ = _request(base + f"/sessions/{session['session_id']}/receipt")
        receipt = json.loads(body)
        assert status == 200
        assert receipt["content_hash"]
        assert receipt["metrics"]["resolved"] is True
        assert receipt["metrics"]["profile_report"]["final_compliance"] == "24/24"

        try:
            _request(base + "/sessions", payload={
                "goal": "set the table",
                "constraints": [{"kind": "keep_local", "value": None}],
            })
        except HTTPError as exc:
            assert exc.code == 400
            assert b"justification" in exc.read()
        else:  # pragma: no cover - assertion guard
            raise AssertionError("server accepted an unjustified constraint")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_health_reports_the_configured_runtime_without_claiming_hardware(monkeypatch):
    monkeypatch.setenv("OMNIQ_UI_RUNTIME", "intel")
    monkeypatch.setenv("OMNIQ_OMNI_REASONER", "omni")

    payload = server_module._health_payload()

    assert payload["ok"] is True
    assert payload["runtime"] == "intel"
    assert payload["execution"] == "simulated"
    assert payload["hardware"] is False
    assert payload["reasoner"] == "omni"
    assert "MuJoCo" in payload["environment"]


def test_ui_factory_selects_mock_runtime_and_preserves_session_event_bus(monkeypatch):
    monkeypatch.setenv("OMNIQ_UI_RUNTIME", "mock")
    bus = EventBus()

    engine = server_module._ui_engine_factory(bus)

    assert engine.bus is bus
    assert getattr(engine.world, "mode", "mock") == "mock"


def test_profile_factory_rejects_unknown_profile_before_session_creation(monkeypatch):
    monkeypatch.setenv("OMNIQ_UI_RUNTIME", "mock")

    try:
        server_module._ui_engine_factory(EventBus(), profile="does-not-exist")
    except ValueError as exc:
        assert "unknown profile" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("unknown profile was accepted")
