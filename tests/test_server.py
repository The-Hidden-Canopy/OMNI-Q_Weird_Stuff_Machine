"""The stdlib judge surface serves only explicit mock/session state."""

from __future__ import annotations

import json
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

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

        status, body, _ = _request(base + "/sessions", payload={
            "goal": "set the table",
            "constraints": [{
                "kind": "keep_local",
                "value": None,
                "justification": "judge demo stays local",
            }],
        })
        assert status == 202
        session = json.loads(body)
        assert session["session_id"]
        assert session["mode"] == "mock"
        assert session["instruction"]["goal"] == "set the table"
        assert session["runtime"]["execution"] == "simulated"
        assert session["runtime"]["capabilities"]

        status, body, _ = _request(base + f"/sessions/{session['session_id']}")
        assert status == 200
        summary = json.loads(body)
        assert summary["mode"] == "mock"
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
