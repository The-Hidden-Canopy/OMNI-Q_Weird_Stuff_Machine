# Judge-facing UI

The Omni Q UI is the operator-facing proof surface for the governed execution
loop. It makes the core story visible in one screen:

```
objective → observation → graph → placement → action → verification → receipt
```

The UI is intentionally thin. It does not invent state or call hardware
directly; it renders the causal event stream emitted by the active session in
`src/omni_q/server.py`.

## Run it

From the repository root, start the session/SSE server:

```bash
PYTHONPATH=src python -m omni_q.server
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m omni_q.server
```

Open <http://127.0.0.1:8770>.

The UI has no separate build step. `ui/index.html`, `ui/styles.css`, and
`ui/app.js` are served directly by the Python server.

## What the screen shows

- **Mission composer** — submits a natural-language objective and an optional
  operator constraint with a required justification.
- **Workspace view** — renders compact structured observations, object zones,
  target zones, confidence boundaries, and whether the workspace still needs
  action.
- **Mission pulse** — tracks Observe, Plan, Act, and Verify as the session
  advances.
- **Execution graph** — shows the compiled steps, current state, arm/device
  placement, and graph revision after a replan.
- **Capability nodes** — makes the separation between function and placement
  visible across perception, reasoning, and the two simulated SO-101 arms.
- **Why Omni did that** — presents authorization, constraint, replan,
  verification, and failure events in causal order.
- **Run receipt** — displays final metrics and the receipt content hash.

## Live constraints

The composer can submit a constraint with the initial mission. While a session
is still running, **Apply live constraint** posts the same operator-scoped
constraint to `/sessions/<session_id>/constraints`. The engine queues it,
narrows the effective world, and emits a `graph.recompiled` event on the next
loop iteration.

Supported UI examples:

- `keep_local` — keep inference on-device and raise autonomy mode to
  `LOCAL_ONLY`.
- `forbid_object=connector_2` — prevent manipulation of the named object.
- `prefer_arm=left` — express a preference without overriding reachability or
  safety checks.

Every operator constraint needs a non-empty justification. Constraints are
scoped to the current session and close when the session reaches a terminal
state.

## Scope boundary

The current server launches `build_mock_engine`, so the UI is explicitly
labelled **MOCK MODE — NOT HARDWARE**. It demonstrates the event contract,
graph compilation, governed authorization, replan behavior, and receipt
surface. It is not evidence of a live camera, physical SO-101 hardware,
Qualcomm inference, or Speechmatics streaming.

The frontend can be reused when those providers are wired in because it reads
the shared event shapes rather than provider-specific APIs.

## UI files and checks

```text
ui/index.html   page structure and accessible labels
ui/styles.css   responsive dark workcell-control-plane styling
ui/app.js       SSE event validation and rendering
```

Useful checks:

```bash
node --check ui/app.js
PYTHONPATH=src python -m pytest -q tests/test_server.py tests/test_sessions.py
```
