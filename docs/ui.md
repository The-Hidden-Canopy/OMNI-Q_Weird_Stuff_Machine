# Judge-facing UI

The OMNI-Q UI is the operator-facing proof surface for the governed execution
loop. It keeps the hackathon story visible in one screen:

```
objective → observation → graph → placement → action → verification → receipt
```

The page is intentionally thin. It does not invent state or call hardware
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

- **Mission composer** — submits a natural-language tabletop objective and an
  optional operator constraint with a required justification. The default
  presentation objective is `Set the table.`.
- **Workspace view** — renders the structured observation, object class,
  object ID, zone, target zone, confidence, data status, frame, state revision,
  and observer source. An empty detection list is shown as **NO OBJECT SIGNAL**;
  it is never presented as confirmed clearance.
- **Mission pulse** — tracks `Observe → Plan → Act → Verify` and reports
  completion against the current graph revision.
- **Execution graph** — shows compiled steps, explicit dependencies,
  current state, arm/device placement, and graph revision after a replan. The
  graph is not drawn as a fake linear chain when the contract supplies deps.
- **Planner decision evidence** — shows candidates considered, candidates
  feasible, selected operations, rejected proposals, constraints, autonomy
  mode, and whether the decision used a labeled fallback.
- **Capability nodes** — renders runtime metadata returned by the session:
  observer, reasoner, simulated SO-101 arms, and available placement devices.
  It does not claim live hardware when the session is synthetic.
- **Why OMNI-Q did that** — presents authorization, constraint, replan,
  verification, world-change, voice, residency, and failure events in causal
  order. The stream guard rejects duplicate, out-of-order, or parent-invalid
  events.
- **Run receipt** — displays resolution, revisions, actions, duration,
  rejected decisions, autonomy mode, run ID, parent hash, provenance, and the
  content hash. The full receipt is available from
  `GET /sessions/<session_id>/receipt` after the run is terminal.

## Live constraints

The composer can submit a constraint with the initial mission. While a session
is still running, **Apply live constraint** posts the same operator-scoped
constraint to `/sessions/<session_id>/constraints`. The engine queues it,
narrows the effective world, and emits a `graph.recompiled` event on the next
loop iteration.

Supported UI examples:

- `keep_local` — keep inference on-device and raise autonomy mode to
  `LOCAL_ONLY`.
- `forbid_object=<observed object ID>` — prevent manipulation of the selected
  object. The object field is populated from the latest observation and also
  accepts a typed ID for an initial constraint.
- `prefer_arm=left` — express a preference without overriding reachability or
  safety checks.

Every operator constraint needs a non-empty justification. Constraints are
scoped to the current session and close when the session reaches a terminal
state. Active constraints remain visible below the composer.

## Current runtime and real-eyes relationship

The current judge server launches `build_mock_engine`, so the page is explicitly
labelled **MOCK / NO HARDWARE**. The mock session proves the event contract,
graph compilation, governed authorization, replan behavior, causal stream
recovery, and receipt surface.

The repository also contains an opt-in real perception seam. `OMNIQ_PERCEPTION=yolo`
uses the table fine-tune, and the Intel/OpenVINO integration is documented in
[`docs/oq-omni-vision-integration-2026-09-11.md`](oq-omni-vision-integration-2026-09-11.md).
Those paths are not silently presented as live by this server. A future real
session adapter can reuse the page by emitting the same event shapes plus
camera/frame metadata and provider capability metadata.

Voice and residency are supported as event vocabulary and contract seams, but
the current server keeps Speechmatics transport and residency control visibly
outside the active mock session.

## UI files and checks

```text
ui/index.html   page structure and accessible labels
ui/styles.css   responsive dark workcell-control-plane styling
ui/app.js       SSE validation, reconnect, and rendering
```

Useful checks:

```bash
node --check ui/app.js
PYTHONPATH=src python -m pytest -q tests/test_server.py tests/test_sessions.py
```
