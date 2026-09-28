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

## Product workflow: profile → mission → receipt

The page is profile-first. Select a venue/event profile, keep or edit the
operator objective, and start one session. The existing OMNI-Q graph,
authorization, transition, verification, and receipt seams remain underneath
the product surface:

```text
GET /profiles
      ↓
profile selection
      ↓
POST /sessions {profile, goal, constraints}
      ↓
profile.loaded → mission.diff → existing OMNI-Q event stream
      ↓
profile report inside the normal chained receipt
```

`GET /profiles` returns the checked-in YAML profiles. The current catalog
includes `formal_dinner_v3`, which has six enabled item classes for four seats
(24 required placements). The UI reports first-pass placements,
self-corrections, replans, retries, unresolved residuals, and final compliance.
It shows unresolved residuals as **HUMAN ASSISTANCE REQUIRED** rather than
turning an absent or stale observation into a success.

The profile runtime is currently a truthful symbolic-zone product fixture. The
mock observer can verify that `fork_2` is in `setting_2`; it cannot prove a
3 mm relationship without a metric pose. Those limits are retained in
`profile_report.metric_verification_gaps`. The Intel profile wrapper likewise
does not synthesize missing inventory: absent profile objects remain unresolved
in the receipt. Profile configuration and constraints are retained in the
receipt; runtime authority still comes from the existing governed mission
envelope and operator-justified constraint path.

The server uses the mock runtime by default. The browser is wired to the same
session/SSE surface for the opt-in Intel MuJoCo runtime:

```powershell
$env:OMNIQ_UI_RUNTIME = "intel"
$env:OMNIQ_OMNI_REASONER = "mock"   # or "omni" with matched checkpoint/receipt
$env:PYTHONPATH = "src"
python -m omni_q.server
```

`OMNIQ_UI_RUNTIME=intel` remains simulation-only and does not activate
hardware. With `OMNIQ_OMNI_REASONER=omni`, also set
`OMNIQ_OMNI_CHECKPOINT`, `OMNIQ_OMNI_RECEIPT`, and optionally
`OMNIQ_OMNI_DEVICE`; the identity-gated reasoner is selected inside the same
governed session, and the UI displays planner backend/fallback evidence from
the event stream and receipt.

## What the screen shows

- **Mission composer** — submits a natural-language tabletop objective and an
  selected venue/event profile plus an optional operator constraint with a
  required justification. The default presentation objective is `Set the table.`.
- **Product summary** — keeps the selected profile, current status, and
  satisfied/required placements visible above the diagnostic surfaces. Current
  residuals are listed as missing, stale, or misplaced rather than hidden in
  the graph.
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
  content hash. For profile runs it also displays the profile report and any
  metric-verification gaps. The full receipt is available from
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

The current judge server launches the mock runtime by default, so an objective-
only session is explicitly labelled **MOCK / NO HARDWARE**. A selected profile
session is labelled **TABLEOPS PROFILE / SIMULATED**; it is still synthetic and
does not control hardware. Selecting the Intel runtime changes
the session factory to the real MuJoCo dual-arm scene; the page then labels the
runtime as simulation and surfaces the Intel observer, scheduled placement, and
reasoner metadata instead of silently calling it mock. Both paths use the same
event contract, graph compilation, governed authorization, replan behavior,
causal stream recovery, and receipt surface.

The repository also contains an opt-in real perception seam. `OMNIQ_PERCEPTION=yolo`
uses the table fine-tune, and the Intel/OpenVINO integration is documented in
[`docs/oq-omni-vision-integration-2026-09-11.md`](oq-omni-vision-integration-2026-09-11.md).
Those paths are not silently presented as live by this server. A camera-backed
session adapter can reuse the page by emitting the same event shapes plus
camera/frame metadata and provider capability metadata; the front end already
renders those fields when they are present.

Voice and residency are supported as event vocabulary and contract seams, but
the current server keeps Speechmatics transport and residency control visibly
outside the active mock session.

## UI files and checks

```text
  profiles/formal_dinner_v3.yaml   checked-in product profile
  src/omni_q/profiles.py           profile loader, diff, mission wrapper, report
  ui/index.html                    product surface and accessible labels
  ui/styles.css                    responsive dark workcell-control-plane styling
  ui/app.js                        profile catalog, SSE validation, reconnect, rendering
```

Useful checks:

```bash
node --check ui/app.js
PYTHONPATH=src python -m pytest -q tests/test_profiles.py tests/test_server.py tests/test_sessions.py
```
