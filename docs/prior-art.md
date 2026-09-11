# Prior art — patterns lifted from sibling repos

Four repos in the The-Hidden-Canopy org already solved pieces of what Omni Q
needs. This is what to borrow and where it lands on the [backlog](../BACKLOG.md).

| Repo | What it is | Omni Q borrows |
|------|------------|----------------|
| **SOCOM_REACT** | Adversarial rolling-horizon replanner for 200-agent swarms under degraded comms | The whole control-loop shape, executive/planner split, decision-reason object, signed authority envelope, monotone degradation modes |
| **Open-World-Model-Harness** | Engine-neutral harness: authoritative world ↔ JSON decision contract ↔ policy/replay/eval | World boundary (never mutate the world directly), append-only causal event log, knowledge-as-bounded-resource with `LIVE/STALE/FALLBACK` status |
| **FALCON-DARPA** | Structured-ML → selective foundation-model → governed-LLM fusion, 1000 reproducible runs | Self-describing per-run evidence package, SHA-256 parent-chained `manifest.jsonl`, deterministic `run_id = sha256(config)[:12]`, provenance block |
| **VIGIL** | RegOS-derived governance runtime: authority/evidence/policy/audit before consequential actions | Decision-receipt canonicalization + `verify_chain`, fail-closed pipeline (finalize receipt *before* the action returns), restrictive-only monotonic decisions, "intelligence interprets, governance authorizes" |

## SOCOM_REACT — the architecture to copy

REACT is "a mission-execution intelligence layer that converts Commander's
Intent into bounded autonomous action" and "continuously manages objectives,
information, time, observability, uncertainty… as conditions change faster than
centralized human replanning can keep up." Swap "Commander's Intent" for
"natural-language objective" and that is Omni Q.

### 1. Control loop — commit ONE action, then re-observe

```
OBSERVE → ESTIMATE → FORECAST → GENERATE COAs → SIMULATE
→ CONSTRAINT FILTER → ROBUST SCORE → COMMIT ONE ACTION → MEASURE → REPLAN
```

> "COAs are time-indexed configuration sequences, not waypoint lists; only the
> first configuration is committed before replanning from fresh observations."

Omni Q today plans a whole graph, runs it, and only replans on failure. The
demo's "I moved the object while it was working and the robot finished anyway"
moment is exactly REACT's commit-one-then-reobserve. → **OQ-018**, engine loop.

### 2. Executive above the planner (`executive.py` vs `planner.py`)

- **Mission Executive** decides *which problem the planner is solving*:
  objective selection, phase management, branches/sequels, time-decaying
  opportunity windows, authority delegation.
- **Planner** just picks the next action for the problem it was handed.

Omni Q's `engine.py` currently does both. Split them. → **OQ-012** (scheduler is
the executive's job), **OQ-015** (style constraints are an executive concern,
not a planner hack).

### 3. `PlanDecision` — every plan carries a reason object

```python
@dataclass
class PlanDecision:
    selected_coa_id: str
    next_action: ...
    candidates_considered: int
    candidates_feasible: int
    predicted_utility: float
    worst_case_utility: float
    governing_constraints: tuple[str, ...]
    rejected_reasons: dict[str, tuple[str, ...]]   # candidate -> why filtered
    mode: AutonomyMode
    state_hash: str
    model_version: str
    latency_ms: float
```

> "auditable without trying to make the optimizer itself 'explainable AI'."

Omni Q steps have a `rationale` string but no per-decision audit object, and
forbidden objects are *silently* skipped. Record them as `rejected_reasons`.
→ **OQ-018**, **OQ-047** ("why Omni did that" display reads this straight off).

### 4. `SignedMissionEnvelope` — fixed operator authority

```python
@dataclass(frozen=True)
class SignedMissionEnvelope:
    mission_id: str
    version: int
    permitted_objectives: tuple[str, ...]
    max_risk: float
    ...
    def digest(self) -> str: ...
```

> "Local planners inherit it unchanged — losing headquarters never broadens an
> element's authority."

Omni Q's constraints (`forbid_object`, `keep_local`) are ad-hoc kwargs. Put the
authority half in a signed, hashable envelope every candidate is gated against;
keep transient constraints separate. → **OQ-025**, **OQ-031** (device routing
inherits the envelope), **OQ-034**.

### 5. `AutonomyMode` — monotone degradation (`envelope.py`)

```python
MODE_PRECEDENCE = {NOMINAL:0, DEGRADED_NAV:1, DEGRADED_COMMS:2,
                   LOCAL_ONLY:3, HOLD:4, SAFE_RETURN:5, PROTECTIVE_STOP:6}

def merge_mode(current, incoming):   # never downgrades without explicit reset
    return incoming if PRECEDENCE[incoming] >= PRECEDENCE[current] else current
```

Plus `OperationalEnvelope`: **lower confidence ⇒ tighter risk ceiling,
automatically**. Omni Q has no graceful-degradation concept. Left arm lost ⇒
`DEGRADED`; "keep everything local" ⇒ `LOCAL_ONLY`; a low-confidence /stale
detection ⇒ contract what the planner may commit. → **OQ-013**, **OQ-018**.

### 6. Event-triggered replanning + tempo

- Replan only when inputs changed beyond `REPLAN_CHANGE_THRESHOLD`, not every
  tick.
- `opportunity_value` decays with time — "a marginally safer answer found too
  late scores lower." Tempo is an optimization dimension, not an afterthought.

### 7. Evidence (`evidence.py`)

`DecisionEvidence` (JSONL, one line per decision: `state_digest`, `planner`,
`candidate_count`, `feasible_count`, `selected`, `utility`, `worst_case`,
`constraints_active`, `latency_ms`, `mode`) + `RunManifest`
(`scenario_sha256`, `config_sha256`, `model_version`, `platform`, `metrics`).

## Open-World-Model-Harness — world boundary + honest knowledge

### 8. Never mutate the world directly

> "The reference simulator owns a mutable internal state, but callers only
> receive copies of state and submit typed transition requests. The same
> contracts can be serialized over HTTP when Unity or Unreal owns the
> authoritative world."

`FakeManipulator` currently reaches into `MockWorld` and mutates it. Route every
effect through `world.apply_transition(request) -> revisioned Observation`. That
seam *is* the MuJoCo boundary (**OQ-006**) and the Arduino boundary
(**OQ-030**) — build it now against the mock so the real ones drop in.

### 9. Append-only causal event log (`core.py::EventLog`)

Domain events carry `parent_event_ids`, `root_cause_id`, `state_revision`;
`compact_domain_events()` folds old raw events into digest-backed episodes.
Omni Q's `EventBus` has no lineage or revision. Add `seq`, `parent_id`,
`revision` to `Event`. → **OQ-037**.

### 10. Knowledge is a bounded resource

`KnowledgeKind = OBSERVED | REPORTED | INFERRED | HISTORICAL`; every claim keeps
confidence, source, and verification tick. "Models… cannot promote an inference
into an authoritative fact." "Hidden evaluator truth status is never presented
as fact." `DataStatus = LIVE | STALE | FALLBACK`.

For Omni Q: a `Detection` needs a status + last-verified frame. The planner must
not treat a stale detection as ground truth — it must re-`Observe` or lower its
commitment. → **OQ-009**, **OQ-018**. Also: `get_state()` (agent, partial) is
separate from `get_evaluation_snapshot()` (evaluator, hidden truth) — keep that
split for the OQ-019 evaluator and OQ-021 breakage tests.

## FALCON-DARPA — reproducible evidence packages

### 11. Self-describing per-run package + tamper-evident chain (`artifacts.py`)

```
artifacts/runs/<run_id>/config.json  predictions.csv  metrics.json  provenance.json
artifacts/manifest.jsonl   # one line/run: sha256 of each file + parent_hash → prev entry
```

- `run_id_for(config) = sha256(canonical(config))[:12]` — deterministic.
- `provenance`: `generated_at`, `was_generated_by`, `platform`, `python`,
  `package_versions` (explicit tracked list), `git_commit`.
- `_last_hash()` returns `"GENESIS"` for the first entry.

Omni Q's `FakeRecorder` hashes three blobs and stops. Adopt the chain, the
deterministic `run_id`, and the provenance block wholesale. → **OQ-037**.

### 12. Preserve negative results

> "A hard missingness gate over-escalated and was rejected as a negative
> result." Failed SFT experiments are kept, not hidden.

Receipts should record rejected candidates and failed attempts, not just the
winning path. Judges (and **OQ-021**, **OQ-048**) want to see the recovery, not
a laundered trace.

### 13. Structured evidence upstream of any LLM

> "The LLM is downstream of structured evidence. It is not treated as the source
> of authoritative statistical truth."

Matches the strategy doc: Qualcomm emits compact structured scene state, Omni
reasons over *that*, never raw video. Bakes into the `Observe` contract.

## VIGIL — governed decisions with receipts

### 14. Decision-receipt canonicalization + chain (`audit/receipt.py`)

```python
def canonical_bytes(r):   # sorted keys, no whitespace, content_hash EXCLUDED
    ...
def content_hash(r) -> str:            # "sha256:" + sha256(canonical_bytes)
def finalize(r):                       # returns copy with content_hash set
def verify_chain(receipts):            # each hash recomputes AND
                                       # parent_receipt == prior.audit_id
```

### 15. Fail-closed pipeline (`runtime/pipeline.py`)

- Any stage error / incomplete evaluation ⇒ `ESCALATE` (could-not-decide).
- The receipt **always carries at least one reason**.
- The receipt is **finalized and appended to the ledger *before* any permitting
  decision returns**; if the append fails, the decision stays non-permitting.
- A running decision only ever moves toward the restrictive end (monotone —
  same shape as REACT's `merge_mode`).

Omni Q's engine returns a receipt at the *end*. Invert it: finalize + chain the
receipt for each committed action before that action is allowed to take effect.
→ **OQ-018**, **OQ-037**, **OQ-038**.

### 16. Decision vocabulary

`ALLOW · LIMIT (reduced scope) · REDACT · REQUIRE_ADDITIONAL_EVIDENCE ·
REQUIRE_APPROVAL · ESCALATE · DENY`. A step-authorization gate for Omni Q
collapses to `ALLOW / LIMIT / REQUIRE_APPROVAL / DENY`.

> "Intelligence interprets. Governance authorizes." — the planner proposes;
> a separate envelope/governance check authorizes. Connectivity ≠ authority.

### 17. Pinned versions in the receipt

`PinnedVersions` records which policy / schema / model versions were in force
for the decision, so a later audit reconstructs the exact ruleset.

## Open-World-Model-Harness (second pass) — evidence integrity + evaluator secrecy

### 18. Re-verifiable evidence bundle (`data_management.py:32-499`)

`RunManifest` + `RunArtifactWriter.write_run` + `validate_run_artifact`
(`data_management.py:32-245`, `:249-405`): one immutable run directory with
`manifest.json`, caller artifacts, and `checksums.json` written **last**;
`_serialize` rejects non-finite floats (`:448-464`), `_sha256` streams
(`:466-471`), `_safe_name` digest-suffixes the directory (`:474-489`), and
`_is_safe_artifact_name` guards traversal (`:492-499`). Validation is
fail-closed: checksum set equality vs the manifest, missing *and* unlisted
files, per-file digest recompute, and count cross-checks. Vendored
(generalized from engine-specific turns/episodes to caller-supplied
metadata + counts) as `omni_q/evidence_bundle.py`
(`EvidenceBundleWriter`, `validate_evidence_bundle`).

### 19. Evaluator-only secrets + deterministic seeding (`core.py:659-720`, `contracts.py:3032-3046`)

- Per-entity hidden scalars: SHA-256 over `session:player:skill:node`,
  first digest byte mapped into `[0.65, 1.35)` (`core.py:659-679`).
  Vendored as `stable_unit_float(domain, *parts)` in
  `omni_q/eval_secrets.py`.
- `get_evaluation_snapshot()` (`core.py:681-720`) is evaluator-only ground
  truth, persisted to its own `final_evaluator_snapshot.json`
  (`data_management.py:220-223`) and never included in observations.
  Vendored as `SealedSnapshot` / `PublicSnapshot` — separate dataclasses,
  so the secret cannot be serialized through the public path.
- `SensoryCue.reliable` (`contracts.py:3032-3046`, emitted at
  `core.py:8070-8094`): `False` marks a hallucination-prone channel;
  consumers must weight accordingly. Vendored as `SensoryCue` with
  `reliable: bool = True`.

## What changed in this pass

Additive, low-risk lifts (see the commit):

- `contracts.py`: `DataStatus`, `AutonomyMode` + `merge_mode`, `PlanDecision`,
  `MissionEnvelope` (frozen, `.digest()`), `ReceiptRecord` extended with
  `parent_hash` / `content_hash` / `provenance` / `rejected` + `verify_chain`.
- `provenance.py` (new): `git_commit`, `package_versions`, `run_id_for`,
  `build_provenance` — lifted from FALCON `artifacts.py`.
- `engine.py`: emits a `PlanDecision` per (re)compile; tracks `AutonomyMode`
  monotonically; finalizes a chained receipt.
- `fakes.py`: `RulePlanner` records forbidden objects as `rejected`, not silent
  skips.
- `evidence_bundle.py` (new, from OWMH `data_management.py`): re-verifiable
  evidence bundles — fail-closed validation, checksums written last.
- `eval_secrets.py` (new, from OWMH `core.py`): `stable_unit_float`,
  `SealedSnapshot`/`PublicSnapshot`, `SensoryCue`.

Deferred (documented, not yet built): executive/planner split (#2), world
transition-request seam (#8), event causal lineage (#9), stale-knowledge
handling in the planner (#10).
