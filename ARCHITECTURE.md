# Architecture

Rationale: [`docs/strategy-notes.md`](docs/strategy-notes.md). This describes
what's built; the mock stack runs today (`python -m omni_q.demo`), the Intel
sim path via `python -m omni_q.demo_intel_sim`.

## The layers — each with a reason to exist

```
Perception    a swarm of cheap specialist detectors      "what appears to be happening?"
   │          (objects, human/hand, hazard, affordance,   → emits Claims (hypotheses + provenance)
   ▼           robot-state) — MXFP2 on the Intel box
Ontology      the fusion authority                        "what do we currently believe exists?"
   │          reconciles Claims into Entities + Relations;
   ▼           IS-A merge, preserved disagreement, deltas
OMNI-Q        objective + world → capability graph         "given that, what should happen next?"
   │          plan · schedule · rewrite · react            (FP16/INT8; wakes on meaningful deltas)
   ▼
Execution     device routing + arm drivers                "what am I permitted and able to do?"
              MissionEnvelope gates every step
```

Full ontology writeup: [`docs/ontology.md`](docs/ontology.md).

## The seven contracts (`src/omni_q/contracts.py`)

`runtime_checkable` Protocols; nothing above them calls hardware directly.

| contract | in | out |
|----------|----|-----|
| **World** | — | authoritative `WorldState` (the only thing that mutates reality, via `TransitionRequest`) |
| **Observe** | `WorldState` | `Observation` (compact scene state, never raw video) |
| **Plan** | goal + `WorldState` | `PlanGraph` (+ recompile) |
| **Manipulate** | `Step` | `ManipResult` |
| **Verify** | expected vs `Observation` | `VerifyResult` |
| **Device** | `Step` + `WorldState` | a device name (placement is separate from function) |
| **Receipt** | a run | parent-chained, hash-verified `ReceiptRecord` + fail-closed `ActionAuthorization` |

Data types are frozen where they cross a boundary. Governed execution: every
manipulate step is authorized (`ALLOW/LIMIT/REQUIRE_APPROVAL/DENY`) and the
receipt is finalized *before* the action runs. Borrowed patterns:
[`docs/prior-art.md`](docs/prior-art.md).

## The engine loop (`omni_q.engine.OmniQ`)

```
start_mission → observe → plan → ┌─ apply queued constraints → recompile
                                 │  next step → route (Device) → authorize (Receipt)
                                 │           → manipulate → apply_transition (World)
                                 │           → observe → verify
                                 └─ on {constraint change · lost capability · step fail ·
                                       verify mismatch} → recompile; AutonomyMode escalates
                                        monotonically (NOMINAL→DEGRADED→LOCAL_ONLY→HOLD)
→ receipt (metrics, decisions, rejected, provenance, content hash)
```

## Composable decorators — no engine edits to add behaviour

**Planner stack** (compose outermost-first):

```
ReactivePlanner( ScheduledPlanner( RewritingPlanner( RulePlanner() ) ) )
```

| decorator | doc | does |
|-----------|-----|------|
| `RulePlanner` | — | goal + world → a linear PICK/MOVE/VERIFY graph (stand-in for a GenieX/VLA planner) |
| `RewritingPlanner` | [`docs/rewrite.md`](docs/rewrite.md) | collapse PICK+MOVE → SLIDE / NUDGE when cheap (OQ-HAND-006); apply spoken `spin`/`nudge` as graph edits (OQ-025) |
| `ScheduledPlanner` | [`docs/scheduler.md`](docs/scheduler.md) | bimanual wave scheduling, reach/load arm choice (OQ-044), collision/gripper barriers (OQ-013), style flourishes + dance-in-slack (OQ-015 / OQ-HAND-011) |
| `ReactivePlanner` | [`docs/ontology.md`](docs/ontology.md) | while the ontology reports a workspace conflict, return a safe hold graph; resume the task when it clears |

**Observe stack:** `ReactiveObserver( OntologyObserver( swarm ) )` — or, without
the swarm, `FrameObserver(detector)` (single detector + IoU/centre tracker for
stable ids), or `FakeObserver` (reads the world directly, mock only).

**Device:** `ProviderRouter([INTEL_PROVIDER, QUALCOMM_PROVIDER])`
([`docs/providers.md`](docs/providers.md)) routes one graph across either
sponsor track — perception, reasoning and each arm land on distinct devices
(complementary work, OQ-031); `keep_local` drops the cloud; a track going
offline mid-run triggers a normal replan onto the other (OQ-034).

## Natural language → graph

`omni_q.nlu.parse(text)` → a canonical goal + `(kind, value)` constraints
(`forbid_object`, `keep_local`, `prefer_arm`, `style`) + structural `mutations`
(`spin`, `nudge`, `spin_on_place`). `omni_q.mutation.RuntimeMutator` applies
them to a live engine: constraints via `add_constraint` (recompiled under the
`MissionEnvelope`), mutations registered on the `RewritingPlanner`.
[`docs/runtime-mutation.md`](docs/runtime-mutation.md).

## Action vocabulary (`omni_q.actions`)

84 ops with metadata (`category`, `requires_contact/grasp`, `supports_bimanual`,
`precision`, `style_action`, `blocking`, preconditions/effects) + composite
routines (`SPIN_PLATE`, `NAPKIN_ROUTINE`, …) with `expand()`. The scheduler and
rewriter read this instead of hard-coded op sets.
[`docs/actions.md`](docs/actions.md).

## Perception model (OQ-008)

`perception/` fine-tunes a 7-class detector (`plate cup fork spoon knife napkin
drawer`) from our own `KissTheHabit/yolov8n-hituav-thermal-finetune` on a large
real-image haul (Open Images V7 + Objects365 + COCO + LVIS) with a synthetic
MuJoCo top-up for the thin classes. [`docs/datasets.md`](docs/datasets.md) ·
[`perception/README.md`](perception/README.md).

Deploy: `best.pt → ONNX → OpenVINO IR` (Intel, run at **MXFP2** for the swarm) ·
`→ QAIRT` (Qualcomm bonus). Real thermal-model OpenVINO benchmark:
`evidence/benchmark_results/openvino_inference_2026-09-10/`.

## Intel online stack (the entry track)

Per the official brief
([`docs/challenge-briefs/intel-online-physical-ai-challenge.md`](docs/challenge-briefs/intel-online-physical-ai-challenge.md)) —
"Bimanual VLA Manipulation with Multi-Modal Reasoning":

```
Simulation   MuJoCo (or a compatible LeRobot Gym env)
Policy       Hugging Face LeRobot fine-tune — SmolVLA / Pi0.5 / ACT / other VLA/IL
Training     local or cloud (Intel provides no training infra)
Inference    Intel OpenVINO (+ OpenVINO Physical AI), Core Ultra Series 2/3
```

100-pt rubric: task completion 30 · VLA reasoning 20 · robustness / 10 seeds 15 ·
OpenVINO optimization 20 · reproducibility 10 · innovation 5. Independent audit:
[`docs/oq-004-requirements-audit.md`](docs/oq-004-requirements-audit.md);
red-team: [`docs/oq-021-red-team-findings.md`](docs/oq-021-red-team-findings.md).
Qualcomm + Speechmatics are bonus layers that stack on this entry.
