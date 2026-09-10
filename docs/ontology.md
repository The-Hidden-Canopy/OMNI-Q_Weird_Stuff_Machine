# The ontology layer — a living scene graph between perception and OMNI-Q

> `src/omni_q/ontology.py`. Turns a swarm of cheap specialist detectors into a
> shared, reconciled world model that OMNI-Q *reacts to*, instead of making
> OMNI-Q re-derive the world from pixels every frame.

```
cheap YOLO specialists          Ontology (fusion authority)          OMNI-Q
─────────────────────           ──────────────────────────           ──────
yolo_objects  ─┐                entities + relations                 reacts to a
yolo_human    ─┼─▶  Claims  ─▶  IS-A reconciliation           ─▶     meaningful
yolo_hazard   ─┤   (hypotheses,  temporal association                state delta
yolo_afford.  ─┘    provenance)  preserved disagreement
                                 workspace-conflict detection
```

## Four layers, each with a reason to exist

| layer | answers | in the code |
|-------|---------|-------------|
| **Perception** | *what appears to be happening?* | detector callables `frame → [Claim]` — deliberately stupid specialists |
| **Ontology** | *what do we currently believe exists?* | `Ontology` — the fusion authority (this file) |
| **OMNI-Q** | *given that world and my objective, what next?* | `omni_q.engine` + planner + `scheduler` |
| **Execution** | *what am I permitted and physically able to do?* | `providers` / `MissionEnvelope` / arm drivers |

## Claims, not mutations

A detector never edits reality. It emits `Claim(source_model, entity_type,
confidence, geometry, device, ts, attrs)` — a hypothesis with provenance. The
ontology decides what becomes authoritative. This matches the receipts /
provenance spine: every authoritative entity traces back to *which model on
which camera said what, when, how sure*.

`claims_from_detections("yolo_objects", dets)` lifts any plain YOLO-style
detector (`frame → [Detection2D]`, from `frame_observer`) into provenance-carrying
claims — swap the stub for real **MXFP2** YOLO inference (the Intel-track
format) and nothing downstream changes.

## Reconciliation

- **IS-A hierarchy** (`_ISA`): `mug`, `wineglass`, `saucer`, `butter_knife` …
  roll up to `cup` / `plate` / `knife`. Two detectors calling the same box
  `cup` and `mug` **fuse to one entity** and their confidences *reinforce*
  (`type_conf` computed over canonical buckets).
- **Temporal association**: IoU + centre-gate, per canonical type; ids are
  stable across frames (`cup_1`, `hand_1`), survive occlusion for `max_missed`
  frames, then emit `entity.lost`.
- **Preserved disagreement**: a big spatial overlap with *incompatible* types
  (`fork` .94 vs `knife` .61) → **one entity, `conflict=True`**, both votes
  kept. A conflicted entity is projected to `WorldState` as `DataStatus.FALLBACK`
  — the planner (which already respects knowledge status) **won't manipulate
  it**; OMNI-Q can reposition a camera / request another observation.

## Relations + workspace conflict

After each ingest the `relation_rules` derive relations over the *live*
entities (this frame, not memory): `near`, `intersects`, and — via
`workspace_conflict_rule` — a `human` entity in an `arm` entity's zone. That
raises a `workspace.conflict` delta (and `workspace.clear` when the hand
leaves). `Observation.workspace_clear` reflects it, so the existing
verify/replan loop reacts with no engine change.

### Reach + place-setting (OQ-ONT-007)

Two more default rules turn the OQ-007 table geometry into scene-graph facts:

- **`reachable_by_rule`** — `<obj> reachable_by left_arm|right_arm` for each
  arm whose mirrored-pair envelope covers the object's zone. It uses
  `omni_q.scheduler`'s `DEFAULT_LAYOUT` + `arm_reaches` — the *same* reach
  model the scheduler plans with — so a reach relation in the ontology means
  the scheduler agrees. Pure geometry: emitted whether or not an arm was
  detected this frame (availability is the provider's concern), and the
  object is a stable `<side>_arm` label, not a per-run entity id.
- **`missing_from_rule`** — reads `Ontology(place_setting=...)`
  (`{zone: {canonical classes}}`, default `DEFAULT_PLACE_SETTING` for the
  mock `setting_1/2`; pass the real spec per scene). Per setting zone it
  emits `<obj|class> missing_from <zone>` for every unfilled slot — naming
  the actual entity when one sits in the wrong zone, else the bare class —
  plus one `<zone> incomplete setting` marker whose `conf` is the fraction
  of the setting still missing. Both clear automatically once the slot fills.

Together they compose the headline read the planner/UI wants:
`setting_1 incomplete setting` + `cup_1 missing_from setting_1` +
`cup_1 reachable_by left_arm` → "left setting incomplete; left_arm can reach
the cup." Relations are projected in `snapshot()["relations"]`; no new deltas
(an incomplete setting is the normal start state, not a wake event).

## The ontology as an attention mechanism

`ingest()` returns `list[Delta]`. `wakes(delta)` says which are worth waking the
expensive reasoner for:

```
camera @ 30 fps → tiny perception workers → ontology
  … nothing interesting … nothing interesting … HAND ENTERED WORKSPACE → wake OMNI-Q
```

So the perception swarm runs at **MXFP2** continuously on the Intel box —
that's the pinned format for the entry track: the UE8M0 power-of-two scale
dequantises with a shift (not a multiply) and enables the multiply-free Linear
path, which is what makes it *cheap on Core Ultra*, not just small on disk.
NVINT2 (4-level signed grid) is the Qualcomm-side / per-specialist fallback for
any detector that fails the cosine / detection-parity gate at MXFP2. OMNI-Q's
own reasoning path stays FP16/INT8 and only fires on a real state change.
`OntologyObserver.wake` is the boolean an event loop checks. (Codec:
`integrations/qualcomm/lowbit/`; evidence: `evidence/benchmark_results/yolo_2bit_cpu_*`,
`omni_quant_2bit_*`.)

## `OntologyObserver` — the drop-in

Implements the `Observe` contract, so it slots into `OmniQ` exactly where
`FakeObserver` / `FrameObserver` did:

```python
from omni_q import build_mock_engine
from omni_q.ontology import OntologyObserver, stub_swarm

engine = build_mock_engine()
engine.observer = OntologyObserver(stub_swarm(engine.world))   # 2 stub specialists
engine.run("inspect and correct the workspace")
engine.observer.ontology.snapshot()      # entities, relations, conflicts, workspace flag
engine.observer.wake                     # did anything meaningful just change?
```

`stub_swarm(world)` is three model-free specialists (objects, robot-state /
both arms, human/hand driven by `world.hand_xy`) for tests and the demo.
Replace each entry with a real detector.

## Reacting to the attention signal — `src/omni_q/reactor.py`

The ontology raising `workspace.conflict` is only useful if OMNI-Q acts on it.
`reactor.py` turns it into **halt → re-observe → resume**, entirely through the
existing engine seams:

- **`ReactiveObserver`** wraps the `OntologyObserver`. Each `observe`, on a
  *new* conflict it records a `HALT` event (`"human_1 intersects left_arm_1"`)
  and, if given the engine, queues a `style=freeze` constraint so the loop
  recompiles promptly; on clear it records `RESUME` and queues
  `style=minimum_time`.
- **`ReactivePlanner`** is a `Plan` decorator (compose it outermost):
  while `observer.blocked`, every `plan`/`replan` returns a **wait graph**
  (`STABILIZE` then `VERIFY`) so the arms hold steady; the `VERIFY` keeps
  failing while the goal is unmet, so the engine recompiles — and the moment
  the workspace clears, the real task plan resumes and finishes.

```python
ro = ReactiveObserver(OntologyObserver(stub_swarm(engine.world)), engine=engine)
engine.observer = ro
engine.planner  = ReactivePlanner(RulePlanner(), observer=ro)
# a hand crossing the left-arm zone mid-task ->
#   PICK MOVE | HALT | STABILIZE VERIFY x3 | RESUME | STABILIZE PICK MOVE VERIFY  (resolved)
```

The receipt / event stream then shows *which model observed the hand, how the
ontology changed, why OMNI-Q halted, and what the arms actually did* — not
"YOLO found a cup".

## Try it

```
PYTHONPATH=src python -m pytest -q tests/test_ontology.py
```

## Not yet

- Real specialist YOLOs (`perception/` fine-tunes the object one; hazard /
  affordance / robot-state detectors are new heads).
- Per-camera / per-modality models and cross-camera entity linking.
- Running the swarm at MXFP2 on real Core Ultra silicon (Series 2/3) — the CPU
  round-trip is measured (`yolo_2bit_cpu_*`); NPU/iGPU MXFP2→INT8 fused
  dequant kernel is the remaining piece.
