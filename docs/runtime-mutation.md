# Runtime constraint mutation (`omni_q.mutation`)

> OQ-025. Depends on OQ-023 (`omni_q.nlu`) and the engine's constraint /
> replan loop (OQ-018).

`RuntimeMutator(engine).apply("<spoken instruction>")` turns a live operator
utterance into a **safe** change to a running or about-to-run `OmniQ`.

```python
from omni_q import build_mock_engine
from omni_q.mutation import RuntimeMutator

engine = build_mock_engine()
mut = RuntimeMutator(engine)
mut.apply("don't touch the red connector")   # -> applied forbid_object=connector_2
mut.apply("keep everything local")           # -> applied keep_local
mut.apply("spin that plate")                 # -> deferred (needs OQ-014 primitive)
engine.run("inspect and correct the workspace")
```

## Two kinds of mutation

| Kind | Examples | Path |
|------|----------|------|
| **Constraint-shaped** — applied | *"don't touch the red cup"*, *"keep it local"*, *"use the left arm"*, *"show off"* | queued via `engine.add_constraint`, which validates it; the engine recompiles atomically on its next loop iteration. `_effective_world()` re-lays the fixed `MissionEnvelope` on top, so a live change can only **narrow** authority, never widen it. |
| **Structural** — deferred | *"spin that plate"*, *"move the fork farther left"* | parsed into a descriptor and reported in `result.deferred`, not executed. `spin` waits on the rotate→handoff→regrasp primitive (OQ-011 / OQ-014); `nudge` waits on within-zone poses (OQ-009 remainder). |

## `MutationResult`

- `applied: list[(kind, value)]` — constraints queued on the engine.
- `rejected: list[(kind, value, reason)]` — failed validation; **reported, never raised**, so a bad utterance can't crash a run.
- `deferred: list[(kind, payload, reason)]` — structural mutations awaiting a primitive.
- `ok` — something applied and nothing rejected.
- `summary()` / `as_dict()`.

`RuntimeMutator.history` keeps every result; `apply_all([...])` runs a transcript.

## Mid-run

The engine drains `_pending_constraints` at the top of its execution loop, so
calling `mut.apply(...)` from a Speechmatics callback (OQ-024) or a UI control
while `engine.run()` is in progress is picked up on the next step and triggers a
`graph.recompiled`. See `tests/test_mutation.py::test_mid_run_mutation_recompiles_the_live_graph`.

## Try it

```
PYTHONPATH=src python -m omni_q.mutation
PYTHONPATH=src python -m pytest -q tests/test_mutation.py
```
