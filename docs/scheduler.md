# Bimanual task scheduler (`omni_q.scheduler`)

> OQ-012 (scheduler) · OQ-013 (collision / resource barriers) · OQ-044
> (dynamic arm-role assignment). Design authority:
> [`../integrations/intel/so101_capability_map.md`](../integrations/intel/so101_capability_map.md).

`schedule(graph, world)` takes a planner `PlanGraph` plus a `WorldState` and
returns a `Schedule`: per-step arm, concurrency waves, and the barriers where a
shared workspace forced serialisation.

The module is **standalone** — it only reads `PlanGraph` / `Step` / `WorldState`
and never writes to the engine. Integration is one call:

```python
from omni_q.scheduler import schedule
annotated = schedule(graph, world).annotate(graph)   # engine runs this unchanged
```

`annotate()` returns a copy of the graph with every manipulate `Step.arm` filled
and each step additionally depending on the whole previous wave, so the current
dep-gated `OmniQ` loop executes the waves in order with no engine change. A real
dual-arm executor reads `Schedule.waves` directly.

## Model

| Concept | Rule |
|---------|------|
| **Regions** | `zone name → Region(id, x_center)` via `DEFAULT_LAYOUT`. `x_center` is signed across the centreline: `<0` right, `>0` left, `~0` shared centre. Unknown zone → centre (conservative). Poses are not modelled yet (OQ-009 gap); zones are the proxy. |
| **Reach** | Mirrored pair: `left` covers `x ≥ -overlap`, `right` covers `x ≤ +overlap` (`overlap=0.15`). Centre is reachable by both. Per the capability map, reach is rarely the binding constraint. |
| **Arm choice** (OQ-044) | 1) explicit `Step.arm`; 2) a `prefer_arm` constraint that reaches; 3) the reaching arm with the lower projected load. A pick/move pair stays on one arm. A chain containing a `HANDOFF` splits: giver keeps the pre-steps, the named `to_actor` takes the rest. Role is chosen, never fixed L/R. |
| **Gripper occupancy** | One step per arm per wave, and a `PICK` needs that arm's hand free (its prior object already moved / handed off). |
| **Waves** | Greedy layering: each wave takes ready steps (deps met, arm free, region clear), ≤1 per arm. `observe` / `verify` steps quiesce both arms → solo wave. |
| **Barriers** (OQ-013) | Two steps conflict if their regions are equal or within `overlap` **and** they would share a wave; the later one is pushed to a new wave and a `Barrier(kind="workspace")` is recorded. Also `kind ∈ {gripper, verify, reach}`. Invariant: no wave holds two manipulate steps in conflicting regions. |

## `Schedule`

- `waves: list[Wave]` — `Wave.index`, `Wave.steps: list[ScheduledStep]` (`step_id`, `arm`, `region_id`, `wave`).
- `barriers: list[Barrier]` — `between`, `kind`, `reason`.
- `arm_timeline: dict[str, list[str]]` — ordered step ids per arm (feeds OQ-020 viz).
- `metrics` — `waves`, `max_parallelism`, `handoffs`, `serialized_conflicts`.
- `assignment` / `regions` — per-step arm and region id.
- `annotate(graph)` / `to_dict()`.

## Try it

```
PYTHONPATH=src python -m omni_q.scheduler          # smoke on the sample world
PYTHONPATH=src python -m pytest -q tests/test_scheduler.py
```

## Not yet

- Wiring `schedule()` into `RulePlanner` (kept out while the core is churning).
- Style-aware flourish scheduling (OQ-015) — keep `present_*` steps but place
  them only in slack waves.
- Real object poses instead of zone→region (needs OQ-006/OQ-008 data).
- Auto-inserting `HANDOFF` when a chain spans both arms (today it flags a
  `reach` barrier).
