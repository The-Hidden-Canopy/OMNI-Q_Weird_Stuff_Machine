# Bimanual task scheduler (`omni_q.scheduler`)

> OQ-012 (scheduler) · OQ-013 (collision / resource barriers) · OQ-044
> (dynamic arm-role assignment) · OQ-015 (style flourishes). Design authority:
> [`../integrations/intel/so101_capability_map.md`](../integrations/intel/so101_capability_map.md).

`schedule(graph, world)` takes a planner `PlanGraph` plus a `WorldState` and
returns a `Schedule`: per-step arm, concurrency waves, and the barriers where a
shared workspace forced serialisation.

For the N-arm expansion, [`docs/manipulation-fleet.md`](manipulation-fleet.md)
defines the resource substrate: individual manipulators, dynamic biarm units,
capability leases, and workspace reservations. The scheduler below remains the
existing two-arm path until the fleet-aware assignment and real-time execution
acceptance gates land; its `max_parallelism` metric must not be read as proof of
physical simultaneity.

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

With a fleet snapshot, the same seam exposes resource participants for
multi-biarm planning:

```python
from omni_q.fleet import ManipulationFleet
from omni_q.scheduler import schedule

fleet = ManipulationFleet.from_ids(
    ("arm_1", "arm_2", "arm_3", "arm_4"),
    capabilities={arm: frozenset({"CO_ROTATE"}) for arm in
                  ("arm_1", "arm_2", "arm_3", "arm_4")},
    workspace_regions={arm: frozenset({"table.NW", "table.SE"}) for arm in
                       ("arm_1", "arm_2", "arm_3", "arm_4")},
    preferred_pairs=(("arm_1", "arm_2"), ("arm_3", "arm_4")),
)
planned = schedule(graph, world, fleet=fleet)
planned.resource_assignment  # step id -> all participating resource ids
```

Fleet mode permits disjoint biarm units to occupy one wave. A bimanual step's
`ScheduledStep.participants` is the authoritative resource set for a future
fleet executor; `Step.arm` remains the legacy primary-arm field for the
current single-step engine.

### Live: `ScheduledPlanner`

Wrap any `Plan` provider so it emits scheduled graphs automatically:

```python
from omni_q import build_mock_engine
from omni_q.fakes import RulePlanner
from omni_q.scheduler import ScheduledPlanner

engine = build_mock_engine()
engine.planner = ScheduledPlanner(RulePlanner())   # plug-compatible with OmniQ
engine.run("inspect and correct the workspace")
engine.planner.last_schedule.waves                 # for the UI (OQ-020)
```

`ScheduledPlanner` forwards `plan` / `replan` / `last_decision` to the inner
planner and applies `schedule(...).annotate(...)` to the result. It keeps the
last `Schedule` on `.last_schedule`; if scheduling raises it degrades to the
inner graph and records `.last_error` — a scheduler bug can't break a run. This
is the zero-touch way to make scheduling live without editing `RulePlanner` or
the engine.

## Model

| Concept | Rule |
|---------|------|
| **Regions** | `zone name → Region(id, x_center)` via `DEFAULT_LAYOUT`. `x_center` is signed across the centreline: `<0` right, `>0` left, `~0` shared centre. Unknown zone → centre (conservative). Poses are not modelled yet (OQ-009 gap); zones are the proxy. |
| **Reach** | Mirrored pair: `left` covers `x ≥ -overlap`, `right` covers `x ≤ +overlap` (`overlap=0.15`). Centre is reachable by both. Per the capability map, reach is rarely the binding constraint. |
| **Arm choice** (OQ-044) | 1) explicit `Step.arm`; 2) a `prefer_arm` constraint that reaches; 3) the reaching arm with the lower projected load. A pick/move pair stays on one arm. A chain containing a `HANDOFF` splits: giver keeps the pre-steps, the named `to_actor` takes the rest. Role is chosen, never fixed L/R. |
| **Gripper occupancy** | One step per arm per wave, and a `PICK` needs that arm's hand free (its prior object already moved / handed off). |
| **Waves** | Greedy layering: each wave takes ready steps (deps met, arm free, region clear), ≤1 per arm. `observe` / `verify` steps quiesce both arms → solo wave. |
| **Barriers** (OQ-013) | Two steps conflict if their regions are equal or within `overlap` **and** they would share a wave; the later one is pushed to a new wave and a `Barrier(kind="workspace")` is recorded. Also `kind ∈ {gripper, verify, reach, flourish}`. Invariant: no wave holds two manipulate steps in conflicting regions. |
| **Flourishes** (OQ-015) | Steps with `op ∈ {PRESENT, FLOURISH, SHOWCASE, SPIN_SHOW}` are showmanship. They're scheduled *after* the mandatory plan: slotted into an existing slack wave if one fits, else one single flourish wave is inserted before `verify` (`allow_flourish_wave=True`), else dropped with a `Barrier(kind="flourish")`. A flourish never blocks another step (`annotate()` excludes it from previous-wave deps) and is never a dependency of `verify` — the goal path is unchanged. Metrics: `flourishes_scheduled`, `flourishes_dropped`, `flourish_waves_added`. Without a `style=show_off` constraint the planner emits none. |
| **Dance-in-slack** (OQ-HAND-011) | With a dance style mode (`dance` / `synchronized` / `mirrored` / `take_turns` — from spoken vocab via `nlu`), `_fill_idle_slack` appends non-blocking `IDLE_FLOURISH` steps to *already-idle* arms in *already-existing* non-final waves — never adding a wave, never reordering real work. `synchronized` uses one primitive (`SWAY`) on every idle arm; `mirrored` uses `MIRROR`; the rest cycle `actions.IDLE_FLOURISH_OPS`. `minimum_time` / `freeze` strip every flourish. Metric: `idle_flourishes`. These are scheduler-invented (not in the plan graph): `annotate()` leaves them out by default, or emits them as runnable leaf `Step`s with `execute_flourishes=True` (below). Ops execute as free-space gestures — no object, no grasp — no-op in `FakeManipulator`, real joint motion in `intel_sim`'s coarse-pose path (`_FLOURISH_GESTURES`). |

## `Schedule`

- `waves: list[Wave]` — `Wave.index`, `Wave.steps: list[ScheduledStep]` (`step_id`, `arm`, `region_id`, `flourish`, `op` — `op` set only on idle-slack flourishes).
- `barriers: list[Barrier]` — `between`, `kind`, `reason`.
- `arm_timeline: dict[str, list[str]]` — ordered step ids per arm (feeds OQ-020 viz).
- `dropped: list[str]` — flourishes omitted for lack of slack.
- `metrics` — `waves`, `max_parallelism`, `handoffs`, `serialized_conflicts`, `flourishes_scheduled`, `flourishes_dropped`, `flourish_waves_added`.
- `assignment` / `regions` — per-step arm and region id.
- `annotate(graph, *, execute_flourishes=False)` / `to_dict()`. `execute_flourishes=True` also emits the idle-slack flourishes as non-blocking leaf `Step`s (arm set, deps on the prior wave, nothing depends on them) so the engine runs the dance alongside the work.

## Try it

```
PYTHONPATH=src python -m omni_q.scheduler          # smoke on the sample world
PYTHONPATH=src python -m pytest -q tests/test_scheduler.py
```

## Not yet

- Wiring `schedule()` into `RulePlanner` (kept out while the core is churning).
- Real object poses instead of zone→region — `Detection.pose` now exists (OQ-009,
  filled by `intel_sim`); the region model still keys on zone strings by design.
- Auto-inserting `HANDOFF` when a chain spans both arms (today it flags a
  `reach` barrier).
