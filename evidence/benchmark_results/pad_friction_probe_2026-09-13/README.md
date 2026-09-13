# Fingertip pad torsional-friction probe — 2026-09-13

**Hypothesis (from [`docs/oq-010-external-sources-crosscheck-2026-09-13.md`](../../../docs/oq-010-external-sources-crosscheck-2026-09-13.md)):**
the SO-101 RL project that solved a cube grasp in this simulator used pad
friction `1 0.05 0.001`; ours runs `3.00 0.020 0.001` — **2.5x lower torsional
friction**. Torsional friction resists a pinched object rotating out of the
grip, which is the suspected failure mode of the thin flat objects
(`fork_1`, `spoon_1`, `napkin_1`) that no controller has ever grasped.

**Result: FALSIFIED.** Ten randomized trials per arm, same seed (900), only the
middle friction term changed via `OMNIQ_PAD_FRICTION`:

| arm | pad friction | cup_1 held | plate_1 held | cup_1 placed | fork/spoon/napkin held |
| --- | --- | --- | --- | --- | --- |
| baseline | `3.00 0.020 0.001` | 10/10 | 9/10 | 8/10 | **0** |
| probe | `3.00 0.050 0.001` | 10/10 | 9/10 | 8/10 | **0** |

Byte-identical per-object outcomes. Not a single trial changed.

## What the receipts actually show

Friction was never the binding constraint, because **the gripper is not reaching
the object**. From `trial-00-seed-900.json`, at the pinch pose:

| object | pinch position error | approach contact force | lift achieved | held |
| --- | --- | --- | --- | --- |
| `cup_1` | 0.0098 m (clearance) | **0.28 N** | +0.0247 m | yes |
| `spoon_1` | 0.039 m | **32.7 N** | 0.0000 m | no |
| `napkin_1` | 0.057 m / 0.187 m | 10.5 N / 32.3 N | −0.0015 m | no |
| `fork_1` | **0.147 m** | **32.7 N** | −0.0007 m | no |

A 147 mm position error is not a contact-quality problem. The arm is pressing
into the table at ~32 N — over 100x the successful grasp's 0.28 N — while its
pad is up to 19 cm from where it was asked to be. It then closes on nothing and
"lifts" the object by −0.7 mm.

Orientation is satisfied at clearance but `orientation_satisfied: false` at the
pinch for every failing object, so the solver is losing the pose on the final
descent specifically.

## Consequences

1. **Contact parameters are not the lever.** Friction, and by extension the
   fingertip-geometry lesson already implemented (8 box pads, elliptic cone,
   `impratio=10`), are not what is stopping flat-object grasps.
2. **Neither is a learned residual controller.** The bounded envelope proposed
   for residual RL is `max_delta_mm: 4`. A 4 mm correction cannot close a
   147 mm error. This is measured support for the tension already recorded in
   [`docs/skill-runtime-architecture-2026-09-13.md`](../../../docs/skill-runtime-architecture-2026-09-13.md):
   residual RL improves a controller that is nearly right; it cannot discover a
   pose the solver never reaches.
3. **The real blocker is IK convergence on the final descent to a low, flat
   target** — consistent with `intel_sim.py`'s own record of `_ik_reach_pad`
   failing to converge for these objects, and *not* resolved by the earlier
   repositioning fix.

## Reproduce

```
OMNIQ_PAD_FRICTION="3.00 0.050 0.001" python -c "
import sys; sys.path.insert(0,'src')
from omni_q.intel_sim import run_intel_table_evaluation_report
print(run_intel_table_evaluation_report('out', trials=10, seed=900)['per_object_summary'])"
```

`PAD_FRICTION` (`intel_sim.py`) defaults to the value that has always run, so
this probe required no edit to tracked physics.
