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

## Follow-up: the real blocker is a grasp-frame mismatch (same day)

Chasing the "gripper never reaches the object" finding above to its root.

**Measured, with the correct arm** (an earlier probe of mine used the wrong
one — `fork_1` is only reachable by the *left* arm at 0.257 m; the right arm is
0.632 m away, well outside the ~0.386 m envelope. Arm routing in the system is
correct; my probe was not):

The fingertip pads run **up** the finger. At a stalled fork pinch:

| pad | world z |
| --- | --- |
| `pad_1` (fingertip) | 0.0076 |
| `pad_2` | 0.0174 |
| `pad_3` | 0.0317 |
| **`pad_4` — the geom the IK tracks** | **0.0510** |

`_do_pick` computes `grasp_z` from the **object's centre** and demands it of
`pad_4`, which sits ~43 mm above the fingertip. For `fork_1` (centre 0.0027 m)
that asks the fingertip to reach ~4 cm **below the table**. Unsatisfiable, so
the solver stalls with the tip at 7.6 mm — grazing the fork (1–3 N contacts)
but never straddling it.

This single mismatch predicts the entire measured success pattern:

| object | grasp height | vs 43 mm offset | held |
| --- | --- | --- | --- |
| `cup_1` | 0.050 m | above | **10/10** |
| `plate_1` | 0.016 m (rim) | below | 9/12 |
| `fork_1`/`spoon_1`/`napkin_1` | 0.003–0.004 m | far below | **0** |

`cup_1` is the only object taller than the offset, and the only reliable grasp.

**Corroboration:** the SO-101 RL post the hosts circulated names this as its #1
IK bug — *"Mixed up `graspframe` (contact point) vs `gripperframe` (TCP)"* —
and reports needing visualization to find it. It also places its fingertip
collision boxes **at the tip**, not up the finger.

**Second defect, independent:** `_ik_reach_pad(tol=0.01)` stops once the pad is
within **10 mm**. `fork_1`/`spoon_1` are 8 mm thick and `napkin_1` 6 mm — the
solver declares success while off by more than the whole object.

### Attempted fix, and why it is OFF

`_fingertip_pinch_offset` (`OMNIQ_FINGERTIP_PINCH=1`, default off) adds the
measured tip↔`pad_4` height to the pinch target. Ten trials:

| object | baseline | fingertip pinch |
| --- | --- | --- |
| `cup_1` held | **10/10** | **0/10** |
| `cup_1` placed | 8/10 | 0/10 |
| `plate_1` held | 9/10 | **10/10** |
| flat objects | 0 | 0 |

Directionally right (`plate_1` improved) but a **net regression**, so it stays
disabled. The implementation is naive: it samples the tip↔pad offset at the
*current* pose, and that offset is wrist-orientation dependent, so it is stale
by the time the pinch executes. A correct version resolves the offset in the
gripper frame at the pinch orientation. Flat objects also stayed at zero, so
the frame mismatch is necessary but not sufficient — the 10 mm tolerance, and
possibly pad geometry against a 6-8 mm target, remain.

## Can the simulator legitimately be changed to help? (checked, mostly no)

The SO-101 post's fingertip collision boxes are **2.5 mm**; ours are 8 mm tall
at the tip (`pad_1` half-extents `0.001 x 0.005 x 0.004`). Tempting to shrink
them — an 8 mm pad cannot straddle an 8 mm fork without its underside reaching
the table exactly, which is the ~11 mm descent floor measured above.

**Checked against the real finger first, and the answer is no.** On
`left_Fixed_Jaw` the two group-3 collision meshes span roughly 74 mm in z and
±16-25 mm in the other axes; the pads are a **2 mm-thin strip** running along
the gripping edge at local y = -0.057 → -0.101. The pads are far *thinner* than
the finger they sit on, not bulkier than it. Shrinking them would not be a
fidelity correction — it would be making the gripper better than the hardware,
i.e. exactly the class of change the no-simulation-cheating rule exists to stop
(cf. the reverted `contype`/`conaffinity` = 0 incident).

So the 8 mm fingertip is a fair model, and a real parallel-jaw gripper genuinely
struggles to pinch an 8 mm object lying on a flat table. **That is a physical
constraint, not a simulator artifact.**

### What that leaves

A pre-grasp manoeuvre, all of which use real physics and stay inside the rules:

1. **push to the table edge**, then grasp from the side with the pads clear of
   the surface;
2. **tilt/scoop** — press one fingertip down to lever the far edge up, then
   close on the raised lip;
3. **slide onto a raised feature** (the plate, the tray) and grasp from there.

Worth noting the SO-101 RL agent independently discovered a *nudge before
grasping* that nobody rewarded — the same conclusion arrived at from the
learned-control side.

This also re-scopes OQ-010-TELEOP: a human teleoperator asked to grasp a fork
will likely discover the same manoeuvre, and whether they can is still the
cheapest available signal.

## Two-arm cooperation is impossible in this scene as authored

Using both arms on a tricky object is the right physical answer — one fingertip
as a wall or lever while the other slides the object against it, which needs no
grasp to begin (pushing is non-prehensile). It is also a scored rubric item:
OQ-004's audit lists two-arm coordination as PARTIAL.

**Measured.** Bases are 0.52 m apart with a ~0.386 m reach each, so a genuine
shared workspace exists — a midline band, x ∈ [-0.12, 0.13], y ∈ [-0.09, 0.30].
**No object is in it:**

| object | dist L | dist R | both arms? |
| --- | --- | --- | --- |
| `cup_1` | 0.494 | **0.279** | no |
| `plate_1` | **0.309** | 0.480 | no |
| `fork_1` | **0.257** | 0.632 | no |
| `spoon_1` | 0.632 | **0.257** | no |
| `napkin_1` | **0.184** | 0.513 | no |

Every item is single-arm-only. So HANDOFF, STABILIZE and COOPERATIVE_ROTATE
(OQ-011) cannot be demonstrated on any current object, and neither can a
two-arm flat-object manoeuvre — not because the arms cannot cooperate, but
because nothing is placed where both can reach.

This is a scene-authoring gap, and repositioning is already the accepted remedy
here: `intel_sim.py` records `fork_1`/`spoon_1` being moved once before for
exactly this reason (they sat outside the reach envelope entirely).

**Proposed, cheapest first:**

1. give the flat objects a **staging pose inside the shared band** — or a
   "slide to centre" step — so a second arm can participate at all;
2. then the two-arm manoeuvre: arm A plants a fingertip as a stop, arm B pushes
   the fork against it so it tips, B closes on the raised edge. All real
   physics, no collision exemptions;
3. `table_grazing` and the wrist cameras now make that manoeuvre observable —
   a few millimetres of fingertip-vs-table gap is invisible from overhead.

Note this does not conflict with the pad finding above: the fingertip cannot
straddle an 8 mm object lying flat, so the object has to be *tipped* rather
than pinched from above. Two arms are what make tipping possible.

## CORRECTION: the gripper closes to 2.6 mm at the fingertip, not 21 mm

The premise underneath this whole investigation was wrong, and the correction
changes the conclusion.

`intel_sim.py` justifies `plate_1`'s 32 mm rim by citing
`so101_capability_map.md`: *"the SO-101 gripper's own fully-closed pad gap is
21.3 mm ... a 14 mm rim is geometrically thinner than the gripper can ever close
to."* Measured directly, in the fixed-jaw frame at the closed limit:

| pad pair | along-finger offset | face gap when closed |
| --- | --- | --- |
| **pad_1 (fingertip)** | 3.1 mm | **2.6 mm** |
| pad_2 | 2.6 mm | 8.3 mm |
| pad_3 | 0.5 mm | 14.1 mm |
| pad_4 (base) | 0.4 mm | 19.3 mm |

The pads pair 1↔1 … 4↔4 (offsets under ~3 mm, so they genuinely oppose), and
**the jaws close as a wedge**: near-touching at the tip, ~19 mm at the base.
That is simply how a pivoting jaw closes — this is a pincer, not a parallel
gripper. Independently corroborated by the collision meshes themselves, whose
minimum jaw-to-jaw gap in the finger region at full close is **3.1–4.6 mm**.

The 21.3 mm figure is the *base* of the wedge, not the tip.

### What this changes

1. **Thin objects are grippable** — at the fingertip. `fork_1`/`spoon_1` (8 mm)
   and `napkin_1` (6 mm) all fit inside a 2.6 mm close.
2. **`plate_1`'s 32 mm rim is unjustified.** A 14 mm rim was always graspable at
   the tip. A real plate rim is ~5-10 mm, so the scene currently carries an
   unrealistically thick plate that makes the task *easier* than reality —
   worth reverting on realism grounds, separately from any grasp work.
3. **It explains the grip failure completely.** `_do_pick` tracks
   `fixed_jaw_pad_4` — the **wide end** of the wedge. At pad_4 the jaws close
   only to 19.3 mm, so an 8 mm object cannot be pinched there *at any height*,
   however good the IK is. Thin objects must be taken at the fingertip.

That makes `_ik_reach_pad(track_tcp=True)` (tracking the pad_1 midpoint, added
above) the correct fix rather than a workaround — it needs the downstream pinch
constants re-tuned to the new tracked point, which is what the earlier naive
attempt got wrong when it regressed `cup_1`.

**Sources checked while resolving this:** the vendored asset is faithful — the
`jaw_pad_*` geoms are in upstream menagerie `trs_so_arm100` at the pinned commit
with identical sizes and positions (so `SOURCE.md`'s "verbatim copy" claim
stands; an earlier suspicion that they were a local addition was wrong).
Menagerie has no SO-101 model, and neither the LeRobot SO-101 page nor
roboticscenter.ai publishes a jaw-gap figure, so the model's own geometry is the
authority here.

## Host guidance applied: jaw stall + contact buffer as grasp sensors

Two notes relayed from the challenge hosts' Discord:

1. *A target angle tells the gripper to close to 0.5 rad, a rigid object blocks
   it at 0.7 rad, and the controller endlessly ramps force to push the finger
   through the solid object. Look at the object collision mesh and the gripper
   closing position.*
2. *Look at the simulation's contact buffer — MuJoCo tracks all active
   intersections between geoms.*

`_set_gripper` used to set `ctrl` and step blind. It now watches the jaw while
closing, declares a **stall** when the jaw has stopped for a sustained window
(25 steps, < 5 mrad travel) while still short of its target, and returns the
actuator's actual angle plus every pad/object contact from `data.contact` with
its force. That evidence rides in every PICK receipt as `grasp_sensor` and maps
directly onto `skills/verification/grasp.py`'s `GraspEvidence`
(`contact_pad_count`, `max_contact_force_n`).

What the new sensor immediately showed, single picks with the correct arm:

| object | held | jaw commanded | jaw actual | stalled | pads | max force |
| --- | --- | --- | --- | --- | --- | --- |
| `cup_1` | yes | −0.174 | **−0.095** | no (creeping) | 7 | 8.4 N |
| `plate_1` | yes | −0.174 | **+1.026** | **yes @ 1.004** | 3 | **14.9 N** |
| `fork_1` | no | −0.174 | −0.012 | no | **0** | 0.0 N |

- `fork_1`: the jaw closes to −0.012 with **zero** pad/object contacts. It
  closed on nothing — the fingertips are above the fork, as diagnosed above.
- `cup_1`: blocked at −0.095 by the cup, seven pads touching. A real grasp,
  and now a *reported* one.
- **`plate_1` is the host's warning, verbatim.** The jaw stalls at ~1.0 rad
  on the rim's top face and never closes on the rim at all; the plate lifts
  because the servo, ramping toward its limit for 180 steps, presses a pad
  down at ~15 N and drags the rim up. Lift is 21.1 mm against a 20 mm "held"
  threshold. That is why plate has always been the fragile 9/10 grasp.

### Why the bounded hold is opt-in

With a bounded squeeze after stall (`OMNIQ_GRIPPER_STALL_HOLD=1`, ~2 N), the
plate goes **9/10 → 0/10** — because its grasp *depends* on the force-ramp.
The realistic behaviour therefore breaks the demo's plate. Detection and
evidence are always on; the hold stays opt-in until the rim approach is fixed
so the jaw actually closes on the rim instead of stalling on top of it. The
receipt no longer hides which grasps are legitimate and which are force-driven.
