# Demo

> One nasty, visible, reproducible behavior — and a repo that proves it wasn't
> smoke and mirrors.

This demo is built for the **Intel Online Physical AI Challenge** — the only
track we qualify to enter. Everything not on that path (Qualcomm, Intel
onsite, Speechmatics) is bonus work layered on top, not the entry.

## The behavior

Intel online entry scenario, "Setting Up a Dinner Table":

> "Open the top drawer, pick up the plate with arm A, place it on the table,
> pick up the mug with arm B, pour water into the mug with arm A."

The system visibly runs:

```
NATURAL-LANGUAGE GOAL → OMNI Q → execution graph → task decomposition
       → LEFT_ARM / RIGHT_ARM capability assignment → dual SO-101 manipulation (MuJoCo)
       → camera verification → success / replan
```

The business wrapper is **OMNI-Q Home / OMNI-Q Chef**: a physical host that
works with a homeowner or personal chef as another set of hands. The benchmark
demonstrates configured etiquette, physical reference resolution, recovery, and
bounded personality. **TableOps** is the customer-facing hospitality
expansion—banquet reset and event changeover—while room-scale inventory and live
multi-camera claims remain roadmap work. See [`docs/tableops.md`](docs/tableops.md).

Bonus, not part of the entry: a Speechmatics voice layer can supply the goal
and live constraints by speech instead of text (see
[`integrations/speechmatics/README.md`](integrations/speechmatics/README.md)).
The realtime transport is built; on stage it needs `SPEECHMATICS_API_KEY`
exported and `--mic`. If the venue network or the key is a risk, the same CLI
replays a recorded session offline and produces the identical committed
mutations — rehearse with `--replay`, and keep a `--record` capture of the real
run as the fallback.

Then, on camera, one of these happens and the audience watches Omni Q react:

| State | Trigger | What Omni Q does |
| --- | --- | --- |
| NORMAL | goal given | builds graph, executes |
| CONSTRAINT CHANGE | *"Don't touch the red object."* / *"Keep inference on-device."* | modifies / re-places the graph |
| FAILURE / WORLD CHANGE | object moved, capability disabled, first attempt fails | replans and finishes anyway |

## Run

Mock mode works today — no hardware, no third-party deps (Python 3.10+):

```
./demo/run_demo.sh                 # or:  PYTHONPATH=src python -m omni_q.demo
```

Prints all three states end-to-end: NORMAL resolves in one graph;
CONSTRAINT CHANGE recompiles onto the right arm when the left goes offline;
WORLD CHANGE replans after an object is knocked away mid-run. Each run emits a
receipt with metrics and SHA-256 hashes of inputs / plan / actions.

Judge UI and event stream (OQ-005):

```
PYTHONPATH=src python -m omni_q.server
# open http://127.0.0.1:8770
# POST /sessions -> GET /sessions/<id>/events?cursor=<seq>
```

The judge-facing dashboard is deliberately labelled **MOCK / NO HARDWARE**. It
renders the causal event stream supplied by the active session:

- mission objective and structured workspace observation
- Observe → Plan → Act → Verify phase state
- compiled graph steps, graph revisions, and arm/device placement
- authorization decisions, constraints, replans, and capability failures
- planner candidates, dependencies, verification results, autonomy mode,
  final metrics, and receipt hash

The **Apply live constraint** control exercises the same session-scoped path as
Speechmatics: the operator supplies a justified `keep_local`, `forbid_object`,
or `prefer_arm` constraint, the engine queues it, and the graph recompiles on
the next loop iteration. The UI only renders events from the active session; it
does not call hardware directly. See [`docs/ui.md`](docs/ui.md) for the full
screen map and scope boundary.

### The MuJoCo table setting — what to record (2026-09-14)

Real contact physics, two SO-101 arms, realistic tableware built from
primitives (rimmed soup plate, hollow cup, bent-handle fork and spoon, folded
napkin). The two-arm plate is the visible thing: both grippers come in
sideways from opposite sides, slide a fingertip under the rim, close, lift
level, carry, set down. Numbers behind it are in
[`docs/contact-honesty-2026-09-13.md`](docs/contact-honesty-2026-09-13.md)
(harness seeds 900–909: 47/50 placements; seeds 910–919: 50/50).

Everything below opens a live MuJoCo viewer paced to real time; screen-record
that window. Close it to exit.

```
# 1. Full engine-driven table setting: two-arm plate first, then cup, fork,
#    spoon, napkin. Seeds 901, 903, 911-919 are known-good 5/5 runs.
.venv/Scripts/python integrations/intel/scripts/watch_sim.py --live --trial --seed 903

# 2. Just the two-arm plate pick, carry and place.
.venv/Scripts/python integrations/intel/scripts/watch_sim.py --live --pick plate_1 --move --seed 900

# 3. Mid-run change of authority. After the plate is set, the operator's
#    "don't use the left arm anymore" goes through the voice NLU into the
#    engine; the graph recompiles, the right arm finishes cup + spoon, fork
#    and napkin are dropped from the plan WITH REASONS in the receipt.
.venv/Scripts/python integrations/intel/scripts/demo_authority_change.py --seed 901 --live

# 4. Two-arm cup handoff (isolated contact scene): left grasps, lifts to the
#    shared point, right takes it while left still holds, left releases.
.venv/Scripts/python integrations/intel/scripts/watch_sim.py --live --handoff --seed 19

# GIF instead of a window (for slides): swap --live for
#    --record tmp/name.gif --camera third_person --every 40
```

What the sim is and is not: the arms, joint limits, servo torque limits and
gripper are the vendored MuJoCo Menagerie SO-ARM100 (unchanged); every grasp
is a real contact event (no welds, no teleports, no collision exemptions —
see the no-cheating rule in `src/omni_q/intel_sim.py`); a failed placement
leaves the object where it fell. Perception in these runs is the simulator's
own object state, not a camera; the camera/perception seam is the opt-in
YOLO/OpenVINO integration in
[`docs/oq-omni-vision-integration-2026-09-11.md`](docs/oq-omni-vision-integration-2026-09-11.md),
and the trained Omni checkpoint on HF is the reasoner (intent and authority),
not a motor policy — say it that way on camera.

Intel simulation smoke (OQ-006 seed):

```
PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_intel_sim.py -q
```

## Presentation timing (≤ 5 min)

- 0:00–0:30  What Omni Q is
- 0:30–1:00  Why fixed AI workflows suck
- 1:00–3:30  Live demo behavior (Intel online entry)
- 3:30–4:20  Intel online integration details (+ any bonus Speechmatics/Qualcomm demo)
- 4:20–5:00  TableOps wedge, pilot metrics, and why this is different

## Screen layout

```
┌──────────────┬─────────────────────┬────────────────┐
│ CAMERA       │ OMNI EXECUTION      │ DEVICE STATE   │
│ structured   │ GRAPH               │ Intel arm A ✓  │
│ scene/frame  │ SEE → PLAN → ACT    │ Intel arm B ✓  │
│              │      → VERIFY       │ Speech (bonus) │
└──────────────┴─────────────────────┴────────────────┘

GOAL:           "Set the table."
CURRENT ACTION: PICK(plate) → ARM_A
WHY:            Required before PLACE(plate, table)
EXECUTING ON:   Dual SO-101 (MuJoCo simulation)
```
