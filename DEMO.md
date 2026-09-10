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

Bonus, not part of the entry: a Speechmatics voice layer can supply the goal
and live constraints by speech instead of text (see
[`integrations/speechmatics/README.md`](integrations/speechmatics/README.md)).

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

The judge-facing dashboard is deliberately labelled **MOCK MODE — NOT
HARDWARE**. It renders the causal event stream supplied by the active session:

- mission objective and structured workspace observation
- Observe → Plan → Act → Verify phase state
- compiled graph steps, graph revisions, and arm/device placement
- authorization decisions, constraints, replans, and capability failures
- verification results, autonomy mode, final metrics, and receipt hash

The **Apply live constraint** control exercises the same session-scoped path as
Speechmatics: the operator supplies a justified `keep_local`, `forbid_object`,
or `prefer_arm` constraint, the engine queues it, and the graph recompiles on
the next loop iteration. The UI only renders events from the active session; it
does not call hardware directly. See [`docs/ui.md`](docs/ui.md) for the full
screen map and scope boundary.

Intel simulation smoke (OQ-006 seed):

```
PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_intel_sim.py -q
```

This is a real MuJoCo dual-arm/controller step using the pinned SO-ARM100
mechanical proxy, but tableware placement remains an explicitly labelled
scripted transition. _TODO: camera perception (OQ-008), contact-rich grasping,
screen recording, VLA/OpenVINO, and hardware evidence._

## Presentation timing (≤ 5 min)

- 0:00–0:30  What Omni Q is
- 0:30–1:00  Why fixed AI workflows suck
- 1:00–3:30  Live demo behavior (Intel online entry)
- 3:30–4:20  Intel online integration details (+ any bonus Speechmatics/Qualcomm demo)
- 4:20–5:00  Business value + why this is different

## Screen layout

```
┌──────────────┬─────────────────────┬────────────────┐
│ CAMERA       │ OMNI EXECUTION      │ DEVICE STATE   │
│ live scene   │ GRAPH               │ Intel arm A ✓  │
│              │ SEE → PLAN → ACT    │ Intel arm B ✓  │
│              │      → VERIFY       │ Speech (bonus) │
└──────────────┴─────────────────────┴────────────────┘

GOAL:           "Set the table."
CURRENT ACTION: PICK(plate) → ARM_A
WHY:            Required before PLACE(plate, table)
EXECUTING ON:   Dual SO-101 (MuJoCo)
```
