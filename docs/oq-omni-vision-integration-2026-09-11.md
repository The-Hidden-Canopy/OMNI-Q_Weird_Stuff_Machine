# Real vision + reasoning-driven planning, composed for the first time

> Damion/Claude, self-directed, 2026-09-11. Prompted directly by the
> team's stated direction: the project sets the full table, and the Omni
> model + YOLO vision are the real control path for the arms. Before this
> pass, real vision (`FrameObserver` + `OpenVINODetector`, see
> `src/omni_q/vision.py`/`frame_observer.py`) and reasoning-driven
> planning (`OmniPlanner`, see `src/omni_q/omni_planner.py`) had each been
> proven independently, but never run together on the same engine. This
> composes them for the first time, finds three real integration bugs
> that would have blocked it, fixes them, and gets a genuine real
> PICK+MOVE success end to end.

## What was composed

```python
engine = build_intel_sim_engine()

# Real vision: swap ground truth for a real render -> real OpenVINO
# detection -> real camera-geometry zone mapping.
cam = MuJoCoCameraSource(world.model, world.data, "table_overhead", ...)
detector = OpenVINODetector(model_xml, device="CPU")
engine.observer = FrameObserver(
    as_frame_detector(detector, (640, 480)),
    zone_map=make_camera_zone_map(cam, zones, (640, 480)),
    frame_source=lambda w: cam.capture(),
)

# Reasoning-driven planning, with the governed deterministic planner as
# fallback -- the same composition documented in demo_intel_reasoner.py.
engine.planner = OmniPlanner(reasoner, fallback=ScheduledPlanner(IntelTablePlanner()))

receipt = engine.run("set the table")
```

The model used here is `MockReasoner` with a scripted `PLAN...END`
response, not the real IDA Omni reference body -- that checkpoint is
still pretraining (per the backlog, "day 6"). This is honestly a test of
the **wiring**, not of model quality: does real perception's output
correctly reach a plan, does that plan correctly reach real physical
execution, and does the governed fallback correctly take over when the
model's advice runs out. A better-trained model changes what gets
*proposed*; it doesn't change whether the pipeline in between is sound,
which is what this actually tests.

## Three real bugs found, all from actually running it, not inspection

**1. Stale re-PICK of an already-held object.** The world layer already
refuses this correctly (`TransitionRejected: "<oid> is already held"`),
but nothing before `OmniPlanner._validate` caught it, so a model that
proposes the same completed PICK again (its context hasn't yet confirmed
the last attempt succeeded) burns a full revision on a proposal that can
never succeed -- measured 6 of 7 revisions spent this way in one run.
Fixed: rejected at validation time, mirroring `RulePlanner`'s own existing
pattern ("carry straight from the gripper instead" of re-issuing a PICK
it doesn't need -- see `fakes.py`).

**2. MOVE routed to the wrong arm.** A MOVE with no `arm=` (or a guessed
one) for an object actually held by the *other* arm was rejected every
time (`"held by intel.right_arm, not intel.left_arm"`). There's no
ambiguity to resolve -- only the holding arm can act on a held object --
so this is now corrected to the true holder rather than merely rejected,
the same way a human operator would just fix an obviously-wrong arm label
instead of stopping to argue about it.

**3. Redundant PICK/MOVE on an already-placed object.** `RulePlanner`
never has this problem -- it generates candidates fresh from
`world.misplaced()` every call, so a done object simply never reappears.
A model carries its own belief of the goal across turns instead, so once
an object is genuinely placed, a model that hasn't caught up yet may
propose PICK/MOVE for it again. Ownership has already been released by
then, so nothing else catches it -- unlike bug #1, this reached a **real**
grasp/carry attempt on a finished object, burning real physics time to
fail (or, worse, risking disturbing a correctly-placed object). Measured
directly: after `cup_1` was genuinely picked and placed, the next
replan's static proposal re-picked it anyway.

All three: found by composing the actual pieces and reading what
happened, not by reasoning about the code in the abstract. Regression
tests for each are in `tests/test_omni_planner.py`.

## The result, after fixing all three

```
omni_pick_cup_1_0   PICK  done    held=True
omni_move_cup_1_0   MOVE  done    placed=True
                    -- cup_1 genuinely picked and placed, real vision,
                       real IK, real MuJoCo physics throughout --
[static mock proposal re-submitted; correctly rejected: "already at its
 target zone" -- clean, labeled fallback to the deterministic planner]
open_drawer         OPEN  done
pick_napkin_1       PICK  failed  (still the known, documented grasp gap)
...
```

Full receipt: `evidence/benchmark_results/omni_vision_integration_2026-09-11/receipt.json`.

`resolved: False` -- this is not a new claim that table-setting works.
`cup_1` was already known to be the one reliably graspable object (see
`intel_sim.py`'s module docstring, "Seventh update": 10/10 across the
randomized harness). What's new here is that the **path to that success
now runs through real perception and a reasoning-shaped planning
interface**, not just ground-truth state and a hand-coded rule planner --
and that path is now verified sound, with the specific defects that would
have silently wasted a real model's turns found and fixed before a real
model exists to hit them.

## What this doesn't prove

- **Not the real model.** `MockReasoner`'s canned response is static and
  scripted by hand for this test -- it proves the interface a real model
  will be held to, not anything about model quality, prompting, or
  whether a trained IDA Omni body will reliably produce a valid `PLAN`
  for this scene. That's still ahead, gated on the model finishing
  pretraining.
- **Not the real YOLO classes.** `OpenVINODetector` here runs the
  already-published thermal YOLO (`KissTheHabit/yolov8n-hituav-thermal-
  finetune`) as a stand-in, same caveat as the earlier OpenVINO benchmark
  and vision-pipeline work -- placeholder class labels until OQ-008's
  7-class table-setting fine-tune lands. The zone-mapping/projection math
  is real regardless of which classes the boxes carry.
- **Not full table-setting.** The other four objects still don't hold
  reliably (see `intel_sim.py`'s "Fifth"/"Sixth" updates for why, and why
  further work there hit real diminishing returns this session).
- **A real, separate integration risk this surfaced**: the zone
  vocabulary a vision system's `zone_map` reports (this test's own
  first draft used `vision.py`'s perception-facing labels like
  `"tray_cup"`) must exactly match what `ZONE_POSITIONS` (the physical
  MOVE-target vocabulary `_do_place` actually uses) accepts, or a
  structurally-valid-looking proposal will still fail at physical
  execution every time. Not fixed here (it was a test-fixture mismatch,
  not a code bug), but worth remembering when wiring a real
  `FrameObserver` deployment's zone_map: use `ZONE_POSITIONS`' own
  vocabulary, not an independently-invented one.
