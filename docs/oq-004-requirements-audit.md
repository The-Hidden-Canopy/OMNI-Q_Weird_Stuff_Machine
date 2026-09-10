# OQ-004 — Intel challenge requirements audit

> Damion/Claude, P0, no dependencies. Independent PASS/GAP matrix against the
> **published brief**
> ([`docs/challenge-briefs/intel-online-physical-ai-challenge.md`](challenge-briefs/intel-online-physical-ai-challenge.md)),
> not the team's description of it. Dated 2026-09-10, against the codebase at
> commit `24e3661` (main). Some findings below come from my own hands-on IK/
> grasp implementation work this session, not just static code reading —
> flagged where that applies. Re-run before submission; several gaps here are
> actively being closed by the team in parallel.
>
> **Addendum, same day, commit `ea7818b`→later:** acted on this audit's own
> priority #3 — exported the published thermal YOLO to OpenVINO IR and
> benchmarked it (FP32 + NNCF INT8) on this machine's real Intel CPU+iGPU:
> [`evidence/benchmark_results/openvino_inference_2026-09-10/`](../evidence/benchmark_results/openvino_inference_2026-09-10/README.md).
> This moves "Intel execution" from a flat GAP to partial — the
> export/optimize/benchmark pipeline is now proven end-to-end on real
> hardware, though not yet on the fine-tuned table-setting detector (OQ-008
> still pending) or on actual Core Ultra Series 2/3 silicon. The verdict
> tables below are left as originally written (point-in-time record); read
> them alongside this note, not as superseded.
>
> **Second addendum, same day, commit `19d9ce7`→later:** the "camera
> reasoning" flat GAP is also now a demonstrated pipeline, from two
> independently-built pieces that landed the same session and composed
> cleanly: a real MuJoCo-render → OpenVINO-inference → camera-geometry
> back-projection stack (`src/omni_q/vision.py`, this work) wired as the
> real detector + zone map for `src/omni_q/frame_observer.py`'s
> `FrameObserver` (built independently by someone else acting on this same
> audit finding — stable-id tracking, full `Observe`-contract compliance).
> End to end: a real detection lands in the zone matching its actual object's
> real position. See `integrations/intel/README.md` for the full writeup.
> Same caveat as the OpenVINO addendum above: real pipeline, placeholder
> class labels until OQ-008's fine-tune lands, and `FrameObserver` isn't
> wired into `build_intel_sim_engine()`'s default engine yet (still opt-in).

> **Third addendum, 2026-09-10 working-tree follow-up:** the general legacy
> table-setting controller now includes an object-aware 6D fingertip-pad solve,
> named pad-only contact masks, calibrated `cup_1` physics, and snapshot-based
> retry rollback.  This improves the physical primitive but does **not** close
> the multi-step table-setting GAP: current local runs still expose
> shared-workspace/order sensitivity plus unresolved cutlery or placement
> failures.  The separate `simulation-contact-handoff` path remains the only
> deterministic 10/10 contact evidence.  The point-in-time matrix below is
> intentionally retained; re-run the legacy randomized report after the
> controller/scene changes before changing its verdict.  A fresh 10-seed
> report is retained at
> [`evidence/benchmark_results/intel_table_eval_2026-09-10-v4/report.json`](../evidence/benchmark_results/intel_table_eval_2026-09-10-v4/report.json):
> 0/10 complete, with 2 grasp failures and 8 placement failures.
>
> **Third addendum, same day:** OQ-021's red-team pass
> ([`docs/oq-021-red-team-findings.md`](oq-021-red-team-findings.md)) found
> two more real issues (a latent perception-failure/success confusion in
> `FrameObserver`, and the scheduler's "concurrent" waves being
> planning-graph-only, not real-time-simultaneous) — see that doc. Separately,
> follow-up on this audit's own grasp-generalization priority (#1) falsified
> the "single wrist-roll scalar" hypothesis and found+fixed a genuine scene
> bug: `fork_1`/`spoon_1` sat ~65% past the arm's own independently measured
> reach envelope, confirmed two ways. Fixing that was necessary but not
> sufficient — the 10-seed harness is still 10/10 grasp failure after the fix
> (`evidence/benchmark_results/intel_table_eval_2026-09-10-v3/`); the real
> remaining blocker is the pad-tracking controller's own position precision
> (~0.05m best case vs ~0.01-0.02m needed), not reach or orientation search
> range. See `integrations/intel/README.md` and `src/omni_q/intel_sim.py`'s
> module docstring for the full writeup. Task completion (30 pts) is still
> the single biggest point risk on the board.
>
> **Fourth addendum, same day:** a teammate independently landed real
> orientation-aware IK (`_grasp_frame`/`_ik_reach_pad_pose`, jointly solving
> position and orientation, verified against actual close+lift rather than
> position error alone) plus a `cup_1` resized to fit the gripper's real
> measured envelope. Result, honestly measured: `world._do_pick(6,"cup_1")`
> now returns a genuine held grasp — the first reliable success this whole
> investigation has produced, from real control + geometry work, not from
> weakened physics. **The same merge also introduced a real problem**,
> caught before it reached a demo or rubric claim: `contype`/`conaffinity`
> set to 0/0 on every non-fingertip-pad arm geom, which under MuJoCo's
> collision rule makes the whole arm mesh except the two pads unable to
> collide with the table, the drawer, or any object — a no-clip exemption,
> not a control improvement, and exactly the category of change the
> no-simulation-cheating rule was established to rule out earlier this
> session. Flagged to the user before touching it, given explicit direction
> to fix it while preserving real-world realism, reverted entirely. Verified
> the real grasp still holds with full collision restored (it does — the
> orientation-aware control + correct geometry was always doing the real
> work). Cost, accepted not hidden: full suite runtime ~95s → ~6min once
> physics is honest. `plate_1`/`fork_1`/`spoon_1`/`napkin_1` still don't
> hold. See `integrations/intel/README.md` and `src/omni_q/intel_sim.py`'s
> module docstring for the full writeup. Task completion (30 pts) risk has
> shifted from "grasping doesn't work at all" to "grasping works for one
> object class, real physics confirmed, four to go."

## Method

Read the five brief-named areas literally off the PDF text, then checked each
against what the running code actually does — not the docstrings, not the
TODO checkmarks. Cross-referenced against the 100-point rubric separately,
since "PASS on the requirement" and "earns the rubric points" aren't always
the same claim.

## PASS/GAP — the five brief-named areas

| Area | Verdict | Evidence |
| --- | --- | --- |
| **Natural-language instructions** | **PARTIAL** | Two real, tested layers exist: `omni_q.nlu` (OQ-023, 24 tests) parses spoken/typed utterances into constraints (`forbid_object`, `keep_local`, `prefer_arm`, `style`), and `omni_q.mutation` (OQ-025, 10 tests) applies them live mid-run. Separately, `omni_q.omni_planner.OmniPlanner` (new this session's pull) lets a reasoner — mock or a real model backend — propose a step-by-step plan from an NL goal + structured world state, governed (every step re-validated before it reaches the graph). **Gap**: none of this has been run against the brief's own example instruction — *"Open the top drawer, pick up the plate with arm A, place it on the table, pick up the mug with arm B, pour water into the mug with arm A."* `POUR` does not exist in `omni_q.actions`' 84-op registry, `omni_q.nlu`, or `omni_q.omni_planner` (checked directly, zero matches). Arm-specific addressing ("with arm A"/"with arm B") is not a parsed pattern in `nlu.py` either — arm assignment today comes from the scheduler's reach/load heuristic, not from the instruction text. |
| **Camera reasoning** | **GAP** | `IntelTableObserver` (the only observer wired into `build_intel_sim_engine`) is explicitly `"Simulation-grounded observation reference; it is never labelled camera live"` — it reads ground-truth `WorldState` objects, not rendered camera pixels through a detector. A real perception pipeline exists (`perception/` — dataset build, fine-tune from the thermal YOLO, ONNX/OpenVINO export) but is **not wired into the observe loop**; OQ-008 is `~` in progress, "run + train pending" per its own backlog row. As of today, the system does not reason over camera observations at all — it reasons over privileged simulator state. |
| **Two-arm coordination** | **PARTIAL** | The scheduling mechanism is real and tested: `omni_q.scheduler` (OQ-012/013, 20+ tests) computes genuine concurrent waves — measured on `set the table`, both arms work the same wave (`max_parallelism` 1→2 after the OQ-007 zone fix), with real workspace-conflict barriers, not just distinct arm labels. A real bimanual **handoff** is proven end-to-end: `_ContactHandoffController` moves a cup from the left arm to the right arm via genuine MuJoCo contact dynamics, 10/10 across its deterministic + randomized trials. **Gap**: that handoff is one hand-tuned choreography for one object (`cup_1`); the general `IntelTablePlanner` path that's supposed to coordinate both arms across the full table-setting task doesn't yet complete a real grasp for *any* of the five tracked objects (see Multi-step table setting below), so "two-arm coordination" is currently demonstrated on the scheduling/handoff mechanism, not on the actual scenario. |
| **Multi-step table setting** | **GAP (currently)** | `IntelTablePlanner` generates the right *shape* of plan — `OPEN(drawer) → PICK/MOVE` per misplaced object → `VERIFY`, matching the brief's drawer-then-retrieve-then-place sequence — and every step genuinely steps real MuJoCo physics (no mock, no skip). But the grasp itself is 3-DOF position-only IK with no orientation control, and the retained evidence is blunt: **`evidence/benchmark_results/intel_table_eval_2026-09-10-v2/README.md` — 10/10 seeds, 10/10 grasp failure, 0 successes.** This is the team's own honestly-labeled evidence, not my inference. I did the IK work that produced this result this session, so I can say precisely why: position converges to ~1cm but wrist orientation is whatever the redundant solver's null-space settles into, not aimed at the object — the jaws frequently aren't angled to actually pinch what they're reaching for. A proven fix pattern exists (`_ContactHandoffController`'s fixed wrist-roll per arm, 10/10 for the cup) but hasn't been generalized to the other four objects or the main planner path yet. |
| **Intel execution** | **GAP** | `pyproject.toml`'s `intel` extra installs `openvino`, `optimum-intel[openvino]`, and `nncf`; this session independently verified a real Intel CPU + iGPU are visible to OpenVINO on the dev machine (`Core().available_devices` → `CPU`, `GPU.0` Intel iGPU, `GPU.1` NVIDIA dGPU). That is dependency and hardware-visibility readiness, not execution: **no model — perception or policy — has actually been exported to OpenVINO IR or run through OpenVINO inference anywhere in the codebase yet.** The arms are driven by hand-written IK/scheduling logic, not a trained VLA/policy, so there is currently nothing for OpenVINO to optimize even once export tooling is wired up. `perception/export.py` exists (ONNX → OpenVINO/QAIRT) but per OQ-008's own status, hasn't been run. |

## Cross-reference against the 100-point rubric

| Rubric item | Pts | Status | Why |
| --- | ---: | --- | --- |
| End-to-End Task Completion & Bimanual Manipulation | 30 | **At risk** | Sequencing/hand-off/coordination *mechanisms* are real and tested; end-to-end task success is currently 0/10 on the retained evidence. This is the single largest line item and the current biggest point risk. |
| VLA / Multi-Modal Reasoning | 20 | **Partial / at risk** | Governed NL→plan infrastructure exists (`OmniPlanner`) but untested against the brief's own instruction complexity (arm-addressed multi-clause commands, `POUR`); "reasons over visual observations" fails outright today since observation is ground-truth, not camera-derived. |
| Robustness & Generalization (10 seeds) | 15 | **Harness PASS, outcome FAIL** | The 10-randomized-seed harness itself is real, correct, and exactly matches what's asked (`IntelSceneConfig` + `run_intel_table_evaluation_report`, hash-checked receipts, honestly labeled "not a promotion claim"). But it currently measures a 0% success rate, so the evaluation *infrastructure* earns credibility, the *result* does not yet earn robustness points. |
| OpenVINO & Intel Core Ultra Optimization | 20 | **Largely unclaimed** | Toolchain installed, hardware confirmed visible; no model exported or benchmarked through OpenVINO yet. Second-largest point risk after task completion, and currently has the least implementation progress of any rubric category. |
| Technical Quality & Reproducibility | 10 | **Strong** | 195 passing tests, hash-verified parent-chained receipts, clear extras-based install, honest documentation discipline throughout (every doc I read this session labels its own limitations rather than overclaiming — genuinely unusual and worth preserving as a habit under submission pressure). |
| Innovation & Technical Demonstration | 5 | **Likely PASS, needs staging** | The capability-graph/replan-on-failure architecture, 84-op action vocabulary, style/choreography modes, and dynamic provider routing are genuinely distinctive relative to a scripted demo. Needs to be *visible in the final demo sequence*, not just present in code, to actually earn the points. |

## What this means, plainly

The **scaffolding is ahead of the outcome**. Contracts, scheduling, governance,
evidence/receipts, evaluation harness, action vocabulary, and NL infrastructure
are all real, tested, and honestly documented — that's the strongest part of
this submission and where the "Technical Quality & Reproducibility" points are
essentially secured already. But the two rubric categories worth the most
(task completion 30 + OpenVINO optimization 20 = half the total score) both
currently point at work that hasn't landed: a reliable grasp, and any model
actually running through OpenVINO. Robustness (15 pts) is architecturally
ready and will move the moment task completion does, since the same harness
already runs it.

## Priority recommendation (my read, not a directive)

1. **Generalize the proven grasp fix.** `_ContactHandoffController`'s fixed
   wrist-roll pattern already works 10/10 for one object under one
   choreography. Extending it — or something like it — to `_do_pick`/
   `_do_place` for the other four objects is the highest-leverage single
   change: it unblocks task completion (30 pts) *and* robustness (15 pts)
   from the same harness that's already built and already runs 10 seeds.
2. **Wire perception into the observe loop**, even narrowly. `IntelTableObserver`
   swapping to a real render→detect path (however imperfect at first) is
   what turns "camera reasoning" from a documented gap into a demonstrable
   one, and unlocks real "VLA / Multi-Modal Reasoning" scoring.
3. **Export one model through OpenVINO and benchmark it**, even a small,
   narrow one (the perception detector is the most exportable candidate
   today, via `perception/export.py`). Partial credit on a 20-point item beats
   zero, and it's currently the least-started rubric category.
4. **Add `POUR` and arm-addressed instruction parsing**, or deliberately
   narrow the demo script to instructions the system actually supports and
   say so in the technical README — either resolves the gap; silently hoping
   judges don't test the brief's own example sentence does not.
5. Re-run this audit (or hand it to OQ-021/042/048, my own later tasks) once
   (1)-(3) land — the verdicts above will likely flip fast, and a stale audit
   is worse than an absent one.
