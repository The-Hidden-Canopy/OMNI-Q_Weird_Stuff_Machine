# Omni Q — Hackathon Backlog

One backlog. **Intel online is the entry track — the only one we qualify
for** and the required deliverable. **Qualcomm** and **Intel onsite** are
exploratory/bonus work, not part of the entry; **Speechmatics** is a bonus
voice layer that stacks on the Intel online entry. Every task is atomic
enough to become an owner's `TASK.md`.

Owners: `Gerron/Claude`, `Gerron/GPT`, `Gerron/Kimi`, `Bryan/Codex`,
`Damion/Claude`. Per-owner filtered lists live in [`agents/`](agents/).

Status legend: ` ` todo · `~` in progress · `x` done.

## P0 — protect the Intel critical path

| ✓ | ID | Owner | Task | Depends on | Done when |
|---|----|-------|------|------------|-----------|
| x | OQ-001 | Gerron/Claude | Freeze Omni capability contracts: Observe, Plan, Manipulate, Verify, Device, Receipt | — | Interfaces compile and fake capabilities execute end-to-end — `src/omni_q/contracts.py` (7 `runtime_checkable` Protocols) + fakes; `python -m omni_q.demo` |
|   | OQ-002 | Gerron/GPT | Build agent-control repo scaffold: five agent dirs, task files, repo maps, coding pointers, active scopes | — | Every agent can bootstrap from repo state without chat history |
| x | OQ-003 | Gerron/Kimi | Inspect SO-101/MuJoCo implementation and write arm capability map | — | Joint names, limits, gripper range, workspace assumptions, wrist-roll limits, control API, cameras documented |
| x | OQ-004 | Damion/Claude | Independently audit Intel challenge requirements against intended design | — | Written PASS/GAP matrix for natural language, camera reasoning, two-arm coordination, multi-step table setting, Intel execution — [`docs/oq-004-requirements-audit.md`](docs/oq-004-requirements-audit.md): 1 PASS-mechanism (scheduling/handoff), 2 PARTIAL (NL, two-arm coordination), 2 GAP (camera reasoning, Intel execution/OpenVINO); cross-referenced against the 100-pt rubric — task completion (30) and OpenVINO (20) are the two biggest point risks, both currently blocked on the same grasp-reliability gap the team is actively closing |
|   | OQ-005 | Bryan/Codex | Build minimal live Omni graph UI shell | OQ-001 | UI can display goal, observations, both arms, graph nodes, current action, verification state |
| x | OQ-006 | Gerron/GPT | Stand up Intel MuJoCo dual-SO-101 environment | OQ-003 | Two simulated arms boot reliably and accept commanded joint/end-effector actions â€” pinned dual proxy loads (`nu=12`) and controller smoke passes; contact-rich grasping remains OQ-010 |
|   | OQ-007 | Gerron/GPT | Create table-setting scene/object pack | OQ-006 | Plates, cups, forks, spoons, napkins and target place settings exist with usable mass/friction/collision properties |
| ~ | OQ-008 | Gerron/GPT | Adapt your YOLO pipeline to tabletop objects | OQ-007 | Detector outputs class, bbox/center, confidence and stable object IDs from simulated camera frames — data + train + export pipeline built in `perception/` (7-class map, Open Images/COCO/LVIS pull, synth top-up, fine-tune from the thermal YOLOv8n, ONNX→OpenVINO/QAIRT); run + train still pending. Export→OpenVINO→benchmark half proven end-to-end using the already-published thermal YOLO as a stand-in: `evidence/benchmark_results/openvino_inference_2026-09-10/` (real Intel CPU+iGPU, FP32 + NNCF INT8) — swap in the real 7-class `.pt` once fine-tuned, same commands |
| ~ | OQ-009 | Gerron/Claude | Implement world-state representation | OQ-001, OQ-008 | Omni maintains objects, positions, orientations, ownership, goals and constraints across frames — `WorldState` has objects/ownership/constraints/goal/revision across frames; object **poses** pending real sim data (OQ-006/OQ-008) |
|   | OQ-010 | Gerron/GPT | Implement basic arm primitives | OQ-006 | PICK, PLACE, MOVE, OPEN, CLOSE, ROTATE, PRESENT execute individually |
|   | OQ-011 | Gerron/GPT | Implement bimanual primitives | OQ-010 | HANDOFF, STABILIZE, REGRASP, COOPERATIVE_ROTATE work between arms |
| x | OQ-012 | Gerron/Claude | Build bimanual task scheduler | OQ-009, OQ-010 | Scheduler considers reachability, occupied grippers, dependencies, workspace conflicts and parallelism — `src/omni_q/scheduler.py` + `ScheduledPlanner`, `docs/scheduler.md`, 20 tests |
| x | OQ-013 | Gerron/Claude | Implement collision/resource barriers | OQ-012 | Two arms cannot simultaneously enter unsafe/conflicting workspace without scheduling resolution — `Barrier(kind=workspace|gripper|verify|reach)`, wave-conflict invariant asserted |
|   | OQ-014 | Gerron/GPT | Build the spin primitive | OQ-011, OQ-013 | Plate can visibly rotate beyond one wrist's usable range through rotate → handoff → regrasp → rotate |
| x | OQ-015 | Gerron/Claude | Add style constraints to planning | OQ-012, OQ-014 | "set the table" generates efficient plan; "set the table and show off" adds safe flourishes without changing final goal — flourishes slot into slack / one pre-verify wave / else dropped; never a dep of verify |
|   | OQ-016 | Gerron/GPT | Execute basic table setting | OQ-007, OQ-012 | Two arms successfully place a minimal setting from initial scene |
|   | OQ-017 | Gerron/GPT | Execute concurrent table setting | OQ-016 | Both arms perform useful independent actions simultaneously rather than alternating |
| x | OQ-018 | Gerron/Claude | Closed-loop verification/replanning | OQ-008, OQ-009, OQ-016 | After every manipulation, vision compares observed vs expected state and retries/replans when necessary — `OmniQ` verifies after every manipulate step + replans on mismatch/lost-capability/constraint; real perception swaps in at OQ-008 |
| x | OQ-019 | Gerron/Kimi | Build table-layout evaluator | OQ-007 | Produces positional/orientation errors and PASS/FAIL for final setting |
|   | OQ-020 | Bryan/Codex | Visualize bimanual execution | OQ-012, OQ-005 | UI shows ARM-A/ARM-B actions, parallel intervals, handoffs, barriers and object ownership live |
| x | OQ-021 | Damion/Claude | Break the Intel demo deliberately | OQ-018 | Test moved objects, failed grasp, unreachable object, collision risk, missing detection, bad instruction; record behavior — see [`docs/oq-021-red-team-findings.md`](docs/oq-021-red-team-findings.md): 2 new latent findings (silent-false-positive perception on zero detections; scheduler "concurrency" is graph-level not real-time), 2 clean passes (unknown-zone, adversarial NL), 1 gap (mid-run perturbation untestable until grasp lands) |
|   | OQ-022 | Gerron/GPT | Intel hardware/runtime packaging | OQ-016 | Demo can run through required Intel execution path rather than generic local-only code |

## P1 — Speechmatics, Qualcomm, unification, submission

| ✓ | ID | Owner | Task | Depends on | Done when |
|---|----|-------|------|------------|-----------|
| x | OQ-023 | Gerron/Claude | Natural-language goal parser | OQ-001 | Instructions such as "plates centered, forks left, cups upper-right, show off" become explicit graph constraints — `src/omni_q/nlu.py`, canonical goal + `(kind,value)` constraints + `mutations`, 24 tests |
|   | OQ-024 | Gerron/GPT | Speechmatics realtime adapter | OQ-023 | Spoken instruction enters same goal parser and can modify an active task |
| x | OQ-025 | Gerron/Claude | Runtime constraint mutation | OQ-023, OQ-018 | "Don't touch the red cup" / "spin that plate" / "move the fork farther left" modifies current plan safely — `src/omni_q/mutation.py` `RuntimeMutator`; constraint-shaped changes queued + recompiled under the envelope, `spin`/`nudge` parsed + deferred to OQ-011/OQ-014; 10 tests |
|   | OQ-026 | Gerron/Kimi | Characterize flourish envelope | OQ-003, OQ-014 | Determine safe plate/cup rotation amounts, handoff poses, velocity limits and failure rates |
|   | OQ-027 | Gerron/GPT | Qualcomm HF model intake | — | Your Hugging Face model can be retrieved/exported into the Qualcomm-supported deployment path |
|   | OQ-028 | Gerron/GPT | Qualcomm X Elite inference node | OQ-027 | Real on-device inference returns structured detections/results |
|   | OQ-029 | Gerron/Kimi | Quantization/performance experiment for Qualcomm | OQ-028 | Record latency, memory and accuracy for viable deployment variants |
|   | OQ-030 | Gerron/GPT | Arduino UNO Q capability node | — | Omni can read at least one physical/simulated input and issue at least one output/action |
| ~ | OQ-031 | Gerron/Claude | Device-to-device task routing | OQ-028, OQ-030 | Omni assigns complementary work between X Elite and Arduino rather than merely sending a serial command — **Intel track done** (`src/omni_q/providers.py` `ProviderRouter`: perception/reasoning/each-arm on distinct devices, verify never shares with manipulation, 9 tests); Qualcomm track scaffolded, activates at OQ-028/OQ-030 |
|   | OQ-032 | Bryan/Codex | Qualcomm device graph UI | OQ-031 | UI visibly shows which node executes on Snapdragon vs Arduino and data moving between them |
|   | OQ-033 | Damion/Claude | Qualcomm rubric audit | OQ-031 | Verify model use, on-device AI, hardware integration and device-to-device collaboration are each demonstrable |
| ~ | OQ-034 | Gerron/Claude | Unified Omni provider abstraction | OQ-022, OQ-031 | Intel and Qualcomm appear as capabilities under the same Omni graph instead of separate demos — `Provider` + `ProviderRouter`; `set_available()` re-routes the SAME graph between tracks, mid-run track loss triggers replan onto the other; `INTEL_PROVIDER` live, `QUALCOMM_PROVIDER` `available=False` until OQ-028/OQ-030 |
|   | OQ-035 | Gerron/GPT | Unified demo launcher | OQ-034 | One command selects Intel simulation, Qualcomm hardware, or mock mode |
|   | OQ-036 | Bryan/Codex | Judge-facing demo mode | OQ-020, OQ-032 | One screen communicates goal → perception → plan → execution → verification without developer explanation |
| x | OQ-037 | Gerron/Kimi | Evidence/receipt collector | OQ-018, OQ-028 | Each run records inputs, model/version, task graph, actions, final metrics and hashes |
|   | OQ-038 | Damion/Claude | End-to-end acceptance suite | OQ-035, OQ-037 | Clean-machine or clean-environment reproduction passes documented demo cases |
|   | OQ-039 | Bryan/Codex | GitHub judge path | OQ-035 | README gets a short "run this" path, architecture image, demo GIF/video link and sponsor-tech mapping |
|   | OQ-040 | Gerron + Bryan | Product description | OQ-036 | ~1-paragraph description explains what it does, why it matters and what is original without jargon sludge |
|   | OQ-041 | Bryan/Codex | Five-minute presentation structure | OQ-036 | Script/demo sequence lands under 5:00 with room for failure margin |
|   | OQ-042 | Damion/Claude | Judge-rubric scorecard | OQ-039–041 | Every rubric item points to a concrete demo behavior/evidence artifact |

## P2 — polish, choreography, red-team

| ✓ | ID | Owner | Task | Depends on | Done when |
|---|----|-------|------|------------|-----------|
|   | OQ-043 | Gerron/GPT | Failure-injection demo control | OQ-018 | Button or scripted event moves an object / disables capability to visibly force replanning |
| x | OQ-044 | Gerron/Claude | Dynamic arm-role reassignment | OQ-012 | Omni chooses which arm picks/holds/spins based on current reach/state, not fixed left/right roles — scheduler picks the reaching arm with lower load; explicit `Step.arm` > `prefer_arm` > reach; HANDOFF splits a chain across arms |
|   | OQ-045 | Gerron/GPT | Fancy synchronized choreography | OQ-017, OQ-026 | Two useful actions overlap with deliberate visual timing while still reducing or preserving task completion time |
|   | OQ-046 | Gerron/Kimi | Run comparative trials | OQ-045 | Compare sequential vs bimanual vs bimanual+flourish for completion, errors, collisions and time |
|   | OQ-047 | Bryan/Codex | "Why Omni did that" display | OQ-018 | UI surfaces one-sentence reasoning/constraint explanation for the current graph transition |
|   | OQ-048 | Damion/Claude | Final red-team pass | All demo-critical tasks | Identify anything scripted, unsupported, unverifiable, flaky or confusing before submission |

## Hand vocabulary expansion — "hands, not clamps on sticks"

Richer manipulation set so the arms *do something once holding it*. Spec + planner/scheduler
= Gerron/Claude (`src/omni_q/actions.py`, `docs/actions.md`); sim primitives = Gerron/GPT.
Feeds the Speechmatics demo (OQ-024): `nlu` choreography phrases + `RuntimeMutator` are the
speech → semantic-state-mutation layer.

| ✓ | ID | Owner | Task | Depends on | Done when |
|---|----|-------|------|------------|-----------|
| x | OQ-HAND-001 | Gerron/Claude | Grip-state actions (PRE_SHAPE, PINCH/WIDE/EDGE_GRIP, SOFT/FIRM_HOLD, RELEASE, MICRO_RELEASE, REGRIP, SHIFT_GRIP) | OQ-001 | in the registry with metadata + pre/post |
| x | OQ-HAND-002 | Gerron/Claude | Contact-manipulation actions (NUDGE, PUSH, PULL, SLIDE, DRAG, SWEEP, PRESS, PIN, BRACE, STABILIZE, TAP, BUMP_ALIGN) | OQ-001 | in the registry; SLIDE is a no-grasp alt to PICK+PLACE |
| x | OQ-HAND-003 | Gerron/Claude | Orientation / flourish actions (ROTATE_IN_HAND, TWIRL, SPIN, ROLL, FLIP, TURN_HANDLE_TO, PRESENT, ORIENT_*, ALIGN_EDGE, NAPKIN_FLICK) | OQ-001 | in the registry, `style_action` flagged |
| x | OQ-HAND-004 | Gerron/Claude | Bimanual hand actions (HANDOFF, RECEIVE, PASS_THROUGH, CO_HOLD, CO_ROTATE, CO_ALIGN, CO_STABILIZE, ASSIST_GRASP, TRANSFER_LOAD, REGRASP_WITH_PARTNER, HOLD_WHILE_OTHER_ACTS) | OQ-HAND-001 | category `BIMANUAL` → scheduler gives them both arms in their wave |
| x | OQ-HAND-005 | Gerron/Claude | Action metadata + preconditions/effects | OQ-HAND-001..004 | `ActionSpec` (category, requires_contact/grasp, supports_bimanual, precision, style_action, blocking, preconditions, effects, args, composes); 84 ops, 17 tests |
| ~ | OQ-HAND-006 | Gerron/Claude | Planner/scheduler chooses SLIDE/NUDGE vs PICK/PLACE | OQ-HAND-005 | scheduler now reads style/bimanual/grasp/blocking off the registry; planner cost-based op choice still to do |
| ~ | OQ-HAND-007 | Gerron/GPT | Plate spin routine | OQ-HAND-004, OQ-014 | `SPIN_PLATE` / `SPIN_AND_PLACE` routines + `expand()` to primitives; MuJoCo primitive pending |
| ~ | OQ-HAND-008 | Gerron/GPT | Cup handle orientation | OQ-HAND-003 | `ROTATE_CUP_HANDLE` / `TURN_HANDLE_TO` in spec; sim primitive pending |
| ~ | OQ-HAND-009 | Gerron/GPT | Utensil alignment micro-actions | OQ-HAND-002 | `TWIRL_UTENSIL`, `ALIGN_PARALLEL`, `STRAIGHTEN`, `NUDGE` in spec; sim primitive pending |
| ~ | OQ-HAND-010 | Gerron/GPT | Napkin spread / fold-ish routine | OQ-HAND-003 | `NAPKIN_ROUTINE` / `SPREAD` / `NAPKIN_FLICK` in spec; sim primitive pending |
| ~ | OQ-HAND-011 | Gerron/Claude | Dance-while-working: speech style-mode → scheduler idle-slack flourishes | OQ-HAND-005, OQ-015 | spoken vocab ("fancy", "together", "opposite", "freeze", "back to work") → `style` mode; scheduler fills idle-arm slack with non-blocking IDLE_FLOURISH, never adding a wave or reordering; last STYLE wins; `minimum_time` strips flourishes. Sim primitives pending |

## Critical path (Intel)

Protect this sequence above everything else:

```
OQ-003 → OQ-006 → OQ-007 → OQ-010 → OQ-011 → OQ-012 → OQ-014
       → OQ-016 → OQ-017 → OQ-018 → OQ-020 → OQ-021 → OQ-022
```

From "learn the damn arms" to "two arms set a table concurrently, spin things
through real bimanual regrasping, and recover when the world changes."

## First milestone (not a model — a demo)

An ugly MuJoCo window where someone can command:

```
LEFT PICK plate
LEFT ROTATE +140
RIGHT RECEIVE plate
LEFT RELEASE
RIGHT ROTATE +140
LEFT REGRASP
PLACE target_1
```

Once that works, Omni owns it as `SPIN_AND_PLACE()`. Everything above that line
becomes intelligence rather than robot debugging.

## Standing notes

- **Kimi:** do **not** spend the morning on the 400M corpus. Immediate
  highest-value work is arm characterization (OQ-003), evaluator (OQ-019), and
  Qualcomm profiling/data (OQ-029/037). The pretraining corpus keeps moving but
  must not block the demo.
- **Deliverables are only:** product description, ≤ 5-min presentation, GitHub
  repo with the demo. The repo is the proof surface, not the pitch deck.
- **Briefs may shift at kickoff** (2026-09-10 15:00 UTC). The capability-node
  abstraction (OQ-001) is safe to build now; it holds for all three sponsors.
- **Reuse over greenfield.** `docs/prior-art.md` maps concrete subsystems from
  `SOCOM_REACT` (rolling-horizon replanning, signed authority envelope,
  degradation modes, decision-reason object), `Open-World-Model-Harness` (world
  boundary, causal event log, honest knowledge status), `FALCON-DARPA`
  (parent-chained evidence packages), and `VIGIL` (receipt canonicalization +
  fail-closed pipeline) onto specific OQ tasks. Check it before building
  OQ-009/012/013/015/018/023/025/031/034/037.
