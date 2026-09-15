# Submission readiness — Intel Online Physical AI Challenge (+ Speechmatics bonus), 2026-09-15

Written overnight 2026-09-14 → 15 against the host brief
([`challenge-briefs/intel-online-physical-ai-challenge.md`](challenge-briefs/intel-online-physical-ai-challenge.md))
and the hosts' later note that **VLA must be the dominant robot-control
policy** (IK alone "not preferable"; IK may be combined with VLA "provided VLA
is the dominant policy"). Everything below is either measured in this repo or
marked as a gap. Numbers are from `evidence/` receipts, not from memory.

## Verdict in one paragraph

The dinner-table task itself is in good shape: both SO-101 arms set the
table under real contact physics, both arms work at the same time, hand-off
and complementary actions exist, the perception loop is closed through four
cameras with a fine-tuned YOLOv8n running under OpenVINO, mid-run voice
authority changes and a physical arm failure are handled by re-planning, and
the 10-seed randomized evaluation is **9/10 resolved, 49/50 placements, 9/10
trials with every object physically in its zone and upright at the end**
under placement / yaw / mass / friction / colour / lighting / background /
prompt variation. The demo video set (25 clips + a 10-seed montage) is
recorded from exactly that build. **The material risk is the hosts' VLA
note**: as of the recorded set, motion is produced by scripted IK + contact
grasp primitives sequenced by a governed rule planner; the IDA Omni reasoner
path exists but produced no valid step advice in a test run (100% fallback),
and the teammate's SmolVLA seam has no trained checkpoint behind it yet. A
VLA-first path (SmolVLA fine-tuned on this stack's own demonstrations,
leading every single-arm PICK/MOVE, IK only realising its Cartesian deltas,
governed primitive completing each step) was built and measured overnight —
8/10 seeds in VLA mode; see the "VLA" section for exactly what the policy
does and does not do yet.

## Deliverables (brief §"Required deliverables")

| # | Deliverable | Status | Where |
|---|---|---|---|
| 1 | Reproducible repo: setup, deps, scene, training/fine-tune code, eval code, inference code, demo commands | Mostly ready. Setup + extras in `pyproject.toml` (`intel`, `smolvla`, `speechmatics`); scene in `src/omni_q/intel_sim.py`; eval harness `run_intel_table_evaluation_report`; perception training `integrations/intel/scripts/train_table_yolo.py`; VLA demo recording + fine-tune + eval in `integrations/intel/vla/`; demo commands in `DEMO.md` and `integrations/intel/README.md` | README needs a final "reproduce the video" section pointing at `record_demo_videos.py` / `record_seed_montage.py` |
| 2 | Reproducible MuJoCo sim incl. randomization + evaluation config | Ready. `IntelSceneConfig` (seeded: placement ±12 mm, yaw ±11°, mass ±25 %, friction ±15 %, colour, light angle/intensity, floor tone; prompt phrasing per seed in the harness); variation embedded in the compiled model (`omniq_variation` text) | `run_intel_table_evaluation_report(..., trials=10, seed=900)` |
| 3 | Intel inference benchmark script (Core Ultra 2/3; latency, throughput, device, precision) | **Partial.** `integrations/intel/scripts/profile_yolo_openvino.py` benchmarks the detector under OpenVINO on CPU + iGPU (FP32 and NNCF INT8) with the same protocol as the ONNX baseline — but it was run on a Core 5 210H, **not a Core Ultra Series 2/3**, and it profiles the earlier thermal YOLO; it must be re-run on the target part with the table detector (`models/table_yolo_v4_hfbase_2026-09-14/best.xml`) and, if the VLA lands, with the SmolVLA policy | `evidence/benchmark_results/openvino_inference_2026-09-10/` |
| 4 | Demo video: 10 randomized seeds; command, variation, outcome easy to verify | Ready (local, not in git): `Desktop/OMNI-Q_demo_videos/` — 01 full runs (single cam, director, grid, six cameras), 02 second seed (overhead, wrists), 03 perception e2e incl. four YOLO-overlay takes, 04 two-arm plate, 05 hand-off, 06 handshake, 07 voice authority change, 08 arm failure, 09 ten-seed montage (title card prints the seed's variation factors; result card prints the end-of-run physical check). README.txt in the folder | To edit into the final video |
| 5 | Technical README / architecture summary (architecture, VLA/VLM choice, bimanual strategy, training, robustness, OpenVINO, Intel mapping) | **Needs a pass.** README.md has the architecture, perception, voice, reasoner and skill-runtime sections; it must state plainly what controls motion in the recorded runs and what the VLA path is (see below), and the Intel hardware mapping (CPU/iGPU/NPU per component) | `README.md` |

## Rubric (100 pts)

| Criterion | Pts | Evidence | Gap / note |
|---|---|---|---|
| End-to-end task + bimanual (sequencing, hand-off, coordination, accuracy, success) | 30 | 10-seed harness `evidence/benchmark_results/parallel_arms_2026-09-14/harness_seed900_x10_final_layout/` (9/10 resolved, 49/50, 9/10 final-state); both arms simultaneous (`apply_transitions_parallel`); two-arm plate carry; cup hand-off `evidence/.../handoff_variation_2026-09-14_recovery` (8/20 honest); placement errors 1–4 cm | Hand-off success (8/20) is the weakest number; it is reported honestly |
| VLA / multi-modal reasoning (language + vision, multi-step context, action selection, adapts plan) | 20 | NLU → constraints (voice authority change), perception-driven planning + verification (`camera_e2e_2026-09-14`), re-planning on failure and on arm fault (`arm_failure_2026-09-14`); the behaviours are all demonstrated | **The model in the loop is the gap.** The IDA Omni reasoner (`OmniPlanner` over `omni_planner_r1_best.pt`) returned "no valid steps" for every decision in a test run (deterministic fallback did all planning, and the scheduled wrapper regressed the plate pick to 1/5). Recorded runs used the governed rule planner. See VLA section |
| Robustness & generalization (placement/weight/friction/shape/lighting/background; 10 seeds) | 15 | Harness above; per-seed factors printed on the montage cards | Shape variation is not randomized (one model per object) |
| OpenVINO & Core Ultra optimization | 20 | Detector exported to IR, FP32 + INT8, CPU/iGPU benchmark; detector runs under OpenVINO in the perception e2e run | Not yet measured on Core Ultra 2/3; NPU not exercised; policy (VLA) not yet under OpenVINO |
| Technical quality & reproducibility | 10 | Deterministic seeds, receipts with content hashes, 34 tests on the sim, harness + montage tooling | — |
| Innovation & demo | 5 | Governed re-planning under voice authority and physical faults; end-of-run physical verification; fleet showcase clips (labelled) | — |

## What actually controls the arms today (be precise in the README/video)

* **Task level:** `IntelTablePlanner` (governed rule planner) sequences PICK/MOVE
  steps from perception/state, honours operator constraints (`prefer_arm`),
  re-plans on failures and faults. The IDA Omni reasoner can advise it
  (`OmniPlanner`) but did not produce valid advice in the 2026-09-15 test.
* **Motion level (recorded set):** scripted primitives — FK-scan-seeded IK,
  contact-driven grasps, wall/rim pinches, lockstep two-arm carry — all under
  real contact physics with verifiers. This is the "IK dominant" pattern the
  hosts flagged.
* **Speech:** Speechmatics realtime/voice transport → NLU → constraints; the
  authority-change clip is driven through `nlu.parse("don't use the left arm
  anymore")`; live latency unmeasured (README says so).

## VLA path (built 2026-09-15, additive; see `integrations/intel/vla/`)

Design (matches the hosts' "combine IK with VLA, VLA dominant"):

```text
overhead + front + wrist cameras, arm state, per-step instruction
    → SmolVLA (lerobot/smolvla_base fine-tuned on this stack's own demonstrations)
    → 7 Cartesian deltas per 10 Hz tick   (the teammate's governed seam: src/omni_q/skills/controllers/smolvla.py)
    → workspace-safety check → pad-pose IK realises the delta (joint targets only)
    → physics; the world's own grasp / placement verifiers decide
    → on budget exhaustion: rewind that arm, governed primitive finishes — counted in the receipt as `vla_attempt` + `fallback`
```

* Demonstrations: `record_expert_demos.py` — 20 seeds × 2 arms, 10 Hz, three
  320×240 cameras, 9-d state, 7-d Cartesian-delta action, per-step language
  (LeRobot v3 dataset, `datasets/so101_table_vla`, local).
* Fine-tune: `lerobot-train --policy.path=lerobot/smolvla_base` on the RTX 4050
  (see `tmp/vla_train.sh` / `integrations/intel/vla/README.md`).
* Evaluation: `run_vla_eval.py` — per op, VLA-completed vs fallback, task
  outcome, end-of-run physical check; `--no-fallback` for the pure-VLA number.

**Status / numbers (measured 2026-09-15, morning):**

* Dataset: 40 episodes / 65,850 frames from 20 randomized seeds (`datasets/so101_table_vla_absjaw`:
  7th action = absolute jaw command; physics-rollback teleports zeroed).
* Fine-tune: `lerobot/smolvla_base` action expert, 8000 steps, batch 8, RTX 4050,
  ~78 min (`outputs/train/smolvla_so101_table_20260915_0419/checkpoints/008000`).
* What the policy does on its own: reaches the cup and closes the jaw above it,
  carries and released the napkin once in the zone (1 step completed solo in
  a diagnostic run); it cannot localise the thin cutlery (dx≈0 for fork/spoon).
* VLA-first 10-seed harness (`evidence/benchmark_results/vla_smolvla_2026-09-15/harness_seed900_x10/summary.json`):
  **8/10 resolved, 48/50 placements, 8/10 fully set at the end.** Every
  single-arm PICK/MOVE (124 steps) was VLA-led — the policy drove the reach
  and the start of the carry from the cameras for its budget (8 s / 6 s) —
  and **all 124 were completed by the governed primitive continuing from
  where the policy left the arm** (0 completed by the policy alone). The
  grasp-integrity guard (object shifting in the pinch) is what hands carries
  over early; without it the spoon was dropped.
* So the honest claim is: *VLA leads the motion of every single-arm step; the
  governed contact primitive completes it; the plate is a governed two-arm
  primitive.* By time, roughly half of each single-arm step's motion is
  policy-driven. That is "VLA + IK combined" but **not yet VLA-dominant in
  the sense of the policy finishing grasps** — say exactly this in the README
  and video; do not claim more.
* Paired (both-arms-at-once) execution under the VLA loops does not work yet
  (1/5 in a test); VLA-mode clips execute one arm-step at a time.
* Next steps that would move the needle (in order): more demonstrations
  (100+ seeds) and longer training; a higher-resolution wrist view for the
  cutlery; ACT as a comparison policy; OpenVINO export of the policy.

**Video set in VLA mode:** `Desktop/OMNI-Q_demo_videos_VLA/` (recorded from
this checkpoint; HUD tags each step VLA / handed to governed primitive; the
governed-planner set stays in `Desktop/OMNI-Q_demo_videos/` for the
simultaneous-arms and vision showcase clips).

## Speechmatics bonus

Voice path is real (transport, aggregation, authority checks, TTS sink) and
the demo uses NLU-parsed commands; what is *not* shown in the video set is a
live microphone session. For the bonus, record one clip of a real
Speechmatics session driving the authority change (the key is rotated; run
`integrations/speechmatics/scripts/run_voice_transport.py` per its README).

## Recommended order for the last day

1. Decide the VLA story from the numbers below; write the README section
   accordingly (what is VLA-driven, what is governed fallback, rates).
2. Re-run the OpenVINO benchmark on a Core Ultra 2/3 machine with the table
   detector (and the SmolVLA policy if it exported); put the numbers in the
   README and the video.
3. Edit the video from `Desktop/OMNI-Q_demo_videos/` in the brief's order
   (01 → 09); use `showcase_experiments/` only with its labels.
4. Record the live Speechmatics clip.
