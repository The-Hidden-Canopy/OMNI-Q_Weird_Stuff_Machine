# OMNI-Q — Intel Physical AI Online Challenge submission (2026-09-16)

**Challenge:** Bimanual VLA Manipulation with Multi-Modal Reasoning — *Setting Up a
Dinner Table* ([official brief](docs/challenge-briefs/intel-online-physical-ai-challenge.md)).
**Bonus track:** Speechmatics (voice → constraints).

This page is the judges' entry point: what was built, what controls the arms,
what was measured, where every number comes from, and how to reproduce each
deliverable. Everything stated here is backed by a receipt under `evidence/`
or by code in this repository; where something is not done, it says so.

OMNI-Q is the custom architecture. YOLO is the open-weight perception
component; SmolVLA is the learned motor-policy component; IK realizes motion;
OpenVINO is the Intel inference runtime; and the robot endpoints are
replaceable capabilities inside the architecture.

## 1. Architecture (Observe → Understand → Plan → Act → Optimize)

```text
 Observe     four scene cameras + two wrist cameras rendered from the MuJoCo scene
             └─ scene-trained YOLOv8n (7 classes) as an OpenVINO INT8 IR on Intel CPU / iGPU
                └─ multi-camera fusion → back-projected object poses on the table plane
 Understand  natural language (typed or Speechmatics voice) → NLU → goal + constraints
             IDA Omni reasoner (our own trained body, fenced plan grammar) proposes the next steps
 Plan        governed core validates every proposal (real object, correct zone, reachable arm,
             PICK-before-MOVE, operator authority) and completes what the model did not address;
             re-plans on failed placements, on "don't use the left arm", on a servo fault
 Act         SmolVLA (lerobot/smolvla_base, action expert fine-tuned on this stack's own
             demonstrations) leads every single-arm PICK / MOVE from the cameras + state + language;
             a small IK solve realises each Cartesian delta; the governed contact primitive
             continues from where the policy left the arm; the plate is a two-arm rim pinch
 Verify      end-of-run physical check: every object in its zone and upright, read from physics
 Optimize    OpenVINO 2026.3 — FP32 / FP16 / INT8 (NNCF) IRs, CPU + iGPU, latency + throughput hints
```

Simulation: MuJoCo 3.13, two **SO-101** arms from the MuJoCo Menagerie mirror of
TheRobotStudio/SO-ARM100 (unchanged joint limits, torque limits, gripper),
tableware modelled from primitives (rimmed plate, hollow cup, bent-handle fork
and spoon, fan-fold napkin), real contact physics — **no welds, no teleports, no
collision exemptions**; a failed placement leaves the object where it fell
(`src/omni_q/intel_sim.py`).

## 2. What controls the arms (say it exactly this way)

| Layer | Component | Learned? | Evidence |
|---|---|---|---|
| Plan | `OmniPlanner` = IDA Omni reasoner (fenced decoding) + governed validation/completion | yes (our checkpoint `models/omni_planner_r1_final.pt`) | HUD prints each decision; receipts label `model-proposed step, core-validated` / `core-completed` |
| Motion, single-arm steps | SmolVLA fine-tune (`integrations/intel/vla/`) leads; governed primitive completes | yes (LeRobot 0.6.1, `outputs/train/smolvla_so101_table_20260915_0419/checkpoints/008000`) | HUD tags `[VLA: SmolVLA]` / `[VLA-led -> governed completion]`; receipts carry `vla_attempt` + `fallback` |
| Motion, plate | governed bimanual rim-pinch primitive (lockstep carry, closed-loop lowering) | no | receipts; `[two-arm primitive]` tag |
| Perception | YOLOv8n trained on renders of this scene, served by OpenVINO INT8 | yes | `evidence/benchmark_results/openvino_table_detector_2026-09-15/` |
| Voice | Speechmatics realtime → NLU → constraints (`prefer_arm`, authority) | — | `integrations/speechmatics/` |

Honest limits: the policy leads the reach and the start of every carry from
the cameras, but in the measured harness **0 of 124 single-arm steps were
finished by the policy alone** — the governed contact primitive completed each
one (it cannot yet localise thin cutlery). The reasoner contributes *which
object next / which arm* (e.g. seed 903: 4 proposed, 1 accepted, run 5/5);
it does not plan the whole table. Paired (both-arms-at-once) execution is
governed-only; in VLA/OMNI mode arms execute one step at a time.

## 3. Results across 10 randomized seeds

Randomization per seed (`IntelSceneConfig`, seeded): placement ±12 mm, yaw
±0.20 rad, mass ±25 %, friction ±15 %, object colours, light angle/intensity,
floor/background tone, prompt phrasing. The variation is compiled into the
model (`omniq_variation` custom text) and printed on each montage title card.

| Mode | Seeds | Resolved | Placements | Final state all in zone & upright | Receipt |
|---|---|---|---|---|---|
| Governed planner + contact primitives, both arms simultaneous | 900–909 | 9/10 | 49/50 | 9/10 | `evidence/benchmark_results/parallel_arms_2026-09-14/harness_seed900_x10_final_layout/` |
| VLA-first (SmolVLA leads, governed completes) | 900–909 | 8/10 | 48/50 | 8/10 | `evidence/benchmark_results/vla_smolvla_2026-09-15/harness_seed900_x10/` |
| VLA-first, policy sampling seeded per scene/step | 900–915 | 12/16 | — | 12/16 | `evidence/benchmark_results/vla_smolvla_2026-09-15/harness_seed900_x16_seeded/` |

Seed 908 is the one recurring outlier in every mode (plate carry overshoot).
The 10-seed montage clips are recorded on passing seeds and print the
end-of-run physical check on each result card.

## 4. OpenVINO optimisation on Intel hardware

Measured on this laptop's **Intel Core 5 210H** (CPU + Intel iGPU `GPU.0`),
OpenVINO 2026.3, real 640×640 val images, 50 warm-up + 200 timed
([`benchmark_openvino_table.py`](integrations/intel/scripts/benchmark_openvino_table.py),
[receipt](evidence/benchmark_results/openvino_table_detector_2026-09-15/README.md)):

| precision | mAP50 / mAP50-95 | IR | CPU latency | CPU FPS | iGPU latency | iGPU FPS |
|---|---|---|---|---|---|---|
| FP32 | 0.980 / 0.917 | 12.4 MB | 45.5 ms | 39 | 13.4 ms | 96 |
| FP16 | 0.979 / 0.916 | 6.4 MB | 45.4 ms | 39 | 14.5 ms | 95 |
| **INT8 (NNCF PTQ, scene-calibrated)** | **0.981 / 0.915** | **3.6 MB** | **18.0 ms** | **98** | **11.4 ms** | **106** |

INT8 is 2.5× faster on the CPU with no accuracy loss on this task; the INT8 IR
(`models/table_yolo_v4_int8_openvino_model/`) is what the perception loop
loads. The script enumerates every OpenVINO device on the box and runs both
`LATENCY` and `THROUGHPUT` hints; on a Core Ultra Series 2/3 it will also list
and benchmark the NPU (`--devices CPU GPU NPU`) — this part has no NPU.

**Intel hardware mapping**

| Component | Runs on | Notes |
|---|---|---|
| MuJoCo physics + camera rendering | Intel CPU; GL on the iGPU | 25 fps recordings at real-time pace |
| Table detector (YOLOv8n) | OpenVINO INT8 IR, CPU or iGPU (`--device CPU|GPU.0`) | NPU untested (no NPU on the dev part) |
| SmolVLA policy | PyTorch (CUDA on the dev laptop's RTX 4050 for the recordings) | **not yet exported to OpenVINO** — the next optimisation step |
| IDA Omni reasoner | PyTorch (same GPU) | ~45–70 s per decision; sim-time video hides decode latency |
| NLU / governed core / verifiers | CPU | pure Python |
| Speechmatics | cloud realtime API | key never stored in the repo |

## 5. Training approach

* **Perception:** 7-class YOLOv8n trained on renders of this exact scene under
  the same randomization (`integrations/intel/scripts/make_table_yolo_dataset.py`,
  `train_table_yolo.py`; `data/table_yolo_v4/`), 268-image val split.
* **Policy:** `record_expert_demos.py` records the governed expert at 10 Hz —
  three 320×240 cameras (overhead, front, wrist), 9-d state, 7-d
  Cartesian-delta action with an absolute jaw command, per-step language —
  into a LeRobot v3 dataset (40 episodes, 65,850 frames, 20 seeds).
  `train_smolvla.sh` fine-tunes the `lerobot/smolvla_base` action expert
  (vision encoder frozen), 8000 steps, batch 8, RTX 4050, ~78 min.
* **Reasoner:** our own IDA Omni body trained on plan-grammar traces
  (`models/omni_planner_r1_final_receipt.json`); at run time the identifier
  slots are fenced to the scene's legal object ids and zones.

## 6. Robustness methods

Seeded randomization above; contact-honest primitives with verifiers
(grasp integrity: object sliding > 10 mm or turning > 8° in the pinch hands
the carry over); clearance-planned rim-pinch retries; one immediate retry
then deferral in the planner; re-planning on failed placements, on operator
authority changes (voice) and on a physical arm fault; end-of-run physical
check independent of the receipts; perception-driven verification through
the cameras in the e2e clip.

## 7. Demonstration video

The complete generated video set is recorded locally and remains outside Git in
the brief's order. The supplied representative full-run clip is checked in at
[`evidence/demo/01_full_run_seed903_director.mp4`](evidence/demo/01_full_run_seed903_director.mp4).
The remaining local clips are:
01 full run (single camera / director / 2×2 grid / six cameras), 02 second
seed, 03 perception e2e with YOLO/OpenVINO overlays, 04 two-arm plate,
05 hand-off, 06 handshake, 07 voice authority change, 08 arm failure,
09 ten-seed montage. Three sets exist, each with a README that states what
controlled the arms:

* governed set (both arms simultaneous; 10/10 montage on seeds 900–907, 909, 910);
* VLA-first set (SmolVLA leads every single-arm step; HUD tags; 10/10 montage on seeds 901, 903, 905–907, 910–914);
* OMNI-advised set (reasoner decisions printed on the HUD + SmolVLA motion; 10/10 montage on seeds 902–906, 910, 911, 913, 917, 918 — OMNI-mode takes vary between runs, so failed takes were swapped for other seeds and every failed take is kept in `replaced/` with its tally).

* Speechmatics bonus (`10_speechmatics_authority_*` in the VLA set): a real Speechmatics realtime session (spoken and transcribed live on 2026-09-13; every provider message recorded in `integrations/speechmatics/samples/live_session_2026-09-13.jsonl`) is replayed through the voice boundary — mapper → utterance aggregator → intent accumulator → authority → `RuntimeMutator` — and withdraws the left arm mid-run; `integrations/speechmatics/scripts/record_voice_authority_clip.py`. Set `SPEECHMATICS_API_KEY` and use `run_voice_transport.py --file/--mic --record` for a fresh live session.

Every clip's end state is checked from physics after the run (all five objects
in zone and upright), not only "placed at some point"; the recording log prints
`final N/5 in zone & upright` per clip.

Extras (labelled on screen): a 4-unit / 8-arm fleet where each unit is a real
engine run; a 6-arm relay and a drone/rover escalation that are scripted
choreography and say so.

The same recording is mirrored at [Google Drive](https://drive.google.com/file/d/1NMlxlWw5OBF0dZCn98CXpPsI7FeGEBAf/view?usp=drivesdk);
Drive access remains controlled by the file's sharing settings. Its checksum
and provenance are recorded in [`evidence/demo/README.md`](evidence/demo/README.md).

## 8. Reproduce

```bash
py -3 -m venv .venv
.venv/Scripts/python -m pip install -e ".[intel,smolvla,speechmatics]"
PYTHONPATH=src .venv/Scripts/python -m pytest tests -q                             # 719 passed, 9 skipped (2026-09-15)

# trained artifacts (gitignored; gated HF repo, access auto-approved after `hf auth login`):
#   our SmolVLA fine-tune + the OMNI planner checkpoint -> models/hf_omni_q_table/
bash scripts/fetch_checkpoints.sh
export OMNIQ_VLA_CHECKPOINT=$PWD/models/hf_omni_q_table/smolvla_so101_table
export OMNIQ_OMNI_CHECKPOINT=models/hf_omni_q_table/omni_planner/omni_planner_r1_final.pt OMNIQ_OMNI_RECEIPT=models/hf_omni_q_table/omni_planner/omni_planner_r1_final_receipt.json
# the scene detector's OpenVINO IRs (FP32, INT8) are committed under models/; nothing else to download

# 10-seed randomized evaluation (governed core; add OMNIQ_VLA_CHECKPOINT / OMNIQ_OMNI_REASONER for the other modes)
PYTHONPATH=src .venv/Scripts/python -c "from omni_q.intel_sim import run_intel_table_evaluation_report as r; print(r('tmp/harness_seed900_x10', trials=10, seed=900))"

# VLA: record demos -> fine-tune -> evaluate (integrations/intel/vla/README.md)
.venv/Scripts/python integrations/intel/vla/record_expert_demos.py --seeds 900 901 902 903 904 --root datasets/so101_table_vla_absjaw --repo-id omni-q/so101_table_vla_absjaw --jaw-mode absolute
bash integrations/intel/vla/train_smolvla.sh 8000 8
.venv/Scripts/python integrations/intel/vla/run_vla_eval.py --seeds 900 901 902 903 904 905 906 907 908 909   # uses OMNIQ_VLA_CHECKPOINT

# Intel inference benchmark (every OpenVINO device on the box, FP32/FP16/INT8, LATENCY + THROUGHPUT, mAP check)
.venv/Scripts/python integrations/intel/scripts/benchmark_openvino_table.py      # falls back to the committed IRs + live scene renders on a fresh clone
.venv/Scripts/python integrations/intel/scripts/plot_openvino_benchmark.py evidence/benchmark_results/openvino_table_detector_2026-09-15/receipt.json

# Demo clips (writes outside the repo, Desktop/OMNI-Q_demo_videos*)
.venv/Scripts/python integrations/intel/scripts/record_demo_videos.py
.venv/Scripts/python integrations/intel/scripts/record_seed_montage.py --seeds 900 901 902 903 904 905 906 907 909 910
#   VLA-first:     with OMNIQ_VLA_CHECKPOINT exported (same commands; --out Desktop/OMNI-Q_demo_videos_VLA)
#   OMNI-advised:  + OMNIQ_OMNI_REASONER=omni (and the two OMNIQ_OMNI_* variables above; --out Desktop/OMNI-Q_demo_videos_OMNI)
#   the montage caches passing seeds per mode (_montage_cache*/) so a failed seed costs one seed's re-run
```

Live viewer versions of every clip: [`DEMO.md`](DEMO.md). Full status against
the rubric, with gaps: [`docs/submission-readiness-2026-09-15.md`](docs/submission-readiness-2026-09-15.md).

## 9. Known gaps (stated, not hidden)

* SmolVLA and the reasoner are not yet under OpenVINO; the detector is.
* OMNI-advised runs are not deterministic across takes (the reasoner's proposals
  depend on the policy's rollouts): ~40 % of OMNI-mode seed attempts failed
  while recording, versus 2/10 in the VLA-first harness. Reported as measured.
* Benchmarks were taken on a Core 5 210H, not a Core Ultra Series 2/3; the
  script is device-agnostic and will pick up the NPU there.
* The policy does not finish grasps on its own yet; the governed primitive
  completes every step (counted, tagged).
* Cup hand-off between arms is 8/20 in its isolated harness (reported as is).
* Object *shape* is not randomized (one model per object).
