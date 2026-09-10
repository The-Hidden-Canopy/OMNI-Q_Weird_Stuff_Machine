# Dataset sourcing — one use case: bimanual dual-SO-101 table setting

> Scoped to the Intel Online Challenge (`docs/challenge-briefs/intel-online-physical-ai-challenge.md`):
> two simulated SO-101 arms in MuJoCo, natural-language instruction, camera
> reasoning, multi-step table setting. Two data problems:
>
> 1. **YOLO** — a tabletop object detector (class, bbox/center, conf, stable IDs)
>    for the perception node (OQ-008).
> 2. **Omni** — (a) a VLA / imitation policy for the arms (SmolVLA / π0.5 / ACT,
>    per the brief) and (b) instruction → capability-graph data for the planner.

Object vocabulary from the OQ-007 pack: **plate, cup/mug, fork, spoon, knife,
napkin, drawer**, plus the table surface and target place-setting zones.

---

## 1. YOLO — tabletop object detection

### 1a. Primary: synthetic from the sim (recommended, do first)

The challenge explicitly wants robustness "under changes in object weight,
friction, shape, lighting, background, initial placement… not tied to one fixed
scene." Rendering labelled frames **from the MuJoCo scene itself** gives
perfectly-accurate boxes (ground-truth poses are known), infinite scale, and
exact in-distribution data.

- **How:** the OQ-006/OQ-007 scene + `dm_control` / `mujoco` renderer, or
  [Kubric](https://github.com/google-research/kubric), or NVIDIA Omniverse
  Replicator. Domain-randomise lighting, table/wall textures, camera pose,
  object count/placement, distractors.
- **Assets:** [YCB object set](https://www.ycbbenchmarks.com/) has scanned
  meshes for mug, bowl, plate, fork, spoon, knife; MuJoCo Menagerie household
  props; the `menagerie_so_arm100` scene already vendored under
  `integrations/intel/assets/`.
- **Output:** COCO-JSON or YOLO-txt, auto-generated. Target ~20–50k frames,
  50/50 randomised vs. "nominal" scene.

### 1b. Real-image pretraining / sim-to-real seasoning

Pretrain or mix in real photos so the detector isn't brittle to sim artefacts.
Ranked by fit:

| Source | Table-setting coverage | Size / licence | Notes |
|--------|------------------------|----------------|-------|
| **LVIS v1** (`lvisdataset.org`) | `plate, saucer, platter, wineglass, mug, coffee_cup, napkin, tablecloth, place_mat, fork, spoon, butter_knife, soupspoon` | 100k COCO images, CC-BY-4.0 anns | Best real-image match for the **full** vocabulary; federated (non-exhaustive) labels |
| **COCO 2017** (`cocodataset.org`) | `cup, fork, knife, spoon, bowl, wine glass, dining table, bottle` | ~123k train imgs, CC-BY-4.0 anns | 6 classes clean and dense; **no plate / napkin / drawer** |
| **Open Images V7** (`storage.googleapis.com/openimages`) | `Tableware, Plate, Coffee cup, Mug, Kitchen knife, Fork, Spoon, Bowl, Wine glass, Napkin, Drawer, Kitchen & dining room table` | ~1.9M imgs w/ boxes, CC-BY-4.0 | Only real set with **`Drawer`**; box quality varies |
| **Roboflow Universe** — [Cutlery Detection](https://universe.roboflow.com/home-detection/cutlery-detection-1ofa0), [Kitchen Utensils Recognition](https://universe.roboflow.com/table-utensils-detector/kitchen-utensils-recognition), [Dinner Plate Detection](https://universe.roboflow.com/platedetection-project/dinner-plate-detection), [Spoon/Fork/Knife](https://universe.roboflow.com/kmitl-qnilk/spoon-fork-or-knife) | narrow, YOLO-ready | small (60–~2k imgs each), mixed licences | fastest to drop in; already YOLOv8 YAML/txt |
| **AI2-THOR kitchen renders** ([Roboflow](https://universe.roboflow.com/ai2thor/ai2thor-kitchen-items-actions)) | kitchen items | synthetic, sim | cross-check for another sim domain |
| EPIC-KITCHENS / Ego4D | plate/cup/cutlery in clutter | large, research licence | egocentric, hand-occluded — low priority |

**Plan:** fine-tune YOLOv8n (proven pipeline —
`KissTheHabit/yolov8n-hituav-thermal-finetune`) on **synthetic (1a) + LVIS/Open
Images subset (1b)**, mapped to the 7-class table vocabulary. Export ONNX →
QAIRT/OpenVINO per `models/README.md`.

### 1c. Not found: a ready-made tableware detector

No dedicated tableware YOLO on the Hub — training our own is the OQ-008
deliverable and matches the "bring your own vision model" strategy.

---

## 2. Omni — VLA / imitation policy (the arms)

Brief names **SmolVLA, π0.5, ACT** and **LeRobot**. Base models:
`lerobot/smolvla_base`, π0.5 via `openpi`.

### 2a. SO-101 / SO-100 policy pretraining

| Dataset | What | Size |
|---------|------|------|
| [`dongyoonkim/so101-pi05-base-dataset`](https://hf.co/datasets/dongyoonkim/so101-pi05-base-dataset) | **150 public SO-101/SO-100 teleop datasets unified** into one LeRobot dataset (π0.5 base) | 10k–100k frames, 5k dl |
| [`lerobot/svla_so101_pickplace`](https://hf.co/datasets/lerobot/svla_so101_pickplace) | official LeRobot SO-101 pick-place (SmolVLA reference) | 50 eps, 43 likes |
| [`armnet/armnetbench_v01_lerobot_so101`](https://hf.co/datasets/armnet/armnetbench_v01_lerobot_so101) | single-arm SO-101 bench, 50 teleop refs/task + 7 policy evals | — |
| [`CoRL2026-CSI/Isaaclab-so101_11task_openpi_v21`](https://hf.co/datasets/CoRL2026-CSI/Isaaclab-so101_11task_openpi_v21) | **IsaacLab SO-101 sim, 11 tasks**, OpenPI v2.1 | 1k–10k |
| [`5hadytru/so101_bench_real_1_v2.1`](https://hf.co/datasets/5hadytru/so101_bench_real_1_v2.1) | 3202 eps, ~2000 instruction-following, ~16 h | large |
| [`hbseong/record-pick-and-place-*-so101`](https://hf.co/datasets/hbseong/record-pick-and-place-pos5-so101) | teleop pick-place, several variants | 100k+ frames |

### 2b. Bimanual (the two-arm skills)

| Dataset | What |
|---------|------|
| [`armnet/armnetbench_v01_lerobot_bimanual_so101`](https://hf.co/datasets/armnet/armnetbench_v01_lerobot_bimanual_so101) | **bimanual SO-101** benchmark — 50 teleop refs/task + policy evals |
| [`andreaskoepf/dk1_cutlery_basket_2026-04-*`](https://hf.co/datasets/andreaskoepf/dk1_cutlery_basket_2026-04-21) | bimanual 2×6-DOF: unload a cutlery basket into a drawer organiser (100k–1M frames) |
| ALOHA / Mobile ALOHA (`lerobot/aloha_*`) | canonical bimanual imitation data (handoffs, co-manipulation) |
| [DROID](https://droid-dataset.github.io/) | 76k eps cross-lab manipulation (broad pretraining, mostly single-arm) |
| Open X-Embodiment / RT-X | ~1M+ eps cross-embodiment (pretrain, then SO-101 fine-tune) |

### 2c. Table-setting task demonstrations

| Dataset | Robot / task |
|---------|--------------|
| [`BAAI-DataCube/AgiBotWorld-Beta_G1_task_525_Placing_tableware_in_the_restaurant`](https://hf.co/datasets/BAAI-DataCube/AgiBotWorld-Beta_G1_task_525_Placing_tableware_in_the_restaurant) | **2267 episodes** placing tableware, LeRobot v3 |
| [`lirislab/franka_prepare_a_set_of_tableware`](https://hf.co/datasets/lirislab/franka_prepare_a_set_of_tableware) · [`prepare_two_sets_of_tableware`](https://hf.co/datasets/lirislab/prepare_two_sets_of_tableware) | Franka, prepare 1–2 place settings (8k / 33k frames) |
| [`RoboCOIN/R1_Lite_tableware_arrangement`](https://hf.co/datasets/RoboCOIN/R1_Lite_tableware_arrangement) · [`..._put_the_tableware_into_the_cupboard`](https://hf.co/datasets/RoboCOIN/R1_Lite_put_the_tableware_into_the_cupboard) | Galaxea R1-Lite, tableware arrange / stow (gated) |
| [`masato-ka/so100_cutlery_handling`](https://hf.co/datasets/masato-ka/so100_cutlery_handling) · [`cijerezg/multi-task-cutlery-v2`](https://hf.co/datasets/cijerezg/multi-task-cutlery-v2) | **SO-100/SO-101** cutlery handling / multi-task |
| [`youliangtan/so101-table-cleanup`](https://hf.co/datasets/youliangtan/so101-table-cleanup) | SO-101 clearing a table (inverse task, useful for perturbation) |

### 2d. Our own MuJoCo demos (primary, like 1a)

Teleop the OQ-006 dual-SO-101 scene with LeRobot's recording tools + OMPL for
scripted trajectories → LeRobot v3 dataset of `set the table` episodes with the
challenge's own physics. This is the data the submitted policy is actually
fine-tuned on; 2a–2c are pretraining and skill priors.

---

## 3. Omni — instruction → capability-graph (the planner)

The planner turns *"set the table, and dance while you do it"* into a graph over
the `omni_q.actions` vocabulary. Data is mostly **synthetic**:

- **Generate** `(instruction, world_state, plan_graph)` triples from templates ×
  the `actions.py` registry (preconditions/effects make plans checkable) ×
  the OQ-007 scene layouts × style modes. Add paraphrases via an LLM.
- **Distil** longer chains from an LLM planner prompted with the action specs.
- **Ground against** language-conditioned benchmarks for realism and eval:
  - **LIBERO** (130 MuJoCo tasks, templated language, 4 generalisation suites) — closest engine match.
  - **CALVIN** (34 tasks, long-horizon language chains, 5-step).
  - **RLBench `set_the_table`** (CoppeliaSim) — reference for the task's success predicates and instruction phrasing, not the data itself.
  - **LoHoRavens** (long-horizon language tabletop), **VLABench**, **ALFRED** (household task→subgoal decomposition).
- **Speechmatics (OQ-024):** the spoken-command side reuses `omni_q.nlu`
  phrasings; collect a small held-out set of real spoken table-setting +
  choreography commands for eval.

---

## Acquisition order

1. **Sim-synthetic YOLO data** from the OQ-006/OQ-007 scene (1a) — unblocks OQ-008 with zero licence risk.
2. **`dongyoonkim/so101-pi05-base-dataset`** + **`armnet/...bimanual_so101`** — policy pretraining / bimanual priors (2a, 2b).
3. **LVIS + Open Images** table-vocab subset (1b) — sim-to-real seasoning for YOLO.
4. **Our MuJoCo `set the table` teleop demos** (2d) — the policy fine-tune set.
5. **Synthetic instruction→graph corpus** (§3) — planner training + eval; ground on **LIBERO**.
6. **AgiBotWorld task_525 / lirislab tableware** (2c) — extra task-demo variety if time.

## Licensing notes

- COCO / LVIS / Open Images annotations: **CC-BY-4.0** — attribution, fine for a
  hackathon repo. Images: individual Flickr licences (Open Images/COCO curate
  permissive subsets).
- LeRobot Hub datasets above: **Apache-2.0** unless marked 🔒 Gated
  (`RoboCOIN/*` — request access or skip).
- Roboflow Universe: per-dataset (CC-BY / MIT / CC-BY-NC — check before training
  a submitted model; NC is eval-only).
- YCB meshes: free for research.
- Sim-synthetic + our own teleop: **ours**, no third-party terms — the safest
  data for the submitted model.

## Not blocked on

Real SO-101 hardware (sim-first); a pre-existing tableware detector (none
exists — we train one); the 400M Omni-Q pretraining corpus (separate track, not
demo-critical per `docs/strategy-notes.md`).
