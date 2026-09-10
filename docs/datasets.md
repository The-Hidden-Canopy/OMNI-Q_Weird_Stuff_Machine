# Dataset sourcing — table setting, ~2 h to deadline

> Decisive path only. Full catalog is the appendix; ignore it unless the demo
> is already locked.

Use case: two simulated SO-101 arms in MuJoCo, natural-language instruction,
multi-step table setting. Object vocab from the OQ-007 pack:
**plate, cup, fork, spoon, knife, napkin, drawer**.

---

## YOLO — fine-tune from OUR model, do not start from COCO

**Base: [`KissTheHabit/yolov8n-hituav-thermal-finetune`](https://huggingface.co/KissTheHabit/yolov8n-hituav-thermal-finetune)**
— YOLOv8n, already fine-tuned (HIT-UAV thermal, 0.825 mAP50). The classes are
wrong (thermal UAV) but the weights are a trained YOLOv8n backbone we own, and
the train→export→deploy pipeline is proven. Transfer-learn from it:

1. Load those `.pt` weights, swap the detection head to **7 classes** (the vocab
   above), keep the backbone.
2. **Data (this is the whole job) — breadth over repetition:** render a *large,
   varied* synthetic set from the OQ-006/OQ-007 MuJoCo scene, every frame
   distinct. Ground-truth object poses → exact YOLO-txt boxes, free.
   - **Keep it similar to deploy:** the challenge's third-person table camera
     intrinsics/extrinsics, the real object meshes, plausible place-setting
     layouts. Don't drift the scene.
   - **Randomise only the nuisances:** lighting, table/wall texture, small
     camera jitter, object count/pose/occlusion, distractors, initial
     placement (the brief's perturbation axes).
   - Target ~15–30k frames, 90/10 split. More unique frames, not more passes.
3. Fine-tune **~2–5 epochs** (one pass warm-up with the backbone frozen, then
   1–2 with it unfrozen), `imgsz=640`, `yolov8n`. Watch val mAP50 per epoch and
   stop when it plateaus — the model should see many images a few times, never
   the same few images many times. Minutes on one GPU.
4. Export `best.pt → ONNX`, then ONNX → **OpenVINO IR** (Intel track) and
   ONNX → **QAIRT/QNN** (Qualcomm track). Detector emits `class, bbox,
   center, conf` + a stable per-object id (track by IoU + class).

That's the entire OQ-008 path inside the time budget. No external download,
no licence check, no scraping.

**Only if the sim renderer isn't ready:** grab the 3 Roboflow Universe sets
([Cutlery Detection](https://universe.roboflow.com/home-detection/cutlery-detection-1ofa0),
[Kitchen Utensils](https://universe.roboflow.com/table-utensils-detector/kitchen-utensils-recognition),
[Dinner Plate](https://universe.roboflow.com/platedetection-project/dinner-plate-detection)) —
already YOLOv8 YAML/txt, ~10 min to merge — and fine-tune on those instead.
Weaker (real photos ≠ sim), but it moves.

---

## Omni — nothing to train in 2 h

The demo runs on what already works: `RulePlanner` + `ScheduledPlanner` +
`RuntimeMutator` (mock/scripted execution), and the real MuJoCo path via
`demo_intel_sim.py`. **Do not** start a VLA fine-tune now.

- **Planner data:** generate `(instruction, world_state, plan_graph)` triples
  from templates × the `omni_q.actions` registry × OQ-007 layouts × style modes.
  Minutes of scripting, produces thousands of pairs, checkable against the
  action preconditions/effects. Enough for eval and a tiny classifier if wanted.
- **If (and only if) a GPU and a spare hour appear:** fine-tune
  `lerobot/smolvla_base` on a handful of MuJoCo `set the table` teleop episodes
  recorded with LeRobot's tools — a skill prior for one arm. Not on the critical
  path; the demo must stand without it.

---

## Appendix — fuller catalog (post-deadline / if the demo is locked)

Perception real-image seasoning: **LVIS v1** (best full-vocab match, CC-BY-4.0),
**COCO 2017** (cup/fork/knife/spoon/bowl), **Open Images V7** (only real set
with `Drawer`).

SO-101 policy pretraining: [`dongyoonkim/so101-pi05-base-dataset`](https://hf.co/datasets/dongyoonkim/so101-pi05-base-dataset)
(150 unified SO-101/SO-100 teleop sets), [`lerobot/svla_so101_pickplace`](https://hf.co/datasets/lerobot/svla_so101_pickplace),
[`CoRL2026-CSI/Isaaclab-so101_11task_openpi_v21`](https://hf.co/datasets/CoRL2026-CSI/Isaaclab-so101_11task_openpi_v21),
DROID / Open X-Embodiment.

Bimanual: [`armnet/armnetbench_v01_lerobot_bimanual_so101`](https://hf.co/datasets/armnet/armnetbench_v01_lerobot_bimanual_so101),
[`andreaskoepf/dk1_cutlery_basket_2026-04-21`](https://hf.co/datasets/andreaskoepf/dk1_cutlery_basket_2026-04-21),
ALOHA / Mobile ALOHA.

Table-setting demos: [`BAAI-DataCube/AgiBotWorld-Beta_G1_task_525_Placing_tableware_in_the_restaurant`](https://hf.co/datasets/BAAI-DataCube/AgiBotWorld-Beta_G1_task_525_Placing_tableware_in_the_restaurant)
(2267 eps), [`lirislab/franka_prepare_a_set_of_tableware`](https://hf.co/datasets/lirislab/franka_prepare_a_set_of_tableware),
[`masato-ka/so100_cutlery_handling`](https://hf.co/datasets/masato-ka/so100_cutlery_handling),
[`youliangtan/so101-table-cleanup`](https://hf.co/datasets/youliangtan/so101-table-cleanup).

Planner grounding / eval benchmarks: **LIBERO** (130 MuJoCo tasks, templated
language — closest engine match), **CALVIN**, **RLBench `set_the_table`**
(CoppeliaSim — reference for success predicates + phrasing), **LoHoRavens**,
**ALFRED**.

Licensing: COCO/LVIS/Open Images anns CC-BY-4.0; LeRobot Hub sets Apache-2.0
unless 🔒 gated (`RoboCOIN/*`); Roboflow per-dataset (watch for CC-BY-NC → eval
only); sim-synthetic + our teleop = ours, safest for the submitted model.
