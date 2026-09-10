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
2. **Data = a large REAL-image subset pulled from public detection datasets**
   (not synthetic). Filter each to the tableware classes, remap to the 7-class
   vocab, merge, export YOLO format. Fastest via [FiftyOne](https://docs.voxel51.com).

   | Dataset | Real images / boxes | Tableware classes it carries | Get it |
   |---------|---------------------|------------------------------|--------|
   | **Open Images V7** | ~1.9M imgs w/ boxes | `Tableware, Plate, Coffee cup, Mug, Wine glass, Fork, Spoon, Kitchen knife, Napkin, Bowl,` **`Drawer`** (only real source for drawer), `Kitchen & dining room table` | `foz.load_zoo_dataset("open-images-v7", classes=[...], label_types=["detections"])` |
   | **Objects365 v2** | 2M imgs, 30M boxes | `Plate (15), Cup, Fork, Knife, Spoon (93), Bowl, Wine Glass, Napkin (102), Dinning Table` | [OpenDataLab](https://opendatalab.com/OpenDataLab/Objects365/download) or Ultralytics `Objects365.yaml` auto-download |
   | **COCO 2017** | 123k train, dense | `cup, fork, knife, spoon, bowl, wine glass, dining table` | `foz.load_zoo_dataset("coco-2017", classes=[...])` |
   | **LVIS v1** | COCO imgs, 1203 cls | `plate, saucer, platter, mug, coffee_cup, wineglass, napkin, place_mat, tablecloth, fork, spoon, butter_knife, soupspoon` | FiftyOne / lvis-api — best real density for **plate + napkin** |
   | **Roboflow Universe** | small, YOLO-ready | cutlery / plate / kitchen-utensil sets — merge for extra | direct download, already YOLOv8 txt |

   **Class map:** plate ← {OI:Plate, O365:Plate, LVIS:plate/saucer/platter};
   cup ← {COCO/O365 cup+wine glass, OI:Coffee cup/Mug/Wine glass, LVIS:mug/coffee_cup/cup/wineglass};
   fork/spoon/knife ← same-named in all four (drop OI weapon "Knife", keep "Kitchen knife");
   napkin ← {O365:Napkin, OI:Napkin, LVIS:napkin/place_mat};
   drawer ← {OI:Drawer} + a Roboflow furniture/drawer set to thicken it.

   Realistic haul: **~150–300k real labelled images** across the 7 classes
   (drawer is the thin one — a few k, supplement from Roboflow). No rendering.

3. Fine-tune **~2–5 epochs** from the thermal weights (1 frozen warm-up, 1–2
   unfrozen), `imgsz=640`, `yolov8n`, early-stop on val mAP50 — breadth comes
   from the real data, so keep passes few.
4. Export `best.pt → ONNX`, then ONNX → **OpenVINO IR** (Intel track) and
   ONNX → **QAIRT/QNN** (Qualcomm track). Detector emits `class, bbox,
   center, conf` + a stable per-object id (track by IoU + class).

That's the OQ-008 path: real public boxes, one FiftyOne merge, a short
fine-tune from our own weights.

**Fastest possible start** (if the big pulls are slow): the 3 Roboflow Universe
sets — [Cutlery Detection](https://universe.roboflow.com/home-detection/cutlery-detection-1ofa0),
[Kitchen Utensils](https://universe.roboflow.com/table-utensils-detector/kitchen-utensils-recognition),
[Dinner Plate](https://universe.roboflow.com/platedetection-project/dinner-plate-detection) —
already YOLOv8 txt, ~10 min to merge. Train on those first, then swap in the
Open Images + Objects365 + LVIS haul when it lands.

### Synthetic — for depth where real data is thin

Real data is the base; render MuJoCo frames to **fill its gaps**, not to replace
it. Ground-truth poses → exact boxes, so scale it up freely. Target where the
real haul is weakest:

- **`drawer`** — only Open Images carries it, sparsely. Render lots.
- **the exact deploy view** — the challenge's third-person table camera
  intrinsics/extrinsics, on the real object meshes.
- **hard combos real photos under-sample** — heavy occlusion, near-empty vs.
  fully-set tables, extreme lighting, tight clutter, the OQ-007 place-setting
  geometry, mid-manipulation frames (an arm across the object).

Blend by **need, not a fixed ratio** — roughly real-heavy overall, but
synthetic-heavy for `drawer` and the deploy-camera slice. Keep frames distinct
(breadth over repetition still applies). Train the merged real+sim set together,
short passes.

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

More real detection data if the 4-source haul isn't enough (all real images):
**EPIC-KITCHENS-100 + VISOR** (egocentric kitchen, plate/cup/cutlery/napkin
masks, cluttered, CC-BY-NC → eval only), **GraspNet-1Billion** (~97k real RGBD
tabletop frames, 88 objects incl. cutlery/cups/bowls, box+mask, CC-BY-NC-SA →
eval only), **ADE20K** (segmentation → boxes: plate/glass/fork/knife/spoon/
napkin/drawer), **V3Det** (13k classes, fine tableware), **SUN RGB-D** (indoor
scenes, furniture 3D boxes incl. drawers).

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

Licensing: COCO / LVIS / Open Images V7 annotations **CC-BY-4.0** (attribution,
fine to train the submitted model); Objects365 free for academic + commercial
(confirm on objects365.org before shipping); EPIC-KITCHENS / GraspNet
**CC-BY-NC** → eval / ablation only, never the submitted model; LeRobot Hub sets
Apache-2.0 unless 🔒 gated (`RoboCOIN/*`); Roboflow per-dataset (watch for
CC-BY-NC); our own MuJoCo top-up frames = ours.
