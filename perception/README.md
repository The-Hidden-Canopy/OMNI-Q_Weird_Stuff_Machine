# `perception/` — the table-setting detector (OQ-008)

Fine-tunes a 7-class YOLO (`plate, cup, fork, spoon, knife, napkin, drawer`)
from **our own** `KissTheHabit/yolov8n-hituav-thermal-finetune`, on a large
real-image set with a synthetic depth top-up. Serves as the perception node
in `omni_q.providers` / `omni_q.scheduler` for the Intel online entry track;
the QAIRT/Qualcomm export path is bonus, not required. Plan:
[`../docs/datasets.md`](../docs/datasets.md).

## Pipeline

| step | script | what |
|------|--------|------|
| 1 | `build_dataset.py` | pull tableware detections from Open Images V7 / COCO / LVIS via FiftyOne, remap every source label to the 7 classes (`classmap.py`), export YOLOv5 layout |
| 1a | `multisource_haul.py` | **no-FiftyOne alternative to step 1** — same classmap, direct-HTTP pulls (COCO val2017 + LVIS val + Open Images v5 validation via CVDF S3), sha256 + dHash dedup with *measured* dup rates in the manifest. First haul 2026-09-10: 2,723 unique images → `data/table_yolo/` (see `evidence/datasets/table_yolo_v1_20260910/`); `perception/data.yaml` consumes it unchanged. Train-scale mode `--source-set train` builds **table_yolo_v2** from COCO train2017 (selective zip extraction, zip deleted before hashing) + LVIS v1 train (images ARE COCO train images → merged by id), with `--dedup-vs` cross-set dedup against v1; see `evidence/datasets/table_yolo_v2_*` |
| 2 | `synth_ingest.py` | fold in MuJoCo-rendered frames (or Objects365 / Roboflow YOLO-txt) — targets `drawer` and the deploy-camera slice where real data is thin |
| 3 | `finetune.py` | Ultralytics fine-tune from the thermal `.pt` (7-class head auto-init), few epochs, early-stop on val mAP50 |
| 4 | `export.py` | `best.pt → ONNX → OpenVINO IR` (Intel) and `ONNX → QAIRT/QNN` (Qualcomm) |

`classmap.py` is dependency-free and unit-tested (`tests/test_classmap.py`) —
it's the deterministic core; everything else is I/O around it.

## Run

```bash
pip install -r perception/requirements.txt

# 0. verify the plumbing (tiny pull, no big download)
python perception/build_dataset.py --sources coco-2017 --smoke

# 1. the real haul  (~150-300k images; drawer will flag as THIN)
python perception/build_dataset.py \
    --sources open-images-v7 coco-2017 lvis \
    --max-per-source 60000 --val-frac 0.10 --out data/table_yolo

# 1b. Objects365 (no FiftyOne loader) — get its YOLO export, then:
python perception/synth_ingest.py --src <objects365_yolo> --out data/table_yolo \
    --src-names <objects365_classes.txt>

# 2. synthetic depth top-up (drawer + deploy camera + hard occlusion)
python perception/synth_ingest.py --src renders/mujoco --out data/table_yolo --tag synth

# 3. fine-tune from our thermal YOLOv8n
python perception/finetune.py --epochs 4 --freeze 10 --imgsz 640

# 4. export for both tracks
python perception/export.py --weights runs/table_yolo/ft/weights/best.pt
```

## Class map (`classmap.py`)

| target | Open Images V7 | COCO | LVIS | Objects365 |
|--------|----------------|------|------|------------|
| plate | Plate | – | plate, saucer, platter | Plate |
| cup | Coffee cup, Mug, Wine glass | cup, wine glass | cup, mug, wineglass, goblet | Cup, Wine Glass |
| fork | Fork | fork | fork | Fork |
| spoon | Spoon | spoon | spoon, soupspoon, wooden_spoon | Spoon |
| knife | **Kitchen knife** (not weapon "Knife") | knife | knife, butter_knife, table_knife | Knife |
| napkin | Napkin | – | napkin | Napkin |
| drawer | **Drawer** | – | drawer | – |

`drawer` has only two real sources — the synthetic top-up carries it.
Anything not listed maps to `None` and is dropped.

## Notes

- Real annotations (OI / COCO / LVIS): **CC-BY-4.0**, fine to train the
  submitted model. Objects365: free academic + commercial (confirm terms).
- Baselines for the exported ONNX live in
  `evidence/benchmark_results/yolo_host_baseline_*` (thermal base: ~83 ms mean
  CPU @ 640).
- Images are hard-linked into `data/table_yolo/` where possible (no copy blow-up).
