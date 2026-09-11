# table_yolo_v3_hf — HF pseudo-labeled component (20260911)

**Status: pseudo-labeling complete.  THIS COMPONENT IS PSEUDO-LABELED, NOT
HUMAN-ANNOTATED.**  Every bounding box was emitted by OUR OWN fine-tuned
detector (`models/table_yolo_v2_ft_2026-09-11.pt`, conf >= 0.5)
running inference over two MIT-licensed HuggingFace image dumps — classic
self-training.  Expect label noise; treat as training signal only, never as
ground truth.

## What was pulled

- `lrad3/kitchen_utensils_13k` (MIT): 13432 images downloaded, 3918 kept after dedup
- `rypow/kitchen_utensils_5k` (MIT): 5000 images downloaded, 1 kept after dedup
  (the 5k dump is largely a near-duplicate of the 13k dump — within-set dHash
  dedup removed the overlap)
- Detected images pre-dedup (both repos, conf >= 0.5): **7303** of 18432 seen
  (18432 - 11129 no-detection skips); per-source pre-dedup detection counts
  were not recorded by this run — `perception/pseudo_label.py` now records them.
- Download date (UTC): 2026-09-11T14:00:20.358476+00:00
- Detector: models/table_yolo_v2_ft_2026-09-11.pt — our own YOLOv8n fine-tune on the table_yolo_v2 haul (OQ-008, perception/finetune.py); 7-class vocabulary plate/cup/fork/spoon/knife/napkin/drawer. Boxes emitted here are PSEUDO-LABELS produced BY this model (self-training), conf>=0.5, NOT human-annotated.

## Dedup (measured, owner requirement)

- Exact: sha256 of image bytes.  Near: 64-bit dHash, Hamming <= 6
  (same helpers as multisource_haul).
- Within-set dups: {'v3_hf': {'exact': 1451, 'near': 1909}}
- Cross-set: **24 of
  3943 detected candidates dropped**:
- `data\table_yolo` (prefix `table_yolo:`): 2723 reference images hashed, **12 v3_hf candidates dropped**
- `data\table_yolo_v2` (prefix `table_yolo_v2:`): 15906 reference images hashed, **12 v3_hf candidates dropped**

## Final numbers

- **3919 unique images** — train 3540
  / val 379 (val-frac 0.1, seed 13).
- Per-class KEPT boxes: plate 97 · cup 2362 · fork 903 · spoon 946 · knife 964 · napkin 7 · drawer 20.
- Skipped (no detections >= conf): 11129 of
  18432 seen; unreadable: 0.
- Elapsed: 5255.4 s.

## License

Source image dumps: lrad3/kitchen_utensils_13k and rypow/kitchen_utensils_5k, both MIT-licensed on HuggingFace (verified at download). Labels are OUR OWN model's pseudo-labels, not the repos' annotations.

Portions derived from *The Hidden Canopy LLC* — [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
