# table_yolo_v3_oi — Open Images train-scale haul (20260911)

**Status: haul complete.** Same 7-class vocabulary (`plate cup fork spoon
knife napkin drawer`), YOLOv5 layout, built by
`perception/multisource_haul.py --source-set openimages-train` (no FiftyOne,
no auth — direct HTTP).  This is the v3 Open Images COMPONENT ONLY; merge
with table_yolo_v2 via `perception/merge_datasets.py` into `data/table_yolo_v3`.

## What was pulled

- Boxes CSV: https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-train-detection-bbox.csv
  (v6 OID train CSV auth-walled: True; failed
  attempts: ['https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bucket.csv -> HTTP ?'])
- Images: `https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg` — S3 train bucket.
- 14098 candidate images carry target
  boxes; offered class boxes {'plate': 5416, 'cup': 20533, 'fork': 1687, 'spoon': 1709, 'knife': 350, 'napkin': 0, 'drawer': 4414}.
- Classes with NO boxable OID label: ['napkin'] —
  Open Images supplies 6 of 7 classes; napkin stays LVIS-only.
- Downloads: 14098 attempted,
  **0 failed (403/404/rate-limit) and
  skipped** — individual image failures never crash the haul.
- min-per-class 7500 (greedy deficit order, seed
  13); kept cap None, attempt cap
  60000.

## Dedup (measured, owner requirement)

- Exact: sha256 of image bytes.  Near: 64-bit dHash, Hamming <= 6.
- Within-v3: {'v3_oi': {'exact': 0, 'near': 31}}
- Cross-set: **85 of 14098
  attempted images dropped** as dups of the reference sets:
- `data\table_yolo` (prefix `table_yolo:`): 2723 reference images hashed, **19 v3 candidates dropped**
- `data\table_yolo_v2` (prefix `table_yolo_v2:`): 15906 reference images hashed, **66 v3 candidates dropped**

## Final numbers

- **13982 unique images** — train 12590
  / val 1392 (val-frac 0.1, seed 13).
- Per-class KEPT boxes: plate 5385 · cup 20461 · fork 1668 · spoon 1699 · knife 347 · napkin 0 · drawer 4325.
- Elapsed: 4157.3 s.

## Usage

```bash
python perception/merge_datasets.py --input data/table_yolo_v2     --input data/table_yolo_v3_oi     --out data/table_yolo_v3 --seed 13
```

`data/` is gitignored; reproducible by re-running the command in the
manifest's `args`.

## License

Annotations: COCO (CC-BY-4.0), LVIS v1 (CC-BY-4.0), Open Images (CC-BY-4.0). Images retain their original Flickr/owner licenses (COCO, Open Images) — redistribution of images is NOT granted by the annotation license; keep this dataset in-house.

Portions derived from *The Hidden Canopy LLC* — [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
