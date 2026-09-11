# table_yolo_v2 — train-scale multi-source haul (20260910)

**Status: haul complete.** Same 7-class vocabulary as v1
(`plate cup fork spoon knife napkin drawer`), YOLOv5 layout, built by
`perception/multisource_haul.py --source-set train` (no FiftyOne, no auth —
direct HTTP).  Copy of `manifest.json` from the dataset dir sits next to this
file.

## What was pulled

| Source | Split | Status | Images kept | Boxes offered |
|---|---|---|---|---|
| COCO 2017 | train2017 | ok | 14485 | cup 28563, fork 5479, spoon 6165, knife 7770 |
| LVIS v1 | train | ok | 10091 | plate 5917, cup 10681, fork 3137, spoon 2278, knife 3515, napkin 3979, drawer 7927 |
| Open Images | v5 validation | skipped | — | — |

- COCO category ids (mapped by NAME, read from the json, never hardcoded):
  {'cup': [46, 47], 'fork': [48], 'knife': [49], 'spoon': [50]}
- LVIS category ids: {'cup': [344, 708, 1190], 'drawer': [390], 'fork': [469], 'knife': [615], 'napkin': [713], 'plate': [818, 819, 915], 'spoon': [988, 1000, 1194]}
- Selection: greedy per-class deficit order (min-per-class 3000,
  seed 13), capped at 25000 images.
- train2017.zip (~19 GB) downloaded whole but ONLY the 17084
  needed members were extracted (python zipfile, namelist-filtered); the zip
  was deleted immediately after extraction.  Projected extract was estimated
  and refused past 24 GB before downloading.

## Dedup (measured, owner requirement)

- Exact: sha256 of image bytes.  Near: 64-bit dHash, Hamming <= 6.
- Within-source: {'v2': {'exact': 4, 'near': 20}}
- Cross-set vs v1 (--dedup-vs data/table_yolo): 1154
  of 17084 v2 candidates dropped (2723
  v1 reference images hashed).

## Final numbers

- **15906 unique images** — train 14302 / val 1604
  (val-frac 0.1, seed 13 — a DIFFERENT seed than v1's 7;
  v2 is a fresh sample from a much larger pool, val images are NOT guaranteed
  disjoint from v1's val).
- Class boxes: plate 5915 · cup 36829 · fork 7958 · spoon 7841 · knife 10498 · napkin 3977 · drawer 7919.
- Elapsed: 1472.6 s.
- Disk (32 GB box): peak overlay usage 27G/32G (85%) during selective
  extraction — zip 19.3 GB + base image ~6.4 GB + extract; the zip was deleted
  right after extraction and usage fell back to ~9.5G. Actual uncompressed
  extract 2.69 GB (avg ~158 KB/img — the 180 KB projection was conservative).

## Usage

```bash
python perception/finetune.py --data data/table_yolo_v2/data.yaml ...
```

`data/` is gitignored; reproducible by re-running the command in the manifest's
`args`.

## License

Annotations: COCO (CC-BY-4.0), LVIS v1 (CC-BY-4.0), Open Images (CC-BY-4.0). Images retain their original Flickr/owner licenses (COCO, Open Images) — redistribution of images is NOT granted by the annotation license; keep this dataset in-house.

Portions derived from *The Hidden Canopy LLC* — [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
