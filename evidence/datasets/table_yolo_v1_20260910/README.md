# table_yolo_v1 — first bounded multi-source haul (2026-09-10)

**Status: first haul complete.** OQ-008 tabletop detector training/val data,
7 classes (`plate cup fork spoon knife napkin drawer`), YOLOv5 layout.
Built by `perception/multisource_haul.py` (no FiftyOne, no auth — direct HTTP).
Copy of `manifest.json` from the dataset dir sits next to this file.

## What was pulled

| Source | Split | Status | Images kept | Boxes offered |
|---|---|---|---|---|
| COCO 2017 | val2017 | ok | 622 (of 5000 val images carry target boxes) | cup 1242, fork 215, spoon 253, knife 326 |
| LVIS v1 | val | ok | merged into COCO images by image id + 1,617 COCO-train images pulled per-image | plate 1191, cup 2113, fork 652, spoon 458, knife 735, napkin 918, drawer 1430 |
| Open Images | v5 validation | ok (partial goal) | 484 unique of 486 candidates | plate 92, cup 386, fork 50, spoon 73, knife 32, drawer 255 |
| Objects365 | — | skipped | v1 requires registration; no no-auth URL. Revisit after registration. | — |

LVIS v1 val has **19,809 images spanning both COCO val2017 and COCO
train2017** and no `file_name` field (names derived from `coco_url`). Only
378 of 1,995 LVIS target-annotated images share COCO val ids (cross-source
dedup rate 18.9%); the rest were fetched per-image from
`images.cocodataset.org/train2017/`.

## Dedup (measured, owner requirement)

- Exact: sha256 of image bytes. Within-source: 0 hits (COCO+LVIS), 0 (OI).
- Near: 64-bit dHash (PIL 9x8 grayscale + numpy diff), Hamming ≤ 6.
  Within-source: 0 (COCO+LVIS), 2 (OI). Cross-source OI-vs-COCO/LVIS: 0 exact, 2 near.
- Cross-source COCO↔LVIS: by COCO image id (LVIS reuses COCO ids) — 378 matched.

## Final numbers

- **2,723 unique images** — train 2,447 / val 276 (seed 7, val-frac 0.10).
- Class boxes (verified by re-parsing every written label file):
  plate 1,283 · cup 3,740 · fork 917 · spoon 784 · knife 1,093 · napkin 918 · drawer 1,684.
- Verification: 2,723/2,723 images ↔ labels paired, 0 empty labels, 0 rows out of
  [0,1], 51 JPEGs spot-verified, manifest histogram == written-file histogram.
- Budget: ~11 min wall clock (downloads ~10.7 min of the 22-min cap),
  ~460 MB dataset + ~1.1 GB warm cache (< 8 GB cap).

## Honest gaps vs the >=1500 boxes/class target

Below target: **plate (1283), fork (917), spoon (784), knife (1093), napkin (918)**.
This is a *validation-split-only* haul by design of the budget; val splits are
small. Drawer (1684) and cup (3740) met target.

## Scale-up path (next hauls)

1. **COCO train2017** — `train2017.zip` 19 GB from images.cocodataset.org;
   `instances_train2017.json` is inside the already-downloaded
   annotations zip. Expect ~10x these numbers for cup/fork/spoon/knife.
2. **Open Images train** via CVDF: `oidv6-train-annotations-bbox.csv` is public
   (verified 200); images at `open-images-dataset.s3.amazonaws.com/train/<id>.jpg`.
   v6/v7 *validation/test* CSVs on storage.googleapis.com/openimages return 403
   without auth — v5 validation was the only public validation boxes file; only
   486 of its 41,620 val images carry our classes.
3. **Objects365** — pending registration.
4. LVIS **train** annotations (`lvis_v1_train.json.zip`, dl.fbaipublicfiles.com)
   reuse the already-cached COCO train2017 images once pulled in step 1 —
   plate/napkin/drawer density comes almost entirely from LVIS.

## Usage

```bash
python perception/finetune.py --data data/table_yolo_v1/data.yaml ...
```

`data/` is gitignored; this dataset is reproducible by re-running
`python perception/multisource_haul.py` (cache under `data/_haul_cache/`
skips already-downloaded archives).

## License

Annotations: COCO (CC-BY-4.0), LVIS v1 (CC-BY-4.0), Open Images (CC-BY-4.0).
Images retain their original Flickr/owner licenses — redistribution of images
is NOT granted by the annotation licenses; keep this dataset in-house.

Portions derived from *The Hidden Canopy LLC* — [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
