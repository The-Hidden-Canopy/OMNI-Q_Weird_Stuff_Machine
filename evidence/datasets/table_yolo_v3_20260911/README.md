# table_yolo_v3 — merged train-scale dataset (20260911)

**Status: merge complete.** Same 7-class vocabulary (`plate cup fork spoon
knife napkin drawer`), YOLOv5 layout, built by `perception/merge_datasets.py`
from the v2 (COCO+LVIS train) and v3 OI component hauls.  Source dataset dirs
were only READ (hardlinks into this tree) — v2 and the OI dir are intact.

## Source composition

- `table_yolo_v2`: 15906 of 15906 images kept
- `table_yolo_v3_oi`: 13982 of 13982 images kept
- `table_yolo_v3_hf`: 3900 of 3919 images kept

## Dedup (belt-and-braces pass on top of haul-time dedup)

- Exact: sha256 of image bytes.  Near: 64-bit dHash, Hamming <= 6.
- **19 cross-input duplicates dropped**
  (matched against: {'table_yolo_v3_oi': 19}).

## Final numbers

- **33788 unique images** — train 30401 /
  val 3387 (val-frac 0.1, seed
  13 — same seed convention as the v2 haul).
- Class boxes: plate 11396 · cup 59646 · fork 10525 · spoon 10480 · knife 11800 · napkin 3984 · drawer 12264.
- Elapsed: 555.4 s.

## Usage

```bash
python perception/finetune.py --data data/table_yolo_v3/data.yaml ...
```

`data/` is gitignored; reproducible by re-running the command in the
manifest's `args`.

## License

Annotations: COCO (CC-BY-4.0), LVIS v1 (CC-BY-4.0), Open Images (CC-BY-4.0). Images retain their original Flickr/owner licenses (COCO, Open Images) — redistribution of images is NOT granted by the annotation license; keep this dataset in-house.

Portions derived from *The Hidden Canopy LLC* — [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
