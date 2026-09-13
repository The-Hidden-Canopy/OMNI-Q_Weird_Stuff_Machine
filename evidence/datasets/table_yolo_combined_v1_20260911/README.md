# table_yolo_combined_v1 — combined YOLO dataset (20260911)

Status: built from the protected seven-class tableware tree plus locally
cached COCO 2017 train/validation images.  The tableware ids remain 0–6;
COCO aliases `wine glass -> cup` and `dining table -> table` are explicit.

## Composition

- `table_yolo_v3`: offered 33788; output images 33788; exact annotation merges 0; near duplicates dropped 0
- `coco_train2017`: offered 1527; output images 1527; exact annotation merges 0; near duplicates dropped 0
- `coco_val2017`: offered 712; output images 712; exact annotation merges 0; near duplicates dropped 0

Final images: **36027** — train 32401 /
val 3626 (seed 13).  Exact duplicate images were
kept once and their annotations unioned; near duplicates were dropped and
recorded in `manifest.json`.

Class boxes: plate 11396 · cup 63241 · fork 11385 · spoon 11319 · knife 12878 · napkin 3984 · drawer 12264 · person 3390 · bicycle 35 · car 85 · motorcycle 11 · airplane 4 · bus 4 · train 1 · truck 9 · boat 9 · traffic light 8 · fire hydrant 1 · stop sign 3 · parking meter 2 · bench 44 · bird 37 · cat 66 · dog 54 · horse 5 · cow 2 · bear 1 · zebra 1 · giraffe 2 · backpack 45 · umbrella 59 · handbag 108 · tie 99 · suitcase 17 · frisbee 4 · skis 3 · snowboard 14 · sports ball 11 · kite 2 · baseball bat 3 · baseball glove 2 · skateboard 4 · surfboard 1 · tennis racket 9 · bottle 1813 · bowl 1260 · banana 177 · apple 162 · sandwich 447 · orange 177 · broccoli 458 · carrot 608 · hot dog 173 · pizza 465 · donut 291 · cake 598 · chair 1717 · couch 162 · potted plant 275 · bed 101 · table 1418 · toilet 33 · tv 195 · laptop 186 · mouse 88 · remote 126 · keyboard 138 · cell phone 148 · microwave 102 · oven 253 · toaster 17 · sink 259 · refrigerator 161 · book 938 · clock 100 · vase 299 · scissors 54 · teddy bear 59 · hair drier 4 · toothbrush 39.

Use with an opt-in combined-head training run:

```text
python perception/finetune.py --data data/table_yolo_combined_v1/data.yaml \
    --base models/table_yolo_v2_ft_2026-09-11.pt --name combined_v1
```

The source images remain local/in-house artifacts.  COCO annotations are
CC-BY-4.0; image redistribution rights remain with their original owners.
