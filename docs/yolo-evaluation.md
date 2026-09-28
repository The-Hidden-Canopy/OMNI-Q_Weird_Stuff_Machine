# YOLO evaluation and improvement plan

Snapshot: 2026-09-11

## Outcome

The current YOLO path is a useful integration baseline, not yet a manipulation-grade perception system. The next improvement should target deploy-domain data and class-specific failure modes. Larger models, quantization, and threshold tuning should be evaluated only after that baseline is measured on the camera views that drive action.

## Evidence reviewed

- Dataset manifest: `evidence/datasets/table_yolo_v2_20260911/manifest.json`
- Fine-tune log: `evidence/datasets/table_yolo_v2_20260911/train_ftv2.log`
- Quantization comparison: `evidence/benchmark_results/yolo_2bit_map_20260911_071815-eb6457d2e88b564e/`
- Runtime seam: `src/omni_q/yolo_perception.py`
- Contract tests: `tests/test_yolo_perception.py`

The v2 dataset contains `15,906` unique images: `14,302` train and `1,604` validation. The retained FP32 validation result is:

| Class | Validation instances | mAP50 |
|---|---:|---:|
| plate | 566 | 0.2183 |
| cup | 3,475 | 0.5054 |
| fork | 837 | 0.2910 |
| spoon | 833 | 0.2417 |
| knife | 1,038 | 0.2356 |
| napkin | 417 | 0.1843 |
| drawer | 771 | 0.5923 |
| **all** | **7,937** | **0.3241** |

The average is materially stronger than the older v1 evidence, but it is not a sufficient release metric. Cup and drawer contribute the strongest scores while plate, napkin, and the utensil classes remain weak. Those are precisely the classes needed for a precision table-reset demo. The broad web-image validation set also does not establish performance for the fixed camera geometry, lighting, object scale, occlusion, or hand intrusion expected at the cell.

The retained compression comparison is a stop signal, not a deployment result: its numeric rows report FP32 mAP50 `0.3241`, MXFP4 `0.0775`, NVINT2 `0.0003`, and MXFP2 `0.0`. However, the receipt's `labels.baseline` and README narrative still describe the same checkpoint as a 4-CPU-epoch run with approximately `0.058` mAP50, which conflicts with the numeric FP32 row and the v2 training log. This is a provenance defect. Treat the quantization deltas as provisional until the receipt is reconciled or the harness is rerun with the actual checkpoint lineage. The experiment used software dequantization; it does not prove native 2-bit hardware behavior.

## Recommended experiment order

1. **Freeze a deploy slice.** Capture or render scene-disjoint views using the intended camera placements, table layouts, object scales, lighting variation, partial occlusion, arm intrusion, and empty/misplaced settings. Keep this slice out of training and model selection. Report per-class precision, recall, AP, confusion, duplicate detections, missed small objects, occlusion misses, and confidence calibration.

2. **Measure downstream utility.** Add detection-to-world mapping, stable identity, and placement/re-observation outcomes to the evaluation. A model can improve mAP while still failing to provide a safe grasp target. Track false actionable detections separately from harmless misses.

3. **Improve data before architecture.** Add high-quality deploy-view labels and controlled synthetic examples, with deliberate coverage for plate, napkin, fork, spoon, and knife. Use scene/frame-disjoint splits and verify that near-duplicate web images do not cross the split. Compare class-balanced sampling or loss weighting as an ablation rather than assuming it helps.

4. **Test small-object/domain hypotheses one at a time.** Compare the current baseline against higher inference resolution, an appropriate larger model, and tiling/cropping only if the deploy-slice error report shows small-object misses. Keep one variable changed per run and retain the complete manifest, configuration, checkpoint lineage, and fixed-subset results.

5. **Prove runtime parity.** Run the same fixed images through the best `.pt`, ONNX, and OpenVINO artifacts. Record box/class/confidence deltas and downstream metric deltas. A successful export is not equivalent to export parity.

6. **Add policy gates.** Calibrate thresholds per class and action context. Low-confidence or disagreeing observations must produce `REOBSERVE`/`NOT_ACTIONABLE`, not an implicit fallback to a guessed object. Preserve camera health, calibration epoch, timestamp alignment, uncertainty, and provenance with each observation.

7. **Reconcile, then revisit compression last.** First repair or rerun the inconsistent quantization receipt so model/checkpoint/training provenance is unambiguous. Quantization is worthwhile only if it preserves the deploy-slice and downstream acceptance metrics on the target runtime. Current evidence gives no reason to prioritize 2-bit work.

## Suggested acceptance gate

Do not promote a YOLO checkpoint for autonomous manipulation based on aggregate mAP50 alone. Require:

- frozen deploy-slice results for every actionable class;
- explicit handling for occlusion, camera disagreement, stale observations, and human intrusion;
- export parity for the selected runtime;
- a downstream grasp/placement or re-observation metric;
- no fabricated live status when weights, calibration, or runtime health are unavailable;
- a retained evidence bundle linking dataset manifest, code revision, checkpoint, export, evaluation configuration, and results.
- internally consistent checkpoint/training provenance in every evidence receipt; conflicting labels are a promotion blocker.

Until those gates exist, YOLO remains an observation source for WorldState and must not silently promote uncertain detections into authoritative object identity or safe action targets.
