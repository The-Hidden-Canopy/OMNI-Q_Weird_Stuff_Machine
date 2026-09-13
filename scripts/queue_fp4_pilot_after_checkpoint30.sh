#!/usr/bin/env bash
set -Eeuo pipefail

# Start the selected-layer Hopper-style packed-FP4 consumer pilot from the
# live combined 82-class run's checkpoint-20 weights.  The active run is not
# interrupted and its checkpoint remains the rollback/control artifact.

CURRENT_PID=39526
CURRENT_RUN=/workspace/omniq/runs/v4_chain/ftv5_combined_oi_v1_yolov8m_b48_fused_120ep_r1
CURRENT_RESULTS="${CURRENT_RUN}/results.csv"
BASE_PT="${CURRENT_RUN}/weights/last.pt"
PROJECT=/workspace/omniq/runs/v4_chain
PILOT_NAME=ftv5_combined_oi_v1_yolov8m_b48_fp4pilot_from20_r1
PILOT_RUN="${PROJECT}/${PILOT_NAME}"
TARGET_EPOCH=20

echo "waiting for ${CURRENT_RUN} to reach checkpoint ${TARGET_EPOCH}"
while true; do
    last_epoch="$(awk -F, 'NR > 1 && $1 ~ /^[0-9]+$/ { last = $1 } END { print last + 0 }' "${CURRENT_RESULTS}")"
    if (( last_epoch >= TARGET_EPOCH )); then
        break
    fi
    if ! kill -0 "${CURRENT_PID}" 2>/dev/null; then
        echo "refusing pilot: source run stopped before checkpoint ${TARGET_EPOCH}" >&2
        exit 1
    fi
    sleep 30
done

if [[ ! -s "${BASE_PT}" ]]; then
    echo "refusing pilot: checkpoint is missing or empty: ${BASE_PT}" >&2
    exit 1
fi

# Do not consume the checkpoint while Ultralytics is still writing it.  Two
# identical non-zero size observations make the handoff boundary explicit;
# the source run continues independently after this point.
previous_size=0
stable_observations=0
while (( stable_observations < 2 )); do
    current_size="$(stat -c '%s' "${BASE_PT}")"
    if [[ "${current_size}" == "${previous_size}" && "${current_size}" -gt 0 ]]; then
        stable_observations=$((stable_observations + 1))
    else
        stable_observations=0
        previous_size="${current_size}"
    fi
    sleep 10
done

# Avoid consuming a second GPU slot if this queue script is restarted after a
# successful handoff.  Do not overwrite an existing pilot run.
if [[ -s "${PILOT_RUN}/results.csv" || -e "${PILOT_RUN}/weights/last.pt" ]]; then
    echo "pilot already exists: ${PILOT_RUN}" >&2
    exit 1
fi

# The checkpoint is used as a weight parent, not as --resume: the pilot keeps
# MXFP8 + BF16 Lion as its authoritative optimizer state and starts a fresh
# optimizer state at the controlled format boundary.
cd /workspace/omniq
export TORCH_CUDA_ARCH_LIST=8.9
export OMNIQ_REQUIRE_NATIVE_FUSION=1
exec /venv/main/bin/python -u perception/finetune.py \
    --data /workspace/omniq/data/table_yolo_combined_oi_v1/data.yaml \
    --base "${BASE_PT}" \
    --w-master mxfp8 \
    --packed-compute mxfp4 \
    --packed-compute-pattern '^model\.(2|4|6|8|12|15|18|21)\.' \
    --epochs 60 \
    --freeze 0 \
    --imgsz 768 \
    --batch 48 \
    --lr0 0.00015 \
    --nbs 128 \
    --patience 30 \
    --workers 8 \
    --project "${PROJECT}" \
    --name "${PILOT_NAME}"
