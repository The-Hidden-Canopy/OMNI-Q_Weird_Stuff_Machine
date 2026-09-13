#!/usr/bin/env bash
set -Eeuo pipefail

CURRENT_PID=21150
RUN_DIR=/workspace/omniq/runs/v4_chain/ftv4_combined_v1_b128_fused_120ep_r1
RESULTS_CSV="${RUN_DIR}/results.csv"
LAST_PT="${RUN_DIR}/weights/last.pt"
BASE_MODEL=/workspace/omniq/yolov8s.pt

echo "waiting for ftv4_combined_v1_b128_fused_120ep_r1 pid=${CURRENT_PID}"
while kill -0 "${CURRENT_PID}" 2>/dev/null; do
    sleep 30
done

if ! awk -F, 'NR > 1 && $1 ~ /^[0-9]+$/ { last = $1 } END { exit !(last >= 120) }' "${RESULTS_CSV}"; then
    echo "refusing continuation: current run did not complete epoch 120" >&2
    exit 1
fi
if [[ ! -s "${LAST_PT}" ]]; then
    echo "refusing continuation: missing ${LAST_PT}" >&2
    exit 1
fi

echo "current run completed; starting 60-epoch YOLOv8s run from ${BASE_MODEL}"
cd /workspace/omniq
export TORCH_CUDA_ARCH_LIST=8.9
export OMNIQ_REQUIRE_NATIVE_FUSION=1
exec /venv/main/bin/python -u perception/finetune.py \
    --data /workspace/omniq/data/table_yolo_combined_v1/data.yaml \
    --base "${BASE_MODEL}" \
    --w-master mxfp8 \
    --epochs 60 \
    --freeze 0 \
    --imgsz 768 \
    --batch 64 \
    --lr0 3e-4 \
    --nbs 128 \
    --patience 100 \
    --workers 8 \
    --project /workspace/omniq/runs/v4_chain \
    --name ftv4_combined_v1_yolov8s_b64_fused_60ep_r1
