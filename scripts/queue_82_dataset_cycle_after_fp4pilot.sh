#!/usr/bin/env bash
set -Eeuo pipefail

# Serial 82-class transfer cycle.  Stage A is the live packed-FP4 pilot; do
# not start Stage B until Stage A has finished so one GPU never hosts two
# trainers.  The later stages use last.pt as the explicit weight handoff and
# start fresh optimizer state at each dataset boundary.

CURRENT_PID=41562
PROJECT=/workspace/omniq/runs/v4_chain
CURRENT_DATA_ROOT=/workspace/omniq/data/table_yolo_combined_oi_v1
OTHER_DATA_ROOT=/workspace/omniq/data/table_yolo_combined_v1

STAGE_A_RUN="${PROJECT}/ftv5_combined_oi_v1_yolov8m_b48_fp4pilot_from20_r1"
STAGE_B_NAME=ftv5_combined_v1_yolov8m_b48_fp4pilot_from_oi60_r1
STAGE_B_RUN="${PROJECT}/${STAGE_B_NAME}"
STAGE_C_NAME=ftv5_combined_oi_v1_yolov8m_b48_fp4pilot_reconsolidate_r1
STAGE_C_RUN="${PROJECT}/${STAGE_C_NAME}"
MIN_TRANSFER_EPOCHS=30
MAX_TRANSFER_EPOCHS=60

last_epoch() {
    local csv="$1"
    awk -F, 'NR > 1 && $1 ~ /^[0-9]+$/ { last = $1 } END { print last + 0 }' "${csv}"
}

wait_for_stage_a() {
    local results="${STAGE_A_RUN}/results.csv"
    echo "waiting for Stage A ${STAGE_A_RUN} to finish at epoch ${MAX_TRANSFER_EPOCHS}"
    while true; do
        local epoch
        epoch="$(last_epoch "${results}")"
        if (( epoch >= MAX_TRANSFER_EPOCHS )); then
            break
        fi
        if ! kill -0 "${CURRENT_PID}" 2>/dev/null; then
            echo "refusing Stage B: Stage A stopped at epoch ${epoch}" >&2
            exit 1
        fi
        sleep 30
    done
    while kill -0 "${CURRENT_PID}" 2>/dev/null; do
        sleep 30
    done
    if [[ ! -s "${STAGE_A_RUN}/weights/last.pt" ]]; then
        echo "refusing Stage B: missing ${STAGE_A_RUN}/weights/last.pt" >&2
        exit 1
    fi
}

check_transfer_stage() {
    local run="$1"
    local epoch
    epoch="$(last_epoch "${run}/results.csv")"
    if (( epoch < MIN_TRANSFER_EPOCHS )); then
        echo "refusing next stage: ${run} stopped at epoch ${epoch}; minimum is ${MIN_TRANSFER_EPOCHS}" >&2
        exit 1
    fi
    if [[ ! -s "${run}/weights/last.pt" ]]; then
        echo "refusing next stage: missing ${run}/weights/last.pt" >&2
        exit 1
    fi
    echo "completed ${run} at epoch ${epoch}"
}

run_stage() {
    local data_root="$1"
    local data_yaml="${data_root}/data.yaml"
    local base_pt="$2"
    local name="$3"
    local lr0="$4"
    cd "${data_root}"
    export TORCH_CUDA_ARCH_LIST=8.9
    export OMNIQ_REQUIRE_NATIVE_FUSION=1
    /venv/main/bin/python -u /workspace/omniq/perception/finetune.py \
        --data "${data_yaml}" \
        --base "${base_pt}" \
        --w-master mxfp8 \
        --packed-compute mxfp4 \
        --packed-compute-pattern '^model\.(2|4|6|8|12|15|18|21)\.' \
        --epochs "${MAX_TRANSFER_EPOCHS}" \
        --freeze 0 \
        --imgsz 768 \
        --batch 48 \
        --lr0 "${lr0}" \
        --nbs 128 \
        --patience 30 \
        --workers 8 \
        --project "${PROJECT}" \
        --name "${name}"
}

wait_for_stage_a

if [[ -s "${STAGE_B_RUN}/results.csv" || -e "${STAGE_B_RUN}/weights/last.pt" ]]; then
    echo "refusing Stage B: existing output would be overwritten: ${STAGE_B_RUN}" >&2
    exit 1
fi
echo "starting Stage B on ${OTHER_DATA_ROOT} from ${STAGE_A_RUN}/weights/last.pt"
run_stage "${OTHER_DATA_ROOT}" "${STAGE_A_RUN}/weights/last.pt" "${STAGE_B_NAME}" 0.0001
check_transfer_stage "${STAGE_B_RUN}"

if [[ -s "${STAGE_C_RUN}/results.csv" || -e "${STAGE_C_RUN}/weights/last.pt" ]]; then
    echo "refusing Stage C: existing output would be overwritten: ${STAGE_C_RUN}" >&2
    exit 1
fi
echo "starting Stage C on ${CURRENT_DATA_ROOT} from ${STAGE_B_RUN}/weights/last.pt"
run_stage "${CURRENT_DATA_ROOT}" "${STAGE_B_RUN}/weights/last.pt" "${STAGE_C_NAME}" 0.00005
check_transfer_stage "${STAGE_C_RUN}"

cd /workspace/omniq
/venv/main/bin/python -u /workspace/omniq/scripts/evaluate_82_dataset_cycle.py \
    --model "${STAGE_C_RUN}/weights/best.pt" \
    --data "${CURRENT_DATA_ROOT}/data.yaml" \
    --data "${OTHER_DATA_ROOT}/data.yaml" \
    --imgsz 768 \
    --device 0 \
    --out "${PROJECT}/ftv5_82_dataset_cycle_evaluation.json"
