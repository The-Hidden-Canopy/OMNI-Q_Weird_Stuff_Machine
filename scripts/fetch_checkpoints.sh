#!/bin/bash
# Fetch the two training outputs the VLA / OMNI demo modes need (gated HF repo,
# access is auto-approved: log in once with `hf auth login`).
#
#   bash scripts/fetch_checkpoints.sh
#   export OMNIQ_VLA_CHECKPOINT=$PWD/models/hf_omni_q_table/smolvla_so101_table
#   export OMNIQ_OMNI_REASONER=omni OMNIQ_OMNI_CHECKPOINT=models/hf_omni_q_table/omni_planner/omni_planner_r1_final.pt \
#          OMNIQ_OMNI_RECEIPT=models/hf_omni_q_table/omni_planner/omni_planner_r1_final_receipt.json
#
# The governed mode, the perception loop (INT8 IR is in git) and the OpenVINO
# benchmark need nothing from here.
set -e
cd "$(dirname "$0")/.."
DEST=models/hf_omni_q_table
PY=${PYTHON:-.venv/Scripts/python}
[ -x "$PY" ] || PY=python
"$PY" - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("The-Hidden-Canopy/omni-q-table-checkpoints", local_dir="models/hf_omni_q_table")
print("checkpoints in", p)
EOF
echo "OMNIQ_VLA_CHECKPOINT=$PWD/$DEST/smolvla_so101_table"
echo "OMNIQ_OMNI_CHECKPOINT=$DEST/omni_planner/omni_planner_r1_final.pt  OMNIQ_OMNI_RECEIPT=$DEST/omni_planner/omni_planner_r1_final_receipt.json"
