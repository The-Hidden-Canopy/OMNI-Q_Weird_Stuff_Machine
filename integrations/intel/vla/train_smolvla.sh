#!/bin/bash
# Fine-tune SmolVLA on the expert demonstrations (RTX 4050, 6 GB).
# usage: bash integrations/intel/vla/train_smolvla.sh [steps] [batch]
# SMOLVLA_BASE: local snapshot dir of lerobot/smolvla_base (hub ids break on Windows paths)
cd "$(dirname "$0")/../../.."
STEPS=${1:-3000}; BATCH=${2:-4}
BASE="${SMOLVLA_BASE:-C:/Users/damio/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205}"
OUT=outputs/train/smolvla_so101_table_$(date +%Y%m%d_%H%M)
RENAME='{"observation.images.overhead": "observation.images.camera1", "observation.images.front": "observation.images.camera2", "observation.images.wrist": "observation.images.camera3"}'
mkdir -p outputs/train
.venv/Scripts/python integrations/intel/vla/train_smolvla.py \
  --policy.path="$BASE" \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=omni-q/so101_table_vla \
  --dataset.root=datasets/so101_table_vla \
  --dataset.video_backend=pyav \
  --rename_map="$RENAME" \
  --output_dir="$OUT" \
  --job_name=smolvla_so101_table \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq=500 \
  --log_freq=50 \
  --num_workers=2 \
  --wandb.enable=false
echo "checkpoints under $OUT/checkpoints"
