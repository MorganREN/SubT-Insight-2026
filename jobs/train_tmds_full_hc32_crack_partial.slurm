#!/bin/bash --login
#SBATCH --job-name=subt-tmds-full-hc32-crack
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --account=你的AccountString
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# TMDS Full, head_channels=32.
# Main supervision: fully labeled Tongji patches.
# Extra supervision:
#   1) Roboflow Tongji raw labels (spalling -> segment_damage, leakage -> 255 ignore)
#   2) Roboflow crack partial labels (crack=1, unlabeled=255 ignore)
#
# Submit:
#   sbatch jobs/train_tmds_full_hc32_crack_partial.slurm
#
# Common overrides:
#   WORK_DIR=/path/to/SubT-Insight-2026 sbatch jobs/train_tmds_full_hc32_crack_partial.slurm
#   RUN_NAME=my_run BATCH_SIZE=2 USE_TTA=0 sbatch jobs/train_tmds_full_hc32_crack_partial.slurm

RUN_NAME="${RUN_NAME:-ablation_tmds_full_hc32_crack_partial}"
WORK_DIR="${WORK_DIR:-$PWD}"

DATA_PATCH="${DATA_PATCH:-dataset/tongji_data_awesome}"
DATA_RAW="${DATA_RAW:-dataset/tongji_data_raw}"
CRACK_PARTIAL_DATA="${CRACK_PARTIAL_DATA:-dataset/roboflow_tunnel_crack_partial_raw}"
ROBOFLOW_TONGJI_DATA="${ROBOFLOW_TONGJI_DATA:-dataset/roboflow_tongji_tunnel_raw_spalling_segment_leakage_ignore}"

HEAD_CHANNELS="${HEAD_CHANNELS:-32}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-6}"
ROUTING_LOSS="${ROUTING_LOSS:-0.2}"
RARE_WEIGHTS="${RARE_WEIGHTS:-1:3.0,3:2.0}"

INFER_SPLIT="${INFER_SPLIT:-valid}"
PRED_SPLITS="${PRED_SPLITS:-valid}"
USE_TTA="${USE_TTA:-1}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-2}"

module purge
module load miniforge/24.11.3-0
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
conda activate subt-2026

cd "$WORK_DIR"
mkdir -p logs

OUTPUT_DIR="outputs/${RUN_NAME}"
CKPT="${OUTPUT_DIR}/best.pth"

echo "======================================================"
echo "  Experiment : TMDS Full + HC32 + Crack Partial"
echo "  SLURM Job  : ${SLURM_JOB_ID:-local}"
echo "  Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "  Run name   : $RUN_NAME"
echo "  Work dir   : $WORK_DIR"
echo "  Output dir : $OUTPUT_DIR"
echo "======================================================"

echo ""
echo ">>> [Check] Dataset / weight paths"
test -d "$DATA_PATCH/img_dir/train"
test -d "$DATA_PATCH/ann_dir/train"
test -d "$DATA_PATCH/img_dir/valid"
test -d "$DATA_PATCH/ann_dir/valid"
test -d "$DATA_RAW/img_dir/$INFER_SPLIT"
test -d "$DATA_RAW/ann_dir/$INFER_SPLIT"
test -d "$ROBOFLOW_TONGJI_DATA/img_dir/train"
test -d "$ROBOFLOW_TONGJI_DATA/ann_dir/train"
test -d "$CRACK_PARTIAL_DATA/img_dir/train"
test -d "$CRACK_PARTIAL_DATA/ann_dir/train"
test -f "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"

TRAIN_CMD=(
  python scripts/train.py
  --use_tmds
  --data_root "$DATA_PATCH"
  --extra_data_roots "$ROBOFLOW_TONGJI_DATA" "$CRACK_PARTIAL_DATA"
  --output_dir "$OUTPUT_DIR"
  --head_channels "$HEAD_CHANNELS"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --routing_loss_weight "$ROUTING_LOSS"
  --use_cmim
  --device cuda
)

if [ -n "$RARE_WEIGHTS" ]; then
  TRAIN_CMD+=(--rare_class_weights "$RARE_WEIGHTS")
fi

echo ""
echo ">>> [Step 1/5] Training start  $(date '+%Y-%m-%d %H:%M:%S')"
echo "${TRAIN_CMD[@]}"
srun "${TRAIN_CMD[@]}"

if [ ! -f "$CKPT" ]; then
  echo "!!! Missing checkpoint: $CKPT"
  exit 1
fi
echo ">>> [Step 1/5] Training done   $(date '+%Y-%m-%d %H:%M:%S')"

INFER_CMD=(
  python scripts/infer.py
  --run raw
  --ckpt "$CKPT"
  --data_root "$DATA_RAW"
  --split "$INFER_SPLIT"
  --output_dir "${OUTPUT_DIR}/infer_raw"
  --num_workers 4
  --use_tiling
)

if [ "$USE_TTA" -eq 1 ]; then
  INFER_CMD+=(--use_tta)
fi

echo ""
echo ">>> [Step 2/5] Tongji raw validation start  $(date '+%Y-%m-%d %H:%M:%S')"
echo "${INFER_CMD[@]}"
"${INFER_CMD[@]}"
echo ">>> [Step 2/5] Tongji raw validation done   $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo ">>> [Step 3/5] Tongji raw full-image prediction start  $(date '+%Y-%m-%d %H:%M:%S')"
python scripts/predict_dataset.py \
  --ckpt "$CKPT" \
  --img_root "${DATA_RAW}/img_dir" \
  --splits $PRED_SPLITS \
  --output_dir "${OUTPUT_DIR}/predict_raw"
echo ">>> [Step 3/5] Tongji raw full-image prediction done   $(date '+%Y-%m-%d %H:%M:%S')"

if [ -d "$ROBOFLOW_TONGJI_DATA/img_dir/train" ]; then
  echo ""
  echo ">>> [Step 4/5] Roboflow Tongji merged-label eval start  $(date '+%Y-%m-%d %H:%M:%S')"
  python -u scripts/infer_roboflow_small_models_merged.py \
    --ckpt "$CKPT" \
    --model_name "$RUN_NAME" \
    --data_root "$ROBOFLOW_TONGJI_DATA" \
    --split train \
    --output_root "${OUTPUT_DIR}/infer_roboflow_tongji" \
    --device cuda \
    --patch_batch_size "$PATCH_BATCH_SIZE"
  echo ">>> [Step 4/5] Roboflow Tongji merged-label eval done   $(date '+%Y-%m-%d %H:%M:%S')"
else
  echo ""
  echo ">>> [Step 4/5] Skip Roboflow Tongji eval: $ROBOFLOW_TONGJI_DATA not found"
fi

echo ""
echo ">>> [Step 5/5] Roboflow crack partial eval start  $(date '+%Y-%m-%d %H:%M:%S')"
python -u scripts/infer_roboflow_crack_partial_four_models.py \
  --ckpt "$CKPT" \
  --model_name "$RUN_NAME" \
  --data_root "$CRACK_PARTIAL_DATA" \
  --split train \
  --output_root "${OUTPUT_DIR}/infer_roboflow_crack_partial" \
  --device cuda \
  --patch_batch_size "$PATCH_BATCH_SIZE"
echo ">>> [Step 5/5] Roboflow crack partial eval done   $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "======================================================"
echo "  All done"
echo "  Best ckpt: $CKPT"
echo "======================================================"
