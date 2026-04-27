#!/bin/bash --login
#SBATCH --job-name=subt-train-c3
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --account=a_civil_eng
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# 可修改区域
RUN_NAME="tmds_run"
DATA_PATCH="dataset/tongji_data_awesome"
DATA_RAW="dataset/tongji_data_raw"
WORK_DIR="${WORK_DIR:-$PWD}"

HEAD_CHANNELS=96
BATCH_SIZE=32
NUM_WORKERS=6
ROUTING_LOSS=0.6
RARE_WEIGHTS="1:3.0,3:2.0"          # 例如 "1:3.0,3:2.0" 表示类别 1 的权重是 3.0，类别 3 的权重是 2.0
INFER_SPLIT="valid"
PRED_SPLITS="valid"      # 多个 split 可写成 "valid test"
USE_TTA=1                # 1=Step 2 开启 TTA，0=关闭

module purge
module load miniforge/24.11.3-0
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
conda activate subt-2026

cd "$WORK_DIR"

echo "======================================================"
echo "  SLURM Job   : $SLURM_JOB_ID"
echo "  Node        : $SLURMD_NODENAME"
echo "  Run name    : $RUN_NAME"
echo "  Work dir    : $WORK_DIR"
echo "======================================================"

OUTPUT_DIR="outputs/${RUN_NAME}"
CKPT="${OUTPUT_DIR}/best.pth"

TRAIN_CMD=(
  python train.py
  --use_tmds
  --data_root "$DATA_PATCH"
  --output_dir "$OUTPUT_DIR"
  --head_channels "$HEAD_CHANNELS"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --routing_loss_weight "$ROUTING_LOSS"
)

if [ -n "$RARE_WEIGHTS" ]; then
  TRAIN_CMD+=(--rare_class_weights "$RARE_WEIGHTS")
fi

echo ""
echo ">>> [Step 1/3] 训练开始  $(date '+%Y-%m-%d %H:%M:%S')"
srun "${TRAIN_CMD[@]}"

TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "!!! 训练失败 (exit=$TRAIN_RC)"
  exit $TRAIN_RC
fi
if [ ! -f "$CKPT" ]; then
  echo "!!! 未找到 checkpoint: $CKPT"
  exit 1
fi
echo ">>> [Step 1/3] 训练完成  $(date '+%Y-%m-%d %H:%M:%S')"

INFER_CMD=(
  python infer.py
  --run raw
  --ckpt "$CKPT"
  --data_root "$DATA_RAW"
  --split "$INFER_SPLIT"
  --output_dir "${OUTPUT_DIR}/infer_raw"
  --num_workers 4
  --use_tiling
)

if [ "${USE_TTA:-0}" -eq 1 ]; then
  INFER_CMD+=(--use_tta)
fi

echo ""
echo ">>> [Step 2/3] 推理评估开始  $(date '+%Y-%m-%d %H:%M:%S')"
"${INFER_CMD[@]}"
INFER_RC=$?
if [ $INFER_RC -ne 0 ]; then
  echo "!!! 推理评估失败 (exit=$INFER_RC)"
fi
echo ">>> [Step 2/3] 推理评估完成  $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo ">>> [Step 3/3] 全图推理开始  $(date '+%Y-%m-%d %H:%M:%S')"
python predict_dataset.py \
  --ckpt "$CKPT" \
  --img_root "${DATA_RAW}/img_dir" \
  --splits $PRED_SPLITS \
  --output_dir "${OUTPUT_DIR}/predict_raw"

PRED_RC=$?
if [ $PRED_RC -ne 0 ]; then
  echo "!!! 全图推理失败 (exit=$PRED_RC)"
fi
echo ">>> [Step 3/3] 全图推理完成  $(date '+%Y-%m-%d %H:%M:%S')"
