#!/bin/bash --login
#SBATCH --job-name=subt-train
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=你的AccountString
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
module load miniforge/24.11.3-0
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
conda activate subt-2026

cd /scratch/user/uqmren2/projects/SubT-Insight-2026


srun python train.py \
  --use_tmds \
  --data_root dataset/tongji_data_awesome \
  --output_dir outputs/tmds_run \
  --head_channels 96 \
  --batch_size 16 \
  --num_workers 6 \
  --routing_loss_weight 0.5 \
