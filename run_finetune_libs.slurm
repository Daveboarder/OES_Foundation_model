#!/bin/bash
#SBATCH --job-name=finetune_libs_binned
#SBATCH --output=/scratch/vrabel/self_supervised_libs_remote/slurm_finetune_libs_%j.out
#SBATCH --error=/scratch/vrabel/self_supervised_libs_remote/slurm_finetune_libs_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# Fine-tune the LIBS encoder on the binned-quantification task on an A100 80GB.
# Uses config/config_libs_a100.yaml + config/libs_data.yaml.
#
# Usage: sbatch run_finetune_libs.slurm <pretrain_run_dir>
#   e.g. sbatch run_finetune_libs.slurm runs/pretrain_2026-05-19_..._libs_pretrain_a100

set -x

WORKSPACE=/scratch/vrabel/self_supervised_libs_remote

PRETRAIN_RUN="${1:?Usage: sbatch run_finetune_libs.slurm <pretrain_run_dir>}"

enroot start --root \
  --mount="$WORKSPACE":/workspace \
  sslibs \
  -- bash -lc "
set -eux
cd /workspace
uv run python train_finetune.py \
    --config config/config_libs_a100.yaml \
    --pretrain_run_dir $PRETRAIN_RUN \
    --libs_data_config config/libs_data.yaml \
    --task quantification_binned \
    --pool cls_mean \
    --experiment_name libs_binned_q_a100 \
    --num_workers 4
"
