#!/bin/bash
#SBATCH --job-name=finetune_libs
#SBATCH --output=/scratch/vrabel/self_supervised_libs_remote/slurm_finetune_%j.out
#SBATCH --error=/scratch/vrabel/self_supervised_libs_remote/slurm_finetune_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

set -x

WORKSPACE=/scratch/vrabel/self_supervised_libs_remote

# Pass pretrain run dir as first argument: sbatch run_finetune.slurm runs/pretrain_2024-...
PRETRAIN_RUN="${1:?Usage: sbatch run_finetune.slurm <pretrain_run_dir>}"

enroot start --root \
  --mount="$WORKSPACE":/workspace \
  sslibs \
  -- bash -lc "
set -eux
cd /workspace
uv run python train_finetune.py \
    --config config/config_a100.yaml \
    --pretrain_run_dir $PRETRAIN_RUN \
    --task both \
    --num_workers 4
"
