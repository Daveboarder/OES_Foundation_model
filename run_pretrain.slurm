#!/bin/bash
#SBATCH --job-name=pretrain_libs
#SBATCH --output=/scratch/vrabel/self_supervised_libs_remote/slurm_pretrain_%j.out
#SBATCH --error=/scratch/vrabel/self_supervised_libs_remote/slurm_pretrain_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

set -x

WORKSPACE=/scratch/vrabel/self_supervised_libs_remote

enroot start --root \
  --mount="$WORKSPACE":/workspace \
  sslibs \
  -- bash -lc '
set -eux
cd /workspace
uv run python train_pretrain.py \
    --config config/config_a100.yaml \
    --experiment_name a100_v1 \
    --num_workers 4
'
