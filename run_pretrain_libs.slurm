#!/bin/bash
#SBATCH --job-name=pretrain_libs_real
#SBATCH --output=/scratch/vrabel/self_supervised_libs_remote/slurm_pretrain_libs_%j.out
#SBATCH --error=/scratch/vrabel/self_supervised_libs_remote/slurm_pretrain_libs_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# Pretrain on the real LIBS pipeline (17,428-bin sequences) on an A100 80GB.
# Uses config/config_libs_a100.yaml + config/libs_data.yaml.

set -x

WORKSPACE=/scratch/vrabel/self_supervised_libs_remote

enroot start --root \
  --mount="$WORKSPACE":/workspace \
  sslibs \
  -- bash -lc '
set -eux
cd /workspace
uv run python train_pretrain.py \
    --config config/config_libs_a100.yaml \
    --libs_data_config config/libs_data.yaml \
    --experiment_name libs_pretrain_a100 \
    --num_workers 4
'
