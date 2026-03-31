#!/bin/bash
#SBATCH --job-name=eval_libs
#SBATCH --output=/scratch/vrabel/self_supervised_libs_remote/slurm_eval_%j.out
#SBATCH --error=/scratch/vrabel/self_supervised_libs_remote/slurm_eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -x

WORKSPACE=/scratch/vrabel/self_supervised_libs_remote

# Pass the run directory as first argument: sbatch run_eval.slurm runs/pretrain_2024-...
RUN_DIR="${1:?Usage: sbatch run_eval.slurm <run_dir>}"

enroot start --root \
  --mount="$WORKSPACE":/workspace \
  sslibs \
  -- bash -lc "
set -eux
cd /workspace
uv run python evaluate_model.py --run_dir $RUN_DIR
"
