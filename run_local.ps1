# ============================================================
# LIBS Foundation Model - Local Run Script (4060 Ti)
# ============================================================
# Runs the full pipeline: pretrain -> evaluate -> finetune -> evaluate
#
# Usage:
#   .\run_local.ps1
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  LIBS Foundation Model - Local Pipeline (4060 Ti)"         -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$CONFIG = "config/config_local.yaml"
$EXPERIMENT = "local_4060ti"

# ------------------------------------------------------------------
# Step 1: Pre-training (self-supervised MIP)
# ------------------------------------------------------------------
Write-Host "[Step 1/4] Pre-training with masked intensity prediction..." -ForegroundColor Yellow
Write-Host ""

uv run python train_pretrain.py `
    --config $CONFIG `
    --experiment_name $EXPERIMENT `
    --save_data `
    --num_workers 0

if ($LASTEXITCODE -ne 0) {
    Write-Host "Pre-training failed!" -ForegroundColor Red
    exit 1
}

# Find the pretrain run directory (most recent)
$PRETRAIN_RUN = Get-ChildItem -Path "runs" -Directory -Filter "pretrain_*$EXPERIMENT*" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName

if (-not $PRETRAIN_RUN) {
    Write-Host "Could not find pretrain run directory!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Pretrain run: $PRETRAIN_RUN" -ForegroundColor Green
Write-Host ""

# ------------------------------------------------------------------
# Step 2: Evaluate pre-trained model
# ------------------------------------------------------------------
Write-Host "[Step 2/4] Evaluating pre-trained model..." -ForegroundColor Yellow
Write-Host ""

uv run python evaluate_model.py --run_dir $PRETRAIN_RUN

if ($LASTEXITCODE -ne 0) {
    Write-Host "Pretrain evaluation failed!" -ForegroundColor Red
    exit 1
}

# ------------------------------------------------------------------
# Step 3: Fine-tuning (supervised, both tasks)
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[Step 3/4] Fine-tuning on labeled data (classification + regression)..." -ForegroundColor Yellow
Write-Host ""

uv run python train_finetune.py `
    --config $CONFIG `
    --pretrain_run_dir $PRETRAIN_RUN `
    --task both `
    --experiment_name $EXPERIMENT `
    --num_workers 0

if ($LASTEXITCODE -ne 0) {
    Write-Host "Fine-tuning failed!" -ForegroundColor Red
    exit 1
}

# Find the finetune run directory
$FINETUNE_RUN = Get-ChildItem -Path "runs" -Directory -Filter "finetune_*$EXPERIMENT*" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName

if (-not $FINETUNE_RUN) {
    Write-Host "Could not find finetune run directory!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Finetune run: $FINETUNE_RUN" -ForegroundColor Green
Write-Host ""

# ------------------------------------------------------------------
# Step 4: Evaluate fine-tuned model
# ------------------------------------------------------------------
Write-Host "[Step 4/4] Evaluating fine-tuned model..." -ForegroundColor Yellow
Write-Host ""

uv run python evaluate_model.py --run_dir $FINETUNE_RUN

if ($LASTEXITCODE -ne 0) {
    Write-Host "Finetune evaluation failed!" -ForegroundColor Red
    exit 1
}

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Pipeline complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results:" -ForegroundColor Cyan
Write-Host "  Pretrain:  $PRETRAIN_RUN\evaluation\"
Write-Host "  Finetune:  $FINETUNE_RUN\evaluation\"
Write-Host ""
Write-Host "TensorBoard:" -ForegroundColor Cyan
Write-Host "  uv run tensorboard --logdir runs/"
Write-Host ""
