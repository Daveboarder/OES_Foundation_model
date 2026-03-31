# Fix environment - run from project root

# Add uv to PATH if not already there
$uvPath = "$env:USERPROFILE\.local\bin"
if ($env:Path -notlike "*$uvPath*") {
    $env:Path = "$uvPath;$env:Path"
    Write-Host "Added uv to PATH" -ForegroundColor Yellow
}

Write-Host "Creating fresh venv and syncing deps (this takes a few minutes)..." -ForegroundColor Cyan
uv sync --reinstall

Write-Host "`nTesting imports..." -ForegroundColor Cyan
uv run python -c "import torch; import pytorch_lightning as pl; print(f'torch={torch.__version__}, lightning={pl.__version__}, CUDA={torch.cuda.is_available()}')"

Write-Host "`nDone! Now you can use:" -ForegroundColor Green
Write-Host "  uv run python train_pretrain.py --config config/config.yaml" -ForegroundColor White
