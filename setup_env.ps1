# PowerShell script to set up the environment with uv and CUDA support
# Run this script from the project root directory

$ErrorActionPreference = "Stop"

Write-Host "Setting up LIBS Foundation Model environment with uv..." -ForegroundColor Cyan

# Check if uv is installed
$uvPath = "$env:USERPROFILE\.local\bin"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Installing uv..." -ForegroundColor Yellow
    irm https://astral.sh/uv/install.ps1 | iex
    
    # Add uv to PATH for this session
    $env:Path = "$uvPath;$env:Path"
    Write-Host "Added uv to PATH for this session" -ForegroundColor Green
}

# Verify uv is now available
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: uv still not found. Please restart your shell and run the script again." -ForegroundColor Red
    exit 1
}

Write-Host "uv version: $(uv --version)" -ForegroundColor Green

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Cyan
uv venv .venv --python 3.11

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.4 (latest stable)
# Adjust cu124 to cu121 or cu118 depending on your CUDA version
Write-Host "`nInstalling PyTorch with CUDA support..." -ForegroundColor Cyan
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
Write-Host "`nInstalling project dependencies..." -ForegroundColor Cyan
uv pip install -e ".[dev]"

# Verify CUDA is available
Write-Host "`nVerifying CUDA installation..." -ForegroundColor Cyan
python -c @"
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA version: N/A')
    print('GPU: N/A')
"@

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "To activate the environment in new terminals, run: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
