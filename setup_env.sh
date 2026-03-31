#!/bin/bash
# Bash script to set up the environment with uv and CUDA support
# Run this script from the project root directory

set -e

echo -e "\033[0;36mSetting up LIBS Foundation Model environment with uv...\033[0m"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "\033[0;33muv not found. Installing uv...\033[0m"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to path for this session
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment
echo -e "\n\033[0;36mCreating virtual environment...\033[0m"
uv venv .venv --python 3.11

# Activate virtual environment
echo -e "\n\033[0;36mActivating virtual environment...\033[0m"
source .venv/bin/activate

# Install PyTorch with CUDA 12.4 (latest stable)
# Adjust cu124 to cu121 or cu118 depending on your CUDA version
echo -e "\n\033[0;36mInstalling PyTorch with CUDA support...\033[0m"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
echo -e "\n\033[0;36mInstalling project dependencies...\033[0m"
uv pip install -e ".[dev]"

# Verify CUDA is available
echo -e "\n\033[0;36mVerifying CUDA installation...\033[0m"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo -e "\n\033[0;32mSetup complete!\033[0m"
echo -e "\033[0;33mTo activate the environment, run: source .venv/bin/activate\033[0m"
