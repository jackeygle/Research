#!/bin/bash
# Setup Script for Action Recognition Environment
# ================================================

# Load mamba
module load mamba

# Create conda environment
echo "Creating conda environment: action_recog"
conda create -n action_recog python=3.10 -y

# Activate environment
source activate action_recog

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing dependencies..."
pip install tensorboard
pip install pyyaml
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install tqdm
pip install av  # For video decoding

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"

echo ""
echo "Environment setup complete!"
echo "Activate with: source activate action_recog"
