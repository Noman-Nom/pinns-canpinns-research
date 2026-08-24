#!/bin/bash
# Setup script for CAN-PINNs deep learning environment
# Compatible with Ubuntu OS and NVIDIA T2000 GPU

set -e  # Exit on error

echo "=========================================="
echo "CAN-PINNs Environment Setup"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

echo "Step 1: Checking NVIDIA GPU and drivers..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
    echo ""
fi

echo "Step 2: Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "Step 3: Activating environment and installing PyTorch with CUDA..."
conda activate pinns

# Install PyTorch with CUDA support via pip (conda environment)
# Using CUDA 11.8 which is widely compatible
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Step 4: Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Step 5: Verifying TensorFlow installation..."
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')"

echo ""
echo "=========================================="
echo "Environment setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate pinns"
echo ""
echo "To verify GPU setup, run:"
echo "  python verify_gpu.py"

