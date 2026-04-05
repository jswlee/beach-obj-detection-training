#!/bin/bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

# Install remaining dependencies
pip install -r requirements.txt
