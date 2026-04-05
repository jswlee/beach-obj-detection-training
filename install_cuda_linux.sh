#!/bin/bash
# Install PyTorch with CUDA 12.1 support for Ubuntu/Linux
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

# Install remaining dependencies
pip install -r requirements.txt
