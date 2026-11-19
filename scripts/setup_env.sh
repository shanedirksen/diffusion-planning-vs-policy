#!/usr/bin/env bash
# Setup Python environment for diffusion-planning-vs-policy
#
# Supports: venv (default), conda (pass --conda)
# Requires: Python 3.10+ and a CUDA-capable GPU (optional but recommended)
set -euo pipefail

USE_CONDA=false
for arg in "$@"; do
  case $arg in --conda) USE_CONDA=true ;; esac
done

if $USE_CONDA; then
  ENV_NAME="${1:-diffplan}"
  echo "=== Creating conda env: ${ENV_NAME} (Python 3.12) ==="
  conda create -n "${ENV_NAME}" python=3.12 -y
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}"
else
  echo "=== Creating venv ==="
  python -m venv .venv
  source .venv/bin/activate  # Linux/macOS
  # On Windows use: .\.venv\Scripts\Activate.ps1
fi

echo "=== Upgrading pip ==="
pip install -U pip

echo "=== Installing PyTorch (CUDA 12.8) ==="
echo "    Change the URL below to match your GPU / CUDA version."
echo "    See: https://pytorch.org/get-started/locally/"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "=== Installing project + dependencies ==="
pip install -r requirements.txt
pip install -e .

echo "=== Pre-downloading Minari datasets ==="
python scripts/download_datasets.py

echo ""
echo "Done!  Activate with:"
if $USE_CONDA; then
  echo "  conda activate ${ENV_NAME}"
else
  echo "  source .venv/bin/activate   (Linux/macOS)"
  echo "  .\.venv\Scripts\Activate.ps1  (Windows)"
fi
