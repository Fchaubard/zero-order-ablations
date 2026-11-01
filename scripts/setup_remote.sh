#!/usr/bin/env bash
# One-shot setup for a fresh Ubuntu + CUDA PyTorch docker image (e.g. vast.ai).
#   bash scripts/setup_remote.sh
# Installs git + tmux, the Python requirements, prefetches every dataset and prints the GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi

echo "== apt: git tmux"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends git tmux

echo "== pip: requirements.txt"
$PY -m pip install --upgrade pip >/dev/null
$PY -m pip install -r requirements.txt

echo "== datasets: MNIST, CIFAR-10, PTB -> ./data"
$PY -m zoo.data --root ./data --download-all

echo "== GPUs"
$PY - <<'PYEOF'
import torch
n = torch.cuda.device_count()
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}, device_count: {n}")
for i in range(n):
    print(f"  gpu{i}: {torch.cuda.get_device_name(i)}")
PYEOF

echo "== models"
$PY -m zoo.models
echo "setup done. Next: export WANDB_API_KEY=...; bash scripts/launch_sweep.sh"
