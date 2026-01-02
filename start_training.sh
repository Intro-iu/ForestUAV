#!/bin/bash

# ForestUAV Training Entry Script (UV Version)

# Stop on error
set -e

echo "========================================"
echo "    ForestUAV Training Launcher (UV)"
echo "========================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' command not found. Please install uv first."
    echo "Curl: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 1. Environment Setup
echo "[1/3] Syncing environment..."
# Sync dependencies from uv.lock/pyproject.toml
uv sync

# 2. Data Preparation
echo "[2/3] Checking and downloading data..."
if [ ! -d "coco128" ]; then
    uv run download_coco128.py
else
    echo "Data directory 'coco128' found. Skipping download."
fi

# 3. Training
echo "[3/3] Starting YOLOv7 training..."
echo "Config: cfg/training/train.yaml"
echo "Data:   data/coco128.yaml"
echo "Device: 0 (GPU)"

# Force UTF-8 encoding
export PYTHONUTF8=1

# Run training with uv
uv run train.py \
    --weights "" \
    --cfg cfg/training/train.yaml \
    --data data/coco128.yaml \
    --epochs 300 \
    --batch-size 8 \
    --img 640 \
    --device 0 \
    --name server_train_run \
    --exist-ok

echo "========================================"
echo "Training finished!"
echo "Results saved to runs/train/server_train_run"
echo "========================================"
