#!/bin/bash

# ForestUAV Training Entry Script

# Stop on error
set -e

echo "========================================"
echo "    ForestUAV Training Launcher"
echo "========================================"

# 1. Environment Setup
echo "[1/3] Installing dependencies..."
# Update pip just in case
pip install --upgrade pip
# Install requirements
pip install -r requirements.txt

# 2. Data Preparation
echo "[2/3] Checking and downloading data..."
if [ ! -d "coco128" ]; then
    python download_coco128.py
else
    echo "Data directory 'coco128' found. Skipping download."
fi

# 3. Training
echo "[3/3] Starting YOLOv7 training..."
echo "Config: cfg/training/train.yaml"
echo "Data:   data/coco128.yaml"
echo "Device: 0 (GPU)"

# Force UTF-8 encoding to avoid python print errors
export PYTHONUTF8=1

# Run training
# Note: Adjusted batch-size to 16 for server (assuming better GPU), change if needed.
python train.py \
    --weights "" \
    --cfg cfg/training/train.yaml \
    --data data/coco128.yaml \
    --epochs 300 \
    --batch-size 16 \
    --img 640 \
    --device 0 \
    --name server_train_run \
    --exist-ok

echo "========================================"
echo "Training finished!"
echo "Results saved to runs/train/server_train_run"
echo "========================================"
