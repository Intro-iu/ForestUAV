#!/bin/bash

# ForestUAV Inference Entry Script

# Stop on error
set -e

echo "========================================"
echo "    ForestUAV Inference Launcher"
echo "========================================"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' command not found."
    exit 1
fi

# Configuration
# Path to the trained weights (from start_training.sh)
WEIGHTS="runs/train/server_train_run/weights/best.pt"
# Source images to detect
SOURCE="coco128/images/train2017" # Run on all 128 images
# Output name
NAME="server_inference"

echo "Weights: $WEIGHTS"
echo "Source:  $SOURCE"
echo "Device:  0 (GPU)"

# Check if weights exist
if [ ! -f "$WEIGHTS" ]; then
    echo "WARNING: Weights file not found at $WEIGHTS"
    echo "If training is not complete, this will fail."
    echo "You can edit this script to use 'yolov7.pt' for testing."
fi

# Force UTF-8
export PYTHONUTF8=1

# Run inference
uv run detect.py \
    --weights "$WEIGHTS" \
    --source "$SOURCE" \
    --device 0 \
    --conf 0.25 \
    --name "$NAME" \
    --exist-ok

echo "========================================"
echo "Inference finished!"
echo "Check results in: runs/detect/$NAME"
echo "========================================"
