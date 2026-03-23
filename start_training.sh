#!/bin/bash

set -e

MODE="${1:-custom}"

echo "========================================"
echo " ForestUAV Training Launcher"
echo " Mode: ${MODE}"
echo " Dataset: M4SFWD"
echo "========================================"

if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' command not found. Please install uv first."
    exit 1
fi

export PYTHONUTF8=1

uv sync
uv run python scripts/run_experiment.py train \
    --mode "${MODE}" \
    --data data/m4sfwd.yaml \
    --device 0 \
    --epochs 300 \
    --img-size 640 \
    --exist-ok

echo "========================================"
echo "Training finished."
echo "========================================"
