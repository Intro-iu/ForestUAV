#!/bin/bash

set -e

MODE="${1:-custom}"
SOURCE="${2:-datasets/M4SFWD/images/test}"

echo "========================================"
echo " ForestUAV Inference Launcher"
echo " Mode: ${MODE}"
echo " Source: ${SOURCE}"
echo "========================================"

if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' command not found."
    exit 1
fi

export PYTHONUTF8=1

uv run python scripts/run_experiment.py infer \
    --mode "${MODE}" \
    --source "${SOURCE}" \
    --device 0 \
    --exist-ok

echo "========================================"
echo "Inference finished."
echo "========================================"
