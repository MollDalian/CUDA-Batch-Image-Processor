#!/bin/bash
# Build and run the CUDA batch image processing pipeline.
set -e

echo "=== CUDA Batch Image Processing Pipeline ==="
echo ""

echo "[1/3] Building..."
make clean 2>/dev/null || true
make
echo ""

echo "[2/3] Generating 100 test images (128x128 pixels)..."
python3 scripts/generate_test_data.py data 100 128
echo ""

echo "[3/3] Running image processor (full pipeline)..."
mkdir -p output
./image_processor --input data --output output --filter edge
echo ""

echo "=== Complete ==="
echo "Input images:  data/"
echo "Output images: output/"
