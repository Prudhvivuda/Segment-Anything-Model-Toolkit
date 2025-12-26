#!/bin/bash
# Production Training Script

echo "🚀 Starting Production Training..."
echo ""

python train_yolo.py \
    --mode train \
    --data-yaml insurance_data.yaml \
    --model yolov8x \
    --epochs 100 \
    --batch 16 \
    --device auto

echo ""
echo "✅ Production training complete!"
echo ""
