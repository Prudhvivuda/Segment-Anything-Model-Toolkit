#!/bin/bash
# Quick Start Training Script

echo "🚀 Starting Quick Training..."
echo ""

# Check if dataset exists
if [ ! -d "insurance_starter_dataset/images/train" ] || [ -z "$(ls -A insurance_starter_dataset/images/train)" ]; then
    echo "⚠️  Warning: Dataset is empty!"
    echo "   Please add images first:"
    echo "   1. Collect images"
    echo "   2. Annotate with SAM tool"
    echo "   3. Convert to YOLO format"
    echo ""
    exit 1
fi

# Train with small model for quick testing
python train_yolo.py \
    --mode train \
    --data-yaml insurance_data.yaml \
    --model yolov8n \
    --epochs 50 \
    --batch 8 \
    --device auto

echo ""
echo "✅ Training complete!"
echo "📁 Check results in: runs/detect/train/"
echo ""
