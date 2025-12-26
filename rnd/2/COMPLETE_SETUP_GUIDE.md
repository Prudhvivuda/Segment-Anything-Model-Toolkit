# 🏠 Insurance Claims Detection - Complete Setup Guide

## ✅ Setup Complete! Everything is Ready

### 📁 Project Structure
```
Segment-Anything-Model-Toolkit/
├── Insurance_Priority_Classes.csv       # 116 insurance classes
├── insurance_data.yaml                   # Training configuration (READY)
├── insurance_starter_dataset/            # Dataset directory (EMPTY - add your data)
├── train_yolo.py                         # Training script
├── download_and_prepare_datasets.py      # Auto-download script
├── merge_roboflow_datasets.py            # Merge multiple datasets
├── prepare_yolo_dataset.py               # Organize your dataset
├── quick_train.sh                        # Quick training script
└── train_production.sh                   # Production training script
```

## 🚀 Quick Start (3 Paths)

### Path 1: Manual Collection + SAM Annotation (RECOMMENDED for Insurance)

**Best for**: Custom, insurance-specific scenarios

```bash
# Step 1: Collect 50-100 rebut per class (use smartphone)
mkdir my_images
# Take photos of: Sofa, TV, Refrigerator, Laptop, etc.

# Step 2: Annotate with SAM (FASTEST!)
python rnd/interactive_sam_folder.py my_images/ sam_output/ --device mps

# Step 3: Convert to YOLO format
python rnd/sam_to_yolo.py sam_output/ my_images/ --output yolo_labels --format segmentation

# Step 4: Organize dataset
python prepare_yolo_dataset.py \
    --images my_images/ \
    --labels yolo_labels/labels \
    --output insurance_starter_dataset

# Step 5: Train!
./quick_train.sh
# Or: python train_yolo.py --mode train --data-yaml insurance_data.yaml --epochs 50
```

**Time**: 2-4 hours for 10-20 classes
**Result**: High-quality, domain-specific dataset

### Path 2: Download from Roboflow (FASTEST to start)

**Best for**: Quick start with existing labeled data

```bash
# See detailed steps in: ROBOFLOW_DOWNLOAD_STEPS.md

# Quick version:
# 1. Visit https://universe.roboflow.com/microsoft/coco
# 2. Download in YOLO v8 format
# 3. Extract to: datasets/roboflow_coco/
# 4. Merge:
python merge_roboflow_datasets.py datasets/roboflow_coco/ merged_datasets/

# 5. Train:
python train_yolo.py --mode train --data-yaml insurance_data.yaml --epochs 50
```

**Time**: 30 minutes
**Result**: 20-30 classes ready immediately

### Path 3: Hybrid Approach (BEST for Production)

```bash
# 1. Download COCO from Roboflow (20-30 classes)
# 2. Download furniture/appliances datasets (10-20 more classes)
# 3. Collect manually for remaining classes (insurance-specific items)
# 4. Merge everything
# 5. Train on complete dataset
```

**Time**: 1-2 weeks
**Result**: Complete 116-class production model

## 📊 Current Status

- ✅ All scripts created and configured
- ✅ Dataset structure ready
- ✅ Training configuration ready (insurance_data.yaml)
- ✅ 116 classes defined
- ⬜ Dataset images (YOU NEED TO ADD)

## 🎯 Recommended Next Steps

### Today (1 hour):
1. Start with **Path 1**: Manual collection for 10 priority classes
   - Sofa, TV, Refrigerator, Laptop, Bed, Dining Table, Microwave, Oven, Dishwasher, Toilet
2. Annotate with SAM tool (interactive, fast!)
3. Train first model
4. See results immediately

### This Week:
1. Expand to 30-40 classes
2. Add more training data
3. Retrain with better model (yolov8m or yolov8l)

### This Month:
1. Complete all 116 classes
2. Production training (yolov8x)
3. Deploy model

## 🔧 All Available Commands

```bash
# Dataset Management
python prepare_yolo_dataset.py --help
python merge_roboflow_datasets.py --help
python download_and_prepare_datasets.py

# Training
python train_yolo.py --mode train --data-yaml insurance_data.yaml --epochs 50
./quick_train.sh  # Quick training
./train_production.sh  # Production training

# Validation
python train_yolo.py --mode validate --model-path runs/detect/train/weights/best.pt

# Prediction
python train_yolo.py --mode predict --model-path runs/detect/train/weights/best.pt --image test.jpg

# Export
python train_yolo.py --mode export --export-format onnx
```

## 📚 Documentation Files

- `QUICK_DOWNLOAD_GUIDE.md` - How to download pre-labeled datasets
- `DATASET_SOURCES.md` - All data sources available
- `INSURANCE_WORKFLOW.md` - Complete workflow guide
- `TRAINING_GUIDE.md` - Detailed training instructions
- `ROBOFLOW_DOWNLOAD_STEPS.md` - Roboflow download guide

## 💡 Pro Tips

1. **Start Small**: 10 classes → 30 classes → 116 classes
2. **Quality > Quantity**: 50 perfect annotations > 200 sloppy ones
3. **Use SAM Tool**: It's FAST! Interactive annotation saves days
4. **Iterate**: Train, test, improve, repeat
5. **Real-World Data**: Use actual insurance claim photos

## 🎉 Everything is Ready!

Just add your images and start training!
