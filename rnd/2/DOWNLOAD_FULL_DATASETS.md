# 📥 Download Full COCO & Open Images - Complete Guide

## ✅ What Just Happened

✅ **COCO128 processed**: 52 images, 18 insurance classes ready in `insurance_ready_dataset/`
✅ **You can start training NOW** with this smaller dataset for testing!

## 🎯 For Production: Download Full Datasets

### Option 1: Roboflow COCO (RECOMMENDED - Easiest)

**Why Roboflow**: Already in YOLO format, no conversion needed!

#### Steps:

1. **Download COCO Dataset**
   - Visit: https://universe.roboflow.com/microsoft/coco
   - Click **"Download"** button
   - Select format: **"YOLO v8"** ⭐
   - Click **"Download ZIP"**
   - File size: ~1-2GB (zipped), ~5-10GB (extracted)
   - Extract to: `downloaded_datasets/roboflow_coco/`

2. **Process the downloaded dataset**
   ```bash
   python download_and_filter_datasets.py --process-coco downloaded_datasets/roboflow_coco/
   ```

3. **Expected Result**
   - ~20,000-50,000 images with insurance classes
   - ~18-25 insurance classes
   - Ready to train!

#### Download Additional Roboflow Datasets:

**Furniture Detection:**
- Search: https://universe.roboflow.com/search?q=furniture
- Download top 2-3 datasets in YOLO v8 format
- Extract to: `downloaded_datasets/roboflow_furniture/`

**Home Appliances:**
- Search: https://universe.roboflow.com/search?q=appliances
- Download 1-2 datasets
- Extract to: `downloaded_datasets/roboflow_appliances/`

**Electronics:**
- Search: https://universe.roboflow.com/search?q=electronics
- Download 1-2 datasets
- Extract to: `downloaded_datasets/roboflow_electronics/`

4. **Merge All Datasets**
   ```bash
   python merge_roboflow_datasets.py downloaded_datasets/ merged_full_dataset/
   ```

5. **Process to Insurance Classes**
   ```bash
   python process_roboflow_to_insurance.py merged_full_dataset/ insurance_final_dataset/
   ```

**Total Time**: 2-4 hours (mostly download time)
**Result**: 50,000+ images, 40-60 insurance classes

---

### Option 2: Official COCO Dataset (Advanced)

**Note**: Requires format conversion from COCO JSON to YOLO

1. **Download COCO 2017**
   ```bash
   # Training images (~18GB)
   wget http://images.cocodataset.org/zips/train2017.zip
   
   # Training annotations
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   
   # Extract
   unzip train2017.zip
   unzip annotations_trainval2017.zip
   ```

2. **Convert COCO to YOLO format**
   ```bash
   # This would require a conversion script
   # You can use: https://github.com/ultralytics/JSON2YOLO
   ```

**Better Alternative**: Just use Roboflow! It's already converted.

---

### Option 3: Open Images Dataset (Largest Coverage)

**Coverage**: ~60-80 of your 116 classes

#### Using FiftyOne (Easiest):

```bash
# Install FiftyOne
pip install fiftyone

# Download Open Images with specific classes
python << EOF
import fiftyone as fo
import fiftyone.zoo as foz

# Download specific insurance-relevant classes
insurance_classes = [
    "Sofa", "Couch", "Chair", "Bed", "Table", "Desk",
    "Refrigerator", "Microwave", "Oven", "Dishwasher",
    "Washing Machine", "TV", "Laptop", "Computer", 
    "Monitor", "Keyboard", "Mouse", "Camera",
    "Guitar", "Piano", "Bicycle", "Bike"
]

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=insurance_classes,
    max_samples=5000  # Limit per class
)

# Export to YOLO format
dataset.export(
    export_dir="./open_images_yolo",
    dataset_type=fo.types.YOLOv5Dataset,
)
EOF
```

#### Manual Download (if FiftyOne doesn't work):

1. Visit: https://storage.googleapis.com/openimages/web/download.html
2. Download images and annotations for specific classes
3. Convert ko YOLO format

**Time**: 4-8 hours (large download)
**Result**: 100,000+ images, 60-80 classes

---

## 🚀 Complete Workflow

### Quick Path (Use Roboflow):

```bash
# 1. Download COCO from Roboflow (manual, takes 30-60 min)
# 2. Process it:
python download_and_filter_datasets.py --process-coco downloaded_datasets/roboflow_coco/

# 3. Download furniture/appliances datasets from Roboflow
# 4. Merge:
python merge_roboflow_datasets.py downloaded_datasets/roboflow_*/ merged/

# 5. Process merged dataset:
python download_and_filter_datasets.py --process-coco merged/ --output insurance_final_dataset

# 6. Train!
./quick_train.sh
```

### Comprehensive Path (All Sources):

```bash
# 1. COCO from Roboflow → Process
# 2. Open Images via FiftyOne → Export to YOLO → Process  
# 3. Additional Roboflow datasets → Merge → Process
# 4. Combine everything → Train production model
```

---

## 📊 Expected Results

| Source | Images | Classes | Time to Download | Time to Process |
Es------|--------|---------|------------------|-----------------|
| COCO128 (done) | 52 | 18 | Already done | ✅ Done |
| Full COCO (Roboflow) | 20K-50K | 18-25 | 30-60 min | 5-10 min |
| Additional Roboflow | 10K-30K | 10-20 | 1-2 hours | 5-10 min |
| Open Images | 50K-200K | 60-80 | 4-8 hours | 30-60 min |
| **TOTAL** | **80K-280K** | **70-90** | **5-10 hours** | **1-2 hours** |

---

## 💡 Pro Tips

1. **Start with Roboflow COCO**: Easiest, fastest, already YOLO format
2. **Download overnight**: Large files take time
3. **Check disk space**: Need 50-100GB free for full datasets
4. **Process incrementally**: Download → Process → Verify → Download more
5. **Keep organized**: Use clear folder names

---

## 🎯 Current Status

✅ **COCO128 Processed**: 52 images ready
✅ **Scripts Ready**: All automation in place
⬜ **Full COCO**: Download from Roboflow
⬜ **Additional Datasets**: Download from Roboflow Universe
⬜ **Open Images**: Optional, for maximum coverage

**Next Step**: Download full COCO from Roboflow and process it!

---

## 📝 Quick Reference

```bash
# Process any downloaded COCO/Roboflow dataset:
python download_and_filter_datasets.py --process-coco <dataset_folder>

# Merge multiple datasets:
python merge_roboflow_datasets.py <source1> <source2> <output>

# Start training:
./quick_train.sh
```

