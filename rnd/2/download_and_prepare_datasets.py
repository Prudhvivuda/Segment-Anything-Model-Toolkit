#!/usr/bin/env python3
"""
Download pre-labeled datasets and prepare for training
Focuses on classes available in COCO and other public datasets
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json

# COCO class mapping to your insurance classes
# COCO has 80 classes (0-79), mapping to Insurance_Priority_Classes.csv
COCO_TO_INSURANCE = {
    # COCO ID: (Insurance ID, Insurance Name)
    # Format: coco_id: (insurance_id, "Insurance Class Name")
    
    # Vehicles & Transportation (not in insurance list, skipping)
    # 0: person, 1: bicycle, 2: car, 3: motorcycle, etc.
    1: (92, "Bike"),  # bicycle -> Bike
    
    # Furniture
    56: (30, "Dining Chair"),  # chair -> Dining Chair
    57: (1, "Couch"),  # couch -> Couch
    59: (14, "Bed"),  # bed -> Bed
    60: (29, "Dining Table"),  # dining table -> Dining Table
    
    # Accessories
    24: (84, "Backpack"),  # backpack -> Backpack
    26: (82, "Handbag"),  # handbag -> Handbag
    28: (83, "Suitcase"),  # suitcase -> Suitcase
    
    # Sports Equipment
    37: (107, "Surfboard"),  # surfboard -> Surfboard
    
    # Electronics
    62: (5, "TV"),  # tv -> TV
    63: (44, "Laptop"),  # laptop -> Laptop
    64: (47, "Mouse"),  # mouse -> Mouse
    66: (46, "Keyboard"),  # keyboard -> Keyboard
    67: (72, "Smartphone"),  # cell phone -> Smartphone
    
    # Kitchen Appliances
    68: (25, "Microwave"),  # microwave -> Microwave
    69: (26, "Oven"),  # oven -> Oven
    70: (33, "Toaster"),  # toaster -> Toaster
    72: (23, "Refrigerator"),  # refrigerator -> Refrigerator
    
    # Bathroom
    61: (55, "Toilet"),  # toilet -> Toilet
    71: (56, "Sink"),  # sink -> Sink
    
    # Accessories (approximate matches)
    74: (76, "Watch"),  # clock -> Watch (closest match)
    
    # Note: COCO also has these classes but no direct insurance match:
    # - 39: bottle, 40: wine glass, 41: cup (could add if needed)
    # - 58: potted plant
    # - 73: book
    # - 75: vase
    # - 76: scissors
    # - 77: teddy bear
    # - 78: hair drier
    # - 79: toothbrush
    # These are skipped as they're not in your insurance priority list
}

def download_file(url: str, destination: str, desc: str = "Downloading"):
    """Download file with progress bar"""
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, destination, reporthook=t.update_to)

def download_coco128():
    """Download COCO128 sample dataset (already in YOLO format)"""
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    output_dir = Path("downloaded_datasets")
    output_dir.mkdir(exist_ok=True)
    
    zip_path = output_dir / "coco128.zip"
    
    print("\n" + "="*60)
    print("📦 Downloading COCO128 Dataset")
    print("="*60)
    print("Size: ~7 MB")
    print("Images: 128 images")
    print("Classes: 80 classes (will filter to your insurance classes)")
    print("="*60 + "\n")
    
    if not zip_path.exists():
        print("Downloading...")
        download_file(url, str(zip_path), "COCO128")
        print("✓ Download complete!")
    else:
        print("✓ Already downloaded")
    
    # Extract
    extract_dir = output_dir / "coco128"
    if not extract_dir.exists():
        print("\nExtracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("✓ Extraction complete!")
    
    return extract_dir

def filter_and_remap_coco(coco_dir: Path, output_dir: Path):
    """
    Filter COCO dataset to only include your insurance classes
    and remap class IDs
    """
    print("\n" + "="*60)
    print("🔄 Filtering and Remapping COCO Dataset")
    print("="*60)
    
    coco_dir = coco_dir / "coco128"
    images_dir = coco_dir / "images" / "train2017"
    labels_dir = coco_dir / "labels" / "train2017"
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ Error: Could not find COCO directories")
        print(f"   Images: {images_dir}")
        print(f"   Labels: {labels_dir}")
        return {'total_images': 0, 'filtered_images': 0, 'classes_found': set(), 'objects_per_class': {}}
    
    output_images = output_dir / "images"
    output_labels = output_dir / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_images': 0,
        'filtered_images': 0,
        'classes_found': set(),
        'objects_per_class': {},
        'coco_classes_seen': set()
    }
    
    # Process each image/label pair
    label_files = list(labels_dir.glob('*.txt'))
    print(f"\nFound {len(label_files)} label files to process")
    
    for label_file in tqdm(label_files, desc="Processing"):
        image_file = images_dir / f"{label_file.stem}.jpg"
        
        if not image_file.exists():
            continue
        
        stats['total_images'] += 1
        
        # Read and filter labels
        filtered_lines = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                coco_class = int(parts[0])
                stats['coco_classes_seen'].add(coco_class)
                
                # Check if this COCO class maps to our insurance classes
                if coco_class in COCO_TO_INSURANCE:
                    insurance_id, insurance_name = COCO_TO_INSURANCE[coco_class]
                    
                    # Remap class ID
                    parts[0] = str(insurance_id)
                    filtered_lines.append(' '.join(parts) + '\n')
                    
                    stats['classes_found'].add(insurance_name)
                    stats['objects_per_class'][insurance_name] = \
                        stats['objects_per_class'].get(insurance_name, 0) + 1
        
        # Only copy if we have valid objects
        if filtered_lines:
            # Copy image
            shutil.copy2(image_file, output_images / image_file.name)
            
            # Write filtered labels
            with open(output_labels / label_file.name, 'w') as f:
                f.writelines(filtered_lines)
            
            stats['filtered_images'] += 1
    
    print(f"\n✓ Filtering complete!")
    print(f"   Total images processed: {stats['total_images']}")
    print(f"   Images with insurance objects: {stats['filtered_images']}")
    print(f"   Insurance classes found: {len(stats['classes_found'])}")
    
    # Show what COCO classes were actually in the dataset
    print(f"\n📋 COCO classes found in dataset: {sorted(stats['coco_classes_seen'])}")
    print(f"   (Looking for: {sorted(COCO_TO_INSURANCE.keys())})")
    
    if stats['classes_found']:
        print(f"\n📊 Objects per class:")
        for cls, count in sorted(stats['objects_per_class'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {cls}: {count} objects")
    else:
        print(f"\n⚠️  No matching insurance classes found in this COCO128 sample")
        print(f"   COCO128 is a small subset - try downloading full COCO or Roboflow datasets")
    
    return stats

def create_train_val_split(dataset_dir: Path, train_ratio: float = 0.8):
    """Split dataset into train/val"""
    print("\n" + "="*60)
    print("✂️  Splitting into Train/Val")
    print("="*60)
    
    import random
    random.seed(42)
    
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    # Get all images
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    # Shuffle
    random.shuffle(image_files)
    
    # Split
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Create directory structure
    for split in ['train', 'val']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Move files
    print(f"Moving {len(train_files)} images to train...")
    for img in tqdm(train_files, desc="Train"):
        label = labels_dir / f"{img.stem}.txt"
        
        shutil.move(str(img), str(dataset_dir / 'images' / 'train' / img.name))
        if label.exists():
            shutil.move(str(label), str(dataset_dir / 'labels' / 'train' / label.name))
    
    print(f"Moving {len(val_files)} images to val...")
    for img in tqdm(val_files, desc="Val"):
        label = labels_dir / f"{img.stem}.txt"
        
        shutil.move(str(img), str(dataset_dir / 'images' / 'val' / img.name))
        if label.exists():
            shutil.move(str(label), str(dataset_dir / 'labels' / 'val' / label.name))
    
    # Remove temp directories
    if images_dir.exists() and not list(images_dir.glob('*')):
        images_dir.rmdir()
    if labels_dir.exists() and not list(labels_dir.glob('*')):
        labels_dir.rmdir()
    
    print(f"\n✓ Split complete!")
    print(f"   Train: {len(train_files)} images")
    print(f"   Val: {len(val_files)} images")

def create_data_yaml(output_dir: Path, insurance_csv: str = "Insurance_Priority_Classes.csv"):
    """Create data.yaml for training"""
    print("\n" + "="*60)
    print("📝 Creating data.yaml")
    print("="*60)
    
    # Read insurance classes
    df = pd.read_csv(insurance_csv)
    
    # Create names dictionary
    names = {}
    for _, row in df.iterrows():
        names[int(row['class_id'])] = row['class_name']
    
    # Create YAML content
    yaml_content = f"""# Insurance Object Detection Dataset
# Auto-generated from COCO and other public datasets

path: {output_dir.absolute()}
train: images/train
val: images/val

# Number of classes
nc: {len(names)}

# Class names
names:
"""
    
    for class_id, class_name in sorted(names.items()):
        yaml_content += f"  {class_id}: {class_name}\n"
    
    yaml_path = "insurance_data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Created {yaml_path}")
    print(f"   Classes: {len(names)}")
    print(f"   Path: {output_dir.absolute()}")
    
    return yaml_path

def verify_dataset(dataset_dir: Path):
    """Verify dataset structure"""
    print("\n" + "="*60)
    print("🔍 Verifying Dataset")
    print("="*60)
    
    for split in ['train', 'val']:
        img_dir = dataset_dir / 'images' / split
        label_dir = dataset_dir / 'labels' / split
        
        if not img_dir.exists() or not label_dir.exists():
            print(f"❌ {split}: Missing directories")
            continue
        
        images = list(img_dir.glob('*.*'))
        labels = list(label_dir.glob('*.txt'))
        
        print(f"\n{split.upper()}:")
        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")
        
        # Check matches
        img_stems = {img.stem for img in images}
        label_stems = {lbl.stem for lbl in labels}
        
        if img_stems == label_stems:
            print(f"  ✅ All files matched!")
        else:
            missing = len(img_stems - label_stems)
            if missing > 0:
                print(f"  ⚠️  {missing} images without labels")
        
        # Count objects
        total_objects = 0
        for label_file in labels:
            with open(label_file, 'r') as f:
                total_objects += len(f.readlines())
        
        if labels:
            print(f"  Objects: {total_objects} ({total_objects/len(labels):.1f} per image)")

def main():
    print("\n" + "="*70)
    print("🚀 INSURANCE DATASET PREPARATION")
    print("="*70)
    print("\nThis script will:")
    print("1. Download COCO128 dataset (pre-labeled, YOLO format)")
    print("2. Filter to only insurance-relevant classes")
    print("3. Remap class IDs to match your Insurance_Priority_Classes.csv")
    print("4. Split into train/val sets")
    print("5. Create data.yaml for training")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    # Step 1: Download COCO128
    coco_dir = download_coco128()
    
    # Step 2: Filter and remap
    output_dir = Path("insurance_ready_dataset")
    stats = filter_and_remap_coco(coco_dir, output_dir)
    
    if stats['filtered_images'] == 0:
        print("\n" + "="*70)
        print("❌ No insurance-relevant objects found in COCO128")
        print("="*70)
        print("\n💡 COCO128 is a very small sample (128 images)")
        print("   It might not contain the specific classes we're looking for.")
        print("\n🎯 RECOMMENDED NEXT STEPS:")
        print("\n1. Download from Roboflow Universe (BEST OPTION):")
        print("   Visit: https://universe.roboflow.com/microsoft/coco")
        print("   - Full COCO dataset with YOLO labels")
        print("   - Select 'YOLO v8' format")
        print("   - Much better class coverage")
        
        print("\n2. Search Roboflow for specific categories:")
        print("   - https://universe.roboflow.com/search?q=furniture")
        print("   - https://universe.roboflow.com/search?q=appliances")
        print("   - https://universe.roboflow.com/search?q=electronics")
        
        print("\n3. Merge multiple datasets:")
        print("   python merge_roboflow_datasets.py roboflow_downloads/ merged_dataset/")
        
        print("\n4. Or start with manual collection:")
        print("   - Collect 50-100 images per class")
        print("   - Annotate with: python rnd/interactive_sam_folder.py")
        
        print("\n" + "="*70)
        print("📄 See QUICK_DOWNLOAD_GUIDE.md for detailed instructions")
        print("="*70 + "\n")
        return
    
    # Step 3: Split train/val
    create_train_val_split(output_dir)
    
    # Step 4: Create data.yaml
    yaml_path = create_data_yaml(output_dir)
    
    # Step 5: Verify
    verify_dataset(output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("✅ DATASET READY!")
    print("="*70)
    print(f"\n📁 Dataset Location: {output_dir.absolute()}")
    print(f"📝 Config File: {yaml_path}")
    print(f"📊 Classes Found: {len(stats['classes_found'])}")
    print(f"🖼️  Total Images: {stats['filtered_images']}")
    
    print("\n🎯 Next Steps:")
    print("\n1. Review the dataset:")
    print(f"   ls {output_dir}/images/train/")
    
    print("\n2. Start training:")
    print(f"   python train_yolo.py \\")
    print(f"       --mode train \\")
    print(f"       --data-yaml {yaml_path} \\")
    print(f"       --model yolov8n \\")
    print(f"       --epochs 50 \\")
    print(f"       --batch 8")
    
    print("\n3. To add more classes:")
    print("   - Download more datasets from Roboflow Universe")
    print("   - Or collect/annotate your own images")
    print("   - Merge with this dataset")
    
    print("\n" + "="*70)
    
    # Save summary
    summary = {
        'dataset_dir': str(output_dir.absolute()),
        'yaml_file': yaml_path,
        'stats': {
            'total_images': stats['filtered_images'],
            'classes_found': list(stats['classes_found']),
            'objects_per_class': stats['objects_per_class']
        }
    }
    
    with open('dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n📄 Summary saved to: dataset_summary.json\n")

if __name__ == "__main__":
    main()

