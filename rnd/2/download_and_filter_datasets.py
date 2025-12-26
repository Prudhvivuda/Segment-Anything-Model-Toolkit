#!/usr/bin/env python3
"""
Download COCO and Open Images datasets
Filter to only insurance-relevant classes
Remap to Insurance_Priority_Classes.csv IDs
Organize into ready-to-train YOLO format
"""

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
import shutil
import random
import yaml

# Complete COCO to Insurance mapping
COCO_TO_INSURANCE = {
    1: 92,  # bicycle -> Bike
    24: 84,  # backpack -> Backpack
    26: 82,  # handbag -> Handbag
    28: 83,  # suitcase -> Suitcase
    37: 107,  # surfboard -> Surfboard
    56: 30,  # chair -> Dining Chair
    57: 1,  # couch -> Couch
    59: 14,  # bed -> Bed
    60: 29,  # dining table -> Dining Table
    61: 55,  # toilet -> Toilet
    62: 5,  # tv -> TV
    63: 44,  # laptop -> Laptop
    64: 47,  # mouse -> Mouse
    66: 46,  # keyboard -> Keyboard
    67: 72,  # cell phone -> Smartphone
    68: 25,  # microwave -> Microwave
    69: 26,  # oven -> Oven
    70: 33,  # toaster -> Toaster
    71: 56,  # sink -> Sink
    72: 23,  # refrigerator -> Refrigerator
    74: 76,  # clock -> Watch
}

# Open Images class name to Insurance mapping
OPEN_IMAGES_TO_INSURANCE = {
    # Furniture
    '/m/03hn7w': (0, "Sofa"),  # Sofa
    '/m/040b_t': (1, "Couch"),  # Couch
    '/m/01mzpv': (12, "Chair"),  # Chair -> Dining Chair (30) or Armchair (7)
    '/m/07c52': (14, "Bed"),  # Bed
    '/m/03fp41': (29, "Table"),  # Table -> Dining Table
    '/m/04bcr3': (39, "Desk"),  # Desk
    
    # Electronics
    '/m/07c52': (5, "Television"),  # TV
    '/m/0k4j': (44, "Laptop"),  # Laptop
    '/m/01x3z': (42, "Monitor"),  # Monitor
    '/m/025s21p': (46, "Keyboard"),  # Keyboard
    '/m/04f5wk': (47, "Mouse"),  # Mouse
    '/m/01mzpv': (72, "Mobile phone"),  # Smartphone
    
    # Appliances
    '/m/03d443': (23, "Refrigerator"),  # Refrigerator
    '/m/0270h': (25, "Microwave"),  # Microwave
    '/m/0bt9lr': (26, "Oven"),  # Oven
    '/m/07pbtc8': (56, "Sink"),  # Sink
    
    # Add more as needed
}

def download_coco_from_roboflow():
    """Download instructions for COCO from Roboflow"""
    print("\n" + "="*70)
    print("📥 COCO DATASET DOWNLOAD")
    print("="*70)
    print("\nCOCO dataset is too large to download automatically.")
    print("Please download from Roboflow (already in YOLO format):")
    print("\n1. Visit: https://universe.roboflow.com/microsoft/coco")
    print("2. Click 'Download' button")
    print("3. Select format: 'YOLO v8'")
    print("4. Click 'Download ZIP'")
    print("5. Extract to: 'downloaded_datasets/roboflow_coco/'")
    print("\nAfter downloading, run this script again with --process-coco flag")
    print("="*70 + "\n")

def process_coco_dataset(coco_dir: Path, output_dir: Path):
    """Process COCO dataset and filter to insurance classes"""
    print("\n" + "="*70)
    print("🔄 Processing COCO Dataset")
    print("="*70)
    
    # Find images and labels directories (Roboflow or COCO structure)
    possible_structures = [
        # Roboflow: train/ with images and labels subdirs
        (coco_dir / "train" / "images", coco_dir / "train" / "labels"),
        # Roboflow: images/train and labels/train
        (coco_dir / "images" / "train", coco_dir / "labels" / "train"),
        # COCO128: images/train2017 and labels/train2017
        (coco_dir / "images" / "train2017", coco_dir / "labels" / "train2017"),
        # Flat structure: train/ with mixed files
        (coco_dir / "train", coco_dir / "train"),
    ]
    
    images_dir = None
    labels_dir = None
    
    for img_path, lbl_path in possible_structures:
        # Check if images directory has image files
        if img_path.exists():
            has_images = any(img_path.glob('*.jpg')) or any(img_path.glob('*.png'))
            if has_images:
                images_dir = img_path
                # Try different label locations
                if lbl_path.exists():
                    labels_dir = lbl_path
                elif (coco_dir / "labels" / "train2017").exists():
                    labels_dir = coco_dir / "labels" / "train2017"
                elif (coco_dir / "labels" / "train").exists():
                    labels_dir = coco_dir / "labels" / "train"
                else:
                    # Try finding labels in same directory
                    labels_dir = img_path
                
                if labels_dir and (labels_dir.exists() and any(labels_dir.glob('*.txt'))):
                    break
    
    # Ensure we have separate directories
    if images_dir == labels_dir and images_dir.exists():
        # If same directory, create proper structure
        actual_images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        actual_labels = list(images_dir.glob('*.txt'))
        if actual_images and actual_labels:
            # This is the COCO128 structure where images and labels might be separate
            pass  # We'll handle this below
    
    if not images_dir or not labels_dir or not labels_dir.exists():
        print(f"\n❌ Could not find COCO structure in {coco_dir}")
        print(f"   Searched for:")
        print(f"   - train/images/ and train/labels/")
        print(f"   - images/train and labels/train")
        print(f"   - images/train2017 and labels/train2017")
        print(f"\n💡 Download COCO from Roboflow:")
        print(f"   1. Visit: https://universe.roboflow.com/microsoft/coco")
        print(f"   2. Download in 'YOLO v8' format")
        print(f"   3. Extract to: downloaded_datasets/roboflow_coco/")
        print(f"\n   OR use existing COCO128:")
        print(f"   python download_and_filter_datasets.py --process-coco downloaded_datasets/coco128")
        return {'processed_files': 0, 'classes_found': {}, 'total_files': 0, 'coco_classes_seen': set()}
    
    print(f"✓ Found images: {images_dir}")
    print(f"✓ Found labels: {labels_dir}")
    
    # Create output structure
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'classes_found': {},
        'coco_classes_seen': set()
    }
    
    # Process train set
    print(f"\nProcessing train set...")
    label_files = list(labels_dir.glob('*.txt'))
    
    for label_file in tqdm(label_files, desc="Processing"):
        # Find corresponding image
        image_file = images_dir / f"{label_file.stem}.jpg"
        if not image_file.exists():
            image_file = images_dir / f"{label_file.stem}.png"
        
        if not image_file.exists():
            continue
        
        stats['total_files'] += 1
        
        # Read and filter labels
        remapped_lines = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                coco_class = int(parts[0])
                stats['coco_classes_seen'].add(coco_class)
                
                # Check if maps to insurance class
                if coco_class in COCO_TO_INSURANCE:
                    insurance_id = COCO_TO_INSURANCE[coco_class]
                    
                    # Remap class ID
                    parts[0] = str(insurance_id)
                    remapped_lines.append(' '.join(parts) + '\n')
                    
                    # Track class
                    df = pd.read_csv("Insurance_Priority_Classes.csv")
                    match = df[df['class_id'] == insurance_id]
                    if not match.empty:
                        class_name = match.iloc[0]['class_name']
                        stats['classes_found'][class_name] = stats['classes_found'].get(class_name, 0) + 1
        
        # Only save if we have insurance-relevant objects
        if remapped_lines:
            # Copy image
            dst_img = output_dir / 'images' / 'train' / image_file.name
            shutil.copy2(image_file, dst_img)
            
            # Write remapped labels
            dst_label = output_dir / 'labels' / 'train' / label_file.name
            with open(dst_label, 'w') as f:
                f.writelines(remapped_lines)
            
            stats['processed_files'] += 1
    
    print(f"\n✅ COCO Processing Complete!")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Files with insurance classes: {stats['processed_files']}")
    print(f"   COCO classes seen: {sorted(stats['coco_classes_seen'])}")
    print(f"   Insurance classes: {len(stats['classes_found'])}")
    
    if stats['classes_found']:
        print(f"\n   Classes found:")
        for cls, count in sorted(stats['classes_found'].items(), key=lambda x: x[1], reverse=True):
            print(f"     {cls}: {count} objects")
    
    return stats

def split_dataset_train_val(output_dir: Path, train_ratio: float = 0.8):
    """Split train into train/val"""
    print(f"\n✂️  Splitting into train/val ({int(train_ratio*100)}/{int((1-train_ratio)*100)})...")
    
    train_images = list((output_dir / 'images' / 'train').glob('*.*'))
    
    if len(train_images) < 10:
        print(f"   ⚠️  Only {len(train_images)} images, skipping split")
        return
    
    random.seed(42)
    random.shuffle(train_images)
    
    split_idx = int(len(train_images) * train_ratio)
    val_images = train_images[split_idx:]
    
    # Move to val
    for img in tqdm(val_images, desc="Moving to val"):
        label = output_dir / 'labels' / 'train' / f"{img.stem}.txt"
        
        shutil.move(str(img), str(output_dir / 'images' / 'val' / img.name))
        if label.exists():
            shutil.move(str(label), str(output_dir / 'labels' / 'val' / label.name))
    
    print(f"   Train: {len(train_images) - len(val_images)} images")
    print(f"   Val: {len(val_images)} images")

def create_data_yaml(output_dir: Path):
    """Create data.yaml"""
    df = pd.read_csv("Insurance_Priority_Classes.csv")
    
    names = {}
    for _, row in df.iterrows():
        names[int(row['class_id'])] = row['class_name']
    
    data = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(names),
        'names': names
    }
    
    yaml_path = "insurance_data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Created {yaml_path}")
    return yaml_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and process COCO/Open Images for insurance classes")
    parser.add_argument("--process-coco", type=str, help="Path to downloaded COCO dataset")
    parser.add_argument("--output", default="insurance_ready_dataset", help="Output directory")
    parser.add_argument("--skip-download", action="store_true", help="Skip download instructions")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    print("\n" + "="*70)
    print("🎯 INSURANCE DATASET PREPARATION")
    print("="*70)
    print("\nThis script processes COCO dataset (in YOLO format) and filters")
    print("to only insurance-relevant classes from Insurance_Priority_Classes.csv")
    print("="*70)
    
    if not args.process_coco:
        download_coco_from_roboflow()
        
        if not args.skip_download:
            print("\n💡 After downloading COCO from Roboflow, run:")
            print(f"   python {sys.argv[0]} --process-coco downloaded_datasets/roboflow_coco/")
        return
    
    # Process COCO
    coco_dir = Path(args.process_coco)
    if not coco_dir.exists():
        print(f"❌ Directory not found: {coco_dir}")
        return
    
    stats = process_coco_dataset(coco_dir, output_dir)
    
    if stats['processed_files'] == 0:
        print("\n❌ No insurance-relevant objects found!")
        print("   Check that COCO dataset is in YOLO format")
        print("   And that download completed successfully")
        return
    
    # Split train/val
    split_dataset_train_val(output_dir)
    
    # Create data.yaml
    yaml_path = create_data_yaml(output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("✅ DATASET READY!")
    print("="*70)
    print(f"\n📁 Dataset: {output_dir.absolute()}")
    print(f"📝 Config: {yaml_path}")
    print(f"📊 Classes: {len(stats['classes_found'])} insurance classes found")
    print(f"🖼️  Images: {stats['processed_files']} images with insurance objects")
    
    print(f"\n🎯 Next Step - Start Training:")
    print(f"   ./quick_train.sh")
    print(f"   Or: python train_yolo.py --mode train --data-yaml {yaml_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

