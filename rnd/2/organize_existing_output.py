#!/usr/bin/env python3
"""
Organize existing SAM output into proper YOLO dataset with correct class mappings
"""

import pandas as pd
from pathlib import Path
import shutil
import json

# Map image names to insurance class IDs
IMAGE_TO_INSURANCE_CLASS = {
    'bag': 82,  # Handbag
    'blender motor + air fryer + microwave': 25,  # Microwave (primary object)
    'blender motor + air fryer': 32,  # Blender
    'dish washer': 24,  # Dishwasher
    'dryer + washer': 50,  # Dryer
    'fan': 63,  # Fan
    'fridge': 23,  # Refrigerator
    'heater': 62,  # Heater
    'instant pot': 37,  # Pressure Cooker
    'kitchen weight scale': 105,  # Could map to Toolbox or create new class
    'laptop': 44,  # Laptop
    'monitor': 42,  # Monitor
    'tv': 5,  # TV
    'vaccum cleaner': 54,  # Vacuum Cleaner
    'window ac': 61,  # Air Conditioner
}

def remap_classes_from_filename(label_file: Path) -> list:
    """Remap class ID based on filename"""
    # Extract base name from filename
    base_name = label_file.stem
    
    # Find matching insurance class
    insurance_class = 0  # Default
    for img_key, class_id in IMAGE_TO_INSURANCE_CLASS.items():
        if img_key.lower() in base_name.lower():
            insurance_class = class_id
            break
    
    # Read and remap
    remapped_lines = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                parts[0] = str(insurance_class)
                remapped_lines.append(' '.join(parts) + '\n')
    
    return remapped_lines

def organize_existing_output():
    """Organize existing output into insurance_starter_dataset"""
    
    print("\n" + "="*70)
    print("🔄 Organizing Existing SAM Output")
    print("="*70)
    
    source_labels = Path("output/yolo_labels/labels")
    source_images = Path("dataset")
    output_dir = Path("insurance_starter_dataset")
    
    if not source_labels.exists():
        print("❌ No labels found at output/yolo_labels/labels")
        return
    
    # Get all label files
    label_files = list(source_labels.glob('*.txt'))
    print(f"\nFound {len(label_files)} label files")
    
    # Count classes
    class_counts = {}
    processed = 0
    
    for label_file in label_files:
        # Find matching image
        base_name = label_file.stem
        
        # Try to find image
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = source_images / f"{base_name}{ext}"
            if candidate.exists():
                image_file = candidate
                break
        
        if not image_file:
            print(f"  ⚠️  No image found for {base_name}")
            continue
        
        # Remap classes
        remapped_lines = remap_classes_from_filename(label_file)
        
        if not remapped_lines:
            continue
        
        # Determine class
        for img_key, class_id in IMAGE_TO_INSURANCE_CLASS.items():
            if img_key.lower() in base_name.lower():
                class_name = None
                # Get class name from CSV
                df = pd.read_csv("Insurance_Priority_Classes.csv")
                match = df[df['class_id'] == class_id]
                if not match.empty:
                    class_name = match.iloc[0]['class_name']
                
                class_counts[class_name or f"Class {class_id}"] = class_counts.get(class_name or f"Class {class_id}", 0) + 1
                break
        
        # Copy to train set (we can split later)
        # For now, put all in train
        dst_img = output_dir / 'images' / 'train' / image_file.name
        dst_label = output_dir / 'labels' / 'train' / label_file.name
        
        shutil.copy2(image_file, dst_img)
        
        with open(dst_label, 'w') as f:
            f.writelines(remapped_lines)
        
        processed += 1
    
    print(f"\n✅ Processed {processed} images")
    print(f"\n📊 Classes found:")
    for cls, count in sorted(class_counts.items()):
        print(f"   {cls}: {count} images")
    
    # Split into train/val manually
    print(f"\n✂️  Splitting into train/val (80/20)...")
    
    import random
    random.seed(42)
    
    # Get all train files
    train_images = list((output_dir / 'images' / 'train').glob('*.*'))
    train_labels = list((output_dir / 'labels' / 'train').glob('*.txt'))
    
    # Shuffle
    random.shuffle(train_images)
    
    # Split 80/20
    split_idx = int(len(train_images) * 0.8)
    val_images = train_images[split_idx:]
    train_images = train_images[:split_idx]
    
    # Move validation files
    for img in val_images:
        label = output_dir / 'labels' / 'train' / f"{img.stem}.txt"
        
        # Move to val
        shutil.move(str(img), str(output_dir / 'images' / 'val' / img.name))
        if label.exists():
            shutil.move(str(label), str(output_dir / 'labels' / 'val' / label.name))
    
    print(f"   Train: {len(train_images)} images")
    print(f"   Val: {len(val_images)} images")
    
    print(f"\n📁 Dataset ready at: {output_dir}")
    print(f"   You can now train with: ./quick_train.sh")

if __name__ == "__main__":
    organize_existing_output()
