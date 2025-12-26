#!/usr/bin/env python3
"""
Helper script to organize images and labels into YOLO dataset structure
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import argparse
from tqdm import tqdm

def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[int, int, int]:
    """
    Split dataset into train/val/test and organize in YOLO format
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO format labels (.txt)
        output_dir: Output directory for organized dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_count, val_count, test_count)
    """
    random.seed(seed)
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Find all images with corresponding labels
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in valid_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    # Filter images that have corresponding labels
    paired_files = []
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            paired_files.append((img_path, label_path))
        else:
            print(f"⚠️  Warning: No label found for {img_path.name}")
    
    if not paired_files:
        print("❌ Error: No images with corresponding labels found!")
        return 0, 0, 0
    
    # Shuffle files
    random.shuffle(paired_files)
    
    # Calculate split points
    total = len(paired_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_files = paired_files[:train_end]
    val_files = paired_files[train_end:val_end]
    test_files = paired_files[val_end:]
    
    print(f"\n{'='*60}")
    print(f"Dataset Split:")
    print(f"{'='*60}")
    print(f"Total: {total} images")
    print(f"Train: {len(train_files)} ({len(train_files)/total*100:.1f}%)")
    print(f"Val:   {len(val_files)} ({len(val_files)/total*100:.1f}%)")
    print(f"Test:  {len(test_files)} ({len(test_files)/total*100:.1f}%)")
    print(f"{'='*60}\n")
    
    # Copy files to respective directories
    def copy_split(files, split_name):
        print(f"Copying {split_name} files...")
        for img_path, label_path in tqdm(files, desc=split_name):
            # Copy image
            dst_img = output_dir / 'images' / split_name / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Copy label
            dst_label = output_dir / 'labels' / split_name / label_path.name
            shutil.copy2(label_path, dst_label)
    
    copy_split(train_files, 'train')
    copy_split(val_files, 'val')
    copy_split(test_files, 'test')
    
    print(f"\n✅ Dataset organized successfully!")
    print(f"📁 Output directory: {output_dir}")
    
    return len(train_files), len(val_files), len(test_files)

def filter_priority_classes(
    labels_dir: str,
    output_labels_dir: str,
    priority_classes: List[int],
    verbose: bool = True
) -> int:
    """
    Filter labels to only include priority classes
    
    Args:
        labels_dir: Input labels directory
        output_labels_dir: Output labels directory
        priority_classes: List of class IDs to keep
        verbose: Print progress
    
    Returns:
        Number of filtered label files
    """
    labels_dir = Path(labels_dir)
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    priority_set = set(priority_classes)
    label_files = list(labels_dir.glob('*.txt'))
    
    filtered_count = 0
    
    for label_file in tqdm(label_files, desc="Filtering labels", disable=not verbose):
        filtered_lines = []
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) in priority_set:
                    filtered_lines.append(line)
        
        # Only save if file has labels after filtering
        if filtered_lines:
            output_file = output_labels_dir / label_file.name
            with open(output_file, 'w') as f:
                f.writelines(filtered_lines)
            filtered_count += 1
    
    print(f"✓ Filtered {filtered_count} label files with priority classes")
    return filtered_count

def verify_dataset(dataset_dir: str):
    """
    Verify dataset structure and print statistics
    
    Args:
        dataset_dir: Root dataset directory
    """
    dataset_dir = Path(dataset_dir)
    
    print(f"\n{'='*60}")
    print("Dataset Verification")
    print(f"{'='*60}")
    
    for split in ['train', 'val', 'test']:
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
        
        # Check for mismatches
        img_stems = {img.stem for img in images}
        label_stems = {lbl.stem for lbl in labels}
        
        missing_labels = img_stems - label_stems
        missing_images = label_stems - img_stems
        
        if missing_labels:
            print(f"  ⚠️  {len(missing_labels)} images without labels")
        if missing_images:
            print(f"  ⚠️  {len(missing_images)} labels without images")
        
        if not missing_labels and not missing_images and len(images) > 0:
            print(f"  ✅ All files matched!")
        
        # Count objects per class
        class_counts = {}
        for label_file in labels:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        class_counts[cls] = class_counts.get(cls, 0) + 1
        
        if class_counts:
            print(f"  Classes: {len(class_counts)} unique classes")
            print(f"  Total objects: {sum(class_counts.values())}")
            print(f"  Avg objects/image: {sum(class_counts.values())/len(labels):.1f}")
    
    print(f"{'='*60}\n")

def load_priority_classes(file_path: str) -> List[int]:
    """Load priority class IDs from file"""
    with open(file_path, 'r') as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]

def main():
    parser = argparse.ArgumentParser(
        description="Organize images and labels into YOLO dataset structure"
    )
    
    parser.add_argument("--images", default="dataset",
                       help="Directory containing images")
    parser.add_argument("--labels", default="yolo_labels/labels",
                       help="Directory containing YOLO format labels")
    parser.add_argument("--output", default="yolo_dataset",
                       help="Output directory for organized dataset")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Training set ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                       help="Validation set ratio (default: 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                       help="Test set ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--priority-classes", type=str,
                       help="File with priority class IDs (one per line)")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing dataset")
    
    args = parser.parse_args()
    
    # Verify ratios sum to 1
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 0.01:
        print(f"❌ Error: Train/val/test ratios must sum to 1.0 (got {ratio_sum})")
        return
    
    if args.verify_only:
        verify_dataset(args.output)
        return
    
    # Filter by priority classes if specified
    labels_dir = args.labels
    if args.priority_classes:
        print(f"Loading priority classes from {args.priority_classes}...")
        priority_classes = load_priority_classes(args.priority_classes)
        print(f"✓ Loaded {len(priority_classes)} priority classes")
        
        # Create filtered labels
        filtered_labels_dir = Path(args.output) / "filtered_labels"
        filter_priority_classes(args.labels, str(filtered_labels_dir), priority_classes)
        labels_dir = str(filtered_labels_dir)
    
    # Split and organize dataset
    train_count, val_count, test_count = split_dataset(
        images_dir=args.images,
        labels_dir=labels_dir,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Verify the organized dataset
    if train_count > 0:
        verify_dataset(args.output)
        
        print("\n📝 Next Steps:")
        print(f"1. Create data.yaml:")
        print(f"   python train_yolo.py --mode create_yaml --dataset {args.output}")
        print(f"\n2. Start training:")
        print(f"   python train_yolo.py --mode train --data-yaml data.yaml")

if __name__ == "__main__":
    main()

