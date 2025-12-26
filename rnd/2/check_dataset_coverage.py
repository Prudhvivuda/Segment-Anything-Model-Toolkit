#!/usr/bin/env python3
"""
Check which of your insurance classes are available in popular datasets
"""

import pandas as pd
from typing import Dict, List, Set

# COCO Dataset 80 classes (most commonly available)
COCO_CLASSES = {
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
}

# Open Images classes (simplified - actual has 600+)
OPEN_IMAGES_FURNITURE = {
    'Sofa', 'Couch', 'Chair', 'Office Chair', 'Armchair', 'Table', 'Desk', 'Coffee Table',
    'Dining Table', 'Bed', 'Nightstand', 'Dresser', 'Wardrobe', 'Bookshelf', 'Cabinet',
    'Drawer', 'Shelf', 'Bench', 'Stool'
}

OPEN_IMAGES_APPLIANCES = {
    'Refrigerator', 'Washing Machine', 'Dishwasher', 'Oven', 'Microwave', 'Toaster',
    'Coffee Maker', 'Blender', 'Mixer', 'Vacuum Cleaner', 'Fan', 'Air Conditioner',
    'Heater'
}

OPEN_IMAGES_ELECTRONICS = {
    'Television', 'TV', 'Computer', 'Laptop', 'Tablet', 'Monitor', 'Keyboard', 'Mouse',
    'Printer', 'Camera', 'Smartphone', 'Telephone', 'Speaker', 'Headphones'
}

OPEN_IMAGES_INSTRUMENTS = {
    'Guitar', 'Piano', 'Drum', 'Violin', 'Saxophone', 'Trumpet'
}

OPEN_IMAGES_SPORTS = {
    'Bicycle', 'Bike', 'Treadmill', 'Surfboard', 'Kayak', 'Golf Club'
}

def check_coverage(csv_path: str = "Insurance_Priority_Classes.csv"):
    """
    Check which classes are available in popular datasets
    """
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*80)
    print("DATASET COVERAGE ANALYSIS")
    print("="*80)
    print(f"\nTotal Insurance Classes: {len(df)}")
    
    # Check COCO coverage
    coco_matches = []
    for _, row in df.iterrows():
        class_name = row['class_name'].lower()
        # Check for matches (fuzzy matching)
        for coco_class in COCO_CLASSES:
            if class_name in coco_class or coco_class in class_name:
                coco_matches.append(row['class_name'])
                break
    
    print(f"\n{'='*80}")
    print(f"📊 COCO Dataset Coverage: {len(coco_matches)}/{len(df)} classes ({len(coco_matches)/len(df)*100:.1f}%)")
    print(f"{'='*80}")
    print("Available in COCO:")
    for cls in sorted(coco_matches):
        print(f"  ✓ {cls}")
    
    # Check Open Images coverage (approximation)
    all_open_images = (OPEN_IMAGES_FURNITURE | OPEN_IMAGES_APPLIANCES | 
                      OPEN_IMAGES_ELECTRONICS | OPEN_IMAGES_INSTRUMENTS | 
                      OPEN_IMAGES_SPORTS)
    
    open_images_matches = []
    for _, row in df.iterrows():
        class_name = row['class_name']
        if class_name in all_open_images or class_name.lower() in [x.lower() for x in all_open_images]:
            open_images_matches.append(class_name)
    
    print(f"\n{'='*80}")
    print(f"📊 Open Images Dataset Coverage: ~{len(open_images_matches)}-{len(open_images_matches)+20}/{len(df)} classes")
    print(f"{'='*80}")
    print("Likely available in Open Images:")
    for cls in sorted(open_images_matches):
        print(f"  ✓ {cls}")
    
    # Classes you'll need to collect yourself
    missing_classes = set(df['class_name']) - set(coco_matches) - set(open_images_matches)
    
    print(f"\n{'='*80}")
    print(f"⚠️  Classes Requiring Manual Collection: {len(missing_classes)}/{len(df)}")
    print(f"{'='*80}")
    print("You'll need to collect images for:")
    for cls in sorted(missing_classes):
        print(f"  ⬜ {cls}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("💡 RECOMMENDATIONS")
    print(f"{'='*80}")
    print(f"\n1. Download COCO Dataset:")
    print(f"   - Covers {len(coco_matches)} of your classes (~{len(coco_matches)/len(df)*100:.0f}%)")
    print(f"   - Already in YOLO format (via Roboflow)")
    print(f"   - ~18GB download")
    
    print(f"\n2. Search Roboflow Universe:")
    print(f"   - Community datasets for specific categories")
    print(f"   - Already YOLO-formatted")
    print(f"   - Free to use (check licenses)")
    
    print(f"\n3. Use Open Images Dataset:")
    print(f"   - Covers ~{len(open_images_matches)+20} of your classes")
    print(f"   - Needs conversion to YOLO format")
    print(f"   - Very large download")
    
    print(f"\n4. Collect Manually:")
    print(f"   - {len(missing_classes)} classes need custom collection")
    print(f"   - Use SAM tool for fast annotation")
    print(f"   - Focus on high-value items first")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Download COCO dataset (quick start):")
    print("   - Roboflow: https://universe.roboflow.com/microsoft/coco")
    print("   - Or use: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip")
    
    print("\n2. Browse Roboflow Universe for your missing classes:")
    print("   - https://universe.roboflow.com/")
    print("   - Search for: 'furniture', 'appliances', 'electronics', etc.")
    
    print("\n3. For remaining classes, use SAM tool:")
    print("   - Collect 50-100 images per class")
    print("   - Annotate with: python rnd/interactive_sam_folder.py")
    
    print("\n" + "="*80 + "\n")
    
    return {
        'coco_matches': coco_matches,
        'open_images_matches': open_images_matches,
        'missing': list(missing_classes)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check dataset coverage for insurance classes")
    parser.add_argument("--csv", default="Insurance_Priority_Classes.csv",
                       help="Path to insurance classes CSV")
    
    args = parser.parse_args()
    
    coverage = check_coverage(args.csv)
    
    # Save results
    import json
    with open('dataset_coverage.json', 'w') as f:
        json.dump(coverage, f, indent=2)
    
    print("📁 Coverage analysis saved to: dataset_coverage.json")

