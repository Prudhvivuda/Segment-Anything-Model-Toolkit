#!/usr/bin/env python3
"""
Process Roboflow datasets and remap classes to insurance classes
"""

import json
from pathlib import Path
import yaml
import shutil
import argparse
from tqdm import tqdm

def load_mapping():
    """Load class mapping"""
    with open("class_mapping.json", "r") as f:
        return json.load(f)

def remap_labels_to_insurance(source_dir: Path, output_dir: Path, mapping_config: dict):
    """Remap labels from source dataset to insurance classes"""
    
    coco_mapping = mapping_config.get('coco_mapping', {})
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'classes_mapped': {}
    }
    
    # Process each split
    for split in ['train', 'val', 'test']:
        source_images = source_dir / 'images' / split
        source_labels = source_dir / 'labels' / split
        
        if not source_images.exists() or not source_labels.exists():
            continue
        
        print(f"\nProcessing {split}...")
        
        for label_file in tqdm(list(source_labels.glob('*.txt')), desc=split):
            image_file = source_images / f"{label_file.stem}.jpg"
            if not image_file.exists():
                image_file = source_images / f"{label_file.stem}.png"
            
            if not image_file.exists():
                continue
            
            stats['total_files'] += 1
            
            # Read and remap labels
            remapped_lines = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    old_class = int(parts[0])
                    
                    # Check if this class maps to insurance
                    if old_class in coco_mapping:
                        insurance_id, insurance_name = coco_mapping[old_class]
                        parts[0] = str(insurance_id)
                        remapped_lines.append(' '.join(parts) + '\n')
                        
                        stats['classes_mapped'][insurance_name] = \
                            stats['classes_mapped'].get(insurance_name, 0) + 1
            
            if remapped_lines:
                # Copy image
                shutil.copy2(image_file, output_dir / 'images' / split / image_file.name)
                
                # Write remapped labels
                with open(output_dir / 'labels' / split / label_file.name, 'w') as f:
                    f.writelines(remapped_lines)
                
                stats['processed_files'] += 1
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Process Roboflow dataset to insurance classes")
    parser.add_argument("source_dir", help="Source Roboflow dataset directory")
    parser.add_argument("output_dir", help="Output directory for insurance-mapped dataset")
    
    args = parser.parse_args()
    
    source = Path(args.source_dir)
    output = Path(args.output_dir)
    
    if not source.exists():
        print(f"❌ Source directory not found: {source}")
        return
    
    print("\n" + "="*70)
    print("🔄 Processing Roboflow Dataset for Insurance Classes")
    print("="*70)
    
    # Load mapping
    try:
        mapping = load_mapping()
    except FileNotFoundError:
        print("❌ class_mapping.json not found. Run auto_setup_complete.py first.")
        return
    
    # Process
    stats = remap_labels_to_insurance(source, output, mapping)
    
    print(f"\n✅ Processing complete!")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Processed: {stats['processed_files']}")
    print(f"   Classes mapped: {len(stats['classes_mapped'])}")
    print(f"\n   Classes found:")
    for cls, count in sorted(stats['classes_mapped'].items(), key=lambda x: x[1], reverse=True):
        print(f"     {cls}: {count}")
    print(f"\n📁 Output: {output.absolute()}\n")

if __name__ == "__main__":
    main()
