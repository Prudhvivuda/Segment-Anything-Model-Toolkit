#!/usr/bin/env python3
"""
Merge multiple Roboflow datasets into one unified dataset
Handles YOLO v8 format from Roboflow Universe downloads
"""

import shutil
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import yaml

def merge_roboflow_datasets(source_dir: str, output_dir: str, remap_classes: bool = False):
    """
    Merge multiple Roboflow datasets
    
    Args:
        source_dir: Directory containing extracted Roboflow dataset folders
        output_dir: Output directory for merged dataset
        remap_classes: If True, remap all classes to sequential IDs starting from 0
    """
    source = Path(source_dir)
    output = Path(output_dir)
    
    print("\n" + "="*70)
    print("🔄 MERGING ROBOFLOW DATASETS")
    print("="*70)
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    counter = {'train': 0, 'val': 0, 'test': 0}
    all_classes = {}
    class_mapping = {}
    
    # Find all dataset folders
    dataset_dirs = [d for d in source.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not dataset_dirs:
        print(f"❌ No dataset folders found in {source_dir}")
        print(f"Expected structure: {source_dir}/<dataset_name>/train/...")
        return
    
    print(f"\nFound {len(dataset_dirs)} datasets to merge:")
    for d in dataset_dirs:
        print(f"  - {d.name}")
    
    print("\n" + "="*70)
    
    # Process each dataset
    for dataset_idx, dataset_dir in enumerate(dataset_dirs, 1):
        print(f"\n[{dataset_idx}/{len(dataset_dirs)}] Processing: {dataset_dir.name}")
        
        # Try to load data.yaml from this dataset
        data_yaml = dataset_dir / 'data.yaml'
        dataset_classes = {}
        
        if data_yaml.exists():
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    dataset_classes = data['names']
                    print(f"  Found {len(dataset_classes)} classes in data.yaml")
                    
                    # Merge class names
                    for cls_id, cls_name in dataset_classes.items():
                        if cls_name not in all_classes.values():
                            new_id = len(all_classes)
                            all_classes[new_id] = cls_name
                            class_mapping[f"{dataset_dir.name}_{cls_id}"] = new_id
        
        # Roboflow structure: train/ val/ test/ directories with images and labels
        for split in ['train', 'val', 'test']:
            split_dir = dataset_dir / split
            
            # Also check images/split pattern
            if not split_dir.exists():
                split_dir = dataset_dir / 'images' / split
            
            if not split_dir.exists():
                continue
            
            # Find images (may be in subdirectory or directly in split dir)
            image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            images = []
            for pattern in image_patterns:
                images.extend(list(split_dir.glob(pattern)))
                # Also check one level deeper
                images.extend(list(split_dir.glob(f'*/{pattern}')))
            
            if not images:
                continue
            
            print(f"  {split}: {len(images)} images")
            
            # Copy images and labels
            for img in tqdm(images, desc=f"  Copying {split}", leave=False):
                # New unique name
                new_name = f"merged_{counter[split]:06d}{img.suffix}"
                
                # Copy image
                dst_img = output / 'images' / split / new_name
                shutil.copy2(img, dst_img)
                
                # Find and copy label
                # Labels might be in same dir or parallel labels/ dir
                label_candidates = [
                    img.with_suffix('.txt'),  # Same directory
                    img.parent.parent / 'labels' / split / f"{img.stem}.txt",  # Parallel labels dir
                    dataset_dir / 'labels' / split / f"{img.stem}.txt",  # Root labels dir
                ]
                
                label_file = None
                for candidate in label_candidates:
                    if candidate.exists():
                        label_file = candidate
                        break
                
                if label_file and label_file.exists():
                    # If remapping, modify label file
                    if remap_classes and dataset_classes:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                        
                        remapped_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                old_cls = int(parts[0])
                                mapping_key = f"{dataset_dir.name}_{old_cls}"
                                if mapping_key in class_mapping:
                                    parts[0] = str(class_mapping[mapping_key])
                                    remapped_lines.append(' '.join(parts) + '\n')
                        
                        dst_label = output / 'labels' / split / f"merged_{counter[split]:06d}.txt"
                        with open(dst_label, 'w') as f:
                            f.writelines(remapped_lines)
                    else:
                        dst_label = output / 'labels' / split / f"merged_{counter[split]:06d}.txt"
                        shutil.copy2(label_file, dst_label)
                
                counter[split] += 1
    
    print("\n" + "="*70)
    print("✅ MERGE COMPLETE!")
    print("="*70)
    print(f"\nMerged Statistics:")
    print(f"  Train images: {counter['train']}")
    print(f"  Val images: {counter['val']}")
    print(f"  Test images: {counter['test']}")
    print(f"  Total images: {sum(counter.values())}")
    
    if all_classes:
        print(f"  Total classes: {len(all_classes)}")
        print(f"\n  Classes found:")
        for cls_id, cls_name in sorted(all_classes.items()):
            print(f"    {cls_id}: {cls_name}")
    
    print(f"\n📁 Output: {output.absolute()}")
    
    # Create data.yaml for merged dataset
    if all_classes:
        yaml_content = f"""# Merged Dataset from Roboflow
# Auto-generated

path: {output.absolute()}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {len(all_classes)}

# Class names
names:
"""
        for cls_id, cls_name in sorted(all_classes.items()):
            yaml_content += f"  {cls_id}: {cls_name}\n"
        
        yaml_path = output / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n📝 Created: {yaml_path}")
    
    print("\n🎯 Next Steps:")
    print(f"   python train_yolo.py --mode train --data-yaml {output}/data.yaml")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple Roboflow YOLO datasets"
    )
    parser.add_argument("source_dir", 
                       help="Directory containing extracted Roboflow datasets")
    parser.add_argument("output_dir",
                       help="Output directory for merged dataset")
    parser.add_argument("--remap-classes", action="store_true",
                       help="Remap all classes to sequential IDs starting from 0")
    
    args = parser.parse_args()
    
    merge_roboflow_datasets(args.source_dir, args.output_dir, args.remap_classes)

if __name__ == "__main__":
    main()

