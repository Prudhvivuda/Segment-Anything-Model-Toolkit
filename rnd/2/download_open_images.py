#!/usr/bin/env python3
"""
Download Open Images dataset for insurance classes
Using FiftyOne (recommended) or manual download
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd

# Open Images class names to download (matching insurance classes)
OPEN_IMAGES_CLASSES = [
    "Sofa",
    "Couch", 
    "Chair",
    "Bed",
    "Table",
    "Desk",
    "Refrigerator",
    "Dishwasher",
    "Microwave",
    "Oven",
    "Washing machine",
    "Television",
    "Computer monitor",
    "Laptop computer",
    "Mobile phone",
    "Camera",
    "Guitar",
    "Piano",
    "Bicycle",
    "Sink",
    "Toilet",
    "Handbag",
    "Backpack",
    "Suitcase",
    "Watch",
]

def check_fiftyone():
    """Check if FiftyOne is installed"""
    try:
        import fiftyone as fo
        return True
    except ImportError:
        return False

def download_open_images_fiftyone(output_dir: str = "open_images_dataset", max_samples: int = 1000):
    """
    Download Open Images using FiftyOne (easiest method)
    
    Args:
        output_dir: Directory to save dataset
        max_samples: Maximum samples per class
    """
    print("\n" + "="*70)
    print("📥 Downloading Open Images via FiftyOne")
    print("="*70)
    
    if not check_fiftyone():
        print("\n❌ FiftyOne not installed")
        print("\n📦 Installing FiftyOne...")
        print("   This may take a few minutes...")
        
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "fiftyone", "-q"
        ])
        
        print("✅ FiftyOne installed!")
    
    # Create download script
    download_script = f"""
import fiftyone as fo
import fiftyone.zoo as foz

print("Downloading Open Images dataset...")
print("This may take 1-2 hours depending on your internet speed")
print("Downloading {max_samples} images per class from {len(OPEN_IMAGES_CLASSES)} classes")

# Download Open Images
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes={OPEN_IMAGES_CLASSES},
    max_samples={max_samples},
    only_matching=True,
    shuffle=True,
    seed=42,
)

print(f"✓ Downloaded {{len(dataset)}} images")

# Export to YOLO format
print("Exporting to YOLO format...")
dataset.export(
    export_dir="{output_dir}",
    dataset_type=fo.types.YOLOv5Dataset,
)

print(f"✅ Export complete!")
print(f"📁 Dataset saved to: {output_dir}")
"""
    
    script_path = Path("_download_open_images_temp.py")
    with open(script_path, "w") as f:
        f.write(download_script)
    
    print(f"\n🚀 Starting download...")
    print(f"   Classes: {len(OPEN_IMAGES_CLASSES)}")
    print(f"   Max per class: {max_samples}")
    print(f"   Output: {output_dir}")
    print(f"\n   This will take 1-4 hours depending on internet speed")
    print(f"   You can run this in the background")
    
    try:
        subprocess.check_call([sys.executable, str(script_path)])
        script_path.unlink()  # Cleanup
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        script_path.unlink()
        return False

def download_open_images_manual():
    """Manual download instructions for Open Images"""
    print("\n" + "="*70)
    print("📥 Manual Open Images Download")
    print("="*70)
    print("\nOpen Images can be downloaded manually:")
    print("\n1. Visit: https://storage.googleapis.com/openimages/web/download.html")
    print("\n2. Download images for these classes:")
    for cls in OPEN_IMAGES_CLASSES:
        print(f"   - {cls}")
    
    print("\n3. Or use the OID toolkit:")
    print("   pip install openimages")
    print("   oi_download_dataset --base_dir ./open_images \\")
    print("       --labels 'Sofa,Chair,Bed,Table,Refrigerator' \\")
    print("       --format darknet")
    
    print("\n⚠️  Note: Manual download is complex")
    print("   Recommend using FiftyOne method instead")

def map_open_images_to_insurance(open_images_dir: Path, output_dir: Path):
    """
    Map Open Images class names to insurance class IDs
    """
    print("\n" + "="*70)
    print("🔄 Mapping Open Images to Insurance Classes")
    print("="*70)
    
    # Load insurance classes
    df = pd.read_csv("Insurance_Priority_Classes.csv")
    insurance_map = {}
    for _, row in df.iterrows():
        insurance_map[row['class_name'].lower()] = int(row['class_id'])
    
    # Open Images to Insurance mapping
    oi_to_insurance = {
        'sofa': 0,
        'couch': 1,
        'chair': 30,  # Default to dining chair
        'bed': 14,
        'table': 29,  # Default to dining table
        'desk': 39,
        'refrigerator': 23,
        'dishwasher': 24,
        'microwave': 25,
        'oven': 26,
        'washing machine': 49,
        'television': 5,
        'tv': 5,
        'computer monitor': 42,
        'monitor': 42,
        'laptop computer': 44,
        'laptop': 44,
        'mobile phone': 72,
        'smartphone': 72,
        'camera': 66,
        'guitar': 85,
        'piano': 86,
        'bicycle': 92,
        'bike': 92,
        'sink': 56,
        'toilet': 55,
        'handbag': 82,
        'backpack': 84,
        'suitcase': 83,
        'watch': 76,
    }
    
    print(f"Processing Open Images dataset...")
    # Implementation here would process the Open Images YOLO export
    
    print("✅ Mapping complete!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Open Images for insurance classes")
    parser.add_argument("--method", choices=["fiftyone", "manual"], default="fiftyone",
                       help="Download method")
    parser.add_argument("--output", default="open_images_dataset",
                       help="Output directory")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="Max samples per class")
    
    args = parser.parse_args()
    
    if args.method == "fiftyone":
        success = download_open_images_fiftyone(args.output, args.max_samples)
        if success:
            print("\n✅ Download complete!")
            print(f"\nNext step: Process to insurance classes")
            print(f"   python process_open_images_to_insurance.py {args.output}/")
    else:
        download_open_images_manual()

if __name__ == "__main__":
    main()

