#!/usr/bin/env python3
"""
Complete automated pipeline to download and prepare datasets
Handles: COCO, Open Images, and additional sources
"""

import sys
from pathlib import Path
import subprocess

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_existing_datasets():
    """Check what datasets are already downloaded"""
    print_section("🔍 Checking Existing Datasets")
    
    downloaded = Path("downloaded_datasets")
    
    datasets_found = []
    
    # Check COCO128
    if (downloaded / "coco128").exists():
        datasets_found.append(("COCO128", downloaded / "coco128"))
        print("  ✓ COCO128 found")
    
    # Check Roboflow COCO
    if (downloaded / "roboflow_coco").exists():
        datasets_found.append(("Roboflow COCO", downloaded / "roboflow_coco"))
        print("  ✓ Roboflow COCO found")
    
    # Check for any other datasets
    for item in downloaded.iterdir():
        if item.is_dir() and item.name not in ["coco128"] and item.name.endswith("_dataset"):
            datasets_found.append((item.name, item))
            print(f"  ✓ {item.name} found")
    
    if not datasets_found:
        print("  ⚠️  No datasets found in downloaded_datasets/")
        print("  📥 You need to download datasets first")
    
    return datasets_found

def process_existing_datasets():
    """Process all existing datasets"""
    print_section("🔄 Processing Existing Datasets")
    
    datasets = check_existing_datasets()
    
    if not datasets:
        print("\n❌ No datasets to process")
        print_download_instructions()
        return
    
    output_dir = Path("insurance_ready_dataset")
    
    # Process each dataset
    for name, dataset_path in datasets:
        print(f"\n📦 Processing: {name}")
        try:
            subprocess.run([
                sys.executable, "download_and_filter_datasets.py",
                "--process-coco", str(dataset_path),
                "--output", str(output_dir)
            ], check=True)
            print(f"  ✅ {name} processed successfully")
        except Exception as e:
            print(f"  ❌ Error processing {name}: {e}")
            continue

def print_download_instructions():
    """Print clear download instructions"""
    print_section("📥 Download Instructions")
    
    print("\n🎯 To get datasets with insurance classes:")
    
    print("\n📦 OPTION 1: Roboflow COCO (RECOMMENDED - Easiest)")
    print("   1. Visit: https://universe.roboflow.com/microsoft/coco")
    print("   2. Click 'Download' → Select 'YOLO v8' format")
    print("   3. Extract ZIP to: downloaded_datasets/roboflow_coco/")
    print("   4. Run this script again to process")
    
    print("\n📦 OPTION 2: Open Images via FiftyOne")
    print("   Run: python download_open_images.py")
    print("   This will download ~20,000 images automatically")
    
    print("\n📦 OPTION 3: Use Current COCO128")
    print("   You already have COCO128 processed!")
    print("   Dataset ready at: insurance_ready_dataset/")
    print("   52 images, 18 classes - ready to train NOW!")

def create_download_helper():
    """Create a helper script for Roboflow downloads"""
    
    helper_script = """#!/bin/bash
# Helper script for Roboflow downloads

echo "📥 ROBOFLOW DOWNLOAD HELPER"
echo "=========================="
echo ""

echo "Step 1: Visit these links in your browser:"
echo ""
echo "COCO Dataset:"
echo "  https://universe.roboflow.com/microsoft/coco"
echo ""
echo "Additional Datasets (optional):"
echo "  Furniture: https://universe.roboflow.com/search?q=furniture"
echo "  Appliances: https://universe.roboflow.com/search?q=appliances"
echo "  Electronics: https://universe.roboflow.com/search?q=electronics"
echo ""

echo "Step 2: For each dataset:"
echo "  1. Click 'Download' button"
echo "  2. Select format: 'YOLO v8'"
echo "  3. Click 'Download ZIP'"
echo "  4. Extract ZIP to: downloaded_datasets/<dataset_name>/"
echo ""

echo "Step 3: After downloading, run:"
echo "  python complete_download_pipeline.py"
echo ""
"""
    
    with open("download_helper.sh", "w") as f:
        f.write(helper_script)
    
    import os
    os.chmod("download_helper.sh", 0o755)
    
    print("  ✓ Created download_helper.sh")

def main():
    print("\n" + "="*70)
    print("  🤖 COMPLETE DATASET DOWNLOAD & PREPARE PIPELINE")
    print("="*70)
    
    # Check what we have
    datasets = check_existing_datasets()
    
    # Process existing
    if datasets:
        process_existing_datasets()
        
        print_section("✅ Processing Complete")
        print("\n📁 Dataset ready at: insurance_ready_dataset/")
        print("\n🚀 Start training:")
        print("   ./quick_train.sh")
    else:
        print_download_instructions()
        create_download_helper()
        
        print("\n" + "="*70)
        print("💡 QUICK START:")
        print("="*70)
        print("\n1. Download COCO from Roboflow (see instructions above)")
        print("2. Extract to: downloaded_datasets/roboflow_coco/")
        print("3. Run: python complete_download_pipeline.py")
        print("\nOR use current COCO128 dataset:")
        print("   python download_and_filter_datasets.py --process-coco downloaded_datasets/coco128")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()

