#!/usr/bin/env python3
"""
Helper script to download images from various sources
"""

import requests
from pathlib import Path
import argparse
from tqdm import tqdm
import time

def download_from_urls(urls_file: str, output_dir: str = "downloaded_images"):
    """
    Download images from a text file containing URLs
    
    Args:
        urls_file: Text file with one URL per line
        output_dir: Directory to save images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read URLs
    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(urls)} URLs to download")
    
    success_count = 0
    fail_count = 0
    
    for idx, url in enumerate(tqdm(urls, desc="Downloading")):
        try:
            # Determine file extension
            ext = url.split('.')[-1].split('?')[0]
            if ext not in ['jpg', 'jpeg', 'png', 'bmp']:
                ext = 'jpg'
            
            # Download
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Save
            filename = output_dir / f"image_{idx:05d}.{ext}"
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ Failed to download {url}: {e}")
            fail_count += 1
        
        # Be nice to servers
        time.sleep(0.5)
    
    print(f"\n✅ Download complete!")
    print(f"   Success: {success_count}")
    print(f"   Failed: {fail_count}")
    print(f"   Output: {output_dir}")

def download_unsplash_images(query: str, count: int = 100, api_key: str = None):
    """
    Download images from Unsplash
    Requires API key from https://unsplash.com/developers
    """
    if not api_key:
        print("❌ Error: Unsplash API key required")
        print("Get one at: https://unsplash.com/developers")
        return
    
    print(f"Downloading {count} images for query: {query}")
    print("Note: Unsplash API has rate limits")
    
    # Implementation left as exercise - needs API key registration

def scrape_google_images(query: str, count: int = 100, output_dir: str = "google_images"):
    """
    Note: This is for educational purposes only.
    Google's Terms of Service should be respected.
    Consider using official APIs or datasets instead.
    """
    print("⚠️  Warning: Web scraping may violate Terms of Service")
    print("Consider these alternatives:")
    print("  1. Use official APIs (Unsplash, Pexels, Flickr)")
    print("  2. Download from open datasets (ImageNet, Open Images)")
    print("  3. Take your own photos")
    print("  4. Use stock photo services with proper licenses")

def main():
    parser = argparse.ArgumentParser(description="Download images for training")
    parser.add_argument("--mode", choices=["urls", "unsplash", "info"],
                       default="info", help="Download mode")
    parser.add_argument("--urls-file", help="Text file with image URLs")
    parser.add_argument("--output", default="downloaded_images",
                       help="Output directory")
    parser.add_argument("--query", help="Search query for image search")
    parser.add_argument("--count", type=int, default=100,
                       help="Number of images to download")
    parser.add_argument("--api-key", help="API key for services")
    
    args = parser.parse_args()
    
    if args.mode == "urls":
        if not args.urls_file:
            print("❌ Error: --urls-file required for urls mode")
            return
        download_from_urls(args.urls_file, args.output)
    
    elif args.mode == "unsplash":
        download_unsplash_images(args.query, args.count, args.api_key)
    
    elif args.mode == "info":
        print("\n" + "="*60)
        print("IMAGE COLLECTION GUIDE")
        print("="*60)
        print("\n📸 Option 1: Take Your Own Photos (RECOMMENDED)")
        print("   - Most control over quality and content")
        print("   - Use smartphone or camera")
        print("   - Vary angles, lighting, backgrounds")
        print("   - 100-500 images per class")
        
        print("\n🌐 Option 2: Download from Open Datasets")
        print("   - Open Images: https://storage.googleapis.com/openimages/web/index.html")
        print("   - ImageNet: https://www.image-net.org/")
        print("   - COCO: https://cocodataset.org/")
        print("   - Roboflow Universe: https://universe.roboflow.com/")
        
        print("\n🔑 Option 3: Use Stock Photo APIs (with license)")
        print("   - Unsplash API: https://unsplash.com/developers")
        print("   - Pexels API: https://www.pexels.com/api/")
        print("   - Pixabay API: https://pixabay.com/api/docs/")
        print("   - Flickr API: https://www.flickr.com/services/api/")
        
        print("\n📝 Option 4: Create URLs File")
        print("   1. Manually collect image URLs")
        print("   2. Save to text file (one URL per line)")
        print("   3. Run: python download_images.py --mode urls --urls-file urls.txt")
        
        print("\n⚠️  IMPORTANT:")
        print("   - Respect copyright and terms of service")
        print("   - Verify you have rights to use images for training")
        print("   - Consider data privacy and ethical AI practices")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()

