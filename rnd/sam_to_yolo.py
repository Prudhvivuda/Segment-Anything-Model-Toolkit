#!/usr/bin/env python3
"""
Convert SAM output masks to YOLO format (both detection and segmentation)
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm

class SAMToYOLOConverter:
    """Convert SAM masks to YOLO formats"""
    
    def __init__(self, sam_output_dir: str, original_images_dir: str):
        """
        Initialize converter
        
        Args:
            sam_output_dir: Directory containing SAM output (masks/ and visualizations/)
            original_images_dir: Directory containing original images
        """
        self.sam_output_dir = Path(sam_output_dir)
        self.masks_dir = self.sam_output_dir / "masks"
        self.viz_dir = self.sam_output_dir / "visualizations"
        self.original_images_dir = Path(original_images_dir)
        
        if not self.masks_dir.exists():
            raise ValueError(f"Masks directory not found: {self.masks_dir}")
    
    def get_image_dimensions(self, image_name: str) -> Tuple[int, int]:
        """
        Get dimensions of the original image
        
        Args:
            image_name: Base name of the image (without extension)
            
        Returns:
            (width, height) of the image
        """
        # Try to find the original image
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_path = self.original_images_dir / f"{image_name}{ext}"
            if image_path.exists():
                img = cv2.imread(str(image_path))
                if img is not None:
                    height, width = img.shape[:2]
                    return width, height
        
        # If original not found, try to infer from mask
        mask_files = list(self.masks_dir.glob(f"{image_name}_mask_*.png"))
        if mask_files:
            mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                height, width = mask.shape
                return width, height
        
        raise ValueError(f"Could not find image dimensions for {image_name}")
    
    def bbox_to_yolo_detection(self, 
                               bbox: List[int], 
                               img_width: int, 
                               img_height: int,
                               class_id: int = 0) -> str:
        """
        Convert bounding box to YOLO detection format
        
        Args:
            bbox: [x_min, y_min, x_max, y_max] in pixels
            img_width: Image width in pixels
            img_height: Image height in pixels
            class_id: Class ID for the object
            
        Returns:
            YOLO format string: "class_id x_center y_center width height"
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Calculate center and dimensions
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        
        # Normalize to [0, 1]
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        # Ensure values are within [0, 1]
        x_center_norm = max(0, min(1, x_center_norm))
        y_center_norm = max(0, min(1, y_center_norm))
        width_norm = max(0, min(1, width_norm))
        height_norm = max(0, min(1, height_norm))
        
        return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
    
    def mask_to_yolo_segmentation(self,
                                 mask_path: str,
                                 img_width: int,
                                 img_height: int,
                                 class_id: int = 0,
                                 simplify: bool = True) -> Optional[str]:
        """
        Convert mask to YOLO segmentation format
        
        Args:
            mask_path: Path to the mask PNG file
            img_width: Image width
            img_height: Image height
            class_id: Class ID
            simplify: Whether to simplify the contour
            
        Returns:
            YOLO segmentation format string or None if no valid contour
        """
        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        
        # Ensure binary mask
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Get largest contour (main object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour if requested
        if simplify:
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Skip if too few points
        if len(largest_contour) < 3:
            return None
        
        # Convert to normalized coordinates
        points = []
        for point in largest_contour:
            x = point[0][0] / img_width
            y = point[0][1] / img_height
            # Ensure normalized values
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            points.extend([x, y])
        
        # Format: class_id x1 y1 x2 y2 ... xn yn
        return f"{class_id} " + " ".join(f"{p:.6f}" for p in points)
    
    def process_image_masks(self, 
                           image_name: str,
                           class_mapping: Optional[Dict[str, int]] = None,
                           output_format: str = "detection") -> Dict:
        """
        Process all masks for a single image
        
        Args:
            image_name: Base name of the image
            class_mapping: Optional mapping of mask patterns to class IDs
            output_format: "detection" or "segmentation"
            
        Returns:
            Dictionary with processing results
        """
        # Get image dimensions
        try:
            img_width, img_height = self.get_image_dimensions(image_name)
        except ValueError as e:
            print(f"Error: {e}")
            return {"error": str(e)}
        
        # Find all masks for this image
        json_files = sorted(self.masks_dir.glob(f"{image_name}_mask_*.json"))
        
        if not json_files:
            return {"error": f"No masks found for {image_name}"}
        
        yolo_labels = []
        
        for json_file in json_files:
            # Read metadata
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            
            # Determine class ID
            class_id = 0  # Default class
            if class_mapping:
                # You can implement custom logic here
                # For example, based on mask_id or other criteria
                pass
            
            if output_format == "detection":
                # Convert bbox to YOLO detection format
                yolo_line = self.bbox_to_yolo_detection(
                    metadata['bbox'],
                    img_width,
                    img_height,
                    class_id
                )
                yolo_labels.append(yolo_line)
                
            elif output_format == "segmentation":
                # Convert mask to YOLO segmentation format
                mask_file = json_file.with_suffix('.png')
                if mask_file.exists():
                    yolo_line = self.mask_to_yolo_segmentation(
                        str(mask_file),
                        img_width,
                        img_height,
                        class_id
                    )
                    if yolo_line:
                        yolo_labels.append(yolo_line)
        
        return {
            "image_name": image_name,
            "num_objects": len(yolo_labels),
            "labels": yolo_labels,
            "dimensions": (img_width, img_height)
        }
    
    def convert_all(self, 
                   output_dir: str,
                   output_format: str = "detection",
                   class_mapping: Optional[Dict[str, int]] = None):
        """
        Convert all SAM masks to YOLO format
        
        Args:
            output_dir: Directory to save YOLO labels
            output_format: "detection" or "segmentation"
            class_mapping: Optional mapping for class IDs
        """
        output_dir = Path(output_dir)
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all unique image names
        json_files = list(self.masks_dir.glob("*_mask_*.json"))
        image_names = set()
        
        for json_file in json_files:
            # Extract image name (everything before _mask_XXX)
            name_parts = json_file.stem.rsplit('_mask_', 1)
            if len(name_parts) == 2:
                image_names.add(name_parts[0])
        
        if not image_names:
            print("No mask files found!")
            return
        
        print(f"Found masks for {len(image_names)} images")
        print(f"Converting to YOLO {output_format} format...")
        
        # Process each image
        results = []
        for image_name in tqdm(sorted(image_names)):
            result = self.process_image_masks(
                image_name, 
                class_mapping, 
                output_format
            )
            
            if "error" not in result:
                # Save YOLO label file
                label_file = labels_dir / f"{image_name}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(result['labels']))
                
                results.append(result)
                print(f"✓ {image_name}: {result['num_objects']} objects")
            else:
                print(f"✗ {image_name}: {result['error']}")
        
        # Save summary
        summary = {
            "format": output_format,
            "total_images": len(results),
            "total_objects": sum(r['num_objects'] for r in results),
            "output_dir": str(labels_dir),
            "images": results
        }
        
        summary_file = output_dir / "conversion_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create classes.txt
        classes_file = output_dir / "classes.txt"
        with open(classes_file, 'w') as f:
            if class_mapping:
                # Sort by class ID
                sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
                for class_name, class_id in sorted_classes:
                    f.write(f"{class_name}\n")
            else:
                f.write("object\n")  # Default single class
        
        print(f"\n✅ Conversion complete!")
        print(f"📁 Labels saved to: {labels_dir}")
        print(f"📄 Summary saved to: {summary_file}")
        print(f"📝 Classes saved to: {classes_file}")
        
        # Print sample
        if results:
            print(f"\n📊 Sample YOLO label (from {results[0]['image_name']}):")
            for line in results[0]['labels'][:3]:  # Show first 3 lines
                print(f"   {line}")

def main():
    parser = argparse.ArgumentParser(description="Convert SAM masks to YOLO format")
    parser.add_argument("sam_output", help="SAM output directory (containing masks/ folder)")
    parser.add_argument("original_images", help="Directory with original images")
    parser.add_argument("--output", default="yolo_labels", help="Output directory")
    parser.add_argument("--format", choices=["detection", "segmentation"], 
                       default="detection", help="Output format")
    parser.add_argument("--classes", help="JSON file with class mapping")
    
    args = parser.parse_args()
    
    # Load class mapping if provided
    class_mapping = None
    if args.classes:
        with open(args.classes, 'r') as f:
            class_mapping = json.load(f)
    
    # Convert
    converter = SAMToYOLOConverter(args.sam_output, args.original_images)
    converter.convert_all(args.output, args.format, class_mapping)

if __name__ == "__main__":
    main()