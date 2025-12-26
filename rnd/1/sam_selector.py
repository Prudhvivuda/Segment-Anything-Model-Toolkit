#!/usr/bin/env python3
"""
SAM (Segment Anything Model) Complete Toolkit
Supports multiple selection modes and export formats
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from PIL import Image
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm

class SAMSelector:
    """
    Complete SAM toolkit for object selection and segmentation
    """
    
    def __init__(self, 
                 model_type: str = "vit_b",  # vit_b, vit_l, or vit_h
                 checkpoint_path: Optional[str] = None,
                 device: str = "mps"):
        """
        Initialize SAM model
        
        Args:
            model_type: SAM model variant (vit_b=base, vit_l=large, vit_h=huge)
            checkpoint_path: Path to SAM checkpoint
            device: Device to run on
        """
        # Setup device
        if device == "mps" and not torch.backends.mps.is_available():
            print("MPS not available, using CPU")
            device = "cpu"
        self.device = device
        
        # Download checkpoint if not provided
        if checkpoint_path is None:
            checkpoint_path = self._download_checkpoint(model_type)
        
        # Load model
        print(f"Loading SAM {model_type} model...")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        
        # Initialize predictor for interactive mode
        self.predictor = SamPredictor(self.sam)
        
        # Initialize automatic mask generator
        self.auto_generator = None
        
    def _download_checkpoint(self, model_type: str) -> str:
        """Download SAM checkpoint if needed"""
        import urllib.request
        import os
        
        checkpoint_urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        }
        
        checkpoint_dir = Path.home() / ".cache" / "sam"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        filename = checkpoint_urls[model_type].split("/")[-1]
        checkpoint_path = checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            print(f"Downloading SAM {model_type} checkpoint...")
            urllib.request.urlretrieve(checkpoint_urls[model_type], checkpoint_path)
            print(f"Downloaded to {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def set_image(self, image: np.ndarray):
        """
        Set the current image for processing
        
        Args:
            image: Image array (HxWx3)
        """
        self.current_image = image
        self.predictor.set_image(image)
        self.image_height, self.image_width = image.shape[:2]
    
    def select_with_points(self, 
                          points: List[Tuple[int, int]], 
                          labels: Optional[List[int]] = None) -> Dict:
        """
        Select objects using point prompts
        
        Args:
            points: List of (x, y) coordinates
            labels: List of labels (1=foreground, 0=background)
        
        Returns:
            Dictionary with mask and metadata
        """
        if labels is None:
            labels = [1] * len(points)  # All foreground by default
        
        input_points = np.array(points)
        input_labels = np.array(labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        return {
            "mask": best_mask,
            "score": scores[best_idx],
            "all_masks": masks,
            "all_scores": scores,
            "area": best_mask.sum(),
            "bbox": self._mask_to_bbox(best_mask)
        }
    
    def select_with_box(self, box: Tuple[int, int, int, int]) -> Dict:
        """
        Select object using bounding box
        
        Args:
            box: (x1, y1, x2, y2) coordinates
        
        Returns:
            Dictionary with mask and metadata
        """
        input_box = np.array(box)
        
        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=True,
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        return {
            "mask": best_mask,
            "score": scores[best_idx],
            "all_masks": masks,
            "all_scores": scores,
            "area": best_mask.sum(),
            "bbox": self._mask_to_bbox(best_mask)
        }
    
    def auto_generate_masks(self, 
                           points_per_side: int = 32,
                           pred_iou_thresh: float = 0.88,
                           stability_score_thresh: float = 0.95,
                           min_mask_region_area: int = 100) -> List[Dict]:
        """
        Automatically generate masks for all objects
        
        Args:
            points_per_side: Number of points to sample
            pred_iou_thresh: IoU threshold for filtering
            stability_score_thresh: Stability threshold
            min_mask_region_area: Minimum mask area
        
        Returns:
            List of mask dictionaries
        """
        if self.auto_generator is None:
            self.auto_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
            )
        
        masks = self.auto_generator.generate(self.current_image)
        
        # Sort by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        return masks
    
    def interactive_selection(self, image_path: str):
        """
        Interactive GUI for selecting objects with clicks
        
        Args:
            image_path: Path to image
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.set_image(image_rgb)
        
        # Variables for interaction
        self.selected_points = []
        self.selected_labels = []
        self.current_mask = None
        self.display_image = image_rgb.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Left click = positive point
                self.selected_points.append([x, y])
                self.selected_labels.append(1)
                cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
                
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right click = negative point
                self.selected_points.append([x, y])
                self.selected_labels.append(0)
                cv2.circle(self.display_image, (x, y), 5, (255, 0, 0), -1)
            
            # Generate mask if we have points
            if len(self.selected_points) > 0:
                result = self.select_with_points(
                    self.selected_points, 
                    self.selected_labels
                )
                self.current_mask = result['mask']
                
                # Update display
                self.display_image = self._overlay_mask(
                    image_rgb.copy(), 
                    self.current_mask
                )
                
                # Draw points
                for point, label in zip(self.selected_points, self.selected_labels):
                    color = (0, 255, 0) if label == 1 else (255, 0, 0)
                    cv2.circle(self.display_image, tuple(point), 5, color, -1)
        
        # Create window
        window_name = 'SAM Interactive Selection'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        print("Instructions:")
        print("- Left click: Add positive point (object)")
        print("- Right click: Add negative point (background)")
        print("- 'r': Reset selection")
        print("- 's': Save current mask")
        print("- 'q': Quit")
        
        while True:
            display_bgr = cv2.cvtColor(self.display_image, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, display_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset
                self.selected_points = []
                self.selected_labels = []
                self.current_mask = None
                self.display_image = image_rgb.copy()
            elif key == ord('s'):
                # Save mask
                if self.current_mask is not None:
                    mask_path = Path(image_path).stem + '_mask.png'
                    cv2.imwrite(mask_path, (self.current_mask * 255).astype(np.uint8))
                    print(f"Mask saved to {mask_path}")
        
        cv2.destroyAllWindows()
        return self.current_mask
    
    def _mask_to_bbox(self, mask: np.ndarray) -> List[int]:
        """Convert mask to bounding box"""
        y_coords, x_coords = np.where(mask)
        if len(x_coords) == 0:
            return [0, 0, 0, 0]
        return [
            int(x_coords.min()),
            int(y_coords.min()),
            int(x_coords.max()),
            int(y_coords.max())
        ]
    
    def _overlay_mask(self, 
                     image: np.ndarray, 
                     mask: np.ndarray, 
                     color: Tuple[int, int, int] = (30, 144, 255),
                     alpha: float = 0.5) -> np.ndarray:
        """Overlay mask on image"""
        overlay = image.copy()
        overlay[mask] = overlay[mask] * (1 - alpha) + np.array(color) * alpha
        return overlay.astype(np.uint8)
    
    def save_masks(self, 
                  masks: List[Dict], 
                  output_dir: str,
                  image_name: str):
        """
        Save masks in various formats
        
        Args:
            masks: List of mask dictionaries
            output_dir: Output directory
            image_name: Base name for the image
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual masks
        masks_dir = output_dir / 'masks'
        masks_dir.mkdir(exist_ok=True)
        
        # Save visualization
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Create composite visualization
        composite = self.current_image.copy()
        
        for i, mask_dict in enumerate(masks):
            mask = mask_dict['mask'] if 'mask' in mask_dict else mask_dict['segmentation']
            
            # Save individual mask
            mask_filename = masks_dir / f"{image_name}_mask_{i:03d}.png"
            cv2.imwrite(str(mask_filename), (mask * 255).astype(np.uint8))
            
            # Add to composite
            color = np.random.randint(0, 255, 3).tolist()
            composite = self._overlay_mask(composite, mask, color, alpha=0.4)
            
            # Save metadata
            metadata = {
                'mask_id': i,
                'bbox': mask_dict.get('bbox', self._mask_to_bbox(mask)),
                'area': int(mask_dict.get('area', mask.sum())),
                'score': float(mask_dict.get('score', 0))
            }
            
            meta_filename = masks_dir / f"{image_name}_mask_{i:03d}.json"
            with open(meta_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Save composite visualization
        viz_filename = viz_dir / f"{image_name}_composite.png"
        cv2.imwrite(str(viz_filename), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        
        print(f"Saved {len(masks)} masks to {output_dir}")
    
    def export_to_coco_rle(self, mask: np.ndarray) -> Dict:
        """
        Export mask to COCO RLE format
        
        Args:
            mask: Binary mask
        
        Returns:
            RLE dictionary
        """
        from pycocotools import mask as mask_utils
        
        # Convert to Fortran order
        mask_fortran = np.asfortranarray(mask.astype(np.uint8))
        rle = mask_utils.encode(mask_fortran)
        rle['counts'] = rle['counts'].decode('utf-8')
        
        return rle
    
    def export_to_yolo_segment(self, 
                              mask: np.ndarray, 
                              class_id: int = 0) -> str:
        """
        Export mask to YOLO segmentation format
        
        Args:
            mask: Binary mask
            class_id: Class ID
        
        Returns:
            YOLO format string
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return ""
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Normalize coordinates
        points = []
        for point in simplified:
            x = point[0][0] / self.image_width
            y = point[0][1] / self.image_height
            points.extend([x, y])
        
        # Format: class_id x1 y1 x2 y2 ... xn yn
        return f"{class_id} " + " ".join(f"{p:.6f}" for p in points)

def main():
    parser = argparse.ArgumentParser(description="SAM toolkit for object selection")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--mode", choices=["interactive", "auto", "box", "points"], 
                       default="interactive", help="Selection mode")
    parser.add_argument("--model", choices=["vit_b", "vit_l", "vit_h"], 
                       default="vit_b", help="SAM model size")
    parser.add_argument("--output", default="sam_output", help="Output directory")
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], 
                       default="mps", help="Device to run on")
    
    # For box mode
    parser.add_argument("--box", nargs=4, type=int, 
                       help="Bounding box: x1 y1 x2 y2")
    
    # For points mode
    parser.add_argument("--points", nargs='+', type=int,
                       help="Points: x1 y1 x2 y2 ...")
    
    args = parser.parse_args()
    
    # Initialize SAM
    sam = SAMSelector(model_type=args.model, device=args.device)
    
    # Load image
    image = cv2.imread(args.image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam.set_image(image_rgb)
    
    image_name = Path(args.image).stem
    
    if args.mode == "interactive":
        # Interactive selection
        mask = sam.interactive_selection(args.image)
        if mask is not None:
            sam.save_masks([{"mask": mask}], args.output, image_name)
    
    elif args.mode == "auto":
        # Automatic segmentation
        masks = sam.auto_generate_masks()
        print(f"Generated {len(masks)} masks")
        sam.save_masks(masks, args.output, image_name)
    
    elif args.mode == "box" and args.box:
        # Box selection
        result = sam.select_with_box(args.box)
        sam.save_masks([result], args.output, image_name)
    
    elif args.mode == "points" and args.points:
        # Points selection
        points = [(args.points[i], args.points[i+1]) 
                 for i in range(0, len(args.points), 2)]
        result = sam.select_with_points(points)
        sam.save_masks([result], args.output, image_name)
    
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()