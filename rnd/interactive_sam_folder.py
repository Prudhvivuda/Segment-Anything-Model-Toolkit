# #!/usr/bin/env python3
# """
# Interactive SAM selector for choosing specific objects across multiple images
# """

# import cv2
# import numpy as np
# from pathlib import Path
# from segment_anything import sam_model_registry, SamPredictor
# import torch
# import json
# from typing import List, Dict, Optional, Tuple
# import argparse

# class InteractiveSAMSelector:
#     """Interactively select specific objects in multiple images"""
    
#     def __init__(self, 
#                  model_type: str = "vit_b",
#                  device: str = "mps",
#                  checkpoint_path: Optional[str] = None):
#         """Initialize SAM for interactive selection"""
        
#         # Setup device
#         if device == "mps" and not torch.backends.mps.is_available():
#             print("MPS not available, using CPU")
#             device = "cpu"
#         self.device = device
        
#         # Download checkpoint if needed
#         if checkpoint_path is None:
#             checkpoint_path = self._download_checkpoint(model_type)
        
#         # Load model
#         print(f"Loading SAM {model_type} model...")
#         self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
#         self.sam.to(device=device)
        
#         # Initialize predictor
#         self.predictor = SamPredictor(self.sam)
        
#         # Interactive state
#         self.reset_selection()
    
#     def _download_checkpoint(self, model_type: str) -> str:
#         """Download SAM checkpoint if needed"""
#         import urllib.request
        
#         checkpoint_urls = {
#             "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
#             "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
#             "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
#         }
        
#         checkpoint_dir = Path.home() / ".cache" / "sam"
#         checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
#         filename = checkpoint_urls[model_type].split("/")[-1]
#         checkpoint_path = checkpoint_dir / filename
        
#         if not checkpoint_path.exists():
#             print(f"Downloading SAM {model_type} checkpoint...")
#             urllib.request.urlretrieve(checkpoint_urls[model_type], checkpoint_path)
#             print(f"Downloaded to {checkpoint_path}")
        
#         return str(checkpoint_path)
    
#     def reset_selection(self):
#         """Reset current selection state"""
#         self.selected_points = []
#         self.selected_labels = []
#         self.current_masks = []
#         self.display_image = None
#         self.original_image = None
#         self.drawing_box = False
#         self.box_start = None
#         self.box_end = None
    
#     def process_folder_interactive(self, 
#                                   input_folder: str,
#                                   output_folder: str,
#                                   class_names: Optional[List[str]] = None):
#         """
#         Process all images in folder with interactive selection
        
#         Args:
#             input_folder: Folder containing images
#             output_folder: Output folder for selected masks
#             class_names: Optional list of class names for labeling
#         """
#         input_path = Path(input_folder)
#         output_path = Path(output_folder)
        
#         # Create output structure
#         masks_dir = output_path / "masks"
#         viz_dir = output_path / "visualizations"
#         masks_dir.mkdir(parents=True, exist_ok=True)
#         viz_dir.mkdir(parents=True, exist_ok=True)
        
#         # Find all images
#         image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
#         image_files = []
#         for ext in image_extensions:
#             image_files.extend(sorted(input_path.glob(f'*{ext}')))
#             image_files.extend(sorted(input_path.glob(f'*{ext.upper}')))
        
#         if not image_files:
#             print(f"No images found in {input_folder}")
#             return
        
#         print(f"\n{'='*60}")
#         print(f"Found {len(image_files)} images to process interactively")
#         print(f"{'='*60}\n")
        
#         # Process each image
#         all_results = []
        
#         for idx, image_path in enumerate(image_files):
#             print(f"\n[{idx+1}/{len(image_files)}] Processing: {image_path.name}")
            
#             # Interactive selection for this image
#             masks = self.select_objects_interactive(
#                 str(image_path), 
#                 class_names=class_names
#             )
            
#             if masks:
#                 # Save masks for this image
#                 image_name = image_path.stem
                
#                 for i, mask_data in enumerate(masks):
#                     # Save mask PNG
#                     mask_filename = masks_dir / f"{image_name}_mask_{i:03d}.png"
#                     cv2.imwrite(str(mask_filename), 
#                               (mask_data['mask'] * 255).astype(np.uint8))
                    
#                     # Save metadata JSON
#                     metadata = {
#                         "mask_id": i,
#                         "bbox": mask_data['bbox'],
#                         "area": int(mask_data['area']),
#                         "class": mask_data.get('class', 'object'),
#                         "class_id": mask_data.get('class_id', 0)
#                     }
                    
#                     meta_filename = masks_dir / f"{image_name}_mask_{i:03d}.json"
#                     with open(meta_filename, 'w') as f:
#                         json.dump(metadata, f, indent=2)
                
#                 # Save composite visualization
#                 composite = self.create_composite_visualization(masks)
#                 viz_filename = viz_dir / f"{image_name}_composite.png"
#                 cv2.imwrite(str(viz_filename), composite)
                
#                 all_results.append({
#                     "image": image_path.name,
#                     "masks_selected": len(masks)
#                 })
                
#                 print(f"✓ Saved {len(masks)} masks for {image_path.name}")
#             else:
#                 all_results.append({
#                     "image": image_path.name,
#                     "masks_selected": 0
#                 })
#                 print(f"✗ No masks selected for {image_path.name}")
        
#         # Save summary
#         summary = {
#             "total_images": len(image_files),
#             "total_masks": sum(r["masks_selected"] for r in all_results),
#             "results": all_results
#         }
        
#         summary_path = output_path / "selection_summary.json"
#         with open(summary_path, 'w') as f:
#             json.dump(summary, f, indent=2)
        
#         print(f"\n{'='*60}")
#         print(f"✅ Interactive selection complete!")
#         print(f"📊 Total masks selected: {summary['total_masks']}")
#         print(f"📁 Output saved to: {output_path}")
#         print(f"{'='*60}\n")
    
#     def select_objects_interactive(self, 
#                                   image_path: str,
#                                   class_names: Optional[List[str]] = None) -> List[Dict]:
#         """
#         Interactive selection for a single image
        
#         Returns:
#             List of selected mask dictionaries
#         """
#         # Read image
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Error: Cannot read {image_path}")
#             return []
        
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         self.predictor.set_image(image_rgb)
        
#         # Reset state
#         self.reset_selection()
#         self.original_image = image_rgb.copy()
#         self.display_image = image_rgb.copy()
#         self.image_height, self.image_width = image.shape[:2]
        
#         # Setup window
#         window_name = 'SAM Interactive Selection'
#         cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#         cv2.resizeWindow(window_name, 1200, 800)
        
#         # Mouse callback
#         def mouse_callback(event, x, y, flags, param):
#             if event == cv2.EVENT_LBUTTONDOWN:
#                 if flags & cv2.EVENT_FLAG_CTRLKEY:
#                     # Ctrl+Click = start box selection
#                     self.drawing_box = True
#                     self.box_start = (x, y)
#                     self.box_end = (x, y)
#                 else:
#                     # Left click = positive point
#                     self.selected_points.append([x, y])
#                     self.selected_labels.append(1)
#                     self._update_mask()
                    
#             elif event == cv2.EVENT_RBUTTONDOWN:
#                 # Right click = negative point
#                 self.selected_points.append([x, y])
#                 self.selected_labels.append(0)
#                 self._update_mask()
            
#             elif event == cv2.EVENT_MOUSEMOVE:
#                 if self.drawing_box:
#                     self.box_end = (x, y)
#                     self._draw_current_state()
            
#             elif event == cv2.EVENT_LBUTTONUP:
#                 if self.drawing_box:
#                     self.drawing_box = False
#                     # Generate mask from box
#                     if self.box_start and self.box_end:
#                         x1 = min(self.box_start[0], self.box_end[0])
#                         y1 = min(self.box_start[1], self.box_end[1])
#                         x2 = max(self.box_start[0], self.box_end[0])
#                         y2 = max(self.box_start[1], self.box_end[1])
                        
#                         if x2 - x1 > 5 and y2 - y1 > 5:  # Minimum box size
#                             self._generate_mask_from_box([x1, y1, x2, y2])
        
#         cv2.setMouseCallback(window_name, mouse_callback)
        
#         # Instructions
#         print("\n" + "="*50)
#         print("INTERACTIVE SELECTION CONTROLS:")
#         print("-"*50)
#         print("🖱️  Left Click        : Add object point (green)")
#         print("🖱️  Right Click       : Add background point (red)")
#         print("🖱️  Ctrl+Drag        : Draw bounding box")
#         print("⌨️  'a' / 'Enter'    : Accept current mask")
#         print("⌨️  '1-9'            : Assign class to mask")
#         print("⌨️  'r'              : Reset current selection")
#         print("⌨️  'u'              : Undo last mask")
#         print("⌨️  's'              : Skip this image")
#         print("⌨️  'q' / 'ESC'      : Finish and save")
#         print("="*50 + "\n")
        
#         # Main interaction loop
#         while True:
#             self._draw_current_state()
#             display_bgr = cv2.cvtColor(self.display_image, cv2.COLOR_RGB2BGR)
            
#             # Add status text
#             status_text = f"Masks: {len(self.current_masks)} | Points: {len(self.selected_points)}"
#             cv2.putText(display_bgr, status_text, (10, 30),
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
#             cv2.imshow(window_name, display_bgr)
            
#             key = cv2.waitKey(1) & 0xFF
            
#             if key == ord('q') or key == 27:  # Quit
#                 break
#             elif key == ord('s'):  # Skip image
#                 self.current_masks = []
#                 break
#             elif key == ord('a') or key == 13:  # Accept mask (Enter key)
#                 if hasattr(self, 'current_mask') and self.current_mask is not None:
#                     self._save_current_mask(class_names)
#             elif key == ord('r'):  # Reset current selection
#                 self.selected_points = []
#                 self.selected_labels = []
#                 self.display_image = self._create_display_with_saved_masks()
#             elif key == ord('u'):  # Undo last mask
#                 if self.current_masks:
#                     self.current_masks.pop()
#                     self.display_image = self._create_display_with_saved_masks()
#             elif ord('1') <= key <= ord('9'):  # Assign class
#                 class_id = key - ord('1')
#                 if hasattr(self, 'current_mask') and self.current_mask is not None:
#                     self._save_current_mask(class_names, class_id)
        
#         cv2.destroyWindow(window_name)
#         return self.current_masks
    
#     def _update_mask(self):
#         """Update mask based on current points"""
#         if not self.selected_points:
#             return
        
#         input_points = np.array(self.selected_points)
#         input_labels = np.array(self.selected_labels)
        
#         masks, scores, logits = self.predictor.predict(
#             point_coords=input_points,
#             point_labels=input_labels,
#             multimask_output=True,
#         )
        
#         # Select best mask
#         best_idx = np.argmax(scores)
#         self.current_mask = masks[best_idx]
        
#         self._draw_current_state()
    
#     def _generate_mask_from_box(self, box):
#         """Generate mask from bounding box"""
#         input_box = np.array(box)
        
#         masks, scores, logits = self.predictor.predict(
#             box=input_box,
#             multimask_output=True,
#         )
        
#         best_idx = np.argmax(scores)
#         self.current_mask = masks[best_idx]
        
#         # Clear points when using box
#         self.selected_points = []
#         self.selected_labels = []
        
#         self._draw_current_state()
    
#     def _draw_current_state(self):
#         """Draw current selection state"""
#         # Start with saved masks
#         self.display_image = self._create_display_with_saved_masks()
        
#         # Add current mask if exists
#         if hasattr(self, 'current_mask') and self.current_mask is not None:
#             overlay = self.display_image.copy()
#             overlay[self.current_mask] = overlay[self.current_mask] * 0.4 + np.array([255, 255, 0]) * 0.6
#             self.display_image = overlay.astype(np.uint8)
        
#         # Draw points
#         for point, label in zip(self.selected_points, self.selected_labels):
#             color = (0, 255, 0) if label == 1 else (255, 0, 0)
#             cv2.circle(self.display_image, tuple(point), 5, color, -1)
#             cv2.circle(self.display_image, tuple(point), 5, (255, 255, 255), 2)
        
#         # Draw box if dragging
#         if self.drawing_box and self.box_start and self.box_end:
#             cv2.rectangle(self.display_image, self.box_start, self.box_end, (255, 255, 0), 2)
    
#     def _create_display_with_saved_masks(self):
#         """Create display image with saved masks"""
#         display = self.original_image.copy()
        
#         for i, mask_data in enumerate(self.current_masks):
#             color = self._get_color_for_index(i)
#             mask = mask_data['mask']
#             display[mask] = display[mask] * 0.5 + np.array(color) * 0.5
        
#         return display.astype(np.uint8)
    
#     def _save_current_mask(self, class_names: Optional[List[str]] = None, class_id: int = 0):
#         """Save current mask to the list"""
#         if not hasattr(self, 'current_mask') or self.current_mask is None:
#             return
        
#         # Calculate bounding box
#         y_coords, x_coords = np.where(self.current_mask)
#         if len(x_coords) == 0:
#             return
        
#         bbox = [
#             int(x_coords.min()),
#             int(y_coords.min()),
#             int(x_coords.max()),
#             int(y_coords.max())
#         ]
        
#         class_name = "object"
#         if class_names and class_id < len(class_names):
#             class_name = class_names[class_id]
        
#         mask_data = {
#             'mask': self.current_mask,
#             'bbox': bbox,
#             'area': self.current_mask.sum(),
#             'class': class_name,
#             'class_id': class_id
#         }
        
#         self.current_masks.append(mask_data)
        
#         # Reset for next selection
#         self.selected_points = []
#         self.selected_labels = []
#         self.current_mask = None
        
#         print(f"  ✓ Saved mask {len(self.current_masks)}: {class_name} (area: {mask_data['area']})")
    
#     def _get_color_for_index(self, idx: int) -> Tuple[int, int, int]:
#         """Get consistent color for mask index"""
#         colors = [
#             (255, 0, 0),    # Red
#             (0, 255, 0),    # Green
#             (0, 0, 255),    # Blue
#             (255, 255, 0),  # Yellow
#             (255, 0, 255),  # Magenta
#             (0, 255, 255),  # Cyan
#             (255, 128, 0),  # Orange
#             (128, 0, 255),  # Purple
#         ]
#         return colors[idx % len(colors)]
    
#     def create_composite_visualization(self, masks: List[Dict]) -> np.ndarray:
#         """Create composite visualization of all masks"""
#         composite = self.original_image.copy()
        
#         for i, mask_data in enumerate(masks):
#             color = self._get_color_for_index(i)
#             mask = mask_data['mask']
#             composite[mask] = composite[mask] * 0.5 + np.array(color) * 0.5
            
#             # Add label
#             x1, y1, x2, y2 = mask_data['bbox']
#             label = f"{mask_data.get('class', 'object')}_{i}"
#             cv2.putText(composite, label, (x1, y1-5),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
#         return cv2.cvtColor(composite.astype(np.uint8), cv2.COLOR_RGB2BGR)

# def main():
#     parser = argparse.ArgumentParser(description="Interactive SAM object selection for folders")
#     parser.add_argument("input_folder", help="Folder containing images")
#     parser.add_argument("output_folder", help="Output folder for selected masks")
#     parser.add_argument("--model", choices=["vit_b", "vit_l", "vit_h"], 
#                        default="vit_b", help="SAM model size")
#     parser.add_argument("--classes", nargs="+", 
#                        help="Class names for labeling (e.g., person car dog)")
#     parser.add_argument("--device", choices=["mps", "cpu", "cuda"], 
#                        default="mps", help="Device to run on")
    
#     args = parser.parse_args()
    
#     # Initialize selector
#     selector = InteractiveSAMSelector(
#         model_type=args.model,
#         device=args.device
#     )
    
#     # Process folder interactively
#     selector.process_folder_interactive(
#         args.input_folder,
#         args.output_folder,
#         class_names=args.classes
#     )

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Interactive SAM selector for choosing specific objects across multiple images
"""

import cv2
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
import torch
import json
from typing import List, Dict, Optional, Tuple
import argparse

class InteractiveSAMSelector:
    """Interactively select specific objects in multiple images"""
    
    def __init__(self, 
                 model_type: str = "vit_b",
                 device: str = "mps",
                 checkpoint_path: Optional[str] = None):
        """Initialize SAM for interactive selection"""
        
        # Setup device
        if device == "mps" and not torch.backends.mps.is_available():
            print("MPS not available, using CPU")
            device = "cpu"
        self.device = device
        
        # Download checkpoint if needed
        if checkpoint_path is None:
            checkpoint_path = self._download_checkpoint(model_type)
        
        # Load model
        print(f"Loading SAM {model_type} model...")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        
        # Initialize predictor
        self.predictor = SamPredictor(self.sam)
        
        # Interactive state
        self.reset_selection()
    
    def _download_checkpoint(self, model_type: str) -> str:
        """Download SAM checkpoint if needed"""
        import urllib.request
        
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
    
    def reset_selection(self):
        """Reset current selection state"""
        self.selected_points = []
        self.selected_labels = []
        self.current_masks = []
        self.display_image = None
        self.original_image = None
        self.drawing_box = False
        self.box_start = None
        self.box_end = None
    
    def process_folder_interactive(self, 
                                  input_folder: str,
                                  output_folder: str):
        """
        Process all images in folder with interactive selection
        
        Args:
            input_folder: Folder containing images
            output_folder: Output folder for selected masks
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Create output structure
        masks_dir = output_path / "masks"
        viz_dir = output_path / "visualizations"
        masks_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(sorted(input_path.glob(f'*{ext}')))
            image_files.extend(sorted(input_path.glob(f'*{ext.upper}')))
        
        if not image_files:
            print(f"No images found in {input_folder}")
            return
        
        print(f"\n{'='*60}")
        print(f"Found {len(image_files)} images to process interactively")
        print(f"{'='*60}\n")
        
        # Process each image
        all_results = []
        
        for idx, image_path in enumerate(image_files):
            print(f"\n[{idx+1}/{len(image_files)}] Processing: {image_path.name}")
            
            # Interactive selection for this image
            masks = self.select_objects_interactive(str(image_path))
            
            if masks:
                # Save masks for this image
                image_name = image_path.stem
                
                for i, mask_data in enumerate(masks):
                    # Save mask PNG
                    mask_filename = masks_dir / f"{image_name}_mask_{i:03d}.png"
                    cv2.imwrite(str(mask_filename), 
                              (mask_data['mask'] * 255).astype(np.uint8))
                    
                    # Save metadata JSON
                    metadata = {
                        "mask_id": i,
                        "bbox": mask_data['bbox'],
                        "area": int(mask_data['area']),
                        "score": 0.0  # Placeholder for compatibility
                    }
                    
                    meta_filename = masks_dir / f"{image_name}_mask_{i:03d}.json"
                    with open(meta_filename, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                # Save composite visualization
                composite = self.create_composite_visualization(masks)
                viz_filename = viz_dir / f"{image_name}_composite.png"
                cv2.imwrite(str(viz_filename), composite)
                
                all_results.append({
                    "image": image_path.name,
                    "masks_selected": len(masks)
                })
                
                print(f"✓ Saved {len(masks)} masks for {image_path.name}")
            else:
                all_results.append({
                    "image": image_path.name,
                    "masks_selected": 0
                })
                print(f"✗ No masks selected for {image_path.name}")
        
        # Save summary
        summary = {
            "total_images": len(image_files),
            "total_masks": sum(r["masks_selected"] for r in all_results),
            "results": all_results
        }
        
        summary_path = output_path / "selection_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ Interactive selection complete!")
        print(f"📊 Total masks selected: {summary['total_masks']}")
        print(f"📁 Output saved to: {output_path}")
        print(f"{'='*60}\n")
    
    def select_objects_interactive(self, image_path: str) -> List[Dict]:
        """
        Interactive selection for a single image
        
        Returns:
            List of selected mask dictionaries
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot read {image_path}")
            return []
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        
        # Reset state
        self.reset_selection()
        self.original_image = image_rgb.copy()
        self.display_image = image_rgb.copy()
        self.image_height, self.image_width = image.shape[:2]
        
        # Setup window
        window_name = 'SAM Interactive Selection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)
        
        # Mouse callback
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    # Ctrl+Click = start box selection
                    self.drawing_box = True
                    self.box_start = (x, y)
                    self.box_end = (x, y)
                else:
                    # Left click = positive point
                    self.selected_points.append([x, y])
                    self.selected_labels.append(1)
                    self._update_mask()
                    
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right click = negative point
                self.selected_points.append([x, y])
                self.selected_labels.append(0)
                self._update_mask()
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing_box:
                    self.box_end = (x, y)
                    self._draw_current_state()
            
            elif event == cv2.EVENT_LBUTTONUP:
                if self.drawing_box:
                    self.drawing_box = False
                    # Generate mask from box
                    if self.box_start and self.box_end:
                        x1 = min(self.box_start[0], self.box_end[0])
                        y1 = min(self.box_start[1], self.box_end[1])
                        x2 = max(self.box_start[0], self.box_end[0])
                        y2 = max(self.box_start[1], self.box_end[1])
                        
                        if x2 - x1 > 5 and y2 - y1 > 5:  # Minimum box size
                            self._generate_mask_from_box([x1, y1, x2, y2])
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # Instructions
        print("\n" + "="*50)
        print("INTERACTIVE SELECTION CONTROLS:")
        print("-"*50)
        print("🖱️  Left Click        : Add object point (green)")
        print("🖱️  Right Click       : Add background point (red)")
        print("🖱️  Ctrl+Drag        : Draw bounding box")
        print("⌨️  'a' / 'Enter'    : Accept and save current mask")
        print("⌨️  'r'              : Reset current selection")
        print("⌨️  'u'              : Undo last mask")
        print("⌨️  's'              : Skip this image")
        print("⌨️  'q' / 'ESC'      : Finish and go to next image")
        print("="*50 + "\n")
        
        # Main interaction loop
        while True:
            self._draw_current_state()
            display_bgr = cv2.cvtColor(self.display_image, cv2.COLOR_RGB2BGR)
            
            # Add status text
            status_text = f"Masks saved: {len(self.current_masks)} | Points: {len(self.selected_points)}"
            cv2.putText(display_bgr, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Quit
                break
            elif key == ord('s'):  # Skip image
                self.current_masks = []
                break
            elif key == ord('a') or key == 13:  # Accept mask (Enter key)
                if hasattr(self, 'current_mask') and self.current_mask is not None:
                    self._save_current_mask()
            elif key == ord('r'):  # Reset current selection
                self.selected_points = []
                self.selected_labels = []
                self.display_image = self._create_display_with_saved_masks()
            elif key == ord('u'):  # Undo last mask
                if self.current_masks:
                    self.current_masks.pop()
                    self.display_image = self._create_display_with_saved_masks()
        
        cv2.destroyWindow(window_name)
        return self.current_masks
    
    def _update_mask(self):
        """Update mask based on current points"""
        if not self.selected_points:
            return
        
        input_points = np.array(self.selected_points)
        input_labels = np.array(self.selected_labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        self.current_mask = masks[best_idx]
        
        self._draw_current_state()
    
    def _generate_mask_from_box(self, box):
        """Generate mask from bounding box"""
        input_box = np.array(box)
        
        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=True,
        )
        
        best_idx = np.argmax(scores)
        self.current_mask = masks[best_idx]
        
        # Clear points when using box
        self.selected_points = []
        self.selected_labels = []
        
        self._draw_current_state()
    
    def _draw_current_state(self):
        """Draw current selection state"""
        # Start with saved masks
        self.display_image = self._create_display_with_saved_masks()
        
        # Add current mask if exists
        if hasattr(self, 'current_mask') and self.current_mask is not None:
            overlay = self.display_image.copy()
            overlay[self.current_mask] = overlay[self.current_mask] * 0.4 + np.array([255, 255, 0]) * 0.6
            self.display_image = overlay.astype(np.uint8)
        
        # Draw points
        for point, label in zip(self.selected_points, self.selected_labels):
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(self.display_image, tuple(point), 5, color, -1)
            cv2.circle(self.display_image, tuple(point), 5, (255, 255, 255), 2)
        
        # Draw box if dragging
        if self.drawing_box and self.box_start and self.box_end:
            cv2.rectangle(self.display_image, self.box_start, self.box_end, (255, 255, 0), 2)
    
    def _create_display_with_saved_masks(self):
        """Create display image with saved masks"""
        display = self.original_image.copy()
        
        for i, mask_data in enumerate(self.current_masks):
            color = self._get_color_for_index(i)
            mask = mask_data['mask']
            display[mask] = display[mask] * 0.5 + np.array(color) * 0.5
        
        return display.astype(np.uint8)
    
    def _save_current_mask(self):
        """Save current mask to the list"""
        if not hasattr(self, 'current_mask') or self.current_mask is None:
            return
        
        # Calculate bounding box
        y_coords, x_coords = np.where(self.current_mask)
        if len(x_coords) == 0:
            return
        
        bbox = [
            int(x_coords.min()),
            int(y_coords.min()),
            int(x_coords.max()),
            int(y_coords.max())
        ]
        
        mask_data = {
            'mask': self.current_mask,
            'bbox': bbox,
            'area': self.current_mask.sum()
        }
        
        self.current_masks.append(mask_data)
        
        # Reset for next selection
        self.selected_points = []
        self.selected_labels = []
        self.current_mask = None
        
        print(f"  ✓ Saved mask {len(self.current_masks)} (area: {mask_data['area']})")
    
    def _get_color_for_index(self, idx: int) -> Tuple[int, int, int]:
        """Get consistent color for mask index"""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]
        return colors[idx % len(colors)]
    
    def create_composite_visualization(self, masks: List[Dict]) -> np.ndarray:
        """Create composite visualization of all masks"""
        composite = self.original_image.copy()
        
        for i, mask_data in enumerate(masks):
            color = self._get_color_for_index(i)
            mask = mask_data['mask']
            composite[mask] = composite[mask] * 0.5 + np.array(color) * 0.5
            
            # Add mask number label
            x1, y1, x2, y2 = mask_data['bbox']
            label = f"Mask {i+1}"
            cv2.putText(composite, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return cv2.cvtColor(composite.astype(np.uint8), cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser(description="Interactive SAM object selection")
    parser.add_argument("input", help="Input folder or single image path")
    parser.add_argument("output_folder", help="Output folder for selected masks")
    parser.add_argument("--model", choices=["vit_b", "vit_l", "vit_h"], 
                       default="vit_b", help="SAM model size")
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], 
                       default="mps", help="Device to run on")
    
    args = parser.parse_args()
    
    # Initialize selector
    selector = InteractiveSAMSelector(
        model_type=args.model,
        device=args.device
    )
    
    # Check if input is a single image or folder
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Path does not exist: {args.input}")
        return
    
    if input_path.is_file():
        # Single image mode
        # Verify it's an image file
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        if input_path.suffix.lower() not in valid_extensions:
            print(f"Error: {input_path.name} is not a valid image file")
            print(f"Supported formats: {', '.join(valid_extensions)}")
            return
        # Single image mode
        print(f"Processing single image: {input_path.name}")
        
        # Create output directories
        output_path = Path(args.output_folder)
        masks_dir = output_path / "masks"
        viz_dir = output_path / "visualizations"
        masks_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Process single image
        masks = selector.select_objects_interactive(str(input_path))
        
        if masks:
            image_name = input_path.stem
            
            # Save masks
            for i, mask_data in enumerate(masks):
                # Save mask PNG
                mask_filename = masks_dir / f"{image_name}_mask_{i:03d}.png"
                cv2.imwrite(str(mask_filename), 
                          (mask_data['mask'] * 255).astype(np.uint8))
                
                # Save metadata JSON
                metadata = {
                    "mask_id": i,
                    "bbox": mask_data['bbox'],
                    "area": int(mask_data['area']),
                    "score": 0.0  # Placeholder for compatibility
                }
                
                meta_filename = masks_dir / f"{image_name}_mask_{i:03d}.json"
                with open(meta_filename, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Save visualization
            composite = selector.create_composite_visualization(masks)
            viz_filename = viz_dir / f"{image_name}_composite.png"
            cv2.imwrite(str(viz_filename), composite)
            
            print(f"\n✅ Saved {len(masks)} masks for {input_path.name}")
            print(f"📁 Output saved to: {output_path}")
        else:
            print(f"No masks selected for {input_path.name}")
    
    elif input_path.is_dir():
        # Folder mode - original functionality
        selector.process_folder_interactive(
            args.input,
            args.output_folder
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main()