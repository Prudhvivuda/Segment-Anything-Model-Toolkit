#!/usr/bin/env python3
"""
SAM (Segment Anything Model) Tool
===================================

A comprehensive toolkit for image segmentation using Meta's SAM model.
Provides interactive selection, automatic segmentation, and multiple export formats.

Features:
---------
1. Interactive selection mode (single image or batch)
2. Automatic segmentation mode
3. Programmatic selection (boxes, points)
4. Multiple export formats (YOLO, COCO RLE, binary masks)
5. Batch processing for folders
6. Visualization capabilities

Author: AI Assistant
Date: 2024
License: MIT
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from PIL import Image
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class SAMTool:
    """
    SAM Tool for all segmentation tasks.
    
    This class provides a complete interface to SAM (Segment Anything Model)
    with support for multiple selection modes, batch processing, and various
    export formats.
    
    Attributes:
        model_type (str): Type of SAM model ('vit_b', 'vit_l', 'vit_h')
        device (str): Computing device ('mps' for Apple Silicon, 'cuda' for NVIDIA, 'cpu')
        sam: The loaded SAM model
        predictor: SAM predictor for interactive/programmatic selection
        auto_generator: SAM automatic mask generator for auto mode
    """
    
    def __init__(self, 
                 model_type: str = "vit_b",
                 checkpoint_path: Optional[str] = None,
                 device: str = "auto"):
        """
        Initialize the SAM Tool.
        
        Args:
            model_type (str): SAM model variant to use
                - 'vit_b': Base model (375MB, fastest, good quality)
                - 'vit_l': Large model (1.2GB, balanced)
                - 'vit_h': Huge model (2.5GB, best quality, slowest)
            checkpoint_path (Optional[str]): Path to SAM checkpoint file.
                If None, will auto-download from Facebook AI
            device (str): Device to run inference on
                - 'auto': Automatically detect best available
                - 'mps': Force Apple Silicon GPU
                - 'cuda': Force NVIDIA GPU
                - 'cpu': Force CPU only
        
        Raises:
            RuntimeError: If specified device is not available
            ValueError: If model_type is not recognized
        """
        # Validate model type
        valid_models = ['vit_b', 'vit_l', 'vit_h']
        if model_type not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}, got {model_type}")
        
        # Setup device - intelligently detect best available option
        self.device = self._setup_device(device)
        print(f"Using device: {self.device}")
        
        # Download checkpoint if not provided
        if checkpoint_path is None:
            checkpoint_path = self._download_checkpoint(model_type)
        
        # Load the SAM model
        print(f"Loading SAM {model_type} model...")
        self.model_type = model_type
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        print(f"✓ Model loaded successfully")
        
        # Initialize components for different modes
        self.predictor = SamPredictor(self.sam)  # For interactive/programmatic selection
        self.auto_generator = None  # Will be initialized when needed for auto mode
        
        # State variables for interactive selection
        self._reset_interactive_state()
        
    def _setup_device(self, device: str) -> str:
        """
        Setup and validate the computing device.
        
        Args:
            device (str): Requested device ('auto', 'mps', 'cuda', 'cpu')
            
        Returns:
            str: The device string to use for PyTorch
            
        Raises:
            RuntimeError: If requested device is not available
        """
        if device == "auto":
            # Automatically detect best available device
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        
        elif device == "mps":
            # Apple Silicon GPU
            if not torch.backends.mps.is_available():
                print("Warning: MPS not available, falling back to CPU")
                return "cpu"
            return "mps"
        
        elif device == "cuda":
            # NVIDIA GPU
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return "cuda"
        
        else:
            # CPU fallback
            return "cpu"
    
    def _download_checkpoint(self, model_type: str) -> str:
        """
        Download SAM checkpoint from Facebook AI if not already cached.
        
        Args:
            model_type (str): Model variant to download
            
        Returns:
            str: Path to the downloaded checkpoint file
        """
        import urllib.request
        
        # URLs for different model checkpoints
        checkpoint_urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        }
        
        # Setup cache directory in user home
        checkpoint_dir = Path.home() / ".cache" / "sam"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract filename and create full path
        filename = checkpoint_urls[model_type].split("/")[-1]
        checkpoint_path = checkpoint_dir / filename
        
        # Download if not already cached
        if not checkpoint_path.exists():
            print(f"Downloading SAM {model_type} checkpoint...")
            print(f"This is a one-time download (~{self._get_model_size(model_type)})")
            
            # Download with progress callback
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                print(f"Progress: {percent:.1f}%", end='\r')
            
            urllib.request.urlretrieve(
                checkpoint_urls[model_type], 
                checkpoint_path,
                reporthook=download_progress
            )
            print(f"\n✓ Downloaded to {checkpoint_path}")
        else:
            print(f"Using cached checkpoint: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def _get_model_size(self, model_type: str) -> str:
        """Get human-readable model size for display."""
        sizes = {"vit_b": "375MB", "vit_l": "1.2GB", "vit_h": "2.5GB"}
        return sizes.get(model_type, "Unknown")
    
    def _reset_interactive_state(self):
        """Reset all interactive selection state variables."""
        self.selected_points = []  # List of clicked points
        self.selected_labels = []  # Labels for points (1=object, 0=background)
        self.current_masks = []  # List of saved masks for current image
        self.current_mask = None  # Currently active mask being edited
        self.display_image = None  # Image shown in GUI
        self.original_image = None  # Original image without annotations
        self.drawing_box = False  # Flag for box drawing mode
        self.box_start = None  # Start point of box selection
        self.box_end = None  # End point of box selection
    
    # ==================== Core Processing Methods ====================
    
    def set_image(self, image: np.ndarray):
        """
        Set the current image for processing.
        
        This method must be called before using selection methods like
        select_with_points() or select_with_box().
        
        Args:
            image (np.ndarray): Image array in RGB format with shape (H, W, 3)
        
        Example:
            >>> tool = SAMTool()
            >>> image = cv2.imread('photo.jpg')
            >>> image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            >>> tool.set_image(image_rgb)
        """
        self.current_image = image
        self.predictor.set_image(image)  # Compute image embeddings
        self.image_height, self.image_width = image.shape[:2]
    
    def select_with_points(self, 
                          points: List[Tuple[int, int]], 
                          labels: Optional[List[int]] = None) -> Dict:
        """
        Select objects using point prompts.
        
        This method allows you to specify points on the image to indicate
        what should be segmented (positive points) or what should be excluded
        (negative points).
        
        Args:
            points (List[Tuple[int, int]]): List of (x, y) pixel coordinates
            labels (Optional[List[int]]): List of labels for each point
                - 1: Positive point (part of object)
                - 0: Negative point (not part of object)
                If None, all points are treated as positive
        
        Returns:
            Dict: Dictionary containing:
                - 'mask': Boolean numpy array of the segmentation
                - 'score': Confidence score of the mask
                - 'all_masks': All generated mask options
                - 'all_scores': Scores for all masks
                - 'area': Number of pixels in the mask
                - 'bbox': Bounding box [x1, y1, x2, y2]
        
        Example:
            >>> # Select object with two positive points
            >>> result = tool.select_with_points(
            ...     points=[(100, 200), (150, 250)],
            ...     labels=[1, 1]
            ... )
            >>> mask = result['mask']
        """
        # Default all points to positive if labels not provided
        if labels is None:
            labels = [1] * len(points)
        
        # Convert to numpy arrays for SAM
        input_points = np.array(points)
        input_labels = np.array(labels)
        
        # Generate masks using SAM
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,  # Generate multiple mask options
        )
        
        # Select the best mask based on score
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        return {
            "mask": best_mask,
            "score": float(scores[best_idx]),
            "all_masks": masks,
            "all_scores": scores.tolist(),
            "area": int(best_mask.sum()),
            "bbox": self._mask_to_bbox(best_mask)
        }
    
    def select_with_box(self, box: Union[List[int], Tuple[int, int, int, int]]) -> Dict:
        """
        Select object using a bounding box prompt.
        
        This method segments the primary object within the provided
        bounding box coordinates.
        
        Args:
            box (Union[List, Tuple]): Bounding box as [x1, y1, x2, y2]
                where (x1, y1) is top-left and (x2, y2) is bottom-right
        
        Returns:
            Dict: Same format as select_with_points()
        
        Example:
            >>> # Select object within bounding box
            >>> result = tool.select_with_box([100, 100, 400, 400])
            >>> mask = result['mask']
        """
        # Convert to numpy array
        input_box = np.array(box)
        
        # Generate masks using SAM
        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=True,
        )
        
        # Select the best mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        return {
            "mask": best_mask,
            "score": float(scores[best_idx]),
            "all_masks": masks,
            "all_scores": scores.tolist(),
            "area": int(best_mask.sum()),
            "bbox": self._mask_to_bbox(best_mask)
        }
    
    def auto_generate_masks(self, 
                           points_per_side: int = 32,
                           pred_iou_thresh: float = 0.88,
                           stability_score_thresh: float = 0.95,
                           min_mask_region_area: int = 100,
                           max_objects: Optional[int] = None) -> List[Dict]:
        """
        Automatically generate masks for all detected objects in the image.
        
        This method uses SAM's automatic mask generation to segment all
        objects without any user input.
        
        Args:
            points_per_side (int): Number of points to sample per side of image.
                Higher values = more thorough but slower
            pred_iou_thresh (float): Filter masks with predicted IoU below this
            stability_score_thresh (float): Filter masks with stability below this
            min_mask_region_area (int): Minimum area in pixels for valid masks
            max_objects (Optional[int]): Limit number of returned objects
        
        Returns:
            List[Dict]: List of mask dictionaries, each containing:
                - 'segmentation': Boolean mask array
                - 'area': Mask area in pixels
                - 'bbox': Bounding box [x, y, w, h]
                - 'predicted_iou': Predicted IoU score
                - 'stability_score': Stability score
        
        Example:
            >>> # Automatically segment all objects
            >>> masks = tool.auto_generate_masks(
            ...     min_mask_region_area=500,  # Ignore tiny objects
            ...     max_objects=10  # Get top 10 largest objects
            ... )
            >>> print(f"Found {len(masks)} objects")
        """
        # Initialize generator if not already done
        if self.auto_generator is None:
            print("Initializing automatic mask generator...")
            self.auto_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
            )
        
        # Generate masks for current image
        print("Generating masks automatically...")
        masks = self.auto_generator.generate(self.current_image)
        
        # Sort by area (largest objects first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Limit number of objects if specified
        if max_objects is not None:
            masks = masks[:max_objects]
        
        print(f"Generated {len(masks)} masks")
        return masks
    
    # ==================== Interactive Selection Methods ====================
    
    def process_interactive(self,
                           input_path: str,
                           output_dir: str,
                           auto_skip: bool = False) -> Dict:
        """
        Process images interactively with GUI for object selection.
        
        This method provides a graphical interface for selecting objects
        using mouse clicks and keyboard shortcuts. It can process either
        a single image or all images in a folder.
        
        Args:
            input_path (str): Path to image file or folder containing images
            output_dir (str): Directory to save output masks
            auto_skip (bool): If True, automatically skip images with no selections
        
        Returns:
            Dict: Summary of processing results including:
                - 'total_images': Number of images processed
                - 'total_masks': Total masks created
                - 'results': List of per-image results
        
        Controls:
            - Left Click: Add positive point (object)
            - Right Click: Add negative point (background)
            - Ctrl+Drag: Draw bounding box
            - 'a' or Enter: Accept current mask
            - 'r': Reset current selection
            - 'u': Undo last mask
            - 's': Skip image
            - 'q' or ESC: Next image / Quit
        
        Example:
            >>> # Process folder interactively
            >>> results = tool.process_interactive(
            ...     'images/', 
            ...     'output/'
            ... )
        """
        input_path = Path(input_path)
        output_path = Path(output_dir)
        
        # Determine if input is file or folder
        if input_path.is_file():
            image_files = [input_path]
        elif input_path.is_dir():
            # Find all image files in folder
            image_files = self._find_images_in_folder(input_path)
            if not image_files:
                print(f"No images found in {input_path}")
                return {"total_images": 0, "total_masks": 0, "results": []}
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
        
        # Create output directories
        masks_dir = output_path / "masks"
        viz_dir = output_path / "visualizations"
        masks_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        all_results = []
        total_masks = 0
        
        print(f"\n{'='*60}")
        print(f"Starting interactive selection for {len(image_files)} image(s)")
        print(f"{'='*60}\n")
        
        for idx, image_file in enumerate(image_files):
            print(f"\n[{idx+1}/{len(image_files)}] Processing: {image_file.name}")
            
            # Interactive selection for current image
            masks = self._interactive_selection_single(str(image_file))
            
            if masks:
                # Save masks and metadata
                image_name = image_file.stem
                self._save_interactive_masks(
                    masks, image_name, masks_dir, viz_dir
                )
                
                all_results.append({
                    "image": image_file.name,
                    "masks_selected": len(masks)
                })
                total_masks += len(masks)
                print(f"✓ Saved {len(masks)} masks")
            
            elif not auto_skip:
                all_results.append({
                    "image": image_file.name,
                    "masks_selected": 0
                })
                print(f"✗ No masks selected")
        
        # Save summary
        summary = {
            "total_images": len(image_files),
            "total_masks": total_masks,
            "results": all_results
        }
        
        summary_path = output_path / "selection_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ Interactive selection complete!")
        print(f"📊 Total masks selected: {total_masks}")
        print(f"📁 Output saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return summary
    
    def _interactive_selection_single(self, image_path: str) -> List[Dict]:
        """
        Interactive selection for a single image using OpenCV GUI.
        
        Internal method that handles the GUI interaction for selecting
        objects in a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            List[Dict]: List of selected mask dictionaries
        """
        # Load and prepare image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot read {image_path}")
            return []
        
        # Convert to RGB and set for SAM
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.set_image(image_rgb)
        
        # Reset interactive state
        self._reset_interactive_state()
        self.original_image = image_rgb.copy()
        self.display_image = image_rgb.copy()
        
        # Setup OpenCV window
        window_name = 'SAM Interactive Selection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)
        
        # Mouse callback for interaction
        def mouse_callback(event, x, y, flags, param):
            """Handle mouse events for interactive selection."""
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    # Ctrl+Click starts box selection
                    self.drawing_box = True
                    self.box_start = (x, y)
                    self.box_end = (x, y)
                else:
                    # Regular left click adds positive point
                    self.selected_points.append([x, y])
                    self.selected_labels.append(1)
                    self._update_current_mask()
            
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right click adds negative point
                self.selected_points.append([x, y])
                self.selected_labels.append(0)
                self._update_current_mask()
            
            elif event == cv2.EVENT_MOUSEMOVE:
                # Update box if drawing
                if self.drawing_box:
                    self.box_end = (x, y)
                    self._draw_current_state()
            
            elif event == cv2.EVENT_LBUTTONUP:
                # Finish box selection
                if self.drawing_box:
                    self.drawing_box = False
                    if self.box_start and self.box_end:
                        # Calculate box coordinates
                        x1 = min(self.box_start[0], self.box_end[0])
                        y1 = min(self.box_start[1], self.box_end[1])
                        x2 = max(self.box_start[0], self.box_end[0])
                        y2 = max(self.box_start[1], self.box_end[1])
                        
                        # Minimum box size check
                        if x2 - x1 > 5 and y2 - y1 > 5:
                            self._generate_mask_from_box([x1, y1, x2, y2])
        
        # Set mouse callback
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # Print instructions
        self._print_interactive_instructions()
        
        # Main interaction loop
        while True:
            # Update display
            self._draw_current_state()
            display_bgr = cv2.cvtColor(self.display_image, cv2.COLOR_RGB2BGR)
            
            # Add status text overlay
            status = f"Masks: {len(self.current_masks)} | Points: {len(self.selected_points)}"
            cv2.putText(display_bgr, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show image
            cv2.imshow(window_name, display_bgr)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Quit (q or ESC)
                break
            elif key == ord('s'):  # Skip image
                self.current_masks = []
                break
            elif key == ord('a') or key == 13:  # Accept mask (a or Enter)
                if self.current_mask is not None:
                    self._save_current_mask_to_list()
            elif key == ord('r'):  # Reset current selection
                self.selected_points = []
                self.selected_labels = []
                self.current_mask = None
                self.display_image = self._create_display_with_saved_masks()
            elif key == ord('u'):  # Undo last mask
                if self.current_masks:
                    self.current_masks.pop()
                    self.display_image = self._create_display_with_saved_masks()
                    print(f"  Undone - now {len(self.current_masks)} masks")
        
        cv2.destroyWindow(window_name)
        return self.current_masks
    
    def _update_current_mask(self):
        """Update the current mask based on selected points."""
        if not self.selected_points:
            return
        
        # Get mask from SAM
        result = self.select_with_points(
            self.selected_points,
            self.selected_labels
        )
        self.current_mask = result['mask']
        
        # Update display
        self._draw_current_state()
    
    def _generate_mask_from_box(self, box):
        """Generate mask from bounding box selection."""
        result = self.select_with_box(box)
        self.current_mask = result['mask']
        
        # Clear points when using box
        self.selected_points = []
        self.selected_labels = []
        
        self._draw_current_state()
    
    def _draw_current_state(self):
        """Update the display image with current selection state."""
        # Start with saved masks
        self.display_image = self._create_display_with_saved_masks()
        
        # Overlay current mask if exists
        if self.current_mask is not None:
            overlay = self.display_image.copy()
            # Yellow overlay for current selection
            overlay[self.current_mask] = overlay[self.current_mask] * 0.4 + np.array([255, 255, 0]) * 0.6
            self.display_image = overlay.astype(np.uint8)
        
        # Draw selection points
        for point, label in zip(self.selected_points, self.selected_labels):
            color = (0, 255, 0) if label == 1 else (255, 0, 0)  # Green for positive, red for negative
            cv2.circle(self.display_image, tuple(point), 5, color, -1)
            cv2.circle(self.display_image, tuple(point), 5, (255, 255, 255), 2)  # White border
        
        # Draw box if in drawing mode
        if self.drawing_box and self.box_start and self.box_end:
            cv2.rectangle(self.display_image, self.box_start, self.box_end, (255, 255, 0), 2)
    
    def _create_display_with_saved_masks(self):
        """Create display image with all saved masks overlaid."""
        display = self.original_image.copy()
        
        # Overlay each saved mask with a different color
        for i, mask_data in enumerate(self.current_masks):
            color = self._get_color_for_index(i)
            mask = mask_data['mask']
            display[mask] = display[mask] * 0.5 + np.array(color) * 0.5
        
        return display.astype(np.uint8)
    
    def _save_current_mask_to_list(self):
        """Save the current mask to the list of masks for this image."""
        if self.current_mask is None:
            return
        
        # Calculate bounding box
        bbox = self._mask_to_bbox(self.current_mask)
        
        # Create mask data dictionary
        mask_data = {
            'mask': self.current_mask.copy(),
            'bbox': bbox,
            'area': int(self.current_mask.sum())
        }
        
        self.current_masks.append(mask_data)
        
        # Reset for next selection
        self.selected_points = []
        self.selected_labels = []
        self.current_mask = None
        
        print(f"  ✓ Saved mask {len(self.current_masks)} (area: {mask_data['area']} pixels)")
    
    # ==================== Export Methods ====================
    
    def export_to_yolo_detection(self,
                                mask_or_bbox: Union[np.ndarray, List[int]],
                                class_id: int = 0) -> str:
        """
        Export mask or bbox to YOLO detection format.
        
        YOLO format: class_id x_center y_center width height (normalized)
        
        Args:
            mask_or_bbox: Either a boolean mask array or [x1, y1, x2, y2] bbox
            class_id: Class ID for the object (default 0)
            
        Returns:
            str: YOLO format string
            
        Example:
            >>> yolo_line = tool.export_to_yolo_detection(mask, class_id=2)
            >>> print(yolo_line)  # "2 0.5 0.5 0.3 0.4"
        """
        # Get bbox from mask if needed
        if isinstance(mask_or_bbox, np.ndarray):
            bbox = self._mask_to_bbox(mask_or_bbox)
        else:
            bbox = mask_or_bbox
        
        x1, y1, x2, y2 = bbox
        
        # Calculate center and dimensions
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # Normalize by image dimensions
        x_center_norm = x_center / self.image_width
        y_center_norm = y_center / self.image_height
        width_norm = width / self.image_width
        height_norm = height / self.image_height
        
        # Ensure values are in [0, 1]
        x_center_norm = np.clip(x_center_norm, 0, 1)
        y_center_norm = np.clip(y_center_norm, 0, 1)
        width_norm = np.clip(width_norm, 0, 1)
        height_norm = np.clip(height_norm, 0, 1)
        
        return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
    
    def export_to_yolo_segmentation(self,
                                   mask: np.ndarray,
                                   class_id: int = 0,
                                   simplify: bool = True) -> Optional[str]:
        """
        Export mask to YOLO segmentation format (polygon).
        
        YOLO seg format: class_id x1 y1 x2 y2 ... xn yn (normalized)
        
        Args:
            mask: Boolean mask array
            class_id: Class ID for the object
            simplify: Whether to simplify the polygon
            
        Returns:
            Optional[str]: YOLO format string or None if no valid contour
            
        Example:
            >>> yolo_seg = tool.export_to_yolo_segmentation(mask)
        """
        # Find contours from mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify if requested
        if simplify:
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Need at least 3 points for valid polygon
        if len(largest_contour) < 3:
            return None
        
        # Normalize coordinates
        points = []
        for point in largest_contour:
            x = point[0][0] / self.image_width
            y = point[0][1] / self.image_height
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            points.extend([x, y])
        
        return f"{class_id} " + " ".join(f"{p:.6f}" for p in points)
    
    def export_to_coco_rle(self, mask: np.ndarray) -> Dict:
        """
        Export mask to COCO RLE (Run-Length Encoding) format.
        
        Args:
            mask: Boolean mask array
            
        Returns:
            Dict: RLE dictionary with 'counts' and 'size' keys
            
        Example:
            >>> rle = tool.export_to_coco_rle(mask)
            >>> print(rle['size'])  # [height, width]
        """
        from pycocotools import mask as mask_utils
        
        # Convert to Fortran order (column-major)
        mask_fortran = np.asfortranarray(mask.astype(np.uint8))
        
        # Encode to RLE
        rle = mask_utils.encode(mask_fortran)
        
        # Convert bytes to string for JSON serialization
        rle['counts'] = rle['counts'].decode('utf-8')
        
        return rle
    
    # ==================== YOLO Export Methods ====================
    
    def export_masks_to_yolo(self,
                            masks_dir: str,
                            original_images_dir: str,
                            output_dir: str,
                            format_type: str = "detection",
                            use_mask_id_as_class: bool = True) -> Dict:
        """
        Export SAM masks to YOLO format.
        
        Args:
            masks_dir: Directory containing SAM mask outputs
            original_images_dir: Directory containing original images
            output_dir: Output directory for YOLO labels
            format_type: "detection" for bboxes or "segmentation" for polygons
            use_mask_id_as_class: If True, use mask_id as class_id; if False, use 0 for all
            
        Returns:
            Dict: Export summary
        """
        masks_path = Path(masks_dir)
        images_path = Path(original_images_dir)
        output_path = Path(output_dir)
        
        # Create output directory
        labels_dir = output_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all mask JSON files
        json_files = list(masks_path.glob("*_mask_*.json"))
        
        # Group by image name
        image_masks = {}
        max_class_id = 0  # Track maximum class ID
        
        for json_file in json_files:
            # Extract image name (everything before _mask_XXX)
            name_parts = json_file.stem.rsplit('_mask_', 1)
            if len(name_parts) == 2:
                image_name = name_parts[0]
                if image_name not in image_masks:
                    image_masks[image_name] = []
                image_masks[image_name].append(json_file)
        
        # Process each image
        total_labels = 0
        for image_name, mask_files in image_masks.items():
            # Find original image to get dimensions
            image_file = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                test_path = images_path / f"{image_name}{ext}"
                if test_path.exists():
                    image_file = test_path
                    break
            
            if not image_file:
                print(f"Warning: Could not find original image for {image_name}")
                continue
            
            # Get image dimensions
            img = cv2.imread(str(image_file))
            if img is None:
                continue
            height, width = img.shape[:2]
            
            # Process masks for this image
            yolo_lines = []
            for json_file in sorted(mask_files):
                # Read metadata
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                
                # Determine class ID
                if use_mask_id_as_class:
                    class_id = metadata.get('mask_id', 0)
                else:
                    class_id = 0
                
                # Track max class ID for classes.txt
                max_class_id = max(max_class_id, class_id)
                
                if format_type == "detection":
                    # Convert bbox to YOLO detection format
                    bbox = metadata['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    # Calculate YOLO format
                    x_center = (x1 + x2) / 2.0 / width
                    y_center = (y1 + y2) / 2.0 / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    
                    # Ensure normalized
                    x_center = np.clip(x_center, 0, 1)
                    y_center = np.clip(y_center, 0, 1)
                    w = np.clip(w, 0, 1)
                    h = np.clip(h, 0, 1)
                    
                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                
                elif format_type == "segmentation":
                    # Load mask PNG for segmentation
                    mask_file = json_file.with_suffix('.png')
                    if mask_file.exists():
                        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            # Set image dimensions for export
                            self.image_width = width
                            self.image_height = height
                            
                            # Convert to YOLO segmentation with class_id
                            yolo_seg = self.export_to_yolo_segmentation(mask > 127, class_id=class_id)
                            if yolo_seg:
                                yolo_lines.append(yolo_seg)
            
            # Save YOLO label file
            if yolo_lines:
                label_file = labels_dir / f"{image_name}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                total_labels += len(yolo_lines)
        
        # Create classes.txt with appropriate number of classes
        classes_file = output_path / "classes.txt"
        with open(classes_file, 'w') as f:
            if use_mask_id_as_class:
                # Create class names based on max_class_id
                for i in range(max_class_id + 1):
                    f.write(f"object_{i}\n")
            else:
                f.write("object\n")  # Single class
        
        print(f"\n✅ Exported {total_labels} labels to YOLO format")
        print(f"📁 Labels saved to: {labels_dir}")
        print(f"📝 Classes file: {classes_file}")
        if use_mask_id_as_class:
            print(f"📊 Number of classes: {max_class_id + 1}")
        
        return {
            "total_images": len(image_masks),
            "total_labels": total_labels,
            "format": format_type,
            "num_classes": max_class_id + 1 if use_mask_id_as_class else 1
        }
    
    # ==================== Utility Methods ====================
    
    def _mask_to_bbox(self, mask: np.ndarray) -> List[int]:
        """
        Convert binary mask to bounding box.
        
        Args:
            mask: Boolean mask array
            
        Returns:
            List[int]: Bounding box as [x1, y1, x2, y2]
        """
        # Find all points where mask is True
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
        """
        Overlay a colored mask on an image.
        
        Args:
            image: Base image
            mask: Boolean mask
            color: RGB color for mask
            alpha: Transparency (0=transparent, 1=opaque)
            
        Returns:
            np.ndarray: Image with mask overlay
        """
        overlay = image.copy()
        overlay[mask] = overlay[mask] * (1 - alpha) + np.array(color) * alpha
        return overlay.astype(np.uint8)
    
    def _get_color_for_index(self, idx: int) -> Tuple[int, int, int]:
        """
        Get a consistent color for a given index.
        
        Args:
            idx: Index number
            
        Returns:
            Tuple[int, int, int]: RGB color
        """
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (128, 255, 0),  # Lime
            (255, 0, 128),  # Pink
        ]
        return colors[idx % len(colors)]
    
    def _find_images_in_folder(self, folder: Path) -> List[Path]:
        """
        Find all image files in a folder.
        
        Args:
            folder: Path to folder
            
        Returns:
            List[Path]: List of image file paths
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        for ext in image_extensions:
            # Check both lowercase and uppercase extensions
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))
        
        return sorted(image_files)
    
    def _print_interactive_instructions(self):
        """Print instructions for interactive mode."""
        print("\n" + "="*50)
        print("INTERACTIVE SELECTION CONTROLS:")
        print("-"*50)
        print("🖱️  Left Click        : Add object point (green)")
        print("🖱️  Right Click       : Add background point (red)")
        print("🖱️  Ctrl+Drag        : Draw bounding box")
        print("⌨️  'a' / 'Enter'    : Accept and save current mask")
        print("⌨️  'r'              : Reset current selection")
        print("⌨️  'u'              : Undo last saved mask")
        print("⌨️  's'              : Skip this image")
        print("⌨️  'q' / 'ESC'      : Next image / Finish")
        print("="*50 + "\n")
    
    def _save_interactive_masks(self,
                               masks: List[Dict],
                               image_name: str,
                               masks_dir: Path,
                               viz_dir: Path):
        """
        Save masks from interactive selection.
        
        Args:
            masks: List of mask dictionaries
            image_name: Base name for files
            masks_dir: Directory for mask files
            viz_dir: Directory for visualizations
        """
        for i, mask_data in enumerate(masks):
            # Save mask as PNG
            mask_filename = masks_dir / f"{image_name}_mask_{i:03d}.png"
            cv2.imwrite(str(mask_filename),
                       (mask_data['mask'] * 255).astype(np.uint8))
            
            # Save metadata as JSON
            metadata = {
                "mask_id": i,
                "bbox": mask_data['bbox'],
                "area": mask_data['area'],
                "score": 0.0  # Placeholder for compatibility
            }
            
            meta_filename = masks_dir / f"{image_name}_mask_{i:03d}.json"
            with open(meta_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Create and save composite visualization
        if self.original_image is not None:
            composite = self._create_composite_visualization(masks)
            viz_filename = viz_dir / f"{image_name}_composite.png"
            cv2.imwrite(str(viz_filename), composite)
    
    def _create_composite_visualization(self, masks: List[Dict]) -> np.ndarray:
        """
        Create a composite visualization with all masks.
        
        Args:
            masks: List of mask dictionaries
            
        Returns:
            np.ndarray: BGR image with all masks overlaid
        """
        composite = self.original_image.copy()
        
        for i, mask_data in enumerate(masks):
            color = self._get_color_for_index(i)
            mask = mask_data['mask']
            composite = self._overlay_mask(composite, mask, color, alpha=0.4)
            
            # Add label
            x1, y1, x2, y2 = mask_data['bbox']
            label = f"Mask {i+1}"
            cv2.putText(composite, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return cv2.cvtColor(composite.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # ==================== High-Level Processing Methods ====================
    
    def process_auto(self,
                    input_path: str,
                    output_dir: str,
                    min_area: int = 100,
                    max_objects: Optional[int] = None) -> Dict:
        """
        Process images with automatic segmentation.
        
        Args:
            input_path: Path to image or folder
            output_dir: Output directory
            min_area: Minimum mask area in pixels
            max_objects: Maximum number of objects per image
            
        Returns:
            Dict: Processing summary
        """
        input_path = Path(input_path)
        output_path = Path(output_dir)
        
        # Setup paths
        if input_path.is_file():
            image_files = [input_path]
        else:
            image_files = self._find_images_in_folder(input_path)
        
        if not image_files:
            print(f"No images found")
            return {"total_images": 0, "total_masks": 0}
        
        # Create output directories
        masks_dir = output_path / "masks"
        viz_dir = output_path / "visualizations"
        masks_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        total_masks = 0
        results = []
        
        for image_file in tqdm(image_files, desc="Processing images"):
            # Load image
            image = cv2.imread(str(image_file))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.set_image(image_rgb)
            self.original_image = image_rgb
            
            # Generate masks
            masks = self.auto_generate_masks(
                min_mask_region_area=min_area,
                max_objects=max_objects
            )
            
            # Save masks
            image_name = image_file.stem
            for i, mask_dict in enumerate(masks):
                # Convert mask format
                mask_data = {
                    'mask': mask_dict['segmentation'],
                    'bbox': self._mask_to_bbox(mask_dict['segmentation']),
                    'area': mask_dict['area']
                }
                
                # Save files
                mask_filename = masks_dir / f"{image_name}_mask_{i:03d}.png"
                cv2.imwrite(str(mask_filename),
                           (mask_data['mask'] * 255).astype(np.uint8))
                
                # Save metadata
                metadata = {
                    "mask_id": i,
                    "bbox": mask_data['bbox'],
                    "area": mask_data['area'],
                    "score": mask_dict.get('predicted_iou', 0.0)
                }
                
                meta_filename = masks_dir / f"{image_name}_mask_{i:03d}.json"
                with open(meta_filename, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            total_masks += len(masks)
            results.append({
                "image": image_file.name,
                "masks_generated": len(masks)
            })
        
        # Save summary
        summary = {
            "total_images": len(image_files),
            "total_masks": total_masks,
            "mode": "automatic",
            "results": results
        }
        
        summary_path = output_path / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Processed {len(image_files)} images")
        print(f"📊 Generated {total_masks} masks")
        print(f"📁 Output saved to: {output_path}")
        
        return summary


def main():
    """
    Main function for command-line interface.
    
    Provides a comprehensive CLI for all SAM operations including
    interactive selection, automatic segmentation, and various
    export formats.
    """
    parser = argparse.ArgumentParser(
        description="SAM Tool - Complete segmentation toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive selection (single image or folder):
    python sam_tool.py input.jpg output/ --mode interactive
    python sam_tool.py images/ output/ --mode interactive
    
  Automatic segmentation:
    python sam_tool.py images/ output/ --mode auto --min-area 500
    
  Box selection:
    python sam_tool.py image.jpg output/ --mode box --box 100 100 400 400
    
  Point selection:
    python sam_tool.py image.jpg output/ --mode points --points 250 300 450 320
        """
    )
    
    # Required arguments
    parser.add_argument("input",
                       help="Input image file or folder containing images")
    parser.add_argument("output",
                       help="Output directory for masks and metadata")
    
    # Mode selection
    parser.add_argument("--mode",
                       choices=["interactive", "auto", "box", "points"],
                       default="interactive",
                       help="Processing mode (default: interactive)")
    
    # Model configuration
    parser.add_argument("--model",
                       choices=["vit_b", "vit_l", "vit_h"],
                       default="vit_b",
                       help="SAM model size: vit_b (fast), vit_l (balanced), vit_h (best)")
    
    parser.add_argument("--device",
                       choices=["auto", "mps", "cuda", "cpu"],
                       default="auto",
                       help="Computing device (default: auto-detect)")
    
    # Mode-specific arguments
    parser.add_argument("--box",
                       nargs=4,
                       type=int,
                       metavar=("X1", "Y1", "X2", "Y2"),
                       help="Bounding box coordinates for box mode")
    
    parser.add_argument("--points",
                       nargs='+',
                       type=int,
                       metavar="COORD",
                       help="Point coordinates (x1 y1 x2 y2 ...) for points mode")
    
    # Auto mode arguments
    parser.add_argument("--min-area",
                       type=int,
                       default=100,
                       help="Minimum mask area in pixels for auto mode (default: 100)")
    
    parser.add_argument("--max-objects",
                       type=int,
                       help="Maximum number of objects to detect per image in auto mode")
    
    # Add YOLO export mode
    parser.add_argument("--export-yolo",
                       action="store_true",
                       help="Export masks to YOLO format after processing")
    
    parser.add_argument("--yolo-format",
                       choices=["detection", "segmentation"],
                       default="detection",
                       help="YOLO format type (default: detection)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize the tool
    print(f"Initializing SAM Tool...")
    tool = SAMTool(
        model_type=args.model,
        device=args.device
    )
    
    # Process based on mode
    try:
        if args.mode == "interactive":
            # Interactive selection mode
            tool.process_interactive(args.input, args.output)
            
            # Export to YOLO if requested
            if args.export_yolo:
                print("\nExporting to YOLO format...")
                input_path = Path(args.input)
                
                # Determine original images directory
                if input_path.is_file():
                    images_dir = input_path.parent
                else:
                    images_dir = input_path
                
                # Export masks to YOLO
                tool.export_masks_to_yolo(
                    masks_dir=Path(args.output) / "masks",
                    original_images_dir=images_dir,
                    output_dir=Path(args.output) / "yolo_labels",
                    format_type=args.yolo_format
                )
            
        elif args.mode == "auto":
            # Automatic segmentation mode
            tool.process_auto(
                args.input,
                args.output,
                min_area=args.min_area,
                max_objects=args.max_objects
            )
            
        elif args.mode == "box":
            # Box selection mode
            if not args.box:
                print("Error: --box coordinates required for box mode")
                return
            
            # Process single image with box
            image = cv2.imread(args.input)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tool.set_image(image_rgb)
            
            result = tool.select_with_box(args.box)
            
            # Save result
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            mask_file = output_path / "mask.png"
            cv2.imwrite(str(mask_file), (result['mask'] * 255).astype(np.uint8))
            
            print(f"✅ Mask saved to: {mask_file}")
            print(f"   Area: {result['area']} pixels")
            print(f"   Score: {result['score']:.3f}")
            
        elif args.mode == "points":
            # Points selection mode
            if not args.points or len(args.points) % 2 != 0:
                print("Error: --points requires pairs of x,y coordinates")
                return
            
            # Convert flat list to points
            points = [(args.points[i], args.points[i+1])
                     for i in range(0, len(args.points), 2)]
            
            # Process single image with points
            image = cv2.imread(args.input)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tool.set_image(image_rgb)
            
            result = tool.select_with_points(points)
            
            # Save result
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            mask_file = output_path / "mask.png"
            cv2.imwrite(str(mask_file), (result['mask'] * 255).astype(np.uint8))
            
            print(f"✅ Mask saved to: {mask_file}")
            print(f"   Area: {result['area']} pixels")
            print(f"   Score: {result['score']:.3f}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()