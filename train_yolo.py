#!/usr/bin/env python3
"""
YOLOv8 Training Script for Custom Object Detection
Trains on 984 classes from Finalized List.csv
"""

from ultralytics import YOLO
import yaml
import pandas as pd
from pathlib import Path
import torch

def create_data_yaml(csv_path: str, dataset_path: str, output_path: str = "data.yaml"):
    """
    Create YOLO data.yaml from your Finalized List.csv
    
    Args:
        csv_path: Path to Finalized List.csv
        dataset_path: Path to dataset root directory
        output_path: Output path for data.yaml
    """
    # Read class list
    df = pd.read_csv(csv_path)
    
    # Create names dictionary
    names = {}
    for _, row in df.iterrows():
        names[int(row['class_id'])] = row['class_name']
    
    # Create YAML structure
    data = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(names),
        'names': names
    }
    
    # Save YAML
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created {output_path} with {len(names)} classes")
    return output_path

def train_yolo_model(
    data_yaml: str = "data.yaml",
    model_size: str = "yolov8x",  # n, s, m, l, x
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "auto",
    resume: bool = False
):
    """
    Train YOLOv8 model
    
    Args:
        data_yaml: Path to data.yaml
        model_size: Model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        epochs: Number of training epochs
        batch_size: Batch size (reduce if out of memory)
        img_size: Input image size
        device: Device to train on (auto, cpu, mps, cuda, 0, 1, etc.)
        resume: Resume from last checkpoint
    """
    print("\n" + "="*60)
    print("YOLOv8 Training Configuration")
    print("="*60)
    print(f"Model: {model_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {img_size}")
    print(f"Device: {device}")
    
    # Check device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Using device: {device}")
    print("="*60 + "\n")
    
    # Load model
    model = YOLO(f'{model_size}.pt')  # Load pretrained model
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        workers=8,
        patience=50,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        cache=False,  # Cache images for faster training (use True if enough RAM)
        pretrained=True,
        optimizer='auto',  # SGD, Adam, AdamW, etc.
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=resume,
        amp=True,  # Automatic Mixed Precision
        fraction=1.0,  # Train on 100% of data
        profile=False,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0
    )
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"📁 Results saved to: runs/detect/train")
    print(f"📊 Best model: runs/detect/train/weights/best.pt")
    print(f"📊 Last model: runs/detect/train/weights/last.pt")
    print("="*60 + "\n")
    
    return results

def validate_model(model_path: str = "runs/detect/train/weights/best.pt", 
                   data_yaml: str = "data.yaml"):
    """Validate trained model"""
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    return results

def export_model(model_path: str = "runs/detect/train/weights/best.pt",
                format: str = "onnx"):
    """
    Export model to different formats
    
    Formats: onnx, torchscript, coreml, tflite, etc.
    """
    model = YOLO(model_path)
    model.export(format=format)
    print(f"✓ Model exported to {format} format")

def predict_sample(model_path: str, image_path: str):
    """Test model on a sample image"""
    model = YOLO(model_path)
    results = model(image_path)
    results[0].show()  # Display results
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 on custom dataset")
    parser.add_argument("--mode", choices=["create_yaml", "train", "validate", "export", "predict"],
                       default="train", help="Operation mode")
    parser.add_argument("--csv", default="Finalized List.csv", 
                       help="Path to class CSV file")
    parser.add_argument("--dataset", default="yolo_dataset",
                       help="Path to dataset directory")
    parser.add_argument("--data-yaml", default="data.yaml",
                       help="Path to data.yaml file")
    parser.add_argument("--model", default="yolov8x",
                       choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                       help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--device", default="auto",
                       help="Device (auto, cpu, mps, cuda, 0, 1, etc.)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from last checkpoint")
    parser.add_argument("--model-path", default="runs/detect/train/weights/best.pt",
                       help="Model path for validation/export/predict")
    parser.add_argument("--image", help="Image path for prediction")
    parser.add_argument("--export-format", default="onnx",
                       help="Export format (onnx, torchscript, coreml, etc.)")
    
    args = parser.parse_args()
    
    if args.mode == "create_yaml":
        # Create data.yaml from CSV
        create_data_yaml(args.csv, args.dataset, args.data_yaml)
        
    elif args.mode == "train":
        # Train model
        train_yolo_model(
            data_yaml=args.data_yaml,
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            device=args.device,
            resume=args.resume
        )
        
    elif args.mode == "validate":
        # Validate model
        results = validate_model(args.model_path, args.data_yaml)
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        
    elif args.mode == "export":
        # Export model
        export_model(args.model_path, args.export_format)
        
    elif args.mode == "predict":
        # Predict on image
        if not args.image:
            print("Error: --image required for prediction mode")
        else:
            results = predict_sample(args.model_path, args.image)
            print(f"✓ Detection complete. Results displayed.")

