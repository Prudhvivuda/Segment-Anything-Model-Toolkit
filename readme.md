# SAM Tool - Segment Anything Model Toolkit

A comprehensive, production-ready toolkit for image segmentation using Meta's Segment Anything Model (SAM). This tool provides interactive selection, automatic segmentation, and multiple export formats including YOLO for training object detection models.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![SAM](https://img.shields.io/badge/SAM-Meta%20AI-purple)
![CUDA](https://img.shields.io/badge/CUDA-11.7%2B-green)
![MPS](https://img.shields.io/badge/Apple-Silicon-black)

## 🌟 Features

- **Interactive Selection Mode** - Click to select specific objects in images
- **Automatic Segmentation** - Detect all objects automatically
- **Batch Processing** - Process entire folders of images
- **Multiple Export Formats** - YOLO, COCO RLE, binary masks
- **Cross-Platform GPU Support** - Apple Silicon (MPS), NVIDIA (CUDA), and CPU
- **Production Ready** - Comprehensive error handling and progress tracking

## 📋 Table of Contents

- [Installation](#-installation)
- [GPU Setup and Usage](#-gpu-setup-and-usage)
  - [Apple Silicon (M1/M2/M3)](#apple-silicon-m1m2m3-mps)
  - [NVIDIA GPUs](#nvidia-gpus-cuda)
  - [Device Selection](#device-selection)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Performance Comparison](#-performance-comparison)
- [Troubleshooting](#-troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV
- PyTorch with appropriate backend support:
  - **MPS** for Apple Silicon (M1/M2/M3)
  - **CUDA** for NVIDIA GPUs
  - **CPU** as fallback

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/sam-tool.git
cd sam-tool
```

### Step 2: Install Dependencies

#### For Apple Silicon (M1/M2/M3 Macs):
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install other requirements
pip install opencv-python segment-anything matplotlib numpy Pillow tqdm pycocotools
```

#### For NVIDIA GPUs:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install opencv-python segment-anything matplotlib numpy Pillow tqdm pycocotools
```

#### For CPU only:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install opencv-python segment-anything matplotlib numpy Pillow tqdm pycocotools
```

### Step 3: Verify GPU Setup

```python
# verify_gpu.py
import torch

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check CUDA availability (NVIDIA)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")

# Check MPS availability (Apple Silicon)
print(f"MPS available: {torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    print("Apple Silicon GPU detected")

# Test tensor creation
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"\nUsing device: {device}")
test_tensor = torch.randn(3, 3).to(device)
print(f"Test tensor device: {test_tensor.device}")
```

Run verification:
```bash
python verify_gpu.py
```

### Step 4: Download SAM Checkpoints (Automatic)

The tool will automatically download the required SAM model checkpoint on first use. Models are cached in `~/.cache/sam/`.

Available models:
- **vit_b** (375MB) - Fast, good quality, best for real-time
- **vit_l** (1.2GB) - Balanced speed and quality
- **vit_h** (2.5GB) - Best quality, slower, needs more VRAM

## 🎮 GPU Setup and Usage

### Apple Silicon (M1/M2/M3) MPS

Metal Performance Shaders (MPS) backend provides GPU acceleration on Mac:

```bash
# Use MPS explicitly
python sam_tool.py images/ output/ --mode interactive --device mps

# Let tool auto-detect (will use MPS if available)
python sam_tool.py images/ output/ --mode interactive --device auto
```

**MPS Optimization Tips:**
- Use `vit_b` model for best speed/quality balance
- Process images in batches for better efficiency
- Close other GPU-intensive applications

### NVIDIA GPUs (CUDA)

CUDA provides acceleration on NVIDIA GPUs:

```bash
# Use CUDA explicitly
python sam_tool.py images/ output/ --mode interactive --device cuda

# Use specific GPU (for multi-GPU systems)
CUDA_VISIBLE_DEVICES=0 python sam_tool.py images/ output/ --mode interactive --device cuda

# Auto-detect (will prefer CUDA if available)
python sam_tool.py images/ output/ --mode interactive --device auto
```

**CUDA Optimization Tips:**
- Use `vit_h` model if you have ≥8GB VRAM
- Enable mixed precision for faster processing
- Monitor GPU memory with `nvidia-smi`

### Device Selection

The `--device` flag controls which processor to use:

```bash
# Auto-detect best available (recommended)
python sam_tool.py images/ output/ --mode interactive --device auto

# Force specific device
python sam_tool.py images/ output/ --mode interactive --device mps   # Apple Silicon
python sam_tool.py images/ output/ --mode interactive --device cuda  # NVIDIA GPU
python sam_tool.py images/ output/ --mode interactive --device cpu   # CPU only
```

**Auto-detection priority:**
1. CUDA (if available)
2. MPS (if available)
3. CPU (fallback)

## ⚡ Quick Start

### Interactive Object Selection with GPU

```bash
# Apple Silicon Mac (M1/M2/M3)
python sam_tool.py image.jpg output/ --mode interactive --device mps --model vit_b

# NVIDIA GPU
python sam_tool.py image.jpg output/ --mode interactive --device cuda --model vit_l

# Auto-detect and use best available
python sam_tool.py image.jpg output/ --mode interactive --device auto

# Process folder with GPU acceleration
python sam_tool.py images/ output/ --mode interactive --device auto --export-yolo
```

### Automatic Segmentation with GPU

```bash
# Fast processing on Apple Silicon
python sam_tool.py images/ output/ --mode auto --device mps --min-area 500

# High quality on NVIDIA GPU
python sam_tool.py images/ output/ --mode auto --device cuda --model vit_h --max-objects 20

# Batch processing with optimal settings
python sam_tool.py images/ output/ --mode auto --device auto --model vit_b
```

## 📖 Usage Guide

### Interactive Mode

The interactive mode provides a GUI for manually selecting objects in images.

```bash
python sam_tool.py input_path output_dir --mode interactive [options]
```

**GPU-Specific Options:**
- `--device {auto,mps,cuda,cpu}` - Select processing device
- `--model {vit_b,vit_l,vit_h}` - Model size (larger = better quality, more VRAM)

**Controls:**
- **Left Click** - Add positive point (part of object)
- **Right Click** - Add negative point (not part of object)
- **Ctrl+Drag** - Draw bounding box around object
- **'a' or Enter** - Accept and save current mask
- **'r'** - Reset current selection
- **'u'** - Undo last saved mask
- **'s'** - Skip current image
- **'q' or ESC** - Move to next image / Finish

**Examples with GPU:**
```bash
# Apple Silicon - Process with MPS acceleration
python sam_tool.py "dataset/image.jpg" output/ --mode interactive --device mps

# NVIDIA GPU - Use larger model for quality
python sam_tool.py images/ output/ --mode interactive --device cuda --model vit_h

# Auto-detect GPU and process batch
python sam_tool.py images/ output/ --mode interactive --device auto --export-yolo
```

### Automatic Mode

Automatically segment all objects in images without user interaction.

```bash
python sam_tool.py input_path output_dir --mode auto [options]
```

**Options:**
- `--min-area` - Minimum mask area in pixels (default: 100)
- `--max-objects` - Maximum objects per image
- `--device` - GPU/CPU selection
- `--model` - Model size selection

**Examples:**
```bash
# Fast processing on GPU
python sam_tool.py images/ output/ --mode auto --device auto --min-area 500

# High quality segmentation on NVIDIA GPU
python sam_tool.py images/ output/ --mode auto --device cuda --model vit_h

# Efficient batch on Apple Silicon
python sam_tool.py images/ output/ --mode auto --device mps --model vit_b --max-objects 10
```

### Box Selection

Select objects using bounding box coordinates.

```bash
python sam_tool.py image.jpg output/ --mode box --box x1 y1 x2 y2 --device auto
```

**Example:**
```bash
# With GPU acceleration
python sam_tool.py photo.jpg output/ --mode box --box 100 100 400 400 --device cuda
```


## 🔧 Python API with GPU

```python
from sam_tool import SAMTool
import cv2

# Initialize with specific device
tool = SAMTool(
    model_type="vit_b",
    device="mps"  # or "cuda", "cpu", "auto"
)

# Check which device is being used
print(f"Using device: {tool.device}")

# Process image
image = cv2.imread("photo.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
tool.set_image(image_rgb)

# Select object
result = tool.select_with_points([(100, 200)])
mask = result['mask']
```

## 🐛 Troubleshooting

### Apple Silicon (MPS) Issues

**Problem: "MPS not available"**
```bash
# Solution: Update PyTorch
pip install --upgrade torch torchvision torchaudio

# Verify MPS
python -c "import torch; print(torch.backends.mps.is_available())"
```

**Problem: "MPS out of memory"**
```bash
# Solution: Use smaller model or reduce batch size
python sam_tool.py images/ output/ --device mps --model vit_b
```

### NVIDIA GPU (CUDA) Issues

**Problem: "CUDA out of memory"**
```bash
# Solution 1: Use smaller model
python sam_tool.py images/ output/ --device cuda --model vit_b

# Solution 2: Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Solution 3: Reduce batch size or image resolution
```

**Problem: "CUDA not available"**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### General GPU Tips

1. **Monitor GPU Usage:**
   - Apple Silicon: Use Activity Monitor → GPU History
   - NVIDIA: Use `nvidia-smi -l 1` for real-time monitoring

2. **Optimize for Speed:**
   - Use `vit_b` model for real-time processing
   - Process images in batches
   - Lower `points_per_side` in auto mode

3. **Optimize for Quality:**
   - Use `vit_h` model when VRAM permits
   - Increase `points_per_side` for thorough detection
   - Use interactive mode for precise selection

## 📚 Additional Resources

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [SAM Paper](https://arxiv.org/abs/2304.02643)

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.