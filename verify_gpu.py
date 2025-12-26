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