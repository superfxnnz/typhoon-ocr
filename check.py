import torch
import sys

print(f"--- System Check ---")
print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    prop = torch.cuda.get_device_properties(0)
    print(f"GPU: {prop.name}")
    print(f"Compute Capability: {prop.major}.{prop.minor}")
    
    # เช็คว่าลง Flash Attention ได้ไหม
    if prop.major >= 8:
        print("GPU รองรับ Flash Attention (Ampere ขึ้นไป)")
    else:
        print("GPU เก่าไปสำหรับ Flash Attention (ต้องใช้ SDPA แทน)")