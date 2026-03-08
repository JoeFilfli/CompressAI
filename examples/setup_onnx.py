"""
Quick setup script to install ONNX Runtime and verify installation
Run this first before using ONNX optimization
"""
import subprocess
import sys

def install_onnx():
    """Install ONNX Runtime and dependencies"""
    
    print("="*60)
    print("ONNX Runtime Setup")
    print("="*60)
    print()
    
    packages = [
        "onnx>=1.12.0",
        "onnxscript>=0.1.0",    # Required for PyTorch ONNX export
        "onnxruntime>=1.13.0",  # CPU version
        # "onnxruntime-gpu>=1.13.0",  # GPU version (uncomment if you have CUDA)
    ]
    
    print("Installing required packages...")
    print()
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed\n")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}\n")
            return False
    
    # Verify installation
    print("\nVerifying installation...")
    try:
        import onnx
        import onnxruntime as ort
        print(f"✓ ONNX version: {onnx.__version__}")
        print(f"✓ ONNX Runtime version: {ort.__version__}")
        print(f"✓ Available providers: {ort.get_available_providers()}")
        print("\n" + "="*60)
        print("✅ ONNX Runtime setup complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Export model: python examples/export_to_onnx.py")
        print("2. Use ONNX: python examples/encode_folder_onnx.py")
        print()
        return True
    except ImportError as e:
        print(f"✗ Verification failed: {e}")
        return False


if __name__ == "__main__":
    success = install_onnx()
    sys.exit(0 if success else 1)
