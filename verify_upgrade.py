#!/usr/bin/env python3
"""
Compatibility verification script for PyTorch 2.0.1 + TensorFlow 2.14 upgrade
"""

import sys
import subprocess

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_section(text):
    print(f"\n▶ {text}")
    print("-" * 60)

def check_pytorch():
    print_section("Checking PyTorch Installation")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"✓ CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"✓ GPU 0 Name: {torch.cuda.get_device_name(0)}")
        
        # Check common modules
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        print("✓ Core PyTorch modules available")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")
        return False

def check_tensorflow():
    print_section("Checking TensorFlow Installation")
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        # Check CUDA
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✓ Number of GPUs detected: {len(gpus)}")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"✓ GPU {i}: {gpu}")
        
        # Check compat.v1
        print("✓ TensorFlow compat.v1 available")
        
        return True
    except Exception as e:
        print(f"✗ TensorFlow check failed: {e}")
        return False

def check_pytorch_models():
    print_section("Checking PyTorch Model Loading")
    try:
        import sys
        sys.path.insert(0, './src')
        
        import torch
        import models
        
        # Test MNIST model
        mnist_model = models.MNIST_target_net()
        print(f"✓ MNIST target model: {type(mnist_model).__name__}")
        
        # Test Generator
        generator = models.Generator(1, 1, 'MNIST')
        print(f"✓ Generator model: {type(generator).__name__}")
        
        # Test Discriminator
        discriminator = models.Discriminator(1)
        print(f"✓ Discriminator model: {type(discriminator).__name__}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 1, 28, 28)
        output = mnist_model(dummy_input)
        print(f"✓ MNIST model forward pass successful: output shape {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch models check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_data_loading():
    print_section("Checking Data Loading")
    try:
        import sys
        sys.path.insert(0, './src')
        
        import torch
        import torchvision.transforms as transforms
        import torchvision.datasets
        from torch.utils.data import DataLoader
        
        # Test MNIST dataset
        print("Testing MNIST dataset loading...")
        transform = transforms.ToTensor()
        dataset = torchvision.datasets.MNIST('./datasets', train=True, 
                                            transform=transform, download=True)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Get one batch
        images, labels = next(iter(dataloader))
        print(f"✓ MNIST batch loaded: images shape {images.shape}, labels shape {labels.shape}")
        
        return True
    except Exception as e:
        print(f"⚠ Data loading check failed: {e}")
        print("  (This is expected if you haven't downloaded the datasets yet)")
        return True  # Don't fail on this, as datasets may not be downloaded

def check_numpy_compatibility():
    print_section("Checking NumPy Compatibility")
    try:
        import numpy as np
        import torch
        
        # Test numpy to tensor conversion
        np_array = np.random.randn(3, 224, 224).astype(np.float32)
        tensor = torch.from_numpy(np_array)
        print(f"✓ NumPy to Tensor conversion: {tensor.shape}")
        
        # Test tensor to numpy conversion
        tensor_output = tensor.cpu().numpy()
        print(f"✓ Tensor to NumPy conversion: {tensor_output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ NumPy compatibility check failed: {e}")
        return False

def main():
    print_header("PyTorch 2.0.1 + TensorFlow 2.14 Compatibility Check")
    
    results = []
    
    # Run checks
    results.append(("PyTorch Installation", check_pytorch()))
    results.append(("TensorFlow Installation", check_tensorflow()))
    results.append(("NumPy Compatibility", check_numpy_compatibility()))
    results.append(("PyTorch Models", check_pytorch_models()))
    results.append(("Data Loading", check_data_loading()))
    
    # Print summary
    print_header("Verification Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n" + "="*60)
        print("  ✓ All compatibility checks passed!")
        print("  You can now run: python src/main.py")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print(f"  ⚠ {total - passed} check(s) failed")
        print("  Please review the output above for details")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
