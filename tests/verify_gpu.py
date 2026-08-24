#!/usr/bin/env python3
"""
GPU Verification Script for CAN-PINNs Environment
This script verifies that CUDA and GPU acceleration are working properly
for both PyTorch and TensorFlow.
"""

import sys

def verify_pytorch():
    """Verify PyTorch CUDA setup"""
    print("=" * 60)
    print("PyTorch GPU Verification")
    print("=" * 60)
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}:")
                print(f"  - Name: {torch.cuda.get_device_name(i)}")
                print(f"  - Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                print(f"  - Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
            
            # Test GPU computation
            print("\nTesting GPU computation...")
            device = torch.device("cuda:0")
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.matmul(x, y)
            print("✓ GPU computation test passed!")
            return True
        else:
            print("✗ CUDA is not available. GPU acceleration will not work.")
            return False
    except ImportError:
        print("✗ PyTorch is not installed")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def verify_tensorflow():
    """Verify TensorFlow GPU setup"""
    print("\n" + "=" * 60)
    print("TensorFlow GPU Verification")
    print("=" * 60)
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✓ GPU devices found: {len(gpus)}")
        
        if len(gpus) > 0:
            for i, gpu in enumerate(gpus):
                print(f"\nGPU {i}:")
                print(f"  - Name: {gpu.name}")
                print(f"  - Details: {gpu}")
            
            # Test GPU computation
            print("\nTesting GPU computation...")
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
            print(f"✓ GPU computation test passed!")
            print(f"  Result: {c.numpy()}")
            return True
        else:
            print("✗ No GPU devices found. GPU acceleration will not work.")
            return False
    except ImportError:
        print("✗ TensorFlow is not installed")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def verify_other_libraries():
    """Verify other required libraries"""
    print("\n" + "=" * 60)
    print("Other Libraries Verification")
    print("=" * 60)
    
    libraries = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
    }
    
    all_ok = True
    for lib_name, import_name in libraries.items():
        try:
            lib = __import__(import_name)
            version = getattr(lib, '__version__', 'unknown')
            print(f"✓ {lib_name}: {version}")
        except ImportError:
            print(f"✗ {lib_name}: Not installed")
            all_ok = False
    
    return all_ok

def main():
    """Main verification function"""
    print("\n" + "=" * 60)
    print("CAN-PINNs Environment GPU Verification")
    print("=" * 60)
    print()
    
    pytorch_ok = verify_pytorch()
    tensorflow_ok = verify_tensorflow()
    other_ok = verify_other_libraries()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"PyTorch GPU: {'✓ Working' if pytorch_ok else '✗ Not working'}")
    print(f"TensorFlow GPU: {'✓ Working' if tensorflow_ok else '✗ Not working'}")
    print(f"Other libraries: {'✓ All installed' if other_ok else '✗ Some missing'}")
    
    if pytorch_ok and tensorflow_ok and other_ok:
        print("\n✓ All checks passed! Environment is ready for CAN-PINNs development.")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

