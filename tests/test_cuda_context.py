"""
Test script to verify CUDA context is properly initialized.
This helps diagnose the cuBLAS warning message.
"""

import torch
import torch.nn as nn

print("="*60)
print("CUDA Context Test")
print("="*60)

# Check basic CUDA availability
print(f"\n1. Basic CUDA Check:")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   Device name: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    
    # Test creating a tensor on GPU
    print(f"\n2. Creating tensor on GPU:")
    try:
        x = torch.randn(10, 10).cuda()
        print(f"   ✓ Successfully created tensor on GPU")
        print(f"   Tensor device: {x.device}")
        
        # Test a simple operation
        print(f"\n3. Testing GPU operations:")
        y = torch.randn(10, 10).cuda()
        z = torch.matmul(x, y)
        print(f"   ✓ Matrix multiplication on GPU successful")
        print(f"   Result device: {z.device}")
        print(f"   Result shape: {z.shape}")
        
        # Test autograd
        print(f"\n4. Testing autograd on GPU:")
        x.requires_grad_(True)
        y.requires_grad_(True)
        z = torch.matmul(x, y)
        loss = z.sum()
        loss.backward()
        print(f"   ✓ Autograd on GPU successful")
        print(f"   x.grad device: {x.grad.device if x.grad is not None else 'None'}")
        
        # Test neural network forward/backward
        print(f"\n5. Testing neural network on GPU:")
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        ).cuda()
        
        output = model(x[:, :10])
        loss = output.sum()
        loss.backward()
        print(f"   ✓ Neural network forward/backward on GPU successful")
        
        print(f"\n" + "="*60)
        print("✓ All CUDA tests passed!")
        print("="*60)
        print("\nNote: The cuBLAS warning is usually harmless.")
        print("It occurs when PyTorch tries to use CUDA before the context is fully initialized.")
        print("The context is automatically set, and CUDA operations work correctly.")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n✗ CUDA is not available!")
    print("   Please check your GPU drivers and CUDA installation.")

