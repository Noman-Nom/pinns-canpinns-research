"""
CUDA initialization utility to suppress warnings and ensure proper context setup.
"""

import torch
import warnings
import os

def init_cuda(device_id=0, suppress_warnings=True):
    """
    Initialize CUDA context and suppress warnings.
    
    Args:
        device_id: GPU device ID to use (default: 0)
        suppress_warnings: If True, suppress cuBLAS warnings
    
    Returns:
        device: torch.device object
    """
    if suppress_warnings:
        # Suppress the cuBLAS warning
        warnings.filterwarnings('ignore', message='.*cuBLAS.*')
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Using CPU.")
        return torch.device('cpu')
    
    # Set device
    device = torch.device(f'cuda:{device_id}')
    
    # Initialize CUDA context by creating a dummy tensor
    # This ensures the context is set before any operations
    try:
        _ = torch.zeros(1, device=device)
        print(f"✓ CUDA context initialized on device {device_id}: {torch.cuda.get_device_name(device_id)}")
    except Exception as e:
        print(f"Warning: Could not initialize CUDA context: {e}")
        return torch.device('cpu')
    
    return device

# Initialize CUDA when module is imported
if torch.cuda.is_available():
    _ = init_cuda(suppress_warnings=True)

