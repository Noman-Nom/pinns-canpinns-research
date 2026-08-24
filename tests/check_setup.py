"""
Setup Verification Script
Checks that all dependencies are correctly installed and accessible.

Usage:
    python check_setup.py
"""

import sys
from importlib import import_module

print("\n" + "="*70)
print("  PINN Project Setup Verification")
print("="*70 + "\n")

# List of required packages
REQUIRED_PACKAGES = {
    'torch': 'PyTorch (neural networks)',
    'numpy': 'NumPy (numerical computing)',
    'scipy': 'SciPy (scientific computing)',
    'matplotlib': 'Matplotlib (plotting)',
    'sympy': 'SymPy (symbolic math)',
    'mcp': 'MCP (Model Context Protocol)',
    'pytest': 'pytest (testing)',
}

OPTIONAL_PACKAGES = {
    'sklearn': 'scikit-learn (ML utilities)',
    'tqdm': 'tqdm (progress bars)',
}

# Track results
passed = 0
failed = 0
warnings = 0

def check_package(package_name: str, display_name: str) -> bool:
    """Check if a package can be imported."""
    global passed, failed
    try:
        module = import_module(package_name)
        version = getattr(module, '__version__', 'unknown version')
        print(f"✓ {display_name:<40} {version}")
        passed += 1
        return True
    except ImportError as e:
        print(f"✗ {display_name:<40} NOT INSTALLED")
        failed += 1
        return False

# Check required packages
print("REQUIRED PACKAGES:")
print("-" * 70)
for pkg, desc in REQUIRED_PACKAGES.items():
    check_package(pkg, desc)

print("\nOPTIONAL PACKAGES:")
print("-" * 70)
for pkg, desc in OPTIONAL_PACKAGES.items():
    check_package(pkg, desc)

# Check virtual environment
print("\nVIRTUAL ENVIRONMENT:")
print("-" * 70)
in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
if in_venv:
    print(f"✓ Virtual environment is active")
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version: {sys.version.split()[0]}")
    passed += 1
else:
    print(f"✗ Virtual environment NOT active")
    print(f"  Please run: .\\venv\\Scripts\\Activate.ps1")
    warnings += 1

# Check GPU/CUDA
print("\nGPU/CUDA STATUS:")
print("-" * 70)
try:
    import torch
    has_cuda = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if has_cuda else 0

    if has_cuda:
        print(f"✓ CUDA is available")
        print(f"  GPU count: {cuda_device_count}")
        print(f"  Current device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA capability: {torch.cuda.get_device_capability(0)}")
        passed += 1
    else:
        print(f"⚠ CUDA not available (will use CPU)")
        print(f"  This is OK for development, but training will be slow")
        warnings += 1
except Exception as e:
    print(f"⚠ Could not check CUDA status: {e}")
    warnings += 1

# Check local project modules
print("\nPROJECT MODULES:")
print("-" * 70)
project_modules = [
    'pinn_model',
    'allen_cahn_pinn',
    'allen_cahn_pinn_improved',
    'residual_adaptive_sampling',
    'train_improved_allen_cahn',
]

for module_name in project_modules:
    try:
        import_module(module_name)
        print(f"✓ {module_name:<40} OK")
        passed += 1
    except Exception as e:
        print(f"✗ {module_name:<40} ERROR: {e}")
        failed += 1

# Summary
print("\n" + "="*70)
if failed == 0:
    print(f"  ✓ ALL CHECKS PASSED!")
    if warnings > 0:
        print(f"  ({warnings} warning(s) - see above)")
    print("="*70 + "\n")
    print("You are ready to run experiments!")
    print("\nQuick start:")
    print("  1. python test_improved.py          (quick test)")
    print("  2. python train_improved_allen_cahn.py  (main experiment)")
    sys.exit(0)
else:
    print(f"  ✗ {failed} CHECK(S) FAILED")
    print(f"  {passed} passed, {warnings} warning(s)")
    print("="*70 + "\n")
    print("Please fix the errors above before continuing.")
    print("\nCommon fixes:")
    print("  - Ensure venv is active: .\\venv\\Scripts\\Activate.ps1")
    print("  - Reinstall dependencies: pip install -r requirements.txt")
    print("  - See SETUP.md for troubleshooting")
    sys.exit(1)
