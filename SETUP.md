# Virtual Environment Setup Guide

**Project**: Hybrid CAN-PINN for Allen-Cahn Equation  
**Environment**: Windows 11, Python 3.10+  
**Date**: 2026-05-02

---

## Quick Start (Automated)

### Option 1: PowerShell (Recommended for Windows 11)

```powershell
cd e:\khokhar-tappa\pinns-canpinns-research
powershell -ExecutionPolicy Bypass -File setup_venv.ps1
```

The script will:
1. ✓ Check Python installation
2. ✓ Create virtual environment at `./venv`
3. ✓ Activate the environment
4. ✓ Upgrade pip, setuptools, wheel
5. ✓ Install all dependencies from `requirements.txt`
6. ✓ Verify GPU/CUDA setup
7. ✓ Print next steps

### Option 2: Command Prompt (Traditional)

```cmd
cd e:\khokhar-tappa\pinns-canpinns-research
setup_venv.bat
```

---

## Manual Setup (Step-by-Step)

If the automated scripts don't work, follow these steps manually:

### Step 1: Verify Python
```powershell
python --version
```
Expected output: `Python 3.10.x` or higher

If Python is not found:
- Install from [python.org](https://www.python.org/downloads/)
- **Important**: Check "Add Python to PATH" during installation
- Restart terminal after installation

### Step 2: Create Virtual Environment
```powershell
cd e:\khokhar-tappa\pinns-canpinns-research
python -m venv venv
```

This creates a `venv/` directory with isolated Python packages.

### Step 3: Activate Virtual Environment
```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# OR Command Prompt
venv\Scripts\activate.bat
```

After activation, your prompt should show `(venv)` prefix:
```
(venv) PS E:\khokhar-tappa\pinns-canpinns-research>
```

**Note**: If PowerShell says "cannot be loaded because running scripts is disabled", run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Upgrade Pip
```powershell
python -m pip install --upgrade pip setuptools wheel
```

### Step 5: Install Dependencies
```powershell
pip install -r requirements.txt
```

This installs all packages from `requirements.txt`. Installation takes 5-10 minutes depending on your internet speed.

### Step 6: Verify Installation
```powershell
python verify_gpu.py
```

Output should show:
- PyTorch version (e.g., `2.0.0`)
- CUDA availability (`True` or `False`)
- GPU name (if CUDA available) or `CPU`

---

## File Breakdown

| File | Purpose |
|------|---------|
| `requirements.txt` | List of all Python packages needed |
| `setup_venv.ps1` | Automated setup script (PowerShell) |
| `setup_venv.bat` | Automated setup script (Command Prompt) |
| `SETUP.md` | This file |

---

## What Gets Installed

### Core Dependencies (from requirements.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| **torch** | ≥2.0.0 | Neural network framework (PyTorch) |
| **numpy** | ≥1.24.0 | Numerical computing |
| **scipy** | ≥1.10.0 | Scientific computing (ODE solvers, etc.) |
| **matplotlib** | ≥3.7.0 | Plotting results |
| **sympy** | ≥1.14.0 | Symbolic math (for reference solver, MCP tools) |
| **mcp** | ≥0.1.0 | Model Context Protocol server |
| **pytest** | ≥7.0.0 | Testing framework |
| **scikit-learn** | ≥1.3.0 | Machine learning utilities |
| **tqdm** | ≥4.65.0 | Progress bars |

### GPU Support (Optional)

If you have an NVIDIA GPU with CUDA:
- PyTorch will automatically use CUDA for GPU acceleration
- Check status with `python verify_gpu.py`

To force CPU-only (not recommended):
```powershell
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## After Setup: Running Experiments

### 1. Activate environment (each time you open a new terminal)
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Quick test (verify everything works)
```powershell
python test_improved.py
```

Expected: Should complete in <5 minutes with test results.

### 3. Run main experiment (all 4 test cases)
```powershell
python train_improved_allen_cahn.py
```

Expected: ~15-30 minutes on GPU (2-3 hours on CPU).

### 4. Generate reference solutions and compute errors
```powershell
python reference_solver.py  # TODO: To be implemented
```

---

## Troubleshooting

### Issue: `python: command not found`
**Solution**: Python is not in your PATH. 
- Reinstall Python, checking "Add Python to PATH"
- Or add Python manually to PATH in Environment Variables

### Issue: `Access to the path 'venv' is denied`
**Solution**: Close any open PowerShell windows and try again. Or:
```powershell
Remove-Item -Recurse -Force venv
python -m venv venv
```

### Issue: `No module named 'torch'`
**Solution**: Virtual environment may not be activated. Run:
```powershell
.\venv\Scripts\Activate.ps1
pip install torch
```

### Issue: GPU not recognized (`CUDA available: False`)
**Solution**: PyTorch may be installed with CPU-only support.
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Replace `cu118` with your CUDA version (run `nvidia-smi` to check).

### Issue: `Permission denied` when running PowerShell script
**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try again:
```powershell
powershell -ExecutionPolicy Bypass -File setup_venv.ps1
```

---

## Virtual Environment Basics

### What is a Virtual Environment?

A venv isolates Python packages for this project. Without it:
- Package conflicts between projects
- Hard to reproduce results on other machines
- Difficult to clean up

With venv:
- Each project has its own `python` and `pip`
- `requirements.txt` lists exact versions
- Easy to recreate on any machine

### Activating/Deactivating

```powershell
# Activate (each new terminal)
.\venv\Scripts\Activate.ps1

# Deactivate (return to system Python)
deactivate
```

### Adding/Removing Packages

```powershell
# Install new package (venv must be active)
pip install package_name

# Remove package
pip uninstall package_name

# Update requirements.txt after changes
pip freeze > requirements.txt
```

### Deleting Virtual Environment

If you need to start over:
```powershell
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Environment Verification Checklist

After setup, verify:

- [ ] `(venv)` prefix shown in terminal
- [ ] `python --version` shows 3.10+
- [ ] `python verify_gpu.py` shows PyTorch info
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` works
- [ ] `python -c "import numpy, scipy, sympy, matplotlib, mcp"` imports without error
- [ ] `pytest --version` shows pytest is installed

---

## Next: Implement Reference Solver

The project requires a Crank-Nicolson reference solver (see CLAUDE.md).

**Blocked until venv is active**: `reference_solver.py` depends on SciPy.

```powershell
# Once venv is set up:
python reference_solver.py  # TODO: Not yet implemented
```

---

## Support & Documentation

- **CLAUDE.md**: Full project architecture, hyperparams, requirements
- **RESULTS.md**: Current experimental results
- **SUPERVISOR_SUMMARY.md**: Summary for supervisor meeting

For questions about the project, see CLAUDE.md.
