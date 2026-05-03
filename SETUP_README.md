# 🚀 Project Setup Guide — Hybrid CAN-PINN Research

**Quick Links:**
- 📋 Full Setup Instructions: [SETUP.md](SETUP.md)
- 🐍 Project Architecture: [CLAUDE.md](CLAUDE.md)
- 📊 Current Results: [RESULTS.md](RESULTS.md)

---

## What You Have

I've created a complete virtual environment setup system for your Windows 11 machine:

### Setup Files Created

| File | Purpose | How to Use |
|------|---------|-----------|
| **requirements.txt** | Complete package list with versions | `pip install -r requirements.txt` |
| **setup_venv.ps1** | Automated PowerShell setup (⭐ **Recommended**) | `powershell -ExecutionPolicy Bypass -File setup_venv.ps1` |
| **setup_venv.bat** | Automated Command Prompt setup | `setup_venv.bat` |
| **check_setup.py** | Verify installation after setup | `python check_setup.py` |
| **SETUP.md** | Detailed manual setup guide | Reference for troubleshooting |

---

## Quick Start (5 Minutes)

### Windows 11 with PowerShell (Recommended)

```powershell
# 1. Navigate to project directory
cd e:\khokhar-tappa\pinns-canpinns-research

# 2. Run the setup script (one command!)
powershell -ExecutionPolicy Bypass -File setup_venv.ps1

# 3. Verify everything works
python check_setup.py
```

That's it! The script will:
- ✓ Create virtual environment (`venv/`)
- ✓ Activate it automatically
- ✓ Install all dependencies (torch, numpy, scipy, sympy, mcp, etc.)
- ✓ Check GPU/CUDA availability
- ✓ Print next steps

### Windows with Command Prompt

If you prefer cmd.exe:
```cmd
cd e:\khokhar-tappa\pinns-canpinns-research
setup_venv.bat
python check_setup.py
```

---

## Package Breakdown

### Installed Packages (9 total)

**Core ML & Numerical:**
- `torch` (2.0+) — Neural networks
- `numpy` (1.24+) — Numerical arrays
- `scipy` (1.10+) — Scientific computing (FD solvers, etc.)
- `matplotlib` (3.7+) — Plotting results

**Research Tools:**
- `sympy` (1.14+) — Symbolic math (reference solver, Crank-Nicolson)
- `mcp` (0.1+) — MCP server for math tools

**Development:**
- `pytest` (7.0+) — Testing framework
- `scikit-learn` (1.3+) — ML utilities
- `tqdm` (4.65+) — Progress bars

### Why Each Package?

| Package | Used For |
|---------|----------|
| **torch** | Training PINN and CAN-PINN models (GPU acceleration) |
| **numpy/scipy** | Numerical computing, finite difference solvers |
| **matplotlib** | Plotting loss curves, solution comparisons |
| **sympy** | Symbolic PDE verification, Crank-Nicolson FD reference solver |
| **mcp** | Running math tools via MCP server (mcp_math_server.py) |
| **pytest** | Unit tests (test_*.py files) |

---

## After Setup: Running Experiments

### Step 1: Activate Environment (Each Time)
```powershell
.\venv\Scripts\Activate.ps1
```
You should see `(venv)` prefix in your terminal.

### Step 2: Quick Verification
```powershell
python check_setup.py
```
Should show all packages installed and GPU status.

### Step 3: Run Tests
```powershell
python test_improved.py
```
Quick sanity check (~2-3 minutes).

### Step 4: Run Main Experiment
```powershell
python train_improved_allen_cahn.py
```
Trains baseline PINN vs hybrid CAN-PINN on 4 test cases (~30 min on GPU, 2-3 hrs on CPU).

### Step 5: Generate Reference Solutions (TODO)
```powershell
python reference_solver.py  # To be implemented
```
Computes Crank-Nicolson reference and error metrics (supervisor requirement).

---

## Virtual Environment Essentials

### What is a venv?

A **virtual environment** isolates Python packages for this project:
- Each project has its own `python` and `pip`
- No conflicts between projects
- Easy to recreate on any machine using `requirements.txt`

### Activate/Deactivate

```powershell
# Activate (each new terminal)
.\venv\Scripts\Activate.ps1

# Deactivate (return to system Python)
deactivate
```

### Install New Packages (While Activated)

```powershell
pip install package_name
pip freeze > requirements.txt  # Update requirements.txt
```

### Delete & Recreate (If Corrupted)

```powershell
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Troubleshooting

### Problem: PowerShell Won't Run setup_venv.ps1
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try again.

### Problem: "No module named 'torch'" After Setup
**Solution:** Virtual environment not activated.
```powershell
.\venv\Scripts\Activate.ps1
pip install torch
```

### Problem: GPU Not Detected
**Solution:** PyTorch may be CPU-only. Reinstall CUDA version:
```powershell
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
(Replace `cu118` with your CUDA version. Run `nvidia-smi` to check.)

### Problem: "Access denied" when deleting venv
**Solution:** Close all PowerShell windows, then:
```powershell
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Full Documentation

| File | Contains |
|------|----------|
| [SETUP.md](SETUP.md) | Detailed setup guide, manual steps, troubleshooting |
| [CLAUDE.md](CLAUDE.md) | **Project architecture**, network design, hyperparams, requirements from supervisor |
| [RESULTS.md](RESULTS.md) | Current experimental results (2 wins, 2 losses vs baseline) |
| [HONEST_RESULTS_REVIEW.md](HONEST_RESULTS_REVIEW.md) | Detailed analysis with caveats |
| [SUPERVISOR_SUMMARY.md](SUPERVISOR_SUMMARY.md) | Presentation-ready summary for meeting |

---

## Project Structure

```
e:\khokhar-tappa\pinns-canpinns-research\
├── Core Models
│   ├── pinn_model.py                    # Base PINN class
│   ├── allen_cahn_pinn.py               # Baseline PINN + Original CAN-PINN
│   ├── allen_cahn_pinn_improved.py      # Hybrid CAN-PINN with enhancements
│   └── residual_adaptive_sampling.py    # Adaptive sampling module
│
├── Training & Testing
│   ├── train_improved_allen_cahn.py     # Main experiment (4 test cases)
│   ├── test_improved.py                 # Quick unit test
│   ├── test_*.py                        # Other tests
│   └── verify_gpu.py                    # GPU status check
│
├── Setup (NEW - Created Today)
│   ├── requirements.txt                 # Package list with versions
│   ├── setup_venv.ps1                   # Automated setup (PowerShell)
│   ├── setup_venv.bat                   # Automated setup (cmd)
│   ├── check_setup.py                   # Verification script
│   ├── SETUP.md                         # Detailed setup guide
│   └── SETUP_README.md                  # This file
│
├── Documentation
│   ├── CLAUDE.md                        # Architecture, hyperparams, requirements
│   ├── RESULTS.md                       # Current results
│   ├── HONEST_RESULTS_REVIEW.md         # Analysis with caveats
│   ├── SUPERVISOR_SUMMARY.md            # Presentation summary
│   └── PINN_DOCUMENTATION.md            # Framework theory
│
├── MCP Math Server
│   └── mcp_math_server.py               # Symbolic + numerical tools
│
└── venv/                               # (Created by setup script)
    ├── Scripts/Activate.ps1             # Activation script
    ├── lib/python3.10/                  # Installed packages
    └── ...
```

---

## Next: Implement Reference Solver

**Blocking your supervisor approval:**
1. ✓ Environment setup (just completed!)
2. ⏳ **Reference solution** (Crank-Nicolson FD) — needs scipy
3. ⏳ Error metrics vs reference (L2 relative, L∞)
4. ⏳ Three-way comparison (PINN vs original CAN-PINN vs hybrid)

With this venv setup, you can now:
```powershell
python reference_solver.py  # TODO: Implement this next
```

See [CLAUDE.md](CLAUDE.md) → "What the Supervisor Requires (Critical)" for full details.

---

## Support

**Questions about setup?** See [SETUP.md](SETUP.md)  
**Questions about project?** See [CLAUDE.md](CLAUDE.md)  
**Questions about results?** See [RESULTS.md](RESULTS.md)

Good luck with your research! 🎓
