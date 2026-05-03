@echo off
REM Batch Script: Setup Virtual Environment for PINN Research
REM Windows 11, Python 3.10+
REM Run: setup_venv.bat

setlocal enabledelayedexpansion

cls
echo.
echo ════════════════════════════════════════════════════════════════
echo   Hybrid CAN-PINN Research Project - Virtual Environment Setup
echo ════════════════════════════════════════════════════════════════
echo.

REM Step 1: Check Python installation
echo [1/7] Verifying Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PYTHON_VER=%%i
echo OK: Found %PYTHON_VER%
echo.

REM Step 2: Create virtual environment
echo [2/7] Creating virtual environment at '.\venv'...
if exist venv (
    echo WARNING: venv directory already exists. Removing...
    rmdir /s /q venv
)
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo OK: Virtual environment created
echo.

REM Step 3: Activate virtual environment
echo [3/7] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo WARNING: Could not activate automatically
    echo Try manually: venv\Scripts\activate.bat
)
echo OK: Virtual environment activated
echo.

REM Step 4: Upgrade pip
echo [4/7] Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel --quiet
echo OK: Pip tools upgraded
echo.

REM Step 5: Install requirements
echo [5/7] Installing project dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Review the error messages above and try again.
    pause
    exit /b 1
)
echo OK: All dependencies installed
echo.

REM Step 6: Check GPU/CUDA
echo [6/7] Checking GPU/CUDA availability...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo.

REM Step 7: Success message
echo.
echo ════════════════════════════════════════════════════════════════
echo   OK: Setup Complete!
echo ════════════════════════════════════════════════════════════════
echo.
echo NEXT STEPS:
echo.
echo 1. Activate the environment (if not already):
echo    venv\Scripts\activate.bat
echo.
echo 2. Verify the setup:
echo    python verify_gpu.py
echo.
echo 3. Run a quick test:
echo    python test_improved.py
echo.
echo 4. Run main training experiment:
echo    python train_improved_allen_cahn.py
echo.
echo DOCUMENTATION:
echo - CLAUDE.md: Architecture, hyperparams, and project structure
echo - RESULTS.md: Current experimental results
echo.
pause
