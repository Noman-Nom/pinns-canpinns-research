# PowerShell Script: Setup Virtual Environment for PINN Research
# Windows 11, Python 3.10+
# Run: powershell -ExecutionPolicy Bypass -File setup_venv.ps1

Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Hybrid CAN-PINN Research Project - Virtual Environment Setup  ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python installation
Write-Host "📋 Step 1: Verifying Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ ERROR: Python not found or not in PATH" -ForegroundColor Red
    Write-Host "  Please install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Step 2: Create virtual environment
Write-Host ""
Write-Host "📦 Step 2: Creating virtual environment at './venv'..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  ⚠ venv directory already exists. Removing..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv
}

python -m venv venv
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✗ ERROR: Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Step 3: Activate virtual environment
Write-Host ""
Write-Host "🔌 Step 3: Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "⚠ Could not activate automatically. Try manually:" -ForegroundColor Yellow
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
}

# Step 4: Upgrade pip, setuptools, wheel
Write-Host ""
Write-Host "🔄 Step 4: Upgrading pip, setuptools, wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
Write-Host "✓ Pip tools upgraded" -ForegroundColor Green

# Step 5: Install requirements
Write-Host ""
Write-Host "📥 Step 5: Installing project dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ All dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ ERROR: Failed to install dependencies" -ForegroundColor Red
    Write-Host "  Review the error messages above and try again." -ForegroundColor Red
    exit 1
}

# Step 6: Verify GPU/CUDA (if available)
Write-Host ""
Write-Host "🔍 Step 6: Checking GPU/CUDA availability..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
Write-Host ""

# Step 7: Success summary
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  ✓ Setup Complete!                                             ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Activate the environment (if not already):" -ForegroundColor Cyan
Write-Host "     .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "  2. Verify the setup:" -ForegroundColor Cyan
Write-Host "     python verify_gpu.py" -ForegroundColor White
Write-Host ""
Write-Host "  3. Run a quick test:" -ForegroundColor Cyan
Write-Host "     python test_improved.py" -ForegroundColor White
Write-Host ""
Write-Host "  4. Run main training experiment:" -ForegroundColor Cyan
Write-Host "     python train_improved_allen_cahn.py" -ForegroundColor White
Write-Host ""
Write-Host "📚 Documentation:" -ForegroundColor Cyan
Write-Host "  - CLAUDE.md: Architecture, hyperparams, and project structure" -ForegroundColor White
Write-Host "  - RESULTS.md: Current experimental results" -ForegroundColor White
Write-Host ""
