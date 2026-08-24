# PowerShell Script: Install GPU PyTorch
# Detects CUDA version and installs correct PyTorch build

Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Installing GPU PyTorch for Your NVIDIA GPU" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════`n" -ForegroundColor Cyan

# Step 1: Check if venv is active
if ($null -eq $env:VIRTUAL_ENV) {
    Write-Host "ERROR: Virtual environment not active!" -ForegroundColor Red
    Write-Host "Please run: .\venv\Scripts\Activate.ps1`n" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Virtual environment active: $env:VIRTUAL_ENV`n" -ForegroundColor Green

# Step 2: Check if nvidia-smi is available
Write-Host "Checking NVIDIA GPU..." -ForegroundColor Yellow
$nvidiaCheck = nvidia-smi 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: nvidia-smi not found or no GPU detected!" -ForegroundColor Red
    Write-Host "Please ensure your NVIDIA GPU drivers are installed.`n" -ForegroundColor Yellow
    exit 1
}

# Step 3: Parse CUDA version
$cudaVersion = $null
foreach ($line in $nvidiaCheck) {
    if ($line -match "CUDA Version:\s+(\d+\.\d+)") {
        $cudaVersion = $matches[1]
        break
    }
}

if ($null -eq $cudaVersion) {
    Write-Host "WARNING: Could not detect CUDA version from nvidia-smi" -ForegroundColor Yellow
    Write-Host "Attempting default installation (CUDA 12.1)..." -ForegroundColor Yellow
    $cudaVersion = "12.1"
}

Write-Host "✓ Detected CUDA Version: $cudaVersion`n" -ForegroundColor Green

# Step 4: Map CUDA version to PyTorch wheel index
$cudaMajor = [int]($cudaVersion.Split('.')[0])
$cudaMinor = [int]($cudaVersion.Split('.')[1])

if ($cudaMajor -eq 12) {
    # CUDA 12.x → cu121
    $wheelIndex = "cu121"
    Write-Host "→ Installing PyTorch for CUDA 12.1..." -ForegroundColor Cyan
} elseif ($cudaMajor -eq 11 -and $cudaMinor -ge 8) {
    # CUDA 11.8+ → cu118
    $wheelIndex = "cu118"
    Write-Host "→ Installing PyTorch for CUDA 11.8..." -ForegroundColor Cyan
} else {
    Write-Host "WARNING: Unsupported CUDA version $cudaVersion" -ForegroundColor Yellow
    Write-Host "Defaulting to CUDA 12.1..." -ForegroundColor Yellow
    $wheelIndex = "cu121"
}

Write-Host "`nUninstalling CPU PyTorch..." -ForegroundColor Yellow
pip uninstall torch -y --quiet

Write-Host "Installing GPU PyTorch..." -ForegroundColor Yellow
$installCmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$wheelIndex"
Invoke-Expression $installCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ PyTorch GPU installation successful!`n" -ForegroundColor Green

    # Verify installation
    Write-Host "Verifying GPU access..." -ForegroundColor Yellow
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Green
        Write-Host "  ✓ GPU PyTorch Ready!" -ForegroundColor Green
        Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
        Write-Host "`nYou can now run GPU-accelerated training:" -ForegroundColor Cyan
        Write-Host "  python train_improved_allen_cahn.py`n" -ForegroundColor White
    }
} else {
    Write-Host "`n✗ Installation failed!" -ForegroundColor Red
    Write-Host "Please check the errors above and try manually:" -ForegroundColor Yellow
    Write-Host "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$wheelIndex`n" -ForegroundColor Cyan
    exit 1
}
