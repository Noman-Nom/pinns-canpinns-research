# PowerShell Script: Install GPU PyTorch for CUDA 13.0
# CUDA 13.0 is newer, but PyTorch provides cu124 (12.4) wheels that work
# cu124 wheels are forward-compatible with CUDA 13.0

Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Installing GPU PyTorch for CUDA 13.0" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════`n" -ForegroundColor Cyan

# Step 1: Verify venv is active
if ($null -eq $env:VIRTUAL_ENV) {
    Write-Host "ERROR: Virtual environment not active!" -ForegroundColor Red
    Write-Host "Run: .\venv\Scripts\Activate.ps1`n" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Virtual environment active`n" -ForegroundColor Green

# Step 2: Try installation with cu124 (CUDA 12.4 - forward compatible with 13.0)
Write-Host "Installing PyTorch with cu124 wheels (CUDA 12.4 - compatible with your CUDA 13.0)..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Installation successful!`n" -ForegroundColor Green

    # Verify GPU access
    Write-Host "Verifying GPU access..." -ForegroundColor Yellow
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'; print(f'GPU: {gpu_name}')"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Green
        Write-Host "  ✓ GPU PyTorch Ready!" -ForegroundColor Green
        Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
        Write-Host "`nYou can now run GPU training:" -ForegroundColor Cyan
        Write-Host "  python train_improved_allen_cahn.py`n" -ForegroundColor White
        exit 0
    }
} else {
    Write-Host "cu124 failed. Trying cu118 (CUDA 11.8) as fallback..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ Installation successful!`n" -ForegroundColor Green
        python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'; print(f'GPU: {gpu_name}')"
        exit 0
    } else {
        Write-Host "`n✗ Both cu124 and cu118 failed!" -ForegroundColor Red
        Write-Host "Try manual installation:" -ForegroundColor Yellow
        Write-Host "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`n" -ForegroundColor Cyan
        exit 1
    }
}
