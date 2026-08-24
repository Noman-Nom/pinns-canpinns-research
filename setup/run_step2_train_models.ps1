# PowerShell: Step 2 - Train PINN and Hybrid CAN-PINN
# ====================================================
# Trains models on all 4 test cases:
#   • Baseline PINN (standard AD)
#   • Hybrid CAN-PINN (AD + uncertainty + adaptive + L-BFGS)
#
# Supervisor Requirement #3: Mathematical clarity
# "Show that hybrid is an improvement over baseline"
#
# Time: ~30 minutes on GPU (2-3 hours on CPU)

Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Step 2: Train PINN and Hybrid CAN-PINN Models" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════`n" -ForegroundColor Cyan

Write-Host "Training Configuration:" -ForegroundColor Yellow
Write-Host "  • Adam optimizer: 10,000 epochs (lr=0.001)" -ForegroundColor Yellow
Write-Host "  • L-BFGS fine-tuning: 1,000 iterations" -ForegroundColor Yellow
Write-Host "  • Hybrid enhancements:" -ForegroundColor Yellow
Write-Host "    - Uncertainty-weighted loss (learnable log_var)" -ForegroundColor Yellow
Write-Host "    - Residual-based adaptive sampling (10% every 3k epochs)" -ForegroundColor Yellow
Write-Host "    - Gradient penalty (λ=1e-5)" -ForegroundColor Yellow
Write-Host "  • GPU: Yes (Quadro T2000)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Estimated time: 30 minutes on GPU" -ForegroundColor Yellow
Write-Host ""

# Check venv is active
if ($null -eq $env:VIRTUAL_ENV) {
    Write-Host "ERROR: Virtual environment not active!" -ForegroundColor Red
    Write-Host "Run: .\venv\Scripts\Activate.ps1`n" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Virtual environment active`n" -ForegroundColor Green

# Check reference solutions exist
if (-not (Test-Path "reference_solutions")) {
    Write-Host "WARNING: reference_solutions/ not found" -ForegroundColor Yellow
    Write-Host "  Did you run Step 1? (Run: .\run_step1_reference.ps1)" -ForegroundColor Yellow
    Write-Host ""
}

# Run training
Write-Host "Starting training..." -ForegroundColor Yellow
Write-Host "  [Processing: TC2 sin ε=0.01, TC2 step ε=0.01, TC3 sin ε=0.01, TC3 sin ε=0.05]" -ForegroundColor Yellow
Write-Host ""

python train_improved_allen_cahn.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "  ✓ Step 2 Complete: Models Trained" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    Write-Host "Output: outputs/" -ForegroundColor Cyan
    Write-Host "  • Loss curves and metrics for each test case" -ForegroundColor White
    Write-Host "  • Solution predictions for error analysis" -ForegroundColor White
    Write-Host ""
    Write-Host "Next: Run Step 3 to analyze errors" -ForegroundColor Cyan
    Write-Host "  .\run_step3_error_analysis.ps1" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "`n✗ FAILED: Model training" -ForegroundColor Red
    exit 1
}
