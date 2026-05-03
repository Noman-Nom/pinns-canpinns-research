# PowerShell: Step 1 - Generate Reference Solutions
# ==========================================
# Supervisor Requirement #1: "I need exact test cases with answers"
#
# This generates Crank-Nicolson reference solutions for all 4 test cases:
#   • TC2: sin(πx) IC, ε=0.01
#   • TC2: step IC, ε=0.01
#   • TC3: sin(πx) IC, ε=0.01
#   • TC3: sin(πx) IC, ε=0.05
#
# Grid: Δx = 1/256, Δt = 0.0001 (high accuracy, ~1e-7 error)
# Time: ~5-10 minutes

Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Step 1: Generate Crank-Nicolson Reference Solutions" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════`n" -ForegroundColor Cyan

Write-Host "Supervisor Requirement: 'I need exact test cases with answers'" -ForegroundColor Yellow
Write-Host ""

# Check venv is active
if ($null -eq $env:VIRTUAL_ENV) {
    Write-Host "ERROR: Virtual environment not active!" -ForegroundColor Red
    Write-Host "Run: .\venv\Scripts\Activate.ps1`n" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Virtual environment active`n" -ForegroundColor Green

# Run reference solver
Write-Host "Generating reference solutions..." -ForegroundColor Yellow
Write-Host "(This may take 5-10 minutes)" -ForegroundColor Yellow
Write-Host ""

python reference_solver.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "  ✓ Step 1 Complete: Reference Solutions Generated" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    Write-Host "Output: reference_solutions/" -ForegroundColor Cyan
    Write-Host "  • TC2_sin_eps001_reference.npz" -ForegroundColor White
    Write-Host "  • TC2_step_eps001_reference.npz" -ForegroundColor White
    Write-Host "  • TC3_sin_eps001_reference.npz" -ForegroundColor White
    Write-Host "  • TC3_sin_eps005_reference.npz" -ForegroundColor White
    Write-Host ""
    Write-Host "Next: Run Step 2 to train PINN models" -ForegroundColor Cyan
    Write-Host "  .\run_step2_train_models.ps1" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "`n✗ FAILED: Reference solution generation" -ForegroundColor Red
    exit 1
}
