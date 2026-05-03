# PowerShell: Step 3 - Compute Error Metrics
# ==========================================
# Supervisor Requirement #1: "Compare against exact solutions"
#
# Compares PINN/CAN-PINN predictions against reference solutions
# Computes:
#   • L2 relative error: ||u_pinn - u_ref|| / ||u_ref||
#   • L∞ absolute error: max|u_pinn - u_ref|
#   • L∞ relative error: max|u_pinn - u_ref| / max|u_ref|
#
# Time: ~10 minutes

Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Step 3: Compute Error Metrics vs Reference Solutions" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════`n" -ForegroundColor Cyan

Write-Host "Supervisor Requirement: 'Show error metrics vs exact solutions'" -ForegroundColor Yellow
Write-Host ""
Write-Host "Error Metrics Computed:" -ForegroundColor Yellow
Write-Host "  • L2 relative error: ||u_pinn - u_ref||_2 / ||u_ref||_2" -ForegroundColor Yellow
Write-Host "  • L∞ absolute error: max|u_pinn - u_ref|" -ForegroundColor Yellow
Write-Host "  • L∞ relative error: max|u_pinn - u_ref| / max|u_ref|" -ForegroundColor Yellow
Write-Host "  • Mean absolute error (MAE)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Estimated time: 10 minutes" -ForegroundColor Yellow
Write-Host ""

# Check venv is active
if ($null -eq $env:VIRTUAL_ENV) {
    Write-Host "ERROR: Virtual environment not active!" -ForegroundColor Red
    Write-Host "Run: .\venv\Scripts\Activate.ps1`n" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Virtual environment active`n" -ForegroundColor Green

# Check prerequisites
$missing = @()
if (-not (Test-Path "reference_solutions")) { $missing += "reference_solutions/" }
if (-not (Test-Path "outputs")) { $missing += "outputs/" }

if ($missing.Count -gt 0) {
    Write-Host "ERROR: Missing prerequisites:" -ForegroundColor Red
    foreach ($item in $missing) { Write-Host "  • $item" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Run Step 1 and Step 2 first:" -ForegroundColor Yellow
    Write-Host "  .\run_step1_reference.ps1" -ForegroundColor White
    Write-Host "  .\run_step2_train_models.ps1" -ForegroundColor White
    Write-Host ""
    exit 1
}

# Run error analysis
Write-Host "Computing error metrics..." -ForegroundColor Yellow
Write-Host "  [Interpolating PINN outputs to reference grid]" -ForegroundColor Yellow
Write-Host "  [Computing L2, L∞ errors]" -ForegroundColor Yellow
Write-Host ""

python error_analysis.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "  ✓ Step 3 Complete: Error Analysis Done" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    Write-Host "Output: error_analysis_results.json" -ForegroundColor Cyan
    Write-Host "  Contains error metrics for all test cases and models" -ForegroundColor White
    Write-Host ""
    Write-Host "Next: Run Step 4 for novel contribution (three-way comparison)" -ForegroundColor Cyan
    Write-Host "  .\run_step4_three_way_comparison.ps1" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "`n✗ FAILED: Error analysis" -ForegroundColor Red
    exit 1
}
