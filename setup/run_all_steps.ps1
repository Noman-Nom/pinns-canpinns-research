# PowerShell: Master Script - Run All Steps
# =========================================
# Runs complete pipeline to satisfy all 3 supervisor requirements:
#   1. Reference solutions (Crank-Nicolson)
#   2. Error metrics vs reference
#   3. Three-way comparison (novel contribution)
#
# Total time: ~2-3 hours on GPU

Write-Host "`n" -ForegroundColor Cyan
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  COMPLETE PIPELINE: Supervisor Requirements                   ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "This script runs all 4 steps:" -ForegroundColor Yellow
Write-Host "  1. Generate reference solutions (5-10 min)" -ForegroundColor Yellow
Write-Host "  2. Train PINN + Hybrid CAN-PINN (30 min)" -ForegroundColor Yellow
Write-Host "  3. Analyze errors vs reference (10 min)" -ForegroundColor Yellow
Write-Host "  4. Three-way comparison (1-2 hours)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Total estimated time: 2-3 hours on GPU" -ForegroundColor Yellow
Write-Host ""

# Check venv is active
if ($null -eq $env:VIRTUAL_ENV) {
    Write-Host "ERROR: Virtual environment not active!" -ForegroundColor Red
    Write-Host "Run: .\venv\Scripts\Activate.ps1`n" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Virtual environment active`n" -ForegroundColor Green

$startTime = Get-Date

# Step 1
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Step 1/4: Generate Reference Solutions                       ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
python reference_solver.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nSTEP 1 FAILED" -ForegroundColor Red
    exit 1
}

# Step 2
Write-Host "`n╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Step 2/4: Train PINN and Hybrid CAN-PINN                     ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
python train_improved_allen_cahn.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nSTEP 2 FAILED" -ForegroundColor Red
    exit 1
}

# Step 3
Write-Host "`n╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Step 3/4: Analyze Errors vs Reference                        ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
python error_analysis.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nSTEP 3 FAILED" -ForegroundColor Red
    exit 1
}

# Step 4
Write-Host "`n╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Step 4/4: Three-Way Comparison (Novel Contribution)          ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
python three_way_comparison.py 0.01 sin 2
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nSTEP 4 FAILED" -ForegroundColor Red
    exit 1
}

# Success
$endTime = Get-Date
$duration = ($endTime - $startTime).TotalSeconds

Write-Host "`n╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  ✓ ALL STEPS COMPLETE!                                        ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Total execution time: $([Math]::Round($duration/60, 1)) minutes" -ForegroundColor Green
Write-Host ""
Write-Host "Generated Results:" -ForegroundColor Cyan
Write-Host "  ✓ reference_solutions/" -ForegroundColor White
Write-Host "    └─ 4 high-accuracy reference solutions" -ForegroundColor White
Write-Host ""
Write-Host "  ✓ outputs/" -ForegroundColor White
Write-Host "    └─ PINN + Hybrid CAN-PINN results for all test cases" -ForegroundColor White
Write-Host ""
Write-Host "  ✓ error_analysis_results.json" -ForegroundColor White
Write-Host "    └─ L2/L∞ error metrics vs reference (Supervisor Req #1)" -ForegroundColor White
Write-Host ""
Write-Host "  ✓ three_way_results/" -ForegroundColor White
Write-Host "    └─ Three-way comparison (Supervisor Req #2, Novel Contribution)" -ForegroundColor White
Write-Host ""
Write-Host "Next Actions:" -ForegroundColor Cyan
Write-Host "  1. Review error metrics:" -ForegroundColor White
Write-Host "     $PSScriptRoot\error_analysis_results.json" -ForegroundColor White
Write-Host ""
Write-Host "  2. Review three-way comparison:" -ForegroundColor White
Write-Host "     $PSScriptRoot\three_way_results\comparison_summary.json" -ForegroundColor White
Write-Host ""
Write-Host "  3. Create final summary for supervisor meeting" -ForegroundColor White
Write-Host ""
