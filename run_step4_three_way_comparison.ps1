# PowerShell: Step 4 - Three-Way Comparison (Novel Contribution)
# ==============================================================
# Supervisor Requirement #2: "For mark 5 we need something new"
#
# Novel Contribution: Show progression from baseline to hybrid
#   1. Baseline PINN (standard AD, no enhancements)
#   2. Original CAN-PINN (uses finite differences for ∂²u/∂x²)
#   3. Hybrid CAN-PINN (uses AD + uncertainty + adaptive + L-BFGS)
#
# Why this is novel:
#   • Shows WHY hybrid is better: replaces FD (O(Δx²) error) with AD
#   • Adds mathematical rigor (uncertainty weighting + adaptive sampling)
#   • Demonstrates systematic improvement
#
# Time: ~1-2 hours on GPU (includes full training of all 3 methods)

Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Step 4: Three-Way Comparison (Novel Contribution)" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════`n" -ForegroundColor Cyan

Write-Host "Supervisor Requirement: 'For mark 5 we need something new'" -ForegroundColor Yellow
Write-Host ""
Write-Host "Novel Contribution: Progressive Improvement" -ForegroundColor Yellow
Write-Host "  1. Baseline PINN" -ForegroundColor Yellow
Write-Host "     • Standard automatic differentiation" -ForegroundColor Yellow
Write-Host "     • No enhancements" -ForegroundColor Yellow
Write-Host ""
Write-Host "  2. Original CAN-PINN (from paper)" -ForegroundColor Yellow
Write-Host "     • Uses finite differences for ∂²u/∂x²" -ForegroundColor Yellow
Write-Host "     • Faster computation" -ForegroundColor Yellow
Write-Host "     • BUT: Has truncation error O(Δx²) ≈ 1e-4" -ForegroundColor Yellow
Write-Host ""
Write-Host "  3. Hybrid CAN-PINN (our contribution)" -ForegroundColor Yellow
Write-Host "     • Restores AD (eliminates FD error)" -ForegroundColor Yellow
Write-Host "     • Keeps uncertainty weighting (learnable log_var)" -ForegroundColor Yellow
Write-Host "     • Adds residual-based adaptive sampling" -ForegroundColor Yellow
Write-Host "     • Adds gradient penalty" -ForegroundColor Yellow
Write-Host "     • Adds L-BFGS fine-tuning" -ForegroundColor Yellow
Write-Host ""
Write-Host "Result: Show quantitatively WHY hybrid wins" -ForegroundColor Yellow
Write-Host ""
Write-Host "Estimated time: 1-2 hours on GPU (trains all 3 methods)" -ForegroundColor Yellow
Write-Host ""

# Check venv is active
if ($null -eq $env:VIRTUAL_ENV) {
    Write-Host "ERROR: Virtual environment not active!" -ForegroundColor Red
    Write-Host "Run: .\venv\Scripts\Activate.ps1`n" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Virtual environment active`n" -ForegroundColor Green

Write-Host "Starting three-way comparison..." -ForegroundColor Yellow
Write-Host "  [Training: Baseline PINN]" -ForegroundColor Yellow
Write-Host "  [Training: Original CAN-PINN (FD)]" -ForegroundColor Yellow
Write-Host "  [Training: Hybrid CAN-PINN (AD + Enhancements)]" -ForegroundColor Yellow
Write-Host ""

# Run three-way comparison for TC2, sin IC, ε=0.01
python three_way_comparison.py 0.01 sin 2

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "  ✓ Step 4 Complete: Three-Way Comparison Done" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    Write-Host "Output: three_way_results/" -ForegroundColor Cyan
    Write-Host "  • comparison_summary.json - Detailed results" -ForegroundColor White
    Write-Host "  • *_predictions.npz - Solution files for plotting" -ForegroundColor White
    Write-Host ""
    Write-Host "Key Result:" -ForegroundColor Cyan
    Write-Host "  Shows that hybrid CAN-PINN is an improvement over both baselines" -ForegroundColor White
    Write-Host "  by eliminating FD truncation error while keeping benefits of CAN-PINN" -ForegroundColor White
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "  1. Review all results:" -ForegroundColor White
    Write-Host "     • error_analysis_results.json (error metrics)" -ForegroundColor White
    Write-Host "     • three_way_results/comparison_summary.json (novel contribution)" -ForegroundColor White
    Write-Host ""
    Write-Host "  2. Create summary document for supervisor:" -ForegroundColor White
    Write-Host "     python create_supervisor_report.py (TODO)" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "`n✗ FAILED: Three-way comparison" -ForegroundColor Red
    exit 1
}
