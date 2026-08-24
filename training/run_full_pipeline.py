"""
Full Pipeline: Generate References → Train Models → Analyze Errors → Compare

This orchestrates all three supervisor requirements:
1. Reference solutions (Crank-Nicolson FD)
2. Error metrics vs reference
3. Three-way comparison (novel contribution)

Run this ONCE to generate all results needed for supervisor.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n✗ FAILED: {description}")
        sys.exit(1)
    print(f"\n✓ Complete: {description}")


def main():
    """Run full pipeline."""
    print("\n" + "="*70)
    print("  FULL PIPELINE: Supervisor Requirements")
    print("="*70)
    print()
    print("This script will:")
    print("  1. Generate reference solutions (Crank-Nicolson)")
    print("  2. Train PINN + Hybrid CAN-PINN on all test cases")
    print("  3. Analyze errors vs reference")
    print("  4. Run three-way comparison (novel contribution)")
    print()
    print("Estimated time: 1-2 hours on GPU")
    print()
    input("Press Enter to start... (or Ctrl+C to cancel)")

    # Step 1: Generate reference solutions
    run_command(
        "python analysis/reference_solver.py",
        "Step 1: Generate Crank-Nicolson Reference Solutions"
    )

    # Step 2: Train models
    run_command(
        "python training/train_improved_allen_cahn.py",
        "Step 2: Train PINN + Hybrid CAN-PINN on All Test Cases"
    )

    # Step 3: Error analysis
    run_command(
        "python analysis/error_analysis.py",
        "Step 3: Analyze Errors vs Reference Solutions"
    )

    # Step 4: Three-way comparison (single test case for now)
    run_command(
        "python analysis/three_way_comparison.py 0.01 sin 2",
        "Step 4: Three-Way Comparison (TC2, sin IC, ε=0.01)"
    )

    # Summary
    print("\n" + "="*70)
    print("  ✓ FULL PIPELINE COMPLETE")
    print("="*70)
    print()
    print("Generated files:")
    print("  • outputs_archive/reference_solutions/  - Reference solutions")
    print("  • outputs_archive/                       - PINN/CAN-PINN outputs")
    print("  • outputs_archive/three_way_results/     - Comparison results")
    print("  • error_analysis_results.json            - Error metrics")
    print()
    print("Next: Review results and prepare for supervisor meeting")
    print()


if __name__ == '__main__':
    main()
