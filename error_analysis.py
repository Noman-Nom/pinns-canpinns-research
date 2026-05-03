"""
Error Analysis: Compare PINN/CAN-PINN vs Crank-Nicolson Reference Solutions

Computes for each model and test case:
  - L2 relative error:  ||u_pinn - u_ref||_2 / ||u_ref||_2
  - L∞ absolute error:  max |u_pinn - u_ref|
  - L∞ relative error:  max |u_pinn - u_ref| / max |u_ref|

Supervisor Requirement: "I need exact test cases with answers"
"""

import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import json


def load_reference(tc_name):
    """Load Crank-Nicolson reference solution."""
    ref_file = Path('reference_solutions') / f"{tc_name}_reference.npz"
    if not ref_file.exists():
        raise FileNotFoundError(f"Run reference_solver.py first. Missing: {ref_file}")
    d = np.load(ref_file)
    return d['u'], d['x'], d['t']


def load_pinn_output(filepath):
    """Load saved PINN/CAN-PINN predictions."""
    d = np.load(filepath)
    return d['u_pred'].ravel(), d['x_test'].ravel(), d['t_test'].ravel()


def compute_errors(u_ref, x_ref, t_ref, u_pred_flat, x_pred_flat, t_pred_flat):
    """
    Interpolate predictions onto the reference grid, then compute error metrics.

    Args:
        u_ref:        Reference solution [nt_ref, nx_ref]
        x_ref, t_ref: 1D reference grid arrays
        u_pred_flat:  Flat PINN predictions [N]
        x_pred_flat:  Flat x coordinates for predictions [N]
        t_pred_flat:  Flat t coordinates for predictions [N]

    Returns:
        dict with L2_relative, Linf_absolute, Linf_relative, MAE
    """
    nx_pred = len(np.unique(x_pred_flat))
    nt_pred = len(np.unique(t_pred_flat))
    x_unique = np.unique(x_pred_flat)
    t_unique = np.unique(t_pred_flat)

    # Reshape predictions to [nt, nx] grid
    u_pred_grid = u_pred_flat.reshape(nt_pred, nx_pred)

    # Build interpolator on prediction grid
    interp = RegularGridInterpolator(
        (t_unique, x_unique),
        u_pred_grid,
        method='linear',
        bounds_error=False,
        fill_value=None
    )

    # Evaluate on reference grid
    T_ref, X_ref = np.meshgrid(t_ref, x_ref, indexing='ij')  # [nt_ref, nx_ref]
    pts = np.column_stack([T_ref.ravel(), X_ref.ravel()])
    u_pred_on_ref = interp(pts).reshape(u_ref.shape)

    diff = u_ref - u_pred_on_ref
    ref_norm_l2 = np.sqrt(np.mean(u_ref**2))
    ref_norm_inf = np.max(np.abs(u_ref))

    return {
        'L2_relative':   float(np.sqrt(np.mean(diff**2)) / (ref_norm_l2 + 1e-12)),
        'Linf_absolute': float(np.max(np.abs(diff))),
        'Linf_relative': float(np.max(np.abs(diff)) / (ref_norm_inf + 1e-12)),
        'MAE':           float(np.mean(np.abs(diff))),
    }


def run_error_analysis():
    """Main entry point: compare all models against reference solutions."""

    print("=" * 70)
    print("  Error Analysis: PINN & CAN-PINN vs Crank-Nicolson Reference")
    print("=" * 70)
    print()

    # Maps: (tc_name used by reference_solver) → (output file prefix from training)
    test_cases = [
        {'ref_name': 'TC2_sin_eps001',  'file_prefix': 'TC2_sin_eps001'},
        {'ref_name': 'TC2_step_eps001', 'file_prefix': 'TC2_step_eps001'},
        {'ref_name': 'TC3_sin_eps001',  'file_prefix': 'TC3_sin_eps001'},
        {'ref_name': 'TC3_sin_eps005',  'file_prefix': 'TC3_sin_eps005'},
    ]

    all_results = {}
    any_missing = False

    for tc in test_cases:
        ref_name = tc['ref_name']
        prefix = tc['file_prefix']
        print(f"Test Case: {ref_name}")
        print("-" * 70)

        # Load reference
        try:
            u_ref, x_ref, t_ref = load_reference(ref_name)
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            any_missing = True
            print()
            continue

        tc_results = {'reference': {'nx': len(x_ref), 'nt': len(t_ref),
                                    'u_min': float(u_ref.min()), 'u_max': float(u_ref.max())}}
        print(f"  Reference: {len(x_ref)}×{len(t_ref)} grid, "
              f"u ∈ [{u_ref.min():.4f}, {u_ref.max():.4f}]")

        for model_label, suffix in [('PINN', 'pinn'), ('CAN-PINN', 'canpinn')]:
            filepath = Path('outputs') / f"{prefix}_{suffix}_solution.npz"
            if not filepath.exists():
                print(f"  ✗ {model_label}: missing {filepath}")
                print(f"      → Run: python train_improved_allen_cahn.py")
                any_missing = True
                continue

            u_pred, x_pred, t_pred = load_pinn_output(filepath)
            errors = compute_errors(u_ref, x_ref, t_ref, u_pred, x_pred, t_pred)
            tc_results[model_label] = errors

            print(f"  {model_label}:")
            print(f"    L2 relative:   {errors['L2_relative']:.4e}")
            print(f"    L∞ absolute:   {errors['Linf_absolute']:.4e}")
            print(f"    L∞ relative:   {errors['Linf_relative']:.4e}")
            print(f"    MAE:           {errors['MAE']:.4e}")

        all_results[ref_name] = tc_results
        print()

    # ── Summary table ──────────────────────────────────────────────────────
    print("=" * 70)
    print("  SUMMARY TABLE — L2 Relative Error vs Crank-Nicolson Reference")
    print("=" * 70)
    print(f"  {'Test Case':<22} {'PINN L2':>12} {'CAN-PINN L2':>14} {'Winner':>12}")
    print("  " + "-" * 64)

    for ref_name, res in all_results.items():
        pinn_l2 = res.get('PINN', {}).get('L2_relative', float('nan'))
        can_l2  = res.get('CAN-PINN', {}).get('L2_relative', float('nan'))
        if not (np.isnan(pinn_l2) or np.isnan(can_l2)):
            winner = 'PINN' if pinn_l2 < can_l2 else 'CAN-PINN'
            pct = abs(pinn_l2 - can_l2) / max(pinn_l2, can_l2) * 100
            note = f"{winner} ({pct:.0f}%↓)"
        else:
            note = "N/A"
        print(f"  {ref_name:<22} {pinn_l2:>12.4e} {can_l2:>14.4e} {note:>12}")

    print()

    # Save JSON
    with open('error_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("  Results saved → error_analysis_results.json")

    if any_missing:
        print()
        print("  ⚠  Some files were missing. Re-run training first:")
        print("     python train_improved_allen_cahn.py")
        print("  Then re-run this script:")
        print("     python error_analysis.py")

    return all_results


if __name__ == '__main__':
    run_error_analysis()
