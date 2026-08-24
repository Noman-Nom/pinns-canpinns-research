"""
Three-Way Comparison: Novel Contribution for Mark 5

Compares:
1. Baseline PINN (standard AD, no enhancements)
2. Original CAN-PINN (uses finite differences for ∂²u/∂x² — from the paper)
3. Hybrid CAN-PINN (uses AD + uncertainty + adaptive sampling + L-BFGS — our contribution)

Why this is novel:
  Original CAN-PINN paper uses FD → truncation error O(Δx²) ≈ 1e-4
  Our hybrid replaces FD with AD (exact), keeps all uncertainty/adaptive enhancements
  This is the original contribution beyond reproducing the paper.

Supervisor: "For mark 5 we need something new"
"""

import sys
import torch
import numpy as np
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from allen_cahn_pinn import AllenCahnPINN, CANAllenCahnPINN
from allen_cahn_pinn_improved import ImprovedCANAllenCahnPINN
from residual_adaptive_sampling import ResidualAdaptiveSampler
from train_allen_cahn import generate_training_data, generate_test_data


def run_three_way_comparison(epsilon=0.01, ic_type='sin', test_case=2,
                              adam_epochs=10000, lbfgs_epochs=1000):
    """
    Train and compare three methods side-by-side.

    Args:
        epsilon: Allen-Cahn diffusivity
        ic_type: 'sin' or 'step'
        test_case: 2, 3, or 4 (controls domain)
        adam_epochs: Adam training epochs
        lbfgs_epochs: L-BFGS iterations (hybrid only)

    Returns:
        results dict with PDE losses, times, predictions
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Domain ---
    if test_case == 4:
        x_min, x_max, t_min, t_max = 0.0, 2.0, 0.0, 2.0
        non_uniform = True
    else:
        x_min, x_max, t_min, t_max = 0.0, 1.0, 0.0, 1.0
        non_uniform = False

    print("\n" + "="*70)
    print(f"  THREE-WAY COMPARISON: ε={epsilon}, IC={ic_type}, TC={test_case}")
    print("="*70)
    print("  Method 1: Baseline PINN (standard AD)")
    print("  Method 2: Original CAN-PINN (FD for ∂²u/∂x²)")
    print("  Method 3: Hybrid CAN-PINN (AD + uncertainty + adaptive + L-BFGS)")
    print()

    # --- Generate data (once, shared by all methods) ---
    (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc), (x_pde, t_pde) = generate_training_data(
        ic_type=ic_type, N_ic=200, N_bc=200, N_pde=20000,
        x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
        non_uniform=non_uniform
    )
    x_test, t_test = generate_test_data(
        x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
        N_x=100, N_t=100
    )

    results = {}

    # ===========================================================
    # Method 1: Baseline PINN (standard AD, no enhancements)
    # ===========================================================
    print("-"*70)
    print("  [1/3] Training Baseline PINN (standard automatic differentiation)")
    print("-"*70)

    model_pinn = AllenCahnPINN(epsilon=epsilon, layers=[2, 50, 50, 50, 1], device=device)

    t0 = time.time()
    history_pinn = model_pinn.train(
        x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde,
        x_test, t_test, None,
        epochs=adam_epochs, lr=0.001, print_every=2000
    )
    t_pinn = time.time() - t0

    model_pinn.model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_test, dtype=torch.float32, device=device).reshape(-1, 1)
        t_t = torch.tensor(t_test, dtype=torch.float32, device=device).reshape(-1, 1)
        u_pred_pinn = model_pinn.model(x_t, t_t).cpu().numpy()

    pinn_pde = history_pinn['loss_pde'][-1]
    results['Baseline PINN'] = {
        'pde_loss': float(pinn_pde),
        'total_loss': float(history_pinn['total_loss'][-1]),
        'time': float(t_pinn),
        'u_pred': u_pred_pinn,
        'method': 'Standard AD',
        'optimizer': 'Adam only',
    }
    print(f"  ✓ Done | PDE Loss: {pinn_pde:.4e} | Time: {t_pinn:.1f}s")

    # ===========================================================
    # Method 2: Original CAN-PINN (finite differences for ∂²u/∂x²)
    # ===========================================================
    print()
    print("-"*70)
    print("  [2/3] Training Original CAN-PINN (FD derivatives, O(Δx²) truncation error)")
    print("-"*70)

    model_can = CANAllenCahnPINN(epsilon=epsilon, layers=[2, 50, 50, 50, 1], device=device)

    t0 = time.time()
    history_can = model_can.train(
        x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde,
        x_test, t_test, None,
        epochs=adam_epochs, lr=0.001, print_every=2000
    )
    t_can = time.time() - t0

    model_can.model.eval()
    with torch.no_grad():
        u_pred_can = model_can.model(x_t, t_t).cpu().numpy()

    can_pde = history_can['loss_pde'][-1]
    results['Original CAN-PINN'] = {
        'pde_loss': float(can_pde),
        'total_loss': float(history_can['total_loss'][-1]),
        'time': float(t_can),
        'u_pred': u_pred_can,
        'method': 'Finite Differences',
        'optimizer': 'Adam only',
    }
    print(f"  ✓ Done | PDE Loss: {can_pde:.4e} | Time: {t_can:.1f}s")

    # ===========================================================
    # Method 3: Hybrid CAN-PINN (AD + all enhancements)
    # ===========================================================
    print()
    print("-"*70)
    print("  [3/3] Training Hybrid CAN-PINN (AD + uncertainty + adaptive + L-BFGS)")
    print("-"*70)

    model_hybrid = ImprovedCANAllenCahnPINN(
        epsilon=epsilon, layers=[2, 50, 50, 50, 1], device=device,
        h_adaptive=True, use_uncertainty_weights=True,
        gradient_penalty_weight=1e-5, fourier_features=False
    )

    sampler = ResidualAdaptiveSampler(
        initial_N=len(x_pde.ravel()),
        resample_frequency=3000,
        resample_fraction=0.1,
        keep_best_fraction=0.9
    )

    t0 = time.time()
    history_hybrid = model_hybrid.train(
        x_ic, t_ic, u_ic, x_bc, t_bc, u_bc,
        x_pde, t_pde,
        x_test, t_test, None,
        epochs=adam_epochs, lr=0.001, print_every=2000,
        use_lbfgs=True, lbfgs_epochs=lbfgs_epochs,
        adaptive_sampler=sampler,
        x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max
    )
    t_hybrid = time.time() - t0

    model_hybrid.base_model.eval()
    with torch.no_grad():
        u_pred_hybrid = model_hybrid.forward(x_t, t_t).cpu().numpy()

    hybrid_pde = history_hybrid['loss_pde'][-1]
    results['Hybrid CAN-PINN'] = {
        'pde_loss': float(hybrid_pde),
        'total_loss': float(history_hybrid['total_loss'][-1]),
        'time': float(t_hybrid),
        'u_pred': u_pred_hybrid,
        'method': 'AD + Uncertainty + Adaptive + L-BFGS',
        'optimizer': 'Adam + L-BFGS',
    }
    print(f"  ✓ Done | PDE Loss: {hybrid_pde:.4e} | Time: {t_hybrid:.1f}s")

    # --- Summary table ---
    print()
    print("="*70)
    print("  THREE-WAY COMPARISON RESULTS")
    print("="*70)
    print(f"  ε={epsilon}, IC={ic_type}, Test Case={test_case}")
    print()
    print(f"  {'Method':<30} {'PDE Loss':<14} {'Time (s)':<12} {'vs Baseline'}")
    print("  " + "-"*65)

    baseline = results['Baseline PINN']['pde_loss']
    for name, r in results.items():
        pde = r['pde_loss']
        t = r['time']
        if name == 'Baseline PINN':
            note = "(reference)"
        elif pde < baseline:
            pct = (baseline - pde) / baseline * 100
            note = f"✓ {pct:.1f}% better"
        else:
            pct = (pde - baseline) / baseline * 100
            note = f"✗ {pct:.1f}% worse"
        print(f"  {name:<30} {pde:<14.4e} {t:<12.1f} {note}")

    print()
    print("  Why Hybrid Wins (when it does):")
    print("  • Original CAN-PINN: FD for ∂²u/∂x² → O(Δx²) ≈ 1e-4 truncation error")
    print("  • Hybrid: AD eliminates truncation error entirely")
    print("  • Hybrid adds: uncertainty weighting + adaptive sampling + L-BFGS")
    print("  → More accurate PDE satisfaction, especially for larger ε")

    return results, x_test, t_test


def save_comparison_results(results, x_test, t_test, output_dir='outputs_archive/three_way_results'):
    """Save results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, r in results.items():
        summary[name] = {k: v for k, v in r.items() if k != 'u_pred'}
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        np.savez(
            output_dir / f"{safe_name}_predictions.npz",
            u_pred=r['u_pred'],
            x=x_test,
            t=t_test
        )

    with open(output_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to: {output_dir}/")
    print(f"  • comparison_summary.json")
    for name in results:
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        print(f"  • {safe_name}_predictions.npz")


if __name__ == '__main__':
    import sys

    epsilon  = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    ic_type  = sys.argv[2]         if len(sys.argv) > 2 else 'sin'
    test_case = int(sys.argv[3])   if len(sys.argv) > 3 else 2

    results, x_test, t_test = run_three_way_comparison(
        epsilon=epsilon, ic_type=ic_type, test_case=test_case,
        adam_epochs=10000, lbfgs_epochs=1000
    )
    save_comparison_results(results, x_test, t_test)
