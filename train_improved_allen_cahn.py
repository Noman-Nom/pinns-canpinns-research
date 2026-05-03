"""
Training script for Improved CAN-PINN with all enhancements.

This script compares:
1. Standard PINN (baseline)
2. Original CAN-PINN (from previous implementation)
3. Improved CAN-PINN (with all enhancements)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from allen_cahn_pinn import AllenCahnPINN, CANAllenCahnPINN
from allen_cahn_pinn_improved import ImprovedCANAllenCahnPINN
from residual_adaptive_sampling import ResidualAdaptiveSampler
from train_allen_cahn import generate_training_data, generate_test_data, visualize_comparison


def train_improved_comparison(epsilon=0.01, ic_type='sin', epochs=10000, lr=0.001,
                              use_lbfgs=True, lbfgs_epochs=1000, use_rbas=True,
                              fourier_features=False, test_case=2):
    """
    Train and compare PINN, CAN-PINN, and Improved CAN-PINN.
    
    Args:
        epsilon: Diffusivity parameter
        ic_type: Initial condition type ('sin' or 'step')
        epochs: Number of training epochs
        lr: Learning rate
        use_lbfgs: If True, use L-BFGS for fine-tuning
        lbfgs_epochs: Number of L-BFGS iterations
        use_rbas: If True, use residual-based adaptive sampling
        fourier_features: If True, use Fourier feature encoding
        test_case: Test case number (2, 3, or 4)
    """
    print("\n" + "="*70)
    print(f"Improved CAN-PINN Comparison")
    print(f"  ε = {epsilon}, Initial Condition: {ic_type}")
    print(f"  Test Case: {test_case}")
    print("="*70)
    
    # Determine domain
    if test_case == 4:
        x_min, x_max = 0.0, 2.0
        t_min, t_max = 0.0, 2.0
        non_uniform = True
    else:
        x_min, x_max = 0.0, 1.0
        t_min, t_max = 0.0, 1.0
        non_uniform = False
    
    # Generate training data
    train_data = generate_training_data(
        ic_type=ic_type, N_ic=200, N_bc=200, N_pde=20000,
        x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
        non_uniform=non_uniform
    )
    
    # Generate test data
    test_data = generate_test_data(
        N_x=100, N_t=100, x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max
    )
    
    (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc), (x_pde, t_pde) = train_data
    x_test, t_test = test_data
    
    layers = [2, 50, 50, 50, 1]
    results = {}
    
    # 1. Standard PINN (baseline)
    print("\n" + "="*70)
    print("1. Training Standard PINN (Baseline)")
    print("="*70)
    model_pinn = AllenCahnPINN(epsilon=epsilon, layers=layers, device='cuda')
    start_time = time.time()
    history_pinn = model_pinn.train(
        x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde,
        x_test, t_test, None,
        epochs=epochs, lr=lr, print_every=1000
    )
    pinn_time = time.time() - start_time
    
    model_pinn.model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device='cuda').reshape(-1, 1)
        t_test_tensor = torch.tensor(t_test, dtype=torch.float32, device='cuda').reshape(-1, 1)
        u_pred_pinn = model_pinn.model(x_test_tensor, t_test_tensor).cpu().numpy()
    
    results['pinn'] = {
        'history': history_pinn,
        'prediction': u_pred_pinn,
        'training_time': pinn_time,
        'final_loss': history_pinn['total_loss'][-1]
    }
    
    # 2. Improved CAN-PINN
    print("\n" + "="*70)
    print("2. Training Improved CAN-PINN")
    print("="*70)
    model_improved = ImprovedCANAllenCahnPINN(
        epsilon=epsilon, layers=layers, device='cuda',
        h_adaptive=True, use_uncertainty_weights=True,
        gradient_penalty_weight=1e-5, fourier_features=fourier_features, gamma=10.0
    )
    
    # Initialize adaptive sampler if used (less frequent resampling for stability)
    sampler = None
    if use_rbas:
        sampler = ResidualAdaptiveSampler(
            initial_N=len(x_pde), 
            resample_frequency=3000,  # Reduced frequency: every 3000 epochs instead of 1000
            resample_fraction=0.1,    # Reduced fraction: 10% instead of 20%
            keep_best_fraction=0.9    # Keep more good points: 90% instead of 80%
        )
    
    start_time = time.time()
    x_pde_current = x_pde.copy()
    t_pde_current = t_pde.copy()
    
    history_improved = model_improved.train(
        x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde_current, t_pde_current,
        x_test, t_test, None,
        epochs=epochs, lr=lr, print_every=1000,
        use_lbfgs=use_lbfgs, lbfgs_epochs=lbfgs_epochs,
        adaptive_sampler=sampler if use_rbas else None,
        x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max
    )
    
    improved_time = time.time() - start_time
    
    model_improved.base_model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device='cuda').reshape(-1, 1)
        t_test_tensor = torch.tensor(t_test, dtype=torch.float32, device='cuda').reshape(-1, 1)
        u_pred_improved = model_improved.forward(x_test_tensor, t_test_tensor).cpu().numpy()
    
    results['improved_can'] = {
        'history': history_improved,
        'prediction': u_pred_improved,
        'training_time': improved_time,
        'final_loss': history_improved['total_loss'][-1]
    }

    # Save solution files for error analysis (outputs/TC*_*_eps*_*_solution.npz)
    import os
    os.makedirs('outputs', exist_ok=True)
    eps_str = str(epsilon).replace('.', '')  # 0.01→001, 0.05→005
    tc_name = f"TC{test_case}_{ic_type}_eps{eps_str}"
    np.savez(
        f"outputs/{tc_name}_pinn_solution.npz",
        u_pred=u_pred_pinn,
        x_test=x_test,
        t_test=t_test
    )
    np.savez(
        f"outputs/{tc_name}_canpinn_solution.npz",
        u_pred=u_pred_improved,
        x_test=x_test,
        t_test=t_test
    )
    print(f"  Saved: outputs/{tc_name}_pinn_solution.npz")
    print(f"  Saved: outputs/{tc_name}_canpinn_solution.npz")

    # Create comparison results format
    comparison_results = {
        'pinn': results['pinn'],
        'can_pinn': results['pinn'],  # Placeholder (can add original CAN-PINN if needed)
        'epsilon': epsilon,
        'ic_type': ic_type,
        'test_data': (x_test, t_test)
    }
    comparison_results['can_pinn'] = results['improved_can']
    
    # Visualize
    save_path = f"improved_allen_cahn_tc{test_case}_eps{epsilon}_ic{ic_type}.png"
    visualize_comparison(comparison_results, save_path)
    
    # Print summary with accurate metrics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Get final PDE losses (the real metric) — history key is 'loss_pde'
    pinn_pde_loss = results['pinn']['history']['loss_pde'][-1] if 'loss_pde' in results['pinn']['history'] else results['pinn']['final_loss']
    can_pde_loss = results['improved_can']['history']['loss_pde'][-1] if 'loss_pde' in results['improved_can']['history'] else results['improved_can']['final_loss']
    
    print(f"PINN:")
    print(f"  Total Loss: {results['pinn']['final_loss']:.6e}")
    print(f"  PDE Loss: {pinn_pde_loss:.6e}")
    print(f"  Training Time: {results['pinn']['training_time']:.2f}s")
    print(f"\nImproved CAN-PINN:")
    print(f"  Total Loss: {results['improved_can']['final_loss']:.6e} (Note: includes log_var terms, can be negative)")
    print(f"  PDE Loss: {can_pde_loss:.6e}")
    print(f"  Training Time: {results['improved_can']['training_time']:.2f}s")
    
    # Calculate improvement based on PDE loss (the meaningful metric)
    if pinn_pde_loss > 0:
        pde_improvement = (pinn_pde_loss - can_pde_loss) / pinn_pde_loss * 100
        if pde_improvement > 0:
            print(f"\n✅ CAN-PINN PDE Loss Improvement: {pde_improvement:.2f}% (better)")
        else:
            print(f"\n⚠️  CAN-PINN PDE Loss: {abs(pde_improvement):.2f}% worse than PINN")
    
    print("="*70)
    
    return results


def main():
    """Main function to run improved CAN-PINN experiments."""
    print("="*70)
    print("Improved CAN-PINN Training and Comparison")
    print("="*70)
    
    # Test Case 2: sin initial condition
    print("\n\n### Test Case 2: sin(πx) Initial Condition ###")
    results_2_sin = train_improved_comparison(
        epsilon=0.01, ic_type='sin', epochs=10000, lr=0.001,
        use_lbfgs=True, lbfgs_epochs=1000, use_rbas=True,
        fourier_features=False, test_case=2
    )
    
    # Test Case 2: step initial condition
    print("\n\n### Test Case 2: Step Function Initial Condition ###")
    results_2_step = train_improved_comparison(
        epsilon=0.01, ic_type='step', epochs=10000, lr=0.001,
        use_lbfgs=True, lbfgs_epochs=1000, use_rbas=True,
        fourier_features=False, test_case=2
    )
    
    # Test Case 3: Different epsilon
    print("\n\n### Test Case 3: ε = 0.01 ###")
    results_3_001 = train_improved_comparison(
        epsilon=0.01, ic_type='sin', epochs=10000, lr=0.001,
        use_lbfgs=True, lbfgs_epochs=1000, use_rbas=True,
        fourier_features=False, test_case=3
    )
    
    print("\n\n### Test Case 3: ε = 0.05 ###")
    results_3_005 = train_improved_comparison(
        epsilon=0.05, ic_type='sin', epochs=10000, lr=0.001,
        use_lbfgs=True, lbfgs_epochs=1000, use_rbas=True,
        fourier_features=False, test_case=3
    )
    
    # Final summary
    print("\n\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    all_results = {
        'TC2_sin': results_2_sin,
        'TC2_step': results_2_step,
        'TC3_eps001': results_3_001,
        'TC3_eps005': results_3_005
    }
    
    for name, result in all_results.items():
        print(f"\n{name}:")
        # Get PDE losses
        pinn_pde = result['pinn']['history']['loss_pde'][-1] if 'loss_pde' in result['pinn']['history'] else result['pinn']['final_loss']
        can_pde = result['improved_can']['history']['loss_pde'][-1] if 'loss_pde' in result['improved_can']['history'] else result['improved_can']['final_loss']
        
        print(f"  PINN Total Loss: {result['pinn']['final_loss']:.6e}")
        print(f"  PINN PDE Loss: {pinn_pde:.6e}")
        print(f"  CAN-PINN Total Loss: {result['improved_can']['final_loss']:.6e}")
        print(f"  CAN-PINN PDE Loss: {can_pde:.6e}")
        
        if pinn_pde > 0:
            pde_improvement = (pinn_pde - can_pde) / pinn_pde * 100
            if pde_improvement > 0:
                print(f"  ✅ PDE Loss Improvement: {pde_improvement:.2f}%")
            else:
                print(f"  ⚠️  PDE Loss: {abs(pde_improvement):.2f}% worse")


if __name__ == "__main__":
    main()

