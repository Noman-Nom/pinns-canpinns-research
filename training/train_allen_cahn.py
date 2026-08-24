"""
Training script for Allen-Cahn Equation using PINN and CAN-PINN

This script implements all test cases from Task 2:
- Test Case 1: Basic PDE (Heat Equation) - Already implemented
- Test Case 2: Varying Initial Conditions for Allen-Cahn
- Test Case 3: Varying Diffusivity (ε) for Allen-Cahn
- Test Case 4: Larger and Non-Uniform Grids

Compares standard PINN vs CAN-PINN performance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from allen_cahn_pinn import AllenCahnPINN, CANAllenCahnPINN


def generate_initial_condition(ic_type='sin', x=None):
    """
    Generate initial condition based on type.
    
    Args:
        ic_type: 'sin' for u(x,0) = sin(πx), 'step' for step function
        x: Spatial coordinates
    
    Returns:
        Initial condition values
    """
    if ic_type == 'sin':
        return np.sin(np.pi * x)
    elif ic_type == 'step':
        # Step function: u(x,0) = 1 for x > 0.5, 0 otherwise
        return np.where(x > 0.5, 1.0, 0.0)
    else:
        raise ValueError(f"Unknown initial condition type: {ic_type}")


def generate_training_data(ic_type='sin', N_ic=200, N_bc=200, N_pde=20000,
                          x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0,
                          non_uniform=False):
    """
    Generate training data for the Allen-Cahn equation.
    
    Args:
        ic_type: Initial condition type ('sin' or 'step')
        N_ic: Number of initial condition points
        N_bc: Number of boundary condition points (per boundary)
        N_pde: Number of PDE collocation points
        x_min, x_max: Spatial domain
        t_min, t_max: Temporal domain
        non_uniform: If True, use non-uniform sampling (denser near boundaries)
    
    Returns:
        Training data: (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc), (x_pde, t_pde)
    """
    # Initial condition: u(x, 0)
    if non_uniform:
        # Denser sampling near boundaries
        x_ic = np.concatenate([
            np.random.uniform(x_min, x_min + 0.1, (N_ic // 4, 1)),
            np.random.uniform(x_min + 0.1, x_max - 0.1, (N_ic // 2, 1)),
            np.random.uniform(x_max - 0.1, x_max, (N_ic // 4, 1))
        ])
    else:
        x_ic = np.random.uniform(x_min, x_max, (N_ic, 1))
    
    t_ic = np.zeros((len(x_ic), 1))
    u_ic = generate_initial_condition(ic_type, x_ic).reshape(-1, 1)
    
    # Boundary conditions: u(0, t) = u(1, t) = 0
    x_bc_left = np.zeros((N_bc, 1))
    t_bc_left = np.random.uniform(t_min, t_max, (N_bc, 1))
    u_bc_left = np.zeros((N_bc, 1))
    
    x_bc_right = np.ones((N_bc, 1)) * x_max
    t_bc_right = np.random.uniform(t_min, t_max, (N_bc, 1))
    u_bc_right = np.zeros((N_bc, 1))
    
    x_bc = np.vstack([x_bc_left, x_bc_right])
    t_bc = np.vstack([t_bc_left, t_bc_right])
    u_bc = np.vstack([u_bc_left, u_bc_right])
    
    # PDE collocation points
    if non_uniform:
        # Denser sampling near boundaries and initial time
        x_pde = np.concatenate([
            np.random.uniform(x_min, x_min + 0.1, (N_pde // 4, 1)),
            np.random.uniform(x_min + 0.1, x_max - 0.1, (N_pde // 2, 1)),
            np.random.uniform(x_max - 0.1, x_max, (N_pde // 4, 1))
        ])
        t_pde = np.concatenate([
            np.random.uniform(t_min, t_min + 0.1, (N_pde // 2, 1)),
            np.random.uniform(t_min + 0.1, t_max, (N_pde // 2, 1))
        ])
    else:
        x_pde = np.random.uniform(x_min, x_max, (N_pde, 1))
        t_pde = np.random.uniform(t_min, t_max, (N_pde, 1))
    
    return (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc), (x_pde, t_pde)


def generate_test_data(N_x=100, N_t=100, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
    """
    Generate test data for validation.
    
    Args:
        N_x: Number of spatial points
        N_t: Number of temporal points
        x_min, x_max: Spatial domain
        t_min, t_max: Temporal domain
    
    Returns:
        Test data: (x_test, t_test)
    """
    x_test = np.linspace(x_min, x_max, N_x)
    t_test = np.linspace(t_min, t_max, N_t)
    X_test, T_test = np.meshgrid(x_test, t_test)
    
    x_test = X_test.flatten()
    t_test = T_test.flatten()
    
    return x_test, t_test


def train_and_compare(model_pinn, model_can_pinn, train_data, test_data, 
                      epsilon, ic_type, epochs=10000, lr=0.001):
    """
    Train both PINN and CAN-PINN models and compare results.
    
    Args:
        model_pinn: Standard PINN model
        model_can_pinn: CAN-PINN model
        train_data: Training data tuple
        test_data: Test data tuple (x_test, t_test)
        epsilon: Diffusivity parameter
        ic_type: Initial condition type
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Results dictionary with training histories and metrics
    """
    (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc), (x_pde, t_pde) = train_data
    x_test, t_test = test_data
    
    results = {}
    
    # Train PINN
    print("\n" + "="*70)
    print(f"Training Standard PINN (ε={epsilon}, IC={ic_type})")
    print("="*70)
    start_time = time.time()
    history_pinn = model_pinn.train(
        x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde,
        x_test, t_test, None,  # No exact solution for Allen-Cahn
        epochs=epochs, lr=lr, print_every=1000
    )
    pinn_time = time.time() - start_time
    
    # Train CAN-PINN
    print("\n" + "="*70)
    print(f"Training CAN-PINN (ε={epsilon}, IC={ic_type})")
    print("="*70)
    start_time = time.time()
    history_can = model_can_pinn.train(
        x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde,
        x_test, t_test, None,
        epochs=epochs, lr=lr, print_every=1000
    )
    can_time = time.time() - start_time
    
    # Get predictions for comparison
    model_pinn.model.eval()
    model_can_pinn.model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32, 
                                    device=model_pinn.device).reshape(-1, 1)
        t_test_tensor = torch.tensor(t_test, dtype=torch.float32, 
                                    device=model_pinn.device).reshape(-1, 1)
        
        u_pred_pinn = model_pinn.model(x_test_tensor, t_test_tensor).cpu().numpy()
        u_pred_can = model_can_pinn.model(x_test_tensor, t_test_tensor).cpu().numpy()
    
    results = {
        'pinn': {
            'history': history_pinn,
            'prediction': u_pred_pinn,
            'training_time': pinn_time,
            'final_loss': history_pinn['total_loss'][-1]
        },
        'can_pinn': {
            'history': history_can,
            'prediction': u_pred_can,
            'training_time': can_time,
            'final_loss': history_can['total_loss'][-1]
        },
        'epsilon': epsilon,
        'ic_type': ic_type,
        'test_data': (x_test, t_test)
    }
    
    return results


def visualize_comparison(results, save_path=None):
    """
    Visualize comparison between PINN and CAN-PINN.
    
    Args:
        results: Results dictionary from train_and_compare
        save_path: Path to save the figure
    """
    x_test, t_test = results['test_data']
    u_pred_pinn = results['pinn']['prediction']
    u_pred_can = results['can_pinn']['prediction']
    
    # Reshape for plotting
    N_x = len(np.unique(x_test))
    N_t = len(np.unique(t_test))
    x_plot = x_test.reshape(N_t, N_x)
    t_plot = t_test.reshape(N_t, N_x)
    u_pinn_plot = u_pred_pinn.reshape(N_t, N_x)
    u_can_plot = u_pred_can.reshape(N_t, N_x)
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    
    # 1. PINN Solution (3D)
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(x_plot, t_plot, u_pinn_plot, cmap='viridis', 
                             alpha=0.8, linewidth=0, antialiased=True)
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(x,t)')
    ax1.set_title('PINN Solution')
    plt.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # 2. CAN-PINN Solution (3D)
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(x_plot, t_plot, u_can_plot, cmap='viridis', 
                             alpha=0.8, linewidth=0, antialiased=True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('u(x,t)')
    ax2.set_title('CAN-PINN Solution')
    plt.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # 3. Difference
    diff = np.abs(u_pinn_plot - u_can_plot)
    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(x_plot, t_plot, diff, cmap='hot', 
                             alpha=0.8, linewidth=0, antialiased=True)
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    ax3.set_zlabel('|Difference|')
    ax3.set_title('|PINN - CAN-PINN|')
    plt.colorbar(surf3, ax=ax3, shrink=0.5)
    
    # 4. Solution at different times (PINN)
    ax4 = fig.add_subplot(3, 3, 4)
    t_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
    x_line = np.unique(x_test)
    for t_val in t_slices:
        idx = np.argmin(np.abs(np.unique(t_test) - t_val))
        ax4.plot(x_line, u_pinn_plot[idx, :], '-', linewidth=1.5, 
                label=f't={t_val:.2f}', alpha=0.7)
    ax4.set_xlabel('x')
    ax4.set_ylabel('u(x,t)')
    ax4.set_title('PINN: Solution at Different Times')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Solution at different times (CAN-PINN)
    ax5 = fig.add_subplot(3, 3, 5)
    for t_val in t_slices:
        idx = np.argmin(np.abs(np.unique(t_test) - t_val))
        ax5.plot(x_line, u_can_plot[idx, :], '-', linewidth=1.5, 
                label=f't={t_val:.2f}', alpha=0.7)
    ax5.set_xlabel('x')
    ax5.set_ylabel('u(x,t)')
    ax5.set_title('CAN-PINN: Solution at Different Times')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Comparison at different times
    ax6 = fig.add_subplot(3, 3, 6)
    for t_val in [0.0, 0.5, 1.0]:
        idx = np.argmin(np.abs(np.unique(t_test) - t_val))
        ax6.plot(x_line, u_pinn_plot[idx, :], '--', linewidth=2, 
                label=f'PINN, t={t_val:.2f}', alpha=0.7)
        ax6.plot(x_line, u_can_plot[idx, :], '-', linewidth=1.5, 
                label=f'CAN-PINN, t={t_val:.2f}', alpha=0.7)
    ax6.set_xlabel('x')
    ax6.set_ylabel('u(x,t)')
    ax6.set_title('Comparison at Different Times')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 7. Loss comparison
    ax7 = fig.add_subplot(3, 3, 7)
    epochs_pinn = range(len(results['pinn']['history']['total_loss']))
    epochs_can = range(len(results['can_pinn']['history']['total_loss']))
    ax7.semilogy(epochs_pinn, results['pinn']['history']['total_loss'], 
                label='PINN', linewidth=2)
    ax7.semilogy(epochs_can, results['can_pinn']['history']['total_loss'], 
                label='CAN-PINN', linewidth=2)
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Total Loss (log scale)')
    ax7.set_title('Loss Convergence Comparison')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Component losses comparison
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.semilogy(epochs_pinn, results['pinn']['history']['loss_pde'], 
                label='PINN PDE', linewidth=1.5, alpha=0.7)
    ax8.semilogy(epochs_can, results['can_pinn']['history']['loss_pde'], 
                label='CAN-PINN PDE', linewidth=1.5, alpha=0.7)
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('PDE Loss (log scale)')
    ax8.set_title('PDE Loss Comparison')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Adaptive weights (if available)
    ax9 = fig.add_subplot(3, 3, 9)
    if results['can_pinn']['history']['weights'] is not None:
        weights = results['can_pinn']['history']['weights']
        # Create epochs array matching weights length
        weights_len = len(weights['ic'])
        epochs_weights = range(weights_len)
        
        # Ensure all weight arrays have same length
        min_len = min(len(weights['ic']), len(weights['bc']), len(weights['pde']))
        epochs_plot = list(range(min_len))
        weights_ic_plot = weights['ic'][:min_len]
        weights_bc_plot = weights['bc'][:min_len]
        weights_pde_plot = weights['pde'][:min_len]
        
        ax9.plot(epochs_plot, weights_ic_plot, label='IC Weight', linewidth=1.5)
        ax9.plot(epochs_plot, weights_bc_plot, label='BC Weight', linewidth=1.5)
        ax9.plot(epochs_plot, weights_pde_plot, label='PDE Weight', linewidth=1.5)
        ax9.set_xlabel('Epoch')
        ax9.set_ylabel('Weight')
        ax9.set_title('CAN-PINN Adaptive Loss Weights')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    else:
        ax9.text(0.5, 0.5, 'No adaptive weights', 
                ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('Adaptive Loss Weights')
    
    plt.suptitle(f"Allen-Cahn Equation: ε={results['epsilon']}, IC={results['ic_type']}", 
                fontsize=14, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def run_test_case(test_case, epsilon=0.01, ic_type='sin', epochs=10000, lr=0.001):
    """
    Run a specific test case.
    
    Args:
        test_case: Test case number (2, 3, or 4)
        epsilon: Diffusivity parameter
        ic_type: Initial condition type ('sin' or 'step')
        epochs: Number of training epochs
        lr: Learning rate
    """
    print("\n" + "="*70)
    print(f"TEST CASE {test_case}: Allen-Cahn Equation")
    print(f"  ε = {epsilon}, Initial Condition: {ic_type}")
    print("="*70)
    
    # Determine domain based on test case
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
    
    # Initialize models
    layers = [2, 50, 50, 50, 1]
    model_pinn = AllenCahnPINN(epsilon=epsilon, layers=layers, device='cuda')
    model_can_pinn = CANAllenCahnPINN(epsilon=epsilon, layers=layers, device='cuda', h=0.01)
    
    # Train and compare
    results = train_and_compare(
        model_pinn, model_can_pinn, train_data, test_data,
        epsilon, ic_type, epochs, lr
    )
    
    # Visualize
    save_path = f"allen_cahn_tc{test_case}_eps{epsilon}_ic{ic_type}.png"
    visualize_comparison(results, save_path)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"PINN - Final Loss: {results['pinn']['final_loss']:.6e}, "
          f"Training Time: {results['pinn']['training_time']:.2f}s")
    print(f"CAN-PINN - Final Loss: {results['can_pinn']['final_loss']:.6e}, "
          f"Training Time: {results['can_pinn']['training_time']:.2f}s")
    print(f"Speedup: {results['pinn']['training_time'] / results['can_pinn']['training_time']:.2f}x")
    print("="*70)
    
    return results


def main():
    """Main function to run all test cases."""
    print("="*70)
    print("Task 2: Allen-Cahn Equation - PINN vs CAN-PINN Comparison")
    print("="*70)
    
    all_results = {}
    
    # Test Case 2: Varying Initial Conditions
    print("\n\n### TEST CASE 2: Varying Initial Conditions ###")
    
    # 2a: sin(πx) initial condition
    results_2a = run_test_case(2, epsilon=0.01, ic_type='sin', epochs=10000)
    all_results['TC2_sin'] = results_2a
    
    # 2b: Step function initial condition
    results_2b = run_test_case(2, epsilon=0.01, ic_type='step', epochs=10000)
    all_results['TC2_step'] = results_2b
    
    # Test Case 3: Varying Diffusivity
    print("\n\n### TEST CASE 3: Varying Diffusivity ###")
    
    # 3a: ε = 0.01 (sharp interface)
    results_3a = run_test_case(3, epsilon=0.01, ic_type='sin', epochs=10000)
    all_results['TC3_eps001'] = results_3a
    
    # 3b: ε = 0.05 (smoother interface)
    results_3b = run_test_case(3, epsilon=0.05, ic_type='sin', epochs=10000)
    all_results['TC3_eps005'] = results_3b
    
    # Test Case 4: Larger and Non-Uniform Grids
    print("\n\n### TEST CASE 4: Larger and Non-Uniform Grids ###")
    results_4 = run_test_case(4, epsilon=0.01, ic_type='sin', epochs=10000)
    all_results['TC4'] = results_4
    
    # Final comparison summary
    print("\n\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    for name, result in all_results.items():
        print(f"\n{name}:")
        print(f"  PINN Loss: {result['pinn']['final_loss']:.6e}, "
              f"Time: {result['pinn']['training_time']:.2f}s")
        print(f"  CAN-PINN Loss: {result['can_pinn']['final_loss']:.6e}, "
              f"Time: {result['can_pinn']['training_time']:.2f}s")
        improvement = (result['pinn']['final_loss'] - result['can_pinn']['final_loss']) / result['pinn']['final_loss'] * 100
        print(f"  Improvement: {improvement:.2f}%")


if __name__ == "__main__":
    main()

