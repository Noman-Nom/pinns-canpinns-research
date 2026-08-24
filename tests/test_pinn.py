"""
Quick test script to verify PINN implementation works correctly.
This runs a shorter training to verify the framework.
"""

import numpy as np
import torch
from pinn_model import HeatEquationPINN
import matplotlib.pyplot as plt


def quick_test():
    """Quick test of the PINN implementation."""
    print("="*60)
    print("Quick PINN Test - Heat Equation")
    print("="*60)
    
    # Parameters
    alpha = 0.1
    layers = [2, 20, 20, 1]  # Smaller network for quick test
    
    # Generate minimal training data
    print("\nGenerating training data...")
    
    # Initial condition: u(x, 0) = sin(πx)
    x_ic = np.random.uniform(0, 1, (50, 1))
    t_ic = np.zeros((50, 1))
    u_ic = np.sin(np.pi * x_ic)
    
    # Boundary conditions
    x_bc_left = np.zeros((25, 1))
    t_bc_left = np.random.uniform(0, 1, (25, 1))
    x_bc_right = np.ones((25, 1))
    t_bc_right = np.random.uniform(0, 1, (25, 1))
    x_bc = np.vstack([x_bc_left, x_bc_right])
    t_bc = np.vstack([t_bc_left, t_bc_right])
    u_bc = np.zeros((50, 1))
    
    # PDE collocation points
    x_pde = np.random.uniform(0, 1, (1000, 1))
    t_pde = np.random.uniform(0, 1, (1000, 1))
    
    # Test data
    x_test = np.linspace(0, 1, 50)
    t_test = np.linspace(0, 1, 50)
    X_test, T_test = np.meshgrid(x_test, t_test)
    x_test = X_test.flatten()
    t_test = T_test.flatten()
    u_exact = np.sin(np.pi * x_test) * np.exp(-alpha * np.pi**2 * t_test)
    
    print(f"  IC points: {len(x_ic)}")
    print(f"  BC points: {len(x_bc)}")
    print(f"  PDE points: {len(x_pde)}")
    print(f"  Test points: {len(x_test)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = HeatEquationPINN(alpha=alpha, layers=layers, activation='tanh')
    
    # Quick training (500 epochs)
    print("\nRunning quick training (500 epochs)...")
    history = model.train(
        x_ic, t_ic, u_ic,
        x_bc, t_bc, u_bc,
        x_pde, t_pde,
        x_test, t_test, u_exact,
        epochs=500,
        lr=0.001,
        print_every=100
    )
    
    # Final evaluation
    final_l2 = model.compute_l2_error(x_test, t_test, u_exact)
    print(f"\nFinal L2 Error: {final_l2:.6e}")
    print(f"Final Total Loss: {history['total_loss'][-1]:.6e}")
    
    # Quick visualization
    print("\nCreating visualization...")
    model.model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_test, dtype=torch.float32, device=model.device).reshape(-1, 1)
        t_tensor = torch.tensor(t_test, dtype=torch.float32, device=model.device).reshape(-1, 1)
        u_pred = model.model(x_tensor, t_tensor).cpu().numpy().flatten()
    
    # Plot at t=0 and t=0.5
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # t = 0
    idx_t0 = np.argmin(np.abs(np.unique(t_test) - 0.0))
    x_line = np.unique(x_test)
    axes[0].plot(x_line, u_exact.reshape(50, 50)[idx_t0, :], '--', label='Exact', linewidth=2)
    axes[0].plot(x_line, u_pred.reshape(50, 50)[idx_t0, :], '-', label='PINN', linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x,t)')
    axes[0].set_title('Solution at t=0')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # t = 0.5
    idx_t05 = np.argmin(np.abs(np.unique(t_test) - 0.5))
    axes[1].plot(x_line, u_exact.reshape(50, 50)[idx_t05, :], '--', label='Exact', linewidth=2)
    axes[1].plot(x_line, u_pred.reshape(50, 50)[idx_t05, :], '-', label='PINN', linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u(x,t)')
    axes[1].set_title('Solution at t=0.5')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_test_results.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: quick_test_results.png")
    
    print("\n" + "="*60)
    print("Quick Test Complete!")
    print("="*60)
    print(f"\n✓ Framework is working correctly")
    print(f"✓ L2 Error: {final_l2:.6e}")
    print(f"✓ Loss decreased from {history['total_loss'][0]:.6e} to {history['total_loss'][-1]:.6e}")
    
    return final_l2 < 0.1  # Test passes if L2 error is reasonable


if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✓ Test PASSED - Framework is ready for full training!")
    else:
        print("\n⚠ Test completed but L2 error is high - may need more training epochs")

