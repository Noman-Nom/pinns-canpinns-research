"""
Training script for solving the Heat Equation using Physics-Informed Neural Networks.

Heat Equation: ∂u/∂t = α * ∂²u/∂x²

Initial Condition: u(x, 0) = sin(πx)
Boundary Conditions: u(0, t) = u(1, t) = 0
Analytical Solution: u(x, t) = sin(πx) * exp(-απ²t)

Domain: x ∈ [0, 1], t ∈ [0, 1]
"""

import numpy as np
import matplotlib.pyplot as plt
from pinn_model import HeatEquationPINN
import torch


def generate_training_data(N_ic=100, N_bc=100, N_pde=10000):
    """
    Generate training data for the Heat equation.
    
    Args:
        N_ic: Number of initial condition points
        N_bc: Number of boundary condition points (per boundary)
        N_pde: Number of PDE collocation points
    
    Returns:
        Training data: (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc), (x_pde, t_pde)
    """
    # Domain
    x_min, x_max = 0.0, 1.0
    t_min, t_max = 0.0, 1.0
    
    # Initial condition: u(x, 0) = sin(πx)
    x_ic = np.random.uniform(x_min, x_max, (N_ic, 1))
    t_ic = np.zeros((N_ic, 1))  # t = 0
    u_ic = np.sin(np.pi * x_ic)  # Initial condition
    
    # Boundary conditions: u(0, t) = u(1, t) = 0
    # Left boundary: x = 0
    x_bc_left = np.zeros((N_bc, 1))
    t_bc_left = np.random.uniform(t_min, t_max, (N_bc, 1))
    u_bc_left = np.zeros((N_bc, 1))
    
    # Right boundary: x = 1
    x_bc_right = np.ones((N_bc, 1))
    t_bc_right = np.random.uniform(t_min, t_max, (N_bc, 1))
    u_bc_right = np.zeros((N_bc, 1))
    
    # Combine boundary conditions
    x_bc = np.vstack([x_bc_left, x_bc_right])
    t_bc = np.vstack([t_bc_left, t_bc_right])
    u_bc = np.vstack([u_bc_left, u_bc_right])
    
    # PDE collocation points (Latin Hypercube Sampling)
    x_pde = np.random.uniform(x_min, x_max, (N_pde, 1))
    t_pde = np.random.uniform(t_min, t_max, (N_pde, 1))
    
    return (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc), (x_pde, t_pde)


def generate_test_data(N_x=100, N_t=100):
    """
    Generate test data for validation.
    
    Args:
        N_x: Number of spatial points
        N_t: Number of temporal points
    
    Returns:
        Test data: (x_test, t_test, u_exact)
    """
    x_test = np.linspace(0, 1, N_x)
    t_test = np.linspace(0, 1, N_t)
    X_test, T_test = np.meshgrid(x_test, t_test)
    
    x_test = X_test.flatten()
    t_test = T_test.flatten()
    
    # Analytical solution
    alpha = 0.1
    u_exact = np.sin(np.pi * x_test) * np.exp(-alpha * np.pi**2 * t_test)
    
    return x_test, t_test, u_exact


def visualize_results(model, x_test, t_test, u_exact, alpha=0.1, save_path=None):
    """
    Visualize the results: predicted vs exact solution.
    
    Args:
        model: Trained PINN model
        x_test: Test spatial coordinates
        t_test: Test temporal coordinates
        u_exact: Exact solution
        alpha: Thermal diffusivity
        save_path: Path to save the figure
    """
    # Reshape for plotting
    N_x = len(np.unique(x_test))
    N_t = len(np.unique(t_test))
    x_plot = x_test.reshape(N_t, N_x)
    t_plot = t_test.reshape(N_t, N_x)
    
    # Get predictions
    model.model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_test, dtype=torch.float32, 
                                device=model.device).reshape(-1, 1)
        t_tensor = torch.tensor(t_test, dtype=torch.float32, 
                                device=model.device).reshape(-1, 1)
        u_pred = model.model(x_tensor, t_tensor).cpu().numpy().flatten()
    
    u_pred_plot = u_pred.reshape(N_t, N_x)
    u_exact_plot = u_exact.reshape(N_t, N_x)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Exact solution
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(x_plot, t_plot, u_exact_plot, cmap='viridis', 
                             alpha=0.8, linewidth=0, antialiased=True)
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(x,t)')
    ax1.set_title('Exact Solution')
    plt.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # 2. Predicted solution
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(x_plot, t_plot, u_pred_plot, cmap='viridis', 
                             alpha=0.8, linewidth=0, antialiased=True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('u(x,t)')
    ax2.set_title('PINN Predicted Solution')
    plt.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # 3. Error
    error = np.abs(u_exact_plot - u_pred_plot)
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(x_plot, t_plot, error, cmap='hot', 
                             alpha=0.8, linewidth=0, antialiased=True)
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    ax3.set_zlabel('|Error|')
    ax3.set_title('Absolute Error')
    plt.colorbar(surf3, ax=ax3, shrink=0.5)
    
    # 4. Solution at different times
    ax4 = fig.add_subplot(2, 3, 4)
    t_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
    x_line = np.linspace(0, 1, N_x)
    for t_val in t_slices:
        idx = np.argmin(np.abs(np.unique(t_test) - t_val))
        ax4.plot(x_line, u_exact_plot[idx, :], '--', linewidth=2, 
                label=f'Exact, t={t_val:.2f}')
        ax4.plot(x_line, u_pred_plot[idx, :], '-', linewidth=1.5, 
                label=f'PINN, t={t_val:.2f}', alpha=0.7)
    ax4.set_xlabel('x')
    ax4.set_ylabel('u(x,t)')
    ax4.set_title('Solution at Different Times')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Solution at different spatial points
    ax5 = fig.add_subplot(2, 3, 5)
    x_slices = [0.25, 0.5, 0.75]
    t_line = np.linspace(0, 1, N_t)
    for x_val in x_slices:
        idx = np.argmin(np.abs(np.unique(x_test) - x_val))
        ax5.plot(t_line, u_exact_plot[:, idx], '--', linewidth=2, 
                label=f'Exact, x={x_val:.2f}')
        ax5.plot(t_line, u_pred_plot[:, idx], '-', linewidth=1.5, 
                label=f'PINN, x={x_val:.2f}', alpha=0.7)
    ax5.set_xlabel('t')
    ax5.set_ylabel('u(x,t)')
    ax5.set_title('Solution at Different Spatial Points')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Error distribution
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(error.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Absolute Error')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Error Distribution')
    ax6.grid(True, alpha=0.3)
    ax6.axvline(np.mean(error), color='r', linestyle='--', 
               label=f'Mean: {np.mean(error):.6f}')
    ax6.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history (losses and L2 error).
    
    Args:
        history: Training history dictionary
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Total loss
    axes[0, 0].semilogy(history['total_loss'], label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (log scale)')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Individual losses
    axes[0, 1].semilogy(history['loss_ic'], label='IC Loss', alpha=0.7)
    axes[0, 1].semilogy(history['loss_bc'], label='BC Loss', alpha=0.7)
    axes[0, 1].semilogy(history['loss_pde'], label='PDE Loss', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss (log scale)')
    axes[0, 1].set_title('Individual Loss Components')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # L2 error
    if history['l2_error']:
        epochs, l2_errors = zip(*history['l2_error'])
        axes[1, 0].semilogy(epochs, l2_errors, 'o-', label='L2 Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('L2 Error (log scale)')
        axes[1, 0].set_title('L2 Error Convergence')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # Loss components comparison (final values)
    final_losses = {
        'IC': history['loss_ic'][-1],
        'BC': history['loss_bc'][-1],
        'PDE': history['loss_pde'][-1]
    }
    axes[1, 1].bar(final_losses.keys(), final_losses.values(), alpha=0.7)
    axes[1, 1].set_ylabel('Loss (log scale)')
    axes[1, 1].set_title('Final Loss Components')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
    
    plt.show()


def main():
    """Main training function."""
    print("="*60)
    print("Heat Equation PINN Training")
    print("="*60)
    
    # Parameters
    alpha = 0.1  # Thermal diffusivity
    layers = [2, 50, 50, 50, 1]  # Network architecture
    epochs = 10000
    lr = 0.001
    
    # Generate training data
    print("\nGenerating training data...")
    (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc), (x_pde, t_pde) = generate_training_data(
        N_ic=100, N_bc=100, N_pde=10000
    )
    print(f"  Initial condition points: {len(x_ic)}")
    print(f"  Boundary condition points: {len(x_bc)}")
    print(f"  PDE collocation points: {len(x_pde)}")
    
    # Generate test data
    print("\nGenerating test data...")
    x_test, t_test, u_exact = generate_test_data(N_x=100, N_t=100)
    print(f"  Test points: {len(x_test)}")
    
    # Initialize model
    print("\nInitializing PINN model...")
    model = HeatEquationPINN(alpha=alpha, layers=layers, activation='tanh')
    
    # Train model
    print("\nStarting training...")
    history = model.train(
        x_ic, t_ic, u_ic,
        x_bc, t_bc, u_bc,
        x_pde, t_pde,
        x_test, t_test, u_exact,
        epochs=epochs,
        lr=lr,
        print_every=1000
    )
    
    # Final L2 error
    final_l2 = model.compute_l2_error(x_test, t_test, u_exact)
    print(f"\nFinal L2 Error: {final_l2:.6e}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(model, x_test, t_test, u_exact, alpha=alpha, 
                     save_path='heat_equation_results.png')
    plot_training_history(history, save_path='training_history.png')
    
    # Save model
    torch.save(model.model.state_dict(), 'heat_equation_pinn_model.pth')
    print("\nModel saved to: heat_equation_pinn_model.pth")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  Final L2 Error: {final_l2:.6e}")
    print(f"  Final Total Loss: {history['total_loss'][-1]:.6e}")
    print(f"  Final IC Loss: {history['loss_ic'][-1]:.6e}")
    print(f"  Final BC Loss: {history['loss_bc'][-1]:.6e}")
    print(f"  Final PDE Loss: {history['loss_pde'][-1]:.6e}")


if __name__ == "__main__":
    main()

