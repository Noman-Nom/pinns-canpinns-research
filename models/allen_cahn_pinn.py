"""
Physics-Informed Neural Network for Allen-Cahn Equation

Allen-Cahn Equation: ∂u/∂t = ε * ∂²u/∂x² + u - u³

This module implements both standard PINN and CAN-PINN (with numerical differentiation
and adaptive loss weighting) for solving the Allen-Cahn equation.
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
from pinn_model import PINN

# Suppress cuBLAS warnings (harmless, context is automatically set)
warnings.filterwarnings('ignore', message='.*cuBLAS.*')
warnings.filterwarnings('ignore', message='.*CUDA context.*')

# Initialize CUDA context early to avoid warnings
if torch.cuda.is_available():
    _ = torch.zeros(1, device='cuda')


class AllenCahnPINN:
    """
    Standard Physics-Informed Neural Network solver for the Allen-Cahn Equation.
    Uses automatic differentiation for all derivatives (Raissi et al. 2019 approach).
    
    The Allen-Cahn equation: ∂u/∂t = ε * ∂²u/∂x² + u - u³
    """
    
    def __init__(self, epsilon=0.01, layers=[2, 50, 50, 50, 1], activation='tanh', device='cuda'):
        """
        Initialize the Allen-Cahn PINN solver.
        
        Args:
            epsilon: Diffusivity parameter (default: 0.01)
            layers: Network architecture (default: [2, 50, 50, 50, 1])
            activation: Activation function (default: 'tanh')
            device: Device to run on ('cuda' or 'cpu')
        """
        self.epsilon = epsilon
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize neural network
        self.model = PINN(layers, activation).to(self.device)
        
        print(f"Initialized Allen-Cahn PINN on device: {self.device}")
        print(f"Model architecture: {layers}")
        print(f"Diffusivity (ε): {epsilon}")
    
    def pde_residual(self, x, t):
        """
        Compute the PDE residual using automatic differentiation.
        PDE: ∂u/∂t - ε * ∂²u/∂x² - u + u³ = 0
        
        Args:
            x: Spatial coordinates (tensor)
            t: Temporal coordinates (tensor)
        
        Returns:
            Residual of the PDE and the solution u
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Get prediction
        u = self.model(x, t)
        
        # Compute derivatives using automatic differentiation
        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Allen-Cahn PDE residual: ∂u/∂t - ε * ∂²u/∂x² - u + u³
        residual = u_t - self.epsilon * u_xx - u + u**3
        
        return residual, u
    
    def loss_function(self, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde, 
                     weights=None):
        """
        Compute the total loss function.
        
        Loss = w_ic * Loss_IC + w_bc * Loss_BC + w_pde * Loss_PDE
        
        Args:
            x_ic, t_ic, u_ic: Initial condition points
            x_bc, t_bc, u_bc: Boundary condition points
            x_pde, t_pde: PDE collocation points
            weights: Dictionary with weights {'ic': w_ic, 'bc': w_bc, 'pde': w_pde}
                    If None, uses equal weights (1.0 each)
        
        Returns:
            Total loss and individual loss components
        """
        if weights is None:
            weights = {'ic': 1.0, 'bc': 1.0, 'pde': 1.0}
        
        # Loss from initial conditions
        u_pred_ic = self.model(x_ic, t_ic)
        loss_ic = torch.mean((u_pred_ic - u_ic)**2)
        
        # Loss from boundary conditions
        u_pred_bc = self.model(x_bc, t_bc)
        loss_bc = torch.mean((u_pred_bc - u_bc)**2)
        
        # Loss from PDE residual
        residual, _ = self.pde_residual(x_pde, t_pde)
        loss_pde = torch.mean(residual**2)
        
        # Weighted total loss
        total_loss = (weights['ic'] * loss_ic + 
                     weights['bc'] * loss_bc + 
                     weights['pde'] * loss_pde)
        
        return total_loss, loss_ic, loss_bc, loss_pde
    
    def compute_l2_error(self, x_test, t_test, u_exact):
        """
        Compute L2 error between predicted and exact solutions.
        
        Args:
            x_test: Test spatial coordinates
            t_test: Test temporal coordinates
            u_exact: Exact solution values
        
        Returns:
            L2 error
        """
        self.model.eval()
        with torch.no_grad():
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=self.device).reshape(-1, 1)
            t_test_tensor = torch.tensor(t_test, dtype=torch.float32, device=self.device).reshape(-1, 1)
            
            u_pred = self.model(x_test_tensor, t_test_tensor).cpu().numpy()
            u_exact = u_exact.reshape(-1, 1)
            
            # L2 error
            l2_error = np.sqrt(np.mean((u_pred - u_exact)**2))
            
        self.model.train()
        return l2_error
    
    def train(self, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde,
              x_test=None, t_test=None, u_exact=None,
              epochs=10000, lr=0.001, print_every=1000, 
              adaptive_weights=False):
        """
        Train the PINN.
        
        Args:
            x_ic, t_ic, u_ic: Initial condition data
            x_bc, t_bc, u_bc: Boundary condition data
            x_pde, t_pde: PDE collocation points
            x_test, t_test, u_exact: Test data for validation (optional)
            epochs: Number of training epochs
            lr: Learning rate
            print_every: Print loss every N epochs
            adaptive_weights: If True, use adaptive loss weighting (for CAN-PINN)
        
        Returns:
            Training history (losses and L2 errors)
        """
        # Convert to tensors and move to device
        x_ic = torch.tensor(x_ic, dtype=torch.float32, device=self.device).reshape(-1, 1)
        t_ic = torch.tensor(t_ic, dtype=torch.float32, device=self.device).reshape(-1, 1)
        u_ic = torch.tensor(u_ic, dtype=torch.float32, device=self.device).reshape(-1, 1)
        
        x_bc = torch.tensor(x_bc, dtype=torch.float32, device=self.device).reshape(-1, 1)
        t_bc = torch.tensor(t_bc, dtype=torch.float32, device=self.device).reshape(-1, 1)
        u_bc = torch.tensor(u_bc, dtype=torch.float32, device=self.device).reshape(-1, 1)
        
        x_pde = torch.tensor(x_pde, dtype=torch.float32, device=self.device).reshape(-1, 1)
        t_pde = torch.tensor(t_pde, dtype=torch.float32, device=self.device).reshape(-1, 1)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1000
        )
        
        # Initialize adaptive weights
        if adaptive_weights:
            weights = {'ic': torch.tensor(1.0, device=self.device),
                      'bc': torch.tensor(1.0, device=self.device),
                      'pde': torch.tensor(1.0, device=self.device)}
        else:
            weights = {'ic': 1.0, 'bc': 1.0, 'pde': 1.0}
        
        # Training history
        history = {
            'total_loss': [],
            'loss_ic': [],
            'loss_bc': [],
            'loss_pde': [],
            'l2_error': [],
            'weights': {'ic': [], 'bc': [], 'pde': []} if adaptive_weights else None
        }
        
        print("\n" + "="*60)
        print("Training PINN for Allen-Cahn Equation")
        print("="*60)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute loss
            total_loss, loss_ic, loss_bc, loss_pde = self.loss_function(
                x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde, weights
            )
            
            # Adaptive loss weighting
            if adaptive_weights and epoch % 100 == 0:
                # Compute gradients for each loss component
                grad_ic = torch.autograd.grad(
                    loss_ic, self.model.parameters(), retain_graph=True, create_graph=False
                )
                grad_bc = torch.autograd.grad(
                    loss_bc, self.model.parameters(), retain_graph=True, create_graph=False
                )
                grad_pde = torch.autograd.grad(
                    loss_pde, self.model.parameters(), retain_graph=True, create_graph=False
                )
                
                # Compute gradient magnitudes
                grad_norm_ic = sum([g.norm() for g in grad_ic])
                grad_norm_bc = sum([g.norm() for g in grad_bc])
                grad_norm_pde = sum([g.norm() for g in grad_pde])
                
                # Update weights inversely proportional to gradient magnitudes
                # Normalize to prevent weights from becoming too large
                total_norm = grad_norm_ic + grad_norm_bc + grad_norm_pde
                if total_norm > 0:
                    weights['ic'] = torch.clamp(total_norm / (grad_norm_ic + 1e-10), 0.1, 10.0)
                    weights['bc'] = torch.clamp(total_norm / (grad_norm_bc + 1e-10), 0.1, 10.0)
                    weights['pde'] = torch.clamp(total_norm / (grad_norm_pde + 1e-10), 0.1, 10.0)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.detach())
            
            # Store history
            history['total_loss'].append(total_loss.item())
            history['loss_ic'].append(loss_ic.item())
            history['loss_bc'].append(loss_bc.item())
            history['loss_pde'].append(loss_pde.item())
            
            if adaptive_weights:
                history['weights']['ic'].append(weights['ic'].item() if isinstance(weights['ic'], torch.Tensor) else weights['ic'])
                history['weights']['bc'].append(weights['bc'].item() if isinstance(weights['bc'], torch.Tensor) else weights['bc'])
                history['weights']['pde'].append(weights['pde'].item() if isinstance(weights['pde'], torch.Tensor) else weights['pde'])
            
            # Compute L2 error periodically (only if exact solution available)
            if (epoch % print_every == 0 or epoch == epochs - 1) and x_test is not None and u_exact is not None:
                l2_error = self.compute_l2_error(x_test, t_test, u_exact)
                history['l2_error'].append((epoch, l2_error))
                l2_str = f"L2 Error: {l2_error:.6e} | "
            else:
                l2_str = ""
            
            # Print progress
            if epoch % print_every == 0 or epoch == epochs - 1:
                weight_str = ""
                if adaptive_weights:
                    w_ic = weights['ic'].item() if isinstance(weights['ic'], torch.Tensor) else weights['ic']
                    w_bc = weights['bc'].item() if isinstance(weights['bc'], torch.Tensor) else weights['bc']
                    w_pde = weights['pde'].item() if isinstance(weights['pde'], torch.Tensor) else weights['pde']
                    weight_str = f" | Weights: IC={w_ic:.3f}, BC={w_bc:.3f}, PDE={w_pde:.3f}"
                
                print(f"Epoch {epoch:5d} | "
                      f"Total Loss: {total_loss.item():.6e} | "
                      f"IC Loss: {loss_ic.item():.6e} | "
                      f"BC Loss: {loss_bc.item():.6e} | "
                      f"PDE Loss: {loss_pde.item():.6e} | "
                      f"{l2_str}{weight_str}")
        
        print("="*60)
        print("Training completed!")
        print("="*60)
        
        return history


class CANAllenCahnPINN(AllenCahnPINN):
    """
    Conservative Allen-Cahn Neural Physics-Informed Neural Network (CAN-PINN).
    
    Enhancements over standard PINN:
    1. Numerical differentiation for spatial second derivatives (∂²u/∂x²)
    2. Adaptive loss weighting
    """
    
    def __init__(self, epsilon=0.01, layers=[2, 50, 50, 50, 1], activation='tanh', 
                 device='cuda', h=0.01):
        """
        Initialize the CAN-PINN solver.
        
        Args:
            epsilon: Diffusivity parameter
            layers: Network architecture
            activation: Activation function
            device: Device to run on
            h: Step size for numerical differentiation (default: 0.01)
        """
        super().__init__(epsilon, layers, activation, device)
        self.h = h  # Step size for central difference
        print(f"CAN-PINN: Using numerical differentiation with h={h}")
    
    def pde_residual(self, x, t):
        """
        Compute the PDE residual using numerical differentiation for spatial derivatives
        and automatic differentiation for time derivatives.
        
        Uses central difference for ∂²u/∂x²:
        ∂²u/∂x² ≈ (u(x+h,t) - 2*u(x,t) + u(x-h,t)) / h²
        
        Args:
            x: Spatial coordinates (tensor)
            t: Temporal coordinates (tensor)
        
        Returns:
            Residual of the PDE and the solution u
        """
        t.requires_grad_(True)
        
        # Get prediction at (x, t)
        u = self.model(x, t)
        
        # Compute time derivative using automatic differentiation
        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute spatial second derivative using numerical differentiation
        # Central difference: ∂²u/∂x² ≈ (u(x+h,t) - 2*u(x,t) + u(x-h,t)) / h²
        h = self.h
        
        # Evaluate at x+h and x-h
        # For boundary points, use one-sided differences instead of clamping
        x_plus_h = x + h
        x_minus_h = x - h
        
        # For points near boundaries, adjust h to stay within domain
        # This preserves gradient flow better than clamping
        mask_left = x < h
        mask_right = x > (1.0 - h)
        
        x_plus_h = torch.where(mask_right, x - h, x_plus_h)
        x_minus_h = torch.where(mask_left, x + h, x_minus_h)
        
        u_plus = self.model(x_plus_h, t)
        u_minus = self.model(x_minus_h, t)
        
        # Central difference approximation
        u_xx = (u_plus - 2.0 * u + u_minus) / (h**2)
        
        # Allen-Cahn PDE residual: ∂u/∂t - ε * ∂²u/∂x² - u + u³
        residual = u_t - self.epsilon * u_xx - u + u**3
        
        return residual, u
    
    def train(self, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde,
              x_test=None, t_test=None, u_exact=None,
              epochs=10000, lr=0.001, print_every=1000):
        """
        Train the CAN-PINN with adaptive loss weighting enabled.
        """
        return super().train(
            x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde,
            x_test, t_test, u_exact, epochs, lr, print_every,
            adaptive_weights=True
        )

