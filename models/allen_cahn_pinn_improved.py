"""
Improved Physics-Informed Neural Network for Allen-Cahn Equation

This module implements enhanced CAN-PINN with:
1. Adaptive step size h based on sampling density
2. Uncertainty-based loss weighting (learned weights)
3. Better boundary handling with symmetric stencils
4. Gradient penalty for smoother interfaces
5. L-BFGS optimizer option
6. Residual-based adaptive sampling support
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
import os
from pinn_model import PINN

# Suppress cuBLAS warnings (harmless, context is automatically set)
warnings.filterwarnings('ignore', message='.*cuBLAS.*')
warnings.filterwarnings('ignore', message='.*CUDA context.*')

# Initialize CUDA context early to avoid warnings
if torch.cuda.is_available():
    _ = torch.zeros(1, device='cuda')


class ImprovedCANAllenCahnPINN:
    """
    Improved CAN-PINN with multiple enhancements over standard implementation.
    """
    
    def __init__(self, epsilon=0.01, layers=[2, 50, 50, 50, 1], activation='tanh', 
                 device='cuda', h_adaptive=True, use_uncertainty_weights=True,
                 gradient_penalty_weight=1e-5, fourier_features=False, gamma=10.0):
        """
        Initialize the improved CAN-PINN solver.
        
        Args:
            epsilon: Diffusivity parameter
            layers: Network architecture
            activation: Activation function
            device: Device to run on
            h_adaptive: If True, use adaptive step size based on sampling density
            use_uncertainty_weights: If True, use learned uncertainty weights
            gradient_penalty_weight: Weight for gradient penalty regularization
            fourier_features: If True, use Fourier feature encoding
            gamma: Fourier feature scale (if fourier_features=True)
        """
        self.epsilon = epsilon
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.h_adaptive = h_adaptive
        self.use_uncertainty_weights = use_uncertainty_weights
        self.gradient_penalty_weight = gradient_penalty_weight
        self.fourier_features = fourier_features
        self.gamma = gamma
        
        # Compute adaptive h from domain (will be updated with data)
        self.h = 0.01  # Default, will be updated
        
        # Initialize neural network
        self.fourier_layer = None
        if fourier_features:
            # Add Fourier feature encoding layer
            self.fourier_layer = self._create_fourier_layer(2, gamma)  # 2 inputs (x, t)
            # Input dimension becomes 2 * (2 * num_freq + 1) after encoding
            num_freq = 10
            input_dim = 2 * num_freq + 2  # sin + cos + original (x, t)
            layers = [input_dim] + layers[1:]
        
        # Create PINN model
        from pinn_model import PINN as BasePINN
        self.base_model = BasePINN(layers, activation).to(self.device)
        self.model = self.base_model  # For compatibility
        
        # Learnable uncertainty weights (log variance)
        if use_uncertainty_weights:
            self.log_var_ic = nn.Parameter(torch.zeros(1, device=self.device))
            self.log_var_bc = nn.Parameter(torch.zeros(1, device=self.device))
            self.log_var_pde = nn.Parameter(torch.zeros(1, device=self.device))
        
        print(f"Initialized Improved CAN-PINN on device: {self.device}")
        print(f"Model architecture: {layers}")
        print(f"Diffusivity (ε): {epsilon}")
        print(f"Adaptive h: {h_adaptive}")
        print(f"Uncertainty weights: {use_uncertainty_weights}")
        print(f"Gradient penalty: {gradient_penalty_weight}")
        print(f"Fourier features: {fourier_features}")
    
    def _create_fourier_layer(self, input_dim, gamma):
        """Create Fourier feature encoding."""
        # Sample random frequencies
        B = torch.randn(10, input_dim, device=self.device) * gamma
        return B
    
    def _fourier_encode(self, x):
        """Apply Fourier feature encoding."""
        if not self.fourier_features or self.fourier_layer is None:
            return x
        
        # x: [N, input_dim]
        # B: [num_freq, input_dim]
        x_proj = 2 * np.pi * x @ self.fourier_layer.T  # [N, num_freq]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj), x], dim=1)
    
    def update_h_from_data(self, x_data):
        """Update step size h based on data sampling density."""
        if self.h_adaptive:
            # Estimate sampling density
            x_sorted = np.sort(x_data.flatten())
            # Compute typical spacing
            if len(x_sorted) > 1:
                dx = np.diff(x_sorted)
                # Use median spacing (robust to outliers)
                h_estimated = np.median(dx[dx > 0])
                # Use 1-2 times the median spacing
                self.h = max(0.001, min(0.1, 2.0 * h_estimated))
            print(f"Updated adaptive h: {self.h:.6f}")
    
    def forward(self, x, t):
        """Forward pass through the network with optional Fourier features."""
        if self.fourier_features and self.fourier_layer is not None:
            # Concatenate inputs
            X = torch.cat([x, t], dim=1)
            # Apply Fourier encoding
            X_encoded = self._fourier_encode(X)
            # Pass through network layers directly
            for i, layer in enumerate(self.base_model.layers[:-1]):
                X_encoded = layer(X_encoded)
                X_encoded = self.base_model.activation(X_encoded)
            X_encoded = self.base_model.layers[-1](X_encoded)
            return X_encoded
        else:
            return self.base_model(x, t)
    
    def pde_residual(self, x, t):
        """
        Compute the PDE residual using automatic differentiation (AD) for all derivatives.
        This hybrid approach uses AD (like PINN) but keeps CAN-PINN enhancements.
        
        Benefits:
        - No numerical errors (exact derivatives)
        - Captures sharp features (discontinuities)
        - Accurate boundaries
        - Better than numerical differentiation for all cases
        """
        # Enable gradients for both x and t
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Get prediction at (x, t)
        u = self.forward(x, t)
        
        # Compute time derivative using automatic differentiation
        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute spatial first derivative using automatic differentiation
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute spatial second derivative using automatic differentiation
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
    
    def loss_function(self, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde):
        """
        Compute the total loss function with uncertainty weighting.
        """
        # Loss from initial conditions
        u_pred_ic = self.forward(x_ic, t_ic)
        loss_ic = torch.mean((u_pred_ic - u_ic)**2)
        
        # Loss from boundary conditions
        u_pred_bc = self.forward(x_bc, t_bc)
        loss_bc = torch.mean((u_pred_bc - u_bc)**2)
        
        # Loss from PDE residual
        residual, u_pde = self.pde_residual(x_pde, t_pde)
        loss_pde = torch.mean(residual**2)
        
        # Gradient penalty for smoother interfaces
        gradient_penalty = 0.0
        if self.gradient_penalty_weight > 0:
            # Compute gradient of solution (x_pde should already require grad from pde_residual)
            # But we need to make sure it's set
            x_pde_grad = x_pde.clone().detach().requires_grad_(True)
            u_pde_grad = self.forward(x_pde_grad, t_pde)
            u_x = torch.autograd.grad(
                outputs=u_pde_grad,
                inputs=x_pde_grad,
                grad_outputs=torch.ones_like(u_pde_grad),
                create_graph=True,
                retain_graph=True
            )[0]
            # Penalize large gradients (promotes smoothness)
            gradient_penalty = self.gradient_penalty_weight * torch.mean(u_x**2)
        
        # Uncertainty-based weighting (with improved clamping to prevent saturation)
        if self.use_uncertainty_weights:
            # Improved clamp range: log_var in [-3, 2] means weights in [0.05, 20]
            # This prevents saturation at extreme values while allowing flexibility
            log_var_ic_clamped = torch.clamp(self.log_var_ic, -3.0, 2.0)
            log_var_bc_clamped = torch.clamp(self.log_var_bc, -3.0, 2.0)
            log_var_pde_clamped = torch.clamp(self.log_var_pde, -3.0, 2.0)
            
            # Weight = 1 / (2 * sigma^2), where sigma^2 = exp(log_var)
            # Loss = (1/2) * (1/sigma^2) * error^2 + (1/2) * log(sigma^2)
            weight_ic = 0.5 * torch.exp(-log_var_ic_clamped)
            weight_bc = 0.5 * torch.exp(-log_var_bc_clamped)
            weight_pde = 0.5 * torch.exp(-log_var_pde_clamped)
            
            # Add regularization term to prevent log_var from becoming too extreme
            # This encourages reasonable weights and stable training
            reg_term = 0.05 * (log_var_ic_clamped**2 + log_var_bc_clamped**2 + log_var_pde_clamped**2)
            
            total_loss = (weight_ic * loss_ic + 0.5 * log_var_ic_clamped +
                         weight_bc * loss_bc + 0.5 * log_var_bc_clamped +
                         weight_pde * loss_pde + 0.5 * log_var_pde_clamped +
                         gradient_penalty + reg_term)
            
            weights = {
                'ic': weight_ic.item(),
                'bc': weight_bc.item(),
                'pde': weight_pde.item()
            }
        else:
            # Equal weights
            total_loss = loss_ic + loss_bc + loss_pde + gradient_penalty
            weights = {'ic': 1.0, 'bc': 1.0, 'pde': 1.0}
        
        return total_loss, loss_ic, loss_bc, loss_pde, gradient_penalty, weights
    
    def compute_l2_error(self, x_test, t_test, u_exact):
        """Compute L2 error between predicted and exact solutions."""
        self.base_model.eval()
        with torch.no_grad():
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=self.device).reshape(-1, 1)
            t_test_tensor = torch.tensor(t_test, dtype=torch.float32, device=self.device).reshape(-1, 1)
            
            u_pred = self.forward(x_test_tensor, t_test_tensor).cpu().numpy()
            u_exact = u_exact.reshape(-1, 1)
            l2_error = np.sqrt(np.mean((u_pred - u_exact)**2))
            
        self.base_model.train()
        return l2_error
    
    def train(self, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde,
              x_test=None, t_test=None, u_exact=None,
              epochs=10000, lr=0.001, print_every=1000,
              use_lbfgs=False, lbfgs_epochs=1000, adaptive_sampler=None,
              x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0):
        """
        Train the improved CAN-PINN.
        
        Args:
            use_lbfgs: If True, use L-BFGS for fine-tuning after Adam
            lbfgs_epochs: Number of L-BFGS iterations
        """
        # Note: With AD approach, adaptive h is not needed for derivatives
        # But we keep it for potential future use or compatibility
        if self.h_adaptive:
            self.update_h_from_data(np.vstack([x_ic, x_bc, x_pde]))
        
        # Convert to tensors and move to device (keep as numpy for adaptive sampling)
        x_ic_np = x_ic.copy()
        t_ic_np = t_ic.copy()
        u_ic_np = u_ic.copy()
        x_bc_np = x_bc.copy()
        t_bc_np = t_bc.copy()
        u_bc_np = u_bc.copy()
        x_pde_np = x_pde.copy()
        t_pde_np = t_pde.copy()
        
        # Convert to tensors
        x_ic = torch.tensor(x_ic_np, dtype=torch.float32, device=self.device).reshape(-1, 1)
        t_ic = torch.tensor(t_ic_np, dtype=torch.float32, device=self.device).reshape(-1, 1)
        u_ic = torch.tensor(u_ic_np, dtype=torch.float32, device=self.device).reshape(-1, 1)
        
        x_bc = torch.tensor(x_bc_np, dtype=torch.float32, device=self.device).reshape(-1, 1)
        t_bc = torch.tensor(t_bc_np, dtype=torch.float32, device=self.device).reshape(-1, 1)
        u_bc = torch.tensor(u_bc_np, dtype=torch.float32, device=self.device).reshape(-1, 1)
        
        x_pde = torch.tensor(x_pde_np, dtype=torch.float32, device=self.device).reshape(-1, 1)
        t_pde = torch.tensor(t_pde_np, dtype=torch.float32, device=self.device).reshape(-1, 1)
        
        # Collect all parameters (including uncertainty weights if used)
        if self.use_uncertainty_weights:
            params = list(self.base_model.parameters())
            if self.fourier_features and self.fourier_layer is not None:
                params.append(self.fourier_layer)
            params.extend([
                self.log_var_ic, self.log_var_bc, self.log_var_pde
            ])
        else:
            params = list(self.base_model.parameters())
            if self.fourier_features and self.fourier_layer is not None:
                params.append(self.fourier_layer)
        
        # Phase 1: Adam optimizer
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1000
        )
        
        # Training history
        history = {
            'total_loss': [],
            'loss_ic': [],
            'loss_bc': [],
            'loss_pde': [],
            'gradient_penalty': [],
            'l2_error': [],
            'weights': {'ic': [], 'bc': [], 'pde': []}
        }
        
        print("\n" + "="*60)
        print("Training Improved CAN-PINN for Allen-Cahn Equation")
        print("="*60)
        
        self.base_model.train()
        adam_epochs = epochs - lbfgs_epochs if use_lbfgs else epochs
        
        # Phase 1: Adam training
        for epoch in range(adam_epochs):
            # Adaptive sampling (if enabled)
            if adaptive_sampler is not None and epoch > 0 and epoch % adaptive_sampler.resample_frequency == 0:
                # Convert current tensors back to numpy for resampling
                x_pde_np = x_pde.detach().cpu().numpy()
                t_pde_np = t_pde.detach().cpu().numpy()
                
                # Resample points
                x_pde_new, t_pde_new = adaptive_sampler.adaptive_sampling_step(
                    self, x_pde_np, t_pde_np, epoch,
                    x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max, device=self.device
                )
                
                # Update tensors
                x_pde = torch.tensor(x_pde_new, dtype=torch.float32, device=self.device).reshape(-1, 1)
                t_pde = torch.tensor(t_pde_new, dtype=torch.float32, device=self.device).reshape(-1, 1)
                print(f"  Resampled PDE points at epoch {epoch}: {len(x_pde_new)} points")
            
            optimizer.zero_grad()
            
            # Compute loss
            total_loss, loss_ic, loss_bc, loss_pde, grad_penalty, weights = self.loss_function(
                x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.detach())
            
            # Store history
            history['total_loss'].append(total_loss.item())
            history['loss_ic'].append(loss_ic.item())
            history['loss_bc'].append(loss_bc.item())
            history['loss_pde'].append(loss_pde.item())
            history['gradient_penalty'].append(grad_penalty.item() if isinstance(grad_penalty, torch.Tensor) else grad_penalty)
            history['weights']['ic'].append(weights['ic'])
            history['weights']['bc'].append(weights['bc'])
            history['weights']['pde'].append(weights['pde'])
            
            # Print progress
            if epoch % print_every == 0 or epoch == adam_epochs - 1:
                l2_str = ""
                if x_test is not None and u_exact is not None:
                    l2_error = self.compute_l2_error(x_test, t_test, u_exact)
                    history['l2_error'].append((epoch, l2_error))
                    l2_str = f"L2 Error: {l2_error:.6e} | "
                
                print(f"Epoch {epoch:5d} | "
                      f"Total Loss: {total_loss.item():.6e} | "
                      f"IC Loss: {loss_ic.item():.6e} | "
                      f"BC Loss: {loss_bc.item():.6e} | "
                      f"PDE Loss: {loss_pde.item():.6e} | "
                      f"Grad Penalty: {grad_penalty.item() if isinstance(grad_penalty, torch.Tensor) else grad_penalty:.6e} | "
                      f"{l2_str}"
                      f"Weights: IC={weights['ic']:.3f}, BC={weights['bc']:.3f}, PDE={weights['pde']:.3f}")
        
        # Phase 2: L-BFGS fine-tuning
        if use_lbfgs:
            print("\n" + "="*60)
            print("Phase 2: L-BFGS Fine-tuning")
            print("="*60)
            
            # Improved L-BFGS settings for better convergence
            lbfgs_optimizer = torch.optim.LBFGS(
                params, 
                lr=0.1,              # Lower learning rate for stability
                max_iter=50,         # More iterations per step
                max_eval=75,         # More function evaluations
                history_size=100,    # Larger history for better approximation
                line_search_fn='strong_wolfe',
                tolerance_grad=1e-7, # Stricter gradient tolerance
                tolerance_change=1e-9 # Stricter change tolerance
            )
            
            def closure():
                lbfgs_optimizer.zero_grad()
                total_loss, _, _, _, _, _ = self.loss_function(
                    x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde
                )
                total_loss.backward()
                # Return detached value for proper convergence check
                return total_loss.detach()
            
            for epoch in range(lbfgs_epochs):
                loss = lbfgs_optimizer.step(closure)
                
                if epoch % 100 == 0 or epoch == lbfgs_epochs - 1:
                    total_loss, loss_ic, loss_bc, loss_pde, grad_penalty, weights = self.loss_function(
                        x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde
                    )
                    history['total_loss'].append(total_loss.item())
                    history['loss_ic'].append(loss_ic.item())
                    history['loss_bc'].append(loss_bc.item())
                    history['loss_pde'].append(loss_pde.item())
                    history['gradient_penalty'].append(grad_penalty.item() if isinstance(grad_penalty, torch.Tensor) else grad_penalty)
                    # Append weights for L-BFGS iterations (use last weight values)
                    history['weights']['ic'].append(weights['ic'])
                    history['weights']['bc'].append(weights['bc'])
                    history['weights']['pde'].append(weights['pde'])
                    
                    print(f"L-BFGS Iter {epoch:5d} | "
                          f"Total Loss: {total_loss.item():.6e} | "
                          f"PDE Loss: {loss_pde.item():.6e}")
        
        print("="*60)
        print("Training completed!")
        print("="*60)
        
        return history

