"""
Residual-Based Adaptive Sampling (RBAS) for PINNs.

This module implements adaptive sampling strategies that focus on regions
where the PDE residual is high, improving training efficiency.
"""

import numpy as np
import torch


class ResidualAdaptiveSampler:
    """
    Adaptive sampler that resamples collocation points based on PDE residual.
    """
    
    def __init__(self, initial_N=10000, resample_frequency=1000, 
                 resample_fraction=0.2, keep_best_fraction=0.8):
        """
        Initialize the adaptive sampler.
        
        Args:
            initial_N: Initial number of collocation points
            resample_frequency: Resample every N epochs
            resample_fraction: Fraction of points to resample
            keep_best_fraction: Fraction of points to keep (highest residual)
        """
        self.initial_N = initial_N
        self.resample_frequency = resample_frequency
        self.resample_fraction = resample_fraction
        self.keep_best_fraction = keep_best_fraction
        
    def compute_residuals(self, model, x_pde, t_pde, device='cuda'):
        """
        Compute PDE residuals at collocation points.
        
        Args:
            model: PINN model
            x_pde: Spatial coordinates
            t_pde: Temporal coordinates
            device: Device to run on
        
        Returns:
            Residuals at each point
        """
        # Set model to eval mode but keep gradients for residual computation
        model.base_model.eval()
        
        # Create tensors with requires_grad
        x_tensor = torch.tensor(x_pde, dtype=torch.float32, device=device, requires_grad=True).reshape(-1, 1)
        t_tensor = torch.tensor(t_pde, dtype=torch.float32, device=device, requires_grad=True).reshape(-1, 1)
        
        # Compute residuals (need gradients for this)
        residual, _ = model.pde_residual(x_tensor, t_tensor)
        
        # Detach and convert to numpy
        residuals = torch.abs(residual).detach().cpu().numpy().flatten()
        
        # Set model back to train mode
        model.base_model.train()
        return residuals
    
    def resample_points(self, x_pde, t_pde, residuals, x_min=0.0, x_max=1.0,
                       t_min=0.0, t_max=1.0, boundary_oversample=True):
        """
        Resample collocation points based on residuals.
        
        Args:
            x_pde: Current spatial coordinates
            t_pde: Current temporal coordinates
            residuals: PDE residuals at current points
            x_min, x_max: Spatial domain
            t_min, t_max: Temporal domain
            boundary_oversample: If True, oversample near boundaries
        
        Returns:
            New x_pde, t_pde arrays
        """
        N = len(x_pde)
        N_resample = int(N * self.resample_fraction)
        N_keep = N - N_resample
        
        # Keep points with highest residuals
        keep_indices = np.argsort(residuals)[-N_keep:]
        x_keep = x_pde[keep_indices]
        t_keep = t_pde[keep_indices]
        
        # Resample new points
        # Option 1: Sample from high-residual regions (importance sampling)
        # Use residuals as probability weights
        residual_weights = residuals / (residuals.sum() + 1e-10)
        resample_indices = np.random.choice(
            N, size=N_resample, p=residual_weights, replace=True
        )
        
        # Add small random perturbations around high-residual points
        x_resample = x_pde[resample_indices] + np.random.normal(0, 0.01, N_resample).reshape(-1, 1)
        t_resample = t_pde[resample_indices] + np.random.normal(0, 0.01, N_resample).reshape(-1, 1)
        
        # Clamp to domain
        x_resample = np.clip(x_resample, x_min, x_max)
        t_resample = np.clip(t_resample, t_min, t_max)
        
        # Option 2: Oversample near boundaries (for sharp interfaces)
        if boundary_oversample:
            # Add some points near boundaries
            N_boundary = int(N_resample * 0.3)
            x_boundary = np.concatenate([
                np.random.uniform(x_min, x_min + 0.1, (N_boundary // 2, 1)),
                np.random.uniform(x_max - 0.1, x_max, (N_boundary // 2, 1))
            ])
            t_boundary = np.random.uniform(t_min, t_max, (N_boundary, 1))
            
            # Replace some resampled points with boundary points
            x_resample[:N_boundary] = x_boundary
            t_resample[:N_boundary] = t_boundary
        
        # Combine kept and resampled points
        x_new = np.vstack([x_keep, x_resample])
        t_new = np.vstack([t_keep, t_resample])
        
        return x_new, t_new
    
    def adaptive_sampling_step(self, model, x_pde, t_pde, epoch, 
                              x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0,
                              device='cuda'):
        """
        Perform one step of adaptive sampling.
        
        Args:
            model: PINN model
            x_pde: Current spatial coordinates
            t_pde: Current temporal coordinates
            epoch: Current training epoch
            x_min, x_max: Spatial domain
            t_min, t_max: Temporal domain
            device: Device to run on
        
        Returns:
            New x_pde, t_pde if resampling occurred, else original
        """
        if epoch % self.resample_frequency == 0 and epoch > 0:
            # Compute residuals
            residuals = self.compute_residuals(model, x_pde, t_pde, device)
            
            # Resample points
            x_new, t_new = self.resample_points(
                x_pde, t_pde, residuals, x_min, x_max, t_min, t_max
            )
            
            print(f"  Resampled {len(x_pde)} points at epoch {epoch}")
            print(f"  Max residual: {residuals.max():.6e}, Mean residual: {residuals.mean():.6e}")
            
            return x_new, t_new
        
        return x_pde, t_pde

