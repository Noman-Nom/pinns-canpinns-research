"""
Physics-Informed Neural Network solver for the Wave Equation.

Wave Equation: ∂²u/∂t² = c² * ∂²u/∂x²

where:
    u(x, t): displacement of the wave
    c: wave speed
"""

import torch
import torch.nn as nn
import numpy as np
from pinn_model import PINN


class WaveEquationPINN:
    """
    Physics-Informed Neural Network solver for the Wave Equation.
    
    The Wave equation: ∂²u/∂t² = c² * ∂²u/∂x²
    """
    
    def __init__(self, c=1.0, layers=[2, 50, 50, 50, 1], activation='tanh', device='cuda'):
        """
        Initialize the Wave Equation PINN solver.
        
        Args:
            c: Wave speed (default: 1.0)
            layers: Network architecture (default: [2, 50, 50, 50, 1])
            activation: Activation function (default: 'tanh')
            device: Device to run on ('cuda' or 'cpu')
        """
        self.c = c
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize neural network
        self.model = PINN(layers, activation).to(self.device)
        
        print(f"Initialized Wave Equation PINN on device: {self.device}")
        print(f"Model architecture: {layers}")
        print(f"Wave speed (c): {c}")
    
    def analytical_solution(self, x, t):
        """
        Analytical solution for the Wave equation with initial conditions:
        u(x,0) = sin(πx)
        u_t(x,0) = 0
        and boundary conditions: u(0,t) = u(1,t) = 0
        
        Solution: u(x,t) = sin(πx) * cos(πct)
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
        
        Returns:
            Analytical solution
        """
        return np.sin(np.pi * x) * np.cos(np.pi * self.c * t)
    
    def compute_derivatives(self, u, x, t):
        """
        Compute derivatives using automatic differentiation.
        
        Args:
            u: Network output u(x, t)
            x: Spatial coordinates
            t: Temporal coordinates
        
        Returns:
            u_t: Time derivative ∂u/∂t
            u_tt: Second time derivative ∂²u/∂t²
            u_x: Spatial derivative ∂u/∂x
            u_xx: Second spatial derivative ∂²u/∂x²
        """
        # First derivatives
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
        
        # Second derivatives
        u_tt = torch.autograd.grad(
            outputs=u_t,
            inputs=t,
            grad_outputs=torch.ones_like(u_t),
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
        
        return u_t, u_tt, u_x, u_xx
    
    def pde_residual(self, x, t):
        """
        Compute the PDE residual: ∂²u/∂t² - c² * ∂²u/∂x²
        
        Args:
            x: Spatial coordinates (tensor)
            t: Temporal coordinates (tensor)
        
        Returns:
            Residual of the PDE
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Get prediction
        u = self.model(x, t)
        
        # Compute derivatives
        _, u_tt, _, u_xx = self.compute_derivatives(u, x, t)
        
        # PDE residual: ∂²u/∂t² - c² * ∂²u/∂x²
        residual = u_tt - self.c**2 * u_xx
        
        return residual, u
    
    def loss_function(self, x_ic, t_ic, u_ic, x_ic_t, t_ic_t, u_ic_t, 
                      x_bc, t_bc, u_bc, x_pde, t_pde):
        """
        Compute the total loss function.
        
        Loss = Loss_IC + Loss_IC_t + Loss_BC + Loss_PDE
        
        where:
            Loss_IC: Loss from initial condition u(x,0)
            Loss_IC_t: Loss from initial condition u_t(x,0)
            Loss_BC: Loss from boundary conditions
            Loss_PDE: Loss from PDE residual
        
        Args:
            x_ic, t_ic, u_ic: Initial condition points (u(x,0))
            x_ic_t, t_ic_t, u_ic_t: Initial condition points (u_t(x,0))
            x_bc, t_bc, u_bc: Boundary condition points
            x_pde, t_pde: PDE collocation points
        
        Returns:
            Total loss and individual loss components
        """
        # Loss from initial condition u(x,0)
        u_pred_ic = self.model(x_ic, t_ic)
        loss_ic = torch.mean((u_pred_ic - u_ic)**2)
        
        # Loss from initial condition u_t(x,0)
        u_pred_ic_t = self.model(x_ic_t, t_ic_t)
        u_t_pred, _, _, _ = self.compute_derivatives(u_pred_ic_t, x_ic_t, t_ic_t)
        loss_ic_t = torch.mean((u_t_pred - u_ic_t)**2)
        
        # Loss from boundary conditions
        u_pred_bc = self.model(x_bc, t_bc)
        loss_bc = torch.mean((u_pred_bc - u_bc)**2)
        
        # Loss from PDE residual
        residual, _ = self.pde_residual(x_pde, t_pde)
        loss_pde = torch.mean(residual**2)
        
        # Total loss
        total_loss = loss_ic + loss_ic_t + loss_bc + loss_pde
        
        return total_loss, loss_ic, loss_ic_t, loss_bc, loss_pde
    
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
    
    def train(self, x_ic, t_ic, u_ic, x_ic_t, t_ic_t, u_ic_t,
              x_bc, t_bc, u_bc, x_pde, t_pde,
              x_test, t_test, u_exact, epochs=10000, lr=0.001, 
              print_every=1000):
        """
        Train the PINN.
        
        Args:
            x_ic, t_ic, u_ic: Initial condition data (u(x,0))
            x_ic_t, t_ic_t, u_ic_t: Initial condition data (u_t(x,0))
            x_bc, t_bc, u_bc: Boundary condition data
            x_pde, t_pde: PDE collocation points
            x_test, t_test, u_exact: Test data for validation
            epochs: Number of training epochs
            lr: Learning rate
            print_every: Print loss every N epochs
        
        Returns:
            Training history (losses and L2 errors)
        """
        # Convert to tensors and move to device
        x_ic = torch.tensor(x_ic, dtype=torch.float32, device=self.device).reshape(-1, 1)
        t_ic = torch.tensor(t_ic, dtype=torch.float32, device=self.device).reshape(-1, 1)
        u_ic = torch.tensor(u_ic, dtype=torch.float32, device=self.device).reshape(-1, 1)
        
        x_ic_t = torch.tensor(x_ic_t, dtype=torch.float32, device=self.device).reshape(-1, 1)
        t_ic_t = torch.tensor(t_ic_t, dtype=torch.float32, device=self.device).reshape(-1, 1)
        u_ic_t = torch.tensor(u_ic_t, dtype=torch.float32, device=self.device).reshape(-1, 1)
        
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
        
        # Training history
        history = {
            'total_loss': [],
            'loss_ic': [],
            'loss_ic_t': [],
            'loss_bc': [],
            'loss_pde': [],
            'l2_error': []
        }
        
        print("\n" + "="*60)
        print("Training PINN for Wave Equation")
        print("="*60)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute loss
            total_loss, loss_ic, loss_ic_t, loss_bc, loss_pde = self.loss_function(
                x_ic, t_ic, u_ic, x_ic_t, t_ic_t, u_ic_t,
                x_bc, t_bc, u_bc, x_pde, t_pde
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)
            
            # Store history
            history['total_loss'].append(total_loss.item())
            history['loss_ic'].append(loss_ic.item())
            history['loss_ic_t'].append(loss_ic_t.item())
            history['loss_bc'].append(loss_bc.item())
            history['loss_pde'].append(loss_pde.item())
            
            # Compute L2 error periodically
            if epoch % print_every == 0 or epoch == epochs - 1:
                l2_error = self.compute_l2_error(x_test, t_test, u_exact)
                history['l2_error'].append((epoch, l2_error))
                
                print(f"Epoch {epoch:5d} | "
                      f"Total Loss: {total_loss.item():.6e} | "
                      f"IC Loss: {loss_ic.item():.6e} | "
                      f"IC_t Loss: {loss_ic_t.item():.6e} | "
                      f"BC Loss: {loss_bc.item():.6e} | "
                      f"PDE Loss: {loss_pde.item():.6e} | "
                      f"L2 Error: {l2_error:.6e}")
        
        print("="*60)
        print("Training completed!")
        print("="*60)
        
        return history

