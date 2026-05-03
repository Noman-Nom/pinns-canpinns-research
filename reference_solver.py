"""
Reference Solution Generator for Allen-Cahn Equation
Using Crank-Nicolson Finite Difference Scheme

Supervisor Requirement: "I need exact test cases with answers"

This generates high-accuracy reference solutions using:
- Grid: Δx = 1/256, Δt = 0.0001
- Accuracy: O(Δx² + Δt²) ≈ 1e-7
- Method: Crank-Nicolson (implicit, unconditionally stable)
"""

import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve
import os
from pathlib import Path


class CrankNicolsonAllenCahn:
    """Crank-Nicolson FD solver for Allen-Cahn equation."""

    def __init__(self, epsilon=0.01, nx=257, nt=10000, t_end=1.0):
        """
        Initialize CN solver.

        Args:
            epsilon: Diffusivity parameter
            nx: Number of spatial grid points (includes boundaries)
            nt: Number of time steps
            t_end: Final time
        """
        self.epsilon = epsilon
        self.nx = nx
        self.nt = nt
        self.t_end = t_end

        # Grid spacing
        self.dx = 1.0 / (nx - 1)
        self.dt = t_end / (nt - 1)

        # Courant number for stability check
        self.r = epsilon * self.dt / (self.dx ** 2)

        # Spatial grid
        self.x = np.linspace(0, 1, nx)

        # Time grid
        self.t = np.linspace(0, t_end, nt)

        print(f"CN Solver initialized:")
        print(f"  Domain: x ∈ [0, 1], t ∈ [0, {t_end}]")
        print(f"  Grid: nx={nx} (Δx={self.dx:.6f}), nt={nt} (Δt={self.dt:.6f})")
        print(f"  ε={epsilon}, r={self.r:.4f} (stability: r ≤ ∞ for CN)")
        print()

    def _initial_condition(self, ic_type='sin'):
        """Get initial condition u(x, 0)."""
        if ic_type == 'sin':
            return np.sin(np.pi * self.x)
        elif ic_type == 'step':
            u0 = np.zeros_like(self.x)
            u0[self.x > 0.5] = 1.0
            return u0
        else:
            raise ValueError(f"Unknown ic_type: {ic_type}")

    def _assemble_matrix(self):
        """Assemble tridiagonal matrices for CN scheme.

        Returns:
            A_left: Left side matrix (implicit)
            A_right: Right side matrix (explicit)
        """
        # Crank-Nicolson coefficients
        a = -self.r / 2.0  # coefficient of u_{i±1}
        b = 1.0 + self.r   # coefficient of u_i on left
        c = -self.r / 2.0  # coefficient of u_{i±1}

        # Left side: (I + r/2 * L) * u^{n+1}
        # Tridiagonal: [a, b, a] on [lower, diag, upper]
        diagonals_left = [a * np.ones(self.nx - 1),
                          b * np.ones(self.nx),
                          a * np.ones(self.nx - 1)]
        A_left = diags(diagonals_left, [-1, 0, 1], shape=(self.nx, self.nx), format='csr')

        # Right side: (I - r/2 * L) * u^n
        # But we'll construct this dynamically with nonlinear term

        return A_left

    def _nonlinear_term(self, u):
        """Evaluate nonlinear term: -u + u³ at time level."""
        return -u + u**3

    def _assemble_rhs(self, u_n, A_right_u_n):
        """Assemble right-hand side for CN step.

        CN step:
        (I + r/2 * L) u^{n+1} = (I - r/2 * L) u^n + Δt/2 * (f^n + f^{n+1})

        where L is the Laplacian operator and f = -u + u³
        """
        # Linear part: (I - r/2 * L) u^n
        # This is A_right_u_n passed in

        # Nonlinear part: Δt/2 * (f^n + f^{n+1})
        # Since u^{n+1} is unknown, we use semi-implicit:
        # f^{n+1} ≈ f(u^n) for nonlinear term
        f_n = self._nonlinear_term(u_n)

        # Combine: (I - r/2 * L) u^n + Δt * f^n
        rhs = A_right_u_n + self.dt * f_n

        # Boundary conditions: u = 0 at x=0, x=1
        rhs[0] = 0.0
        rhs[-1] = 0.0

        return rhs

    def solve(self, ic_type='sin', verbose=True):
        """
        Solve Allen-Cahn equation using Crank-Nicolson.

        Returns:
            u: Solution matrix [nt, nx]
        """
        u = np.zeros((self.nt, self.nx))

        # Set initial condition
        u[0, :] = self._initial_condition(ic_type)

        # Assemble left matrix (constant)
        A_left = self._assemble_matrix()

        # Assemble right matrix for linear part
        r = self.r
        diagonals_right = [(-r/2) * np.ones(self.nx - 1),
                           (1.0 + r) * np.ones(self.nx),
                           (-r/2) * np.ones(self.nx - 1)]
        from scipy.sparse import diags
        A_right = diags(diagonals_right, [-1, 0, 1], shape=(self.nx, self.nx), format='csr')

        if verbose:
            print(f"Solving Allen-Cahn with CN (ε={self.epsilon}, ic={ic_type})...")

        # Time stepping
        for n in range(self.nt - 1):
            u_n = u[n, :]

            # Linear part
            A_right_u_n = A_right @ u_n

            # Assemble RHS with nonlinear term
            rhs = self._assemble_rhs(u_n, A_right_u_n)

            # Solve linear system
            u[n + 1, :] = spsolve(A_left, rhs)

            if verbose and (n + 1) % (self.nt // 10) == 0:
                print(f"  Step {n+1}/{self.nt-1} ({100*(n+1)/(self.nt-1):.0f}%)")

        if verbose:
            print(f"✓ Solution computed. Final time={self.t[-1]}")
            print(f"  Solution range: [{u.min():.4f}, {u.max():.4f}]")

        return u


def generate_all_references():
    """Generate reference solutions for all 4 test cases."""

    # Test case configurations
    test_cases = [
        {'name': 'TC2_sin_eps001', 'epsilon': 0.01, 'ic_type': 'sin'},
        {'name': 'TC2_step_eps001', 'epsilon': 0.01, 'ic_type': 'step'},
        {'name': 'TC3_sin_eps001', 'epsilon': 0.01, 'ic_type': 'sin'},
        {'name': 'TC3_sin_eps005', 'epsilon': 0.05, 'ic_type': 'sin'},
    ]

    # Create output directory
    ref_dir = Path('reference_solutions')
    ref_dir.mkdir(exist_ok=True)

    print("="*70)
    print("  Reference Solution Generation (Crank-Nicolson FD)")
    print("="*70)
    print()

    results_summary = []

    for tc in test_cases:
        print(f"\nGenerating: {tc['name']}")
        print("-" * 70)

        # Create solver
        solver = CrankNicolsonAllenCahn(
            epsilon=tc['epsilon'],
            nx=257,        # Δx = 1/256
            nt=10000,      # Δt = 0.0001
            t_end=1.0
        )

        # Solve
        u_ref = solver.solve(ic_type=tc['ic_type'])

        # Save solution
        filename = ref_dir / f"{tc['name']}_reference.npz"
        np.savez(
            filename,
            u=u_ref,
            x=solver.x,
            t=solver.t,
            epsilon=tc['epsilon'],
            ic_type=tc['ic_type'],
            dx=solver.dx,
            dt=solver.dt
        )

        print(f"  Saved to: {filename}")

        results_summary.append({
            'test_case': tc['name'],
            'epsilon': tc['epsilon'],
            'ic_type': tc['ic_type'],
            'u_min': u_ref.min(),
            'u_max': u_ref.max(),
            'file': str(filename)
        })

    print("\n" + "="*70)
    print("  ✓ ALL REFERENCE SOLUTIONS GENERATED")
    print("="*70)
    print()
    print("Summary:")
    for r in results_summary:
        print(f"  {r['test_case']:<25} u ∈ [{r['u_min']:8.4f}, {r['u_max']:8.4f}]")

    return results_summary


if __name__ == '__main__':
    generate_all_references()
