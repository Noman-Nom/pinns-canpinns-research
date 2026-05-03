"""
MCP Math Server for PINN / CAN-PINN Research
Provides symbolic and numerical math tools via Model Context Protocol.

Install dependencies:
    pip install mcp sympy numpy scipy

Run:
    python mcp_math_server.py
"""

import asyncio
import json
import numpy as np
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("math-tools")


# ─── Tool definitions ────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="symbolic_pde_check",
            description=(
                "Check whether a candidate function u(x,t) satisfies the Allen-Cahn PDE "
                "symbolically using SymPy. Returns the PDE residual in symbolic form."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "u_expr": {
                        "type": "string",
                        "description": "Expression for u(x,t) as a Python/SymPy string, e.g. 'sin(pi*x)*exp(-t)'"
                    },
                    "epsilon": {
                        "type": "number",
                        "description": "Diffusivity parameter ε in the Allen-Cahn equation"
                    }
                },
                "required": ["u_expr", "epsilon"]
            }
        ),
        Tool(
            name="symbolic_differentiate",
            description="Compute symbolic derivatives of a mathematical expression using SymPy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expr": {
                        "type": "string",
                        "description": "Mathematical expression as string, e.g. 'sin(pi*x)*exp(-epsilon*pi**2*t)'"
                    },
                    "variable": {
                        "type": "string",
                        "description": "Variable to differentiate with respect to: 'x', 't', etc."
                    },
                    "order": {
                        "type": "integer",
                        "description": "Order of differentiation (1 or 2)",
                        "default": 1
                    }
                },
                "required": ["expr", "variable"]
            }
        ),
        Tool(
            name="allen_cahn_reference_solution",
            description=(
                "Compute a high-accuracy numerical reference solution for the Allen-Cahn equation "
                "using a Crank-Nicolson finite difference scheme. Returns solution values on a grid."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "epsilon": {
                        "type": "number",
                        "description": "Diffusivity ε (e.g. 0.01 or 0.05)"
                    },
                    "ic_type": {
                        "type": "string",
                        "description": "'sin' for sin(πx) or 'step' for step function at x=0.5",
                        "enum": ["sin", "step"]
                    },
                    "nx": {
                        "type": "integer",
                        "description": "Number of spatial grid points (default 256)",
                        "default": 256
                    },
                    "nt": {
                        "type": "integer",
                        "description": "Number of time steps (default 10000)",
                        "default": 10000
                    },
                    "t_end": {
                        "type": "number",
                        "description": "End time (default 1.0)",
                        "default": 1.0
                    }
                },
                "required": ["epsilon", "ic_type"]
            }
        ),
        Tool(
            name="compute_error_metrics",
            description=(
                "Compute L2 relative error and L-infinity error between a PINN solution "
                "and a reference solution. Pass both as flat lists of numbers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "u_pinn": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "PINN solution values (flattened)"
                    },
                    "u_reference": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Reference solution values (flattened, same shape)"
                    }
                },
                "required": ["u_pinn", "u_reference"]
            }
        ),
        Tool(
            name="fourier_mode_analysis",
            description=(
                "Decompose a 1D solution profile into Fourier modes. "
                "Returns the dominant modes and their amplitudes. Useful for understanding "
                "how PINN captures the spectral content of the Allen-Cahn solution."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "u_values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Solution values on a uniform spatial grid"
                    },
                    "x_min": {"type": "number", "default": 0.0},
                    "x_max": {"type": "number", "default": 1.0},
                    "n_modes": {
                        "type": "integer",
                        "description": "Number of dominant modes to report",
                        "default": 5
                    }
                },
                "required": ["u_values"]
            }
        ),
        Tool(
            name="cn_stability_check",
            description=(
                "Check the stability of the Crank-Nicolson scheme for Allen-Cahn. "
                "Returns the CFL number and whether the scheme is stable for given Δx, Δt, ε."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "epsilon": {"type": "number"},
                    "dx": {"type": "number", "description": "Spatial step Δx"},
                    "dt": {"type": "number", "description": "Time step Δt"}
                },
                "required": ["epsilon", "dx", "dt"]
            }
        ),
        Tool(
            name="heat_equation_exact_solution",
            description=(
                "Compute the analytical solution to the heat equation ∂u/∂t = α ∂²u/∂x² "
                "with IC u(x,0)=sin(πx) and BC u(0,t)=u(1,t)=0. "
                "Solution: u(x,t) = sin(πx) * exp(-α π² t). "
                "Useful as a sanity-check case since an exact answer exists."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "alpha": {"type": "number", "description": "Thermal diffusivity α"},
                    "x_values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Spatial coordinates"
                    },
                    "t_values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Time coordinates (same length as x_values)"
                    }
                },
                "required": ["alpha", "x_values", "t_values"]
            }
        ),
        Tool(
            name="energy_functional_analysis",
            description=(
                "Compute the Ginzburg-Landau energy functional E[u] = ∫(ε/2 |∇u|² + F(u)) dx "
                "where F(u) = (1/4)(u²-1)². The Allen-Cahn equation is the L² gradient flow "
                "of this energy, so E should decrease monotonically during correct evolution."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "u_snapshots": {
                        "type": "array",
                        "description": "List of solution arrays at different times (each is a 1D list)",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"}
                        }
                    },
                    "epsilon": {"type": "number"},
                    "dx": {"type": "number", "description": "Spatial step for integration"}
                },
                "required": ["u_snapshots", "epsilon", "dx"]
            }
        ),
        Tool(
            name="sympy_solve_equation",
            description="Solve algebraic or simple ODE equations symbolically using SymPy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "equation": {
                        "type": "string",
                        "description": "Equation as string, e.g. 'x**2 - 4' or 'Eq(f(x).diff(x), f(x))'"
                    },
                    "solve_for": {
                        "type": "string",
                        "description": "Variable or function to solve for, e.g. 'x' or 'f(x)'"
                    }
                },
                "required": ["equation", "solve_for"]
            }
        ),
    ]


# ─── Tool implementations ─────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        if name == "symbolic_pde_check":
            return await _symbolic_pde_check(**arguments)
        elif name == "symbolic_differentiate":
            return await _symbolic_differentiate(**arguments)
        elif name == "allen_cahn_reference_solution":
            return await _allen_cahn_reference(**arguments)
        elif name == "compute_error_metrics":
            return await _compute_error_metrics(**arguments)
        elif name == "fourier_mode_analysis":
            return await _fourier_mode_analysis(**arguments)
        elif name == "cn_stability_check":
            return await _cn_stability_check(**arguments)
        elif name == "heat_equation_exact_solution":
            return await _heat_exact(**arguments)
        elif name == "energy_functional_analysis":
            return await _energy_functional(**arguments)
        elif name == "sympy_solve_equation":
            return await _sympy_solve(**arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error in {name}: {type(e).__name__}: {e}")]


async def _symbolic_pde_check(u_expr: str, epsilon: float) -> list[TextContent]:
    import sympy as sp

    x, t, eps = sp.symbols("x t epsilon", real=True)
    pi = sp.pi

    local_ns = {"x": x, "t": t, "epsilon": eps, "pi": pi, "sin": sp.sin,
                "cos": sp.cos, "exp": sp.exp, "tanh": sp.tanh, "sqrt": sp.sqrt}
    u = sp.sympify(u_expr, locals=local_ns)

    u_t = sp.diff(u, t)
    u_x = sp.diff(u, x)
    u_xx = sp.diff(u_x, x)

    residual = u_t - eps * u_xx - u + u**3
    residual_subst = residual.subs(eps, epsilon)
    residual_simplified = sp.simplify(residual_subst)

    result = (
        f"Allen-Cahn PDE check for u = {u_expr}, ε = {epsilon}\n"
        f"{'='*60}\n"
        f"∂u/∂t       = {sp.simplify(u_t)}\n"
        f"ε·∂²u/∂x²  = {sp.simplify(epsilon * u_xx)}\n"
        f"u - u³      = {sp.simplify(u - u**3)}\n\n"
        f"Residual = ∂u/∂t - ε·∂²u/∂x² - u + u³\n"
        f"         = {residual_simplified}\n\n"
    )
    if residual_simplified == 0:
        result += "✅ This function EXACTLY satisfies the Allen-Cahn equation."
    else:
        result += "⚠️ This function does NOT satisfy the Allen-Cahn equation (residual ≠ 0)."

    return [TextContent(type="text", text=result)]


async def _symbolic_differentiate(expr: str, variable: str, order: int = 1) -> list[TextContent]:
    import sympy as sp

    x, t, epsilon = sp.symbols("x t epsilon", real=True)
    local_ns = {"x": x, "t": t, "epsilon": epsilon, "pi": sp.pi,
                "sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "tanh": sp.tanh}
    sym_expr = sp.sympify(expr, locals=local_ns)
    var_sym = local_ns.get(variable, sp.Symbol(variable))

    result_sym = sp.diff(sym_expr, var_sym, order)
    simplified = sp.simplify(result_sym)

    text = (
        f"d^{order}/d{variable}^{order} [{expr}]\n"
        f"= {simplified}\n"
        f"= {sp.latex(simplified)}  (LaTeX)"
    )
    return [TextContent(type="text", text=text)]


async def _allen_cahn_reference(
    epsilon: float, ic_type: str, nx: int = 256, nt: int = 10000, t_end: float = 1.0
) -> list[TextContent]:
    """
    Crank-Nicolson semi-implicit scheme for Allen-Cahn:
      (u^{n+1} - u^n)/dt = ε * (u_xx^{n+1} + u_xx^n)/2 + u^n - (u^n)^3
    Linear part (diffusion) treated implicitly, nonlinear part explicitly.
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    dx = 1.0 / (nx - 1)
    dt = t_end / nt
    x = np.linspace(0, 1, nx)

    # Initial condition
    if ic_type == "sin":
        u = np.sin(np.pi * x)
    elif ic_type == "step":
        u = np.where(x > 0.5, 1.0, 0.0).astype(float)
    else:
        return [TextContent(type="text", text=f"Unknown ic_type: {ic_type}")]

    # Enforce BCs
    u[0] = 0.0
    u[-1] = 0.0

    # Build tridiagonal matrix for implicit diffusion
    r = epsilon * dt / (2.0 * dx**2)
    diag_main = (1 + 2 * r) * np.ones(nx)
    diag_off = -r * np.ones(nx - 1)
    A = diags([diag_off, diag_main, diag_off], [-1, 0, 1], format="csr")
    # Fix BCs in matrix
    A = A.tolil()
    A[0, :] = 0; A[0, 0] = 1
    A[-1, :] = 0; A[-1, -1] = 1
    A = A.tocsr()

    # Also build explicit diffusion operator
    B = diags([r * np.ones(nx - 1), (1 - 2 * r) * np.ones(nx), r * np.ones(nx - 1)],
              [-1, 0, 1], format="csr").tolil()
    B[0, :] = 0; B[0, 0] = 1
    B[-1, :] = 0; B[-1, -1] = 1
    B = B.tocsr()

    # Store snapshots at t = 0, 0.25, 0.5, 0.75, 1.0
    snapshots = {}
    save_times = {0: "t=0.00", nt // 4: "t=0.25", nt // 2: "t=0.50",
                  3 * nt // 4: "t=0.75", nt: "t=1.00"}

    for step in range(nt + 1):
        if step in save_times:
            snapshots[save_times[step]] = u.copy()
        if step == nt:
            break
        # RHS: explicit diffusion + explicit nonlinear reaction
        rhs = B @ u + dt * (u - u**3)
        rhs[0] = 0.0
        rhs[-1] = 0.0
        u = spsolve(A, rhs)
        u[0] = 0.0
        u[-1] = 0.0

    # Summary stats
    lines = [
        f"Allen-Cahn Reference Solution (Crank-Nicolson)",
        f"{'='*55}",
        f"ε = {epsilon}, IC = {ic_type}",
        f"Grid: Δx = {dx:.6f} ({nx} points), Δt = {dt:.6f} ({nt} steps)",
        f"Scheme accuracy: O(Δx²) + O(Δt²) ≈ {max(dx**2, dt**2):.2e}",
        "",
        "Solution statistics at each saved time:",
    ]
    for label, snap in snapshots.items():
        lines.append(
            f"  {label}: min={snap.min():.6f}, max={snap.max():.6f}, "
            f"||u||_2={np.linalg.norm(snap):.6f}"
        )

    lines += [
        "",
        "Energy (Ginzburg-Landau) at saved times:",
    ]
    for label, snap in snapshots.items():
        grad = np.gradient(snap, dx)
        energy = np.trapz(0.5 * epsilon * grad**2 + 0.25 * (snap**2 - 1)**2, dx=dx)
        lines.append(f"  {label}: E = {energy:.8f}")

    lines += [
        "",
        "✅ Reference solution computed. Use compute_error_metrics to compare with PINN output.",
        f"   Final solution at t=1.0: {list(np.round(snapshots['t=1.00'][::nx//8], 6))}"
    ]

    return [TextContent(type="text", text="\n".join(lines))]


async def _compute_error_metrics(u_pinn: list, u_reference: list) -> list[TextContent]:
    u_p = np.array(u_pinn, dtype=float)
    u_r = np.array(u_reference, dtype=float)

    if u_p.shape != u_r.shape:
        return [TextContent(type="text", text=f"Shape mismatch: {u_p.shape} vs {u_r.shape}")]

    diff = u_p - u_r
    l2_abs = np.sqrt(np.mean(diff**2))
    l2_rel = l2_abs / (np.sqrt(np.mean(u_r**2)) + 1e-14)
    linf_abs = np.max(np.abs(diff))
    linf_rel = linf_abs / (np.max(np.abs(u_r)) + 1e-14)
    mean_abs = np.mean(np.abs(diff))

    text = (
        f"Error Metrics (PINN vs Reference)\n"
        f"{'='*40}\n"
        f"N points        : {len(u_p)}\n"
        f"L2 absolute     : {l2_abs:.6e}\n"
        f"L2 relative     : {l2_rel:.6e}  ({l2_rel*100:.4f}%)\n"
        f"L∞ absolute     : {linf_abs:.6e}\n"
        f"L∞ relative     : {linf_rel:.6e}  ({linf_rel*100:.4f}%)\n"
        f"Mean abs error  : {mean_abs:.6e}\n\n"
        f"Interpretation:\n"
        f"  L2 relative < 1%  → Excellent\n"
        f"  L2 relative < 5%  → Good\n"
        f"  L2 relative > 10% → Needs improvement\n"
        f"  Current: {'Excellent ✅' if l2_rel < 0.01 else 'Good ✅' if l2_rel < 0.05 else 'Needs work ⚠️'}"
    )
    return [TextContent(type="text", text=text)]


async def _fourier_mode_analysis(
    u_values: list, x_min: float = 0.0, x_max: float = 1.0, n_modes: int = 5
) -> list[TextContent]:
    u = np.array(u_values, dtype=float)
    n = len(u)
    dx = (x_max - x_min) / (n - 1)

    # Sine transform (appropriate for Dirichlet BCs)
    coeffs = np.zeros(n // 2)
    for k in range(1, n // 2 + 1):
        x = np.linspace(x_min, x_max, n)
        coeffs[k - 1] = (2.0 / (x_max - x_min)) * np.trapz(u * np.sin(k * np.pi * (x - x_min) / (x_max - x_min)), dx=dx)

    top_idx = np.argsort(np.abs(coeffs))[::-1][:n_modes]

    lines = [
        "Fourier Mode Analysis (Sine Series)",
        "="*45,
        f"Domain: [{x_min}, {x_max}], {n} points",
        f"u(x) ≈ Σ a_k · sin(kπx/(x_max-x_min))",
        "",
        f"Top {n_modes} dominant modes:",
        f"{'Mode k':>8} {'Coefficient a_k':>18} {'|a_k|/||a||':>14}",
        "-"*42,
    ]
    norm = np.sqrt(np.sum(coeffs**2)) + 1e-14
    for idx in top_idx:
        k = idx + 1
        lines.append(f"  k={k:4d}   {coeffs[idx]:>16.8f}   {abs(coeffs[idx])/norm:>12.6f}")

    captured = np.sum(coeffs[top_idx]**2) / (np.sum(coeffs**2) + 1e-14)
    lines.append(f"\nEnergy captured by top {n_modes} modes: {captured*100:.2f}%")
    return [TextContent(type="text", text="\n".join(lines))]


async def _cn_stability_check(epsilon: float, dx: float, dt: float) -> list[TextContent]:
    r = epsilon * dt / dx**2
    cfl = dt / dx

    # For Crank-Nicolson: unconditionally stable for diffusion part
    # But nonlinear reaction u - u³ adds explicit instability if dt too large
    # Stability condition for explicit reaction: dt < 2 (from linearization around u=0)
    reaction_stable = dt < 2.0
    diffusion_note = "Unconditionally stable (Crank-Nicolson diffusion)"

    text = (
        f"Crank-Nicolson Stability Analysis\n"
        f"{'='*40}\n"
        f"ε = {epsilon}, Δx = {dx}, Δt = {dt}\n\n"
        f"Diffusion number r = ε·Δt/Δx² = {r:.6f}\n"
        f"CFL number          = Δt/Δx   = {cfl:.6f}\n\n"
        f"Diffusion (C-N):  {diffusion_note}\n"
        f"Reaction (explicit): dt = {dt:.4f} {'< 2 ✅ stable' if reaction_stable else '≥ 2 ⚠️ may be unstable'}\n\n"
        f"Recommended Δt: < {min(0.1, dx**2 / (4 * epsilon)):.6f} for safety\n"
        f"Truncation error: O(Δx²) + O(Δt²) = O({dx**2:.2e}) + O({dt**2:.2e})"
    )
    return [TextContent(type="text", text=text)]


async def _heat_exact(alpha: float, x_values: list, t_values: list) -> list[TextContent]:
    x = np.array(x_values, dtype=float)
    t = np.array(t_values, dtype=float)

    u_exact = np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)

    lines = [
        f"Heat Equation Exact Solution",
        f"{'='*40}",
        f"∂u/∂t = α ∂²u/∂x²,  α = {alpha}",
        f"IC: u(x,0) = sin(πx)",
        f"BC: u(0,t) = u(1,t) = 0",
        f"",
        f"Analytical solution: u(x,t) = sin(πx) · exp(-α·π²·t)",
        f"Decay rate: α·π² = {alpha * np.pi**2:.6f}",
        f"",
        f"Values at provided (x, t) pairs:",
    ]
    for xi, ti, ui in zip(x[:10], t[:10], u_exact[:10]):
        lines.append(f"  u({xi:.4f}, {ti:.4f}) = {ui:.8f}")
    if len(x) > 10:
        lines.append(f"  ... ({len(x)} total points)")

    lines += [
        f"",
        f"Range: [{u_exact.min():.6f}, {u_exact.max():.6f}]",
        f"||u||_2 = {np.linalg.norm(u_exact):.6f}",
    ]
    return [TextContent(type="text", text="\n".join(lines))]


async def _energy_functional(
    u_snapshots: list, epsilon: float, dx: float
) -> list[TextContent]:
    lines = [
        "Ginzburg-Landau Energy Analysis",
        "="*45,
        f"E[u] = ∫ [ (ε/2)|∇u|² + (1/4)(u²-1)² ] dx",
        f"ε = {epsilon}, Δx = {dx}",
        "",
        "Allen-Cahn is gradient flow of this energy → E must decrease.",
        "",
        f"{'Snapshot':>10} {'Energy E[u]':>16} {'ΔE':>14} {'Decreasing?':>13}",
        "-"*55,
    ]

    energies = []
    for snap in u_snapshots:
        u = np.array(snap, dtype=float)
        grad = np.gradient(u, dx)
        e = np.trapz(0.5 * epsilon * grad**2 + 0.25 * (u**2 - 1)**2, dx=dx)
        energies.append(e)

    prev = None
    for i, e in enumerate(energies):
        delta = e - prev if prev is not None else float("nan")
        ok = "✅" if (prev is None or e <= prev + 1e-12) else "❌"
        lines.append(f"  t_{i:3d}     {e:>16.10f} {delta:>14.6e}  {ok}")
        prev = e

    total_drop = energies[0] - energies[-1] if len(energies) > 1 else 0
    lines += [
        "",
        f"Total energy dissipation: {total_drop:.8f}",
        f"Energy decreasing overall: {'✅ Yes' if total_drop >= 0 else '❌ No — check solution'}",
    ]
    return [TextContent(type="text", text="\n".join(lines))]


async def _sympy_solve(equation: str, solve_for: str) -> list[TextContent]:
    import sympy as sp

    local_ns = {
        "x": sp.Symbol("x"), "t": sp.Symbol("t"),
        "f": sp.Function("f"), "u": sp.Function("u"),
        "Eq": sp.Eq, "Derivative": sp.Derivative,
        "sin": sp.sin, "cos": sp.cos, "exp": sp.exp,
        "pi": sp.pi, "sqrt": sp.sqrt, "symbols": sp.symbols,
        "dsolve": sp.dsolve,
    }

    parsed = eval(equation, {"__builtins__": {}}, local_ns)
    target = eval(solve_for, {"__builtins__": {}}, local_ns)

    try:
        solution = sp.solve(parsed, target)
        sol_str = "\n".join(f"  {s}" for s in solution) if solution else "  No solution found"
    except Exception:
        try:
            solution = sp.dsolve(parsed, target)
            sol_str = f"  {solution}"
        except Exception as e2:
            sol_str = f"  Could not solve: {e2}"

    text = (
        f"SymPy Solver\n"
        f"{'='*40}\n"
        f"Equation : {equation}\n"
        f"Solve for: {solve_for}\n\n"
        f"Solutions:\n{sol_str}"
    )
    return [TextContent(type="text", text=text)]


# ─── Entry point ──────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
