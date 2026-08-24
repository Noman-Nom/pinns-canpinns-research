# Enhanced Adaptive PINNs for the Allen–Cahn Equation

A controlled, three-way comparison of Physics-Informed Neural Networks (PINNs) on the
Allen–Cahn phase-field equation, built to answer one question precisely: **how much
accuracy does it cost to compute a PINN's spatial derivatives by finite differences
instead of automatic differentiation, and can more training buy that accuracy back?**

```
∂u/∂t = ε · ∂²u/∂x² + u − u³,   x ∈ [0,1], t ∈ [0,1]
```

`u(x,t)` is the phase order parameter, and `ε` sets the width of the interface between
the two phases — small `ε` gives a thin, near-discontinuous interface that punishes any
method that can't represent steep gradients.

## Three methods, one fixed setup

Every method below shares the same `[2 → 50 → 50 → 50 → 1]` Tanh network, the same
20,000/1,000/200 interior/IC/BC collocation points, and the same 10,000-epoch Adam
schedule, so the only thing that varies is the method itself.

| Method | How `∂²u/∂x²` is computed | PDE loss | Time |
|---|---|---|---|
| **Baseline PINN** | Exact automatic differentiation | `2.08e-5` | 210 s |
| **Original CAN-PINN** | 3-point finite-difference stencil | `4.37e-3` (≈210× worse) | 167 s |
| **Enhanced Adaptive PINN** | Exact AD + 4 enhancements (uncertainty-weighted loss, residual-based adaptive sampling, gradient penalty, Adam→L-BFGS) | `1.19e-5` (≈43% better than baseline) | 572 s |

The Enhanced method is *not* a CAN-PINN — it keeps exact differentiation throughout, so
none of the finite-difference truncation error is present.

## The two findings

1. **The finite-difference error floor is structural, not a training artifact.** A
   Taylor-series argument predicts a truncation floor of `E_FD = (h²/12)·∂⁴u/∂x⁴ ≈ 4e-3`
   at `h = 1/256`, `ε = 0.01` — computed *before* any training. The measured floor after
   full training is `4.37e-3`. The two agree to within ~9%, and no amount of extra
   training closes the gap, because the error lives in the discretization, not in the
   network weights.

2. **Both AD-based methods still plateau at ~37% relative L² field error at ε=0.01**
   (falling to ~21% at the gentler ε=0.05), *regardless* of how well they satisfy the
   PDE residual. A smooth Tanh network cannot represent a near-discontinuous interface
   exactly — this is a representational ceiling, not a training failure, and it is the
   dominant limit on solution accuracy even though the PDE-loss comparison above makes
   automatic differentiation look 210× better.

See [CLAUDE.md](CLAUDE.md) for the full derivations, hyperparameters, and exact numbers
across all four test cases, and the accompanying thesis for the complete write-up.

## Repository structure

```
pinns-canpinns-research/
├── models/                       # Network + method implementations
│   ├── pinn_model.py                    Base PINN class (Tanh MLP, shared by all methods)
│   ├── allen_cahn_pinn.py               Baseline PINN (AD) + Original CAN-PINN (FD)
│   ├── allen_cahn_pinn_improved.py      Enhanced Adaptive PINN (all 4 enhancements)
│   └── residual_adaptive_sampling.py    Residual-based adaptive collocation sampler
│
├── analysis/                     # Reference solver + evaluation
│   ├── reference_solver.py              Crank–Nicolson reference solution (Δx=1/256)
│   ├── error_analysis.py                L², L∞, MAE vs. the reference
│   └── three_way_comparison.py          The baseline/CAN-PINN/Enhanced comparison
│
├── training/                     # Entry points
│   ├── run_full_pipeline.py             Reference → train → analyze → compare, in order
│   ├── run_single_test.py               Run one test case (--test_case, --epsilon, --ic_type)
│   ├── train_allen_cahn.py              Data generation + baseline training utilities
│   └── train_improved_allen_cahn.py     Trains all methods on all four test cases
│
├── visualization/                # Figure generation for the thesis/report
├── tests/                        # Sanity checks and GPU/environment verification
├── setup/                        # requirements.txt, environment.yml, install scripts
├── mcp_servers/                  # MCP server exposing symbolic/numerical math tools
├── report_figures/               # Generated result figures
├── outputs_archive/              # Generated results (git-ignored — see below)
├── error_analysis_results.json   # L²/L∞/MAE summary for all four test cases
└── CLAUDE.md                     # Full project reference: math, hyperparameters, results
```

## Getting started

```bash
# 1. Install dependencies (conda or pip — pick one)
conda env create -f setup/environment.yml && conda activate pinns
# or
pip install -r setup/requirements.txt

# 2. Verify the environment / GPU
python tests/verify_gpu.py
```

Run everything (reference solutions → training → error analysis → three-way comparison,
in order) from the project root:

```bash
python training/run_full_pipeline.py
```

Or run pieces individually:

```bash
python analysis/reference_solver.py                          # Crank–Nicolson reference solutions
python training/run_single_test.py --test_case 2 --epsilon 0.01 --ic_type sin
python analysis/error_analysis.py                             # L2/Linf vs. reference
python analysis/three_way_comparison.py 0.01 sin 2             # Baseline vs CAN-PINN vs Enhanced
```

All generated results (`.npz` solution files, reference solutions, comparison summaries)
are written to `outputs_archive/`, which is git-ignored — regenerate them by running the
pipeline rather than expecting them to be checked in.

## Test cases

| ID | Initial condition `u₀(x)` | `ε` | Interface |
|---|---|---|---|
| TC2-sin | `sin(πx)` | 0.01 | Sharp (main benchmark) |
| TC2-step | Step function | 0.01 | Very sharp |
| TC3-sin | `sin(πx)` | 0.01 | Sharp |
| TC3-sin | `sin(πx)` | 0.05 | Smooth (control case, isolates the ε effect) |

## Reference solution

Ground truth comes from an implicit Crank–Nicolson finite-difference solver
(`Δx = 1/256`, `Δt = 10⁻⁴`, accuracy `O(10⁻⁷)`), solved with the Thomas algorithm at each
time step — several orders of magnitude more accurate than any PINN considered here.

## Citation

```bibtex
@mastersthesis{shahid2026enhanced,
  title  = {Adaptive Physics-Informed Neural Networks with Uncertainty-Weighted Loss
            Functions for the Allen-Cahn Phase-Field Equation},
  author = {Shahid, Hassan},
  school = {National University of Science and Technology "MISIS"},
  year   = {2026}
}
```
