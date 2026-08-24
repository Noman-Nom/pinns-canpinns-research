# CLAUDE.md — Allen-Cahn PINNs Research Project
# Complete Project Reference: Code + Thesis + Supervisor Context

---

## 1. WHO IS THIS PROJECT

| Field | Value |
|---|---|
| **Author** | Hassan Shahid |
| **Student ID** | 2416443 |
| **Email** | m2416443@edu.misis.ru |
| **Group** | MIVT-24-6A (Second year, Master's) |
| **Specialty** | 09.04.01 Computer Science and Engineering |
| **Funding** | Full-time, Budgetary |
| **Supervisor** | Assoc. Prof. L. A. Artemyeva, Institute of Computer Science (IC) |
| **University** | National University of Science and Technology "MISIS", Moscow |
| **Thesis location** | `D:\thesis\` |
| **Code location** | `e:\khokhar-tappa\pinns-canpinns-research\` |

---

## 2. WHAT THE PROJECT IS ABOUT

**One sentence:** A controlled three-way comparison of Physics-Informed Neural Networks (PINNs) for the Allen–Cahn phase-field equation, measuring the accuracy cost of computing spatial derivatives by finite differences instead of automatic differentiation.

**The governing equation:**
```
∂u/∂t = ε · ∂²u/∂x² + u − u³,   x ∈ [0,1], t ∈ [0,T]
u(x,0) = u₀(x)
u(0,t) = u(1,t) = 0
```

- `u(x,t)` = order parameter (+1 and −1 are the two stable phases)
- `ε` = interface width parameter (small ε → sharp, near-discontinuous interface)
- `u − u³` = reaction term from double-well potential F(u) = ¼(1−u²)²
- Energy functional: E[u] = ∫[(ε/2)(∂u/∂x)² + F(u)]dx
- Allen–Cahn is the L² gradient flow: ∂u/∂t = −δE/δu

**The research gap:** CAN-PINN replaces exact automatic differentiation with a finite-difference stencil for speed. Nobody had quantified what that costs on a stiff, sharp-interface problem — or whether training can overcome the deficit.

---

## 3. THE THREE METHODS COMPARED

### Method 1 — Baseline PINN (Automatic Differentiation)
- Computes `∂²u/∂x²` exactly via PyTorch autograd (chain rule through computation graph)
- Accuracy: floating-point round-off ~10⁻¹⁶. No approximation, no error floor.
- Loss: L_PDE + L_IC + L_BC (plain, equal weights)
- Optimizer: Adam only, 10,000 epochs
- **PDE loss: 2.08 × 10⁻⁵ | Training time: 210 s**

### Method 2 — Original CAN-PINN (Finite Differences)
- Replaces `∂²u/∂x²` with the 3-point finite-difference stencil (h = 1/256):
  ```
  ∂²u/∂x² ≈ [u(x+h) − 2u(x) + u(x−h)] / h²
  ```
- Truncation error (from Taylor series): `E_FD = (h²/12) · ∂⁴u/∂x⁴`
- At ε=0.01: E_FD ≈ 4×10⁻³ (predicted before training — confirmed by experiment)
- **Permanent error floor** — training cannot remove it (weights θ do not appear in E_FD)
- Cross-term danger: L_FD = ⟨f²⟩ − 2⟨f·E_FD⟩ + ⟨E_FD²⟩  →  optimiser is actively pulled toward wrong solution near interface
- **PDE loss: 4.37 × 10⁻³ | Training time: 167 s (fastest)**

### Method 3 — Enhanced Adaptive PINN (AD + 4 Enhancements)
- Uses **exact AD** (NOT finite differences — this is NOT a CAN-PINN)
- **CRITICAL NAMING:** Always say "Enhanced Adaptive PINN". NEVER say "Hybrid CAN-PINN" (professor corrected this — without FD it cannot be called a CAN-PINN)

  **Enhancement 1: Uncertainty-Weighted Loss** (Kendall et al., 2018)
  ```
  L_UW = Σᵢ [ (1/2σᵢ²)·Lᵢ + ½·log σᵢ² ]   for i ∈ {pde, ic, bc}
  wᵢ = ½·exp(−log σᵢ²)
  ```
  Learns loss weights automatically. When term i is hard, wᵢ increases automatically.

  **Enhancement 2: Residual-Based Adaptive Sampling**
  Every 3,000 epochs: replace the 10% lowest-residual collocation points with fresh uniform samples → effort migrates toward the interface.

  **Enhancement 3: Gradient Penalty**
  ```
  L_GP = λ · (1/N) · Σ (∂u_θ/∂x)²,   λ = 10⁻⁵
  ```
  Suppresses spurious oscillations near the interface. λ is tiny enough not to flatten real gradients.

  **Enhancement 4: Two-Stage Optimizer (Adam → L-BFGS)**
  - Adam: 10,000 epochs (robust global progress, first-order)
  - L-BFGS: 1,000 iterations with strong-Wolfe line search (curvature-aware local refinement)

  **Full objective:**
  ```
  L_total(θ, σ) = L_UW(θ, σ) + L_GP(θ)
  {θ*, σ*} = argmin L_total   over θ ∈ ℝ⁵³⁰¹, σ ∈ ℝ³
  ```
- **PDE loss: 1.19 × 10⁻⁵ | Training time: 572 s (most expensive)**

---

## 4. NETWORK ARCHITECTURE (Shared by All Three Methods)

```
Input (x, t)  →  [2 → 50 → 50 → 50 → 1]  →  Output u_θ(x,t)
Activation:   tanh   (C∞, necessary for ∂²u/∂x² via AD)
Init:         Xavier uniform weights, zero biases
Parameters:   5,301 total
```

**Why tanh and not ReLU:** The PDE residual requires ∂²u/∂x². ReLU has zero second derivative almost everywhere — the diffusion term ε·∂²u/∂x² would vanish from the residual. tanh ∈ C∞ has well-defined derivatives of all orders.

---

## 5. COLLOCATION / TRAINING SETUP

| Parameter | Value |
|---|---|
| Interior collocation points N_PDE | 20,000 |
| Initial condition points N_IC | 1,000 |
| Boundary points N_BC | 200 |
| Adam learning rate | 1×10⁻³ |
| Adam epochs | 10,000 |
| L-BFGS iterations (Enhanced only) | 1,000 |
| Adaptive resampling interval | every 3,000 epochs |
| Resampled fraction | 10% lowest-residual |
| Gradient penalty weight λ | 10⁻⁵ |
| Domain | x ∈ [0,1], t ∈ [0,1] |
| Device | CUDA GPU (CPU fallback) |

---

## 6. REFERENCE SOLUTION (Ground Truth)

**Method:** Implicit Crank–Nicolson finite-difference scheme:
```
(u_j^{n+1} − u_j^n)/Δt = (ε/2)(δ²ₓu^{n+1} + δ²ₓu^n) + u_j^n − (u_j^n)³
```
Solved by Thomas algorithm (tridiagonal) at each time step.

| Parameter | Value |
|---|---|
| Grid spacing Δx | 1/256 ≈ 0.0039 |
| Time step Δt | 10⁻⁴ |
| Accuracy | O(Δx², Δt²) ≈ O(10⁻⁷) |
| Stability | Unconditionally stable (implicit diffusion) |
| Interface resolution at ε=0.01 | ~25 grid points across interface |

Reference files: `outputs_archive/reference_solutions/TC*_reference.npz`

---

## 7. TEST CASES

| ID | Initial condition u₀(x) | ε | Interface type |
|---|---|---|---|
| TC2-sin | sin(πx) | 0.01 | Sharp (MAIN benchmark for all comparisons) |
| TC2-step | Step function | 0.01 | Very sharp (near-discontinuity, L∞=0.925) |
| TC3-sin | sin(πx) | 0.01 | Sharp |
| TC3-sin | sin(πx) | 0.05 | Smooth (CONTROL case — isolates ε effect) |

**Why TC3-sin ε=0.05 matters:** Softening the interface while holding everything else fixed proves the L² error plateau is architectural. If it were a training failure it would not track ε so cleanly.

---

## 8. KEY RESULTS (Always Use These Exact Numbers)

### Three-Way PDE Loss Comparison (TC2-sin, ε=0.01)

| Method | L_PDE | Time (s) | vs. Baseline |
|---|---|---|---|
| Baseline PINN (AD) | 2.08 × 10⁻⁵ | 210 | — |
| Original CAN-PINN (FD) | 4.37 × 10⁻³ | 167 | **≈ 210× worse** |
| Enhanced Adaptive PINN | 1.19 × 10⁻⁵ | 572 | **≈ 43% better** |

### FD Floor: Predicted vs. Measured
```
Predicted by Taylor series (BEFORE training):  E_FD ≈ 4 × 10⁻³
Measured after full training:                  L_PDE = 4.37 × 10⁻³
Agreement: ~9%  →  deficit is mathematical, not undertraining
```

### Field Accuracy vs. Crank–Nicolson Reference

| Test case | ε | Baseline L²_rel | CAN-PINN L²_rel | L∞ (Baseline) | MAE |
|---|---|---|---|---|---|
| TC2-sin  | 0.01 | 37.3% | 37.4% | 0.550 | 0.174 |
| TC2-step | 0.01 | 35.1% | 34.7% | 0.925 | 0.139 |
| TC3-sin  | 0.01 | 37.2% | 37.4% | 0.543 | 0.172 |
| TC3-sin  | 0.05 | 20.7% | 20.7% | 0.310 | 0.101 |

### The Key Paradox (Must Understand This)
The baseline satisfies the equation **210× better** than CAN-PINN in L_PDE,
yet both reproduce the true field to within **0.1%** of each other in L²_rel.
→ The dominant limit on field accuracy is NOT differentiation — it is the smooth tanh network's inability to represent a sharp interface.

---

## 9. THE TWO KEY FINDINGS

### Finding 1: The Finite-Difference Error Floor (Mathematical)

Derivation (Taylor series, 2nd-order central difference):
```
u(x+h) = u + h·u_x + (h²/2)·u_xx + (h³/6)·u_xxx + (h⁴/24)·u_xxxx + O(h⁵)
u(x−h) = u − h·u_x + (h²/2)·u_xx − (h³/6)·u_xxx + (h⁴/24)·u_xxxx + O(h⁵)

Add:  [u(x+h) − 2u(x) + u(x−h)]/h² = u_xx + (h²/12)·u_xxxx + O(h⁴)

Therefore: E_FD = (h²/12) · ∂⁴u/∂x⁴
```

Key properties:
- E_FD does not depend on θ → training cannot remove it
- At ε=0.01: ∂⁴u/∂x⁴ ~ ε⁻² = 10⁴ → floor ~ 4×10⁻³
- Floor scales as ε⁻⁴ (gets drastically worse as interface sharpens)
- The cross-term in L_FD actively misleads the optimiser

### Finding 2: The Architectural (Representational) Ceiling

Mathematical statement:
```
u_θ ∈ C∞(ℝ²)  ⟹  inf_θ ||u_θ − u_true||_L2 ≥ δ(ε) > 0
where δ(ε) → 0 as ε → ∞ (smoother target → lower floor)
```

Evidence it is architectural (not a training failure):
1. Plateau at ~37% for ALL ε=0.01 cases regardless of differentiation method
2. Plateau drops to ~21% when ε relaxes from 0.01 to 0.05 — tracks ε cleanly
3. Error concentrates spatially in a thin band at the interface
4. Connection to spectral bias: networks learn low frequencies first; sharp interfaces need high frequencies

Fix (future work): Fourier feature embeddings (Tancik et al., 2020)

---

## 10. ERROR METRICS (Definitions)

```
L²_rel  = ||u_θ − u_ref||₂ / ||u_ref||₂         (relative field error, %)
L∞      = max|u_θ(xᵢ,tⱼ) − u_ref(xᵢ,tⱼ)|       (worst-case pointwise error)
MAE     = mean|u_θ − u_ref|                       (average pointwise error)
L_PDE   = (1/N) Σ f(xₖ,tₖ)²                     (mean-squared PDE residual)
  where f = ∂u_θ/∂t − ε·∂²u_θ/∂x² − u_θ + u_θ³
```

**Critical distinction:** L_PDE asks "is the equation obeyed?"; L²_rel asks "is the field correct?" — they can disagree dramatically, as this project proves. Always report both.

---

## 11. WHAT WE MINIMISE vs. MAXIMISE

| Quantity | Direction | Why |
|---|---|---|
| L_PDE, L_IC, L_BC | Minimise | Physics/data mismatch must go to zero |
| L_GP | Minimise | Suppress spurious oscillations |
| Gaussian log-likelihood | Maximise | Mathematical derivation of L_UW |
| Learned weight wᵢ on hard term | Auto-increase | Focus effort where physics worst satisfied |

**If asked why total loss can be negative:** The ½·log σᵢ² terms can be large negative when σᵢ² < 1. This is correct likelihood behaviour. Always report L_PDE, never the total loss.

---

## 12. PROJECT FILE STRUCTURE

```
e:\khokhar-tappa\pinns-canpinns-research\
│
├── CLAUDE.md                          ← THIS FILE
├── README.md
├── error_analysis_results.json        ← L2/Linf/MAE for all 4 test cases
│
├── models/
│   ├── pinn_model.py                  ← Base PINN class (shared by all methods)
│   ├── allen_cahn_pinn_improved.py    ← Enhanced Adaptive PINN (all 4 enhancements)
│   └── residual_adaptive_sampling.py ← Enhancement 2 implementation
│
├── analysis/
│   ├── error_analysis.py              ← Computes L2/Linf vs CN reference
│   └── three_way_comparison.py        ← Generates comparison_summary.json
│
├── visualization/
│   ├── generate_report_figures.py     ← All thesis figures (run from project root)
│   ├── canpinn_paper_comparison.py
│   ├── loss_function_detailed_breakdown.py
│   └── optimizer_documentation.py
│
├── outputs_archive/
│   ├── three_way_results/
│   │   ├── comparison_summary.json    ← THE KEY RESULTS FILE
│   │   ├── Baseline_PINN_predictions.npz
│   │   ├── Original_CAN-PINN_predictions.npz
│   │   └── Hybrid_CAN-PINN_predictions.npz  (= Enhanced Adaptive PINN, old name)
│   ├── reference_solutions/           ← Crank-Nicolson .npz files
│   └── TC2/TC3 solution .npz files
│
├── tests/
├── training/
├── setup/
├── mcp_servers/
├── venv/                              ← Python virtual environment
└── data/
```

```
D:\thesis\
├── main.tex / main.pdf                ← 77-page thesis, 0 errors
├── settings.tex                       ← natbib, hyperref, etc.
├── references.bib                     ← 20 references (all resolve)
├── chapters/
│   ├── 01_introduction.tex
│   ├── 02_literature_review.tex
│   ├── 03_methodology.tex
│   ├── 04_results.tex
│   ├── 05_discussion.tex
│   └── 06_conclusion.tex
├── preliminary/
│   ├── title_page.tex                 ← Hassan Shahid, 2416443, MIVT-24-6A
│   ├── declaration.tex
│   └── certificate.tex
├── figures/                           ← All 300 DPI (fig_ch<N>_<desc>.png)
├── meeting_guide_v2.tex / .pdf        ← BEST meeting guide (16 pages, full math)
├── meeting_guide.tex / .pdf           ← Old version (9 pages)
└── upgrade_original_figures.py        ← Regenerates all figures at 300 DPI
```

---

## 13. THESIS STRUCTURE

| Chapter | Key content | Status |
|---|---|---|
| 1 Introduction | Allen-Cahn, energy functional, objectives, novel contributions | Done |
| 2 Literature Review | PINNs, CAN-PINN naming, spectral bias, enhancements lit | Done |
| 3 Methodology | Full math: CN reference, network, losses, Taylor proof, algorithm | Done |
| 4 Results | Three-way comparison, FD floor validation, L² tables, solution plots | Done |
| 5 Discussion | Floor inequality expansion, ceiling proof, guidance table, limitations | Done |
| 6 Conclusion | Findings, contributions, 5-subsection future work | Done |
| Appendix B | Data table + environment table | **Pending (placeholders)** |
| Abstract | Proper abstract | **Pending (placeholder)** |
| Acknowledgements | Text | **Pending (placeholder)** |

**Current page count: 77. Target: 85–90.**

**Compile commands:**
```
cd D:\thesis
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

**LaTeX important notes:**
- Package `natbib` with `\citet{}` / `\citep{}` must be loaded (in settings.tex)
- University logo is optional — wrapped in `\IfFileExists`
- Standard `report` class (not KOMA)
- **Use pdflatex directly, NOT latexmk** (latexmk requires Perl, not installed)
- VS Code PDF viewer shows cached PDFs — close and reopen tab after recompile

---

## 14. FIGURES (All in D:\thesis\figures\, All 300 DPI)

| Filename | Content | Used in |
|---|---|---|
| fig_ch1_allen_cahn_intro.png | 3-panel: evolution / ε effect / double-well | Ch1, meeting guide |
| fig_ch2_pinn_framework.png | PINN training loop | Ch2 |
| fig_ch2_innovation_flow.png | Baseline → CAN-PINN → Enhanced | Ch2, meeting guide |
| fig_ch2_innovation_pyramid.png | Literature context | Ch2 |
| fig_ch3_network_architecture.png | [2→50→50→50→1] | Ch3 |
| fig_ch3_computational_graph.png | AD backward graph | Ch3 |
| fig_ch3_cn_reference_solutions.png | All 4 CN reference fields | Ch3/4 |
| fig_ch3_ad_vs_fd_error.png | FD truncation error vs ε | Ch3 |
| fig_ch3_derivatives_comparison.png | AD exact vs FD stencil | Ch3 |
| fig_ch3_loss_real.png | Training loss curves | Ch3 |
| fig_ch3_optimizer_table.png | Adam vs L-BFGS | Ch3 |
| fig_ch4_results_three_way.png | **MAIN** 3-way bar chart (log scale) | Ch4, meeting guide |
| fig_ch4_three_way_comparison.png | Per-metric breakdown | Ch4 |
| fig_ch4_l2_error_summary.png | L2/Linf per test case | Ch4 |
| fig_ch4_tanh_limitation.png | L² ceiling vs ε | Ch4/5, meeting guide |
| fig_ch4_solution_panels.png | Reference vs predicted + error band | Ch4 |
| fig_ch4_solution_comparison.png | Multi-slice profiles | Ch4 |
| fig_ch5_error_breakdown.png | FD floor + ceiling diagram | Ch5, meeting guide |

---

## 15. SUPERVISOR STYLE AND WHAT SHE EXPECTS

Assoc. Prof. Artemyeva asks in this order, almost every meeting:

1. **"What is your project?"** → One sentence. Not a paragraph.
2. **"What is a PINN?"** → She expects the loss function written out, collocation points explained.
3. **"What is a CAN-PINN?"** → She wants the exact stencil formula and the exact one change.
4. **"What method did you introduce?"** → Must say "Enhanced Adaptive PINN". She corrected "Hybrid CAN-PINN" explicitly.
5. **"What do you maximise and what do you minimise?"** → She always asks this. Minimise L_total over θ; uncertainty weighting = maximising Gaussian likelihood.
6. **"Show me the numbers."** → She wants exact values from JSON files, not approximations.

**What she required for Mark 5 (both previously met):**
- Reference solution (Crank-Nicolson, Δx=1/256) ✅
- Novel contribution beyond reproducing CAN-PINN paper ✅ (three-way + FD floor quantification)
- Mathematical clarity: state explicitly that original CAN-PINN uses FD for ∂²u/∂x² ✅
- Ability to explain Adam, loss function derivation, CAN concept with math ← always be ready

**Her previous concerns (from two recorded meetings):**
- Worried team does not understand the math behind CAN-PINN
- Wants to see Taylor series proof of truncation error, not just the claim
- Checks that reported numbers match actual output files
- Does not accept "better" without a number and a reason

---

## 16. MEETING GUIDES

| File | Pages | Use for |
|---|---|---|
| `D:\thesis\meeting_guide_v2.pdf` | 16 | **Use this** — full math, Taylor proof, algorithm, glossary |
| `D:\thesis\meeting_guide.pdf` | 9 | Old version — Q&A style with blue boxes |

The v2 guide contains:
- Full Taylor series derivation of the FD floor
- Complete algorithm pseudocode (Algorithm 1)
- All 4 enhancement equations
- Results with real numbers
- Plain-language glossary for every technical term

---

## 17. PYTHON ENVIRONMENT

```
Virtual environment:  e:\khokhar-tappa\pinns-canpinns-research\venv\
Activate:             venv\Scripts\activate   (PowerShell)
Platform:             Windows 11 Pro
Key packages:         PyTorch (CUDA), NumPy, Matplotlib, SciPy
Run training:         python training/run_single_test.py  (from project root)
Run figures:          python D:\thesis\upgrade_original_figures.py
                      (script sets cwd automatically)
```

**Important:** `generate_report_figures.py` uses relative paths from project root.
Always run with `cwd = e:\khokhar-tappa\pinns-canpinns-research\` or via `upgrade_original_figures.py` which handles cwd automatically.

---

## 18. IMPLEMENTATION NOTES (Do Not Break These)

- The JSON key in `comparison_summary.json` is `"Hybrid CAN-PINN"` (old name). The thesis and all documents call it **"Enhanced Adaptive PINN"**.
- Total loss of Enhanced method is **negative (−3.15)** — correct behaviour (Gaussian log-likelihood). Report `L_PDE` only when comparing.
- **tanh is mandatory** — do not switch to ReLU. ReLU second derivative = 0 almost everywhere → breaks the PDE residual.
- Single training runs only (no seed averaging) — small differences between AD and FD field errors are within seed-to-seed variation; do not over-interpret them.
- VS Code PDF viewer caches PDFs — must close/reopen tab after recompile to see changes.

---

## 19. SIX DEFENSE REFERENCES

1. **Raissi, Perdikaris & Karniadakis (2019)** — *J. Comput. Phys.* — foundational PINN paper. Defends the whole framework.
2. **Chiu et al. (2022)** — *CMAME* — CAN-PINN error analysis. Defends the FD floor derivation.
3. **Krishnapriyan et al. (2024)** — *Nature Machine Intelligence* — AD essential for accurate PINNs. Defends the 210× finding.
4. **Kendall, Gal & Cipolla (2018)** — *CVPR* — uncertainty-based loss weighting. Defends Enhancement 1.
5. **Han et al. (2022) / Gao et al. (2023)** — residual-based adaptive sampling. Defends Enhancement 2.
6. **Tancik et al. (2020)** — *NeurIPS* — Fourier features, spectral bias. Defends the ceiling explanation and future work.

---

## 20. FUTURE WORK (If Asked)

1. **Fourier feature embeddings** — map (x,t) through sinusoids before first layer → fixes both expressive capacity and spectral bias → directly attacks the 37% ceiling
2. **Ablation study** — apply the 4 enhancements one at a time to isolate individual contributions (not possible with current bundle)
3. **Higher dimensions** — 2D/3D Allen-Cahn; Cahn-Hilliard (4th-order operator makes FD gap even wider)
4. **Operator learning** — DeepONet / FNO to learn a solution map across all ε values at once
5. **Publication** — the quantified FD floor + controlled three-way comparison is publishable in a computational-mathematics venue

---

## 21. QUICK-REFERENCE NUMBERS (Memorise These)

```
Allen-Cahn:           ∂u/∂t = ε·∂²u/∂x² + u − u³
Interface width:      ~ √ε        (0.1 at ε=0.01,  0.22 at ε=0.05)
4th derivative scale: ~ ε⁻²       (10⁴ at ε=0.01)
FD stencil step:      h = 1/256
FD floor (predicted): ~4 × 10⁻³
FD floor (measured):  4.37 × 10⁻³
AD baseline L_PDE:    2.08 × 10⁻⁵
Enhanced L_PDE:       1.19 × 10⁻⁵
FD / AD ratio:        ~210×
Enhancement gain:     ~43%
L² ceiling ε=0.01:   ~37%
L² ceiling ε=0.05:   ~21%
L∞ (step IC):         0.925  (worst case)
Network parameters:   5,301
CN grid:              Δx=1/256, Δt=10⁻⁴, accuracy O(10⁻⁷)
Training (Adam):      10,000 epochs
Training (L-BFGS):    1,000 iterations
Collocation points:   20,000 interior + 1,000 IC + 200 BC
Thesis pages:         77 (target 85–90)
```
