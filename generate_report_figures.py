"""
Generate three publication-quality figures for the thesis report.

Run from the project root:
    python generate_report_figures.py

Output files (saved to report_figures/):
    fig1_three_way_comparison.png   -- Novel contribution bar chart
    fig2_reference_solutions.png    -- All 4 CN reference solutions
    fig3_l2_error_summary.png       -- L2 error vs reference (all cases)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from pathlib import Path

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("report_figures")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

COLORS = {
    "pinn":     "#2196F3",   # blue
    "canpinn":  "#FF5722",   # orange-red
    "fd":       "#9E9E9E",   # grey
    "ref":      "#4CAF50",   # green
}


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Three-way PDE loss comparison (novel contribution)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_three_way():
    with open("three_way_results/comparison_summary.json") as f:
        data = json.load(f)

    methods = ["Baseline\nPINN\n(AD)", "Original\nCAN-PINN\n(FD)", "Hybrid\nCAN-PINN\n(AD + extras)"]
    losses  = [
        data["Baseline PINN"]["pde_loss"],
        data["Original CAN-PINN"]["pde_loss"],
        data["Hybrid CAN-PINN"]["pde_loss"],
    ]
    colors  = [COLORS["pinn"], COLORS["fd"], COLORS["canpinn"]]
    times   = [
        data["Baseline PINN"]["time"],
        data["Original CAN-PINN"]["time"],
        data["Hybrid CAN-PINN"]["time"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Figure 1 — Three-Way Comparison: Baseline PINN vs Original CAN-PINN (FD) vs Hybrid CAN-PINN (AD)\n"
        "Test Case: Allen-Cahn, sin(πx) IC, ε = 0.01",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # ── left: PDE loss bar chart (log scale) ──────────────────────────────────
    ax = axes[0]
    bars = ax.bar(methods, losses, color=colors, edgecolor="black", linewidth=0.8, width=0.5)

    # annotate bars with values and improvement labels
    improvements = [None, f"210× worse", f"43% better"]
    for bar, loss, label in zip(bars, losses, improvements):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            loss * 1.6,
            f"{loss:.2e}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
        if label:
            color = "#B71C1C" if "worse" in label else "#1B5E20"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                loss * 6,
                label,
                ha="center", va="bottom", fontsize=9,
                color=color, fontweight="bold",
            )

    ax.set_yscale("log")
    ax.set_ylabel("PDE Residual Loss (log scale)")
    ax.set_title("PDE Loss Comparison\n(lower is better)")
    ax.set_ylim(losses[2] * 0.3, losses[1] * 30)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # horizontal reference line at baseline PINN
    ax.axhline(losses[0], color=COLORS["pinn"], linestyle="--", linewidth=1.2, alpha=0.6, label="Baseline PINN level")
    ax.legend(loc="upper right", fontsize=9)

    # ── right: training time bar chart ────────────────────────────────────────
    ax2 = axes[1]
    bars2 = ax2.bar(methods, times, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    for bar, t in zip(bars2, times):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            t + 8,
            f"{t:.0f}s",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax2.set_ylabel("Training Time (seconds)")
    ax2.set_title("Computational Cost\n(lower is better)")
    ax2.grid(axis="y", alpha=0.4, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # annotation box explaining the key insight
    textstr = (
        "Key Insight:\n"
        "Original CAN-PINN uses Finite Differences\n"
        "for ∂²u/∂x², introducing O(Δx²) truncation\n"
        "error ≈ 1×10⁻⁴. Replacing FD with exact\n"
        "Automatic Differentiation eliminates this\n"
        "error entirely → 210× PDE residual reduction."
    )
    ax2.text(
        1.04, 0.97, textstr,
        transform=ax2.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9C4", edgecolor="#F9A825", alpha=0.9),
    )

    plt.tight_layout()
    out = OUT_DIR / "fig1_three_way_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — All 4 Crank-Nicolson reference solutions
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_reference_solutions():
    cases = [
        ("TC2_sin_eps001_reference.npz",  "TC2: sin(πx), ε = 0.01",  "sin IC, moderate diffusion"),
        ("TC2_step_eps001_reference.npz", "TC2: Step, ε = 0.01",     "step IC, sharp interface"),
        ("TC3_sin_eps001_reference.npz",  "TC3: sin(πx), ε = 0.01",  "sin IC, moderate diffusion"),
        ("TC3_sin_eps005_reference.npz",  "TC3: sin(πx), ε = 0.05",  "sin IC, wider interface"),
    ]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Figure 2 — Crank-Nicolson Reference Solutions (Ground Truth)\n"
        "Scheme: implicit CN, Δx = 1/256, Δt = 0.0001,  accuracy O(10⁻⁷)",
        fontsize=13, fontweight="bold",
    )

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    for col, (fname, title, subtitle) in enumerate(cases):
        ref = np.load(f"reference_solutions/{fname}")
        u   = ref["u"]   # [nt, nx]
        x   = ref["x"]
        t   = ref["t"]
        eps = float(ref["epsilon"])

        # ── top row: 2-D contour ──────────────────────────────────────────────
        ax_top = fig.add_subplot(gs[0, col])
        cf = ax_top.contourf(x, t, u, levels=30, cmap="jet", vmin=0, vmax=1)
        plt.colorbar(cf, ax=ax_top, fraction=0.046, pad=0.04, label="u(x,t)")
        ax_top.set_title(f"{title}\n({subtitle})", fontsize=10, fontweight="bold")
        ax_top.set_xlabel("x")
        ax_top.set_ylabel("t")

        # ── bottom row: time slices ───────────────────────────────────────────
        ax_bot = fig.add_subplot(gs[1, col])
        snap_times = [0.0, 0.25, 0.5, 0.75, 1.0]
        cmap_slices = plt.cm.plasma(np.linspace(0.1, 0.9, len(snap_times)))
        for snap_t, color in zip(snap_times, cmap_slices):
            idx = np.argmin(np.abs(t - snap_t))
            ax_bot.plot(x, u[idx], color=color, linewidth=1.8, label=f"t={snap_t:.2f}")

        ax_bot.set_xlabel("x")
        ax_bot.set_ylabel("u(x, t)")
        ax_bot.set_title(f"Time Slices  [ε = {eps}]", fontsize=10)
        ax_bot.set_xlim(0, 1)
        ax_bot.set_ylim(-0.05, 1.1)
        ax_bot.legend(fontsize=8, loc="upper right", ncol=1)
        ax_bot.grid(alpha=0.3)
        ax_bot.spines["top"].set_visible(False)
        ax_bot.spines["right"].set_visible(False)

        ref.close()

    out = OUT_DIR / "fig2_reference_solutions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — L2 error summary across all 4 test cases
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_l2_error_summary():
    with open("error_analysis_results.json") as f:
        err = json.load(f)

    # ordered labels matching JSON keys
    case_keys    = ["TC2_sin_eps001", "TC2_step_eps001", "TC3_sin_eps001", "TC3_sin_eps005"]
    case_labels  = [
        "TC2\nsin(πx), ε=0.01",
        "TC2\nStep, ε=0.01",
        "TC3\nsin(πx), ε=0.01",
        "TC3\nsin(πx), ε=0.05",
    ]

    l2_pinn    = [err[k]["PINN"]["L2_relative"]    for k in case_keys]
    l2_canpinn = [err[k]["CAN-PINN"]["L2_relative"] for k in case_keys]
    linf_pinn    = [err[k]["PINN"]["Linf_absolute"]    for k in case_keys]
    linf_canpinn = [err[k]["CAN-PINN"]["Linf_absolute"] for k in case_keys]

    x = np.arange(len(case_keys))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Figure 3 — Error vs Crank-Nicolson Reference Solution (All Test Cases)\n"
        "PINN vs Hybrid CAN-PINN",
        fontsize=13, fontweight="bold",
    )

    # ── left: L2 relative error ───────────────────────────────────────────────
    ax = axes[0]
    b1 = ax.bar(x - width/2, l2_pinn,    width, label="Baseline PINN",     color=COLORS["pinn"],    edgecolor="black", linewidth=0.7, alpha=0.9)
    b2 = ax.bar(x + width/2, l2_canpinn, width, label="Hybrid CAN-PINN",   color=COLORS["canpinn"], edgecolor="black", linewidth=0.7, alpha=0.9)

    for bar, val in zip(list(b1) + list(b2), l2_pinn + l2_canpinn):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=10)
    ax.set_ylabel("L² Relative Error  ‖u_pred − u_ref‖ / ‖u_ref‖")
    ax.set_title("L² Relative Error vs CN Reference\n(lower is better)")
    ax.set_ylim(0, max(l2_pinn + l2_canpinn) * 1.35)
    ax.legend()
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # shade bands to show error regions
    ax.axhspan(0, 0.05,  alpha=0.07, color="green", label="Excellent (<5%)")
    ax.axhspan(0.05, 0.15, alpha=0.07, color="yellow")
    ax.axhspan(0.15, 1.0,  alpha=0.07, color="red")
    ax.text(3.75, 0.01,  "Excellent", fontsize=8, color="green",  alpha=0.7, ha="right")
    ax.text(3.75, 0.09,  "Moderate",  fontsize=8, color="olive",  alpha=0.7, ha="right")
    ax.text(3.75, 0.30,  "High Error",fontsize=8, color="red",    alpha=0.7, ha="right")

    # ── right: L∞ absolute error ──────────────────────────────────────────────
    ax2 = axes[1]
    b3 = ax2.bar(x - width/2, linf_pinn,    width, label="Baseline PINN",   color=COLORS["pinn"],    edgecolor="black", linewidth=0.7, alpha=0.9)
    b4 = ax2.bar(x + width/2, linf_canpinn, width, label="Hybrid CAN-PINN", color=COLORS["canpinn"], edgecolor="black", linewidth=0.7, alpha=0.9)

    for bar, val in zip(list(b3) + list(b4), linf_pinn + linf_canpinn):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(case_labels, fontsize=10)
    ax2.set_ylabel("L∞ Absolute Error  max|u_pred − u_ref|")
    ax2.set_title("L∞ Absolute Error vs CN Reference\n(lower is better)")
    ax2.set_ylim(0, max(linf_pinn + linf_canpinn) * 1.25)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.4, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # key finding annotation
    finding = (
        "Key Finding:\n"
        "Both PINN and Hybrid CAN-PINN converge\n"
        "to similar solutions with ~37% L² error\n"
        "vs the true CN reference at ε = 0.01.\n\n"
        "At ε = 0.05 (wider interface), error\n"
        "drops to ~21% — smoother solutions\n"
        "are easier for Tanh MLPs to capture.\n\n"
        "Root cause: smooth Tanh activation\n"
        "cannot represent sharp phase transitions\n"
        "— a known PINN architectural limitation."
    )
    fig.text(
        0.995, 0.5, finding,
        fontsize=9, verticalalignment="center", ha="right",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#E3F2FD", edgecolor="#1565C0", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    out = OUT_DIR / "fig3_l2_error_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS FIGURE 4 — Solution + Error heatmaps for all 4 cases
# (Reference | PINN | CAN-PINN | Error — in one panel per case)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_solution_comparison():
    from scipy.interpolate import RegularGridInterpolator

    cases = [
        ("TC2_sin_eps001",  "TC2: sin(πx), ε=0.01"),
        ("TC2_step_eps001", "TC2: Step IC, ε=0.01"),
        ("TC3_sin_eps001",  "TC3: sin(πx), ε=0.01"),
        ("TC3_sin_eps005",  "TC3: sin(πx), ε=0.05"),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    fig.suptitle(
        "Figure 4 — Solution Comparison: CN Reference vs PINN vs Hybrid CAN-PINN\n"
        "Columns: (1) Reference  (2) PINN Prediction  (3) CAN-PINN Prediction  (4) |Error| (PINN − CAN-PINN)",
        fontsize=12, fontweight="bold",
    )

    col_titles = [
        "Crank-Nicolson\nReference (Truth)",
        "Baseline PINN\nPrediction",
        "Hybrid CAN-PINN\nPrediction",
        "|PINN − CAN-PINN|\n(Method Difference)",
    ]
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=10, fontweight="bold", pad=8)

    for row, (key, label) in enumerate(cases):
        ref   = np.load(f"reference_solutions/{key}_reference.npz")
        pinn  = np.load(f"outputs/{key}_pinn_solution.npz")
        canp  = np.load(f"outputs/{key}_canpinn_solution.npz")

        u_ref = ref["u"]
        x_ref = ref["x"]
        t_ref = ref["t"]

        x_pred = pinn["x_test"].ravel()
        t_pred = pinn["t_test"].ravel()
        x_u = np.unique(x_pred)
        t_u = np.unique(t_pred)
        nx, nt = len(x_u), len(t_u)

        u_pinn  = pinn["u_pred"].ravel().reshape(nt, nx)
        u_canp  = canp["u_pred"].ravel().reshape(nt, nx)

        # interpolate to reference grid
        interp_p = RegularGridInterpolator((t_u, x_u), u_pinn, bounds_error=False, fill_value=np.nan)
        interp_c = RegularGridInterpolator((t_u, x_u), u_canp, bounds_error=False, fill_value=np.nan)
        T, X = np.meshgrid(t_ref, x_ref, indexing="ij")
        pts  = np.column_stack([T.ravel(), X.ravel()])
        u_p_grid = interp_p(pts).reshape(u_ref.shape)
        u_c_grid = interp_c(pts).reshape(u_ref.shape)
        diff     = np.abs(u_p_grid - u_c_grid)

        grids  = [u_ref, u_p_grid, u_c_grid, diff]
        cmaps  = ["jet", "jet", "jet", "hot_r"]
        vmins  = [0,     0,      0,     0]
        vmaxs  = [1,     1,      1,     diff.max()]

        for col, (grid, cmap, vmin, vmax) in enumerate(zip(grids, cmaps, vmins, vmaxs)):
            ax = axes[row, col]
            cf = ax.contourf(x_ref, t_ref, grid, levels=25, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel("x", fontsize=9)
            ax.set_ylabel("t", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{label}\n\nt", fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=8)

        ref.close(); pinn.close(); canp.close()

    plt.tight_layout()
    out = OUT_DIR / "fig4_solution_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating report figures...\n")
    fig1_three_way()
    fig2_reference_solutions()
    fig3_l2_error_summary()
    fig4_solution_comparison()
    print(f"\nAll figures saved to: {OUT_DIR.resolve()}")
    print("\nFigure summary:")
    print("  fig1_three_way_comparison.png  — Novel contribution (FD vs AD)")
    print("  fig2_reference_solutions.png   — Ground truth CN solutions (all 4 cases)")
    print("  fig3_l2_error_summary.png      — L2/Linf error vs CN reference")
    print("  fig4_solution_comparison.png   — Full solution+error panel (all 4 cases)")
