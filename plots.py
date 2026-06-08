"""
plots.py
========
Final publication figures for:

    A Formal Framework for Layered Asset Claims and Resolution Outcomes
    in Rehypothecation Networks
    Blessing Honmane, 2026

This script generates all cross-scenario comparison figures for the paper.
It reads outputs from dynamics.py, monitoring.py, and resolution.py and
produces figures that demonstrate the paper's core claims:

    1. The two-layer early warning system (Phi and Delta)
    2. The intervention window across crisis types
    3. The netting underestimation theorem
    4. Selection rule divergence as fragility signal
    5. Regime classification across scenarios

All figures saved to simulation_results/paper_figures/
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------

SCENARIOS       = ["economic_drift", "legal_shock", "compound"]
SCENARIO_LABELS = {
    "economic_drift": "Economic Drift",
    "legal_shock":    "Legal Regime Shock",
    "compound":       "Compound",
}
COLORS = {
    "economic_drift": "#1f77b4",
    "legal_shock":    "#d62728",
    "compound":       "#2ca02c",
}
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR  = os.path.join(_SCRIPT_DIR, "simulation_results", "paper_figures")

os.makedirs(FIGURE_DIR, exist_ok=True)


def load_all(scenario: str) -> dict:
    d = {}
    base_dyn = os.path.join(_SCRIPT_DIR, f"simulation_results/{scenario}/dynamics")
    base_mon = os.path.join(_SCRIPT_DIR, f"simulation_results/{scenario}/monitoring")
    base_res = os.path.join(_SCRIPT_DIR, f"simulation_results/{scenario}/resolution")
    base_dat = os.path.join(_SCRIPT_DIR, f"simulation_data/{scenario}")

    d["kappa"]     = np.load(f"{base_dyn}/ts_kappa.npy")
    d["L"]         = np.load(f"{base_dyn}/ts_L.npy")
    d["dG"]        = np.load(f"{base_dyn}/ts_dG.npy")
    d["Phi"]       = np.load(f"{base_dyn}/ts_Phi.npy")
    d["mu"]        = np.load(f"{base_dyn}/ts_mu.npy")
    d["rho"]       = np.load(f"{base_dyn}/ts_rho.npy")

    d["Delta"]     = np.load(f"{base_mon}/ts_Delta.npy")
    d["beta"]      = np.load(f"{base_mon}/ts_beta.npy")
    d["r_lex"]     = np.load(f"{base_mon}/ts_r_lex.npy")
    d["r_regret"]  = np.load(f"{base_mon}/ts_r_regret.npy")
    d["r_hist"]    = np.load(f"{base_mon}/ts_r_hist.npy")

    d["r_lex_res"]     = np.load(f"{base_res}/r_lex.npy")
    d["r_regret_res"]  = np.load(f"{base_res}/r_regret.npy")
    d["r_hist_res"]    = np.load(f"{base_res}/r_hist.npy")
    d["r_netting_res"] = np.load(f"{base_res}/r_netting.npy")
    d["pi"]            = np.load(f"{base_res}/pi_scores.npy")
    d["nominal"]       = np.load(f"{base_dat}/nominal_amounts.npy")
    d["rho_matrix"]    = np.load(f"{base_dat}/rho_matrix_init.npy")
    d["V_A"]           = np.load(f"{base_dat}/V_A.npy")

    with open(f"{base_mon}/monitoring_summary.json") as f:
        d["mon_summary"] = json.load(f)
    with open(f"{base_res}/resolution_summary.json") as f:
        d["res_summary"] = json.load(f)
    with open(f"{base_dyn}/dynamics_meta.json") as f:
        d["dyn_meta"] = json.load(f)

    d["tau"]        = d["mon_summary"]["tau_info"]["tau_intervention"]
    d["t_phi"]      = d["mon_summary"]["tau_info"]["t_phi_star"]
    d["t_delta"]    = d["mon_summary"]["tau_info"]["t_delta_star"]
    d["t_crisis"]   = d["mon_summary"]["transition"]["t_transition"]
    d["shock_t"]    = d["dyn_meta"]["shock_timestep"]
    d["legal_t"]    = d["dyn_meta"]["cfg"].get("legal_shock_timestep")
    d["T"]          = d["dyn_meta"]["T"]

    return d


# -----------------------------------------------------------------------
# Figure 1: PRIMARY — Two-layer early warning system, all three scenarios
# -----------------------------------------------------------------------

def fig1_early_warning(all_data: dict) -> None:
    """
    Three-panel figure showing Phi(t) and Delta(t) for all scenarios.
    Primary paper figure demonstrating the two-layer early warning system
    and the intervention window tau_intervention.

    Paper: Remark -- two-layer early warning system.
           Definition -- Regulatory Intervention Window.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Two-Layer Early Warning System: $\\Phi(A,t)$ and $\\Delta(t)$\n"
        "Across All Three Crisis Scenarios",
        fontsize=14, fontweight='bold'
    )

    for ax, scenario in zip(axes, SCENARIOS):
        d    = all_data[scenario]
        T    = d["T"]
        ts   = np.arange(T)
        tau  = d["tau"]
        label = SCENARIO_LABELS[scenario]

        Phi   = d["Phi"]
        Delta = d["Delta"]

        # Normalise both to [0,1] for overlay
        Phi_n   = (Phi   - Phi.min())   / (Phi.max()   - Phi.min()   + 1e-10)
        Delta_n = (Delta - Delta.min()) / (Delta.max() - Delta.min() + 1e-10)

        ax.plot(ts, Phi_n,   color="#9467bd", lw=2.2,
                label=r"$\Phi(A,t)$ — structural fragility")
        ax.plot(ts, Delta_n, color="#d62728", lw=2.2,
                label=r"$\Delta(t)$ — resolution sensitivity")

        t_phi   = d["t_phi"]
        t_delta = d["t_delta"]

        ax.axvline(t_phi,   color="#9467bd", ls="--", lw=1.5, alpha=0.7)
        ax.axvline(t_delta, color="#d62728", ls="--", lw=1.5, alpha=0.7)

        if tau > 0:
            ax.axvspan(t_phi, t_delta, alpha=0.13, color="#2ca02c")
            ax.text((t_phi + t_delta)/2, 0.88,
                    f"$\\tau={tau}$", ha='center', fontsize=10,
                    color="#2ca02c", fontweight='bold')
        elif tau < 0:
            ax.text(0.5, 0.88, f"$\\tau={tau}$\n(Δ precedes Φ)",
                    ha='center', transform=ax.transAxes,
                    fontsize=9, color="#d62728")
        else:
            ax.text(0.5, 0.88, f"$\\tau\\approx 0$",
                    ha='center', transform=ax.transAxes,
                    fontsize=10, color="black")

        # Mark shocks
        ax.axvline(d["shock_t"], color="gray", ls="-.", lw=1, alpha=0.5)
        if d["legal_t"]:
            ax.axvline(d["legal_t"], color="navy", ls="-.", lw=1.5, alpha=0.6)

        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel("Timestep", fontsize=10)
        ax.set_ylabel("Normalised value [0,1]", fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)

        if ax == axes[0]:
            ax.legend(fontsize=8, loc="upper left")

    # Shared legend for shock markers
    legend_elements = [
        Line2D([0],[0], color="gray",  ls="-.", lw=1.5, label="Collateral shock"),
        Line2D([0],[0], color="navy",  ls="-.", lw=1.5, label="Legal regime shock"),
        Patch(facecolor="#2ca02c", alpha=0.3, label=r"$\tau_{\mathrm{intervention}}$ window"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fname = f"{FIGURE_DIR}/fig1_early_warning_system.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# -----------------------------------------------------------------------
# Figure 2: Phi decomposition — diagnostic vs aggregate
# -----------------------------------------------------------------------

def fig2_phi_decomposition(all_data: dict) -> None:
    """
    Three-column, four-row figure showing kappa, L, dG, Phi for each scenario.
    Demonstrates that Phi collapses three structurally different drivers into
    one number — the diagnostic limitation stated in the paper.

    Paper: Remark -- Limitation of Phi as Diagnostic.
    """
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    fig.suptitle(
        "$\\Phi(A,t)$ Decomposition: Three Drivers Across Scenarios\n"
        "High $\\Phi$ signals fragility but not its source — "
        "all three inputs must be examined for diagnosis.",
        fontsize=13, fontweight='bold'
    )

    row_labels = [
        r"$\kappa(G^{\mathrm{econ}})$ — Coupling Instability",
        r"$L(t) = \sum w_i / V_A$ — Overclaim Ratio",
        r"$\|dG^{\mathrm{econ}}/dt\|$ — Kernel Velocity",
        r"$\Phi(A,t)$ — Aggregate Fragility",
    ]
    row_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]
    row_keys   = ["kappa", "L", "dG", "Phi"]

    for col, scenario in enumerate(SCENARIOS):
        d   = all_data[scenario]
        T   = d["T"]
        ts  = np.arange(T)

        for row, (key, label, color) in enumerate(
                zip(row_keys, row_labels, row_colors)):
            ax = axes[row, col]
            ax.plot(ts, d[key], color=color, lw=1.8)
            ax.axvline(d["shock_t"], color="gray", ls="--", lw=1, alpha=0.5)
            if d["legal_t"]:
                ax.axvline(d["legal_t"], color="navy", ls=":", lw=1.5, alpha=0.7)
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(SCENARIO_LABELS[scenario],
                             fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            if row == 3:
                ax.set_xlabel("Timestep", fontsize=9)
                ax.fill_between(ts, d[key], alpha=0.15, color=color)

    plt.tight_layout()
    fname = f"{FIGURE_DIR}/fig2_phi_decomposition.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# -----------------------------------------------------------------------
# Figure 3: Intervention window comparison — the key quantitative result
# -----------------------------------------------------------------------

def fig3_intervention_window(all_data: dict) -> None:
    """
    Bar chart comparing tau_intervention across the three scenarios.
    The core quantitative result of the simulation.

    Paper: Definition -- Regulatory Intervention Window.
    Prediction: economic_drift > compound > legal_shock.
    """
    taus   = [all_data[s]["tau"]   for s in SCENARIOS]
    labels = [SCENARIO_LABELS[s]   for s in SCENARIOS]
    colors_bar = [COLORS[s]        for s in SCENARIOS]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(
        "Regulatory Intervention Window $\\tau_{\\mathrm{intervention}}$ "
        "by Crisis Type\n"
        "$\\tau = t_{\\Delta^*} - t_{\\Phi^*}$: "
        "steps between structural warning and resolution sensitivity warning",
        fontsize=12, fontweight='bold'
    )

    bars = ax.bar(labels, taus, color=colors_bar, alpha=0.85,
                  edgecolor="black", linewidth=0.8, width=0.5)
    ax.axhline(0, color="black", lw=1.2)

    for bar, val, scenario in zip(bars, taus, SCENARIOS):
        offset = 1.5 if val >= 0 else -3
        ax.text(bar.get_x() + bar.get_width()/2,
                val + offset,
                f"{val} steps",
                ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=11, fontweight='bold')

    ax.set_ylabel("$\\tau_{\\mathrm{intervention}}$ (timesteps)", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotation
    ax.text(0.98, 0.95,
            "Positive $\\tau$: $\\Phi$ warns before $\\Delta$\n"
            "Negative $\\tau$: $\\Delta$ fires before $\\Phi$\n"
            "(resolution sensitive before geometry unstable)",
            transform=ax.transAxes, fontsize=9,
            va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fname = f"{FIGURE_DIR}/fig3_intervention_window.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# -----------------------------------------------------------------------
# Figure 4: Netting underestimation — Theorem 2.1 visualised
# -----------------------------------------------------------------------

def fig4_netting_underestimation(all_data: dict) -> None:
    """
    Three-panel plot showing per-claim allocation gap between
    framework methods and classical netting for all scenarios.
    Direct visual demonstration of Theorem 2.1.

    Paper: Theorem -- Classical Netting Underestimation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Netting Underestimation Gap: $r_{\\mathrm{lex},i} - r_{\\mathrm{net},i}$\n"
        "Theorem 2.1: Classical bilateral netting systematically "
        "misallocates when claims share collateral ancestry ($\\rho_{ij}>0$)",
        fontsize=12, fontweight='bold'
    )

    for ax, scenario in zip(axes, SCENARIOS):
        d       = all_data[scenario]
        n       = len(d["nominal"])
        gap     = d["r_lex_res"] - d["r_netting_res"]
        rho_mat = d["rho_matrix"]
        claims  = np.arange(n)

        # Identify correlated claims
        corr_set = set()
        for i in range(n):
            for j in range(i+1, n):
                if rho_mat[i, j] > 0:
                    corr_set.add(i)
                    corr_set.add(j)

        bar_colors = []
        for i in range(n):
            if i in corr_set:
                bar_colors.append("#d62728" if gap[i] >= 0 else "#ff7f0e")
            else:
                bar_colors.append("#aec7e8" if gap[i] >= 0 else "#c5b0d5")

        ax.bar(claims, gap, color=bar_colors, alpha=0.85, width=0.8)
        ax.axhline(0, color="black", lw=1.2)

        res = d["res_summary"]
        gap_pct = res["gap_lex"]["total_gap_pct"]
        n_corr  = res["gap_lex"]["n_correlated_pairs"]
        rho_mean = res["gap_lex"]["mean_rho"]

        ax.set_title(
            f"{SCENARIO_LABELS[scenario]}\n"
            f"Total gap: {gap_pct:.1f}% | "
            f"Corr. pairs: {n_corr} | "
            f"$\\bar{{\\rho}}$={rho_mean:.3f}",
            fontsize=10, fontweight='bold'
        )
        ax.set_xlabel("Claimant index", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel(
                r"$r_{\mathrm{lex},i} - r_{\mathrm{net},i}$", fontsize=11)
        ax.grid(True, alpha=0.3)

    legend_elements = [
        Patch(facecolor="#d62728", alpha=0.85,
              label="Correlated: framework > netting"),
        Patch(facecolor="#ff7f0e", alpha=0.85,
              label="Correlated: framework < netting"),
        Patch(facecolor="#aec7e8", alpha=0.85,
              label="Uncorrelated: framework > netting"),
        Patch(facecolor="#c5b0d5", alpha=0.85,
              label="Uncorrelated: framework < netting"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fname = f"{FIGURE_DIR}/fig4_netting_underestimation.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# -----------------------------------------------------------------------
# Figure 5: Selection rule divergence — monitoring mode in action
# -----------------------------------------------------------------------

def fig5_selection_rule_divergence(all_data: dict) -> None:
    """
    Three-panel plot showing Delta(t) and mean allocations per rule over time.
    Demonstrates monitoring mode — the rules diverge as the system approaches
    the crisis fixed point.

    Paper: Definition -- Selection Rule Divergence Measure.
           Remark -- two-layer early warning system.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Monitoring Mode: Selection Rule Divergence $\\Delta(t)$ "
        "and Mean Allocations\n"
        "Rule divergence is the operational signal of approaching "
        "irresolvability",
        fontsize=13, fontweight='bold'
    )

    for col, scenario in enumerate(SCENARIOS):
        d  = all_data[scenario]
        T  = d["T"]
        ts = np.arange(T)

        # Top row: Delta(t)
        ax_top = axes[0, col]
        Delta_n = (d["Delta"] - d["Delta"].min()) / \
                  (d["Delta"].max() - d["Delta"].min() + 1e-10)
        ax_top.plot(ts, Delta_n, color="#d62728", lw=2,
                    label=r"$\Delta(t)$ normalised")
        ax_top.fill_between(ts, Delta_n, alpha=0.12, color="#d62728")
        ax_top.axvline(d["t_delta"], color="#d62728", ls="--", lw=1.5)
        ax_top.axvline(d["shock_t"], color="gray", ls="-.", lw=1, alpha=0.5)
        if d["legal_t"]:
            ax_top.axvline(d["legal_t"], color="navy", ls="-.", lw=1.5, alpha=0.6)
        ax_top.set_title(f"{SCENARIO_LABELS[scenario]}\n"
                         f"$\\tau={d['tau']}$ steps",
                         fontsize=10, fontweight='bold')
        ax_top.set_ylabel(r"$\Delta(t)$ normalised", fontsize=9)
        ax_top.grid(True, alpha=0.3)
        ax_top.set_ylim(-0.05, 1.1)

        # Bottom row: mean allocation per rule
        ax_bot = axes[1, col]
        mean_lex    = d["r_lex"].mean(axis=1)
        mean_regret = d["r_regret"].mean(axis=1)
        mean_hist   = d["r_hist"].mean(axis=1)

        ax_bot.plot(ts, mean_lex,    color="#1f77b4", lw=2,
                    label=r"$\hat{R}_{\mathrm{lex}}$")
        ax_bot.plot(ts, mean_regret, color="#d62728", lw=2, ls="--",
                    label=r"$\hat{R}_{\mathrm{regret}}$")
        ax_bot.plot(ts, mean_hist,   color="#2ca02c", lw=2, ls=":",
                    label=r"$\hat{R}_{\mathrm{hist}}$")
        ax_bot.fill_between(ts, mean_lex, mean_regret,
                            alpha=0.1, color="#9467bd")
        ax_bot.axvline(d["shock_t"], color="gray", ls="-.", lw=1, alpha=0.5)
        if d["legal_t"]:
            ax_bot.axvline(d["legal_t"], color="navy", ls="-.", lw=1.5, alpha=0.6)
        ax_bot.set_xlabel("Timestep", fontsize=9)
        ax_bot.set_ylabel("Mean allocation per claim", fontsize=9)
        ax_bot.legend(fontsize=8)
        ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{FIGURE_DIR}/fig5_selection_rule_divergence.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# -----------------------------------------------------------------------
# Figure 6: Regime classification — all scenarios
# -----------------------------------------------------------------------

def fig6_regime_classification(all_data: dict) -> None:
    """
    Three-panel regime classification showing Regime 1/2/3 over time.
    Demonstrates the bifurcation-based regime transitions.

    Paper: Section 1.9 -- The Three Regimes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Regime Classification Over Time\n"
        "Regime 1: Stable — Regime 2: Transitional — Regime 3: Critical",
        fontsize=13, fontweight='bold'
    )

    regime_colors = {1: "#2ca02c", 2: "#ff7f0e", 3: "#d62728"}
    regime_labels = {
        1: "Regime 1 — Stable",
        2: "Regime 2 — Transitional",
        3: "Regime 3 — Critical"
    }

    for ax, scenario in zip(axes, SCENARIOS):
        d   = all_data[scenario]
        T   = d["T"]
        ts  = np.arange(T)

        beta  = d["beta"]
        Delta = d["Delta"]
        nominal_sum = d["nominal"].sum()

        Delta_n = (Delta - Delta.min()) / \
                  (Delta.max() - Delta.min() + 1e-10)
        Phi_n   = (d["Phi"] - d["Phi"].min()) / \
                  (d["Phi"].max() - d["Phi"].min() + 1e-10)

        # Classify regimes
        beta_max = beta.max() if beta.max() > 0 else 1
        regimes = np.ones(T, dtype=int)
        for t in range(T):
            if beta[t] > beta_max * 0.5:
                regimes[t] = 3
            elif Delta_n[t] > 0.3:
                regimes[t] = 2

        for regime_id, color in regime_colors.items():
            mask = regimes == regime_id
            ax.fill_between(ts, 0, 1, where=mask,
                            alpha=0.3, color=color,
                            transform=ax.get_xaxis_transform(),
                            label=regime_labels[regime_id])

        ax.plot(ts, Phi_n,   color="#9467bd", lw=1.8, alpha=0.9,
                label=r"$\Phi(t)$ normalised")
        ax.plot(ts, Delta_n, color="#d62728", lw=1.8, alpha=0.9,
                ls="--", label=r"$\Delta(t)$ normalised")

        ax.axvline(d["shock_t"], color="gray",  ls="-.", lw=1,   alpha=0.5)
        if d["legal_t"]:
            ax.axvline(d["legal_t"], color="navy", ls="-.", lw=1.5, alpha=0.6)

        ax.set_title(SCENARIO_LABELS[scenario], fontsize=11, fontweight='bold')
        ax.set_xlabel("Timestep", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
        if ax == axes[0]:
            ax.set_ylabel("Regime / Signal", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    fname = f"{FIGURE_DIR}/fig6_regime_classification.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# -----------------------------------------------------------------------
# Figure 7: Resolution outcomes — per-claimant shortfall distribution
# -----------------------------------------------------------------------

def fig7_resolution_outcomes(all_data: dict) -> None:
    """
    For each scenario: side-by-side violin/bar plots showing how different
    selection rules distribute the shortfall across claimants differently.

    Total shortfall is identical across methods (structurally fixed at 0.6
    of total claims). What differs is HOW it is distributed — concentrated
    on low-priority claimants under R_lex vs spread under R_regret.

    Paper: Definitions R_lex, R_regret, R_hist.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Per-Claimant Shortfall Distribution at Crisis Fixed Point\n"
        "Total shortfall is structurally fixed — the selection rule "
        "determines its distribution across claimants",
        fontsize=12, fontweight='bold'
    )

    methods = [
        ("r_lex_res",     r"$\hat{R}_{\mathrm{lex}}$",     "#1f77b4"),
        ("r_regret_res",  r"$\hat{R}_{\mathrm{regret}}$",  "#d62728"),
        ("r_hist_res",    r"$\hat{R}_{\mathrm{hist}}$",     "#2ca02c"),
        ("r_netting_res", "Classical\nNetting",              "#ff7f0e"),
    ]

    for ax, scenario in zip(axes, SCENARIOS):
        d       = all_data[scenario]
        nominal = d["nominal"]

        shortfalls_by_method = []
        labels_m = []
        colors_m = []

        for key, label, color in methods:
            r = d[key]
            shortfall_per_claim = np.maximum(0, nominal - r)
            shortfalls_by_method.append(shortfall_per_claim)
            labels_m.append(label)
            colors_m.append(color)

        # Violin plot of per-claim shortfall distribution
        parts = ax.violinplot(
            shortfalls_by_method,
            positions=range(len(methods)),
            showmeans=True,
            showmedians=True
        )

        for pc, color in zip(parts['bodies'], colors_m):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        # Also plot max_regret as annotation
        max_regrets = [d["res_summary"]["metrics"][k]["max_regret"]
                       for k in ["R_lex","R_regret","R_hist","R_netting"]]
        ax2 = ax.twinx()
        ax2.scatter(range(len(methods)), max_regrets,
                    color=colors_m, s=80, zorder=5, marker="D",
                    label="Max regret")
        ax2.set_ylabel("Max regret (worst-off claimant)", fontsize=8,
                       color="gray")
        ax2.tick_params(axis='y', labelcolor='gray')

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels_m, fontsize=9)
        ax.set_title(SCENARIO_LABELS[scenario],
                     fontsize=11, fontweight='bold')
        ax.set_ylabel("Per-claimant shortfall", fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = f"{FIGURE_DIR}/fig7_resolution_outcomes.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# -----------------------------------------------------------------------
# Figure 8: Kernel evolution — interaction geometry across scenarios
# -----------------------------------------------------------------------

def fig8_kernel_evolution(all_data: dict) -> None:
    """
    kappa(t) and rho(t) for all three scenarios on one figure.
    Shows how the interaction geometry deteriorates differently
    under economic drift vs legal shock vs compound.

    Paper: Definition -- Decomposed Interaction Kernel.
           dG_econ/dt != 0 even when asset unchanged.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Interaction Kernel Evolution: "
        "$\\kappa(G^{\\mathrm{econ}})$ and $\\rho(t)$ Across Scenarios\n"
        "$dG^{\\mathrm{econ}}/dt \\neq 0$ even when underlying assets unchanged",
        fontsize=12, fontweight='bold'
    )

    for scenario in SCENARIOS:
        d     = all_data[scenario]
        T     = d["T"]
        ts    = np.arange(T)
        color = COLORS[scenario]
        label = SCENARIO_LABELS[scenario]

        # Log kappa for readability
        kappa_log = np.log1p(d["kappa"])
        ax1.plot(ts, kappa_log, color=color, lw=2, label=label)

        ax2.plot(ts, d["rho"], color=color, lw=2, label=label)

        # Mark shocks
        ax1.axvline(d["shock_t"], color=color, ls="--", lw=1, alpha=0.4)
        ax2.axvline(d["shock_t"], color=color, ls="--", lw=1, alpha=0.4)
        if d["legal_t"]:
            ax1.axvline(d["legal_t"], color=color, ls=":", lw=1.5, alpha=0.6)
            ax2.axvline(d["legal_t"], color=color, ls=":", lw=1.5, alpha=0.6)

    ax1.set_title(r"$\log(1+\kappa(G^{\mathrm{econ}}))$ — Coupling Instability",
                  fontsize=11)
    ax1.set_xlabel("Timestep", fontsize=10)
    ax1.set_ylabel(r"$\log(1+\kappa)$", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_title(r"$\rho(t)$ — Mean Ancestry Overlap (Chain Correlation)",
                  fontsize=11)
    ax2.set_xlabel("Timestep", fontsize=10)
    ax2.set_ylabel(r"$\rho(t)$", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{FIGURE_DIR}/fig8_kernel_evolution.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# -----------------------------------------------------------------------
# Figure 9: Summary table figure
# -----------------------------------------------------------------------

def fig9_summary_table(all_data: dict) -> None:
    """
    A clean summary table figure of all key metrics across scenarios.
    Suitable for inclusion directly in the paper.
    """
    rows = []
    for scenario in SCENARIOS:
        d   = all_data[scenario]
        res = d["res_summary"]
        rows.append([
            SCENARIO_LABELS[scenario],
            f"{d['tau']}",
            f"{d['kappa'].max():.0f}",
            f"{d['rho'].max():.3f}",
            f"{res['gap_lex']['total_gap_pct']:.1f}%",
            f"{res['metrics']['R_regret']['max_regret']:.0f}",
            f"{res['gap_lex']['n_correlated_pairs']}",
        ])

    col_labels = [
        "Scenario",
        r"$\tau_{\mathrm{int}}$",
        r"$\kappa_{\max}$",
        r"$\rho_{\max}$",
        "Netting Gap",
        "Min MaxRegret",
        "Corr. Pairs",
    ]

    fig, ax = plt.subplots(figsize=(13, 3))
    ax.axis('off')
    fig.suptitle(
        "Simulation Summary: Key Metrics Across Crisis Scenarios",
        fontsize=13, fontweight='bold', y=0.98
    )

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.2)

    # Header styling
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight='bold')

    # Row styling
    row_bg = ["#d6eaf8", "#fdebd0", "#d5f5e3"]
    for i, color in enumerate(row_bg):
        for j in range(len(col_labels)):
            table[(i+1, j)].set_facecolor(color)

    plt.tight_layout()
    fname = f"{FIGURE_DIR}/fig9_summary_table.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print("Generating all paper figures")
    print(f"{'='*60}\n")

    # Load all data once
    all_data = {s: load_all(s) for s in SCENARIOS}

    print("Generating figures...")
    fig1_early_warning(all_data)
    fig2_phi_decomposition(all_data)
    fig3_intervention_window(all_data)
    fig4_netting_underestimation(all_data)
    fig5_selection_rule_divergence(all_data)
    fig6_regime_classification(all_data)
    fig7_resolution_outcomes(all_data)
    fig8_kernel_evolution(all_data)
    fig9_summary_table(all_data)

    print(f"\n{'='*60}")
    print(f"All 9 figures saved to: {FIGURE_DIR}/")
    print(f"{'='*60}")
    print("""
Figure index:
  fig1  Two-layer early warning system (PRIMARY)
  fig2  Phi decomposition — diagnostic limitation
  fig3  Intervention window comparison (KEY RESULT)
  fig4  Netting underestimation — Theorem 2.1
  fig5  Selection rule divergence — monitoring mode
  fig6  Regime classification
  fig7  Resolution outcomes — shortfall distribution
  fig8  Kernel evolution — interaction geometry
  fig9  Summary table
""")
