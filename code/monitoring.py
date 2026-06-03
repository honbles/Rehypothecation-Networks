"""
monitoring.py
=============
Monitoring mode implementation for:

    A Formal Framework for Layered Asset Claims and Resolution Outcomes
    in Rehypothecation Networks
    Blessing Honmane, 2026

This script reads the dynamics time series produced by dynamics.py
and implements the monitoring mode of the clearing operator R_hat.

At every timestep it computes:
    - Delta(t): selection rule divergence measure
    - beta(t):  bifurcation parameter
    - Phi(t):   fragility observable (already in dynamics, reloaded here)
    - tau_intervention: regulatory intervention window

It detects the crisis fixed point transition condition:
    Delta(t) >= Delta* AND beta(t) >= beta*

And records the intervention window:
    tau_intervention = t_Delta* - t_Phi*

Paper references
----------------
Monitoring Mode          : Section 3.9.2 (subsubsection)
Selection Rule Divergence: Definition -- Delta(t)
Bifurcation Parameter    : Definition -- beta(G, Psi)
Fragility Observable     : Definition -- Phi(A,t)
Crisis Transition        : Definition -- Crisis Fixed Point Transition Condition
Intervention Window      : Definition -- Regulatory Intervention Window
Phi Limitation Remark    : Remark -- Phi collapses three drivers into one scalar
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import linprog

# -----------------------------------------------------------------------
# Thresholds — match scenario configs in network_generator.py
# -----------------------------------------------------------------------

DELTA_STAR_PCT = 0.10  # Delta threshold as fraction of total nominal claims
                       # 0.10 = rules diverge by more than 10% of total claims
BETA_STAR      = 0.5  # beta threshold for crisis transition
PHI_STAR_PCT   = 0.15 # Phi warning threshold as fraction of Phi max

# -----------------------------------------------------------------------
# Load dynamics output
# -----------------------------------------------------------------------

def load_dynamics(dynamics_dir: str, data_dir: str) -> dict:
    """Load all dynamics time series and network data."""

    d = {}
    d["ts_kappa"]  = np.load(os.path.join(dynamics_dir, "ts_kappa.npy"))
    d["ts_L"]      = np.load(os.path.join(dynamics_dir, "ts_L.npy"))
    d["ts_dG"]     = np.load(os.path.join(dynamics_dir, "ts_dG.npy"))
    d["ts_Phi"]    = np.load(os.path.join(dynamics_dir, "ts_Phi.npy"))
    d["ts_mu"]     = np.load(os.path.join(dynamics_dir, "ts_mu.npy"))
    d["ts_rho"]    = np.load(os.path.join(dynamics_dir, "ts_rho.npy"))
    d["ts_G_full"]   = np.load(os.path.join(dynamics_dir, "ts_G_econ_full.npy"))
    d["ts_C_full"]   = np.load(os.path.join(dynamics_dir, "ts_C_econ_full.npy"))   # (T, n, 3)
    d["ts_GL_full"]  = np.load(os.path.join(dynamics_dir, "ts_G_legal_full.npy"))  # (T, n, n)

    # Network data
    d["nominal"]   = np.load(os.path.join(data_dir, "nominal_amounts.npy"))
    d["V_A"]       = np.load(os.path.join(data_dir, "V_A.npy"))
    d["asset_map"] = np.load(os.path.join(data_dir, "claim_asset_map.npy"))
    d["econ_init"] = np.load(os.path.join(data_dir, "econ_vectors_init.npy"))

    with open(os.path.join(dynamics_dir, "dynamics_meta.json")) as f:
        d["meta"] = json.load(f)

    d["T"]        = d["meta"]["T"]
    d["n_claims"] = d["meta"]["n_claims"]
    d["scenario"] = d["meta"]["scenario"]
    d["cfg"]      = d["meta"]["cfg"]

    return d


# -----------------------------------------------------------------------
# Selection rules — evaluated on current state (monitoring mode)
# -----------------------------------------------------------------------

def build_Psi(nominal: np.ndarray, econ_vectors: np.ndarray) -> np.ndarray:
    """
    Build asset claim state vector Psi(A,t).
    Returns stacked weighted economic state (n_claims x 3).
    Paper: Definition -- Asset Claim State.
    """
    return nominal[:, None] * econ_vectors   # (n, 3)


def R_lex(
    Psi: np.ndarray,
    nominal: np.ndarray,
    V_A_total: float,
    pi_scores: np.ndarray
) -> np.ndarray:
    """
    Lexicographic priority selection rule R_lex.

    Pure strict waterfall: allocate V_A to claimants in descending
    pi_scores order, each receiving up to their nominal claim w_i.
    The aggregate state Psi(A,t) is not used directly here — the
    clearing operator acts on individual claimant states (w_i, c_i)
    and derived priority scores, not on the compressed aggregate vector.

    Paper: Definition R_lex.
    """
    n = len(nominal)
    order = np.argsort(-pi_scores)   # descending priority

    r = np.zeros(n)
    remaining = V_A_total

    for idx in order:
        if remaining <= 0:
            break
        r[idx] = min(float(nominal[idx]), remaining)
        remaining -= r[idx]

    return r


def R_regret(
    Psi: np.ndarray,
    nominal: np.ndarray,
    V_A_total: float
) -> np.ndarray:
    """
    Minimal aggregate regret selection rule R_regret.

    Uses effective claim values from Psi (nominal * econ state norm)
    so the allocation responds to changing economic conditions.

        min max_i (effective_i - r_i)
        subject to sum(r_i) = V_A, 0 <= r_i <= effective_i

    Paper: Definition R_regret.
    """
    n = len(nominal)

    # R_regret uses nominal claim amounts w_i as the entitlements.
    # The clearing operator acts on individual claimant states, not
    # the compressed aggregate Psi. Psi is not used here.
    effective = nominal.copy()
    effective = np.maximum(effective, 1e-6)

    # Ensure feasibility
    if effective.sum() < V_A_total:
        effective = effective * (V_A_total / effective.sum() * 1.01)

    c = np.zeros(n + 1)
    c[-1] = 1.0

    A_ub = np.zeros((n, n + 1))
    for i in range(n):
        A_ub[i, i]  = -1.0
        A_ub[i, -1] = -1.0
    b_ub = -effective

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([V_A_total])

    bounds = [(0, float(effective[i])) for i in range(n)] + [(0, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                     A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        return np.clip(result.x[:n], 0, nominal)
    else:
        return nominal * (V_A_total / nominal.sum())



def R_hist(
    Psi: np.ndarray,
    nominal: np.ndarray,
    V_A_total: float,
    r_hist: np.ndarray
) -> np.ndarray:
    """
    Path-dependent historical precedent selection rule R_hist.

        min ||r - r_hist||
        subject to sum(r_i) = V_A, 0 <= r_i <= w_i

    Projection of r_hist onto the feasible allocation simplex.
    Paper: Definition R_hist.
    """
    n = len(nominal)

    if r_hist is None:
        # No historical precedent: fall back to proportional
        return nominal * (V_A_total / nominal.sum())

    # Project r_hist onto {sum(r_i)=V_A, 0<=r_i<=w_i}
    # Simple iterative projection
    r = np.clip(r_hist, 0, nominal)
    for _ in range(200):
        # Project onto budget constraint
        r = r + (V_A_total - r.sum()) / n
        # Project onto box
        r = np.clip(r, 0, nominal)
        if abs(r.sum() - V_A_total) < 1e-6:
            break

    # Rescale to exactly satisfy budget
    if r.sum() > 0:
        r = r * V_A_total / r.sum()

    return np.clip(r, 0, nominal)


# -----------------------------------------------------------------------
# Priority scoring function f
# -----------------------------------------------------------------------

def compute_pi_scores(
    claim_vectors_full: np.ndarray,
    econ_vectors: np.ndarray,
    G_legal: np.ndarray = None,
) -> np.ndarray:
    """
    Compute local priority scores pi_i.

    G_legal gates neighbourhood construction: when G_legal changes
    (legal regime shock), the effective jurisdiction enforceability
    of cross-jurisdiction claims is reduced, directly impacting pi_i
    and therefore Delta(t).

    This is the mechanism through which legal shocks propagate into
    resolution sensitivity — consistent with Remark 3.10 of the paper.
    """
    alpha_s   = 0.05
    alpha_j   = 0.03
    alpha_tau = 0.02
    alpha_e   = 0.45
    alpha_p   = 0.30
    alpha_l   = 0.15

    s   = claim_vectors_full[:, 0]
    j   = claim_vectors_full[:, 3].copy()   # jurisdiction enforceability
    tau = claim_vectors_full[:, 5]
    p   = econ_vectors[:, 0]
    e   = econ_vectors[:, 1]
    l   = econ_vectors[:, 2]

    # Apply G_legal: for each claimant i, compute the fraction of
    # other claimants it is legally admissible to interact with.
    # A legal shock reducing admissibility degrades effective j_i.
    if G_legal is not None:
        n = len(j)
        for i in range(n):
            admissible_fraction = G_legal[i].sum() / max(n, 1)
            j[i] = j[i] * admissible_fraction   # scale down j by legal reach

    pi = (alpha_s   * s
        + alpha_j   * j
        + alpha_tau * tau
        - alpha_e   * e
        - alpha_p   * (1.0 - p)
        - alpha_l   * (1.0 - l))

    return pi


# -----------------------------------------------------------------------
# Compute Delta(t) — selection rule divergence measure
# -----------------------------------------------------------------------

def compute_Delta(
    r_lex: np.ndarray,
    r_regret: np.ndarray,
    r_hist: np.ndarray,
    V_A_total: float
) -> float:
    """
    Delta(t) = max_{j != k} || R_hat_j(Psi(t)) - R_hat_k(Psi(t)) || / V_A_total

    This is the formal definition from the paper (Definition 3.20), now
    implemented directly. The three selection rule outputs are already
    computed at each timestep; this function simply takes their maximum
    pairwise L2 distance, normalised by V_A_total so Delta is dimensionless.

    Properties:
    - Regime 1: all three rules agree -> Delta near zero
    - Regime 2/3: rules diverge -> Delta grows
    - Legal shock: G_legal restructuring changes pi scores -> R_lex ordering
      shifts -> divergence from R_hist (anchored at t=0) and R_regret jumps
    - Economic drift: encumbrance grows -> pi scores compress -> R_lex
      ordering becomes unstable -> distance from priority-blind R_hist grows

    This replaces the IQR proxy used in earlier versions. The formal
    definition is directly computable since all three selection rules are
    evaluated at every timestep.

    Paper: Definition 3.20 -- Selection Rule Divergence Measure.
    """
    d_lex_regret  = np.linalg.norm(r_lex    - r_regret)
    d_lex_hist    = np.linalg.norm(r_lex    - r_hist)
    d_regret_hist = np.linalg.norm(r_regret - r_hist)
    return float(max(d_lex_regret, d_lex_hist, d_regret_hist)) / (V_A_total + 1e-10)



# -----------------------------------------------------------------------
# Compute beta(t) — bifurcation parameter
# -----------------------------------------------------------------------

def compute_beta(
    kappa: float,
    Delta_current: float,
    Delta_prev: float,
    epsilon: float = 1.0
) -> float:
    """
    beta(G, Psi) = kappa(G) * sup_{||dPsi||<=eps} (|R(Psi+dPsi)| - |R(Psi)|) / eps

    Simulation approximation: the sensitivity of solution set cardinality
    is proxied by the finite difference of Delta(t) with respect to time
    (since Delta itself measures how multi-valued the solution set is).

        beta(t) ≈ kappa(t) * |Delta(t) - Delta(t-1)| / epsilon

    Paper: Definition -- Bifurcation Parameter.
    Note: |R(Psi)| (cardinality) is proxied by Delta(t) normalised to [0,1].
    """
    sensitivity = abs(Delta_current - Delta_prev) / epsilon
    # Log-scale kappa to prevent extreme spikes dominating beta
    kappa_log = float(np.log1p(kappa))
    return float(kappa_log * sensitivity)


# -----------------------------------------------------------------------
# Compute intervention window tau_intervention
# -----------------------------------------------------------------------

def compute_tau_intervention(
    ts_Phi: np.ndarray,
    ts_Delta: np.ndarray,
    nominal_total: float,
    baseline_end: int = 25,
    k_phi: float = 2.0,
    k_delta: float = 2.0,
) -> dict:
    """
    tau_intervention = t_Delta* - t_Phi*

    Asymmetric k values reflect different signal structures:
    - k=2.0 symmetric for both Phi and Delta: threshold = mean + 2*std
      of the baseline period. Same multiplier for both signals for
      consistency. Earlier design used asymmetric k_delta=1; corrected.

    Paper: Definition -- Regulatory Intervention Window.
    """
    T = len(ts_Phi)

    phi_base   = ts_Phi[1:baseline_end]
    delta_base = ts_Delta[1:baseline_end]

    phi_thresh   = phi_base.mean()   + k_phi   * (phi_base.std()   + 1e-10)
    delta_thresh = delta_base.mean() + k_delta * (delta_base.std() + 1e-10)

    t_phi   = None
    t_delta = None

    # Start scan AFTER baseline period — avoids early transient noise
    scan_start = baseline_end

    for t in range(scan_start, T):
        if t_phi   is None and ts_Phi[t]   >= phi_thresh:
            t_phi = t
        if t_delta is None and ts_Delta[t] >= delta_thresh:
            t_delta = t

    if t_phi   is None: t_phi   = T - 1
    if t_delta is None: t_delta = T - 1

    tau = t_delta - t_phi

    return {
        "t_phi_star":        t_phi,
        "t_delta_star":      t_delta,
        "tau_intervention":  tau,
        "phi_threshold":     float(phi_thresh),
        "delta_threshold":   float(delta_thresh),
        "phi_at_t_phi":      float(ts_Phi[t_phi]),
        "delta_at_t_delta":  float(ts_Delta[t_delta]),
        "transition_detected": bool(t_delta > t_phi),
    }


# -----------------------------------------------------------------------
# Detect crisis fixed point transition
# -----------------------------------------------------------------------

def detect_crisis_transition(
    ts_Delta: np.ndarray,
    ts_beta: np.ndarray,
    nominal_total: float,
) -> dict:
    """
    Detect when Delta(t) >= DELTA_STAR_PCT * nominal_total
    AND beta(t) >= BETA_STAR simultaneously.
    Paper: Definition -- Crisis Fixed Point Transition Condition.
    """
    T           = len(ts_Delta)
    delta_star  = DELTA_STAR_PCT * nominal_total

    for t in range(1, T):
        if ts_Delta[t] >= delta_star and ts_beta[t] >= BETA_STAR:
            return {
                "detected":            True,
                "t_transition":        t,
                "Delta_at_transition": float(ts_Delta[t]),
                "beta_at_transition":  float(ts_beta[t]),
            }
    return {
        "detected":            False,
        "t_transition":        T - 1,
        "Delta_at_transition": float(ts_Delta[-1]),
        "beta_at_transition":  float(ts_beta[-1]),
    }


# -----------------------------------------------------------------------
# Main monitoring loop
# -----------------------------------------------------------------------

def run_monitoring(
    dynamics_dir: str,
    data_dir: str,
    output_dir: str
) -> dict:
    """Run full monitoring pass over dynamics time series."""

    d = load_dynamics(dynamics_dir, data_dir)
    T        = d["T"]
    n        = d["n_claims"]
    scenario = d["scenario"]
    cfg      = d["cfg"]

    print(f"\n{'='*60}")
    print(f"Running monitoring: {scenario}")
    print(f"  T={T} steps, n={n} claims")
    print(f"{'='*60}")

    # Load static claim vectors (legal dimensions don't change)
    # We reload from data_dir
    claim_vectors_full = np.load(
        os.path.join(data_dir, "claim_vectors_init.npy"))   # (n, 6)
    nominal   = d["nominal"]
    V_A       = d["V_A"]
    asset_map = d["asset_map"]

    # Compute V_A total (sum across all assets)
    V_A_total = float(V_A.sum())

    # Time series storage
    ts_Delta = np.zeros(T)
    ts_beta  = np.zeros(T)
    ts_r_lex    = []
    ts_r_regret = []
    ts_r_hist   = []

    # Historical precedent: frozen at t=0 lex allocation forever.
    # This anchors R_hist to the initial state so Delta measures
    # how far the current allocation has drifted from the starting point.
    # As the system evolves, R_lex changes but R_hist stays fixed —
    # Delta grows monotonically with drift, which is what we want.
    # R_hist anchor: proportional allocation (completely priority-blind).
    # This creates maximum structural divergence from R_lex (pure priority)
    # and R_regret (equitable-but-Psi-weighted). Delta measures how much
    # the priority-based and equity-based rules disagree with blind proportion.
    r_hist_anchor = nominal * (V_A_total / nominal.sum())   # fixed, never updated

    # Load real per-timestep C_econ and G_legal saved by dynamics.py
    ts_C_full  = d["ts_C_full"]   # (T, n, 3)
    ts_GL_full = d["ts_GL_full"]  # (T, n, n) boolean

    for t in range(T):
        # Use real econ vectors from dynamics — not interpolated
        econ_t   = ts_C_full[t]    # (n, 3)
        G_legal_t = ts_GL_full[t]  # (n, n) — captures exact moment of legal shock

        # Compute priority scores — G_legal gates neighbourhood construction
        pi = compute_pi_scores(claim_vectors_full, econ_t, G_legal_t)

        # Build Psi(t)
        Psi_t = build_Psi(nominal, econ_t)

        # Evaluate three selection rules (kept for resolution plots)
        r_lex_t    = R_lex(Psi_t, nominal, V_A_total, pi)
        r_regret_t = R_regret(Psi_t, nominal, V_A_total)
        r_hist_t   = R_hist(Psi_t, nominal, V_A_total, r_hist_anchor)

        # Compute Delta(t) — formal Definition 3.20
        # Maximum pairwise L2 distance between the three selection rule outputs
        Delta_t = compute_Delta(r_lex_t, r_regret_t, r_hist_t, V_A_total)

        # Compute beta(t)
        if t == 0:
            beta_t = 0.0
        else:
            beta_t = compute_beta(
                d["ts_kappa"][t], Delta_t, ts_Delta[t-1]
            )

        ts_Delta[t] = Delta_t
        ts_beta[t]  = beta_t
        ts_r_lex.append(r_lex_t)
        ts_r_regret.append(r_regret_t)
        ts_r_hist.append(r_hist_t)

    # Smooth Delta with rolling mean (window=7) to reduce noise
    ts_Delta_smooth = np.convolve(ts_Delta, np.ones(7)/7, mode='same')
    ts_Delta_smooth[:3]  = ts_Delta[:3]
    ts_Delta_smooth[-3:] = ts_Delta[-3:]

    # Compute tau_intervention using smoothed Delta, baseline_end=25
    tau_info = compute_tau_intervention(
        d["ts_Phi"], ts_Delta_smooth, V_A_total, baseline_end=25
    )

    # Detect crisis fixed point transition
    transition = detect_crisis_transition(
        ts_Delta, ts_beta, V_A_total
    )

    print(f"\n  Monitoring results:")
    print(f"    Delta(T-1)         = {ts_Delta[-1]:.4f}")
    print(f"    beta(T-1)          = {ts_beta[-1]:.4f}")
    print(f"    t_Phi*             = {tau_info['t_phi_star']}")
    print(f"    t_Delta*           = {tau_info['t_delta_star']}")
    print(f"    tau_intervention   = {tau_info['tau_intervention']} steps")
    print(f"    Crisis transition  = {transition['detected']}"
          f" (t={transition['t_transition']})")

    results = {
        "scenario":       scenario,
        "T":              T,
        "n_claims":       n,
        "ts_Delta":       ts_Delta,
        "ts_beta":        ts_beta,
        "ts_Phi":         d["ts_Phi"],
        "ts_kappa":       d["ts_kappa"],
        "ts_L":           d["ts_L"],
        "ts_dG":          d["ts_dG"],
        "ts_rho":         d["ts_rho"],
        "ts_r_lex":       np.array(ts_r_lex),
        "ts_r_regret":    np.array(ts_r_regret),
        "ts_r_hist":      np.array(ts_r_hist),
        "tau_info":       tau_info,
        "transition":     transition,
        "nominal":        nominal,
        "V_A_total":      V_A_total,
        "cfg":            cfg,
    }

    return results


# -----------------------------------------------------------------------
# Save monitoring output
# -----------------------------------------------------------------------

def save_monitoring(results: dict, output_dir: str) -> None:
    """Save monitoring time series for use by resolution.py and plots.py."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "ts_Delta.npy"),    results["ts_Delta"])
    np.save(os.path.join(output_dir, "ts_beta.npy"),     results["ts_beta"])
    np.save(os.path.join(output_dir, "ts_r_lex.npy"),    results["ts_r_lex"])
    np.save(os.path.join(output_dir, "ts_r_regret.npy"), results["ts_r_regret"])
    np.save(os.path.join(output_dir, "ts_r_hist.npy"),   results["ts_r_hist"])

    summary = {
        "scenario":          results["scenario"],
        "tau_info":          results["tau_info"],
        "transition":        results["transition"],
        "Delta_star_pct":    DELTA_STAR_PCT,
        "beta_star":         BETA_STAR,
        "phi_star_pct":      PHI_STAR_PCT,
        "V_A_total":         results["V_A_total"],
        "cfg":               results["cfg"],
    }
    with open(os.path.join(output_dir, "monitoring_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Monitoring saved to: {output_dir}/")


# -----------------------------------------------------------------------
# Plot monitoring results
# -----------------------------------------------------------------------

def plot_monitoring(results: dict, output_dir: str) -> None:
    """
    Generate monitoring plots.

    Plot 1: PRIMARY PAPER PLOT — Phi(t) and Delta(t) on same axes
        Shows the intervention window tau_intervention as the gap
        between Phi rising and Delta following.
        Paper: Remark -- two-layer early warning system.

    Plot 2: Beta(t) bifurcation parameter over time
        Shows approach to crisis fixed point.
        Paper: Definition -- Bifurcation Parameter.

    Plot 3: Selection rule divergence breakdown
        Shows mean allocation from each rule over time.
        Illustrates when rules start to disagree.
    """
    T        = results["T"]
    scenario = results["scenario"]
    ts       = np.arange(T)
    tau      = results["tau_info"]
    trans    = results["transition"]
    cfg      = results["cfg"]
    shock_t  = cfg.get("shock_timestep", 35)
    legal_t  = cfg.get("legal_shock_timestep")

    os.makedirs(output_dir, exist_ok=True)

    # Normalise Phi to same scale as Delta for overlay plot
    Phi      = results["ts_Phi"]
    Delta    = results["ts_Delta"]
    Phi_norm = Phi / (Phi.max() + 1e-10)
    Delta_norm = Delta / (Delta.max() + 1e-10)

    # ----------------------------------------------------------------
    # Plot 1: PRIMARY — Phi and Delta intervention window
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle(
        f"Two-Layer Early Warning System: $\\Phi(t)$ and $\\Delta(t)$\n"
        f"Scenario: {scenario}",
        fontsize=13, fontweight='bold'
    )

    color_phi   = "#9467bd"
    color_delta = "#d62728"
    color_window = "#2ca02c"

    ax.plot(ts, Phi_norm, color=color_phi, lw=2.2,
            label=r"$\Phi(A,t)$ normalised — structural fragility")
    ax.plot(ts, Delta_norm, color=color_delta, lw=2.2,
            label=r"$\Delta(t)$ normalised — resolution sensitivity")

    # Mark t_Phi* and t_Delta*
    t_phi   = tau["t_phi_star"]
    t_delta = tau["t_delta_star"]
    tau_val = tau["tau_intervention"]

    ax.axvline(t_phi, color=color_phi, ls="--", lw=1.5,
               alpha=0.8, label=f"$t_{{\\Phi^*}}={t_phi}$")
    ax.axvline(t_delta, color=color_delta, ls="--", lw=1.5,
               alpha=0.8, label=f"$t_{{\\Delta^*}}={t_delta}$")

    # Shade intervention window
    if tau_val > 0:
        ax.axvspan(t_phi, t_delta, alpha=0.12, color=color_window,
                   label=f"$\\tau_{{\\mathrm{{intervention}}}}={tau_val}$ steps")

    # Mark crisis transition
    if trans["detected"]:
        t_trans = trans["t_transition"]
        ax.axvline(t_trans, color="black", ls=":", lw=2,
                   label=f"Crisis fixed point t={t_trans}")

    # Mark external shocks
    ax.axvline(shock_t, color="gray", ls="-.", lw=1,
               alpha=0.6, label=f"Collateral shock t={shock_t}")
    if legal_t:
        ax.axvline(legal_t, color="navy", ls="-.", lw=1.5,
                   alpha=0.7, label=f"Legal shock t={legal_t}")

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Normalised value", fontsize=12)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Annotation box
    ann_text = (
        f"$\\tau_{{\\mathrm{{intervention}}}}$ = {tau_val} steps\n"
        f"$\\Phi$ warning at t={t_phi}\n"
        f"$\\Delta$ warning at t={t_delta}"
    )
    ax.text(0.98, 0.97, ann_text,
            transform=ax.transAxes,
            fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fname1 = os.path.join(output_dir,
                          f"intervention_window_{scenario}.png")
    fig.savefig(fname1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname1}")

    # ----------------------------------------------------------------
    # Plot 2: Beta(t) — bifurcation parameter
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        f"Bifurcation Parameter $\\beta(t)$\nScenario: {scenario}",
        fontsize=13, fontweight='bold'
    )

    beta = results["ts_beta"]
    ax.plot(ts, beta, color="#ff7f0e", lw=2,
            label=r"$\beta(G,\Psi)$")
    ax.axhline(BETA_STAR, color="red", ls="--", lw=1.5,
               label=f"$\\beta^*={BETA_STAR}$ threshold")

    if trans["detected"]:
        t_trans = trans["t_transition"]
        ax.axvline(t_trans, color="black", ls=":", lw=2,
                   label=f"Crisis transition t={t_trans}")

    ax.axvline(shock_t, color="gray", ls="-.", lw=1, alpha=0.6,
               label=f"Collateral shock t={shock_t}")
    if legal_t:
        ax.axvline(legal_t, color="navy", ls="-.", lw=1.5, alpha=0.7,
                   label=f"Legal shock t={legal_t}")

    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel(r"$\beta(t)$", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.fill_between(ts, beta, 0, where=(beta >= BETA_STAR),
                    alpha=0.2, color="#ff7f0e",
                    label="Above threshold")

    fname2 = os.path.join(output_dir, f"beta_{scenario}.png")
    fig.savefig(fname2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname2}")

    # ----------------------------------------------------------------
    # Plot 3: Selection rule mean allocations over time
    # ----------------------------------------------------------------
    mean_lex    = results["ts_r_lex"].mean(axis=1)
    mean_regret = results["ts_r_regret"].mean(axis=1)
    mean_hist   = results["ts_r_hist"].mean(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        f"Selection Rule Mean Allocations Over Time\n"
        f"Scenario: {scenario}",
        fontsize=13, fontweight='bold'
    )

    ax.plot(ts, mean_lex,    color="#1f77b4", lw=2,
            label=r"$\hat{R}_{\mathrm{lex}}$ — lexicographic priority")
    ax.plot(ts, mean_regret, color="#d62728", lw=2, ls="--",
            label=r"$\hat{R}_{\mathrm{regret}}$ — minimal regret")
    ax.plot(ts, mean_hist,   color="#2ca02c", lw=2, ls=":",
            label=r"$\hat{R}_{\mathrm{hist}}$ — historical precedent")

    # Shade divergence region
    divergence = np.abs(mean_lex - mean_regret)
    ax.fill_between(ts, mean_lex, mean_regret,
                    alpha=0.12, color="#9467bd",
                    label="Lex vs Regret divergence")

    if trans["detected"]:
        t_trans = trans["t_transition"]
        ax.axvline(t_trans, color="black", ls=":", lw=2,
                   label=f"Crisis fixed point t={t_trans}")

    ax.axvline(shock_t, color="gray", ls="-.", lw=1, alpha=0.6)
    if legal_t:
        ax.axvline(legal_t, color="navy", ls="-.", lw=1.5, alpha=0.7)

    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Mean allocation per claim", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fname3 = os.path.join(output_dir,
                          f"selection_rules_{scenario}.png")
    fig.savefig(fname3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname3}")

    # ----------------------------------------------------------------
    # Plot 4: Regime classification over time
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(
        f"Regime Classification Over Time\nScenario: {scenario}",
        fontsize=13, fontweight='bold'
    )

    # Classify each timestep into regime 1, 2, or 3
    regimes = np.ones(T, dtype=int)   # default Regime 1
    for t in range(T):
        delta_star_abs = DELTA_STAR_PCT * results["nominal"].sum()
        if results["ts_beta"][t] > BETA_STAR:
            regimes[t] = 3
        elif results["ts_Delta"][t] > delta_star_abs * 0.5:
            regimes[t] = 2

    colors = {1: "#2ca02c", 2: "#ff7f0e", 3: "#d62728"}
    labels = {1: "Regime 1 — Stable",
              2: "Regime 2 — Transitional",
              3: "Regime 3 — Critical"}

    for regime_id, color in colors.items():
        mask = regimes == regime_id
        ax.fill_between(ts, 0, 1, where=mask,
                        alpha=0.4, color=color,
                        label=labels[regime_id],
                        transform=ax.get_xaxis_transform())

    ax.plot(ts, Delta_norm, color="#d62728", lw=1.5,
            label=r"$\Delta(t)$ normalised")
    ax.plot(ts, Phi_norm * 0.8, color="#9467bd", lw=1.5, ls="--",
            label=r"$\Phi(t)$ normalised (scaled)")

    ax.axvline(shock_t, color="gray", ls="-.", lw=1, alpha=0.6)
    if legal_t:
        ax.axvline(legal_t, color="navy", ls="-.", lw=1.5, alpha=0.7)

    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Regime / Signal", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2)

    fname4 = os.path.join(output_dir, f"regime_classification_{scenario}.png")
    fig.savefig(fname4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname4}")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":

    from scipy.optimize import linprog   # verify scipy available

    data_base     = "simulation_data"
    dynamics_base = "simulation_results"
    output_base   = "simulation_results"

    tau_summary = {}

    for scenario in ["economic_drift", "legal_shock", "compound"]:
        data_dir     = os.path.join(data_base, scenario)
        dynamics_dir = os.path.join(dynamics_base, scenario, "dynamics")
        output_dir   = os.path.join(output_base, scenario, "monitoring")

        results = run_monitoring(data_dir=data_dir,
                                 dynamics_dir=dynamics_dir,
                                 output_dir=output_dir)
        save_monitoring(results, output_dir)
        plot_monitoring(results, output_dir)

        tau_summary[scenario] = {
            "tau_intervention":  results["tau_info"]["tau_intervention"],
            "t_phi_star":        results["tau_info"]["t_phi_star"],
            "t_delta_star":      results["tau_info"]["t_delta_star"],
            "crisis_detected":   results["transition"]["detected"],
            "t_crisis":          results["transition"]["t_transition"],
        }

        # Copy primary plot to figures folder
        src = os.path.join(output_dir,
                           f"intervention_window_{scenario}.png")
        dst = os.path.join("simulation_results", "..",
                           "rehypothecation_paper", "figures",
                           f"intervention_window_{scenario}.png")
        try:
            import shutil
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
        except Exception:
            pass

    print(f"\n{'='*60}")
    print("INTERVENTION WINDOW SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scenario':<20} {'tau':>6} {'t_Phi*':>8} "
          f"{'t_Delta*':>10} {'Crisis':>8}")
    print("-" * 60)
    for sc, info in tau_summary.items():
        print(f"{sc:<20} {info['tau_intervention']:>6} "
              f"{info['t_phi_star']:>8} "
              f"{info['t_delta_star']:>10} "
              f"{str(info['crisis_detected']):>8}")

    print(f"\n{'='*60}")
    print("Monitoring complete for all three scenarios.")
    print(f"{'='*60}")
    print("""
Next script:
  resolution.py -- executes R_hat at crisis fixed point
                   compares all three rules vs classical netting
""")
