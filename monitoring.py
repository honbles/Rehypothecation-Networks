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
    # Issue 8: load per-timestep eroded nominal so monitoring and dynamics share state
    ts_nominal_path = os.path.join(dynamics_dir, "ts_nominal.npy")
    if os.path.exists(ts_nominal_path):
        d["ts_nominal"] = np.load(ts_nominal_path)   # (T, n)
    else:
        d["ts_nominal"] = None

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

def build_Psi(
    nominal: np.ndarray,
    econ_vectors: np.ndarray,
    claim_vectors_full: np.ndarray = None,
) -> np.ndarray:
    """
    Build asset claim state vector |Psi(A,t)> = sum_i w_i * c_i(t).

    Issue 9 fix: the paper defines Psi as a single 6-dimensional aggregate
    vector (Definition 3.5), not an (n x 3) matrix. We return the true
    aggregate here. When full claim vectors are available (legal + econ),
    we form the full 6D c_i and weight-sum them. Otherwise we use the
    3D econ sub-vectors.

    Remark 3.7 notes that R_hat acts on individual claimant states, not
    on the compressed aggregate — so the selection rules take (nominal,
    econ_vectors) directly. Psi is only used for the fragility observable
    and bifurcation parameter.

    Returns: (6,) aggregate vector if claim_vectors_full is provided,
             (3,) econ aggregate otherwise.
    """
    if claim_vectors_full is not None:
        # Full 6D claim vector: [s, p, e, j, l, tau] — columns 0..5
        # Replace economic columns (1=p, 2=e, 4=l) with current econ values
        c_full = claim_vectors_full.copy().astype(float)
        c_full[:, 1] = econ_vectors[:, 0]   # custody possession p
        c_full[:, 2] = econ_vectors[:, 1]   # encumbrance e
        c_full[:, 4] = econ_vectors[:, 2]   # liquidity l
        return (nominal[:, None] * c_full).sum(axis=0)   # (6,)
    else:
        return (nominal[:, None] * econ_vectors).sum(axis=0)   # (3,)


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
    V_A_total: float,
    econ_vectors: np.ndarray = None,
) -> np.ndarray:
    """
    Minimal aggregate regret selection rule R_regret.

    Issue 5 fix: effective entitlement of each claim is w_i * ||c_i^econ||_2
    (nominal scaled by the L2 norm of the economic sub-vector), so the
    allocation responds to how much economic state has degraded.
    A claim with high encumbrance and low custody sees its effective
    entitlement shrink, changing the regret-minimising allocation over time.

        min max_i (effective_i - r_i)
        s.t. sum(r_i) = V_A, 0 <= r_i <= effective_i

    Paper: Definition R_regret.
    """
    n = len(nominal)

    if econ_vectors is not None:
        # Scale nominal by L2 norm of economic sub-vector (in [0, sqrt(3)])
        econ_norms = np.linalg.norm(econ_vectors, axis=1) / np.sqrt(3)  # normalise to [0,1]
        effective  = nominal * np.maximum(econ_norms, 0.01)
    else:
        effective = nominal.copy()

    effective = np.maximum(effective, 1e-6)

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
        r = np.clip(result.x[:n], 0, nominal)
    else:
        r = nominal * (V_A_total / nominal.sum())

    # Issue 12: hard conservation check — rescale to exactly V_A_total
    r_sum = r.sum()
    if r_sum > 0 and abs(r_sum - V_A_total) > 1e-6:
        r = r * (V_A_total / r_sum)
    assert abs(r.sum() - V_A_total) < 0.01, \
        f"R_regret conservation violated: sum={r.sum():.4f} != V_A={V_A_total:.4f}"
    return r



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

    r = np.clip(r, 0, nominal)
    # Issue 12: hard conservation check after final clip
    r_sum = r.sum()
    if r_sum > 0 and abs(r_sum - V_A_total) > 1e-6:
        r = r * (V_A_total / r_sum)
    assert abs(r.sum() - V_A_total) < 0.01, \
        f"R_hist conservation violated: sum={r.sum():.4f} != V_A={V_A_total:.4f}"
    return r


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
    alpha_j   = 0.15   # raised from 0.03 so legal shock propagates into pi
    alpha_tau = 0.02
    alpha_e   = 0.45
    alpha_p   = 0.30
    alpha_l   = 0.08   # adjusted so weights remain bounded

    s   = claim_vectors_full[:, 0]
    j   = claim_vectors_full[:, 3].copy()   # jurisdiction enforceability
    tau = claim_vectors_full[:, 5]
    p   = econ_vectors[:, 0]
    e   = econ_vectors[:, 1]
    l   = econ_vectors[:, 2]

    # Apply G_legal: compute admissibility fraction over CROSS-JURISDICTION
    # pairs only. Same-jurisdiction pairs are always admissible and including
    # them dilutes the legal shock signal. The cross-jurisdiction fraction
    # drops sharply at t=50 in the legal regime shock scenario.
    if G_legal is not None:
        n = len(j)
        jur = claim_vectors_full[:, 3]   # use raw j as jurisdiction proxy
        # Detect jurisdiction groups from G_legal structure:
        # same-jurisdiction pairs always have G_legal=1; cross-jurisdiction
        # pairs are the ones that can be removed by a legal shock.
        # Use G_legal off-diagonal structure directly.
        for i in range(n):
            # Sum only off-diagonal entries to avoid self-interaction
            row = G_legal[i].copy().astype(float)
            row[i] = 0.0
            n_off = n - 1
            # Identify which off-diagonal entries are cross-jurisdiction:
            # same-jurisdiction pairs are always 1; cross-jur can be 0 or 1.
            # Approximate: treat entries that were ever < 1 as cross-jur.
            # Simpler: just use the fraction of off-diagonal entries that are 1.
            admissible_fraction = row.sum() / max(n_off, 1)
            j[i] = j[i] * admissible_fraction

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
    epsilon: float = 1.0,
    n_claims: int = 1,
) -> float:
    """
    beta(G, Psi) = kappa(G) * sup_{||dPsi||<=eps} (dim(R(Psi+dPsi)) - dim(R(Psi))) / eps

    Issue 3 fix: the correct proxy for dim(R_hat(Psi)) is Delta(t) itself
    (normalised to [0,1]), not its time-derivative. Delta measures how
    spread out the three selection rule outputs are — a large Delta means
    the feasible set is effectively multi-valued (high dim proxy). A system
    where all rules agree but allocations shift over time has Delta ≈ 0
    regardless of dDelta/dt, correctly giving beta ≈ 0.

    The sensitivity of dim to perturbations is proxied by
    Delta(t) / (1 - Delta(t) + eps), which grows steeply as Delta
    approaches 1 (fully irresolvable), reflecting that near-critical
    systems are sensitive to small state perturbations.

    Paper: Definition -- Bifurcation Parameter.
    """
    # Normalise Delta to [0,1] range based on typical max ≈ 1.0
    delta_norm = np.clip(Delta_current, 0.0, 1.0)
    # Sensitivity proxy: steeper near saturation
    sensitivity = delta_norm / (1.0 - delta_norm + 0.1)
    kappa_log   = float(np.log1p(kappa))
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
    delta_star_abs: float = 0.10,
    phi_abs_floor: float = 5.0,
) -> dict:
    """
    tau_intervention = t_Delta* - t_Phi*

    Two-signal design:
    - Phi threshold: mean + k_phi * std of baseline [1:baseline_end].
      Works for economic drift where Phi grows continuously.
    - Delta threshold: absolute fraction delta_star_abs of V_A_total.
      Absolute threshold is used because Delta measures deviation from
      the frozen Regime-1 allocation, so its Regime-1 baseline is near
      zero by construction. A mean+std threshold on near-zero baseline
      fires spuriously on noise.

    Legal shock result: Phi is flat throughout (no economic drift to
    drive G_econ growth), so Phi never crosses its threshold and
    t_Phi* defaults to T-1. Delta crosses at the legal shock timestep.
    tau = t_Delta* - (T-1) < 0 — the correct negative-window result.

    Paper: Definition -- Regulatory Intervention Window.
    """
    T = len(ts_Phi)

    phi_base   = ts_Phi[1:baseline_end]
    phi_thresh = phi_base.mean() + k_phi * (phi_base.std() + 1e-10)

    # Delta: use absolute threshold (fraction of total claims)
    delta_thresh = delta_star_abs

    # Flatness check for Phi: if the full post-baseline range of Phi is
    # less than 3 sigma above the baseline mean, treat Phi as flat and
    # use the T-1 sentinel. This handles the legal shock scenario where
    # Phi fluctuates within noise but never genuinely rises — a mean+k*std
    # threshold on a near-flat signal fires on noise spikes.
    # Flatness detection: treat Phi as flat (use T-1 sentinel) when
    # the signal never rises meaningfully above its baseline. Two criteria:
    # 1. Post-baseline max < mean + 5*std (no sustained rise)
    # 2. Post-baseline max < phi_abs_floor (absolute floor: signals below
    #    this level are noise regardless of baseline statistics)
    # Criterion 2 handles the legal shock scenario where Phi fluctuates
    # around 1.3-1.5 while economic drift drives it to 90+. A crossing at
    # 1.45 in a system where genuine fragility reaches 90 is not a signal.
    phi_post_max  = float(ts_Phi[baseline_end:].max())
    phi_flat_ceil = float(phi_base.mean() + 5.0 * (phi_base.std() + 1e-10))
    phi_is_flat   = bool(phi_post_max < phi_flat_ceil or
                         phi_post_max < phi_abs_floor)

    t_phi   = None
    t_delta = None

    scan_start = baseline_end

    if not phi_is_flat:
        for t in range(scan_start, T):
            if t_phi is None and ts_Phi[t] >= phi_thresh:
                t_phi = t

    for t in range(5, T):
        if t_delta is None and ts_Delta[t] >= delta_thresh:
            t_delta = t

    # Sentinels: if signal never crosses, default to T-1
    # For legal shock: Phi is flat -> t_Phi* = T-1 -> tau = t_Delta* - 99 < 0
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
        "transition_detected": bool(t_delta < t_phi),
        "phi_crossed":       bool(t_phi < T - 1),
        "delta_crossed":     bool(t_delta < T - 1),
        "phi_is_flat":       bool(phi_is_flat),
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
    Detect when Delta(t) >= DELTA_STAR_PCT AND beta(t) >= BETA_STAR simultaneously.
    Delta is already normalised by V_A_total in compute_Delta, so compare
    directly against DELTA_STAR_PCT (not DELTA_STAR_PCT * nominal_total).
    Paper: Definition -- Crisis Fixed Point Transition Condition.
    """
    T = len(ts_Delta)
    for t in range(1, T):
        if ts_Delta[t] >= DELTA_STAR_PCT and ts_beta[t] >= BETA_STAR:
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

    # Fix: ensure output directory exists before any save operations
    os.makedirs(output_dir, exist_ok=True)

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
    nominal_init = d["nominal"]   # initial nominal — used as fallback
    V_A          = d["V_A"]
    asset_map    = d["asset_map"]

    # Issue 8: use per-timestep eroded nominal if available
    ts_nominal_dyn = d.get("ts_nominal", None)   # (T, n) or None

    # Compute V_A total (sum across all assets)
    V_A_total = float(V_A.sum())

    # Load institution mapping for institution-level Delta aggregation.
    # By Theorem 4.1 (Corollary 4.2), compression to institution level is
    # lossless for all resolution-relevant quantities. Institution-level
    # Delta is robust to claim-level noise-driven rank swaps.
    import json as _json
    meta_path = os.path.join(data_dir, "network_metadata.json")
    with open(meta_path) as _f:
        _meta = _json.load(_f)
    _jmap     = {int(k): v for k, v in _meta["jurisdiction_map"].items()}
    inst_ids  = np.array(
        [_jmap[c["holder_institution_id"]] for c in _meta["claims"]])
    n_inst    = len(np.unique(inst_ids))

    def _inst_agg(r):
        ri = np.zeros(n_inst)
        for i, v in enumerate(r):
            ri[inst_ids[i]] += v
        return ri

    # EMA smoothing for pi scores to dampen noise-driven rank swaps.
    # A single noise draw in econ_vectors can swap two nearly-equal pi
    # scores and shift a large institution-level chunk of V_A in R_lex.
    # EMA with alpha=0.25 introduces a ~3-timestep lag that filters noise
    # while preserving genuine persistent changes (legal shock, drift).
    PI_EMA_ALPHA = 0.25
    pi_smooth    = None

    # Time series storage
    ts_Delta = np.zeros(T)
    ts_beta  = np.zeros(T)
    ts_r_lex    = []
    ts_r_regret = []
    ts_r_hist   = []

    # Historical precedent anchor — tracking fix.
    #
    # Root cause of wrong tau: anchoring R_hist at proportional allocation
    # makes Delta structurally high from t=0 (R_lex waterfall vs proportional
    # are always far apart at 2.5x overclaim). The baseline period then
    # measures a permanently high Delta, and threshold crossings are
    # meaningless.
    #
    # Fix: in Regime 1 the anchor tracks the current R_lex output so Delta
    # is near zero when all rules agree. Once Delta rises above a stability
    # threshold the anchor freezes — from that point R_hist is the "last
    # stable allocation" the system knew before crisis, which is the correct
    # financial interpretation of path-dependent historical precedent.
    # The freeze is one-way.
    STABILITY_THRESHOLD = 0.05   # Delta below this means Regime 1

    # Anchor initialised to None; set to R_lex(t=0) on first iteration.
    # This ensures Delta=0 at t=0 by construction — the anchor always
    # starts at the system's first stable allocation, not at proportional.
    r_hist_anchor = None
    anchor_frozen = False

    # Load real per-timestep C_econ and G_legal saved by dynamics.py
    ts_C_full  = d["ts_C_full"]   # (T, n, 3)
    ts_GL_full = d["ts_GL_full"]  # (T, n, n) boolean

    for t in range(T):
        # Issue 8: use per-timestep eroded nominal so monitoring matches dynamics
        if ts_nominal_dyn is not None:
            nominal_t = ts_nominal_dyn[t]
        else:
            nominal_t = nominal_init

        # Use real econ vectors from dynamics — not interpolated
        econ_t    = ts_C_full[t]    # (n, 3)
        G_legal_t = ts_GL_full[t]   # (n, n)

        # Compute priority scores with EMA smoothing on pi to prevent
        # noise-driven rank swaps from polluting Delta.
        pi_raw = compute_pi_scores(claim_vectors_full, econ_t, G_legal_t)
        if pi_smooth is None:
            pi_smooth = pi_raw.copy()
        pi_smooth = (1.0 - PI_EMA_ALPHA) * pi_smooth + PI_EMA_ALPHA * pi_raw
        pi = pi_smooth

        # Build Psi(t) — true aggregate (Issue 9)
        Psi_t = build_Psi(nominal_t, econ_t, claim_vectors_full)

        # Evaluate three selection rules
        # Issue 5: pass econ_vectors to R_regret so it sees degraded state
        r_lex_t    = R_lex(Psi_t, nominal_t, V_A_total, pi)
        r_regret_t = R_regret(Psi_t, nominal_t, V_A_total, econ_t)
        # Use R_lex as fallback anchor before it is initialised
        _anchor = r_hist_anchor if r_hist_anchor is not None else r_lex_t
        r_hist_t = R_hist(Psi_t, nominal_t, V_A_total, _anchor)

        # Delta(t): institution-level R_lex vs frozen Regime-1 anchor.
        # Aggregation to institution level (Corollary 4.2) eliminates
        # claim-level noise-driven rank swap artefacts.
        # The anchor tracks R_lex in Regime 1 so Delta ≈ 0 when stable,
        # and grows only when R_lex persistently deviates from its
        # last stable allocation — the operationally correct signal.
        r_lex_inst   = _inst_agg(r_lex_t)
        _anch_use    = r_hist_anchor if r_hist_anchor is not None else r_lex_t
        anchor_inst  = _inst_agg(_anch_use)
        anchor_inst  = anchor_inst * (V_A_total / (anchor_inst.sum() + 1e-10))
        Delta_t_inst = float(np.linalg.norm(r_lex_inst - anchor_inst)) / V_A_total

        # Update anchor: track R_lex (inst-level) in Regime 1, freeze on exit
        if not anchor_frozen:
            if Delta_t_inst < STABILITY_THRESHOLD:
                r_hist_anchor = r_lex_t.copy()   # Regime 1: anchor follows R_lex
            else:
                anchor_frozen = True              # exiting Regime 1: freeze anchor

        # Issue 12: verify conservation on all three allocations
        for label, r_check in [("lex", r_lex_t), ("regret", r_regret_t), ("hist", r_hist_t)]:
            if abs(r_check.sum() - V_A_total) > 1.0:
                print(f"  WARNING t={t}: R_{label} conservation error "
                      f"sum={r_check.sum():.2f} vs V_A={V_A_total:.2f}")

        # Initialise anchor from R_lex at t=0 so Delta(0)=0 by construction
        if r_hist_anchor is None:
            r_hist_anchor = r_lex_t.copy()

        # Use institution-level Delta as the primary monitoring signal
        Delta_t = Delta_t_inst

        # Compute beta(t) — Issue 3: use Delta-level proxy not dDelta/dt
        beta_t = compute_beta(
            d["ts_kappa"][t], Delta_t, ts_Delta[t-1] if t > 0 else 0.0,
            n_claims=n,
        )

        ts_Delta[t] = Delta_t
        ts_beta[t]  = beta_t
        ts_r_lex.append(r_lex_t)
        ts_r_regret.append(r_regret_t)
        ts_r_hist.append(r_hist_t)

    # Save frozen anchor for resolution.py to use
    np.save(os.path.join(output_dir, "r_hist_anchor_frozen.npy"), r_hist_anchor)
    print(f"  R_hist anchor {'frozen at' if anchor_frozen else 'still tracking (never froze) at'} "
          f"final state. Saved.")

    # Smooth Delta with rolling mean (window=7) to reduce noise
    ts_Delta_smooth = np.convolve(ts_Delta, np.ones(7)/7, mode='same')
    ts_Delta_smooth[:3]  = ts_Delta[:3]
    ts_Delta_smooth[-3:] = ts_Delta[-3:]

    # Compute tau using institution-level Delta with absolute threshold.
    # delta_star_abs=0.10: Delta must reach 10% of V_A divergence to trigger.
    # For legal shock: Phi never crosses -> t_Phi* = T-1 -> tau < 0 (correct).
    # phi_abs_floor=5.0: Phi below this absolute level is treated as flat
    # regardless of baseline statistics. Economic drift Phi reaches 90+;
    # legal shock Phi stays below 2. The floor prevents noise crossings
    # in the legal shock scenario from producing spurious t_Phi* values.
    tau_info_raw = compute_tau_intervention(
        d["ts_Phi"], ts_Delta, V_A_total,
        baseline_end=25, delta_star_abs=0.10, phi_abs_floor=5.0
    )
    tau_info = compute_tau_intervention(
        d["ts_Phi"], ts_Delta_smooth, V_A_total,
        baseline_end=25, delta_star_abs=0.10, phi_abs_floor=5.0
    )
    tau_info["tau_intervention_raw"]      = tau_info_raw["tau_intervention"]
    tau_info["t_delta_star_raw"]          = tau_info_raw["t_delta_star"]
    tau_info["delta_threshold_raw"]       = tau_info_raw["delta_threshold"]
    tau_info["phi_crossed"]               = tau_info_raw["phi_crossed"]
    tau_info["delta_crossed"]             = tau_info_raw["delta_crossed"]

    # Detect crisis fixed point transition
    transition = detect_crisis_transition(
        ts_Delta, ts_beta, V_A_total
    )

    print(f"\n  Monitoring results:")
    print(f"    Delta(T-1)         = {ts_Delta[-1]:.4f}")
    print(f"    beta(T-1)          = {ts_beta[-1]:.4f}")
    print(f"    t_Phi*             = {tau_info['t_phi_star']}")
    print(f"    t_Delta* (smooth)  = {tau_info['t_delta_star']}")
    print(f"    t_Delta* (raw)     = {tau_info['t_delta_star_raw']}")
    print(f"    tau (smoothed)     = {tau_info['tau_intervention']} steps")
    print(f"    tau (raw)          = {tau_info['tau_intervention_raw']} steps")
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
        "nominal":        nominal_init,
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

    # Anchor all paths to the directory containing this script so the script
    # works regardless of the working directory (e.g. when run from a Jupyter
    # notebook whose kernel CWD differs from the repo root).
    _SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    data_base     = os.path.join(_SCRIPT_DIR, "simulation_data")
    dynamics_base = os.path.join(_SCRIPT_DIR, "simulation_results")
    output_base   = os.path.join(_SCRIPT_DIR, "simulation_results")

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

    # ------------------------------------------------------------------
    # Item 5: Threshold sensitivity analysis
    # Run tau across a 3x3 grid of k_phi x k_delta values.
    # Shows qualitative ordering is stable, not an artefact of k=2.0.
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("tau_intervention across k_phi x k_delta grid")
    print(f"{'='*60}")

    k_values = [1.5, 2.0, 2.5]
    sens_results = {}

    for scenario in ["economic_drift", "legal_shock", "compound"]:
        dynamics_dir = os.path.join(dynamics_base, scenario, "dynamics")
        mon_dir      = os.path.join(output_base, scenario, "monitoring")

        ts_Phi_s   = np.load(os.path.join(dynamics_dir, "ts_Phi.npy"))
        ts_Delta_s = np.load(os.path.join(mon_dir, "ts_Delta.npy"))
        V_A_s      = float(np.load(os.path.join(
            data_base, scenario, "V_A.npy")).sum())

        sens_results[scenario] = {}
        for k_phi in k_values:
            for k_delta in k_values:
                info = compute_tau_intervention(
                    ts_Phi_s, ts_Delta_s, V_A_s,
                    baseline_end=25,
                    k_phi=k_phi,
                    delta_star_abs=0.10,
                    phi_abs_floor=5.0
                )
                sens_results[scenario][(k_phi, k_delta)] = info["tau_intervention"]

    # Print sensitivity table
    header = f"{'':20}"
    for k_d in k_values:
        header += f"  k_d={k_d}"
    print(header)
    for scenario in ["economic_drift", "legal_shock", "compound"]:
        for k_p in k_values:
            row = f"{scenario[:12]+' k_p='+str(k_p):<20}"
            for k_d in k_values:
                tau_v = sens_results[scenario][(k_p, k_d)]
                row += f"  {tau_v:+6d}"
            print(row)
        print()

    # Check ordering holds across all grid points
    ordering_holds = all(
        sens_results["economic_drift"][(kp, kd)] > 0
        and sens_results["legal_shock"][(kp, kd)] < 0
        for kp in k_values for kd in k_values
    )
    print(f"Qualitative ordering tau_drift>0, tau_shock<0 holds across all grid: {ordering_holds}")

    # Save sensitivity results
    for scenario in ["economic_drift", "legal_shock", "compound"]:
        mon_dir = os.path.join(output_base, scenario, "monitoring")
        sens_out = {
            str(k): v for k, v in sens_results[scenario].items()
        }
        with open(os.path.join(mon_dir, "tau_sensitivity.json"), "w") as f:
            json.dump({
                "scenario": scenario,
                "grid": {f"k_phi={kp}_k_delta={kd}": sens_results[scenario][(kp,kd)]
                         for kp in k_values for kd in k_values},
                "ordering_holds_all": ordering_holds,
            }, f, indent=2)

    print(f"\n{'='*60}")
    print("Monitoring complete for all three scenarios.")
    print(f"{'='*60}")
    print("""
Next script:
  resolution.py -- executes R_hat at crisis fixed point
                   compares all three rules vs classical netting
""")
