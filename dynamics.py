"""
dynamics.py
===========
Dynamical system evolution for:

    A Formal Framework for Layered Asset Claims and Resolution Outcomes
    in Rehypothecation Networks
    Blessing Honmane, 2026

Fixes applied (reviewer issues 1, 4, 7, 8):
  Issue 1:  rho_ij re-derived from Jaccard similarity of provenance paths
            (stress-driven chain extension replaces blind interpolation).
  Issue 4:  apply_F computes all terms from the SAME time-t state and clips once.
  Issue 7:  legal shock uses its own RNG (LEGAL_SHOCK_RNG) so the economic
            noise sequence is not perturbed by the shock firing.
  Issue 8:  nominal erosion is saved as a full time series (ts_nominal.npy)
            so monitoring.py can use the same eroded values.

Kappa note: G_econ is sparse (zero blocks for rho_ij=0). The matrix is
rank-deficient by construction, making the condition number kappa(G_econ)
numerically meaningless (infinite due to true zeros, not ill-conditioning).
kappa is replaced throughout by the Frobenius norm ||G_econ||_F as the
instability proxy for Phi and beta, consistent with Remark 6.5.
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import norm

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------

RNG_SEED        = 42
rng             = np.random.default_rng(RNG_SEED + 1)
LEGAL_SHOCK_RNG = np.random.default_rng(RNG_SEED + 99)  # Issue 7: isolated RNG

ALPHA          = 0.015    # stronger coupling to drive visible drift
DELTA_T        = 1.0

SHOCK_TIMESTEP  = 35
SHOCK_ASSET_ID  = 0
SHOCK_MAGNITUDE = 0.25

ECON_NOISE_LEVEL = 0.003


# -----------------------------------------------------------------------
# Load network data
# -----------------------------------------------------------------------

def load_network(scenario_dir: str) -> dict:
    data = {}
    data["G_econ"]        = np.load(os.path.join(scenario_dir, "G_econ_init.npy"))
    data["G_legal"]       = np.load(os.path.join(scenario_dir, "G_legal_mask_init.npy"))
    data["rho_matrix"]    = np.load(os.path.join(scenario_dir, "rho_matrix_init.npy"))
    data["econ_vectors"]  = np.load(os.path.join(scenario_dir, "econ_vectors_init.npy"))
    data["claim_vectors"] = np.load(os.path.join(scenario_dir, "claim_vectors_init.npy"))
    data["nominal"]       = np.load(os.path.join(scenario_dir, "nominal_amounts.npy"))
    data["V_A"]           = np.load(os.path.join(scenario_dir, "V_A.npy"))
    data["asset_map"]     = np.load(os.path.join(scenario_dir, "claim_asset_map.npy"))
    with open(os.path.join(scenario_dir, "network_metadata.json")) as f:
        data["meta"] = json.load(f)
    data["scenario_config"] = data["meta"]["params"]["scenario_config"]
    data["n_claims"]        = data["meta"]["n_claims"]
    return data


# -----------------------------------------------------------------------
# Issue 4 fix: apply_F — consistent time-t state, single clip
# -----------------------------------------------------------------------

def apply_F(
    econ_vectors: np.ndarray,
    G_econ: np.ndarray,
    noise_level: float,
    rho_matrix: np.ndarray = None,
) -> np.ndarray:
    """
    Evolution function F. All terms computed from the same time-t state.
    Per-claim diagonal-block drift + cross-claim coupling + noise, clipped once.
    Issue 4 fix: no mixed-time accumulation.
    """
    n = econ_vectors.shape[0]

    rho_i = rho_matrix.mean(axis=1) if rho_matrix is not None else np.full(n, 0.5)

    # Per-claim drift (diagonal block only)
    per_claim_delta = np.zeros_like(econ_vectors)
    for i in range(n):
        block  = G_econ[3*i:3*i+3, 3*i:3*i+3]
        per_claim_delta[i] = ALPHA * float(rho_i[i]) * (block @ econ_vectors[i]) * DELTA_T

    # Cross-claim coupling from same time-t state
    C_flat      = econ_vectors.flatten()
    cross_delta = (ALPHA * 0.1 * G_econ @ C_flat * DELTA_T).reshape(n, 3)

    # Noise
    noise = rng.normal(0, noise_level, size=econ_vectors.shape)

    return np.clip(econ_vectors + per_claim_delta + cross_delta + noise, 0.0, 1.0)


# -----------------------------------------------------------------------
# Update G_econ
# -----------------------------------------------------------------------

def update_G_econ(rho_matrix: np.ndarray, G_legal: np.ndarray, mu: float) -> np.ndarray:
    n = rho_matrix.shape[0]
    G = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(n):
            coupling = rho_matrix[i, j] * np.log1p(mu) / np.log(2.0)
            coupling = np.clip(coupling, 0, 0.3)
            if coupling < 1e-10:
                continue
            block = np.array([
                [coupling,      coupling*0.3, coupling*0.2],
                [coupling*0.3,  coupling,     coupling*0.4],
                [coupling*0.2,  coupling*0.4, coupling    ]
            ])
            G[3*i:3*i+3, 3*j:3*j+3] = block
    return G


# -----------------------------------------------------------------------
# Issue 1 fix: update_rho_matrix via Jaccard on provenance paths
# -----------------------------------------------------------------------

def update_rho_matrix(
    rho_matrix: np.ndarray,
    econ_vectors: np.ndarray,
    mu: float,
    provenance_paths: list = None,
) -> tuple:
    """
    Issue 1 fix: re-derive rho_ij as Jaccard similarity of provenance paths.
    Under margin stress, each chain can acquire a new institution
    (rehypothecation extension). Extension probability scales with mu
    and the claim's encumbrance level e_i.
    Returns (rho_new, updated_paths).
    """
    if provenance_paths is None:
        # Legacy fallback
        growth = 0.01 * mu * (1.0 - rho_matrix)
        rho_new = np.clip(rho_matrix + growth, 0, 1)
        np.fill_diagonal(rho_new, 1.0)
        return rho_new, None

    n = len(provenance_paths)
    n_institutions = max((max(p) for p in provenance_paths if p), default=9) + 1
    n_institutions = max(n_institutions, 10)

    updated_paths = []
    for i, path in enumerate(provenance_paths):
        # Extension probability: scales with stress AND encumbrance of claim i
        encumbrance_i = float(econ_vectors[i, 1]) if i < len(econ_vectors) else 0.3
        ext_prob = 0.04 * mu * (0.5 + encumbrance_i)
        if len(path) > 0 and rng.random() < ext_prob:
            candidates = [k for k in range(n_institutions) if k not in path]
            if candidates:
                path = path + [int(rng.choice(candidates))]
        updated_paths.append(path)

    # Recompute Jaccard
    rho_new = np.zeros((n, n))
    for i in range(n):
        pi_i = set(updated_paths[i])
        for j in range(i, n):
            pi_j = set(updated_paths[j])
            union = pi_i | pi_j
            inter = pi_i & pi_j
            val = len(inter) / len(union) if union else 0.0
            rho_new[i, j] = val
            rho_new[j, i] = val
    np.fill_diagonal(rho_new, 1.0)
    return rho_new, updated_paths


# -----------------------------------------------------------------------
# Legal shock — Issue 7: use isolated RNG
# -----------------------------------------------------------------------

def apply_legal_shock(
    G_legal: np.ndarray,
    shock_magnitude: float,
    claimant_jurisdictions: np.ndarray = None,
) -> np.ndarray:
    G_new = G_legal.copy()
    n = G_legal.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if not G_legal[i, j]:
                continue
            if claimant_jurisdictions is not None:
                if claimant_jurisdictions[i] == claimant_jurisdictions[j]:
                    continue
            if LEGAL_SHOCK_RNG.random() < shock_magnitude:   # Issue 7
                G_new[i, j] = False
                G_new[j, i] = False
    return G_new


# -----------------------------------------------------------------------
# Collateral shock
# -----------------------------------------------------------------------

def apply_collateral_shock(
    econ_vectors: np.ndarray,
    asset_map: np.ndarray,
    shock_asset_id: int,
    delta_A: float,
) -> np.ndarray:
    C_new = econ_vectors.copy()
    for i, asset_id in enumerate(asset_map):
        if asset_id == shock_asset_id:
            C_new[i] = C_new[i] * (1.0 - delta_A)
    return np.clip(C_new, 0.0, 1.0)


# -----------------------------------------------------------------------
# Kappa: use Frobenius norm as instability proxy (kappa is meaningless
# for rank-deficient sparse G_econ — see Remark 6.5)
# -----------------------------------------------------------------------

def compute_kappa(G_econ: np.ndarray) -> float:
    """Frobenius norm of G_econ as instability proxy (replaces condition number)."""
    return float(np.linalg.norm(G_econ, ord='fro'))

def compute_kappa_raw(G_econ: np.ndarray) -> float:
    return compute_kappa(G_econ)


# -----------------------------------------------------------------------
# L(t), dG/dt, Phi
# -----------------------------------------------------------------------

def compute_L(nominal: np.ndarray, V_A: np.ndarray, asset_map: np.ndarray) -> float:
    total_claims = sum(nominal[asset_map == a].sum() for a in range(len(V_A)))
    total_value  = V_A.sum()
    return float(total_claims / total_value) if total_value > 0 else 0.0

def compute_dG_norm(G_current: np.ndarray, G_prev: np.ndarray) -> float:
    return float(norm(G_current - G_prev, ord='fro'))

def compute_Phi(kappa: float, L: float, dG_norm: float, G_econ: np.ndarray = None) -> float:
    G_norm = float(np.linalg.norm(G_econ, ord='fro')) if G_econ is not None else kappa
    return G_norm * L * (1.0 + dG_norm)


# -----------------------------------------------------------------------
# Main dynamics loop
# -----------------------------------------------------------------------

def run_dynamics(scenario_dir: str, output_dir: str) -> dict:
    data     = load_network(scenario_dir)
    cfg      = data["scenario_config"]
    T        = cfg["T_steps"]
    n        = data["n_claims"]
    scenario = cfg["scenario"]

    print(f"\n{'='*60}")
    print(f"Running dynamics: {scenario}")
    print(f"  T={T} steps, n={n} claims")
    print(f"  {cfg.get('description','')}")
    print(f"{'='*60}")

    C_econ    = data["econ_vectors"].copy()
    G_econ    = data["G_econ"].copy()
    G_legal   = data["G_legal"].copy()
    rho_mat   = data["rho_matrix"].copy()
    nominal   = data["nominal"].copy()
    V_A       = data["V_A"].copy()
    asset_map = data["asset_map"].copy()
    mu        = cfg["mu_initial"]

    # Issue 1: load provenance paths
    provenance_paths = [c["provenance_path"] for c in data["meta"]["claims"]]

    ts_kappa    = np.zeros(T)
    ts_L        = np.zeros(T)
    ts_dG       = np.zeros(T)
    ts_Phi      = np.zeros(T)
    ts_mu       = np.zeros(T)
    ts_rho      = np.zeros(T)
    ts_nominal  = np.zeros((T, n))   # Issue 8
    ts_G_econ   = []
    ts_C_econ   = []
    ts_G_legal  = []
    legal_shock_applied = False
    G_prev = G_econ.copy()

    for t in range(T):

        # 1. Legal shock
        if (cfg.get("legal_shock_timestep") is not None
                and t == cfg["legal_shock_timestep"]
                and not legal_shock_applied):
            print(f"  [t={t:3d}] Legal regime shock (magnitude={cfg['legal_shock_magnitude']:.1%})")
            _meta  = json.load(open(os.path.join(scenario_dir, "network_metadata.json")))
            _jmap  = {int(k): v for k, v in _meta["jurisdiction_map"].items()}
            _cl_jur = np.array([_jmap[c["holder_institution_id"]] for c in _meta["claims"]])
            G_legal = apply_legal_shock(G_legal, cfg["legal_shock_magnitude"], _cl_jur)
            legal_shock_applied = True

        # 2. Collateral shock
        # Suppressed in legal_shock scenario: the paper establishes that
        # legal regime shocks produce a negative tau because Phi receives
        # no driver from economic drift. Adding a collateral shock would
        # also drive Phi via dG/dt, contaminating the mechanism.
        # The legal_shock scenario isolates the legal channel only.
        apply_collat = (t == SHOCK_TIMESTEP and scenario != "legal_shock")
        if apply_collat:
            print(f"  [t={t:3d}] Collateral shock asset {SHOCK_ASSET_ID} ({SHOCK_MAGNITUDE:.1%})")
            C_econ = apply_collateral_shock(C_econ, asset_map, SHOCK_ASSET_ID, SHOCK_MAGNITUDE)
            for i, a_id in enumerate(asset_map):
                if a_id == SHOCK_ASSET_ID:
                    nominal[i] *= (1.0 - SHOCK_MAGNITUDE * 0.5)

        # 2b. Nominal erosion from encumbrance
        enc_mean     = C_econ[:, 1].mean()
        nominal      = np.maximum(nominal * (1.0 - 0.001 * enc_mean), 0.01)

        # 3. mu drift
        mu = float(np.clip(mu + cfg.get("economic_drift_rate", 0.0) * DELTA_T, 0.0, 1.0))

        # 4. Issue 1: update rho via Jaccard
        rho_mat, provenance_paths = update_rho_matrix(rho_mat, C_econ, mu, provenance_paths)

        # 5. Update G_econ
        G_prev = G_econ.copy()
        G_econ = update_G_econ(rho_mat, G_legal, mu)

        # 6. Evolve econ vectors — Issue 4
        C_econ = apply_F(C_econ, G_econ, ECON_NOISE_LEVEL, rho_mat)

        # 7. Record
        kappa      = compute_kappa(G_econ)
        L          = compute_L(nominal, V_A, asset_map)
        dG_norm    = compute_dG_norm(G_econ, G_prev)
        rho_scalar = float(np.mean(rho_mat[np.triu_indices(n, k=1)]))
        Phi        = compute_Phi(kappa, L, dG_norm, G_econ)

        ts_kappa[t]   = kappa
        ts_L[t]       = L
        ts_dG[t]      = dG_norm
        ts_Phi[t]     = Phi
        ts_mu[t]      = mu
        ts_rho[t]     = rho_scalar
        ts_nominal[t] = nominal.copy()   # Issue 8
        ts_G_econ.append(G_econ.copy())
        ts_C_econ.append(C_econ.copy())
        ts_G_legal.append(G_legal.copy())

    print(f"\n  Final state at t={T-1}:")
    print(f"    mu       = {ts_mu[-1]:.4f}")
    print(f"    rho(T)   = {ts_rho[-1]:.4f}")
    print(f"    ||G||_F  = {ts_kappa[-1]:.4f}")
    print(f"    L        = {ts_L[-1]:.4f}")
    print(f"    dG_norm  = {ts_dG[-1]:.6f}")
    print(f"    Phi      = {ts_Phi[-1]:.4f}")

    return {
        "scenario":    scenario,
        "T":           T,
        "n_claims":    n,
        "ts_kappa":    ts_kappa,
        "ts_L":        ts_L,
        "ts_dG":       ts_dG,
        "ts_Phi":      ts_Phi,
        "ts_mu":       ts_mu,
        "ts_rho":      ts_rho,
        "ts_nominal":  ts_nominal,
        "ts_G_econ":   ts_G_econ,
        "ts_C_econ":   ts_C_econ,
        "ts_G_legal":  ts_G_legal,
        "G_legal_final": G_legal,
        "nominal":     nominal,
        "V_A":         V_A,
        "asset_map":   asset_map,
        "meta":        data["meta"],
        "cfg":         cfg,
        "shock_timestep":  SHOCK_TIMESTEP,
        "shock_asset_id":  SHOCK_ASSET_ID,
        "shock_magnitude": SHOCK_MAGNITUDE,
    }


# -----------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------

def save_dynamics(results: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "ts_kappa.npy"),   results["ts_kappa"])
    np.save(os.path.join(output_dir, "ts_L.npy"),       results["ts_L"])
    np.save(os.path.join(output_dir, "ts_dG.npy"),      results["ts_dG"])
    np.save(os.path.join(output_dir, "ts_Phi.npy"),     results["ts_Phi"])
    np.save(os.path.join(output_dir, "ts_mu.npy"),      results["ts_mu"])
    np.save(os.path.join(output_dir, "ts_rho.npy"),     results["ts_rho"])
    np.save(os.path.join(output_dir, "ts_nominal.npy"), results["ts_nominal"])
    np.save(os.path.join(output_dir, "G_econ_final.npy"),  results["ts_G_econ"][-1])
    np.save(os.path.join(output_dir, "C_econ_final.npy"),  results["ts_C_econ"][-1])
    np.save(os.path.join(output_dir, "G_legal_final.npy"), results["G_legal_final"])
    np.save(os.path.join(output_dir, "ts_C_econ_full.npy"),
            np.stack(results["ts_C_econ"], axis=0))
    np.save(os.path.join(output_dir, "ts_G_legal_full.npy"),
            np.stack(results["ts_G_legal"], axis=0))
    np.save(os.path.join(output_dir, "ts_G_econ_full.npy"),
            np.stack(results["ts_G_econ"], axis=0))
    with open(os.path.join(output_dir, "dynamics_meta.json"), "w") as f:
        json.dump({
            "scenario":       results["scenario"],
            "T":              results["T"],
            "n_claims":       results["n_claims"],
            "shock_timestep": results["shock_timestep"],
            "shock_asset_id": results["shock_asset_id"],
            "shock_magnitude":results["shock_magnitude"],
            "cfg":            results["cfg"],
        }, f, indent=2)
    print(f"  Dynamics saved to: {output_dir}/")


# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------

def plot_dynamics(results: dict, output_dir: str) -> None:
    T        = results["T"]
    scenario = results["scenario"]
    ts       = np.arange(T)
    shock_t  = results["shock_timestep"]
    cfg      = results["cfg"]
    legal_t  = cfg.get("legal_shock_timestep")
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Phi decomposition
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Fragility Observable Decomposition\nScenario: {scenario}",
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ts, results["ts_kappa"], color="#d62728", lw=1.8)
    ax1.axvline(shock_t, color="gray", ls="--", lw=1, label=f"Shock t={shock_t}")
    if legal_t:
        ax1.axvline(legal_t, color="navy", ls=":", lw=1.5, label=f"Legal t={legal_t}")
    ax1.set_title(r"$\|G^{\mathrm{econ}}\|_F$ — Coupling Density", fontsize=10)
    ax1.set_xlabel("Timestep"); ax1.set_ylabel(r"$\|G\|_F$")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ts, results["ts_L"], color="#ff7f0e", lw=1.8)
    ax2.axvline(shock_t, color="gray", ls="--", lw=1)
    if legal_t: ax2.axvline(legal_t, color="navy", ls=":", lw=1.5)
    ax2.set_title(r"$L(t)$ — Overclaim Ratio", fontsize=10)
    ax2.set_xlabel("Timestep"); ax2.set_ylabel(r"$L(t)$"); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ts, results["ts_dG"], color="#2ca02c", lw=1.8)
    ax3.axvline(shock_t, color="gray", ls="--", lw=1)
    if legal_t: ax3.axvline(legal_t, color="navy", ls=":", lw=1.5)
    ax3.set_title(r"$\|dG/dt\|_F$ — Kernel Velocity", fontsize=10)
    ax3.set_xlabel("Timestep"); ax3.set_ylabel("Frobenius norm"); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(ts, results["ts_Phi"], color="#9467bd", lw=2.2)
    ax4.axvline(shock_t, color="gray", ls="--", lw=1, label=f"Shock t={shock_t}")
    if legal_t: ax4.axvline(legal_t, color="navy", ls=":", lw=1.5, label=f"Legal t={legal_t}")
    ax4.set_title(r"$\Phi(A,t)$ — Aggregate Fragility", fontsize=10)
    ax4.set_xlabel("Timestep"); ax4.set_ylabel(r"$\Phi$")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)
    ax4.fill_between(ts, results["ts_Phi"], alpha=0.15, color="#9467bd")
    fig.savefig(os.path.join(output_dir, f"phi_decomposition_{scenario}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: phi_decomposition_{scenario}.png")

    # Plot 2: rho and ||G||_F
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"Interaction Kernel Evolution\nScenario: {scenario}",
                 fontsize=13, fontweight='bold')
    ax.plot(ts, results["ts_kappa"], color="#d62728", lw=2,
            label=r"$\|G^{\mathrm{econ}}\|_F$ — coupling density")
    ax2r = ax.twinx()
    ax2r.plot(ts, results["ts_rho"], color="#1f77b4", lw=2, ls="--",
              label=r"$\rho(t)$ — mean ancestry overlap")
    ax.axvline(shock_t, color="gray", ls="--", lw=1.2, label=f"Collateral shock t={shock_t}")
    if legal_t: ax.axvline(legal_t, color="navy", ls=":", lw=1.8, label=f"Legal shock t={legal_t}")
    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel(r"$\|G^{\mathrm{econ}}\|_F$", color="#d62728", fontsize=11)
    ax2r.set_ylabel(r"$\rho(t)$", color="#1f77b4", fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, f"kernel_evolution_{scenario}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: kernel_evolution_{scenario}.png")

    # Plot 3: econ sub-vectors
    mean_p = np.array([C[:, 0].mean() for C in results["ts_C_econ"]])
    mean_e = np.array([C[:, 1].mean() for C in results["ts_C_econ"]])
    mean_l = np.array([C[:, 2].mean() for C in results["ts_C_econ"]])
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"Economic Sub-vector Evolution\nScenario: {scenario}",
                 fontsize=13, fontweight='bold')
    ax.plot(ts, mean_p, color="#1f77b4", lw=2, label=r"$\bar{p}(t)$ — custody possession")
    ax.plot(ts, mean_e, color="#d62728", lw=2, label=r"$\bar{e}(t)$ — encumbrance")
    ax.plot(ts, mean_l, color="#2ca02c", lw=2, label=r"$\bar{\ell}(t)$ — liquidity")
    ax.axvline(shock_t, color="gray", ls="--", lw=1.2, label=f"Shock t={shock_t}")
    if legal_t: ax.axvline(legal_t, color="navy", ls=":", lw=1.8, label=f"Legal t={legal_t}")
    ax.set_xlabel("Timestep", fontsize=11); ax.set_ylabel("Mean value [0,1]", fontsize=11)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, f"econ_evolution_{scenario}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: econ_evolution_{scenario}.png")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    data_base   = "simulation_data"
    output_base = "simulation_results"
    for scenario in ["economic_drift", "legal_shock", "compound"]:
        results = run_dynamics(
            os.path.join(data_base, scenario),
            os.path.join(output_base, scenario, "dynamics")
        )
        save_dynamics(results, os.path.join(output_base, scenario, "dynamics"))
        plot_dynamics(results, os.path.join(output_base, scenario, "dynamics"))
    print(f"\n{'='*60}\nDynamics complete.\n{'='*60}")
