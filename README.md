# Claim State Dynamics and Crisis-Triggered Resolution in Rehypothecation Networks
## Simulation Package — README

---

## Overview

This package implements the companion simulation for:

> **Claim State Dynamics and Crisis-Triggered Resolution in Rehypothecation Networks:
> An Endogenous Geometry Approach** — Blessing Honmane (2026)

---

## Quick Start

```bash
cd code/
python network_generator.py   # Step 1: generate synthetic network
python dynamics.py            # Step 2: run T=100 timestep co-evolution
python monitoring.py          # Step 3: compute Phi, Delta, tau_intervention
python resolution.py          # Step 4: execute clearing operator at crisis fixed point
python plots.py               # Step 5: generate all 9 paper figures
```

Figures saved to `simulation_results/paper_figures/`.

---

## Simulation Setup

### Network Parameters

| Parameter | Value | Description |
|---|---|---|
| Institutions | 10 | Financial institutions |
| Assets | 5 | Underlying collateral assets |
| Claims | 30 | Total claims (6 per asset) |
| Jurisdictions | 3 | Legal jurisdictions |
| Overclaim ratio | 2.5× | Sum of claims / total asset value |
| Chain length (max) | 2 | Maximum provenance path hops |
| Reuse probability | 0.25 | Per-institution rehypothecation probability |
| Timesteps | 100 | Simulation horizon (T = 100, so T−1 = 99) |
| Random seed | 42 | All three scenarios share identical base network |

> **Design note — shared topology.** All three scenarios are initialised from
> the **same base network** (same ρ_ij, same nominals, same claim vectors).
> Only dynamics parameters differ. The netting gap figure is therefore
> identical across all three scenarios as a *direct consequence of the shared
> seed*, not as independent confirmation of a topology-independent result.

---

## Three Crisis Scenarios

| Scenario | μ path | Legal shock | tau_intervention |
|---|---|---|---|
| Economic Drift | 0.02 → ~1.2 at rate 0.012/step | None | **+68** steps |
| Legal Regime Shock | Stable at 0.02 | t=50: 80% cross-jurisdiction pairs removed | **−23** steps |
| Compound | 0.02 → ~1.2 | At t=50 | **−6** steps |

Collateral shock δ_A = 0.25 applied to asset A₀ at t=35 in all scenarios.

> **τ_shock = −23 depends on a fallback convention.** In the legal shock
> scenario Φ never crosses its warning threshold during T=100. The code
> assigns t_Φ* = T−1 = 99 as a fallback sentinel. The reported value
> τ = 76 − 99 = −23 depends on this convention; the horizon-independent
> conclusion is that Φ provides *no advance warning* (τ < 0) for this
> crisis type.

---

## Key Implementation Specifications

### Interaction Kernel G_econ (Proxy 2)

```python
coupling = rho_matrix[i,j] * np.log1p(mu) / np.log1p(1.0)   # ÷ log(2) ≈ 0.693
coupling = np.clip(coupling, 0, 0.3)                          # hard cap
```

Actual formula applied in simulation:

    G_econ_ij(t) = min( ρ_ij(t) · log(1 + μ(t)) / log(2),  0.3 )

The division by log(2) normalises entries to [0, 1] for μ ≤ 1. The hard cap
at 0.3 prevents saturation before T=100. **Neither constant appears in
Equation (8) of the paper** — both are implementation constraints disclosed in
Remark 6.4.

### Evolution Function F (Open Problem 1)

Two-part update, both required to reproduce Figure 1 trajectories exactly:

    # Claim-specific drift (per-claim block):
    delta_i = ALPHA * rho_i_bar * G_block_i @ C_i * DELTA_T

    # Cross-claim coupling (reduced weight):
    cross_delta = ALPHA * 0.1 * G_econ @ C_flat * DELTA_T

    C_new[i] = C[i] + delta_i + cross_delta[i] + noise_i

where `rho_i_bar = mean_j(rho_ij)` is the per-claim connectivity weight.
ALPHA = 0.002, DELTA_T = 1, noise ~ N(0, 0.005).

The 10%-weighted cross-claim term is not in Equation (50) of the paper;
it is disclosed in Remark 6.3 of the updated LaTeX.

### Nominal Erosion

At each timestep, nominal claim values erode continuously:

    w_i(t+1) = w_i(t) × (1 − 0.001 × mean_encumbrance(t))

This drives the slowly declining L(t) visible in Figure 2 between shock
events. Disclosed in Remark rem:nominal_erosion.

### Selection Rule Divergence Δ(t) — Formal Definition (not IQR proxy)

```python
d_lex_regret  = ||r_lex − r_regret||
d_lex_hist    = ||r_lex − r_hist||
d_regret_hist = ||r_regret − r_hist||
Delta(t)      = max(d_lex_regret, d_lex_hist, d_regret_hist) / V_A_total
```

**This is the formal definition from Definition 3.26**, not the IQR proxy
described in earlier drafts. The IQR proxy (`Δ = 1 / (1 + IQR(π_legal) / 0.03)`)
was used in earlier versions and remains in the paper as a closed-form
approximation, labelled as superseded in Remark rem:delta_proxy.

### Priority Score Weights (Open Problem 2)

```python
pi_i = +0.05*s + 0.03*j_prime + 0.02*tau
      − 0.45*e − 0.30*(1−p)  − 0.15*(1−l)
```

where `j_prime = j * mean_admissibility_fraction`. These weights are
**hardcoded unvalidated proxies** not derived from the formal theory.
They are required to reproduce Figure 7. See Table rem:priority_weights
in the paper for full disclosure.

### Fragility Observable Φ

    Phi(t) = ||G_econ(t)||_F × L(t) × (1 + ||dG_econ/dt||_F)

Frobenius norm is used in place of κ(G) — the condition number is
numerically ill-conditioned for dense near-uniform coupling matrices.

> **Note on Figure 2.** The panel labelled "κ(G_econ)" plots the true
> 2-norm condition number. Φ itself uses ‖G_econ‖_F. These are different
> signals — see Remark rem:phi_proxy in the paper.

### τ_intervention Parameters

- Baseline window: t ∈ [1, 25)
- Threshold: mean + 2σ above baseline (k = 2.0, symmetric for both Φ and Δ)
- Scan starts at t = 25
- Δ smoothed with rolling window w = 7 before threshold comparison

---

## Results Summary

| Scenario | τ | t_Φ* | t_Δ* | Netting Gap |
|---|---|---|---|---|
| Economic Drift | **+68** | 31 | 99 | ~47% |
| Legal Regime Shock | **−23** | 99 (fallback, never crossed) | 76 | ~47% |
| Compound | **−6** | 31 | 25 | ~47% |

**Netting gap note.** The ~47% figure is the allocation-distance between
R_lex and proportional allocation (R_net). It is identical across scenarios
because all three share the same base topology by design. It is not a direct
measure of Theorem 5.8 (which concerns expected systemic loss under joint
shocks). A control with ρ_ij = 0 also produces ~47%, confirming the gap is
primarily structural to the 2.5× overclaim ratio, not ancestry-specific.

---

## G_legal: Exact Role in This Simulation

`G_legal` is an (n × n) boolean admissibility mask with **exactly two roles**:

1. **Priority scoring:** effective enforceability j_i is scaled by mean
   admissibility `a_bar_i = (1/n) × Σ_j G_legal[i,j]`, giving
   `j_prime_i = j_i × a_bar_i`.

2. **Legal shock event:** at t=50 (legal shock / compound), G_legal
   restructures discretely — 80% of cross-jurisdiction admissible pairs
   become inadmissible — instantly reordering π_i scores and causing Δ(t)
   to jump with no preceding Φ warning.

**G_legal does NOT suppress entries of G_econ.** G_econ is built entirely
from ancestry overlap ρ_ij and log-scaled margin stress (see formula above).

---

## Figure Index

| Figure | File | What it shows |
|---|---|---|
| Fig 1 | `fig1_early_warning_system.png` | Φ(t) and Δ(t) across all three scenarios. Green = τ_intervention window |
| Fig 2 | `fig2_phi_decomposition.png` | Φ decomposed into κ (condition number), L(t), ‖dG/dt‖; Φ uses Frobenius norm (different from κ panel) |
| Fig 3 | `fig3_intervention_window.png` | τ bar chart: positive = time to act, negative = structurally unavoidable late detection |
| Fig 4 | `fig4_netting_underestimation.png` | Per-claimant gap r_lex − r_net (allocation-distance measure, not Thm 5.8 loss measure) |
| Fig 5 | `fig5_selection_rule_divergence.png` | Δ(t) and mean allocations per selection rule across all scenarios |
| Fig 6 | `fig6_regime_classification.png` | Regime over time: green=stable, orange=transitional, red=critical |
| Fig 7 | `fig7_resolution_outcomes.png` | Shortfall distributions at crisis fixed point (requires priority weights in Table rem:priority_weights) |
| Fig 8 | `fig8_kernel_evolution.png` | ‖G_econ‖_F and ρ(t) — kernel evolves independently of legal scaffold |
| Fig 9 | `fig9_summary_table.png` | Summary table of key metrics across all three scenarios |

---

## File Structure

```
rehypothecation_paper/
├── README.md
├── code/
│   ├── network_generator.py   # Step 1
│   ├── dynamics.py            # Step 2
│   ├── monitoring.py          # Step 3
│   ├── resolution.py          # Step 4
│   └── plots.py               # Step 5
├── latex/
│   ├── layered_claims_framework.tex
│   └── layered_claims_framework.pdf
├── figures/paper_figures/     # All 9 figures (PNG)
└── simulation_results/        # JSON summaries + figures per scenario
```

## Dependencies

```
numpy >= 1.24   scipy >= 1.10   matplotlib >= 3.7
```
`pip install numpy scipy matplotlib`

## Reproducibility

Seed 42 reset before each scenario generation. Running the five scripts in
order produces identical figures and summaries on every run.

**Figures reproducible from paper text alone:**
- Fig 3 (τ values), Fig 4 (netting gap), Fig 6 (regime classification) ✓

**Figures requiring information from code (not fully specified in paper text):**
- Fig 1 (exact trajectories): requires cross-coupling weight 0.1 and
  log(2) normalisation from Remark 6.3 / Remark 6.4 (now disclosed in LaTeX)
- Fig 7 (priority score distributions): requires weights in Table rem:priority_weights
  (now disclosed in LaTeX)
