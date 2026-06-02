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
| Overclaim ratio | 2.5x | Sum of claims / total asset value |
| Chain length (max) | 2 | Maximum provenance path hops |
| Reuse probability | 0.25 | Per-institution rehypothecation probability |
| Timesteps | 100 | Simulation horizon |
| Random seed | 42 | All three scenarios share identical base network |

All three scenarios are initialised from the **same base network** (same rho_ij,
same nominals, same claim vectors). Only dynamics parameters differ. This ensures
the 40.45% netting gap is a structural property of network topology, not a scenario
artefact.

---

## Three Crisis Scenarios

| Scenario | mu path | Legal shock | Prediction |
|---|---|---|---|
| Economic Drift | 0.02 → ~1.2 at rate 0.012/step | None | tau = **+68** |
| Legal Regime Shock | Stable at 0.02 | At t=50: 80% cross-jurisdiction pairs removed | tau = **-49** |
| Compound | 0.02 → ~1.2 | At t=50 | tau = **+18** |

Collateral shock delta_A = 0.25 applied to asset A0 at t=35 in all scenarios.

---

## G_legal: Exact Role in This Simulation

`G_legal` is an (n x n) boolean admissibility mask. It has **exactly two roles**:

1. **Priority scoring**: effective enforceability j_i is scaled by mean admissibility
   a_bar_i = (1/n) * sum_j G_legal[i,j], giving pi_legal_i = pi_i * a_bar_i.

2. **Legal shock event**: at t=50 (legal shock / compound), G_legal restructures
   discretely — 80% of cross-jurisdiction admissible pairs become inadmissible —
   instantly compressing pi_legal and causing Delta(t) to jump.

**G_legal does NOT zero or suppress entries of G_econ.** G_econ is built entirely
from ancestry overlap rho_ij and log-scaled margin stress:

    G_econ_ij(t) = rho_ij(t) * log(1 + mu(t))   for all (i,j) where rho_ij > 0

---

## Key Proxy Specifications

### Evolution Function F (Open Problem 1)
Claim-specific Euler update. Each claim drifts at rate proportional to its own
mean ancestry overlap rho_i_bar = mean_j(rho_ij). High-reuse claims drift faster,
creating genuine differential evolution essential for Delta sensitivity.
ALPHA = 0.002, dt = 1.

### Fragility Observable Phi (Open Problem 5)
    Phi(t) = ||G_econ(t)||_F * L(t) * (1 + ||dG_econ/dt||)

Frobenius norm used instead of condition number kappa — kappa is ill-conditioned
for dense near-uniform coupling matrices.

### Selection Rule Divergence Delta (Open Problem 6)
    Delta(t) = 1 / (1 + IQR(pi_legal(t)) / 0.03)

Inverse priority spread. When pi_legal compresses (drift) or G_legal drops
admissibility (shock), Delta rises — capturing feasible set expansion.

### Tau Intervention Window
- Baseline: t=[1,25], threshold = mean + 2*std above baseline
- Scan starts at t=25 to avoid early transient noise
- Delta smoothed with rolling window=7 before threshold comparison

---

## Results Summary

| Scenario | tau | t_Phi* | t_Delta* | Netting Gap |
|---|---|---|---|---|
| Economic Drift | **+68** | 31 | 99 | 40.45% |
| Legal Regime Shock | **-74** | 99 (never) | 25 | 40.45% |
| Compound | **+18** | 31 | 49 | 40.45% |

Netting gap is identical across all scenarios — confirms structural origin (Theorem 5.8).

---

## Figure Index

| Figure | File | What it shows |
|---|---|---|
| Fig 1 | `fig1_early_warning_system.png` | Phi(t) and Delta(t) across all three scenarios. Green = tau_intervention window |
| Fig 2 | `fig2_phi_decomposition.png` | Phi decomposed into its three drivers: kappa/Frobenius, L(t), dG/dt |
| Fig 3 | `fig3_intervention_window.png` | tau bar chart: positive = time to act, negative = unavoidable late detection |
| Fig 4 | `fig4_netting_underestimation.png` | Per-claimant gap r_lex - r_net. Red = correlated pairs. Theorem 5.8 confirmed |
| Fig 5 | `fig5_selection_rule_divergence.png` | Delta(t) and mean allocations per selection rule across all scenarios |
| Fig 6 | `fig6_regime_classification.png` | Regime over time: green=stable, orange=transitional, red=critical |
| Fig 7 | `fig7_resolution_outcomes.png` | Shortfall distributions at crisis fixed point under all four resolution methods |
| Fig 8 | `fig8_kernel_evolution.png` | Frobenius norm of G_econ and rho(t) — kernel evolves independently of assets |
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

Seed 42 reset before each scenario generation. Running the five scripts
in order produces identical figures and summaries on every run.
