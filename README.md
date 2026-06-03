# Claim State Dynamics and Crisis-Triggered Resolution in Rehypothecation Networks
## An Endogenous Geometry Approach — Blessing Honmane (2026)

---

## Overview

This repository contains the companion simulation for the paper submitted to the
Journal of Financial Stability. The simulation implements the framework across
three synthetic crisis scenarios and produces all nine figures in the paper.

The core question the paper addresses is: when the same underlying asset
simultaneously supports multiple legally valid but mutually incompatible claims,
how does the geometry of those claims evolve toward crisis, and how much warning
does a regulator have before resolution becomes irresolvable?

---

## Quick Start

Run the five scripts in order from the `code/` directory:

```bash
cd code/
python network_generator.py   # generates the synthetic rehypothecation network
python dynamics.py            # runs T=100 timestep co-evolution of claims and kernel
python monitoring.py          # computes Phi(t), Delta(t), and tau_intervention
python resolution.py          # executes the clearing operator at the crisis fixed point
python plots.py               # produces all 9 paper figures
```

Figures are saved to `simulation_results/paper_figures/`. Running the scripts in
order on any machine with the dependencies below reproduces all figures exactly,
since the random seed is fixed at 42.

**Dependencies:** `numpy >= 1.24`, `scipy >= 1.10`, `matplotlib >= 3.7`

```bash
pip install numpy scipy matplotlib
```

---

## Network Parameters

| Parameter | Value | Notes |
|---|---|---|
| Institutions | 10 | Financial institutions in the network |
| Assets | 5 | Underlying collateral assets |
| Claims | 30 | 6 claims per asset, distributed across institutions |
| Jurisdictions | 3 | Legal jurisdictions |
| Overclaim ratio | 2.5× | Sum of all claims / total asset value, enforced from t=0 |
| Max chain length | 2 | Maximum rehypothecation hops |
| Reuse probability | 0.25 | Per-institution probability of rehypothecation |
| Timesteps | 100 | Simulation horizon T |
| Random seed | 42 | Fixed for reproducibility |

**Important design note.** All three scenarios are generated from the same seed
and therefore share an identical base network topology: same claim sizes, same
ancestry graph, same ρ_ij matrix at t=0. Only the dynamics parameters differ
across scenarios. The netting gap is evaluated at t=0 before any dynamics run,
so it is identical across scenarios by construction. This is a deliberate design
choice to isolate the effect of dynamics; it is not independent cross-scenario
confirmation of a topology-invariant result.

---

## Three Crisis Scenarios

| Scenario | Margin stress μ | Legal shock | τ_intervention |
|---|---|---|---|
| Economic Drift | 0.02 → ~1.2, rate 0.012/step | None | **+68 steps** |
| Legal Regime Shock | Stable at 0.02 throughout | t=50: 80% cross-jurisdiction admissible pairs removed | **−23 steps** |
| Compound | 0.02 → ~1.2, rate 0.012/step | t=50 | **−6 steps** |

A collateral shock δ_A = 0.25 is applied to asset A₀ at t=35 in all three
scenarios, propagating through economic sub-vectors via the shock operator.

**Reading the τ values.** A positive τ means Φ(t) crossed its warning threshold
before Δ(t): there is genuine time to intervene. A negative τ means Δ(t) crossed
first: the system became resolution-sensitive before structural fragility was
detectable. For the legal shock scenario, Φ never crosses its threshold at all
during T=100. The code assigns t_Φ* = T−1 = 99 as a sentinel, giving
τ = 76 − 99 = −23. The specific magnitude depends on T; the qualitative
conclusion (τ < 0, no advance warning is structurally possible) does not.

---

## Implementation Specifications

### The Claim Vector

Each claim carries six dimensions split into a legal sub-vector (seniority,
jurisdiction enforceability, temporal priority) and an economic sub-vector
(custody possession, encumbrance, liquidity speed). Only the economic sub-vector
evolves continuously. The legal sub-vector changes only at discrete regime-change
events.

### Interaction Kernel G_econ

```python
coupling = rho_matrix[i,j] * np.log1p(mu) / np.log1p(1.0)  # divide by log(2)
coupling = np.clip(coupling, 0, 0.3)                         # hard cap
```

The formal equation in the paper is G_econ_ij = ρ_ij · log(1+μ). The simulation
adds two implementation constants not in the formal equation: division by log(2)
to normalise entries to [0,1] for μ ≤ 1, and a hard cap at 0.3 to prevent
saturation before T=100. Both are disclosed in the paper. G_econ is built from
ρ_ij and μ entirely independently of G_legal.

### G_legal: Exact Role

G_legal is an (n × n) boolean admissibility mask with exactly two roles.

1. **Priority score scaling.** The jurisdiction enforceability dimension j_i is
   scaled by mean admissibility: `j_prime_i = j_i × mean(G_legal[i,:])`. This
   is the channel through which the legal scaffold enters Δ(t).

2. **Legal shock event.** At t=50 (legal shock and compound scenarios), G_legal
   restructures discretely: 80% of cross-jurisdiction admissible pairs flip to
   inadmissible. This reorders priority scores instantly, causing Δ(t) to jump
   with no preceding Φ warning.

G_legal does not suppress or modify entries of G_econ. Economic coupling through
shared ancestry persists even when legal admissibility collapses.

### Evolution Function F (Open Problem 1)

```python
# Claim-specific drift
delta_i = ALPHA * rho_i_bar * G_block_i @ C_i * DELTA_T

# Cross-claim coupling
cross_delta = ALPHA * 0.1 * G_econ @ C_flat * DELTA_T

C_new[i] = C[i] + delta_i + cross_delta[i] + noise_i
```

where `rho_i_bar = mean_j(rho_ij)` is the per-claim connectivity weight.
ALPHA = 0.002, DELTA_T = 1, noise ~ N(0, 0.005). Both terms are required to
reproduce Figure 1 trajectories exactly. The 10% cross-claim coupling weight
prevents lockstep saturation while preserving the network effect.

### ρ_ij Dynamic Update

```python
rho[i,j](t+1) = rho[i,j](t) + 0.01 * mu(t) * (1 - rho[i,j](t))
```

Ancestry overlap grows monotonically with margin stress. Provenance paths are
held fixed in this simulation.

### Nominal Erosion

```python
w_i(t+1) = w_i(t) * (1 - 0.001 * mean_encumbrance(t))
```

This drives the slowly declining L(t) visible in Figure 2 between shock events.

### Selection Rule Divergence Δ(t)

```python
Delta(t) = max(
    ||r_lex - r_regret||,
    ||r_lex - r_hist||,
    ||r_regret - r_hist||
) / V_A_total
```

This is the formal definition from the paper, implemented directly. All three
selection rules are evaluated at every timestep in monitoring mode. The dominant
term is ||r_lex - r_hist|| because r_hist is anchored at fixed proportional
allocation, creating maximum structural contrast with the priority waterfall.

### Priority Score Weights (Open Problem 2)

```python
pi_i = +0.05*s + 0.03*j_prime + 0.02*tau
       - 0.45*e - 0.30*(1-p)  - 0.15*(1-l)
```

These weights are hardcoded unvalidated proxies, not derived from the formal
theory. They are required to reproduce Figures 5 and 7 and are disclosed in
Table 1 of the paper. They do not affect the sign or ordering of τ_intervention
across crisis types.

### Fragility Observable Φ

```python
Phi(t) = ||G_econ(t)||_F * L(t) * (1 + ||dG_econ/dt||)
```

The Frobenius norm substitutes for the condition number κ, which is numerically
unstable at high ρ. The diagnostic panels in Figure 2 plot the true condition
number κ(G_econ), while the aggregate Φ uses the Frobenius norm. These are
different signals and should not be read as equivalent.

### τ Threshold Parameters

- Baseline window: t ∈ [1, 25)
- Threshold: mean + 2σ above baseline (symmetric k=2.0 for both Φ and Δ)
- Scan starts at t=25
- Δ smoothed with rolling window w=7

---

## Results

| Scenario | τ | t_Φ* | t_Δ* | Netting gap |
|---|---|---|---|---|
| Economic Drift | **+68** | 31 | 99 | ~47% |
| Legal Regime Shock | **−23** | 99 (fallback) | 76 | ~47% |
| Compound | **−6** | 31 | 25 | ~47% |

**On the netting gap.** The ~47% figure is the allocation-distance between
R_lex (priority waterfall) and R_net (proportional allocation) in a 2.5×
overclaimed network. A control experiment with ρ_ij = 0 also produces ~47%,
confirming the gap is primarily structural to the overclaim ratio, not
ancestry-specific. It is identical across scenarios because all share the same
base network by design. Theorem 5.8 in the paper is a narrower and distinct
result: bilateral netting underestimates expected systemic loss under joint
shocks whenever claims share ancestry. The 47% figure illustrates allocation
distance; the theorem establishes a risk measurement error.

---

## Figure Index

| Figure | Filename | Description |
|---|---|---|
| 1 | `fig1_early_warning_system.png` | Φ(t) and Δ(t) across all three scenarios. Green region is τ_intervention. |
| 2 | `fig2_phi_decomposition.png` | Φ decomposed into κ(G_econ), L(t), and \|\|dG/dt\|\|. Note: top panels show condition number κ; aggregate Φ uses Frobenius norm. |
| 3 | `fig3_intervention_window.png` | τ bar chart. Positive = time to act. Negative = structurally unavoidable late detection. |
| 4 | `fig4_netting_underestimation.png` | Per-claimant gap r_lex − r_net. Red/orange bars are correlated pairs (ρ_ij > 0). |
| 5 | `fig5_selection_rule_divergence.png` | Δ(t) and mean allocations per selection rule across all three scenarios. |
| 6 | `fig6_regime_classification.png` | Regime over time: green = Regime 1 (Stable), orange = Regime 2 (Transitional), red = Regime 3 (Critical). |
| 7 | `fig7_resolution_outcomes.png` | Shortfall distributions at the crisis fixed point under all four resolution methods. |
| 8 | `fig8_kernel_evolution.png` | log(1+κ) and ρ(t) across scenarios. Kernel evolves independently of the underlying assets. |
| 9 | `fig9_summary_table.png` | Summary table of key metrics across all three crisis scenarios. |

---

## File Structure

```
rehypothecation_paper/
├── README.md
├── code/
│   ├── network_generator.py        # Step 1: builds the synthetic network
│   ├── dynamics.py                 # Step 2: T=100 claim and kernel co-evolution
│   ├── monitoring.py               # Step 3: Phi, Delta, tau computation
│   ├── resolution.py               # Step 4: clearing operator at crisis fixed point
│   └── plots.py                    # Step 5: all 9 paper figures
├── latex/
│   ├── layered_claims_framework.tex
│   └── layered_claims_framework_v9.pdf
├── figures/
│   └── paper_figures/              # 9 figures as PNG
└── simulation_results/
    └── paper_figures/              # same figures, generated by plots.py
```

---

## Reproducibility Notes

Running the five scripts in order reproduces all figures exactly on any platform.
The following figures can be reproduced from the paper text alone without
consulting the code:

- Figure 3 (τ values and ordering), Figure 4 (netting gap), Figure 6 (regime
  classification).

The following require implementation details now disclosed in the paper:

- Figure 1 (exact trajectories): requires cross-coupling weight 0.1 from
  Remark (Proxy 1) and log(2) normalisation from Remark (Proxy 2).
- Figure 7 (shortfall distributions): requires priority score weights from
  Table 1 of the paper.

The sign and ordering of τ across crisis types, which is the paper's headline
result, is robust across all threshold parameter combinations tested. See
Table 2 in the paper for the sensitivity analysis.
