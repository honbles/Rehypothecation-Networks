# Rehypothecation Network: Simulation Code (v8, fixed)

Companion simulation for:

**Claim State Dynamics and Crisis-Triggered Resolution in Rehypothecation
Networks: An Endogenous Geometry Approach**

---

## Contents

```
rehypothecation_fixed_v8/
├── network_generator.py
├── dynamics.py
├── monitoring.py          <- fix applied (mkdir bug)
├── resolution.py
├── plots.py
├── README.md
└── latex/
    ├── paper.tex
    ├── references.bib
    ├── cover_letter.tex
    └── figures/           <- all 21 simulation figures
```

---

## Changes from v7

**Fix 1. monitoring.py: mkdir bug**
`os.makedirs(output_dir, exist_ok=True)` added to `run_monitoring()`.
Running verbatim from a clean directory no longer raises FileNotFoundError.

**Fix 2. Table 3 legal shock values updated in paper**
R�_regret = 583 (was 667), Classical Netting = 786 (was 754).

**Fix 3. kappa vs Frobenius norm disclosed**
Proxy 5 remark now states that kappa(G^econ) is replaced by the Frobenius
norm throughout, because G^econ is rank-deficient by construction.

**Fix 4. Compound tau raw = -2 disclosed**
Figure 3 caption notes EMA-dependence of the simultaneous-crossing result.

**Fix 5. Phi floor of 5.0 disclosed**
Monitoring-mode section states the absolute floor and its purpose.

---

## Requirements

Python 3.9+. Install: `pip install numpy scipy matplotlib`

## Running

```bash
python network_generator.py
python dynamics.py
python monitoring.py
python resolution.py
python plots.py
```

## Key Results (seed 42)

| Scenario           | tau   | Theorem 5.5 gap | Null control |
|--------------------|-------|-----------------|--------------|
| Economic Drift     | +39   | 12.93% of V_A   | 0.00%        |
| Legal Regime Shock | -38   | 10.54% of V_A   | 0.00%        |
| Compound           | 0     | 12.93% of V_A   | 0.00%        |

## Compiling the LaTeX

```bash
cd latex/
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Robustness Script (Section 6.5)

```bash
python robustness_wilcoxon.py
```

Runs the full pipeline across 50 random network seeds and produces the
Wilcoxon signed-rank test reported in Section 6.5. Results are written to
`robustness_results/wilcoxon_results.json` and `wilcoxon_results.txt`.
Runtime approximately 2-5 minutes.
