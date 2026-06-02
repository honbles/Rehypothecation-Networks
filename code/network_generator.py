"""
network_generator.py
====================
Synthetic rehypothecation network generator for:

    A Formal Framework for Layered Asset Claims and Resolution Outcomes
    in Rehypothecation Networks
    Blessing Honmane, 2026

This script generates the synthetic dataset consumed by all subsequent
simulation scripts. It produces no model results. Its sole responsibility
is to construct a internally consistent synthetic rehypothecation network
whose structure matches every formal definition in the paper.

Objects generated
-----------------
- Institutions B = {B_1, ..., B_n}
- Assets A = {A_1, ..., A_m}
- Claims C_i with full claim vectors c_i = (s, p, e, j, l, tau)
- Provenance paths P_i for every claim
- Initial interaction kernel G_econ (3n x 3n)
- Legal scaffold G_legal (encoded as admissibility mask)
- Three scenario configurations: economic drift, legal shock, compound

Paper references
----------------
Definition 1.1  : Claim Vector
Definition 1.2  : Legal Sub-vector
Definition 1.3  : Economic Sub-vector
Definition 1.5  : Asset Claim State
Definition 1.6  : Decomposed Interaction Kernel
Definition 2.1  : Collateral Provenance Path
Definition 2.2  : Collateral Ancestry Set
Definition 2.3  : Pairwise Chain Correlation
Section 1.3     : Coordinate Interpretation
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from itertools import combinations

# -----------------------------------------------------------------------
# Random seed for reproducibility
# -----------------------------------------------------------------------
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# -----------------------------------------------------------------------
# Simulation parameters
# All parameters here correspond directly to paper definitions.
# -----------------------------------------------------------------------

# Network scale
N_INSTITUTIONS   = 10    # |B| number of institutions
N_ASSETS         = 5     # |A| number of underlying assets
N_CLAIMS_PER_ASSET = 6   # number of claims generated per asset
                          # total claims = N_ASSETS * N_CLAIMS_PER_ASSET

# Asset recoverable values V_A (Definition: conservation constraint)
# Set so that sum(w_i) > V_A is structurally guaranteed --- this is
# the overclaim condition the paper formalises.
ASSET_BASE_VALUE = 1000.0       # base recoverable value per asset
OVERCLAIM_RATIO  = 2.5          # sum(w_i) / V_A target at initialisation
                                 # ensures structural overclaim from t=0

# Rehypothecation chain parameters
MAX_CHAIN_LENGTH = 2     # shorter chains at initialisation — system starts sparse
REUSE_PROBABILITY = 0.25 # lower reuse so rho_ij starts near zero, grows with mu
                          # an asset it receives

# Legal jurisdiction parameters
# Three jurisdictions: 0=common_law, 1=civil_law, 2=mixed
N_JURISDICTIONS = 3
# Jurisdiction assignment: each institution is assigned one jurisdiction
# G_legal admissibility: interactions between institutions in the same
# jurisdiction are always admissible; cross-jurisdiction interactions
# are admissible with probability CROSS_JURISDICTION_ADMISSIBILITY
CROSS_JURISDICTION_ADMISSIBILITY = 0.7   # matches paper (p_cross = 0.7)

# Scenario-specific shock parameters
ECONOMIC_DRIFT_RATE   = 0.012   # mu reaches ~1.2 by t=100 from 0.02 start
LEGAL_SHOCK_TIMESTEP  = 50      # timestep at which legal regime shock fires
LEGAL_SHOCK_MAGNITUDE = 0.8     # fraction of cross-jurisdiction interactions
                                 # that become inadmissible after shock

# Total simulation timesteps
T_STEPS = 100

# -----------------------------------------------------------------------
# Data structures
# These mirror the formal objects in the paper exactly.
# -----------------------------------------------------------------------

@dataclass
class Institution:
    """
    Represents one institution B_k in B.
    Paper: Section 3 Setup, Definition 1.6
    """
    id: int                          # index k
    name: str                        # human-readable label
    jurisdiction: int                # 0=common_law, 1=civil_law, 2=mixed
    n_internal_accounts: int         # |W_k| number of internal accounts


@dataclass
class Asset:
    """
    Represents one underlying asset A in A.
    Paper: conservation constraint sum(w_i) > V_A
    """
    id: int
    name: str
    recoverable_value: float         # V_A


@dataclass
class ClaimVector:
    """
    The full six-dimensional claim vector c_i.
    Paper: Definition 1.1

    Legal sub-vector  : (s_i, j_i, tau_i)  -- Definition 1.2
    Economic sub-vector: (p_i, e_i, l_i)   -- Definition 1.3

    All dimensions in [0,1].
    Classification is phenomenological not ontological (Section 1.3):
    dimensions classified by dominant dynamical behaviour.

    s_i   : legal seniority         -- 1=most senior, 0=equity
    p_i   : custody possession      -- 1=full physical, 0=contractual only
    e_i   : encumbrance level       -- 0=clean, 1=fully pledged
    j_i   : jurisdiction enforceability -- 1=immediately enforceable
    l_i   : liquidity/enforcement speed -- 1=immediate, 0=years
    tau_i : temporal priority       -- normalised creation time, 1=oldest
    """
    s:   float    # legal seniority
    p:   float    # custody possession        [economic]
    e:   float    # encumbrance level         [economic]
    j:   float    # jurisdiction enforceability [legal]
    l:   float    # liquidity/enforcement speed [economic]
    tau: float    # temporal priority          [legal]

    @property
    def legal(self) -> np.ndarray:
        """Legal sub-vector (s, j, tau). Paper: Definition 1.2"""
        return np.array([self.s, self.j, self.tau])

    @property
    def economic(self) -> np.ndarray:
        """Economic sub-vector (p, e, l). Paper: Definition 1.3"""
        return np.array([self.p, self.e, self.l])

    @property
    def full(self) -> np.ndarray:
        """Full six-dimensional vector."""
        return np.array([self.s, self.p, self.e, self.j, self.l, self.tau])


@dataclass
class Claim:
    """
    Represents one claim C_i on asset A.
    Paper: Definition 1.5, Definition 2.1

    A claim is held by one institution (the current holder).
    Its provenance path records every institution through which
    the collateral passed before generating this claim.
    This is causal dependency, not legal title (Section 4.1).
    """
    id: int                              # claim index i
    asset_id: int                        # which asset A
    holder_institution_id: int           # current legal holder B_k
    nominal_amount: float                # w_i nominal claim amount
    vector: ClaimVector                  # c_i full claim vector
    provenance_path: List[int]           # P_i: ordered list of institution ids
                                         # from original holder to current
    creation_timestep: int               # timestep at which claim was created
                                         # used to compute tau_i

    @property
    def ancestry_set(self) -> set:
        """
        pi(C_i): set of all institutions in provenance path.
        Paper: Definition 2.2
        Note: typed space V = B sqcup A; here we store institution ids only.
        Asset ancestry is handled separately in rho_iA computation.
        """
        return set(self.provenance_path)


@dataclass
class Network:
    """
    The full synthetic rehypothecation network at t=0.
    This is the primary output of this script.
    """
    institutions:     List[Institution]
    assets:           List[Asset]
    claims:           List[Claim]
    G_econ:           np.ndarray          # (3n x 3n) initial economic kernel
    G_legal_mask:     np.ndarray          # (n x n) boolean admissibility mask
    jurisdiction_map: Dict[int, int]      # institution_id -> jurisdiction
    rho_matrix:       np.ndarray          # (n_claims x n_claims) pairwise rho_ij
    rho_scalar:       float               # rho(0) mean pairwise ancestry overlap
    scenario:         str                 # 'economic_drift' | 'legal_shock' | 'compound'
    params:           Dict                # all parameters used


# -----------------------------------------------------------------------
# Step 1: Generate institutions
# -----------------------------------------------------------------------

def generate_institutions(n: int) -> List[Institution]:
    """
    Generate n institutions each assigned to one jurisdiction.
    Jurisdictions are assigned roughly evenly across institutions.
    """
    institutions = []
    for k in range(n):
        jurisdiction = k % N_JURISDICTIONS
        n_accounts = rng.integers(3, 8)   # 3-7 internal accounts per institution
        institutions.append(Institution(
            id=k,
            name=f"B_{k}",
            jurisdiction=jurisdiction,
            n_internal_accounts=int(n_accounts)
        ))
    return institutions


# -----------------------------------------------------------------------
# Step 2: Generate assets
# -----------------------------------------------------------------------

def generate_assets(m: int) -> List[Asset]:
    """
    Generate m underlying assets with recoverable values V_A.
    Values are drawn from a uniform distribution around the base value.
    """
    assets = []
    for a in range(m):
        value = ASSET_BASE_VALUE * rng.uniform(0.7, 1.3)
        assets.append(Asset(
            id=a,
            name=f"A_{a}",
            recoverable_value=round(float(value), 2)
        ))
    return assets


# -----------------------------------------------------------------------
# Step 3: Generate rehypothecation chains and provenance paths
# -----------------------------------------------------------------------

def generate_provenance_path(
    institutions: List[Institution],
    originating_institution_id: int
) -> List[int]:
    """
    Generate a provenance path P_i for one claim.
    The path is a directed sequence of institution ids representing
    the chain of custody before the current holder received the claim.

    Paper: Definition 2.1
    This is causal dependency, not legal title. The path records
    which institutions the collateral passed through.

    The path always starts with the originating institution.
    Each subsequent institution is chosen from the remaining pool
    with probability REUSE_PROBABILITY, stopping at MAX_CHAIN_LENGTH.
    """
    path = [originating_institution_id]
    available = [i.id for i in institutions if i.id != originating_institution_id]
    rng.shuffle(available)

    for candidate in available:
        if len(path) >= MAX_CHAIN_LENGTH:
            break
        if rng.random() < REUSE_PROBABILITY:
            path.append(candidate)

    return path


# -----------------------------------------------------------------------
# Step 4: Generate claim vectors
# -----------------------------------------------------------------------

def generate_claim_vector(
    holder_institution: Institution,
    provenance_path: List[int],
    creation_time: int,
    total_timesteps: int,
    seniority_tier: int,
    n_seniority_tiers: int
) -> ClaimVector:
    """
    Generate a claim vector c_i consistent with the paper definitions.

    Legal dimensions (s, j, tau):
    - s: determined by seniority tier, spread evenly across [0.1, 1.0]
    - j: determined by jurisdiction of holder institution
    - tau: normalised creation time (older = higher tau)

    Economic dimensions (p, e, l):
    - p: custody possession degrades with chain length
         (longer chain = less direct possession)
    - e: encumbrance grows with number of times asset was rehypothecated
         (longer path = more encumbered)
    - l: liquidity degrades with cross-jurisdiction hops in path

    All values in [0,1]. Classification is phenomenological (Section 1.3).
    """

    # --- Legal dimensions ---

    # Seniority: evenly spaced by tier, 1=most senior
    s = 1.0 - (seniority_tier / max(n_seniority_tiers - 1, 1)) * 0.9
    s = float(np.clip(s, 0.05, 1.0))

    # Jurisdiction enforceability: based on holder's jurisdiction
    # common_law=0.9, civil_law=0.75, mixed=0.6
    jurisdiction_enforceability = {0: 0.90, 1: 0.75, 2: 0.60}
    j = jurisdiction_enforceability[holder_institution.jurisdiction]
    j += float(rng.uniform(-0.05, 0.05))   # small noise
    j = float(np.clip(j, 0.0, 1.0))

    # Temporal priority: normalised creation time, 1=oldest
    # older claims have higher tau
    tau = 1.0 - (creation_time / max(total_timesteps - 1, 1))
    tau = float(np.clip(tau, 0.0, 1.0))

    # --- Economic dimensions ---

    chain_length = len(provenance_path)

    # Custody possession: degrades with chain length
    # direct holder (chain_length=1) has p near 1.0
    # long chains reduce physical possession
    p = 1.0 / (1.0 + 0.4 * (chain_length - 1))
    p += float(rng.uniform(-0.05, 0.05))
    p = float(np.clip(p, 0.05, 1.0))

    # Encumbrance: grows with chain length
    # each rehypothecation hop adds encumbrance
    e = 1.0 - (1.0 / (1.0 + 0.5 * (chain_length - 1)))
    e += float(rng.uniform(-0.05, 0.05))
    e = float(np.clip(e, 0.0, 0.95))

    # Liquidity/enforcement speed: degrades with cross-jurisdiction hops
    # same jurisdiction throughout = fast enforcement
    # multiple jurisdictions = slow
    l = 0.9 - 0.15 * (chain_length - 1)
    l += float(rng.uniform(-0.05, 0.05))
    l = float(np.clip(l, 0.05, 1.0))

    return ClaimVector(s=s, p=p, e=e, j=j, l=l, tau=tau)


# -----------------------------------------------------------------------
# Step 5: Generate all claims
# -----------------------------------------------------------------------

def generate_claims(
    institutions: List[Institution],
    assets: List[Asset],
    n_claims_per_asset: int
) -> List[Claim]:
    """
    Generate all claims across all assets.

    For each asset, generate n_claims_per_asset claims.
    Each claim is assigned to a random institution as current holder.
    Nominal amounts are set so sum(w_i) = OVERCLAIM_RATIO * V_A
    ensuring the structural overclaim condition.

    Paper: Definition 1.5, conservation constraint
    """
    claims = []
    claim_id = 0

    for asset in assets:
        # Distribute claims across institutions
        # Each claim assigned to a randomly selected institution
        holder_ids = rng.choice(
            len(institutions),
            size=n_claims_per_asset,
            replace=True
        )

        # Set nominal amounts so sum = OVERCLAIM_RATIO * V_A
        # This enforces sum(w_i) > V_A structurally
        raw_weights = rng.dirichlet(np.ones(n_claims_per_asset))
        nominal_amounts = raw_weights * asset.recoverable_value * OVERCLAIM_RATIO

        for k in range(n_claims_per_asset):
            holder = institutions[holder_ids[k]]

            # Generate provenance path from a random originating institution
            originating_id = int(rng.choice(len(institutions)))
            path = generate_provenance_path(institutions, originating_id)

            # Seniority tier: spread claims across tiers
            seniority_tier = k % n_claims_per_asset

            # Creation time: spread across first half of simulation
            creation_time = int(rng.integers(0, T_STEPS // 2))

            vec = generate_claim_vector(
                holder_institution=holder,
                provenance_path=path,
                creation_time=creation_time,
                total_timesteps=T_STEPS,
                seniority_tier=seniority_tier,
                n_seniority_tiers=n_claims_per_asset
            )

            claims.append(Claim(
                id=claim_id,
                asset_id=asset.id,
                holder_institution_id=holder.id,
                nominal_amount=float(nominal_amounts[k]),
                vector=vec,
                provenance_path=path,
                creation_timestep=creation_time
            ))
            claim_id += 1

    return claims


# -----------------------------------------------------------------------
# Step 6: Compute pairwise chain correlation rho_ij
# -----------------------------------------------------------------------

def compute_rho_matrix(claims: List[Claim]) -> np.ndarray:
    """
    Compute the (n_claims x n_claims) pairwise chain correlation matrix.

    rho_ij = |pi(C_i) intersect pi(C_j)| / |pi(C_i) union pi(C_j)|

    This is the Jaccard similarity of ancestry sets.
    Paper: Definition 2.3

    rho_ij = 0 when claims share no ancestry (fully independent)
    rho_ij = 1 when claims have identical provenance paths
    rho_ij > 0 whenever paths share any upstream institution

    This is a topological measure, not a statistical correlation.
    """
    n = len(claims)
    rho = np.zeros((n, n))

    for i in range(n):
        rho[i, i] = 1.0    # self-correlation is 1 by convention
        for j in range(i + 1, n):
            pi_i = claims[i].ancestry_set
            pi_j = claims[j].ancestry_set
            intersection = len(pi_i & pi_j)
            union = len(pi_i | pi_j)
            if union == 0:
                rho_ij = 0.0
            else:
                rho_ij = intersection / union
            rho[i, j] = rho_ij
            rho[j, i] = rho_ij    # symmetric

    return rho


def compute_rho_scalar(rho_matrix: np.ndarray) -> float:
    """
    rho(t) = mean pairwise ancestry overlap across all claim pairs.
    Paper: Network Reuse Intensity definition.
    """
    n = rho_matrix.shape[0]
    off_diagonal = rho_matrix[np.triu_indices(n, k=1)]
    return float(np.mean(off_diagonal))


# -----------------------------------------------------------------------
# Step 7: Build G_legal admissibility mask
# -----------------------------------------------------------------------

def build_G_legal(
    institutions: List[Institution],
    claims: List[Claim]
) -> np.ndarray:
    """
    Build the legal admissibility mask G_legal.

    G_legal is an (n_claims x n_claims) boolean matrix.
    Entry (i,j) = True if the interaction between C_i and C_j
    is admissible under the governing legal regime.

    Admissibility rules (paper: Definition 1.6):
    - Same jurisdiction institutions: always admissible
    - Cross-jurisdiction institutions: admissible with probability
      CROSS_JURISDICTION_ADMISSIBILITY

    G_legal is symmetric. It changes only at legal regime-change events
    (paper: coupled evolution equations).
    """
    n = len(claims)
    mask = np.zeros((n, n), dtype=bool)

    inst_jur = {inst.id: inst.jurisdiction for inst in institutions}

    for i in range(n):
        mask[i, i] = True   # self-interaction always admissible
        for j in range(i + 1, n):
            jur_i = inst_jur[claims[i].holder_institution_id]
            jur_j = inst_jur[claims[j].holder_institution_id]

            if jur_i == jur_j:
                admissible = True
            else:
                admissible = bool(rng.random() < CROSS_JURISDICTION_ADMISSIBILITY)

            mask[i, j] = admissible
            mask[j, i] = admissible

    return mask


# -----------------------------------------------------------------------
# Step 8: Build initial G_econ
# -----------------------------------------------------------------------

def build_G_econ(
    claims: List[Claim],
    rho_matrix: np.ndarray,
    G_legal_mask: np.ndarray,
    mu: float = 0.1
) -> np.ndarray:
    """
    Build the initial economic interaction kernel G_econ_t.

    Paper: Definition 1.6
    G_econ_t in R^{3n x 3n} where n = number of claims.
    Acts on the stacked economic state vector C_econ_t in R^{3n}.

    Simulation-specific form (stated as proxy in paper):
        G_econ_{ij}(t) = rho_ij(t) * mu(t)

    Where:
    - rho_ij is ancestry overlap (computed above)
    - mu is margin stress scalar (0=normal, 1=full stress)

    Conditioned on G_legal:
        (G_econ | G_legal)_{ij} = G_econ_{ij} if admissible, else 0

    The 3x3 block structure reflects the three economic dimensions
    (p, e, l) for each claim pair.

    Sparsity: G_econ_{ij} = 0 when rho_ij = 0 (no shared ancestry)
    or when G_legal_mask[i,j] = False (inadmissible interaction).
    This is consistent with the paper's sparsity statement.
    """
    n = len(claims)
    G = np.zeros((3 * n, 3 * n))

    for i in range(n):
        for j in range(n):
            # Economic coupling is driven by ancestry overlap only.
            # G_legal is NOT applied here -- it conditions priority scoring
            # in monitoring.py, not economic coupling existence.
            # Claims can be economically coupled through shared ancestry
            # even when legally inadmissible for direct priority comparison.
            coupling = rho_matrix[i, j] * mu

            if coupling < 1e-10:
                continue

            # Fill 3x3 block (i,j) in G_econ
            # Diagonal entries: direct coupling of same economic dimension
            # Off-diagonal entries: cross-dimension coupling (weaker)
            block = np.array([
                [coupling,           coupling * 0.3,      coupling * 0.2],
                [coupling * 0.3,     coupling,            coupling * 0.4],
                [coupling * 0.2,     coupling * 0.4,      coupling      ]
            ])

            row_start = 3 * i
            col_start = 3 * j
            G[row_start:row_start+3, col_start:col_start+3] = block

    return G


# -----------------------------------------------------------------------
# Step 9: Validate overclaim condition
# -----------------------------------------------------------------------

def validate_overclaim(claims: List[Claim], assets: List[Asset]) -> Dict:
    """
    Verify sum(w_i) > V_A for each asset.
    Paper: conservation constraint.
    Returns a summary dict for logging.
    """
    summary = {}
    asset_values = {a.id: a.recoverable_value for a in assets}

    for asset in assets:
        asset_claims = [c for c in claims if c.asset_id == asset.id]
        total_claims = sum(c.nominal_amount for c in asset_claims)
        V_A = asset_values[asset.id]
        overclaim_ratio = total_claims / V_A
        summary[asset.name] = {
            "V_A": V_A,
            "sum_w_i": round(total_claims, 2),
            "overclaim_ratio": round(overclaim_ratio, 3),
            "overclaim_satisfied": bool(total_claims > V_A)
        }

    return summary


# -----------------------------------------------------------------------
# Step 10: Assemble scenario configurations
# -----------------------------------------------------------------------

def make_scenario_config(scenario: str) -> Dict:
    """
    Return scenario-specific parameters.

    Three scenarios matching paper Section 6.5:

    economic_drift:
        G_legal is stable throughout.
        mu increases gradually at ECONOMIC_DRIFT_RATE per timestep.
        Wide intervention window expected.

    legal_shock:
        mu is stable (low stress).
        At LEGAL_SHOCK_TIMESTEP, G_legal is restructured:
        LEGAL_SHOCK_MAGNITUDE fraction of cross-jurisdiction
        interactions become inadmissible.
        Near-zero intervention window expected.

    compound:
        Both economic drift and legal shock occur simultaneously.
        Most severe scenario.
    """
    base = {
        "T_steps":                 T_STEPS,
        "mu_initial":              0.02,   # start very quiet — Regime 1
        "legal_shock_timestep":    None,
        "legal_shock_magnitude":   0.0,
        "economic_drift_rate":     0.0,
        "delta_star":              0.15,   # Delta(t) threshold for transition
        "beta_star":               2.0,    # beta threshold for transition
    }

    if scenario == "economic_drift":
        base["economic_drift_rate"] = ECONOMIC_DRIFT_RATE
        base["description"] = (
            "G_legal stable. mu drifts upward at {:.3f}/step. "
            "Wide intervention window expected."
        ).format(ECONOMIC_DRIFT_RATE)

    elif scenario == "legal_shock":
        base["legal_shock_timestep"]  = LEGAL_SHOCK_TIMESTEP
        base["legal_shock_magnitude"] = LEGAL_SHOCK_MAGNITUDE
        base["description"] = (
            "mu stable. Legal regime shock at t={} removes {:.0f}% "
            "of cross-jurisdiction admissibility. "
            "Near-zero intervention window expected."
        ).format(LEGAL_SHOCK_TIMESTEP, LEGAL_SHOCK_MAGNITUDE * 100)

    elif scenario == "compound":
        base["economic_drift_rate"]    = ECONOMIC_DRIFT_RATE
        base["legal_shock_timestep"]   = LEGAL_SHOCK_TIMESTEP
        base["legal_shock_magnitude"]  = LEGAL_SHOCK_MAGNITUDE
        base["description"] = (
            "Both economic drift and legal shock. "
            "Most severe scenario. Shortest intervention window expected."
        )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    base["scenario"] = scenario
    return base


# -----------------------------------------------------------------------
# Main: generate and save all three scenario networks
# -----------------------------------------------------------------------

def generate_network(scenario: str) -> Network:
    """Generate one complete network for a given scenario."""

    print(f"\n{'='*60}")
    print(f"Generating network: {scenario}")
    print(f"{'='*60}")

    institutions = generate_institutions(N_INSTITUTIONS)
    assets       = generate_assets(N_ASSETS)
    claims       = generate_claims(institutions, assets, N_CLAIMS_PER_ASSET)

    print(f"  Institutions : {len(institutions)}")
    print(f"  Assets       : {len(assets)}")
    print(f"  Claims       : {len(claims)}")

    rho_matrix  = compute_rho_matrix(claims)
    rho_scalar  = compute_rho_scalar(rho_matrix)
    G_legal     = build_G_legal(institutions, claims)
    G_econ      = build_G_econ(claims, rho_matrix, G_legal, mu=0.1)

    print(f"  rho(0)       : {rho_scalar:.4f}")
    print(f"  G_econ shape : {G_econ.shape}")
    print(f"  G_legal admissible pairs: {G_legal.sum()} / {G_legal.size}")

    overclaim = validate_overclaim(claims, assets)
    print(f"\n  Overclaim validation:")
    for asset_name, info in overclaim.items():
        status = "OK" if info["overclaim_satisfied"] else "FAIL"
        print(f"    [{status}] {asset_name}: "
              f"sum(w_i)={info['sum_w_i']:.1f}, "
              f"V_A={info['V_A']:.1f}, "
              f"ratio={info['overclaim_ratio']:.2f}")

    scenario_config = make_scenario_config(scenario)

    params = {
        "N_INSTITUTIONS":               N_INSTITUTIONS,
        "N_ASSETS":                     N_ASSETS,
        "N_CLAIMS_PER_ASSET":           N_CLAIMS_PER_ASSET,
        "ASSET_BASE_VALUE":             ASSET_BASE_VALUE,
        "OVERCLAIM_RATIO":              OVERCLAIM_RATIO,
        "MAX_CHAIN_LENGTH":             MAX_CHAIN_LENGTH,
        "REUSE_PROBABILITY":            REUSE_PROBABILITY,
        "N_JURISDICTIONS":              N_JURISDICTIONS,
        "CROSS_JURISDICTION_ADMISSIBILITY": CROSS_JURISDICTION_ADMISSIBILITY,
        "T_STEPS":                      T_STEPS,
        "RNG_SEED":                     RNG_SEED,
        "scenario_config":              scenario_config,
        "overclaim_validation":         overclaim,
    }

    return Network(
        institutions=institutions,
        assets=assets,
        claims=claims,
        G_econ=G_econ,
        G_legal_mask=G_legal,
        jurisdiction_map={i.id: i.jurisdiction for i in institutions},
        rho_matrix=rho_matrix,
        rho_scalar=rho_scalar,
        scenario=scenario,
        params=params,
    )


def save_network(network: Network, output_dir: str) -> None:
    """
    Save network to disk for consumption by dynamics.py.
    NumPy arrays saved as .npy files.
    All other data saved as JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save numpy arrays
    np.save(
        os.path.join(output_dir, "G_econ_init.npy"),
        network.G_econ
    )
    np.save(
        os.path.join(output_dir, "G_legal_mask_init.npy"),
        network.G_legal_mask
    )
    np.save(
        os.path.join(output_dir, "rho_matrix_init.npy"),
        network.rho_matrix
    )

    # Save claim vectors as stacked array (n_claims x 6)
    claim_vectors = np.array([c.vector.full for c in network.claims])
    np.save(
        os.path.join(output_dir, "claim_vectors_init.npy"),
        claim_vectors
    )

    # Save economic sub-vectors as stacked array (n_claims x 3)
    econ_vectors = np.array([c.vector.economic for c in network.claims])
    np.save(
        os.path.join(output_dir, "econ_vectors_init.npy"),
        econ_vectors
    )

    # Save nominal amounts (n_claims,)
    nominal_amounts = np.array([c.nominal_amount for c in network.claims])
    np.save(
        os.path.join(output_dir, "nominal_amounts.npy"),
        nominal_amounts
    )

    # Save asset recoverable values
    V_A = np.array([a.recoverable_value for a in network.assets])
    np.save(
        os.path.join(output_dir, "V_A.npy"),
        V_A
    )

    # Save claim-to-asset mapping
    claim_asset_map = np.array([c.asset_id for c in network.claims])
    np.save(
        os.path.join(output_dir, "claim_asset_map.npy"),
        claim_asset_map
    )

    # Save provenance paths and metadata as JSON
    metadata = {
        "n_claims":       len(network.claims),
        "n_institutions": len(network.institutions),
        "n_assets":       len(network.assets),
        "scenario":       network.scenario,
        "rho_scalar":     network.rho_scalar,
        "params":         network.params,
        "institutions": [
            {
                "id":           i.id,
                "name":         i.name,
                "jurisdiction": i.jurisdiction,
                "n_accounts":   i.n_internal_accounts
            }
            for i in network.institutions
        ],
        "assets": [
            {
                "id":                a.id,
                "name":              a.name,
                "recoverable_value": a.recoverable_value
            }
            for a in network.assets
        ],
        "claims": [
            {
                "id":                    c.id,
                "asset_id":              c.asset_id,
                "holder_institution_id": c.holder_institution_id,
                "nominal_amount":        round(c.nominal_amount, 4),
                "provenance_path":       c.provenance_path,
                "ancestry_set":          list(c.ancestry_set),
                "creation_timestep":     c.creation_timestep,
                "vector": {
                    "s":   round(c.vector.s,   4),
                    "p":   round(c.vector.p,   4),
                    "e":   round(c.vector.e,   4),
                    "j":   round(c.vector.j,   4),
                    "l":   round(c.vector.l,   4),
                    "tau": round(c.vector.tau, 4),
                }
            }
            for c in network.claims
        ],
        "jurisdiction_map": {
            str(k): v for k, v in network.jurisdiction_map.items()
        }
    }

    with open(os.path.join(output_dir, "network_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved to: {output_dir}/")
    print(f"    G_econ_init.npy          {network.G_econ.shape}")
    print(f"    G_legal_mask_init.npy    {network.G_legal_mask.shape}")
    print(f"    rho_matrix_init.npy      {network.rho_matrix.shape}")
    print(f"    claim_vectors_init.npy   {claim_vectors.shape}")
    print(f"    econ_vectors_init.npy    {econ_vectors.shape}")
    print(f"    nominal_amounts.npy      {nominal_amounts.shape}")
    print(f"    V_A.npy                  {V_A.shape}")
    print(f"    claim_asset_map.npy      {claim_asset_map.shape}")
    print(f"    network_metadata.json")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":

    output_base = "simulation_data"

    # Reset rng to same seed before each scenario so all three share
    # an identical base network topology. Only scenario-specific
    # parameters (mu_initial, drift_rate, legal_shock) differ.
    # This ensures the netting gap is structurally consistent across
    # crisis types — it depends only on rho_ij which is identical.
    for scenario in ["economic_drift", "legal_shock", "compound"]:
        # Reassign module-level rng to same seed before each scenario
        # so all three share identical base topology
        rng = np.random.default_rng(RNG_SEED)
        # Patch into generate_network via closure — update module global
        import sys
        sys.modules[__name__].rng = rng
        net = generate_network(scenario)
        save_network(net, os.path.join(output_base, scenario))

    print(f"\n{'='*60}")
    print("Network generation complete.")
    print(f"All three scenario networks saved under: {output_base}/")
    print(f"{'='*60}")
    print("""
Next scripts:
  dynamics.py   -- runs T-step co-evolution of c_i and G_econ
  monitoring.py -- computes Phi, Delta, beta, tau_intervention
  resolution.py -- executes R_hat at crisis fixed point
  plots.py      -- generates all figures
""")
