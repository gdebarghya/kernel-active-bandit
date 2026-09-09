# experiments.md — Experimental design for KABC (kernel-based active clustering)

Companion design doc for the paper "Kernel-based Active Clustering of Bandit Arms"
(Thuot, Vogt, Ghoshdastidar, Verzelen). All experiments are laptop-scale.
Language: Python. One Jupyter notebook per experiment; shared logic in `src/`.

## 0. Common infrastructure

### 0.1 Algorithms to implement (src/kabc/algorithms/)

| ID | Algorithm | Source | Notes |
|----|-----------|--------|-------|
| `kabc` | KABC, variance-aware threshold (Wolfer–Alquier style) | our paper | main method |
| `kabc-sg` | KABC, kernel-uniform sub-Gaussian threshold (Tolstikhin style) | our paper, Discussion | ablation |
| `kabc-oracle` | Non-adaptive CLUSTER(n*, δ) with oracle n* | our paper | measures price of adaptivity (doubling trick) |
| `acb` | ACB (SRI + ADC), mean-based | Thuot et al. 2024 (arXiv 2406.11485) | re-implement; shared codebase style |
| `tas-fw` | Track-and-Stop + Frank-Wolfe, finite alphabet | Yavas et al. (arXiv 2409.05072) | small instances only (N ≤ 20) |
| `unif-kkm` | Uniform sampling (n per arm) + kernel k-means | folklore baseline | sweep n; non-adaptive, fixed-budget |

Design rule: every algorithm implements a common interface
`fit(env, delta, budget_cap, rng) -> Result(partition, tau, stopped, history)`.
Environments implement `pull(arm_index, rng) -> sample`.

### 0.2 Calibrated constants protocol

Theoretical threshold constants are conservative. We report the **practical variant**:
replace the leading constant in each stopping threshold by `c_cal * constant_shape`,
where `c_cal` is calibrated ONCE, globally, as follows:

1. Calibration instances: 20 GMM instances (Exp-1 family, Δ at mid-grid), disjoint seeds.
2. Binary-search the smallest `c_cal ∈ [0.01, 1]` such that empirical error over
   200 calibration runs stays ≤ δ/2 (δ = 0.05).
3. Freeze `c_cal`; use it unchanged in ALL experiments (including MovieLens).
4. Paper reports the value of `c_cal` and this protocol. Appendix figure: one
   Exp-1 panel with theoretical constants for comparison.

Same protocol applied independently to `acb` and `tas-fw` so no method is
disadvantaged. Store calibrated values in `configs/calibration.yaml`.

### 0.3 Kernel and bandwidth

- Kernel: Gaussian RBF, characteristic on R^d.
- Bandwidth: median heuristic on a pilot sample — pull each arm `n_pilot = 5`
  times before the algorithm starts (pilot pulls ARE counted in the budget τ).
  Median of pairwise distances across all pilot samples pooled.
- Bandwidth is frozen after the pilot (kernel fixed during the run, as theory assumes).

### 0.4 Metrics, protocol, reproducibility

- δ = 0.05 everywhere. Monte Carlo: 100 repetitions per configuration
  (50 for TaS-FW configs). Seeds: `seed = 1000*exp_id + rep`.
- Metrics:
  - `tau`: stopping budget (adaptive methods). `tau = 0` specifically signals
    that `budget_cap` was too tight to even complete the pilot phase (no
    pulls were made) — distinct from a run that pilots successfully and is
    then capped mid-run, which always has `tau >= N*n_pilot`.
  - `error`: 1{partition != ground truth} averaged over reps (must be ≤ δ).
  - `ARI(budget)`: for non-stopping/capped methods, Adjusted Rand Index of the
    current partition estimate at budget checkpoints (log-spaced grid).
- Budget cap: `budget_cap = 200 * N * s_star^{-2} * log(N/delta)` when s* known
  (synthetic), else `2e6` pulls (MovieLens). A run hitting the cap records
  `stopped=False`, ARI at cap.
- Parallelism: `joblib` over repetitions, `n_jobs=-2`. Every notebook has a
  `SMOKE_TEST = True` flag (reduces reps to 5, grid to 2 points) for fast iteration.
- All results saved as parquet in `results/expN/`; notebooks only load + plot,
  never recompute silently.

---

## Experiment 1 — Gaussian mixtures: competitiveness in the parametric regime

**Claim tested:** in the mean-separated Gaussian setting (home turf of ACB/BOC),
KABC's budget is within a modest factor of ACB; budget scales as s*^{-2} log(N/δ)
(Theorem 3.1); variance-aware threshold beats sub-Gaussian threshold.

**Instances.** N arms, K clusters (balanced), arm i in cluster k has distribution
N(mu_k, sigma^2 I_d), sigma = 1. Cluster means: mu_k = Delta/sqrt(2) * (e_{2k-1} + e_{2k})
so pairwise separation is exactly Delta (equidistant means).

**Grid (one knob at a time, others at base):**
- Base: N = 60, K = 3, d = 10, Delta = 3.
- Sweep Delta in {1, 1.5, 2, 3, 4, 6} (main plot).
- Sweep d in {6, 10, 50, 100} at Delta = 3. (d=2 excluded: GaussianMixtureEnv's mean construction requires d ≥ 2K = 6 for K=3, to place K orthogonal mean-pairs.)
- Sweep N in {30, 60, 120} at Delta = 3.
- TaS-FW subset: N = 15, K = 3, d = 2, Delta in {2, 3, 4} only.

**Algorithms:** kabc, kabc-sg, acb, unif-kkm (n in log-grid), tas-fw (subset).

**Figures:**
1. (Headline) log tau vs log Delta, all methods; dashed overlay of the theoretical
   slope s*^{-2} (compute s* in closed form for RBF + Gaussians numerically).
2. tau box plots vs d (does the nonparametric price grow with d?).
3. Table: empirical error per method per config (validate ≤ δ).
4. (Appendix) theoretical-constants vs calibrated-constants panel.

**Expected outcome:** acb < kabc < kabc-sg in budget, gaps modest; error ≤ δ everywhere.

---

## Experiment 2 — Identical means, different shapes: nonparametric separation

**Claim tested:** when clusters share the same mean, mean-based methods are blind
(ARI at chance for any budget) while KABC recovers the partition with budget
governed by the MMD gap. Also: price of adaptivity (kabc vs kabc-oracle).

**Instance families (d = 1 unless stated):**
- (2a) **Scale pair:** cluster A = N(0, 1), cluster B = N(0, sigma_B^2),
  sigma_B in {1.5, 2, 3, 4}. Same mean, different variance.
- (2b) **Moment-matched mixture:** cluster A = N(0,1),
  cluster B = 0.5 N(-a, b^2) + 0.5 N(a, b^2) with b^2 = 1 - a^2 (matches mean AND
  variance), a in {0.5, 0.7, 0.8, 0.9, 0.95}. Only shape differs.
- (2c) **2D geometry (figure candidate):** ring (radius 2, Gaussian noise 0.1) vs
  N(0, I_2) blob, both centered at origin. Single configuration.

N = 40, K = 2, balanced. 100 reps.

**Algorithms:** kabc, kabc-sg, kabc-oracle, acb (budget-capped), unif-kkm.

**Figures:**
1. (Headline, family 2b) two-panel: left = the two densities at a = 0.8 (visual:
   same mean/variance, different shape); right = tau vs MMD^2 gap (log-log) for
   kabc/kabc-sg/kabc-oracle, plus a horizontal band showing acb's ARI stuck at
   ~0 for ALL budgets up to cap.
2. (2a) tau vs sigma_B; acb ARI-vs-budget flatline inset.
3. Price of adaptivity: ratio tau(kabc)/tau(kabc-oracle) across all Exp-2 configs
   (box plot). Expect a small constant (~2-4 from the doubling trick).
4. (2c) scatter of one ring/blob dataset + recovered partition (pretty figure).

**Expected outcome:** acb at chance ARI at every budget; kabc exact recovery with
tau tracking MMD^{-2}; oracle gap a small constant.

---

## Experiment 3 — MovieLens: real rating distributions

**Claim tested:** the same-mean-different-shape phenomenon occurs in real data
(polarizing vs consensus movies), and KABC exploits it where mean-based methods fail.

**Data.** MovieLens-1M, local copy at `data/ml-1m/` (movies.dat, ratings.dat,
users.dat, README). `scripts/download_movielens.py` checks for
`data/ml-1m/ratings.dat` first and only downloads from grouplens.org if missing.
Filter to movies with ≥ 1000 ratings.

**Prototype selection (deterministic, scripted — `notebooks/03` shows the result):**
1. For each retained movie, compute its empirical rating distribution p_m on {1..5},
   mean(p_m), and pairwise MMD^2 (RBF on R, bandwidth = median heuristic on ratings).
2. Find the pair (m1, m2) maximizing MMD^2 subject to |mean(p_m1) - mean(p_m2)| ≤ 0.05
   → the "polarizing vs consensus" pair (report the movie titles; histogram figure).
3. Add K-2 further prototypes greedily maximizing the minimum pairwise MMD^2
   (these may differ in mean — realism, not adversarial construction). K = 4 total.
4. Arms: m = 10 arms per prototype (N = 40). Pulling an arm = draw one rating
   uniformly WITH replacement from its prototype's rating pool. Ground truth =
   prototype identity. (With-replacement sampling makes same-cluster arms
   identically distributed, matching the theory's premise; state this in the paper.)

**Algorithms:** kabc, kabc-sg, acb (ratings as 1D sub-Gaussian, sigma^2 estimated
empirically per arm — more favorable to acb than a conservative bounded-support
proxy), unif-kkm. tas-fw on a reduced instance: K = 3 prototypes, 2 arms each
(N = 6) — its finite-alphabet assumption holds exactly here, making it the
strongest fair baseline.

**Figures:**
1. (Headline) left: rating histograms of the same-mean prototype pair (titles shown);
   right: ARI vs budget — acb plateaus < 1, kabc reaches 1 and stops (mark tau).
2. Table: tau and error for kabc/kabc-sg; ARI-at-cap for acb.
3. Small-instance table: tas-fw vs kabc on the N = 6 instance.

**Expected outcome:** acb cannot separate the same-mean pair → ARI plateau ~
(K-2 correct clusters + merged pair); kabc exact recovery; tas-fw competitive on
the tiny instance but orders of magnitude more compute per decision.

---

## Repository layout

Workflow: `make data` → `make calibrate` → `make smoke` (all three experiments,
SMOKE_TEST mode, < 10 min laptop) → `make exp1 exp2 exp3` → notebooks for figures.
```