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
| `tas-fw` | Track-and-Stop + Frank-Wolfe, finite alphabet | Yavas et al. (arXiv 2409.05072) | small instances only (N ≤ 20); Exp-3's reduced instance ONLY, not Exp-1 — see note there |
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

**Deviation from steps 1-2 above, as actually implemented in
`scripts/calibrate_constants.py` (compute-constraint, illustrative-supplement
scope):** a real (non-smoke) calibration run against the literal protocol —
200 runs/binary-search step, 15 bisection iterations — was benchmarked before
committing to it: kabc's own per-run cost at the generous calibration
`budget_cap` ranges up to ~100s+ near `c_cal=1.0` (the theoretical constant,
tested first in the search), and `calibrate_constants.py` has no equivalent
of `run_experiment.py`'s `joblib.Parallel` across its within-step runs at the
time this was scoped, making the literal protocol impractical for this
repo's laptop-scale, illustrative-supplement goals (not a paper-submission
compute budget). Implemented instead:

- 100 calibration runs per binary-search step (20 pull-noise seed groups ×
  5 reps each, halving step 1's implied reps-per-instance), and again for
  held-out validation. **Correction to step 1 above:** step 1's "20 GMM
  instances" was never achievable with `GaussianMixtureEnv` as implemented —
  its constructor takes no seed and has no other source of randomness
  (cluster means are a deterministic closed-form function of
  `(N,K,d,Delta,sigma)`), matching how `run_experiment.py`/`configs/exp1.yaml`
  already use it (one fixed instance per grid point). So all 20 "instance
  seeds" build the identical instance; only the per-rep pull-noise RNG
  varies. What actually runs is 100 flat, independently-seeded reps against
  one fixed instance — a well-defined empirical error estimate, just not "20
  distinct instances" the way step 1's prose implies. `configs/calibration.yaml`'s
  header comment and `scripts/calibrate_constants.py`'s module docstring
  record this correction too.
- 8 bisection iterations instead of 15 (still resolves `c_cal` to within
  `(1-0.01)/2^8 ≈ 0.004` of the true smallest-safe value on `[0.01,1]` —
  ample precision for a leading multiplicative constant used only to reduce
  the theoretical threshold's known conservatism, not for a value with
  application-level sensitivity at the 0.001 scale).
- Runs within one binary-search step ARE now parallelized (`joblib.Parallel`,
  `n_jobs=-2`, matching `run_experiment.py`'s own pattern), each run capped
  to a single BLAS thread (env vars + `threadpoolctl`) -- without the cap,
  the 11 concurrent worker processes each independently multi-threaded their
  own Gram-matrix computations, oversubscribing the 12 physical cores and
  making a real calibration run's wall-clock badly unpredictable (confirmed
  empirically: one worker process burned ~2.7x more CPU-seconds than
  wall-clock elapsed before this was caught and fixed).
- The search no longer empirically re-verifies `c_cal=1.0` (the theoretical
  constant) before bisecting -- it is the algorithm's own proven threshold
  constant (Eq. 6), safe by construction, and that single upfront check
  measured as ~79 of kabc's ~100-minute total calibration wall-clock, the
  most expensive single step by far, spent reconfirming something the proof
  already guarantees. Bisection instead searches `[LO, SEARCH_HI]` directly
  (see below for both bounds), trusting `1.0` as a safe-but-unsearched upper
  endpoint; if no candidate the search actually tries passes, that is
  reported as a real negative result (`c_cal: null` in
  `configs/calibration.yaml`, with an explanatory comment distinguishing it
  from "never run") -- NOT silently papered over by falling back to an
  unverified expensive candidate.
- `SEARCH_HI = 0.6` for kabc/kabc-sg (not `1.0`): real single-run timing
  (each run capped to 1 BLAS thread -- see the oversubscription bullet
  above) showed cost is cheap for `const` in roughly `[0.1, 0.5]`
  (single-digit to ~30s/run) and explodes near `1.0` (~520s/run); letting
  the search itself climb back toward `1.0` would re-walk into that region
  for no benefit, since the search only ever moves toward smaller `const`
  on a pass. **acb uses a WIDER `SEARCH_HI = 0.95` instead** -- confirmed
  empirically (single runs, not the full batch) that acb stays sub-second
  (0.02-0.04s) across the whole range up to `1.0`, so the cost reason for
  capping at `0.6` doesn't apply to it; the original `[0.1, 0.6]` search
  found no passing candidate anywhere (best: `const=0.598`, error 0.03,
  still above `delta/2`), and widening to `0.95` found `c_cal=0.6611` with
  both calibration-set and held-out error comfortably under threshold. See
  `configs/calibration.yaml`'s `acb` entry for the recorded detail.
- `LO = 0.1` (raised from `0.01`, NOT for compute-cost reasons -- see the
  safeguards bullet below, which already makes any `const` in `[0.01,1]`
  safe to attempt): investigating a run that took 448.6s and still failed to
  converge at `const=0.0192` found that `const=0.022` (barely larger)
  converges trivially (2.1s, 1 iteration), and `const=0.025` does too. The
  region just below ~0.02 is not a smooth continuation of "smaller const,
  harder problem" -- it is an unstable discontinuity where a run avoids
  reporting a wrong partition only by exhausting `budget_cap` first
  (`stopped=False`), not because the threshold is genuinely resolving edges.
  `const=0.126` (comfortably above the new `LO`) behaves normally (7.8s,
  real convergence). A `c_cal` the search could otherwise land on near the
  old `LO=0.01` would be numerically fast to validate but not a genuine
  reflection of the threshold separating clusters -- not a constant this
  protocol should ever freeze.
- **Safeguards added to `kabc.py`'s doubling-trick loop** (`_fit`, shared by
  `fit`/`fit_sg`, and `fit_oracle`'s one-shot `_cluster_once` call), found
  necessary by the same investigation: a `MAX_ITERATIONS=50` backstop (far
  above where `budget_cap` or the memory guard below would already have
  stopped the loop on any real instance in this repo -- a true backstop, not
  a binding constraint) and an explicit memory guard inside `_cluster_once`
  (before pulling any samples, estimate the Gram-matrix memory footprint at
  a given `n`: `_cluster_once`'s actual peak simultaneous allocation is 3
  `n x n` matrices, not `N*(N-1)/2` of them -- confirmed by reading the code,
  since each pair's matrices are computed and freed before the next pair
  starts -- so `11 workers * 3 matrices * 4x safety factor * n**2 * 8 bytes`,
  capped at 8 GiB total, trips at `n_k > 2852`). Both produce the existing
  graceful `Result(stopped=False, ...)` pattern with a new
  `history[-1] = {"stop_reason": "budget_cap"|"max_iterations"|"memory_guard"}`
  marker (the existing budget_cap early-exits were retrofitted with the same
  marker too, for consistency) -- not a new `Result` field, since that
  dataclass is shared across every algorithm module per this doc's fixed
  `fit` interface. These make every `const` in `[0.01,1]` safe to attempt
  (nothing hangs indefinitely or risks OOM) -- they do NOT make the
  near-`LO=0.01` region fast OR numerically meaningful, which is why `LO`
  was raised regardless of the safeguards existing.

This is a real, documented deviation from the protocol as literally stated
above, not a silent substitution -- `configs/calibration.yaml`'s header
comment and `scripts/calibrate_constants.py`'s module docstring both record
it too.

**PROTOCOL CORRECTION: calibration budget regime must match deployment.**
All of the above was originally run under a `budget_cap` 5x the real
formula-derived value ("generous", meant to keep calibration runs from ever
hitting the cap, so error would isolate the threshold's own tightness).
This was wrong: budget exhaustion IS part of what a real experiment
experiences, and calibrating against an artificially generous cap doesn't
validate error under the conditions deployment will actually use.
Discovered concretely: kabc-sg calibrated under the 5x cap measured 2%
error; re-run at Exp 1's real base-grid-point cap (71,055) it showed 14% --
nearly 3x over `delta/2=0.025`, entirely from `stopped=False` runs that
would have resolved given more budget but don't get it in a real run.
Fixed: `scripts/calibrate_constants.py`'s `_real_budget_cap` (replacing
`_generous_budget_cap`) uses the exact formula `run_experiment.py` uses for
a real grid point (`constant=200`, `s_star_sq` via `estimate_s_star_sq`),
no margin. `SEARCH_HI` also widened back to `1.0` (from `0.6`) for this
re-run: under the real (tighter) cap, cost near `const=1.0` dropped from
~520s/run to ~117s/run (kabc) -- the tighter budget makes the algorithm
give up sooner regardless of constant -- so the earlier cost argument for
excluding that region no longer holds as strongly, and const=1.0 is no
longer safely assumable to satisfy `error<=delta/2` under a fixed budget
purely because it's the theoretical constant (the theorem guarantees
eventual correctness given unlimited patience, not convergence within any
particular fixed budget).

Re-calibrating under the real cap: **kabc** was essentially unaffected
(`0.1035` vs the generous-cap run's `0.1020`, zero error both ways) --
confirms kabc's earlier calibration was already representative. **acb**
was also confirmed fine (`0.6590` vs `0.6611`, held-out error 0.01 both
times) -- checked directly, not assumed safe just because Exp 1's
real-cap benchmark happened to show a low error rate for it.

**kabc-sg hit its known cliff again, closer this time, and its OWN
held-out check caught a real problem:** the bisection's literal answer
(`0.117578`, calibration-set error 2%) FAILED held-out validation at 5%
error -- on a fresh sample of the same fixed instance, not a different
one. Rather than accept this literal answer, `0.121094` was frozen
instead -- the bisection's own iteration 7 candidate, one step before the
search descended into the cliff (`const=0.1141`, 22% error) -- trading a
little search-set efficiency for real margin above demonstrated
instability. Re-validated on the SAME held-out block that failed
`0.117578`: `0.0000` error. `configs/calibration.yaml`'s `kabc-sg` entry
records this override and the reasoning inline.

This incident is itself evidence for why calibration must target the real
budget regime: under the generous cap, kabc-sg's cliff sat further from
where bisection actually landed (`0.115625`, comfortably passing both
checks); under the real cap, the cliff sits close enough that a single
100-run sample isn't always enough to reliably characterize which side of
it a candidate is on -- exactly why the held-out check exists, and exactly
why it was trusted here instead of the raw calibration-set result.

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
- **ARI-at-cap applies to ANY algorithm's `stopped=False` runs, not just ACB's**
  (ACB was simply the algorithm this design originally expected to hit that
  state routinely, being mean-blind on same-mean families -- it is not a
  metric restricted to ACB). `metrics.ari`/`run_experiment.py`'s result row
  already compute `ari` unconditionally for every run regardless of
  algorithm or `stopped` value (checked directly, no code change needed) --
  a capped kabc or kabc-sg run reports a real ARI-at-cap exactly the same
  way a capped ACB run does. This matters in practice: pre-calibration (or
  at a hard grid point) kabc/kabc-sg CAN also end up `stopped=False`, and
  that run's ARI is still meaningful diagnostic information, not just a
  binary error=1 to discard.
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

**Algorithms:** kabc, kabc-sg, acb, unif-kkm (n in log-grid). No tas-fw here:
its finite-alphabet assumption doesn't hold for these continuous Gaussian
instances, and the originally-planned N=15, K=3, d=2 subset is moot anyway
since GaussianMixtureEnv's mean construction requires d ≥ 2K = 6 for K=3 (see
the d-sweep note above). Running tas-fw here would require an off-label
discretization approximation outside the algorithm's proven guarantees;
tas-fw is instead run only on Experiment 3's reduced instance, where the
finite-alphabet assumption holds natively (MovieLens ratings on {1..5}) — see
Experiment 3 below.

**Figures:**
1. (Headline) log tau vs log Delta, all methods; dashed overlay of the theoretical
   slope s*^{-2} (compute s* in closed form for RBF + Gaussians numerically).
2. tau box plots vs d (does the nonparametric price grow with d?).
3. Table: empirical error per method per config (validate ≤ δ), alongside
   ARI-at-cap for any `stopped=False` rows (per §0.4's ARI-at-cap
   generalization -- not exclusive to acb; see the calibration-diagnostic
   note below for why this matters specifically for kabc-sg).
4. (Appendix) theoretical-constants vs calibrated-constants panel.

**Expected outcome:** acb < kabc < kabc-sg in budget, gaps modest; error ≤ δ everywhere.

**Calibration diagnostic finding, relevant to interpreting kabc-sg's error
here:** while calibrating `c_cal` (`configs/calibration.yaml`), kabc-sg's
rare failures at constants near its frozen value were investigated directly
(reproduced misclassified partitions, not just aggregate error rates).
Every failure observed was a **near-miss** (ARI 0.926-0.975), not a
qualitatively wrong clustering: typically 1-3 "straggler" arms failing to
join their true cluster while every other arm is placed correctly, and the
run exhausts its budget (`stopped=False`) rather than converging to a
confidently wrong partition. The mechanism: `subgaussian_threshold` is
**not per-pair adaptive** — it uses only the fixed kernel bound `g_bar=1.0`
and the sample size `n`, identical for every arm pair regardless of that
pair's own empirical spread. One shared cutoff must simultaneously satisfy
every pair, so whichever single pair is noisiest for a given draw determines
whether the whole clustering resolves; since the statistic's sampling noise
and the threshold shrink at the same asymptotic rate in `n`, a borderline
pair does not resolve with more samples, it stays borderline until the
budget runs out. `variance_aware_threshold` (kabc's own threshold, Eq. 6)
sizes each pair's cutoff from that pair's *empirical* RKHS variance, so a
noisier pair automatically gets a proportionally larger threshold — and
correspondingly, kabc showed zero comparable instability across its entire
calibration search (constants from 1.0 down to 0.102, uniformly zero
error), while kabc-sg's search hit a sharp, narrow cliff specifically near
its frozen constant. This is direct empirical evidence for the paper's own
claim about the benefit of variance-aware bounds over the kernel-uniform
sub-Gaussian ablation -- worth citing in the writeup discussion of kabc-sg's
results, not only as internal calibration QA.

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
strongest fair baseline. Budget cap: main instance uses the flat 2e6 (below);
the reduced instance uses its own budget_cap = 20000 instead — the main
instance's 2e6 would be a meaningless cap at N=6 scale (that number predates
this reduced instance). Not derived from a formula; chosen to comfortably
allow convergence for both kabc and tas-fw at this scale while keeping
tas-fw's expensive per-decision compute bounded to a reasonable total step
count. See configs/exp3.yaml.

**Figures:**
1. (Headline) left: rating histograms of the same-mean prototype pair (titles shown);
   right: ARI vs budget — acb plateaus < 1, kabc reaches 1 and stops (mark tau).
2. Table: tau and error for every algorithm; ARI-at-cap wherever `stopped=False`
   occurs (expected routinely for acb on this same-mean-blind instance, but not
   assumed exclusive to it -- report it for kabc/kabc-sg too if either ends up
   capped on a given run, per §0.4's ARI-at-cap generalization).
3. Small-instance table: tas-fw vs kabc on the N = 6 instance.

**Expected outcome:** acb cannot separate the same-mean pair → ARI plateau ~
(K-2 correct clusters + merged pair); kabc exact recovery; tas-fw competitive on
the tiny instance but orders of magnitude more compute per decision.

---

## Known limitation: unbounded Gram-matrix cost at hard grid points (open problem)

`kabc.py`'s (and hence `kabc-sg`/`kabc-oracle`'s) per-iteration `mmd2_plugin`/
`rkhs_variance` calls (`kernels.py`) have no sample-size safeguard -- they
always materialize a full `n x n` RBF Gram matrix. The doubling trick's own
`n_k` schedule can legitimately grow large on hard instances (small `s*`),
and at Exp 2's `moment_matched_mixture` family near `a=0.5` (where the true
MMD gap approaches 0, `s_star_sq` measured ~0.0024) this is not hypothetical:
`n_k` reaches ~135k by iteration 13 (one Gram matrix ~146 GB) and ~1.1M by
iteration 16 (~9.8 TB) before that grid point's real `budget_cap` (~21M,
correctly computed from the formula in section 0.4 -- not a bug) would stop
it. `scripts/run_experiment.py`'s `SMOKE_TEST` mode caps `budget_cap` at a
small ceiling (`SMOKE_TEST_MAX_BUDGET_CAP`) specifically to avoid this during
smoke tests; the real (non-smoke) `budget_cap` formula is untouched.

This means a real (non-smoke) `make exp2` run will hit the same wall on hard
grid points like `moment_matched_mixture` near `a=0.5` -- capping the
budget there for real, or excluding that grid point, would be scientifically
wrong (section 0.4's budget formula is exactly the quantity Exp 2 is
designed to measure, and `a=0.5` is the most informative point in that
sweep, closest to the MMD-gap-to-0 limit). Fixing this needs a real design
decision -- whether some subsampling approximation inside the Gram-matrix
computations is acceptable given KABC's theoretical guarantee, or whether
the algorithm needs actual algorithmic changes (e.g. incremental Gram-matrix
updates across doubling-trick iterations instead of full recomputation each
time) -- not attempted here. This is a known, documented open problem
blocking a real (non-smoke) Experiment 2 run at its hardest grid points, not
something resolved by this document.

---

## Repository layout

Workflow: `make data` → `make calibrate` → `make smoke` (all three experiments,
SMOKE_TEST mode, < 10 min laptop) → `make exp1 exp2 exp3` → notebooks for figures.
​```
kernel-active-bandit/
├── configs/                  # yaml per experiment + calibration.yaml
├── data/                     # gitignored; download script fills it
├── docs/
│   └── experiments.md        # this file
├── notebooks/
│   ├── 01_gaussian_mixtures.ipynb
│   ├── 02_same_mean_shapes.ipynb
│   └── 03_movielens.ipynb
├── results/                  # gitignored parquet outputs
├── scripts/
│   ├── download_movielens.py
│   ├── calibrate_constants.py
│   └── run_experiment.py     # CLI: python -m scripts.run_experiment configs/exp1.yaml
├── src/kabc/
│   ├── algorithms/           # kabc.py, acb.py, tas_fw.py, baselines.py
│   ├── environments.py       # GMM, same-mean families, MovieLens env
│   ├── kernels.py            # RBF, median heuristic, MMD estimators (incl. variance est.)
│   ├── thresholds.py         # variance-aware + sub-Gaussian, calibrated constants
│   ├── metrics.py            # ARI, error, s_star computation
│   └── plotting.py           # shared figure style (matplotlib, paper-ready)
├── tests/                    # pytest: MMD estimator correctness, threshold coverage,
│                             # env sanity (means/variances), interface contracts
├── Makefile                  # data, calibrate, exp1/2/3, smoke, test
└── pyproject.toml            # numpy, scipy, scikit-learn, joblib, pyarrow, matplotlib
​```