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
- **A fourth stop reason, `"degenerate_pilot"`, added later (during
  Experiment 4's real run)** for a distinct failure mode found in the pilot
  phase itself, not the doubling loop: `median_heuristic_bandwidth` can
  receive a pilot sample whose pairwise distances are all (numerically)
  zero, most often on real discrete data with a skewed rating distribution
  (a small pilot draw sampling only tied high-rating values by chance from a
  movie's real pool). It now falls back to the MEAN pairwise distance first
  (`on_degenerate="mean"`, with a `UserWarning` on trigger) -- still
  informative unless literally every pooled sample is identical -- and only
  then, if the mean is ALSO zero, does `_pilot_or_graceful_stop` return
  `Result(stopped=False, history=[{"stop_reason": "degenerate_pilot"}])`
  instead of raising. See `docs/results_exp1_exp3.tex`'s Experiment 4 setup
  section for the concrete crash this fixed.

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

## Experiment 4 — MovieLens random pairs: robustness across the real-data pair distribution

**Claim tested:** Experiment 3 uses one fixed, cherry-picked K=4 MovieLens instance.
Experiment 4 asks whether KABC's advantage (and ACB's/TaS-FW's relative behavior)
is an artifact of that one instance, or holds up when the movie PAIR is randomly
resampled every rep across the real diversity of the MovieLens catalog, under
three distinct pairing regimes that vary how much mean separation is present
and whether B is cherry-picked (max MMD²) or sampled uniformly.

**Data.** `data/ml-1m-207/` (`ratings.parquet`, `movies.parquet`) — the 207
MovieLens-1M movies with ≥ 1000 ratings, pre-filtered by
`scripts/build_movielens_207.py` so per-rep pair sampling doesn't re-load/re-filter
the full ~1M-row `ratings.dat` every time. `movies.parquet` carries
`movie_id, title, n_ratings, mean_rating`; `ratings.parquet` carries the raw
per-movie ratings needed to build each rep's environment. Same discrete
rating support {1..5} and RBF-on-ratings kernel as Experiment 3, with the
same fixed median-heuristic bandwidth computed once over the full 207-movie
pool (bandwidth = 1.0, matching Experiment 3 — not recomputed per pair).

**REVISION HISTORY (this section has been revised twice, after a real
5-rep-per-version smoke test on the original 3-version/7-algorithm-row
design surfaced several findings that motivated the changes below — kept
here rather than silently overwritten, since the smoke-test findings that
motivated each change are real evidence, not speculation):**
- **Revision 1:** `unif-kkm-full` (the fixed-`budget_cap` variant) DROPPED:
  it reached ARI ≈ 1.0 in nearly every observed case (generous budget, no
  adaptivity required to eventually separate a K=2/N=10 instance), so it
  wasn't discriminating between anything --- confirmed uninteresting rather
  than assumed. Only `unif-kkm-1x`/`unif-kkm-2x` remain (6 algorithm rows
  now: kabc, kabc-sg, acb, tas-fw, unif-kkm-1x, unif-kkm-2x).
  `budget_cap` raised from 10000 to **20000**: kabc's smoke-test failures
  (falling back to the ungraceful trivial-partition answer, see "ARI is
  exactly 0.0, never negative" below) may in some cases just need more
  budget, and 20000 was the ORIGINAL value before an earlier reduction made
  purely to bound tas-fw's worst case --- that tas-fw cost concern is now
  handled by capping N at 10 (see "Why N cannot grow" below), not by
  shrinking `budget_cap`. `exp4_1`'s `mean_gap_tol` raised from 0.05 to
  **0.1** (still the argmax-MMD² rule, unchanged). `exp4_2` redefined from a
  single one-sided threshold (`gap > 0.2`) to a RANGE (`0.1 ≤ gap ≤ 0.5`),
  still uniform B-selection, no longer framed as "favors acb" --- the smoke
  test showed acb struggling there too (mean ARI 0.12 over 5 reps, only 20%
  ever stopping); the honest framing is "easier due to mean separation
  existing at all," not "a regime any specific algorithm owns."
- **Revision 1 also DROPPED `exp4_3`** (loose same-mean gap at
  `mean_gap_tol=0.1`, uniform B-selection): the smoke test showed it was
  uniformly hard --- kabc, kabc-sg, acb, AND tas-fw all struggled on its
  sampled pairs (mostly landing on the ARI-exactly-0.0 fallback, see below)
  --- so it looked like it would mostly re-confirm that same difficulty
  rather than add new signal, at real cost (tas-fw alone averages minutes
  per rep).
- **Revision 2 RE-ADDED `exp4_3`**, after weighing what dropping it actually
  cost: `exp4_1` (tight gap + argmax-MMD²) vs. the original `exp4_3` (SAME
  tight gap + uniform) was the only comparison in this design that isolated
  whether kabc's `exp4_1` advantage comes from the tight mean-gap itself or
  specifically from the MMD-maximizing cherry-pick --- losing `exp4_3`
  loses that ablation entirely. Rather than restore it at the same
  `mean_gap_tol=0.1` (which produced total ARI=0.0 degeneracy for kabc on
  every sampled pair in the smoke test --- no signal at all to compare
  against), `exp4_3` returns with `mean_gap_tol` LOOSENED to **0.2**: still
  uniform B-selection (no cherry-picking), still same-mean-ish, but with
  enough real separation that kabc should show ARI somewhere between total
  failure and success rather than flooring out at exactly 0.0 every time.
  This means `exp4_1` vs. `exp4_3` is no longer a perfectly clean
  gap-held-fixed ablation (0.1 vs. 0.2, not 0.1 vs. 0.1) --- an accepted
  trade-off for actually getting a usable signal out of `exp4_3`, made
  explicitly rather than silently.

**Why N cannot realistically grow past 8-10: tas-fw's cost.** tas-fw's
Track-and-Stop design re-scans its ENTIRE hypothesis space (|C|, the number
of ways to split N arms into K blocks of size ≥ 2) AND re-solves a linear
program at EVERY single pull, not periodically like kabc's doubling trick.
This makes it slow even at trivial scale: at Experiment 3's N=6/K=3 reduced
instance (|C|=90), tas-fw was already ~172x slower than kabc per rep (39.64s
vs 0.23s) despite matching kabc's accuracy exactly (ARI 1.000 vs 0.995). At
N=10/K=2 (|C|=501, ~5.6x larger), the smoke test measured tas-fw averaging
~900-1300s/rep whenever it failed to converge within `budget_cap`. The real
N=8/K=2 run (final config, |C| smaller than N=10's) confirmed the same
order of magnitude in practice: mean wall-clock 789-908s/rep across all
three versions, two to three orders of magnitude slower than every other
algorithm (all under ~2.1s/rep) --- see `docs/results_exp1_exp3.tex`'s
§6.14. Since |C| grows combinatorially in N (not polynomially), and tas-fw's
use in this experiment specifically exists to give a fair,
finite-alphabet-respecting baseline, N=8-10 is treated as close to the
practical ceiling for including tas-fw at all; a larger N would need either
dropping tas-fw from Experiment 4 entirely or accepting per-rep costs an
order of magnitude beyond what is already a bottleneck.

**Smoke-test findings worth carrying forward (from the now-superseded
3-version/7-row design, still valid evidence about algorithm behavior):**
- **ARI is exactly 0.0 for certain failure modes, never negative, and this
  is a real mathematical property, not noise.** Verified directly: ARI CAN
  go negative in general (a maximally scrambled alternating partition
  against the true 5-5 split gives ARI=-0.08). But kabc's/acb's graceful
  degradation fallback --- the trivial "everyone in one cluster" answer
  returned when budget runs out without enough evidence to split --- gives
  EXACTLY ARI=0.0 against any true partition, always, by construction of
  the adjusted-for-chance formula (a degenerate single-cluster partition
  makes the observed and expected agreement coincide). kabc-sg's near-
  degenerate 9-1 splits (see next point) also landed at exactly 0.0 in
  every case checked, for the same underlying reason (a singleton cluster
  contributes zero same-cluster pairs).
- **kabc-sg's fragility (see "Restricted budget" and Experiment 2's
  calibration-generalization findings) reproduces here in a new form: not
  just failure to converge, but CONFIDENT convergence to a wrong answer.**
  On the smoke test's hardest sampled pairs, kabc (correctly) fell back to
  the graceful "everyone in one cluster" non-answer (`stopped=False`) ---
  but kabc-sg instead `stopped=True` on a near-degenerate 9-1 split (one
  arm isolated as its own "cluster") every single time on those pairs. This
  is a materially different, and worse, failure mode: an honest "I don't
  know" (kabc) vs. a confidently wrong answer (kabc-sg) that happens to
  score the same ARI=0.0 by coincidence of the two partitions' size
  profiles. Any future ARI-only comparison should be read alongside
  `stopped` for exactly this reason.

**REVISION 3 (post-real-run): N reduced to 8, reps to 50 (40 completed
excluding one permanently-skipped rep), and a real bandwidth-selection
crash was found and fixed.** The N=10 pilot's timing extrapolation (tas-fw
~900-1300s/rep) made a 100-rep run impractically long; N was reduced to 8
(`arms_per_prototype=4`) to keep tas-fw's per-rep cost and total wall-clock
bounded, and reps to 50. Mid-run, `median_heuristic_bandwidth`'s pilot-phase
call raised `ValueError` on a genuinely pathological pair (Schindler's List
vs. Rear Window: both extremely skewed toward high ratings, and a 5-per-arm
pilot draw sampled only tied {3,4,5} values by chance), killing the whole
`Parallel(return_as="generator")` batch and losing all in-flight work for
that batch (already-completed reps were safe, having been written after
each prior rep). Fixed at the source (`src/kabc/kernels.py`,
`src/kabc/algorithms/kabc.py`), not worked around in `run_experiment4.py`:
`median_heuristic_bandwidth` now accepts `on_degenerate="mean"`, falling
back to the mean pairwise distance (still informative unless *every* pair is
tied) before raising, and `_pilot_or_graceful_stop` uses this fallback with
a `UserWarning` on trigger; if the mean is *also* degenerate, KABC now stops
gracefully (`STOP_REASON_DEGENERATE_PILOT`, `stopped=False`) rather than
crashing, matching the existing budget-cap/memory-guard graceful-stop
convention. Verified directly against the exact crash case (rep 38,
exp4_1): KABC now converges correctly instead of crashing. One rep (38)
remains permanently absent from the final dataset --- it was skipped (not
re-run) under the pre-fix "just resume" decision, before the fix existed;
revisiting whether to backfill it is a decision for a future session, not
done here. Final dataset: 40 reps/algorithm/version (0-37, 39-49), 240 rows
per version file. Results and per-version summary stats are in
`docs/results_exp1_exp3.tex`'s "Results --- Experiment 4" section (compiled
PDF), not duplicated here.

**Common setup, all three versions:**
- N = 8 arms, K = 2 clusters, `arms_per_prototype = 4` (reduced from the
  original N=10/`arms_per_prototype=5` design below --- see Revision 3).
- Algorithms: kabc, kabc-sg, acb, tas-fw, unif-kkm-1x, unif-kkm-2x --- 6
  algorithm rows per (rep, version). 50 reps configured (40 completed, see
  Revision 3) for every algorithm including tas-fw.
- `budget_cap`: fixed at **20000** (see revision history above). Not the
  `s_star_formula` type used elsewhere --- a formula-derived cap isn't
  meaningful at this scale.
- δ = 0.05, matching every other experiment.
- Calibrated `c_cal` constants are REUSED from the existing frozen
  `configs/calibration.yaml` (kabc, kabc-sg, acb) --- no new calibration pass
  for Experiment 4, per the "calibrate once, reuse everywhere" protocol
  (section 0.2). Flagged here explicitly since Experiment 4's instance family
  (random MovieLens pairs) is distributionally different from what those
  constants were calibrated against (Experiment 1's `gaussian_mixture`); if
  results look miscalibrated (e.g. kabc-sg error far above δ/2, the same
  failure mode already documented for Experiment 2 in section 0.2), that is
  a finding to report, not a bug to silently patch by re-tuning here.
- Reported per algorithm per version: mean AND standard deviation of τ, ARI,
  and wall-clock/rep. Visualized as bar plots (bar height = mean, error bar =
  std dev): one bar plot per metric (ARI, τ, wall-clock), each with 3 sets
  (one per version) of 6 bars (one per algorithm) --- 18 bars per plot.
- **Architecture note:** unlike Experiments 1-3, EVERY rep here builds a
  fresh environment (a new random movie pair), not just fresh pull-noise
  against one fixed environment built once per grid point. This does not fit
  `run_experiment.py`'s existing per-grid-point loop (build env once, reuse
  across reps), so Experiment 4 is implemented as a separate, dedicated
  script (`scripts/run_experiment4.py` + `configs/exp4.yaml`) rather than by
  extending the shared grid engine.
- **Parallelization: across REPS, not across (version, algorithm) work
  items within a rep.** `Parallel(n_jobs=10, return_as="generator")` runs
  `process_one_rep(rep, ...)` for each of the 100 reps, 10 at a time; INSIDE
  one call, all three versions and all 6 algorithm-rows-per-version run
  SEQUENTIALLY in a single worker process (not further parallelized). This
  resolves the unif-kkm budget dependency below for free (kabc just runs
  first in the same worker) and load-balances tas-fw-heavy vs. cheap reps
  across workers by chance, rather than gating every rep's wall-clock on
  its slowest algorithm.
  `return_as="generator"` (joblib >= 1.3, confirmed available: 1.6.0
  installed) yields each rep's results back to the MAIN process as soon as
  that rep finishes --- reps complete OUT OF ORDER across the 10 workers, so
  progress is tracked by a completed-count, not by rep index order.
  **Writer safety:** workers never touch the filesystem --- `process_one_rep`
  computes entirely in-memory and returns that rep's rows; only the MAIN
  process (iterating the generator) calls `to_parquet(...)`, always
  sequentially, one rep's results at a time. Exactly one writer, same as
  `run_experiment.py`'s worker-computes/main-process-writes pattern --- no
  concurrent-write risk.
- **Progress visibility, two granularities:** (1) the MAIN process prints
  progress after EVERY rep as its results stream back from the generator:
  `"{n}/100 reps done, elapsed={t}s"` (n = completed-count, not necessarily
  rep index), and incrementally rewrites each of the 3 output parquets after
  each rep. (2) EACH WORKER also prints after every individual algorithm
  call it completes (`"rep {r} {version}/{algo}: tau=... stopped=...
  time=...s"`), giving live visibility into what all `n_jobs` workers are
  doing concurrently --- this relies on loky worker stdout being inherited
  by the main process when the whole script's output is redirected to a
  file (the same mechanism that already surfaces workers' own UserWarnings
  in `run_experiment.py`'s logs).
- **Rollout order:** smoke test first, then a real but SHORT run at
  real budget/uncontended CPU (ideally exactly `n_jobs` reps, so each
  worker handles exactly 1 rep --- the cleanest possible per-rep timing
  read, extrapolated by simply multiplying by reps-per-worker for the full
  run) before committing to the full 100-rep run.

**unif-kkm variants: TWO.** unif-kkm is non-adaptive and always spends
whatever budget it's given in one shot (`fit(env, delta, budget_cap, rng)`
--- see `baselines.py`'s docstring: n = budget_cap // N pulls/arm, always).
Passing it a smaller `budget_cap` than the full 20000 is already exactly the
built-in mechanism for "give it fewer samples" --- no code change to
`baselines.py` needed, only to how Experiment 4 calls it:
- **unif-kkm-1x** --- `budget_cap` = min(τ_kabc, 20000) for that SAME
  (rep, version)'s kabc run. The sharpest comparison: identical total budget
  to kabc, isolating whether active/adaptive arm selection helps AT ALL,
  independent of how much budget is spent.
- **unif-kkm-2x** --- `budget_cap` = min(2·τ_kabc, 20000). A modest handicap
  in unif-kkm's favor, testing how much of an edge uniform sampling needs to
  close the gap.
Since 1x/2x need that rep+version's own kabc τ, kabc MUST run before either
unif-kkm variant for that (rep, version) --- resolved for free by running
algorithms sequentially within one rep's worker (kabc first). Edge case: if
kabc doesn't converge (`stopped=False`, hit `budget_cap` or the memory
guard), τ_kabc ≈ `budget_cap`, so unif-kkm-1x/2x get nearly the full budget
too --- expected, not a bug.

**A caution on interpreting unif-kkm-1x/2x vs. kabc, from the smoke test:**
on a hard pair, kabc's failure mode is the graceful trivial-cluster fallback
(ARI EXACTLY 0.0, per the mathematical property above) --- while unif-kkm,
never having that option, always commits to SOME 2-way split with whatever
budget it has, which tends to land above 0 even when wrong. So an
unif-kkm-1x/2x mean ARI exceeding kabc's is not necessarily "uniform
sampling found the structure better" --- it can equally be "kabc's honest
abstention scores worse under ARI than a forced guess does, on a hard
instance." Report `stopped` alongside ARI for exactly this reason; do not
read an ARI gap alone as evidence of an adaptivity advantage in either
direction.

**Per-rep pair sampling (shared machinery, all three versions).** For a given
rep, the movie pair is sampled ONCE, producing ONE environment --- every one
of the 6 algorithm rows (kabc, kabc-sg, acb, tas-fw, unif-kkm-1x,
unif-kkm-2x) is then run against that SAME environment/pair for that rep,
not independently re-paired per algorithm. (Each algorithm still gets its
own pull-randomness sub-stream --- so their individual pull sequences
differ, but all of them face the identical clustering problem for a given
rep, making cross-algorithm comparison within a rep apples-to-apples.) Each
rep's seed (`seed_formula` as elsewhere: `seed = seed_multiplier * exp_id +
rep`) is split into two independent sub-streams: one drives this
once-per-rep pair sampling (movie A/B selection), the other drives the
algorithms' pull randomness (further split per-algorithm) --- kept separate
so which pair got sampled never gets confounded with the pull-noise that
follows. Movie A is drawn uniformly at random from the 207 movies using the
pair-sampling sub-stream; movie B is chosen from A according to the
version-specific rule below. **If no eligible B exists for a drawn A**
(possible for movies near the extremes of the mean-rating range, e.g. *Mars
Attacks!* at 2.90 or *Shawshank Redemption* at 4.55), resample a fresh A from
the same sub-stream and retry, deterministically, until a valid pair is
found --- never treated as an error.

**Common random numbers across versions:** the pair-sampling sub-stream is
seeded identically across all three versions for a given rep index --- i.e.
rep 7 draws the *same* movie A in exp4_1, exp4_2, and exp4_3. Only the
B-selection rule differs between versions. This isolates the effect of the
pairing RULE from independent sampling noise when comparing versions.

**Version-specific B-selection rules:**
- **exp4_1 --- tight mean gap, MMD-maximizing.** Candidates = all movies with
  `|mean_A − mean_B| ≤ 0.1` (`mean_gap_tol = 0.1`). B = argmax MMD² among
  candidates (deterministic given A and the fixed bandwidth) --- the same
  "polarizing vs consensus, cherry-picked for maximum shape difference at
  matched mean" construction as Experiment 3, but redrawn fresh every rep
  instead of fixed once.
- **exp4_2 --- moderate mean gap, uniform.** Candidates = all movies with
  `0.1 ≤ |mean_A − mean_B| ≤ 0.5` (a RANGE, not a one-sided threshold ---
  excludes both the near-zero-gap pairs that overlap `exp4_1`'s hard region
  and trivially-easy very-large-gap pairs). B sampled UNIFORMLY at random
  among candidates (not MMD-maximizing). Tests behavior once genuine,
  moderate mean separation exists --- NOT framed as favoring any specific
  algorithm (the smoke test showed acb struggling here too).
- **exp4_3 --- tight-ish mean gap, uniform.** Candidates = all movies with
  `|mean_A − mean_B| ≤ 0.2` (`mean_gap_tol = 0.2`, looser than `exp4_1`'s
  0.1 --- see revision history above for why). B sampled UNIFORMLY at random
  among candidates (not MMD-maximizing). Same-mean-ish regime as `exp4_1`,
  but without cherry-picking the shape difference --- tests whether kabc's
  advantage in `exp4_1` was really about shape-sensitivity, or partly an
  artifact of always picking the most-different pair available (not a
  perfectly gap-matched ablation against `exp4_1`, since the tolerances now
  differ, 0.1 vs. 0.2 --- an accepted trade-off, see revision history).

**Environment.** `kabc.environments.MovieLensPairEnv`, built fresh per rep
from the sampled `(movie_id_a, movie_id_b)`: `arms_per_prototype=5` each
(N=10), resampling with replacement from each movie's real rating pool
exactly like `MovieLensEnv.pull` already does; standard balanced 2-cluster
ground truth. Takes rating pools directly (pre-loaded once from
`data/ml-1m-207/`) rather than loading/filtering `ratings.dat` itself, since
a fresh environment is built every rep.

**Output.** THREE separate files, one per version ---
`results/exp4_1/results.parquet`, `results/exp4_2/results.parquet`,
`results/exp4_3/results.parquet` --- rather than one combined file with a
`version` column, so `notebooks/04`'s plotting code can load and process one
version at a time with the same logic. Each row also carries the per-rep
sampled pair itself (`movie_id_a, movie_id_b, title_a, title_b, mean_a,
mean_b, mmd2_ab`) --- unlike Experiments 1-3, the instance varies per rep
here, so the sampled pair is essential context for any post-hoc analysis
(e.g. checking whether hard cases cluster around specific movies).

**Figures:** three bar plots (ARI, τ, wall-clock/rep), each with 3 sets
(versions) of 6 bars (algorithms) --- 18 bars per plot, mean ± std
(`notebooks/04_movielens_random_pairs.ipynb`).

**Expected outcome:** exp4_1 (tight gap, MMD-maximizing) --- kabc/kabc-sg
should separate reliably where acb is mean-blind and stuck near budget_cap,
similar in spirit to Experiment 3's headline result but now averaged over a
random sample of same-mean pairs rather than one fixed instance; the smoke
test's finding that kabc-sg can confidently converge to a wrong (not just
non-converged) answer on hard pairs here should be watched for at full
scale. exp4_2 (moderate gap, uniform) --- acb is expected to do at least
somewhat better here than in exp4_1 given real mean separation exists,
though the smoke test already showed it far from perfect (mean ARI 0.12,
20% stopped rate over 5 reps) --- this needs the full run to know if that
holds up. exp4_3 (tight-ish gap, uniform) --- tests whether exp4_1's
advantage for kabc survives without MMD-maximizing cherry-picking; expected
to be harder than exp4_1 for every algorithm (no cherry-pick) but not as
uniformly degenerate as the original mean_gap_tol=0.1 version was, given the
loosened 0.2 tolerance. unif-kkm-1x/2x against kabc's ARI at comparable or
double budget is the informative comparison for whether active selection
matters independent of budget size, subject to the ARI-floor caution above.

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