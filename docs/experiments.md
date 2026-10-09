# Experimental design for KABC (kernel-based active clustering)

Companion design document for the paper "Nonparametric Kernel Clustering with Bandit Feedback"
(Thuot, Vogt, Ghoshdastidar, Verzelen; arXiv:2601.07535).
It describes the four experiments as they are currently configured and run. Configs live in
`configs/`, drivers in `scripts/`, algorithm and environment code in `src/kabc/`, and figures are
produced by the notebooks in `notebooks/`.

Numbering conventions follow the revised version of the paper (to appear on arXiv): KABC is Algorithm 1, its CLUSTER subroutine is
Algorithm 2, the threshold is Eq. (4), s\* is Eq. (5), the budget bound is Eq. (6), and the main
result is Theorem 1. The paper presents three experiments: the paper's Experiment 3 covers both the
two-cluster MovieLens pairs (this document's Exp 3) and the many-cluster unbalanced settings (this
document's Exp 4).

All experiments use δ = 0.05 and are laptop-scale. Every algorithm implements
`fit(env, delta, budget_cap, rng) -> Result(partition, tau, stopped, history)`;
environments implement `pull(arm_index, rng)`.

## 0. Common infrastructure

### 0.1 Algorithms

| ID | Algorithm | Source | Used in |
|----|-----------|--------|---------|
| `kabc` | KABC with the variance-aware threshold (Eq. (4)) and doubling trick | this paper | Exp 1–4 |
| `kabc-2x`, `kabc-0.5x` | `kabc` with its calibrated constant doubled / halved (calibration ablation, not a budget variant) | this paper | Exp 1, 2 |
| `kabc-sg` | KABC with the kernel-uniform sub-Gaussian threshold (Section 3 ablation) | this paper | Exp 1, 2 |
| `acb` | ACB\* (Algorithm 2 of Thuot et al., arXiv:2406.11485): adaptive, mean-based (SRI + ADC) | Thuot et al. 2025 | Exp 1–4 |
| `acb-gaussian` | ACB Algorithm 1 (Appendix A): Gaussian-specialized, non-adaptive, reads Δ, θ, σ from the environment | Thuot et al. 2025 | Exp 1 only |
| `tas-fw` | Track-and-Stop with Frank–Wolfe, finite alphabet (Yavaş et al., arXiv:2409.05072) | Yavaş et al. 2025 | Exp 3 only |
| `unif-kkm` | Uniform sampling plus kernel k-means, at a budget matched to KABC's realized τ | folklore baseline | Exp 3, 4 |

Notes.

- `acb-gaussian` is evaluated with privileged information (Δ, θ, σ are read from the
  `GaussianMixtureEnv`), which is legitimate only for Exp 1 and is stated wherever its numbers
  are reported. It has no calibrated constant: every quantity is an exact quantile of a known
  distribution. It raises `TypeError` on any other environment.
- `tas-fw` is restricted to Exp 3 (N = 8, finite rating alphabet): its per-step cost grows
  combinatorially with N and K, which makes it infeasible for Exp 4 (N = 40/76) and inapplicable to
  continuous Gaussian observations (Exp 1, 2).
- `unif-kkm` runs after `kabc` in the same rep with budget `min(τ_kabc, budget_cap)`; it does not
  depend on the other baselines.
- **ACB\* partition at the cap.** When ACB\* hits `budget_cap` without certifying, it returns a
  best-effort partition: the largest SRI representative set found so far, with every arm assigned
  to its nearest representative on the pooled means (certified runs are unchanged). If SRI finds only
  one representative the partition is trivial and ARI is 0. This is always the case on Exp 2
  (identical means) and on most non-certified Exp 3 reps; it is not the case on Exp 4, where K = 7/12 and
  mean separation exists.
- Related active-clustering methods that were deliberately not included as baselines: BOC
  (Yang et al., 2024; per-round optimization too costly at this scale), BanditClustering
  (Graf et al., 2025; different observation model), Clustered-BAI (Yash et al., 2025; different
  objective), ATBOC and EBC (Chandran et al.; natural future additions).

### 0.2 Calibration of threshold constants

Theoretical threshold constants are conservative, so every method is run in a "practical" variant in which the
leading constant is replaced by `c_cal` times the theoretical shape. `c_cal` is calibrated once and frozen
(`configs/calibration.yaml`, written by `make calibrate` — never hand-edited) and reused unchanged in all experiments:

- Binary search for the smallest `c_cal ∈ [0.01, 1]` whose empirical error stays ≤ δ/2 over 100 runs
  (8 bisection steps), followed by validation on a fresh block of 100 runs with disjoint seeds.
  All runs use one fixed Exp 1 instance (Δ = 3), differing only in pull-noise seeds. (This is a reduced version
  of a 200-run/15-step protocol, chosen for laptop-scale compute.)
- Frozen values: `kabc` 0.1035, `kabc-sg` 0.1211 (manually raised from the raw bisection result 0.1176, which
  failed held-out validation), `acb` 0.659 (one shared multiplier for ACB's six constants). Held-out error is
  ≤ 0.01 for each.
- `tas-fw` cannot run on Gaussian instances, so it has its own protocol (`scripts/calibrate_tas_fw.py` →
  `configs/calibration_tas_fw.yaml`, const ≈ 0.241): synthetic finite-alphabet pair instances at three difficulty
  regimes, with a trajectory-recording trick. The held-out check did not fully generalize (10% error in the
  moderate regime), which is flagged in that file (`held_out_generalizes: false`).

### 0.3 Kernel and bandwidth

Gaussian RBF kernel, bandwidth from the median heuristic on a pilot of `n_pilot = 5` pulls per arm
(pooled pairwise distances), frozen after the pilot. Pilot pulls count toward the reported budget τ. For the MovieLens
experiments the bandwidth is computed once over the full 207-movie pool (= 1.0).

### 0.4 Metrics, seeds and protocol

- Per run: `tau` (budget used, pilots included), `stopped` (certified before `budget_cap`), `ari` (Adjusted Rand Index of the
  returned partition vs. truth; for non-stopped runs this is ARI at the cap), `error` (0/1 indicator), `n_iterations`,
  wall-clock time. Exp 4 adds per-pair ARI (`ari_close_pair`, `ari_far_pair`) and per-size misclassification rates.
- Repetitions: 100 per configuration for all four experiments (see status table below).
- Seeds: `seed = seed_multiplier · exp_id + rep` with `seed_multiplier = 1000`. Within a rep, each algorithm gets its own
  derived seed (Exp 2: `rep_seed·100 + algorithm_index`; Exp 4: `rep_seed·1000 + 100 + setting_index·10 + algorithm_index`
  with index 0/1/2 for kabc/unif-kkm/acb), so adding or re-running one algorithm never changes another's draws.
  Exp 2 budget caps come from a dedicated RNG (`CAP_SEED_OFFSET + exp_id·100000 + grid_point_index`).
- Budget cap: `200 · N · s*⁻² · log(N/δ)` with `s*` Monte-Carlo-estimated from the instance (Exp 1, 2); a fixed cap per
  experiment for the real-data experiments (Exp 3: 30000; Exp 4: 400000 / 800000). A run reaching the cap records
  `stopped = False`.
- Parallelism: `joblib` over repetitions (`n_jobs` in each config; multiprocessing backend, BLAS pinned to 1 thread).
- Results are written as parquet to `results/<exp>/results.parquet` (gitignored). Notebooks only load results and plot;
  they never recompute.
- Resume/append: each script reads `rep_start` from its config; if `rep_start > 0` and results exist, new reps are
  appended. Because seeds are functions of the rep index, appended reps are independent of existing ones. The
  grid/roster must be unchanged between runs.
- Smoke test: `SMOKE_TEST=1` caps reps at 5 and reduces sweeps to their end points.
- ACB-only passes: `scripts/run_experiment2_acb.py` and `scripts/run_experiment4_acb.py` re-run only `acb` on the same
  instances/caps/seeds as the existing runs and merge the rows into the results files (backups are written first, and
  non-ACB rows are asserted unchanged).

### Status of the generated results

| Experiment | Results file | Reps | Algorithms | Rows |
|---|---|---|---|---|
| 1 | `results/exp1/results.parquet` | 100 | 6 | 7200 (12 grid points) |
| 2 | `results/exp2/results.parquet` | 100 | 5 | 1500 (3 grid points) |
| 3 | `results/exp3_3/results.parquet` | 100 | 4 | 400 |
| 4 | `results/exp4_baseline/`, `results/exp4_amplified_imbalance/` | 100 | 3 | 300 each |

The `results_*_backup.parquet` files next to the results are safety copies made by the merge/rerun scripts.

---

## Experiment 1 — Gaussian mixtures: competitiveness in the parametric regime

**Claim.** In the Gaussian setting where mean-based methods are well suited, KABC remains competitive in budget and
accuracy, and the calibrated constant is not what makes the comparison work (calibration ablation).

**Instance.** N arms in K = 3 balanced clusters, observations N(μ_k, σ²I_d) with σ = 1 and cluster means
μ_k = Δ/√2 (e_{2k−1} + e_{2k}) (`GaussianMixtureEnv`). Base point: N = 60, d = 10, Δ = 3. One parameter is swept at a time:

| Sweep | Values |
|---|---|
| Δ | 1, 2, 3, 4 |
| d | 6, 10, 20, 50 |
| N | 30, 60, 90, 120 |

(d ≥ 2K = 6 is required by the mean construction.) The budget cap is the formula in §0.4, shared by all algorithms at a grid point.

**Algorithms.** `kabc`, `kabc-2x`, `kabc-0.5x`, `kabc-sg`, `acb`, `acb-gaussian`.

**Config / script.** `configs/exp1.yaml`, `scripts/run_experiment1.py`, `make exp1`. Figures: notebook 01
(`exp-gmm-main`, `exp-gmm-ablation`).

**Headline (100 reps, averaged over all grid points).** Every algorithm except `kabc-sg` has mean ARI ≥ 0.99 and certifies in
essentially all runs. Mean τ is ≈ 1.1k for `acb-gaussian` (privileged information), ≈ 7.2k for `acb`, ≈ 9.3k for `kabc`,
and ≈ 3.5k / 22k for `kabc-0.5x` / `kabc-2x`. `kabc-sg` is the fragile one: mean ARI 0.94, certifying in only 72% of runs
and with a much larger budget.

## Experiment 2 — Identical means, different shapes: nonparametric separation

**Claim.** When clusters share the same mean, mean-based methods (ACB) are structurally unable to separate them,
whereas KABC separates them through the kernel mean embedding.

**Instances (N = 40, K = 2, one grid point per family).**

- `scale_pair`: N(0, 1) vs. N(0, σ_B²), σ_B = 2 (same mean, different variance).
- `moment_matched_mixture`: N(0, 1) vs. ½N(−a, 1−a²) + ½N(a, 1−a²), a = 0.9 (same mean and variance, different shape).
- `ring_blob`: a 2-D ring (radius 2, noise 0.1) vs. N(0, I₂) (same center).

**Algorithms.** `kabc`, `kabc-2x`, `kabc-0.5x`, `kabc-sg`, `acb`. Budget cap from the formula in §0.4 with `s*` estimated per grid point.

**Config / script.** `configs/exp2.yaml`, `scripts/run_experiment2.py`, `make exp2`; ACB-only rerun with
`scripts/run_experiment2_acb.py` (`make exp2-acb`). Figures: notebook 02.

**Headline (100 reps).** `kabc` has mean ARI 1.0 and certifies in all runs at a mean τ of ≈ 17k–56k. `acb` never certifies
(ARI 0, budgets of 3.4·10⁵ – 2.0·10⁶ at the cap): with identical means its SRI step finds a single representative, so its
best-effort partition is trivial. `kabc-sg` is unreliable (mean ARI 0.26 on `moment_matched_mixture`, 0.78 on `scale_pair`),
consistent with the variance-aware bound being strictly tighter than the sub-Gaussian one when per-arm variance is small.
`kabc-0.5x` is slightly less reliable on `moment_matched_mixture` (ARI 0.91); `kabc-2x` is always correct but expensive.

## Experiment 3 — MovieLens random pairs

**Claim.** KABC's behavior holds up when the pair of clusters is a randomly resampled pair of real movies with no
cherry-picked shape difference and no easy mean separation, and it can be compared directly with TaS-FW on real data.

**Data.** `data/ml-1m-207/` (built by `make movielens-207`): the 207 MovieLens-1M movies with ≥ 1000 ratings. Ratings
live on {1, …, 5}; arm distributions are the empirical rating distributions of the movies.

**Instance.** Each rep draws movie A uniformly at random and movie B uniformly among movies with |mean_A − mean_B| ≤ 1.0
(version `exp3_3`; the only version kept). Each movie is replicated over 4 arms, giving N = 8 and K = 2.

**Algorithms.** `kabc`, `acb`, `tas-fw` (with its separately calibrated constant), `unif-kkm` (budget = τ_kabc).

**Budget cap.** 30000 for all algorithms. This clears KABC's doubling-trick round-7 threshold at N = 8 (28560), so KABC's
behavior on hard draws is not artificially truncated. The earlier 15000-cap results are kept as
`results/exp3_3/results_cap15000_backup.parquet` for reference only.

**Config / script.** `configs/exp3.yaml`, `scripts/run_experiment3.py`, `make exp3`. Figures: notebook 03
(`exp-movielens`).

**Headline (100 reps).** Mean ARI 0.93 for `kabc` (certifies in 93%), 0.98 for `tas-fw` (certifies in 88%), 0.86 for
`unif-kkm` at matched budget, and 0.15 for `acb` (certifies in 27%). `kabc` and `unif-kkm` use ≈ 9.1k pulls on average;
`tas-fw` ≈ 9.1k; `acb` ≈ 25k (it frequently reaches the cap).

## Experiment 4 — MovieLens prototypes: KABC's decision rule under real imbalance

**Claim.** KABC's decision rule (pairwise variance-aware threshold plus connected components) is more robust than k-means
on kernel features under heterogeneous difficulty and cluster-size imbalance. `unif-kkm` is run at KABC's own realized
budget, so any gap is due to the decision rule and not to sampling allocation. `acb` is run at the same cap as `kabc`.

**Resampling mechanism.** Every rep draws a fresh real-movie instance (instances depend only on the rep seed):

1. Close pair (`close_a`, `close_b`): a movie pair with |mean gap| ≤ 0.25, uniform among eligible partners.
2. Far pair (`far_large`, `far_small`): a pair with |mean gap| ≥ 0.5, with very different cluster sizes.
3. Generic anchors: uniform random draws from the remaining movies.
4. The roles within each pair are randomized per rep.

| Setting | N | K | Arms per role | Budget cap |
|---|---|---|---|---|
| `baseline` | 40 | 7 | far 12:2, close 5:4, generic 8, 6, 3 | 400,000 |
| `amplified_imbalance` | 76 | 12 | far 20:2, close 5:4, generic 10, 8, 7, 6, 5, 4, 3, 2 | 800,000 |

The fixed cap (not a formula) is deliberate: a hard random draw then ends as a graceful non-certified near-miss rather than
a runaway Gram-matrix computation (see "Known limitation").

**Memory guard.** `kabc.fit` refuses doubling-trick rounds whose per-worker Gram matrices would not fit in memory. This
experiment runs 5 workers with safety factor 2 (`configs/exp4.yaml: memory_guard`), verified by measurement (≈ 628 MB peak
per worker at the k = 8 scale). That raises the convergence ceiling to doubling round k = 8; round 9 would need ≈ 14.5 GiB at
5 workers and was not pursued. Experiments 1–3 run 10 workers but keep the default guard assumption (11 workers, factor 4).

**Algorithms.** `kabc`, `unif-kkm` (budget = min(τ_kabc, cap)), `acb`. `tas-fw` is infeasible at this size.

**Metrics.** The standard ones plus per-pair ARI and the misclassification rate on the arms of large vs. small prototypes
(Hungarian alignment of recovered to true clusters).

**Config / script.** `configs/exp4.yaml`, `scripts/run_experiment4.py` (kabc, unif-kkm), `scripts/run_experiment4_acb.py`
(acb; `make exp4-acb`). Figures: notebook 03 (`exp-movielens`).

**Status.** Complete: 100 reps per setting. Reps 0–49 were run first (≈ 18.7 h for kabc/unif-kkm) and reps 50–99 were
appended afterwards via `rep_start` in `configs/exp4.yaml` (existing rows are never overwritten); the full run is ≈ 38 h of
laptop time. The ACB rows for reps 0–49 came from the separate ACB-only pass (≈ 5 min); reps 50–99 ran ACB in the main script.
The KABC rounds at k = 8 dominate the cost.

**Headline (100 reps).**

| Setting | Algorithm | Mean ARI | Certified | Mean τ |
|---|---|---|---|---|
| `baseline` | `kabc` | 0.93 | 61% | 278k |
| `baseline` | `unif-kkm` | 0.64 | – | 278k (matched) |
| `baseline` | `acb` | 0.26 | 0% | 400k (cap) |
| `amplified_imbalance` | `kabc` | 0.91 | 33% | 689k |
| `amplified_imbalance` | `unif-kkm` | 0.50 | – | 689k (matched) |
| `amplified_imbalance` | `acb` | 0.21 | 0% | 800k (cap) |

(`unif-kkm` is a fixed-budget method and does not certify.) A large share of `kabc` reps end as non-certified near-misses at the
cap, with a still-high ARI.

## Known limitation: unbounded Gram-matrix cost on hard instances

`kabc`'s per-round MMD and RKHS-variance estimators materialize full n × n Gram matrices, with n growing along the doubling
schedule. On instances with very small `s*` (nearly equal distributions) the schedule would reach sizes that cannot be computed
(e.g., a configuration with `s*² ≈ 0.0024` was seen to need n ≈ 10⁵ – 10⁶). The current grids avoid this regime, and the
Exp 4 memory guard turns the issue into a clean non-certified result. A real fix (e.g., subsampling or incremental Gram updates
across rounds) would require an algorithmic decision and is an open problem.

## Repository layout

Workflow: `make data` → `make movielens-207` → `make calibrate` → `make smoke` (Exp 1–2, reduced) →
`make exp1 exp2 exp3 exp4` → run the notebooks for figures. See the README for the file tree and Makefile targets.
