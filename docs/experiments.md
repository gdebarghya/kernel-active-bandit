# experiments.md — Experimental design for KABC (kernel-based active clustering)

Companion design doc for the paper "Kernel-based Active Clustering of Bandit Arms"
(Thuot, Vogt, Ghoshdastidar, Verzelen). All experiments are laptop-scale.
Language: Python. One Jupyter notebook per experiment; shared logic in `src/`.

## 0. Common infrastructure

### 0.1 Algorithms to implement (src/kabc/algorithms/)

| ID | Algorithm | Source | Notes |
|----|-----------|--------|-------|
| `kabc` | KABC, variance-aware threshold (Eq. 6, builds on Wolfer & Alquier, "Variance-aware estimation of kernel mean embedding," JMLR 26(57):1–48, 2025) | our paper | main method |
| `kabc-sg` | KABC, kernel-uniform sub-Gaussian threshold (Discussion section ablation, substitutes Tolstikhin, Sriperumbudur & Muandet, "Minimax estimation of kernel mean embeddings," JMLR 18(86):1–47, 2017, Prop. A.1) | our paper, Discussion | ablation |
| `kabc-oracle` | Non-adaptive CLUSTER(n*, δ) with oracle n* | our paper | measures price of adaptivity (doubling trick) |
| `acb` | ACB\* (Algorithm 2), adaptive, mean-based (SRI + ADC) | Thuot et al. 2024 (arXiv 2406.11485) | re-implement; shared codebase style |
| `acb-gaussian` | ACB (Algorithm 1), Gaussian-specialized non-adaptive variant (Appendix A) — Δ, θ known | Thuot et al. 2024 (arXiv 2406.11485), Appendix A | Experiment 1 ONLY; no calibration (see below); NOT a generic registry entry |
| `tas-fw` | Track-and-Stop + Frank-Wolfe, finite alphabet | Yavas et al. (arXiv 2409.05072) | small instances only (N ≤ 20); Experiment 3 ONLY, not Exp-1 — see note there |
| `unif-kkm` | Uniform sampling (n per arm) + kernel k-means | folklore baseline | sweep n; non-adaptive, fixed-budget |

Design rule: every algorithm implements a common interface
`fit(env, delta, budget_cap, rng) -> Result(partition, tau, stopped, history)`.
Environments implement `pull(arm_index, rng) -> sample`. `acb-gaussian` is
the one exception noted in the table above: it matches this signature in
call SHAPE only, since Δ and θ are read directly off `env`
(`GaussianMixtureEnv`-only attributes) rather than estimated — see
`src/kabc/algorithms/acb_gaussian.py`'s own module docstring for the full
justification and its explicit non-oracle-cheat framing, and
`docs/results_exp1_exp2_exp3.tex`'s Baselines section for the paper-facing
statement of the same.

**`acb-gaussian` added (2026-09-15): the ACB paper's own Gaussian-
specialized, non-adaptive variant (Appendix A), for Experiment 1 only.**
Beyond the main-text `acb` (ACB\*, Algorithm 2, adaptive, Δ\*/θ\* unknown),
the paper's own Appendix A describes a SEPARATE variant used for its own
Figure 1 numerical experiment (the one reporting ≤1% error at δ=0.1):
Algorithm 1 (non-adaptive, Δ and θ assumed known), with every
concentration-inequality-derived constant (Eq. 9-13's `c1`-`c5`) replaced
by an EXACT δ-level quantile (`n_max` via a χ² quantile; `I`/`J` via a
normal quantile `β` and a Monte-Carlo product-quantile `α`) — see
`src/kabc/algorithms/acb_gaussian.py`'s module docstring for the exact
formulas, quoted directly from a full read of the rendered PDF (all 50
pages, including Appendix A/C), same verification standard as
`tas_fw.py`'s own formula citations. Implemented as a NINTH, independent
row in Experiment 1's algorithm roster (`configs/exp1.yaml`), with no
`-0.5x` self-comparison counterpart (not requested) and, deliberately, NO
calibrated constant — see `configs/calibration.yaml`'s own `acb-gaussian`
entry: every quantity here is an exact quantile of a known finite-sample
distribution, not a proof-derived bound with slack to tighten, so `make
calibrate` explicitly skips it by design, not by omission. It reads
Δ, θ, σ directly from the `GaussianMixtureEnv` instance rather than
estimating them from pulled data — legitimate specifically for Experiment 1
(these values are literally what the environment was constructed from, in
the same sense `N`/`K`/`d` already are known to every algorithm here), but
this DOES mean `acb-gaussian` is evaluated with privileged information the
other baselines don't get, and that fact is stated explicitly wherever its
numbers are reported (not left for a reader to infer) — the justification
is: this is the fairest, most literal reproduction of the specific
comparison the paper's own authors validated, not a general-purpose
competitor. Hard-fails immediately (raises `TypeError`) if ever called on
a non-`GaussianMixtureEnv` instance, rather than silently misapplying a
Gaussian-only method to Experiment 2/3's other families.

**Threshold formula verification (2026-09-15), against a fresh read of the
paper (arXiv:2601.07535).** `src/kabc/thresholds.py`'s
`variance_aware_threshold` and `subgaussian_threshold` were checked
directly against the paper's Eq. 6 and the Discussion section's substitute
formula and match exactly, term for term (the `32/3` coefficient in Eq.
6's second term; the `(sqrt(log_term)+2)` form of the sub-Gaussian bound).
Confirmed the paper uses two conceptually DISTINCT kernel constants that
happen to coincide numerically for the RBF kernel used throughout this
repo: g̃ (the kernel's *range*, sup − inf, used in Eq. 6) vs. ḡ (the
kernel's *supremum*, used in the sub-Gaussian formula) — both equal `1.0`
for RBF (inf → 0 as distance → ∞), but the code correctly keeps them as
separate named parameters (`g_tilde` vs. `g_bar`) rather than silently
conflating them, matching the paper's own distinction.

**Paper's own theoretical claims relevant to interpreting results (Section
4, "Benefit of variance-aware bounds").** Since the per-arm RKHS variance
𝒱ᵢ* never exceeds the kernel's global bound ḡ, the variance-aware bound is
provably never worse than the sub-Gaussian one, and can substantially
improve the budget whenever 𝒱ᵢ* ≪ ḡ — this is the paper's own
justification for why `kabc-sg` (the sub-Gaussian ablation) should be
expected to underperform `kabc` specifically on instances where per-arm
variance is informative and small relative to ḡ. Experiment 2's
`moment_matched_mixture` family (cluster A and B share mean AND variance,
differ only in higher-order shape — see Experiment 2 section below) is
exactly this regime: `kabc-sg`'s $83.2\%$ ARI-violation rate on that
family (79/95 `stopped=True` runs with ARI≠1, `results/exp2/results.parquet`,
50-rep run) is the paper's own analytically-anticipated worst case for the
sub-Gaussian threshold, not an unexplained empirical artifact — see
`docs/results_exp1_exp2_exp3.tex`'s Experiment 2 results section for the
full derivation.

**Paper's Gaussian/linear-kernel special case (Section 4) and Experiment
1's Δ-sweep.** Section 4 gives an explicit analytic condition: for
Gaussian arms under a linear kernel, KABC's budget bound matches the
known-optimal Gaussian linear-kernel rate of `acb`'s own paper (Thuot et
al. 2024) whenever ‖μᵢ−μⱼ‖ ≤ σⱼ²/√ḡ and the ambient dimension is moderate.
For Experiment 1's instances (σ=1 throughout, RBF kernel ḡ=1), this
condition reduces to Δ ≤ 1 — exactly the smallest, hardest point on
Experiment 1's own Δ-sweep grid ({1, 1.5, 2, 3, 4, 6}), sitting at the
boundary of the regime the theorem addresses rather than comfortably
inside or outside it. This is worth keeping in mind when interpreting
Experiment 1's Δ=1 results specifically (see that experiment's section
below and `docs/results_exp1_exp2_exp3.tex`) — the theorem concerns the
two algorithms' theoretical bounds, not either one's realized empirical τ,
so treat any observed budget-competitiveness there as consistent with,
not a literal restatement of, this special case.

**Related active-clustering methods NOT included as baselines (recorded
2026-09-15 for the paper's related-work framing; full paragraph-length
draft lives in `docs/results_exp1_exp2_exp3.tex`'s Baselines section).**
Beyond `acb`/`tas-fw` (both implemented) and `unif-kkm` (folklore, no
paper), five other papers address closely related active/bandit
clustering problems but were deliberately excluded, each for a specific,
non-dismissive reason:
- **BOC** (Yang, Zhong, Tan, "Optimal Clustering with Bandit Feedback,"
  JMLR 2024, arXiv:2202.04294): same fixed-confidence clustering problem,
  asymptotically optimal, but Track-and-Stop-style — solves a constrained
  optimization problem at EVERY round, making per-decision cost
  prohibitive at this benchmark's scale (same cost issue `tas-fw` has,
  which is why `tas-fw` is restricted to Experiment 3's small N=8
  instance).
- **BanditClustering** (Graf, Thuot, Verzelen, "Clustering Items through
  Bandit Feedback: Finding the Right Feature out of Many," ICML 2025,
  arXiv:2503.11209): a genuinely different observation model (selects an
  item AND a specific feature per round, not a full feature vector) — not
  directly comparable without reformulating the environment.
- **Clustered-BAI** (Yash, Karamchandani, Ghosh, "Near Optimal Best Arm
  Identification for Clustered Bandits," ICML 2025, arXiv:2505.10147):
  different objective entirely — identifying each of N agents' best arm
  (clustering is a means to that end via information-sharing), not
  partition recovery as the end goal.
- **ATBOC** (Chandran, Reddy, Bhashyam, "Online Clustering of Data
  Sequences with Bandit Information," IEEE ISIT 2025, arXiv:2501.11421):
  closest in spirit (same fixed-confidence M-arms-into-K-groups problem),
  not implemented in the current benchmark for lack of time, not
  incompatibility — a natural future addition as an additional parametric
  competitor alongside `acb` (different algorithmic structure: average
  tracking, vs. ACB's mean-based approach and BOC's Track-and-Stop).
- **EBC** (Chandran, Reddy, Bhashyam, "Efficient Clustering in Stochastic
  Bandits," arXiv:2601.09162, 2026 preprint): addresses the SAME
  per-round optimization bottleneck excluding BOC/limiting `tas-fw`, via a
  single gradient-like step per round instead of solving the full
  optimization problem, while retaining asymptotic optimality — emerged
  concurrently with this work; the natural next inclusion once available,
  not an implied criticism of BOC.

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
  `(N,K,d,Delta,sigma)`), matching how `configs/exp1.yaml`/
  `scripts/run_experiment1.py` use it (one fixed instance per grid point).
  So all 20 "instance
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
- **A fourth stop reason, `"degenerate_pilot"`, added later (during what was
  then Experiment 4's real run -- see this file's Experiment 3 section for
  the renumbering that later folded that experiment into slot "3")** for a
  distinct failure mode found in the pilot phase itself, not the doubling
  loop: `median_heuristic_bandwidth` can receive a pilot sample whose
  pairwise distances are all (numerically) zero, most often on real discrete
  data with a skewed rating distribution (a small pilot draw sampling only
  tied high-rating values by chance from a movie's real pool). It now falls
  back to the MEAN pairwise distance first (`on_degenerate="mean"`, with a
  `UserWarning` on trigger) -- still informative unless literally every
  pooled sample is identical -- and only then, if the mean is ALSO zero,
  does `_pilot_or_graceful_stop` return `Result(stopped=False,
  history=[{"stop_reason": "degenerate_pilot"}])` instead of raising. See
  `docs/results_exp1_exp2_exp3.tex`'s pre-2026-09-14-rework git history for
  the old Experiment 4 setup section (describes the pre-renumbering run)
  documenting the concrete crash this fixed -- that file was substantially
  reworked on 2026-09-14 to describe the CURRENT design/results instead, so
  this specific old section no longer exists in the file's current content.
- **The same failure mode, found again in `unif-kkm` and fixed the same way
  (during Experiment 3's redesigned real run, REVISION 4 config).**
  `baselines.py`'s `fit` had never received the `kabc.py` fix above --
  `_pooled_bandwidth` called `median_heuristic_bandwidth` with the default
  `on_degenerate="raise"`, and crashed the whole run (not just one rep --
  `run_experiment3.py`'s batched `Parallel()` calls are fail-fast, so an
  unhandled exception in one worker loses the entire in-flight batch, not
  only the rep that hit it) on a degenerate pooled sample from
  `unif-kkm-0.5x` -- a new, smaller-budget row this round, which draws
  fewer pulls per arm than any unif-kkm variant before it and so is more
  likely to land on an all-tied sample from a skewed-rating movie. Fixed
  identically in spirit: `_pooled_bandwidth` now uses
  `on_degenerate="mean"`; `fit` catches the `ValueError` this still raises
  if even the mean is degenerate and returns
  `Result(stopped=False, history=[{"stop_reason": "degenerate_bandwidth"}])`
  (named `_bandwidth` not `_pilot` -- unif-kkm has no separate pilot phase;
  the samples that fail to yield a bandwidth are the same ones `tau`
  already paid for and that clustering would otherwise use, so `tau` in
  this graceful-stop path still reflects the full `N*n` pulls actually
  made, unlike kabc's pilot-phase analogue). Two regression tests added
  (`tests/test_baselines.py`, mirroring `test_kabc.py`'s `_MostlyTiedEnv`/
  `_FullyDegenerateEnv` doubles), 151/151 passing. The crashed run's first
  10 completed reps (batch 1, written before the crash) were kept, not
  discarded -- see this file's Experiment 3 section for how that dataset is
  used pending a full re-run.

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
`_generous_budget_cap`) uses the exact formula `scripts/common.py`'s
`_resolve_budget_cap` uses for a real grid point (`constant=200`,
`s_star_sq` via `estimate_s_star_sq`), no margin. `SEARCH_HI` also widened back to `1.0` (from `0.6`) for this
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

- δ = 0.05 everywhere. Monte Carlo: **20 repetitions per configuration**
  (current first-pass `reps` for Experiments 1-3 alike, per each
  experiment's own `configs/expN.yaml` -- originally scoped as 100/50 here,
  scaled down for the current dedicated-script architecture's first real
  passes; see each experiment's own REDESIGN/REVISION note for its reps
  history). Seeds: `seed = 1000*exp_id + rep`.
- Metrics:
  - `tau`: stopping budget (adaptive methods). `tau = 0` specifically signals
    that `budget_cap` was too tight to even complete the pilot phase (no
    pulls were made) — distinct from a run that pilots successfully and is
    then capped mid-run, which always has `tau >= N*n_pilot`.
  - `error`: 1{partition != ground truth} averaged over reps (must be ≤ δ).
  - `ARI(budget)`: for non-stopping/capped methods, Adjusted Rand Index of the
    current partition estimate at budget checkpoints (log-spaced grid).
- Budget cap: `budget_cap = 200 * N * s_star^{-2} * log(N/delta)` when s* known
  (synthetic, Experiments 1-2), else a fixed value chosen per experiment at
  non-synthetic scale (Experiment 3's MovieLens random pairs: 15000 -- a
  formula-derived cap isn't meaningful at that scale, see Experiment 3's own
  section). A run hitting the cap records `stopped=False`, ARI at cap.
- **ARI-at-cap applies to ANY algorithm's `stopped=False` runs, not just ACB's**
  (ACB was simply the algorithm this design originally expected to hit that
  state routinely, being mean-blind on same-mean families -- it is not a
  metric restricted to ACB). `metrics.ari`/every experiment script's result
  row (`run_experiment1.py`, `run_experiment2.py`, `run_experiment3.py`)
  already compute `ari` unconditionally for every run regardless of
  algorithm or `stopped` value (checked directly, no code change needed) --
  a capped kabc or kabc-sg run reports a real ARI-at-cap exactly the same
  way a capped ACB run does. This matters in practice: pre-calibration (or
  at a hard grid point) kabc/kabc-sg CAN also end up `stopped=False`, and
  that run's ARI is still meaningful diagnostic information, not just a
  binary error=1 to discard.
- Parallelism: `joblib` over repetitions, `n_jobs=10` (each dedicated
  script's own `configs/expN.yaml` sets this explicitly; not a fixed
  `n_jobs=-2` across all cores). Backend forced to `multiprocessing`, not
  joblib's default `loky` -- an infrastructural workaround, not a design
  choice; see `scripts/common.py`'s module docstring. Every dedicated
  script (`scripts/run_experiment1/2/3.py`) reads a `SMOKE_TEST` env var
  (reps capped to `SMOKE_TEST_MAX_REPS=5`, sweeps reduced to their
  first/last value) for fast iteration -- NOT a notebook-level flag:
  notebooks only load `results/expN/*.parquet` and plot, they never
  recompute (see `CLAUDE.md`), so `SMOKE_TEST` only ever applies to a
  `make smoke`/`SMOKE_TEST=1 python -m scripts.run_experimentN` run.
- All results saved as parquet in `results/expN/`; notebooks only load + plot,
  never recompute silently.
- **Resume/append across separate runs (all three experiments,
  2026-09-14):** each dedicated script reads an optional `rep_start` from
  its config (default 0). Because each rep's seed is a direct function of
  its own rep index (`seed = seed_multiplier * exp_id + rep`, further split
  per algorithm), reps `[rep_start, reps)` are always independent, disjoint
  draws from reps `[0, rep_start)` regardless of which invocation computed
  them. If `rep_start > 0` and the target results file(s) already exist,
  the script loads them first and appends the newly-computed reps on top
  (same incremental-write pattern as always) rather than overwriting from
  scratch. Intended workflow: a fast pilot at a small `reps` (e.g. 10,
  `rep_start` unset/0), then a follow-up run with `rep_start` set to that
  pilot's `reps` and a larger final `reps` (e.g. `rep_start: 10, reps: 50`)
  to append the remaining reps without recomputing the pilot's. Assumes the
  grid/algorithm/version list is unchanged between the two runs -- changing
  the grid between a pilot and its follow-up produces a file with some
  points at fewer reps than others, silently, since nothing currently
  validates this. See each `scripts/run_experimentN.py`'s `main()` and the
  `rep_start` comment in each `configs/expN.yaml` for the exact mechanism
  (originally built for Experiment 3's crash-resume needs; generalized to
  all three scripts as a deliberate pilot-then-extend workflow).

---

## Experiment 1 — Gaussian mixtures: competitiveness in the parametric regime

**REDESIGN NOTE.** This experiment was substantially rewritten from its
original form. The original design ran through `run_experiment.py`'s
shared grid engine, had a separate restricted-budget pass
(`budget_multiplier ∈ {1.0, 0.25}`, applied uniformly to every algorithm's
shared budget_cap), swept `unif-kkm` over its own independent budget grid,
and reported an error-rate table, ARI-at-cap for capped runs, and an
appendix theoretical-vs-calibrated-constants panel. All of that is gone.
The new design (below) moved to a dedicated script
(`scripts/run_experiment1.py`, mirroring Experiment 3's architecture),
replaced the restricted-budget pass with per-algorithm
0.5x-of-its-own-realized-tau self-comparison rows (the same idea
Experiment 3 introduced for kabc, now generalized to kabc-sg and acb too),
ties `unif-kkm` to kabc's own tau instead of an independent sweep, and
drops the error table / ARI-at-cap / appendix panel entirely (the new
convergence-rate and ARI figures below make them redundant). See "Pipeline
architecture", "Algorithms", and "Figures" below for the concrete design;
the OLD design's results were described in a since-superseded version of
`docs/results_exp1_exp2_exp3.tex`'s Experiment 1 section (that file was
substantially reworked on 2026-09-14 to describe the CURRENT
design/results -- see git history for the old content).

**MULTIPLIER CHANGED from 0.9x to 0.5x (2026-09-14), after directly
inspecting per-rep `kabc`/`kabc-0.9x` rows in Experiment 2's data.** KABC's
doubling-trick schedule quantizes its own realized tau to a small set of
round-boundary checkpoints (n_k roughly doubles each round), and the FINAL
round alone typically costs roughly half of the total realized tau. A 0.9x
budget cut therefore almost always lands short of that final round
regardless of how close to 1.0 the multiplier nominally is -- there is no
way to construct a real "kabc re-run with 10% less budget" against this
schedule; 0.9x and, say, 0.6x test essentially the same thing (does the
PREVIOUS round's partition already resolve correctly), not a fine-grained
margin. Concretely, on Experiment 2's `scale_pair` sigma_B=2 data: kabc's
own realized tau only ever lands on 8920 or 19560 (two round checkpoints);
`kabc-0.9x` (budget = 0.9 * that rep's own tau) either gets stuck exactly
at the 8920 checkpoint when the base run converged at 19560 (sometimes
already correct there, sometimes not, ARI bimodal at 0/1), or falls back
even further (to 3760) when the base run converged at 8920 itself, always
scoring ARI 0.0 in that case. The unif-kkm baselines, by contrast, degrade
smoothly under the same cut (non-adaptive, continuous n = budget_cap // N),
which is why unif-kkm-0.9x looked deceptively strong relative to
kabc-0.9x's collapse -- the comparison was not apples-to-apples given
KABC's schedule. 0.5x makes what is actually being tested explicit -- a
meaningfully smaller, still-real budget cut -- rather than implying false
precision the doubling schedule cannot honor. All three experiments'
`-0.5x` rows (formerly `-0.9x`) were renamed together; see
`scripts/run_experiment1.py`'s module docstring for the identical
reasoning applied there.

**REVISION 6 (2026-09-15): `acb-gaussian` added as a ninth, independent
algorithm row.** Beyond `acb` (ACB*, Algorithm 2, adaptive, $\Delta_*$/
$\theta_*$ unknown), the ACB paper's own Appendix A describes a SEPARATE
variant, used for the paper's own Figure 1 numerical experiment (the one
reporting $\le 1\%$ error at $\delta=0.1$): Algorithm 1 (non-adaptive,
$\Delta$/$\theta$ assumed known), with every concentration-inequality-
derived constant (Eq. 9-13's `c1`-`c5`) replaced by an EXACT $\delta$-level
quantile ($n_{max}$ via a $\chi^2$ quantile; $I$/$J$ via a normal quantile
$\beta$ and a Monte-Carlo product-quantile $\alpha$) -- see
`src/kabc/algorithms/acb_gaussian.py`'s module docstring for the exact
formulas (quoted directly from a full read of the rendered PDF, same
verification standard as `tas_fw.py`'s own citations) and §0.1's algorithm
table entry. Implemented as `acb-gaussian`, Experiment 1 ONLY (hard-fails
on any non-`GaussianMixtureEnv` instance), reading $\Delta$/$\theta$/
$\sigma$ directly off the environment rather than estimating them --
legitimate here specifically because `GaussianMixtureEnv` is *constructed
from* these exact values, so nothing is being estimated from data that a
fair comparison would otherwise require estimating (see §0.1's algorithm
table note and `docs/results_exp1_exp2_exp3.tex`'s dedicated
"ACB-Gaussian" baselines subsection for the full non-oracle-cheat framing,
stated explicitly rather than left for a reader to infer).

**No `-0.5x` self-comparison row for `acb-gaussian`, deliberately, not an
oversight.** Every other base algorithm's `-0.5x` row (`budget_cap =
min(0.5*tau_own, budget_cap)`) measures how much MARGIN that algorithm's
own stopping rule carries beyond what a correct answer needed -- margin
that exists because kabc/kabc-sg/acb's thresholds are conservative
concentration-inequality bounds with real, calibratable slack (see §0.2's
`c_cal` protocol). `acb-gaussian` has no such slack: every constant is an
exact quantile, not a bound with room to tighten (see its own CALIBRATION
note in `acb_gaussian.py` and `configs/calibration.yaml`'s dedicated
`acb-gaussian: null` entry, generated by `scripts/calibrate_constants.py`
itself as a permanent, by-design skip -- not a "not yet calibrated"
placeholder like `tas-fw`'s). A `-0.5x` row here would not test "how
conservative is this threshold" the way it does for the other three; it
would only show that halving an already-tight budget breaks the algorithm,
a different and far less informative question.

**Merge mechanism (avoiding a repeat of the earlier $N=90$ indexing
issue).** `acb-gaussian` was added to an already-complete 50-rep,
8-algorithm Experiment 1 dataset via a scratch-run-and-merge, run OUTSIDE
`rep_start`'s resume/append mechanism (which assumes the algorithm list is
unchanged between invocations and cannot backfill a newly-added row's
reps 0-49). Two indexing pitfalls were identified and avoided/fixed, both
the same underlying class of bug as the earlier $N=90$ merge:
1. **Grid-point indexing:** the scratch run reused the SAME `instances`/
   `sweeps` structure as the real config (same 14 grid points, same
   enumeration order), so `grid_point_index` -- and therefore the
   Monte-Carlo `s_star`/`budget_cap` seed (`CAP_SEED_OFFSET + exp_id*
   100_000 + grid_point_index`) -- matched exactly, verified directly by
   recomputing all 14 grid points' `budget_cap` fresh and comparing
   against every base algorithm's own stored value before merging.
2. **Algorithm-index seeding (new, not present in the $N=90$ case):**
   `acb-gaussian`'s per-rep RNG seed (`algo_seed = rep_seed*100 +
   algo_idx`) had to use `algo_idx=8` -- its position in the canonical
   bases-first-reordered 9-algorithm list (see
   `scripts/run_experiment1.py`'s `_reorder_bases_first`) -- NOT `0` (its
   position if run alone in an isolated scratch list), which would have
   collided with `kabc`'s own per-rep seed at every single rep.
3. **A genuinely separate, pre-existing inconsistency this check
   incidentally surfaced:** $N=120$'s `kabc`/`kabc-sg`/`acb` base rows had
   two different `budget_cap` values across the pilot (reps 0-9,
   `grid_point_index=12`, computed BEFORE $N=90$ was permanently added to
   the sweep) vs. the extension (reps 10-49, `grid_point_index=13`, AFTER)
   -- a ~1.2% drift, unrelated to `acb-gaussian` but caught by the same
   verification pass. Confirmed zero practical effect (every algorithm's
   realized $\tau$ at $N=120$ is far below both candidate cap values, so
   neither was ever binding) and patched for provenance consistency:
   reps 0-9's stored `budget_cap` for those three algorithms at $N=120$
   was overwritten to the correct, current value (matching reps 10-49).

**Claim tested:** in the mean-separated Gaussian setting (home turf of ACB/BOC),
KABC's budget is within a modest factor of ACB; budget scales as s*^{-2} log(N/δ)
(Theorem 3.1); variance-aware threshold beats sub-Gaussian threshold. Also
(new): how conservative each adaptive method's own stopping rule is, and
whether active/adaptive sampling beats naive uniform sampling at a matched
budget.

**Instances.** N arms, K clusters (balanced), arm i in cluster k has distribution
N(mu_k, sigma^2 I_d), sigma = 1. Cluster means: mu_k = Delta/sqrt(2) * (e_{2k-1} + e_{2k})
so pairwise separation is exactly Delta (equidistant means). Unchanged from
the original design.

**Grid (one knob at a time, others at base) — unchanged, 13 grid points:**
- Base: N = 60, K = 3, d = 10, Delta = 3.
- Sweep Delta in {1, 1.5, 2, 3, 4, 6} (main plot).
- Sweep d in {6, 10, 50, 100} at Delta = 3. (d=2 excluded: GaussianMixtureEnv's mean construction requires d ≥ 2K = 6 for K=3, to place K orthogonal mean-pairs.)
- Sweep N in {30, 60, 120} at Delta = 3.

**Pipeline architecture.** Moved to `scripts/run_experiment1.py` (config:
`configs/exp1.yaml`), a dedicated script, not `run_experiment.py`'s shared
grid engine — the new algorithm roster below has real cross-algorithm tau
dependencies within a (grid point, rep) that don't fit a flat,
order-independent work-item list (the same reason Experiment 3 already has
its own script). Reps are the parallelization unit (`n_jobs`-sized batches,
mirroring Experiment 3): the ENVIRONMENT is still built ONCE PER GRID
POINT, not per rep — `GaussianMixtureEnv`'s cluster means are a
deterministic closed-form function of `(N,K,d,Delta,sigma)`, no randomness
at all, so every rep at a grid point faces the identical instance; only
each algorithm's own pull-noise rng substream differs by rep. Within one
rep's worker, all 8 algorithm rows (below) run SEQUENTIALLY, needed for the
tau-dependency chains. (Experiment 2 later adopted this identical roster
and moved to its own dedicated script too, at which point `run_experiment.py`
had zero remaining users and was deleted outright -- see Experiment 2's
own REDESIGN NOTE and `scripts/common.py`'s module docstring.)

**Algorithms: 9 rows per (grid point, rep).** Three independent BASE
algorithms — `kabc`, `kabc-sg`, `acb` — each run once at the grid point's
full `budget_cap`. Each base is then, separately, re-run at
`budget_cap = min(0.5 * its_own_realized_tau, budget_cap)`, with a FRESH
independent RNG substream (not the base run's exact same pulls truncated
early — seed reuse was considered and deliberately not done, same reasoning
as Experiment 3: results are averaged over reps anyway) — `kabc-0.5x`,
`kabc-sg-0.5x`, `acb-0.5x` respectively, each testing how conservative
THAT algorithm's own stopping rule is. `unif-kkm-1x` and `unif-kkm-0.5x`
are tied specifically to `kabc`'s own tau (not kabc-sg's or acb's, even
though those have their own 0.5x rows) — `budget_cap = min(1.0 or 0.5 *
tau_kabc, budget_cap)` — testing whether active/adaptive selection beats
naive uniform sampling at a matched budget. `unif-kkm` no longer sweeps its
own independent n-grid (the original design); this ties it to a real
realized adaptive budget instead. No `tas-fw` here: its finite-alphabet
assumption doesn't hold for these continuous Gaussian instances; it runs
only in Experiment 3, where the assumption holds natively (MovieLens
ratings on {1..5}). `acb-gaussian` (REVISION 6 above) is a ninth,
independent base row with no `-0.5x` counterpart and no calibration — see
that note for the full reasoning.

**Budget_cap: one value per grid point, shared across all 8 algorithm rows
and every rep** — `s_star_formula`, `constant=200`, exactly the paper's
proven bound (Theorem 3.1): `constant * N / s_star**2 * log(N/delta)`.
`s_star^2` itself is NOT computed in closed form — it's a Monte Carlo
estimate (`kabc.metrics.estimate_s_star_sq`, 3000-sample unbudgeted oracle
draw per cluster, one dedicated fixed-seed rng per grid point, independent
of the experiment's own per-rep seed stream) — a true closed-form `s*^2` is
in principle derivable for isotropic Gaussians under the RBF kernel but is
not implemented; the MC estimate is treated as accurate enough and is
unchanged from the original design. No `budget_multiplier` dimension
anymore — the restricted-budget question it existed to answer is now asked
per-algorithm instead, via the 0.5x rows above.

**Figures — three per sweep (Delta, d, N), 9 total:**
1. **Stopping budget** (τ) vs. sweep parameter, `kabc`/`kabc-sg`/`acb` only
   (no unif-kkm, no 0.5x rows in this figure). For the Delta sweep: dashed
   overlay of the theoretical slope `s*^{-2}` (Theorem 3.1), the design's
   original headline claim. The d/N sweeps show τ vs. parameter without an
   analytic overlay (no simple closed-form slope is claimed for those).
2. **Convergence rate** (bar plot, new): fraction of reps with
   `stopped=True`, `kabc`/`kabc-sg`/`acb` only. One block per sweep-config
   (grid point), 3 bars per block (one per algorithm) — mirrors
   Experiment 3's convergence-rate plot.
3. **ARI** (mean ± std over reps, new): ALL 8 algorithm rows. Lets a reader
   see whether the 0.5x variants' ARI drop tells us the bounds/stopping
   rules are conservative (a small drop = little margin was needed; a large
   drop = the extra half of budget bought real accuracy), and how kabc/kabc-sg/acb
   compare against unif-kkm at matched budget.

**Expected outcome:** acb < kabc < kabc-sg in budget, gaps modest; kabc's
error stays at (or near) 0 across the grid, consistent with prior runs of
this instance family. The new comparisons (0.5x rows, unif-kkm-1x/0.5x)
are this redesign's actual new questions — whether kabc's stopping rule
margin is similar in size to kabc-sg's/acb's, and whether active selection's
edge over naive uniform sampling holds at every point in the grid or only
some — genuinely open until a real run is available.

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

**REDESIGN NOTE.** This experiment was substantially rewritten, matching
Experiment 1's redesign almost exactly: it moved to a dedicated script
(`scripts/run_experiment2.py`), adopted the identical 8-algorithm roster
and tau-dependency structure, dropped `kabc-oracle` and the "price of
adaptivity" claim/figure entirely, dropped the old restricted-budget
(1.0x/0.25x) pass in favor of per-algorithm 0.5x-of-its-own-tau
self-comparison rows, and shrank the grid (10 points, down from 21). See
"Pipeline architecture", "Algorithms", and "Instance families" below for
the concrete design; the OLD design's results were described in a
since-superseded version of `docs/results_exp1_exp2_exp3.tex`'s Experiment 2
section (that file was substantially reworked on 2026-09-14 to describe the
CURRENT design/results -- see git history for the old content).

**MULTIPLIER CHANGED from 0.9x to 0.5x (2026-09-14):** identical change and
identical reasoning as Experiment 1's own REDESIGN NOTE above (KABC's
doubling-trick round-quantization means 0.9x cannot actually test "10% less
budget") -- see that note for the full evidence, which was in fact
discovered by inspecting THIS experiment's own data (`scale_pair`
sigma_B=2).

**GRID TRIM NOTE (2026-09-14, follow-up to the REDESIGN above).** The
10-point grid was trimmed further, to exactly **6 points**: `scale_pair`
sigma_B in {2,3} (dropped 4), `moment_matched_mixture` a in {0.85,0.9}
(dropped 0.8,0.95), `ring_blob` ring_radius in {2,3} (dropped 4). This was
decided alongside, but is a SEPARATE decision from, the plotting redesign
below (see "PLOTTING REVISION NOTE") -- the 6 settings are now the
CANONICAL grid in `configs/exp2.yaml` itself (not just what the notebook
happens to plot out of a larger grid), `reps` raised from 20 to 50 to match
the actual first full production run at this trimmed grid. Rationale:
these 6 settings are meant as 6 distinct illustrative examples rather than
points on a sweep whose in-between variation matters (matching the
plotting redesign's own reasoning), and a=0.8 in particular was the single
slowest point in the whole original grid (`acb` alone ran to tau~8M there
without converging in the 10-rep pilot) -- dropping it also meaningfully
cuts wall-clock for the reps increase.

**Claim tested:** when clusters share the same mean, mean-based methods are
blind (ARI at chance for any budget) while KABC recovers the partition with
budget governed by the MMD gap. Also (new, matching Experiment 1): how
conservative each adaptive method's own stopping rule is, and whether
active/adaptive sampling beats naive uniform sampling at a matched budget.
The old "price of adaptivity" claim (kabc vs kabc-oracle) is DROPPED --
`kabc-oracle`'s one-shot required budget exceeded the shared budget_cap
formula at all 21 of the old grid points (0% convergence, always), so that
figure never actually produced real data under the old design either;
removing it is not a loss of working signal.

**Instance families (d = 1 unless stated) -- 6 grid points total (see GRID
TRIM NOTE above; was 10, before that 21):**
- (2a) **Scale pair:** cluster A = N(0, 1), cluster B = N(0, sigma_B^2),
  sigma_B in {2, 3} -- 2 points. `sigma_B=1.5` DROPPED in the original
  REDESIGN: it was the least-separated (closest to A's own variance) and
  by far the slowest point in the old 4-value sweep. `sigma_B=4` DROPPED
  in the later GRID TRIM.
- (2b) **Moment-matched mixture:** cluster A = N(0,1),
  cluster B = 0.5 N(-a, b^2) + 0.5 N(a, b^2) with b^2 = 1 - a^2 (matches mean AND
  variance), a in {0.85, 0.9} -- 2 points. Only shape differs. The old
  design's extended 0.25x-only range (a in {0.5,0.6,0.7}) was dropped in
  the original REDESIGN -- it existed purely to get ARI-at-cap signal at
  grid points that hit the Gram-matrix memory guard even at full budget
  (see "Known limitation" below); with the restricted-budget pass gone
  entirely, that range no longer served the purpose it was added for.
  `a=0.8`/`a=0.95` DROPPED in the later GRID TRIM (`a=0.8` was also the
  single slowest point in the whole grid: `acb` alone ran to tau~8M there
  without converging).
- (2c) **2D geometry:** ring (Gaussian noise 0.1, radius swept) vs N(0, I_2)
  blob, both centered at origin. ring_radius in {2, 3} -- 2 points, up
  from a single fixed ring_radius=2 config in the original REDESIGN (which
  made it a genuine sweep, larger radius = more geometric separation from
  the blob, matching 2a/2b's "one knob controls separation" shape).
  `ring_radius=4` DROPPED in the later GRID TRIM.

N = 40, K = 2, balanced, unchanged. **50 reps** (raised from the original
REDESIGN's 20, matching Experiment 1's canonical first-pass reps at the
time -- see GRID TRIM NOTE above for why reps went up alongside the grid
shrinking down).

**Pipeline architecture.** `scripts/run_experiment2.py` (config:
`configs/exp2.yaml`) -- a dedicated script, not a shared grid engine (there
is no shared grid engine anymore; see Experiment 1's "Pipeline
architecture" note and `scripts/common.py`'s module docstring). Reps are
the parallelization unit, exactly like Experiment 1: the ENVIRONMENT is
built ONCE PER GRID POINT (none of scale_pair/moment_matched_mixture/
ring_blob's constructors take an rng or seed -- their instance geometry is
a deterministic closed-form function of their kwargs), and within one
rep's worker, all 8 algorithm rows run SEQUENTIALLY, needed for the
tau-dependency chains. The one structural difference from Experiment 1:
this experiment has THREE instance families (each its own environment
class), not one, so the grid-point loop iterates over all three, the way
the old shared engine used to.

**Algorithms: 8 rows per (grid point, rep), IDENTICAL to Experiment 1's.**
Three independent BASE algorithms -- `kabc`, `kabc-sg`, `acb` -- each run
once at the grid point's full `budget_cap`. Each base is then, separately,
re-run at `budget_cap = min(0.5 * its_own_realized_tau, budget_cap)`, with
a FRESH independent RNG substream -- `kabc-0.5x`, `kabc-sg-0.5x`,
`acb-0.5x` respectively, each testing how conservative THAT algorithm's
own stopping rule is. `unif-kkm-1x` and `unif-kkm-0.5x` are tied
specifically to `kabc`'s own tau, testing whether active/adaptive
selection beats naive uniform sampling at a matched budget. `kabc-oracle`
is GONE (see REDESIGN NOTE above). No `tas-fw` here, same reason as
Experiment 1: its finite-alphabet assumption doesn't hold for these
continuous instances.

**Budget_cap: one value per grid point, shared across all 8 algorithm rows
and every rep** -- identical mechanism to Experiment 1: `s_star_formula`,
`constant=200`, `s*^2` Monte-Carlo-estimated. No `budget_multiplier`
dimension anymore.

**PLOTTING REVISION NOTE (2026-09-14).** The Figures below were substantially
redesigned from per-family sweep line plots (3 plots per family x 3
families = 9, plus a 3-panel headline) to a fixed **6-setting bar-plot
structure**, mirroring Experiment 3's "3 versions" block convention
exactly. Rationale: the variation WITHIN a family as its swept parameter
changes (the way Experiment 1's Delta/d/N sweeps are the whole point) isn't
the interesting signal here -- what matters is contrasting 6 distinct,
illustrative instances as their own examples, not a continuous trend. The 6
settings plotted, `scale_pair` sigma_B in {2,3}, `moment_matched_mixture` a
in {0.85,0.9}, `ring_blob` r in {2,3}, are now ALSO the canonical grid
itself (see the GRID TRIM NOTE above, which resolves what was originally an
open question here) -- `configs/exp2.yaml` and
`notebooks/02_same_mean_shapes.ipynb`'s `SETTINGS` constant now agree
exactly, nothing is computed and left unplotted.

**Figures -- 4 total: 1 headline (6 panels) + 3 grouped bar plots (6 blocks each):**
1. **Headline (1 figure, 6 panels in a single row, one per setting):** the
   actual instance geometry for each of the 6 kept settings individually --
   `scale_pair`/`moment_matched_mixture` panels are exact analytic densities
   (A's N(0,1) vs. that setting's B); `ring_blob` panels are real 2D sample
   scatters from `RingBlobEnv` itself (blob vs. that setting's ring), so
   they stay in sync with the actual data-generating code.
2. **Budget** (tau): one bar plot, 6 blocks (settings) x 4 bars (`kabc`,
   `kabc-sg`, `acb`, plus a `theoretical bound` bar -- the grid point's
   formula-based `budget_cap`, a single deterministic value with no error
   bar since it doesn't vary by rep). No unif-kkm, no 0.5x rows in this
   figure -- mirrors Experiment 1's Figure 1 restriction to base algorithms.
3. **Convergence rate:** one bar plot, 6 blocks x 3 bars (`kabc`/`kabc-sg`/
   `acb` only) -- fraction of reps with `stopped=True`, mean ± std.
4. **ARI:** one bar plot, 6 blocks x 8 bars (ALL algorithm rows) -- mean ±
   std over reps. Same reading convention as Experiment 1's ARI figure: a
   small ARI drop from a base algorithm to its own -0.5x variant means
   little margin was needed; a large drop means real accuracy was still
   being bought in that last half of budget.

**Expected outcome:** acb at chance ARI at every budget for all three
families (mean-blind on same-mean data, by construction); kabc exact
recovery with tau tracking the MMD gap. The 0.5x rows and unif-kkm
comparisons are, like Experiment 1, this redesign's genuinely new
questions -- whether kabc's stopping-rule margin looks similar across all
three families, and whether active selection's edge over naive uniform
sampling holds throughout the (now narrower, easier) kept grid -- open
until a real run is available.

---

## Experiment 3 — MovieLens random pairs: robustness across the real-data pair distribution

**RENUMBERING NOTE.** This experiment was originally built and run as
"Experiment 4", alongside a separate, older "Experiment 3" (one fixed,
cherry-picked K=4 prototype instance, plus a small N=6 reduced instance built
just for tas-fw). The old Experiment 3 has been **dropped entirely**: the
whole point of keeping MovieLens in this benchmark is testing against tas-fw
on real data, which this random-pair design already does, and does more
robustly (averaged over many sampled pairs rather than one cherry-picked
instance) than the old fixed instance did. This experiment now occupies the
"3" slot; `configs/exp3.yaml`/`scripts/run_experiment3.py` are dedicated to
this design. REVISION HISTORY below (kept verbatim from when this was
"Experiment 4") uses that old `exp4_N` naming for anything predating this
renumbering; **REVISION 4** at the end of this history is where the
renumbering and this round of config changes are recorded. Current version
names are `exp3_1`/`exp3_2`/`exp3_3`.

**Claim tested:** whether KABC's advantage (and ACB's/TaS-FW's relative
behavior) holds up when the movie PAIR is randomly resampled every rep
across the real diversity of the MovieLens catalog, under three distinct
pairing regimes that vary how much mean separation is present and whether B
is cherry-picked (max MMD²) or sampled uniformly. Also (new, REVISION 4):
how conservative KABC's own stopping rule is, by re-evaluating it at a
fraction of the budget it actually used.

**Data.** `data/ml-1m-207/` (`ratings.parquet`, `movies.parquet`) — the 207
MovieLens-1M movies with ≥ 1000 ratings, pre-filtered by
`scripts/build_movielens_207.py` so per-rep pair sampling doesn't re-load/re-filter
the full ~1M-row `ratings.dat` every time. `movies.parquet` carries
`movie_id, title, n_ratings, mean_rating`; `ratings.parquet` carries the raw
per-movie ratings needed to build each rep's environment. Discrete rating
support {1..5}, RBF-on-ratings kernel, fixed median-heuristic bandwidth
computed once over the full 207-movie pool (bandwidth = 1.0, not recomputed
per pair).

**REVISION HISTORY (kept here rather than silently overwritten at each
step, since the evidence that motivated each change is real, not
speculation):**
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
- **REVISION 3 (post-real-run): N reduced to 8, reps to 50 (40 completed
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
  `src/kabc/algorithms/kabc.py`): `median_heuristic_bandwidth` now accepts
  `on_degenerate="mean"`, falling back to the mean pairwise distance (still
  informative unless *every* pair is tied) before raising, and
  `_pilot_or_graceful_stop` uses this fallback with a `UserWarning` on
  trigger; if the mean is *also* degenerate, KABC now stops gracefully
  (`STOP_REASON_DEGENERATE_PILOT`, `stopped=False`) rather than crashing,
  matching the existing budget-cap/memory-guard graceful-stop convention.
  Verified directly against the exact crash case (rep 38, exp4_1): KABC now
  converges correctly instead of crashing. One rep (38) remained permanently
  absent from that dataset --- skipped (not re-run) under the pre-fix "just
  resume" decision, before the fix existed. Final dataset from that run: 40
  reps/algorithm/version (0-37, 39-49), 240 rows per version file. Results
  and per-version summary stats from that run were in a since-superseded
  version of `docs/results_exp1_exp2_exp3.tex`'s "Results --- Experiment 4"
  section (compiled PDF) --- describes the OLD `exp4_N` results, now
  superseded by REVISION 4's fresh run under the new config below (that
  file was itself substantially reworked on 2026-09-14 -- see git history
  for the old content).
- **REVISION 4 (this round): renumbered to Experiment 3 (see RENUMBERING
  NOTE above), old fixed-instance Experiment 3 dropped, and the algorithm
  roster/version thresholds/scale all changed for a fresh run.**
  - `budget_cap` lowered from 20000 to **15000**. Deliberately NOT raised to
    clear kabc's doubling-trick round-7 threshold (28560 at N=8, δ=0.05,
    per the "Why N cannot grow" arithmetic below): kabc's own non-convergent
    cutoff (τ=13832) is IDENTICAL for any `budget_cap` in `[13832, 28559]`
    (both 15000 and the old 20000 sit in this same range), so this change
    does not alter kabc's own τ/stopped/ARI numbers, or the tau_kabc-derived
    budgets of kabc-0.9x/unif-kkm-1x/unif-kkm-0.9x below (all bounded by
    kabc's own τ, which under non-convergence is always 13832 regardless of
    which cap in that range is configured). What it DOES do is give acb and
    tas-fw --- both of which consume budget continuously rather than in
    kabc's big geometric jumps --- less runway, intentionally: kabc's
    behavior was already fine in the prior (`exp4_N`) run, and the goal here
    is not spending extra wall-clock feeding acb/tas-fw more budget.
  - **`kabc-sg` and `unif-kkm-2x` DROPPED.** kabc-sg's fragility /
    confident-wrong-answer failure mode (see "Smoke-test findings" below,
    and §6.13 of a since-superseded version of `docs/results_exp1_exp2_exp3.tex`
    -- reworked on 2026-09-14, see git history for that section's old
    content) is already thoroughly documented on this exact real-data
    setting from the prior run and wouldn't add new information;
    unif-kkm-2x never showed much beyond what -1x already showed in that
    same prior run.
  - **Two new rows added: `kabc-0.9x` and `unif-kkm-0.9x`**, both testing
    how conservative kabc's stopping rule is. `kabc-0.9x` is a second,
    independent `kabc` run per (rep, version) at
    `budget_cap = min(0.9 * τ_kabc, budget_cap)`, where `τ_kabc` is that
    same (rep, version)'s primary kabc row's own realized τ --- i.e. "what
    does kabc's partition estimate look like with 10% less budget than it
    actually decided it needed?" Uses a FRESH, independent RNG substream
    (not the primary kabc run's exact same pulls truncated early) ---
    considered reusing the same seed for an exact "stopped-early" replay,
    but decided against it, since results here are averaged over many reps
    anyway, so an independent draw at the reduced budget is simpler and
    equally informative in aggregate. `unif-kkm-0.9x` is the analogous
    0.9x-of-τ_kabc budget for the non-adaptive baseline, alongside
    `unif-kkm-1x` (unchanged). 6 algorithm rows total now: kabc, kabc-0.9x,
    acb, tas-fw, unif-kkm-1x, unif-kkm-0.9x. `kabc` must run before ANY of
    the three τ_kabc-dependent rows, for a given (rep, version) --- enforced
    in `scripts/run_experiment3.py`'s `main()` regardless of the config's
    own algorithm order. **(Multiplier later changed to 0.5x -- see REVISION
    5 below; kept at 0.9x here since that is what REVISION 4 itself actually
    introduced.)**
  - **Version thresholds changed, all three versions.** `exp3_1` (was
    `exp4_1`): `mean_gap_tol` raised from 0.1 to **0.25** (still `max_mmd`).
    `exp3_2` (was `exp4_2`): range raised from `[0.1, 0.5]` to
    **`[0.25, 1.0]`** (still uniform), to give acb more real mean separation
    to work with, per the prior run's finding that it was still weak (mean
    ARI 0.089, 17.5% stopped) even in the old moderate-gap range. `exp3_3`
    (was `exp4_3`): `mean_gap_tol` raised from 0.2 to **0.25** (still
    uniform) --- now MATCHING `exp3_1`'s tolerance, which the old 0.1-vs-0.2
    version of this design explicitly could not do (see Revision 2 above):
    `exp3_1` vs. `exp3_3` is now a clean, matched-tolerance ablation of
    max_mmd (cherry-picked shape difference) vs. uniform (no cherry-pick)
    B-selection.
  - **Scale: reps reduced from 50 (or the originally-configured 100) to
    20**, as a first pass under the new config, before committing to more.
    N=8/K=2/`arms_per_prototype=4` unchanged from Revision 3.
  - **Parallelization mechanism changed for an unrelated, purely
    infrastructural reason:** this repo's dev sandbox blocks a syscall
    (`os.sysconf`) that joblib's default `loky` backend needs at worker-pool
    startup. `scripts/run_experiment3.py` now forces
    `joblib.parallel_config(backend="multiprocessing")` and batches
    `rep_indices` into `n_jobs`-sized blocking `Parallel()` calls instead of
    the single `Parallel(..., return_as="generator")` call the original
    `exp4_N` script used (`multiprocessing` doesn't support generator
    streaming). Still real `n_jobs`-way process parallelism and still
    periodic progress printing / incremental parquet writes, just at batch
    granularity instead of single-rep-as-it-completes granularity. See
    `scripts/run_experiment3.py`'s module docstring for the full detail.
    This does not change anything about the experiment's statistical design
    (grid, seeds, algorithms) --- purely how the same computation gets
    scheduled.
- **REVISION 5 (2026-09-14): `kabc-0.9x`/`unif-kkm-0.9x` renamed to
  `kabc-0.5x`/`unif-kkm-0.5x`, multiplier changed from 0.9 to 0.5.**
  Discovered by direct per-rep inspection of Experiment 2's `kabc`/
  `kabc-0.9x` data (`scale_pair` sigma_B=2): KABC's doubling-trick schedule
  quantizes its own realized τ to a small set of round-boundary checkpoints,
  and the FINAL round alone typically costs roughly half of the total
  realized τ. A 0.9x budget cut therefore almost always lands short of that
  final round regardless of how close to 1.0 the multiplier nominally is ---
  there is no way to construct a real "kabc re-run with 10% less budget"
  against this schedule; 0.9x and, say, 0.6x test essentially the same
  thing (does the PREVIOUS round's partition already resolve correctly),
  not a fine-grained margin. The non-adaptive `unif-kkm` baselines, by
  contrast, degrade smoothly under the same cut (continuous n = budget_cap
  // N), which made `unif-kkm-0.9x` look deceptively strong relative to
  `kabc-0.9x`'s collapse in early comparisons --- not evidence of a real
  performance gap, an artifact of comparing a lumpy-schedule cut against a
  smooth one. 0.5x is applied everywhere `0.9x` previously was (Experiments
  1, 2, and this one) to make what is actually being tested explicit: a
  meaningfully smaller, still-real budget cut, not false precision the
  doubling schedule cannot honor. **UPDATE (2026-09-15): this experiment's
  results were re-run under the new multiplier and are no longer pending.**
  `results/exp3_{1,2,3}/results.parquet` (300 rows each, 50 reps/algorithm/
  version) were regenerated by `scripts/run_experiment3.py` after the
  rename (file mtimes 2026-09-15, after this commit's 2026-09-14 rename of
  `KABC_TAU_MULTIPLE_VARIANTS` to `{"kabc-0.5x": 0.5}`) --- confirmed
  directly by re-deriving each `kabc-0.5x` row's `budget_cap` from its
  matching same-(rep,version) `kabc` row's realized tau: every one of the
  150 `kabc-0.5x` rows across all three versions has
  `budget_cap / tau_kabc == 0.5` exactly (std 0.0), not `0.9`. The results
  reported in `docs/results_exp1_exp2_exp3.tex` for Experiment 3 are
  genuine `0.5x` re-runs, not relabeled `0.9x` data.

**Why N cannot realistically grow past 8-10: tas-fw's cost.** tas-fw's
Track-and-Stop design re-scans its ENTIRE hypothesis space (|C|, the number
of ways to split N arms into K blocks of size ≥ 2) AND re-solves a linear
program at EVERY single pull, not periodically like kabc's doubling trick.
At N=10/K=2 (|C|=501), the original smoke test measured tas-fw averaging
~900-1300s/rep whenever it failed to converge within `budget_cap`. The
N=8/K=2 run this design is based on (Revision 3, |C| smaller than N=10's)
confirmed the same order of magnitude in practice: mean wall-clock
789-908s/rep across all three (old) versions, two to three orders of
magnitude slower than every other algorithm (all under ~2.1s/rep) --- see
§6.14 of a since-superseded version of `docs/results_exp1_exp2_exp3.tex`
(describes the old `exp4_N` run; that file was reworked on 2026-09-14, see
git history for the old content). Since
|C| grows combinatorially in N (not polynomially), and tas-fw's use in this
experiment specifically exists to give a fair, finite-alphabet-respecting
baseline, N=8-10 is treated as close to the practical ceiling for including
tas-fw at all; a larger N would need either dropping tas-fw from this
experiment entirely or accepting per-rep costs an order of magnitude beyond
what is already a bottleneck.

**Smoke-test / prior-run findings worth carrying forward (from the
now-superseded `exp4_N` design, still valid evidence about algorithm
behavior in this real-data setting):**
- **ARI is exactly 0.0 for certain failure modes, never negative, and this
  is a real mathematical property, not noise.** Verified directly: ARI CAN
  go negative in general (a maximally scrambled alternating partition
  against the true 5-5 split gives ARI=-0.08). But kabc's/acb's graceful
  degradation fallback --- the trivial "everyone in one cluster" answer
  returned when budget runs out without enough evidence to split --- gives
  EXACTLY ARI=0.0 against any true partition, always, by construction of
  the adjusted-for-chance formula (a degenerate single-cluster partition
  makes the observed and expected agreement coincide).
- **kabc-sg's fragility reproduced here in a new form: not just failure to
  converge, but CONFIDENT convergence to a wrong answer** (part of why it
  was dropped in REVISION 4 above). On the smoke test's hardest sampled
  pairs, kabc (correctly) fell back to the graceful "everyone in one
  cluster" non-answer (`stopped=False`) --- but kabc-sg instead
  `stopped=True` on a near-degenerate 9-1 split (one arm isolated as its own
  "cluster") every single time on those pairs. This is a materially
  different, and worse, failure mode: an honest "I don't know" (kabc) vs. a
  confidently wrong answer (kabc-sg) that happens to score the same
  ARI=0.0 by coincidence of the two partitions' size profiles.

**Common setup, all three versions:**
- N = 8 arms, K = 2 clusters, `arms_per_prototype = 4`.
- Algorithms: kabc, kabc-0.5x, acb, tas-fw, unif-kkm-1x, unif-kkm-0.5x ---
  6 algorithm rows per (rep, version). See REVISION 4 above for what changed
  and why.
- `budget_cap`: fixed at **15000** (see REVISION 4 above). Not the
  `s_star_formula` type used elsewhere --- a formula-derived cap isn't
  meaningful at this scale.
- δ = 0.05, matching every other experiment.
- Calibrated `c_cal` constants are REUSED from the existing frozen
  `configs/calibration.yaml` (kabc, acb) --- no new calibration pass for
  this experiment, per the "calibrate once, reuse everywhere" protocol
  (section 0.2). Flagged here explicitly since this experiment's instance
  family (random MovieLens pairs) is distributionally different from what
  those constants were calibrated against (Experiment 1's
  `gaussian_mixture`); if results look miscalibrated, that is a finding to
  report, not a bug to silently patch by re-tuning here.
- Reported per algorithm per version: mean AND standard deviation of τ, ARI,
  wall-clock/rep, AND stopped/convergence rate. Visualized as bar plots (bar
  height = mean, error bar = std dev): one bar plot per metric (ARI, τ,
  wall-clock, convergence rate), each with 3 sets (one per version) of 6
  bars (one per algorithm) --- 18 bars per plot.
- **Architecture note:** unlike Experiments 1-2, EVERY rep here builds a
  fresh environment (a new random movie pair), not just fresh pull-noise
  against one fixed environment built once per grid point -- a structurally
  different loop shape from either of those, hence its own dedicated script
  (`scripts/run_experiment3.py` + `configs/exp3.yaml`).
- **Parallelization: across REPS, not across (version, algorithm) work
  items within a rep.** See REVISION 4 above for the current (sandbox-driven)
  mechanism; conceptually still "reps in parallel, `n_jobs` at a time, all
  versions/algorithm-rows sequential within one rep's worker" as originally
  designed. This resolves the kabc-derived-variant budget dependency below
  for free (kabc just runs first in the same worker) and load-balances
  tas-fw-heavy vs. cheap reps across workers by chance, rather than gating
  every rep's wall-clock on its slowest algorithm.
  **Writer safety:** workers never touch the filesystem --- `_process_one_rep`
  computes entirely in-memory and returns that rep's rows; only the MAIN
  process calls `to_parquet(...)`, always sequentially. Exactly one writer,
  same worker-computes/main-process-writes pattern every dedicated script
  in this repo follows --- no concurrent-write risk.
- **Progress visibility, two granularities:** (1) the MAIN process prints
  progress after every completed BATCH of `n_jobs` reps, and incrementally
  rewrites each of the 3 output parquets. (2) EACH WORKER also prints after
  every individual algorithm call it completes (`"rep {r} {version}/{algo}:
  tau=... stopped=... time=...s"`), giving live visibility into what all
  `n_jobs` workers are doing concurrently.

**KABC-0.5X and UNIF-KKM VARIANTS.** unif-kkm is non-adaptive and always
spends whatever budget it's given in one shot (`baselines.py`'s docstring:
n = budget_cap // N pulls/arm, always) --- passing it a smaller `budget_cap`
than the configured value is already exactly the built-in mechanism for
"give it fewer samples", no code change to `baselines.py` needed. Both
`kabc-0.5x` and `unif-kkm-1x`/`unif-kkm-0.5x` need that SAME (rep,
version)'s kabc row's own realized τ --- so kabc MUST run before any of
them, for a given (rep, version) --- resolved for free by running algorithms
sequentially within one rep's worker (kabc forced first, see "Common setup"
above). Edge case: if kabc doesn't converge (`stopped=False`, hit
`budget_cap` or the memory guard), τ_kabc ≈ `budget_cap` (specifically
13832 at this N/δ, per the doubling-trick arithmetic in "Why N cannot grow"
--- see also REVISION 4's `budget_cap` note above), so the derived rows get
budgets scaled off that same ceiling too --- expected, not a bug.

**A caution on interpreting unif-kkm-1x/0.5x vs. kabc, from the prior run's
smoke test:** on a hard pair, kabc's failure mode is the graceful
trivial-cluster fallback (ARI EXACTLY 0.0, per the mathematical property
above) --- while unif-kkm, never having that option, always commits to SOME
2-way split with whatever budget it has, which tends to land above 0 even
when wrong. So an unif-kkm-1x/0.5x mean ARI exceeding kabc's is not
necessarily "uniform sampling found the structure better" --- it can equally
be "kabc's honest abstention scores worse under ARI than a forced guess
does, on a hard instance." Report `stopped` alongside ARI for exactly this
reason; do not read an ARI gap alone as evidence of an adaptivity advantage
in either direction. The same caution applies, symmetrically, to reading
`kabc-0.5x` vs. the primary `kabc` row: a lower ARI at 0.5x budget is the
INTENDED signal (how much the extra half of budget bought), not a failure.

**Per-rep pair sampling (shared machinery, all three versions).** For a given
rep, the movie pair is sampled ONCE, producing ONE environment --- every one
of the 6 algorithm rows (kabc, kabc-0.5x, acb, tas-fw, unif-kkm-1x,
unif-kkm-0.5x) is then run against that SAME environment/pair for that rep,
not independently re-paired per algorithm. (Each algorithm still gets its
own pull-randomness sub-stream --- so their individual pull sequences
differ, but all of them face the identical clustering problem for a given
rep, making cross-algorithm comparison within a rep apples-to-apples.) Each
rep's seed (`seed_formula` as elsewhere: `seed = seed_multiplier * exp_id +
rep`) is split into two independent sub-streams: one drives this
once-per-rep pair sampling (movie A/B selection), the other drives the
algorithms' pull randomness (further split per-algorithm) --- kept separate
so which pair got sampled never gets confounded with the pull-noise that
follows. **Corrected description (matches `scripts/run_experiment3.py`'s actual
`_choose_pair`/`candidate_order`, checked directly -- the description
originally written here was a simplified approximation, not what the code
does):** for a given rep, the pair-sampling sub-stream produces ONE random
permutation of all 207 movie indices (`candidate_order`), shared identically
across all three versions for that rep index. Each version independently
scans `candidate_order` IN THAT SAME ORDER and picks the FIRST candidate A
with at least one eligible B under THAT VERSION's own gap rule; B is then
chosen from A's eligible set (argmax MMD² for `max_mmd`, uniform-random for
`uniform`). There is no separate "resample and retry" step distinct from
this scan --- continuing to the next candidate in the shared permutation IS
the retry mechanism, and it never runs out (every movie in the 207-pool has
at least one eligible partner under every version's rule in practice, per
`_choose_pair`'s `RuntimeError` never having fired on a real run).

**Common random numbers across versions:** because all three versions scan
the SAME shared permutation, they ordinarily land on the SAME movie A for a
given rep --- but NOT guaranteed identical: they diverge specifically when
the shared permutation's first eligible-under-some-version candidate isn't
eligible under a DIFFERENT (typically stricter) version's own gap rule, in
which case that stricter version continues scanning past it to its own
first eligible candidate, landing on a different A than the looser
version(s) did for that same rep. This is a weaker guarantee than "always
literally the same A" but still isolates most of the pairing RULE's effect
from independent sampling noise, since the shared starting point means
divergence is the exception (a stricter version skipping ineligible
candidates), not the rule (independent draws per version).

**Version-specific B-selection rules (current, REVISION 4):**
- **exp3_1 --- tight mean gap, MMD-maximizing.** Candidates = all movies with
  `|mean_A − mean_B| ≤ 0.25` (`mean_gap_tol = 0.25`). B = argmax MMD² among
  candidates (deterministic given A and the fixed bandwidth) --- a
  "polarizing vs consensus, cherry-picked for maximum shape difference at
  matched mean" construction, redrawn fresh every rep.
- **exp3_2 --- moderate mean gap, uniform.** Candidates = all movies with
  `0.25 ≤ |mean_A − mean_B| ≤ 1.0` (a RANGE, not a one-sided threshold ---
  excludes both the near-zero-gap pairs that overlap `exp3_1`'s hard region
  and trivially-easy very-large-gap pairs). B sampled UNIFORMLY at random
  among candidates (not MMD-maximizing). Tests behavior once genuine,
  moderate-to-large mean separation exists.
- **exp3_3 --- tight mean gap, uniform.** Candidates = all movies with
  `|mean_A − mean_B| ≤ 0.25` (`mean_gap_tol = 0.25` --- SAME as `exp3_1`,
  see REVISION 4 above for why this is now a clean matched-tolerance
  ablation). B sampled UNIFORMLY at random among candidates (not
  MMD-maximizing). Same-mean-ish regime as `exp3_1`, but without
  cherry-picking the shape difference --- tests whether kabc's advantage in
  `exp3_1` is really about shape-sensitivity, or partly an artifact of
  always picking the most-different pair available.

**Environment.** `kabc.environments.MovieLensPairEnv`, built fresh per rep
from the sampled `(movie_id_a, movie_id_b)`: `arms_per_prototype=4` each
(N=8), resampling with replacement from each movie's real rating pool;
standard balanced 2-cluster ground truth. Takes rating pools directly
(pre-loaded once from `data/ml-1m-207/`) rather than loading/filtering
`ratings.dat` itself, since a fresh environment is built every rep.

**Output.** THREE separate files, one per version ---
`results/exp3_1/results.parquet`, `results/exp3_2/results.parquet`,
`results/exp3_3/results.parquet` --- rather than one combined file with a
`version` column, so `notebooks/03_movielens_random_pairs.ipynb`'s plotting
code can load and process one version at a time with the same logic. Each
row also carries the per-rep sampled pair itself (`movie_id_a, movie_id_b,
title_a, title_b, mean_a, mean_b, mmd2_ab`) --- unlike Experiments 1-2, the
instance varies per rep here, so the sampled pair is essential context for
any post-hoc analysis (e.g. checking whether hard cases cluster around
specific movies).

**Figures -- 5 total: 1 headline (3 panels) + 4 grouped bar plots (18 bars
each):**
1. **Headline (added 2026-09-15, 1 figure, 3 panels in a single row, one
   per version):** the ACTUAL movie pair sampled at a single fixed rep
   (`rep=0`), plotted as real rating histograms
   (`data/ml-1m-207/ratings.parquet`, not recomputed from any algorithm
   output -- same category as Experiment~2's headline figure, which plots
   real 2D sample scatters directly from `RingBlobEnv`). At `rep=0`, all
   three versions happen to share the same shared-candidate-order first
   eligible movie A (``Rain Man'', 1988), so the figure directly shows how
   each version's own $B$-selection rule picks a different movie $B$
   against the identical $A$ -- `exp3_1`'s cherry-picked max-MMD$^2$ choice
   is visibly the most separated of the three, `exp3_3`'s uniformly-chosen
   $B$ under the same tight tolerance is visibly the most overlapping, a
   real-data illustration of why `exp3_3` is the hardest version.
2. Four bar plots (ARI, τ, wall-clock/rep, convergence/stopped rate), each
   with 3 sets (versions) of 6 bars (algorithms) --- 18 bars per plot, mean
   ± std. The convergence-rate plot (REVISION 4) is the fraction of reps
   with `stopped=True` per algorithm per version --- read alongside the
   ARI plot per the "caution on interpreting" note above, not in
   isolation.

All five figures live in `notebooks/03_movielens_random_pairs.ipynb`.

**Expected outcome (first pass at 20 reps under the new config -- treat as
directional, not confirmed, until this run actually completes):** `exp3_1`
(tight gap, MMD-maximizing) --- kabc should separate reliably where acb is
mean-blind and stuck near `budget_cap`, similar in spirit to the prior
`exp4_1` finding. `exp3_2` (moderate-to-large gap, uniform, range raised to
`[0.25, 1.0]`) --- acb is hoped to do noticeably better here than in the
prior `exp4_2` (mean ARI 0.089 there), given the wider/shifted range gives
it more real mean separation to work with; whether it does is exactly what
this version is testing. `exp3_3` (tight gap, uniform, now tolerance-matched
to `exp3_1`) --- the cleanest available test of whether `exp3_1`'s advantage
for kabc is really about shape-sensitivity or partly the MMD-maximizing
cherry-pick, now uncontaminated by a tolerance mismatch. `kabc-0.5x` vs.
primary `kabc`, and `unif-kkm-0.5x` vs. `unif-kkm-1x`, are the new
evidence this round specifically adds: how much of kabc's own budget is
"extra margin" beyond what a correct answer needed, and whether that margin
looks different across the three pairing regimes.

---

## Known limitation: unbounded Gram-matrix cost at hard grid points (open problem)

`kabc.py`'s (and hence `kabc-sg`'s) per-iteration `mmd2_plugin`/
`rkhs_variance` calls (`kernels.py`) have no sample-size safeguard -- they
always materialize a full `n x n` RBF Gram matrix. The doubling trick's own
`n_k` schedule can legitimately grow large on hard instances (small `s*`).
This was concretely observed at Exp 2's OLD `moment_matched_mixture` grid
point `a=0.5` (where the true MMD gap approaches 0, `s_star_sq` measured
~0.0024): `n_k` reached ~135k by iteration 13 (one Gram matrix ~146 GB) and
~1.1M by iteration 16 (~9.8 TB) before that grid point's real `budget_cap`
(~21M, correctly computed from the formula in section 0.4 -- not a bug)
would stop it. `scripts/common.py`'s `_resolve_budget_cap` caps `budget_cap`
at a small ceiling (`SMOKE_TEST_MAX_BUDGET_CAP`) specifically to avoid this
during SMOKE_TEST runs; the real (non-smoke) `budget_cap` formula is
untouched.

**Status for the CURRENT Experiment 2 grid (REDESIGN + later GRID TRIM, see
that section's notes): no longer triggered.** `a=0.5` (and the rest of the
old 0.25x-only range, `a in {0.5,0.6,0.7}`) was dropped in the REDESIGN;
`a=0.8` (the next-closest point to the cliff) was later dropped too in the
GRID TRIM -- the current grid's `moment_matched_mixture` setting only goes
down to `a=0.85`, comfortably away from this cliff. This is a side effect
of those changes, not a fix to the
underlying issue: `kabc.py`'s Gram-matrix computations still have no
sample-size safeguard at all, so **this remains a real, open problem** for
any future grid point (in this experiment or a new one) that pushes `s*`
close enough to 0 -- capping the budget there for real, or silently
excluding such a point, would be scientifically wrong for the same reason
it would have been for the old `a=0.5` point (section 0.4's budget formula
is exactly the quantity these experiments are designed to measure). Fixing
this needs a real design decision -- whether some subsampling approximation
inside the Gram-matrix computations is acceptable given KABC's theoretical
guarantee, or whether the algorithm needs actual algorithmic changes (e.g.
incremental Gram-matrix updates across doubling-trick iterations instead of
full recomputation each time) -- not attempted here.

---

## Repository layout

Workflow: `make data` → `make movielens-207` → `make calibrate` → `make smoke`
(experiments 1-2, SMOKE_TEST mode, < 10 min laptop) →
`make exp1 exp2 exp3` (every experiment via its own dedicated script --
there is no shared grid engine anymore, see `scripts/common.py`'s module
docstring -- see `make smoke-exp3` for a slower, separate exp3 sanity check
that includes tas-fw) → notebooks for figures.
​```
kernel-active-bandit/
├── configs/                  # yaml per experiment + calibration.yaml
├── data/                     # gitignored; download script + build_movielens_207.py fill it
├── docs/
│   └── experiments.md        # this file
├── notebooks/
│   ├── 01_gaussian_mixtures.ipynb
│   ├── 02_same_mean_shapes.ipynb
│   └── 03_movielens_random_pairs.ipynb
├── results/                  # gitignored parquet outputs
├── scripts/
│   ├── download_movielens.py
│   ├── build_movielens_207.py    # builds data/ml-1m-207/ for Experiment 3
│   ├── calibrate_constants.py
│   ├── common.py             # shared calibrated-fit wrappers, grid/budget_cap helpers -- see its module docstring
│   ├── run_experiment1.py    # CLI: python -m scripts.run_experiment1 configs/exp1.yaml
│   ├── run_experiment2.py    # CLI: python -m scripts.run_experiment2 configs/exp2.yaml
│   └── run_experiment3.py    # CLI: python -m scripts.run_experiment3 configs/exp3.yaml
├── src/kabc/
│   ├── algorithms/           # kabc.py, acb.py, tas_fw.py, baselines.py
│   ├── environments.py       # GMM, same-mean families, MovieLens pair env
│   ├── kernels.py            # RBF, median heuristic, MMD estimators (incl. variance est.)
│   ├── thresholds.py         # variance-aware + sub-Gaussian, calibrated constants
│   ├── metrics.py            # ARI, error, s_star computation
│   └── plotting.py           # shared figure style (matplotlib, paper-ready)
├── tests/                    # pytest: MMD estimator correctness, threshold coverage,
│                             # env sanity (means/variances), interface contracts
├── Makefile                  # data, movielens-207, calibrate, exp1/2/3, smoke, smoke-exp3, test
└── pyproject.toml            # numpy, scipy, scikit-learn, joblib, pyarrow, matplotlib
​```