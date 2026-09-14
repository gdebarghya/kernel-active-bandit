# Session notes — 2026-09-14 — Experiment 4 build-out, results, and open decisions

Purpose of this file: a self-contained recap of this session's work and
findings, written specifically so a **future session can replan the
configuration of Experiments 1-4** without having to re-derive the
reasoning below from scratch. It is not part of the polished report
(`docs/results_exp1_exp3.tex`) or the canonical design doc
(`docs/experiments.md`) — both of those were also updated this session with
the parts of this that belong there permanently; this file is the
discussion/rationale layer that sits behind those edits, plus everything
that is explicitly **not yet decided**.

## 1. What got built this session

Experiment 4 (MovieLens random pairs) was built end-to-end: dataset builder
(`scripts/build_movielens_207.py`, producing `data/ml-1m-207/`), environment
class (`MovieLensPairEnv` in `src/kabc/environments.py`), config
(`configs/exp4.yaml`), a dedicated run script (`scripts/run_experiment4.py`,
separate from `run_experiment.py` because every rep needs a fresh
environment, not just fresh pull-noise against one grid point), a notebook
(`notebooks/04_movielens_random_pairs.ipynb`), and plotting support. Full
design rationale, including the two prior config revisions (dropping
`unif-kkm-full`, redefining `exp4_2`, dropping-then-re-adding `exp4_3`), is
in `docs/experiments.md`'s "Experiment 4" section — not repeated here.

**Final configuration actually run:** N=8 (`arms_per_prototype=4`), K=2, 6
algorithm rows (kabc, kabc-sg, acb, tas-fw, unif-kkm-1x, unif-kkm-2x), 3
pairing versions (exp4_1 tight+MMD-max, exp4_2 moderate+uniform, exp4_3
tight-ish+uniform), `budget_cap=20000`, δ=0.05, 50 reps configured, **40
completed** (0-37, 39-49 — rep 38 permanently skipped, see §2). This is
`configs/exp4.yaml`'s current on-disk state — **not yet updated** to the
`budget_cap=30000` recommendation from §3 below.

## 2. Real crash found and fixed: degenerate pilot bandwidth

Mid-run (N=8, 50-rep attempt), `median_heuristic_bandwidth` raised
`ValueError` on rep 38/exp4_1: the pair was Schindler's List (mean 4.51)
vs. Rear Window (mean 4.48), both extremely skewed toward high ratings
(Schindler's: only 0.8% of ratings are "1"). A 5-pulls-per-arm pilot draw
(40 samples total at N=8) happened to sample only {3,4,5}, with >50% of
pairwise distances exactly tied — median distance exactly zero. Because
`Parallel(..., return_as="generator")` is fail-fast, this killed the entire
in-flight batch, not just rep 38 (already-completed reps were safe, since
the incremental writer persists after each rep).

**Decision path (both explicit user instructions, in order):**
1. First: "just resume from rep 39, skip investigating kabc.py further" —
   implemented (`rep_start` config field + resume-from-existing-parquet
   logic in `run_experiment4.py`), then reversed before the resumed run
   finished.
2. Then, reversed: **"Look at the bandwidth selection of kabc and, if it
   fails, then resort to a backup."** Implemented properly:
   - `src/kabc/kernels.py`: `median_heuristic_bandwidth(X, on_degenerate=
     "raise"|"mean", warn_on_fallback=False)`. `"mean"` falls back to the
     mean pairwise distance (informative unless literally every pair is
     tied), only raising if the mean is *also* ~0, with an optional
     `UserWarning` on trigger.
   - `src/kabc/algorithms/kabc.py`'s `_pilot_or_graceful_stop` (shared by
     `fit`/`fit_sg`/`fit_oracle`) now calls this with `on_degenerate="mean"`,
     and if even that fails, returns the existing graceful-stop pattern with
     a new `STOP_REASON_DEGENERATE_PILOT` marker, instead of crashing.
   - 2 new tests added (`tests/test_kabc.py`), 149/149 passing.
   - Verified directly against the exact crash case: kabc now converges
     correctly (τ=6624, `stopped=True`) using the recovered mean-based
     bandwidth.
   - **Confirmed working in production, not just in the unit test**: the
     re-run (with the fix in place) hit the mean-fallback path for real,
     twice, on different pairs, with the `UserWarning` firing each time —
     and the run completed cleanly both times instead of crashing.

**This fix is in shared code** (`kernels.py`/`kabc.py`), so it protects
every experiment (1-4), not just Experiment 4, from this failure mode going
forward.

**Loose end:** rep 38 is permanently absent from the final 40-rep dataset —
it was skipped under decision (1), before the fix existed under decision
(2). Whether to backfill it now (the fix would very likely let it converge)
is an **open decision for the next session**, not done here.

**A related idea that was investigated and rejected:** the user proposed
filtering out movies unless all 5 rating values are present ("avoid the
concentrated ones"). Checked directly against `data/ml-1m-207/`: **this
filter would not have caught the actual crash** — both Schindler's List and
Rear Window already have nonzero counts for all 5 rating values; the real
risk factor is a rating value being *rare* (e.g. <1% share), not *absent*.
A principled version would threshold on minimum rating-share per movie
(computed thresholds: ≥0.01 → 157/207 movies survive, ≥0.02 → 102, ≥0.03 →
57). This is now moot for crash-prevention specifically (the bandwidth fix
makes the crash a non-issue regardless of which movies are included), but
may still be relevant if a future session wants "cleaner"/more-separated
instances for other reasons (e.g. making ACB's regime easier, see §5).
**Not implemented — deferred, per the user's instruction to hold off on
Experiment 4 config changes this session.**

## 3. Budget-cap / doubling-trick interaction (relevant to replanning ANY experiment's cap)

Every `kabc`/`kabc-sg` repetition that failed to converge under
`budget_cap=20000` (at N=8, δ=0.05) stopped at the **exact same** τ value
per version (e.g. 13832 for exp4_1) — not a coincidence. This is how the
doubling-trick loop's budget check works (`kabc.py`, `_fit`): at each round
k, `cost_k = N * n_k` where `n_k = ceil(2^k * log(8(N²-N)/δ_k))`,
`δ_k = δ/(4k²)`; if `tau + cost_k > budget_cap`, the loop stops
**immediately, without charging that round's cost** — so the reported τ is
whatever was accumulated through the *last successful* round, which can be
well below the nominal cap.

Concretely, for N=8, δ=0.05: pilot cost 40, then round costs 168, 384, 816,
1704, 3512, 7208 (cumulative 13832 after 6 rounds) — round 7 alone would
cost **14728 more** (cumulative 28560), which is why `budget_cap=20000`
cuts every affected repetition off one full round short, at 13832, never
28560. Note this is **not** exactly double (13832×2=27664 ≠ 28560) because
`n_k`'s ratio between consecutive rounds is slightly more than 2× (the
`k²` term inside `δ_k`'s denominator grows `log_term` a little each round).

**Decision reached: for the next Experiment 4 configuration, set
`budget_cap=30000`** (covers 28560 with headroom; a clean round number was
preferred over the exact 28560/27664 figures). **Not yet applied to
`configs/exp4.yaml`** — this was explicitly deferred to a future
config-planning session, per the user's "let's not plan the new
configuration here" instruction.

**Reusable recipe for any future N/δ choice** (any experiment, not just
Exp 4): iterate `delta_k = delta/(4*k**2)`, `n_k = ceil(2**k *
log(8*(N**2-N)/delta_k))`, `cost_k = N*n_k`, accumulate `tau += cost_k`
starting from `tau = 5*N` (pilot), until the round where `tau + cost_k`
would exceed the candidate cap — that round's `tau + cost_k` (not simply
2×the previous tau) is the number to design the cap around, if the goal is
"every doubling algorithm gets to attempt at least one more full round
before hitting the cap."

## 4. Exp1/Exp2's `budget_multiplier` vs. Exp4's `unif-kkm-1x`/`-2x`: NOT the same mechanism

Asked and answered directly from `scripts/run_experiment.py`
(`_resolve_budget_cap`, lines ~243-285): Exp1/Exp2's **0.25× multiplier
scales the resolved, per-grid-point `budget_cap`** — either a literal
`type: fixed` config value, or a `type: s_star_formula`-derived value
(`constant * N / s_star² * log(N/delta)`, using the instance's *true* `s*`)
— applied uniformly to **every algorithm** racing on that grid point, all
under the *same* shared 0.25×-reduced cap. `s_star` itself is **not**
rescaled — it's a property of the instance, not of how much budget this
pass allocates.

This is architecturally different from Experiment 4's `unif-kkm-1x`/`-2x`,
which scale relative to **kabc's own realized per-rep τ** (`min(τ_kabc,
budget_cap)` / `min(2·τ_kabc, budget_cap)`) — a per-repetition, per-algorithm
relative comparison, not a shared instance-level theoretical cap.
**Do not conflate the two "multiplier" concepts when replanning** — a
future "restricted budget" figure for Experiment 4 (if wanted) would need
its own, third mechanism, distinct from both of the above.

## 5. Results interpretation from the completed N=8, 40-rep run

Full numeric tables are in `docs/results_exp1_exp3.tex` (§"Results ---
Experiment 4", plus new conclusions §6.12-6.14 and a new caveat bullet in
§6.11). Headlines relevant to replanning:

- **ACB is essentially blind on the tight-mean-gap versions** (mean ARI
  0.000 on exp4_1 and exp4_3, 0% ever stopping) **and still weak on the
  moderate-gap version** (0.089 on exp4_2, 17.5% stopped) — this reproduces
  Experiment 3's single-instance finding, now averaged over a random sample
  of real movie pairs. This is the concrete evidence behind the user's "ACB
  doesn't show anything in any version" observation — confirmed, not
  anecdotal. **The user has flagged ACB as a separate topic they will give
  explicit instructions on in a future session — nothing about ACB's
  algorithm/config should be changed without those instructions.**
- **KABC-SG remains fragile everywhere** (mean ARI ≤0.464 on every version),
  consistent with the same single-non-adaptive-threshold mechanism
  documented for Experiments 1-3. One new nuance worth carrying forward:
  on exp4_3, kabc-sg's *stopped rate* (87.5%) is actually *higher* than
  kabc's (52.5%), while its ARI (0.160) is far *lower* than kabc's (0.488)
  — i.e. kabc-sg converges more often here but to a worse answer, the
  "confident wrong answer" failure mode already documented in Exp4's
  smoke-test findings, now visible in the full-scale numbers too.
- **exp4_3 shows kabc itself is not immune to hard pairs**: dropping the
  MMD-maximizing cherry-pick (same tight-ish mean gap as exp4_1, but
  uniform B-selection) drops kabc's mean ARI from 0.942 (exp4_1) to 0.488,
  and its stopped rate from 97.5% to 52.5%. This means exp4_1's strong
  result is not purely a "tight mean gap" story — the cherry-pick is doing
  real work too. Worth keeping in mind if a future config wants to isolate
  one factor at a time.
- **TaS-FW is accurate but the wall-clock cost is stark**: mean ARI ≥0.987
  on every version (best or tied-best of all six algorithms), at 789-908s
  mean wall-clock/rep — two to three orders of magnitude slower than every
  other algorithm (all <2.1s/rep). Interestingly it rarely reports
  `stopped=True` (2.5-35% across versions) yet its partition estimate is
  already essentially exact under the shared budget — accuracy isn't in
  question, only scalability.
- unif-kkm-1x/2x track kabc's ARI closely wherever kabc converges often,
  per the already-documented "ARI-floor" caution (kabc's graceful abstention
  scores ARI=0.0 exactly, which can make a forced-guess baseline look
  artificially competitive) — read `stopped` alongside ARI, not ARI alone.

## 6. Open decisions for the next (replanning) session

None of the following were acted on this session — listed here specifically
so the next session can pick them up without rediscovery:

1. **Experiment 4 `budget_cap` → 30000** (derivation in §3) — agreed in
   discussion, not yet written into `configs/exp4.yaml`.
2. **ACB-improvement approach** — explicitly deferred; user will give
   instructions.
3. **Movie concentration/rating-share filter** — investigated, current
   literal "all-5-ratings-present" proposal shown insufficient, a
   min-rating-share alternative sketched (§2) but not implemented; only
   relevant if pursued for reasons other than crash-prevention (already
   fixed at the source).
4. **Rep 38 backfill** — currently permanently skipped in the 40-rep
   dataset; the bandwidth fix would very likely let it converge now, but
   whether to re-run it (and whether that changes any reported numbers
   meaningfully at n=41 vs n=40) has not been decided.
5. **Whether Experiments 1-3's configs need any revisiting** in light of
   what was learned here (e.g. the doubling-trick/budget-cap interaction in
   §3 is a general mechanism, not exp4-specific — worth checking whether
   any exp1/exp2 grid points' `budget_cap`/`s_star_formula` values similarly
   truncate kabc/kabc-sg one round short of a natural stopping point). Not
   investigated this session.

## 7. Housekeeping notes

- `.gitignore` currently contains corrupted content (appears to have had a
  shell heredoc block pasted into it verbatim in some earlier session,
  rather than executed) — flagged here but **not fixed**, since it's outside
  this session's explicit scope (per `CLAUDE.md`'s file-editing convention:
  flag and stop for confirmation rather than fold in silently).
- Scratch/temporary config files from this session's pilots
  (`configs/_exp4_N8_50reps.yaml`, `configs/_exp4_N8_50reps_resume.yaml`,
  `configs/_exp4_pilot10.yaml`) are **not** part of the canonical config set
  and were not committed to git; `configs/exp4.yaml` is the canonical file.
- `results/exp4_backup_N10_10reps/` holds a backup of the earlier N=10,
  10-rep pilot data and its plots, kept for reference, separate from the
  final N=8 dataset used in the report.
