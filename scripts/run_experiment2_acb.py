"""Experiment 2 -- ACB-only rerun, replacing the existing acb rows in place.

ACB was changed to return a best-effort partition at the budget cap
(src/kabc/algorithms/acb.py). Every exp2 row is independent of the others, so
only acb is re-run: same grid points, same dedicated-seed budget caps (asserted
equal to the stored ones), same per-run seed (rep_seed*100 + acb's roster index).
results/exp2/results.parquet is backed up once to results_pre_acb_rerun_backup.parquet;
non-acb rows are kept untouched and the acb rows are replaced at their original
row positions. Usage: python -m scripts.run_experiment2_acb [configs/exp2.yaml]
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import threadpoolctl
import yaml
from joblib import Parallel, delayed

from kabc import metrics
from scripts.common import CAP_SEED_OFFSET, _acb_fit, _expand_grid, _resolve_budget_cap
from scripts.run_experiment2 import ALL_INSTANCE_PARAMS, ENVIRONMENTS


def _acb_one_rep(rep, exp_id, seed_multiplier, acb_idx, env, budget_cap, delta, ctx):
    algo_seed = (seed_multiplier * exp_id + rep) * 100 + acb_idx
    rng = np.random.default_rng(algo_seed)
    t0 = time.time()
    with threadpoolctl.threadpool_limits(limits=1):
        result = _acb_fit(env, delta, budget_cap, rng)
    wall = time.time() - t0
    print(f"    rep {rep} {ctx['label']}/acb: tau={result.tau} stopped={result.stopped} time={wall:.1f}s", flush=True)
    row = {
        "exp_id": exp_id, "family": ctx["family"], "swept_param": ctx["swept_param"],
        "swept_value": ctx["swept_value"], "algorithm": "acb", "rep": rep, "seed": algo_seed,
        "budget_cap": budget_cap, "s_star": ctx["s_star"], "tau": result.tau, "stopped": result.stopped,
        "wall_clock_seconds": wall, "partition": result.partition.tolist(),
        "error": metrics.error(result.partition, env.true_partition),
        "ari": metrics.ari(result.partition, env.true_partition), "n_iterations": len(result.history),
    }
    for p in ALL_INSTANCE_PARAMS:
        row[p] = ctx["resolved_kwargs"].get(p)
    return row


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/exp2.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    exp_id, delta, reps = config["exp_id"], config["delta"], config["reps"]
    seed_multiplier, n_jobs = config["seed_multiplier"], config.get("n_jobs", 10)

    path = Path("results/exp2/results.parquet")
    base = pd.read_parquet(path)
    backup = path.with_name("results_pre_acb_rerun_backup.parquet")
    if not backup.exists():
        base.to_parquet(backup, index=False)
    old_acb = base[base["algorithm"] == "acb"]
    assert len(old_acb) > 0

    new_rows = []
    gp_idx = 0
    t0 = time.time()
    for family_cfg in config["instances"]:
        env_cls = ENVIRONMENTS[family_cfg["family"]]
        acb_idx = family_cfg["algorithms"].index("acb")
        for swept_param, swept_value, kwargs in _expand_grid(family_cfg):
            env = env_cls(**kwargs)
            cap_rng = np.random.default_rng(CAP_SEED_OFFSET + exp_id * 100_000 + gp_idx)
            gp_idx += 1
            budget_cap, s_star = _resolve_budget_cap(family_cfg["budget_cap"], env.n_arms, delta, env, cap_rng, budget_multiplier=1.0)
            sel = old_acb[(old_acb["family"] == family_cfg["family"]) & (old_acb["swept_value"] == swept_value)]
            assert len(sel) == reps and (sel["budget_cap"] == budget_cap).all() and np.allclose(sel["s_star"], s_star), \
                f"stored cap/s_star mismatch at {family_cfg['family']}/{swept_value}"
            ctx = dict(family=family_cfg["family"], swept_param=swept_param, swept_value=swept_value,
                       resolved_kwargs=kwargs, s_star=s_star, label=f"{family_cfg['family']}/{swept_param}={swept_value}")
            for b in range(0, reps, n_jobs):
                new_rows.extend(Parallel(n_jobs=n_jobs)(
                    delayed(_acb_one_rep)(r, exp_id, seed_multiplier, acb_idx, env, budget_cap, delta, ctx)
                    for r in range(b, min(b + n_jobs, reps))))
            print(f"exp2-acb: {ctx['label']} done, {len(new_rows)} rows, elapsed={time.time() - t0:.1f}s", flush=True)

    new = pd.DataFrame(new_rows)
    out = base.copy()
    key = ["family", "swept_value", "rep"]
    new_idx = new.set_index(key)
    for i in old_acb.index:
        r = base.loc[i]
        n = new_idx.loc[(r["family"], r["swept_value"], r["rep"])]
        assert n["seed"] == r["seed"], "seed mismatch"
        for c in new.columns:
            if c not in key:
                out.at[i, c] = n[c]
    out = out[list(base.columns)]
    assert len(out) == len(base) and not out.duplicated(["family", "swept_value", "algorithm", "rep"]).any()
    assert out[out["algorithm"] != "acb"].reset_index(drop=True).drop(columns="partition").equals(
        base[base["algorithm"] != "acb"].reset_index(drop=True).drop(columns="partition"))
    tmp = path.with_name("results.parquet.tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    print(f"Wrote {len(out)} rows to {path}")
    print("EXP2 ACB EXIT CODE: 0")


if __name__ == "__main__":
    main()
