"""Experiment 4 -- ACB-only pass, merged into the existing kabc/unif-kkm results.

Runs `acb` (same budget_cap as kabc, best-effort partition at the cap) for reps
[0, reps) in every exp4 setting, on the SAME instances the existing rows used
(prototype selection depends only on the rep seed), with seed index 2
(scripts.run_experiment4.ALGO_SEED_IDX). Existing kabc/unif-kkm rows are never
re-run or modified.

Safety: the existing results.parquet of each setting is backed up once to
results_pre_acb_backup.parquet (never overwritten if present), refuses to run if
acb rows already exist, and after every batch the merged file is rewritten
atomically as (original rows + acb rows so far), ordered by rep then algorithm
(kabc, unif-kkm, acb) so acb rows sit next to their rep's other rows.

Usage: python -m scripts.run_experiment4_acb [configs/exp4.yaml]
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd
import yaml
from joblib import Parallel, delayed

from scripts.run_experiment4 import ALGO_SEED_IDX, _load_movie_pool, _process_one_rep_one_setting

ALGO_ORDER = ["kabc", "unif-kkm", "acb"]


def _acb_rows_for_rep(rep, exp_id, seed_multiplier, conditions, movie_ids, means, titles, pools, delta, memory_guard_kwargs):
    out = {}
    for setting_idx, (name, cfg) in enumerate(conditions.items()):
        out[name] = _process_one_rep_one_setting(
            rep, exp_id, seed_multiplier, name, cfg, movie_ids, means, titles, pools, delta, ["acb"], setting_idx,
            memory_guard_kwargs,
        )
    return out


def _merge(base_df: pd.DataFrame, acb_rows: list[dict]) -> pd.DataFrame:
    merged = pd.concat([base_df, pd.DataFrame(acb_rows)], ignore_index=True)
    merged["_o"] = merged["algorithm"].map({a: i for i, a in enumerate(ALGO_ORDER)})
    merged = merged.sort_values(["rep", "_o"], kind="stable").drop(columns="_o").reset_index(drop=True)
    return merged[list(base_df.columns)]


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/exp4.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    exp_id, delta = config["exp_id"], config["delta"]
    seed_multiplier = config["seed_multiplier"]
    n_jobs = config.get("n_jobs", 5)
    conditions = config["conditions"]
    memory_guard_kwargs = {
        "memory_guard_n_workers": config["memory_guard"]["n_workers"],
        "memory_guard_safety_factor": config["memory_guard"]["safety_factor"],
    }

    base, paths = {}, {}
    for name in conditions:
        path = Path(f"results/exp4_{name}/results.parquet")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run scripts.run_experiment4 first")
        df = pd.read_parquet(path)
        if (df["algorithm"] == "acb").any():
            raise RuntimeError(f"{path} already contains acb rows -- refusing to run (would duplicate)")
        backup = path.with_name("results_pre_acb_backup.parquet")
        if not backup.exists():
            df.to_parquet(backup, index=False)
        base[name], paths[name] = df, path
    reps_by_setting = {n: sorted(df["rep"].unique()) for n, df in base.items()}
    rep_list = sorted(set.intersection(*[set(r) for r in reps_by_setting.values()]))
    print(f"exp4-acb: reps {rep_list[0]}..{rep_list[-1]} ({len(rep_list)}), settings {list(conditions)}, n_jobs={n_jobs}", flush=True)

    movie_ids, means, titles, pools, _ = _load_movie_pool(Path(config["data_dir"]))
    acb_rows = {name: [] for name in conditions}
    t0 = time.time()
    for b in range(0, len(rep_list), n_jobs):
        batch = rep_list[b:b + n_jobs]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_acb_rows_for_rep)(r, exp_id, seed_multiplier, conditions, movie_ids, means, titles, pools, delta,
                                       memory_guard_kwargs) for r in batch
        )
        for res in results:
            for name, rows in res.items():
                acb_rows[name].extend(rows)
        for name in conditions:
            merged = _merge(base[name], acb_rows[name])
            tmp = paths[name].with_name("results.parquet.tmp")
            merged.to_parquet(tmp, index=False)
            os.replace(tmp, paths[name])
        print(f"exp4-acb: {b + len(batch)}/{len(rep_list)} reps done, elapsed={time.time() - t0:.1f}s", flush=True)

    for name in conditions:
        df = pd.read_parquet(paths[name])
        assert not df.duplicated(["setting", "algorithm", "rep"]).any()
        assert len(df) == len(base[name]) + len(rep_list), (name, len(df))
        print(f"Wrote {len(df)} rows to {paths[name]}", flush=True)
    print("EXP4 ACB EXIT CODE: 0")


if __name__ == "__main__":
    main()
