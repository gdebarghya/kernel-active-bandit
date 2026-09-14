"""Build a smaller, derived MovieLens dataset: only the 207 movies with
>=1000 ratings in the full MovieLens-1M ratings.dat (the same filter
Experiment 3's `MovieLensEnv` applies via `min_ratings=1000`), plus their
titles/rating counts/means. Written once to data/ml-1m-207/ so downstream
work (e.g. Experiment 4's per-rep random movie-pair sampling) doesn't have
to re-load and re-filter the full ~1M-row ratings.dat every time.

Requires data/ml-1m/ (see `make data` / scripts/download_movielens.py).
Run directly: `python -m scripts.build_movielens_207`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kabc.environments import _load_movie_titles, _load_ratings, _movie_rating_distributions

SOURCE_DIR = Path("data/ml-1m")
OUT_DIR = Path("data/ml-1m-207")
MIN_RATINGS = 1000


def main() -> None:
    ratings = _load_ratings(SOURCE_DIR)
    titles = _load_movie_titles(SOURCE_DIR)
    probs, means = _movie_rating_distributions(ratings, min_ratings=MIN_RATINGS)

    retained_ids = sorted(means.keys())
    small_ratings = ratings[ratings["movie_id"].isin(retained_ids)].reset_index(drop=True)
    counts = small_ratings.groupby("movie_id").size()

    movies_table = pd.DataFrame(
        {
            "movie_id": retained_ids,
            "title": [titles[mid] for mid in retained_ids],
            "n_ratings": [int(counts[mid]) for mid in retained_ids],
            "mean_rating": [float(means[mid]) for mid in retained_ids],
        }
    ).sort_values("mean_rating").reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    small_ratings.to_parquet(OUT_DIR / "ratings.parquet", index=False)
    movies_table.to_parquet(OUT_DIR / "movies.parquet", index=False)

    print(f"movies retained (>= {MIN_RATINGS} ratings): {len(movies_table)}")
    print(f"ratings rows: {len(small_ratings)} (full dataset had {len(ratings)})")
    print(f"wrote {OUT_DIR / 'ratings.parquet'}")
    print(f"wrote {OUT_DIR / 'movies.parquet'}")


if __name__ == "__main__":
    main()
