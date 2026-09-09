"""Fetch/verify MovieLens-1M into data/ml-1m/, per docs/experiments.md's
Experiment 3 "Data" section: checks for data/ml-1m/ratings.dat first and
only downloads from grouplens.org if missing. Invoked by `make data`.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = Path("data/ml-1m")
REQUIRED_FILES = ["ratings.dat", "movies.dat", "users.dat"]


def already_present(data_dir: Path) -> bool:
    return all((data_dir / name).exists() for name in REQUIRED_FILES)


def download_and_extract(data_dir: Path) -> None:
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "ml-1m.zip"
        print(f"Downloading {ML_1M_URL} ...")
        urllib.request.urlretrieve(ML_1M_URL, zip_path)

        print(f"Extracting to {data_dir.parent} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(Path(tmp) / "extracted")

        extracted_ml1m = Path(tmp) / "extracted" / "ml-1m"
        if not extracted_ml1m.exists():
            raise RuntimeError(
                f"expected an 'ml-1m/' folder inside the downloaded archive, "
                f"got: {list((Path(tmp) / 'extracted').iterdir())}"
            )
        if data_dir.exists():
            shutil.rmtree(data_dir)
        shutil.move(str(extracted_ml1m), str(data_dir))


def main() -> None:
    if already_present(DATA_DIR):
        print(f"{DATA_DIR} already has {REQUIRED_FILES} -- nothing to do.")
        return

    download_and_extract(DATA_DIR)

    missing = [name for name in REQUIRED_FILES if not (DATA_DIR / name).exists()]
    if missing:
        print(f"ERROR: download/extract completed but still missing {missing} in {DATA_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {DATA_DIR} now has {REQUIRED_FILES}.")


if __name__ == "__main__":
    main()
