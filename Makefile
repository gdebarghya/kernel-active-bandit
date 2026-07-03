.PHONY: data calibrate smoke exp1 exp2 exp3 test clean

data:
	python scripts/download_movielens.py

calibrate:
	python scripts/calibrate_constants.py

smoke:
	SMOKE_TEST=1 python -m scripts.run_experiment configs/exp1.yaml
	SMOKE_TEST=1 python -m scripts.run_experiment configs/exp2.yaml
	SMOKE_TEST=1 python -m scripts.run_experiment configs/exp3.yaml

exp1:
	python -m scripts.run_experiment configs/exp1.yaml

exp2:
	python -m scripts.run_experiment configs/exp2.yaml

exp3:
	python -m scripts.run_experiment configs/exp3.yaml

test:
	pytest tests/ -v

clean:
	rm -rf results/*.parquet
	rm -rf .pytest_cache __pycache__ src/kabc/__pycache__