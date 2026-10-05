.PHONY: data movielens-207 calibrate smoke smoke-exp3 smoke-exp4 exp1 exp2 exp3 exp4 exp2-acb exp4-acb test clean

data:
	python scripts/download_movielens.py

movielens-207:
	python -m scripts.build_movielens_207

calibrate:
	python scripts/calibrate_constants.py

# exp3 (MovieLens random pairs) is deliberately NOT part of `smoke`: even
# its SMOKE_TEST=1 mode (5 reps, N/K not shrunk) includes tas-fw, which
# alone can take minutes per rep -- inconsistent with smoke's "quick sanity
# check" goal. Use `make smoke-exp3` separately if you want that check.
# exp1/exp2/exp3 are each their own dedicated script now (see each
# scripts/run_experimentN.py's module docstring for why) -- there is no
# shared grid engine anymore.
smoke:
	SMOKE_TEST=1 python -m scripts.run_experiment1 configs/exp1.yaml
	SMOKE_TEST=1 python -m scripts.run_experiment2 configs/exp2.yaml

smoke-exp3:
	SMOKE_TEST=1 python -m scripts.run_experiment3 configs/exp3.yaml

# exp4 (MovieLens prototypes: kabc / unif-kkm / acb under imbalance) is also
# excluded from `smoke`, same reasoning as exp3: even SMOKE_TEST=1 (5 reps)
# runs kabc's real doubling trick on the K=7 and K=12 settings, which alone
# can take many minutes.
smoke-exp4:
	SMOKE_TEST=1 python -m scripts.run_experiment4 configs/exp4.yaml

exp1:
	python -m scripts.run_experiment1 configs/exp1.yaml

exp2:
	python -m scripts.run_experiment2 configs/exp2.yaml

exp3:
	python -m scripts.run_experiment3 configs/exp3.yaml

exp4:
	python -m scripts.run_experiment4 configs/exp4.yaml

# ACB-only passes on the instances/seeds of an existing exp2/exp4 run (the main
# run must exist first; the results file is backed up before being merged).
exp2-acb:
	python -m scripts.run_experiment2_acb configs/exp2.yaml

exp4-acb:
	python -m scripts.run_experiment4_acb configs/exp4.yaml

test:
	pytest tests/ -v

clean:
	rm -rf .pytest_cache
	find . -name __pycache__ -not -path "./.git/*" -prune -exec rm -rf {} +