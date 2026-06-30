# Fed-CORE run manifest (paper artifact -> exact command -> golden oracle).
# Deterministic CPU targets diff their output against tests/golden/ (must be identical).
# See REPRODUCE.md for the full table and the GPU training targets.
.PHONY: help install test smoke agg-main agg-covtype agg-t8 agg-selftrain figs repro-check
PY ?= python
EXP := experiments/fedcore

# Prereq: `make install` (editable install) once after checkout so `import fedcore` resolves
# from the project-root fedcore/ package. The golden gate self-bootstraps and needs no install.
help:
	@echo "targets: install test smoke agg-main agg-covtype agg-t8 agg-selftrain figs repro-check"
	@echo "  install        editable install so 'import fedcore' resolves (run once after checkout)"
	@echo "  test           golden bit-for-bit regression (the commit gate)"
	@echo "  smoke          CPU sanity (exp_lemma_L, exp_pooling_fail, run_smoke)"
	@echo "  agg-*          re-run an aggregator and diff its output vs tests/golden"
	@echo "  figs           regenerate the figure family"
	@echo "  repro-check    test + all fast agg diffs"

install:
	$(PY) -m pip install -e .

test:
	$(PY) tests/golden_check.py

smoke:
	cd $(EXP) && $(PY) exp_lemma_L.py && $(PY) exp_pooling_fail.py && $(PY) run_smoke.py

agg-covtype:
	cd $(EXP) && $(PY) aggregate_covtype.py >/dev/null
	diff runs/agg_covtype.csv tests/golden/agg_covtype.golden.csv && echo "agg_covtype OK"

agg-t8:
	cd $(EXP) && $(PY) aggregate_T8.py >/dev/null
	diff runs/T8_fedosr_bases_agg.csv tests/golden/T8_fedosr_bases_agg.golden.csv && echo "T8 agg OK"

agg-selftrain:
	cd $(EXP) && $(PY) aggregate_selftrain.py --src runs/selftrain_pkg.csv >/dev/null
	cd $(EXP) && $(PY) aggregate_selftrain.py --src runs/selftrain_lowlabel.csv >/dev/null
	cd $(EXP) && $(PY) aggregate_selftrain.py --src tests/golden/fixtures/selftrain_subguard.csv --out runs/selftrain_subguard_agg.csv >/dev/null
	diff runs/selftrain_pkg_agg.csv tests/golden/selftrain_pkg_agg.golden.csv && \
	diff runs/selftrain_lowlabel_agg.csv tests/golden/selftrain_lowlabel_agg.golden.csv && \
	diff runs/selftrain_subguard_agg.csv tests/golden/selftrain_subguard_agg.golden.csv && echo "selftrain agg OK (incl sub-guard drop)"

agg-main:   ## HEAVY (all runs/*_logits.npz; minutes)
	cd $(EXP) && $(PY) aggregate.py >/dev/null
	diff runs/agg_main.csv tests/golden/agg_main.golden.csv && echo "agg_main OK"

figs:
	cd $(EXP) && $(PY) make_composites.py && $(PY) make_F8.py && \
	  $(PY) make_corruption_curve.py && $(PY) make_selftrain_gain.py && $(PY) make_problem_diagram.py

repro-check: test agg-covtype agg-t8 agg-selftrain
	@echo "repro-check PASS (golden + fast aggregators identical)"
