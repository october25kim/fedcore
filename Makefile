# Fed-CORE run manifest (paper artifact -> exact command -> golden oracle).
# Deterministic CPU targets diff their output against tests/golden/ (must be identical).
# See REPRODUCE.md for the full table and the GPU training targets.
.PHONY: help install unit test test-core smoke agg-main agg-covtype agg-t8 agg-selftrain figs repro-check reproduce-wr-v3 reproduce-v18
PY ?= python
# All targets run from the project root and invoke the installed package with `python -m`
# (no path shims). Modules use a CWD-relative base so runs/ and experiments/fedcore/figs/
# resolve to the same locations as before.

# Prereq: `make install` (editable install) once after checkout so `import fedcore` resolves
# from the project-root fedcore/ package. The golden gate self-bootstraps and needs no install.
help:
	@echo "targets: install unit test-core test smoke agg-main agg-covtype agg-t8 agg-selftrain figs repro-check reproduce-wr-v3 reproduce-v18"
	@echo "  install        editable install so 'import fedcore' resolves (run once after checkout)"
	@echo "  unit           artifact-free current-theorem unit tests"
	@echo "  test-core      deterministic golden check; allows absent frozen run artifacts"
	@echo "  test           strict gate; fails if required frozen run artifacts are absent"
	@echo "  smoke          CPU sanity (pooling counterexample + fake-logit wiring)"
	@echo "  agg-*          re-run an aggregator and diff its output vs tests/golden"
	@echo "  figs           regenerate the figure family"
	@echo "  repro-check    test + all fast agg diffs"
	@echo "  reproduce-wr-v3 verify the current WR-v3 manuscript count-to-decision release"
	@echo "  reproduce-v18  verify the historical v0.2.0 count-to-decision release"

install:
	$(PY) -m pip install -e .

test: unit
	$(PY) tests/golden_check.py

unit:
	$(PY) -m pytest -q tests/test_current_certificate.py tests/test_family_contract.py \
	  tests/test_recertify_holm_output.py tests/test_conservative_solver_contract.py \
	  tests/test_brief_s0.py tests/test_officehome_selector_rescue.py

test-core: unit
	FEDCORE_ALLOW_MISSING_ARTIFACTS=1 $(PY) tests/golden_check.py

smoke:
	$(PY) -m fedcore.experiments.exp_pooling_fail && $(PY) -m fedcore.experiments.run_smoke

agg-covtype:
	@if [ -f runs/covtype_seed0_logits.npz ]; then \
	  $(PY) -m fedcore.aggregate.covtype >/dev/null; \
	  diff runs/agg_covtype.csv tests/golden/agg_covtype.golden.csv && echo "agg_covtype OK"; \
	else \
	  echo "agg_covtype ERROR: runs/covtype_seed*_logits.npz absent"; exit 2; \
	fi

agg-t8:
	@if [ -f runs/T8_fedosr_bases.csv ]; then \
	  $(PY) -m fedcore.aggregate.t8 >/dev/null; \
	  diff runs/T8_fedosr_bases_agg.csv tests/golden/T8_fedosr_bases_agg.golden.csv && echo "T8 agg OK"; \
	else \
	  echo "T8 agg ERROR: runs/T8_fedosr_bases.csv absent"; exit 2; \
	fi

agg-selftrain:
	@if [ -f runs/selftrain_pkg.csv ] && [ -f runs/selftrain_lowlabel.csv ]; then \
	  $(PY) -m fedcore.aggregate.selftrain --src runs/selftrain_pkg.csv >/dev/null; \
	  $(PY) -m fedcore.aggregate.selftrain --src runs/selftrain_lowlabel.csv >/dev/null; \
	  $(PY) -m fedcore.aggregate.selftrain --src tests/golden/fixtures/selftrain_subguard.csv --out runs/selftrain_subguard_agg.csv >/dev/null; \
	  diff runs/selftrain_pkg_agg.csv tests/golden/selftrain_pkg_agg.golden.csv && \
	  diff runs/selftrain_lowlabel_agg.csv tests/golden/selftrain_lowlabel_agg.golden.csv && \
	  diff runs/selftrain_subguard_agg.csv tests/golden/selftrain_subguard_agg.golden.csv && echo "selftrain agg OK (incl sub-guard drop)"; \
	else \
	  echo "selftrain agg ERROR: runs/selftrain_{pkg,lowlabel}.csv absent"; exit 2; \
	fi

agg-main:   ## HEAVY (all runs/*_logits.npz; minutes)
	$(PY) -m fedcore.aggregate.main >/dev/null
	diff runs/agg_main.csv tests/golden/agg_main.golden.csv && echo "agg_main OK"

figs:
	$(PY) -m fedcore.plotting.make_composites && $(PY) -m fedcore.plotting.make_F8 && \
	  $(PY) -m fedcore.plotting.make_corruption_curve && $(PY) -m fedcore.plotting.make_selftrain_gain && $(PY) -m fedcore.plotting.make_problem_diagram

repro-check: test agg-covtype agg-t8 agg-selftrain
	@echo "repro-check PASS: strict frozen-artifact and aggregator checks completed"

reproduce-v18:
	$(PY) paper/v18/scripts/verify_code_contract.py
	$(PY) paper/v18/scripts/verify_release.py

reproduce-wr-v3:
	$(PY) paper/wr-v3/scripts/verify_release.py
