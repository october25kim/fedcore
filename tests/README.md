# Golden regression suite (structure-only refactor gate)

Pins deterministic outputs (abs diff <= 1e-9). The suite distinguishes an
artifact-free partial check from the strict frozen-artifact gate; it never labels
an absent-artifact run as a full pass.

## Strict check (host or `bash scripts/docker_test.sh`)
`python tests/golden_check.py` re-runs `golden_capture.py` into a temp dir and diffs against
`tests/golden/`:
- `certificate_math.json` — CP UCB/LCB, current simplex, conservative-endpoint normalized-box,
  matched-i.i.d. pooled, and historical floor calculations (fixed inputs)
- `scores_selector.json` — 4 scores + selector threshold/mask on a fixed logit fixture
- `split_determinism.json` — calibration fold sizes/index sums; prop/cert/test disjoint
- `certify_frozen.json` — full canonical schema on frozen cifar10_d5_resnet18_seed0 + cifar100_d5 npz

The strict command fails when either frozen NPZ is absent. For source-only CI,
run `make test-core`; it prints `PARTIAL PASS` and tests only the shipped inputs.

## CPU sanity (`bash scripts/docker_smoke.sh`)
`exp_pooling_fail.py` and `run_smoke.py`. The former `exp_lemma_L.py` claim is
withdrawn; its compatibility shim exits non-zero by design.

`test_conservative_solver_contract.py` is the strict-mixture numerical gate. It
checks one-sided risk/coverage validity against exact small-dimensional vertex
enumeration, includes fixed floating-point boundary-flip and subnormal-underflow
regressions, and requires every nonconvergence/infeasibility/residual failure to
fail closed.

## Aggregator references (for the 4-into-1 consolidation)
Static golden CSVs (diffed when the aggregators are consolidated into one `aggregate.py`):
- `selftrain_pkg_agg.golden.csv`, `selftrain_lowlabel_agg.golden.csv` (re-runnable via
  `aggregate_selftrain.py --src ...`; fast)
- `agg_covtype.golden.csv` (`aggregate_covtype.py`; reads covtype npz; seconds)
- `T8_fedosr_bases_agg.golden.csv` (`aggregate_T8.py`; reads foogd/fedpd npz; ~30s)
- `agg_main.golden.csv` (`aggregate.py`; reads ALL runs/*_logits.npz; minutes — HEAVY, verify
  manually after the consolidation, not in the fast gate)

After consolidating, re-run each aggregator and `diff` its output against the corresponding
`*.golden.csv` (must be identical). Re-snapshot ONLY if a numeric change is intended (it is not here).

## Note on agg_main.golden.csv
Re-snapshotted to the CURRENT full `runs/*_logits.npz` set (the original commit-2 copy was
from a smaller npz set, predating the resnet18gn-noise exports). The aggregate.py atomic-io
refactor was verified behaviour-preserving by an OLD-vs-NEW byte diff on the SAME npz set
(identical) — the row-count change is npz-set growth, NOT a refactor regression.
