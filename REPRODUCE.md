# Fed-CORE reproduction contract

This repository separates three levels of evidence. A command is a full
reproduction only when its required inputs are present and hash-bound; a
source-only check or fake-logit smoke is not manuscript evidence.

## 1. Install the certification core

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

The historical deterministic-capture pins are in `requirements.lock`. GPU
training uses the image named in the Docker scripts and additionally requires
the dataset-specific optional dependencies.

## 2. Source-only checks (no paper artifacts)

```bash
make unit
make test-core
python -m fedcore.experiments.exp_pooling_fail
```

`make test-core` must say `PARTIAL PASS` when the frozen run artifacts below are
absent. That is the expected public-source state, not a full reproduction.
`fedcore.experiments.exp_lemma_L` is retired and exits non-zero by design.

`python -m fedcore.experiments.run_smoke` exercises the proposal/certification
wiring on generated logits. It must never be used as a paper result.

## 3. Strict frozen-logit gate

The strict gate requires these files at the exact repository-relative paths:

```text
runs/cifar10_d5_resnet18_seed0_logits.npz
runs/cifar100_d5_none0.0_seed0_logits.npz
```

After restoring them from the author artifact archive, record their SHA-256
digests and the archive identifier in a local manifest, then run:

```bash
shasum -a 256 runs/cifar10_d5_resnet18_seed0_logits.npz \
  runs/cifar100_d5_none0.0_seed0_logits.npz
make test
```

The public repository intentionally contains neither files nor invented
digests. `make test` fails when they are absent.

## 4. Paper-campaign recertification

The current full-simplex fixed-member theorem uses `delta_r/M` and `delta_c/M`
for a proposal-frozen family of size `M`, with no further division by the number
of strata. The compact API is:

```python
from fedcore.certificate import full_simplex_fixed_member_certificate

cert = full_simplex_fixed_member_certificate(
    A, K, n, delta_r=0.05, delta_c=0.05, family_size=M
)
```

The old campaign's `delta/(MJ)` coverage values are valid conservative legacy
calculations, but they are not the theorem-aligned headline procedure. The
complete candidate-level count tensor and its frozen family hashes are released
under `paper/v18/`. Run the strict no-GPU gate with:

```bash
make reproduce-v18
```

This command recomputes the H, S, and legacy B decisions from all candidate
counts. Do not re-rank candidates or rescue failed cells from a selected-member
archive that omits the frozen family.

For a strict normalized-box mixture, use `conditional_risk_certificate(...,
Lambda="box")`. The mathematical optimization is deterministic and global; the
reported numerical certificate is deliberately one-sided rather than a raw
approximate optimum:

- risk supremum: outward-rounded upper endpoint of a validated bisection bracket;
- coverage infimum: outward-rounded lower endpoint of a validated bracket;
- feasible primal objective: diagnostic witness only;
- nonconverged, unknown, infeasible, overflow/non-finite, invalid-bracket, unsafe
  underflow/subnormal-product, or residual-validation failure: fail closed
  (`risk_ucb=inf`, `coverage_lcb=0`, no deploy).

Result rows record `solver_status`, `solver_certificate_valid`, per-objective
`tolerance`, `iterations`, bracket endpoints, residual signs, and witness or
feasibility checks. A strict-mixture manuscript row is unusable if
`solver_certificate_valid` is not true. Run the adversarial numerical gate with:

```bash
python -m pytest -q tests/test_conservative_solver_contract.py tests/test_mixture.py
```

This includes fixed floating-point boundary-flip, subnormal-underflow, and
huge-box overflow regressions, random small-cell vertex enumeration,
nonconvergence, infeasible-mixture, and vanishing-denominator checks.

For pooled CP, the caller must explicitly acknowledge a matched-mixture i.i.d.
audit with `matched_mixture_iid=True`.

## 5. Training-to-logits binding checklist

The v18 package closes count-to-decision reproduction. A stronger claim that
reproduces model training and logits additionally requires all of the following:

- clean repository commit and version tag;
- environment/container digest and dependency lock;
- dataset source, version, split indices, preprocessing, and label mapping;
- fixed checkpoint and complete candidate-family hash;
- proposal/certification/test fold hashes and fixed audit sizes/stopping rule;
- all candidate-level `(A, K, n)` tensors plus uniqueness accounting;
- command, seed, exit code, output SHA-256, and table/figure source mapping.

Those raw-data and training inputs are outside the v18 public numerical release.
The repository therefore makes the narrower, testable claim documented in
`paper/v18/README.md`.
