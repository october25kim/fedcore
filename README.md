# Fed-CORE — Federated Certified Open-Set Recognition

Fed-CORE certifies, with a **finite-sample distribution-free upper confidence
bound**, the **accepted selective risk** of a federated-trained open-set
classifier — the probability that an *accepted* prediction is wrong — under
client heterogeneity and an unknown deployment mixture, computing the certificate
from selector-indexed stratum count triples `(A, K, n)`. Raw examples, logits,
and scores do not enter the certification decision. Cryptographic secure
aggregation and differential privacy are separate deployment mechanisms and are
not implemented by this research code.

This is **not** a federated-accuracy paper and **not** a noisy-label-robust
training method. The object is *certification of which open-set predictions can be
safely accepted*, not improving the model.

> **One-line thesis.** Heterogeneity (and client-side label corruption) deforms
> the confidence–correctness ranking; a small trusted clean calibration set held
> across clients is used to **certify** which predictions are safe to accept, not
> to repair the model.

**Pipeline.**
```
heterogeneous / corrupted FL-trained classifier
  -> score-agnostic accept/reject proposal (risk-buffered, gamma*alpha)
  -> federated independent certification under partial exchangeability
  -> certified accepted coverage from released count triples
```

Headline metric: **CertifiedCoverage@alpha**. Judge results by the `cert_*`
risk / coverage fields — **never** by accuracy or AUROC.

## Theory contract implemented by the current API

- **Controlled object:** `R_sel(lambda) = sum_j lam_j m_j / sum_j lam_j a_j`,
  where per client `j`: `a_j = P_j(accept)`, `m_j = P_j(accept & error)`,
  `r_j = m_j / a_j`. Goal: certify `R_sel(lambda) <= alpha` while maximizing
  accepted coverage.
- **Theorem 1 (full simplex, fixed member).** For a selector fixed independently
  of the certification fold, use `rbar_j = U+(K_j, A_j; delta_r)` and
  `Ubar = max_j rbar_j`. The tail does **not** divide by the number of clients:
  failure of the reported scalar maximum implies failure of the marginal bound
  for a fixed true worst client. Minimum coverage uses the analogous
  `min_j L-(A_j,n_j;delta_c)`. This is not a collection of simultaneous
  clientwise intervals.
- **Frozen selector family.** A simple simultaneous family of `M` proposal-frozen
  members uses `delta_r/M` and `delta_c/M` per member, again with no additional
  client-count division inside a full-simplex member. The public theorem-facing
  functions are `full_simplex_fixed_member_certificate` and
  `simple_simultaneous_family_certificate`. In the strict bounded-mixture family
  branch, the corresponding tails are `delta_r/(3*S*M)` for each risk-side
  endpoint and `delta_c/(S*M)` for the separate coverage endpoints.
- **Holm/IUT scope.** The Holm route is restricted to a full-simplex,
  fixed-`alpha` risk decision. It returns the familywise decision and adjusted
  p-values plus a family-simultaneous coverage LCB; it does not report a
  numerical risk UCB unless the test is explicitly inverted. The dataset-neutral
  import path is `fedcore.certificate.holm.holm_family_certificate`.
- **Theorem 2 (strict bounded mixture).** A proper mixture restriction needs
  simultaneous risk and acceptance endpoints. The normalized-box implementation
  spends `delta_r/(3J)` on each risk-side endpoint family and solves the robust
  positive-denominator linear-fractional program by deterministic global
  optimization. Numerically, the risk certificate is the validated bisection
  bracket's outward-rounded **upper endpoint**, and the coverage certificate is
  its outward-rounded **lower endpoint**. Feasible primal objectives are logged
  only as witnesses. Nonconvergence, an invalid sign bracket, a non-positive
  denominator, unsafe underflow/subnormal product arithmetic, or failed numerical
  validation returns `risk_ucb=inf` and/or `coverage_lcb=0` and cannot certify.
  The former random mixture sampling
  approximation is no longer used by the certificate API.
- **Non-reducibility.** Naive pooling of the federated accepted set into one
  binomial is invalid under heterogeneity (per-client `r_j` differ → the pooled
  accepted-error count is Poisson-binomial, not binomial). The certificate is not
  a corollary of centralized conformal prediction, nor of federated conformal
  coverage.
- **Pooled CP (subordinate and narrow).** It is certifying only for a
  matched-mixture i.i.d. audit, where the pooled accepted-error count is actually
  binomial. The API requires an explicit `matched_mixture_iid=True`
  acknowledgement. The former finite-search “Lemma L” claim for a heterogeneous
  Poisson-binomial mean is withdrawn; its old module now exits non-zero.
- **Privacy taxonomy.** The full-simplex stratified certificate requires one
  `(A_j, K_j, n_j)` triple per declared stratum. These aggregates disclose less
  than observation-level logits or labels, but they may still be sensitive.
  Grouping clients into predeclared public strata can reduce granularity. The
  repository does not claim that count release alone provides cryptographic or
  differential-privacy protection.
- **Calibration assumption (stated openly).** Certifying unknown rejection needs
  the certification fold to contain *labeled* unknown-class points.
  "Distribution-free" is w.r.t. the calibration distribution.

## Repository layout

```
fedcore/                       importable core package (pip install -e .)
  certificate/    CP primitives; full-simplex and bounded-mixture certificates; pooled diagnostic
  certify.py config.py scores.py selector.py   numpy certification core + fixed metric schema
  grouping.py atomic_io.py                      grouped-certification + atomic/locked CSV writers
  data/          FedOSR split (open-set + Dirichlet non-IID + calibration folds), clients, noise
  models/        FedAvg + logit export (torch)
  medical/       MedMNIST / medical open-set data helpers
  experiments/   runnable entry points  (python -m fedcore.experiments.<name>)
  aggregate/     seed-aware table builders for frozen logits and run CSVs
  plotting/      figure generators
  accounting/ campaign/                         provenance, budget, run-matrix helpers
tests/           golden bit-for-bit regression gate (tests/golden/) + unit tests
scripts/         Docker run wrappers (docker_*.sh) + baseline run scripts
docker/          container definitions
pyproject.toml requirements.lock Makefile
```

## Install

```bash
# Certification core.
pip install -e .

# Source-only test and aggregation dependencies.
pip install -e '.[test]'
```

The core runtime dependencies are declared in `pyproject.toml`; the historical
CPU capture pins are retained in `requirements.lock`. GPU
training runs Docker-first in `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`
(the `scripts/docker_*.sh` wrappers add `pip install -e .` automatically).

## Quickstart (CPU, no torch, no data)

```bash
python tests/test_current_certificate.py         # theorem-facing, no data
python -m fedcore.experiments.exp_pooling_fail  # invalid pooled diagnostic under shift
python -m fedcore.experiments.run_smoke         # fake-logit wiring only
```

Regression gate (deterministic outputs, bit-for-bit within 1e-9):

```bash
make unit       # current theorem-facing source-only tests
make test-core  # deterministic partial check; clearly reports missing artifacts
make test       # strict gate; fails when required frozen NPZ artifacts are absent
make smoke      # two CPU sanity scripts above
make reproduce-wr-v3 # current manuscript WR-v3 count-to-decision release gate
make reproduce-v18   # historical v0.2.0 release gate
```

`make test` verifies the certificate math, scores/selector, split determinism,
and the frozen-logit path against `tests/golden/`. Public source checkouts do not
contain the required frozen logits, so a strict run is expected to fail with the
missing paths instead of printing a false full PASS. See [REPRODUCE.md](REPRODUCE.md).

## Real experiments (GPU)

Run any experiment entry point as a module, e.g.:

```bash
python -m fedcore.experiments.run_cifar --dataset cifar10 --n_known 6 \
    --n_clients 5 --dirichlet_alpha 0.1 --rounds 50 --local_epochs 2 \
    --alpha 0.10 --delta 0.10

python -m fedcore.experiments.run_officehome --help
python -m fedcore.experiments.run_fed_isic   --help
python -m fedcore.experiments.run_oneshot_posthoc --help
```

Or use the Docker wrappers (see the scripts for the env vars they read):

```bash
bash scripts/docker_smoke.sh        # CPU sanity in the container
bash scripts/docker_test.sh         # golden gate in the container
bash scripts/docker_cifar.sh        # CIFAR training + logit export + certification
bash scripts/docker_officehome.sh
```

Training writes frozen logits to `runs/` and certificates to `results/`; both are
gitignored. Certification then runs on the frozen `runs/*_logits.npz`. The
versioned manuscript package under `paper/wr-v3/` contains the WR-v3 benchmark
count tensor and numerical source artifacts, not raw datasets, checkpoints, or
per-example logits. It supersedes `paper/v18/` for current manuscript numbers.
The v18 directory remains available as a historical release. Fake-logit smoke
output must not be cited as manuscript evidence.

## Canonical metric schema (do not rename)

Numerical-UCB and fixed-alpha Holm/IUT rows share the common identity, count,
coverage, and decision fields. `risk_output_type` determines the risk payload:

```text
common: certified, risk_output_type, risk_pass, cert_coverage_lcb, cert_n,
        cert_k, prop_coverage, prop_risk, test_coverage, test_risk, score_name,
        gamma, alpha, delta, delta_r, delta_c, Lambda, family_procedure,
        dirichlet_alpha, n_clients
numerical_ucb: cert_risk_ucb
fixed_alpha_decision: iut_raw_pvalue, holm_adjusted_pvalue, holm_rank
```

For `fixed_alpha_decision`, `cert_risk_ucb` is null because Holm/IUT tests the
predeclared risk level rather than constructing a numerical upper bound.

Split hygiene is enforced: the proposal / certification / test folds are disjoint;
the selector is chosen on the proposal fold only (never on certification labels).

## Datasets (not included)

No datasets ship with this package. `runs/`, `data/`, and `results/` are
gitignored. The data loaders expect the standard public sources:

- **CIFAR-10 / CIFAR-100** — torchvision downloads.
- **MedMNIST (PathMNIST, TissueMNIST)** — the `medmnist` package / official `.npz`.
- **Fed-ISIC2019** — via [FLamby](https://github.com/owkin/FLamby).
- **Office-Home** — the official Office-Home release.

## Baselines (external)

The federated open-set baselines (**FedPD**, **FedOSS**, **FOOGD**) are **not
vendored** here. The runners that invoke them
(`fedcore/experiments/run_fedpd_cifar.py`, `run_fedoss_cifar.py`,
`run_foogd_cifar.py`, `foogd_score.py`, and the `scripts/docker_fedpd.sh` /
`docker_foogd.sh` / `docker_fedoss.sh` wrappers) expect the corresponding upstream
repositories to be placed under a local `third_party/` directory. Fetch them from
their original sources and mind their licenses (FedPD's released code is GPLv3).
The Fed-CORE certification core and its primary experiments run **without** these
baselines.

## License

Fed-CORE source: MIT — see [LICENSE](LICENSE). Optional dependencies, datasets,
and external baselines retain their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
