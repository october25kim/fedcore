# Fed-CORE — Federated Certified Open-Set Recognition

Fed-CORE certifies, with a **finite-sample distribution-free upper confidence
bound**, the **accepted selective risk** of a federated-trained open-set
classifier — the probability that an *accepted* prediction is wrong — under
client heterogeneity and an unknown deployment mixture, computing the certificate
from **secure-aggregatable counts only**.

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
  -> certified accepted coverage (secure-aggregation-only leakage)
```

Headline metric: **CertifiedCoverage@alpha**. Judge results by the `cert_*`
risk / coverage fields — **never** by accuracy or AUROC.

## Theory (what the code certifies)

- **Controlled object:** `R_sel(lambda) = sum_j lam_j m_j / sum_j lam_j a_j`,
  where per client `j`: `a_j = P_j(accept)`, `m_j = P_j(accept & error)`,
  `r_j = m_j / a_j`. Goal: certify `R_sel(lambda) <= alpha` while maximizing
  accepted coverage.
- **Theorem 1 / 1′ (main, conditional selective-risk certificate).** Use the
  conditional law `K_j | A_j ~ Bin(A_j, r_j)` to bound `r_j` directly via a
  Clopper–Pearson upper bound `rbar_j = U+(K_j, A_j; delta/J)`. Full simplex:
  `Ubar = max_j rbar_j` (deploy iff `<= alpha`). Bounded `Lambda` (Thm 1′): also
  bound `a_j in [alow_j, ahigh_j]` and solve the robust linear-fractional program
  `sup (sum lam a rbar) / (sum lam a)`.
- **Non-reducibility.** Naive pooling of the federated accepted set into one
  binomial is invalid under heterogeneity (per-client `r_j` differ → the pooled
  accepted-error count is Poisson-binomial, not binomial). The certificate is not
  a corollary of centralized conformal prediction, nor of federated conformal
  coverage.
- **Theorem 2 (feasibility).** Per-client observed accepted count
  `A_j >= ln(J/delta) / (-ln(1-alpha))`.
- **Proposition 3 (pooled, subordinate).** A tighter pooled bound holds only under
  matched-mixture i.i.d. calibration; kept below Thm 1/1′.
- **Privacy taxonomy.** Only the pooled certificate is sum-only secure-
  aggregatable. The stratified certificate needs per-client `(A_j, K_j)`; a
  grouped-stratified variant (public strata, ≥k clients each) secure-aggregates
  within groups as a tunable compromise.
- **Calibration assumption (stated openly).** Certifying unknown rejection needs
  the certification fold to contain *labeled* unknown-class points.
  "Distribution-free" is w.r.t. the calibration distribution.

## Repository layout

```
fedcore/                       importable core package (pip install -e .)
  certificate/    CP primitives; conditional cert (Thm 1/1'); pooled (Prop 3); feasibility (Thm 2)
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
# Editable install so `import fedcore` resolves with no sys.path hacks.
pip install -e .
```

Pinned CPU deps are in `requirements.lock` (numpy / scipy / scikit-learn). GPU
training runs Docker-first in `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`
(the `scripts/docker_*.sh` wrappers add `pip install -e .` automatically).

## Quickstart (CPU, no torch, no data)

```bash
python -m fedcore.experiments.exp_lemma_L       # Lemma-L numerical support
python -m fedcore.experiments.exp_pooling_fail  # pooled-binomial invalidity under heterogeneity
python -m fedcore.experiments.run_smoke         # end-to-end certification wiring on fake logits
```

Regression gate (deterministic outputs, bit-for-bit within 1e-9):

```bash
make test        # python tests/golden_check.py   (the commit gate)
make smoke       # the three CPU sanity scripts above
```

`make test` verifies the certificate math, scores/selector, and split
determinism against `tests/golden/`. It self-bootstraps and needs no install.

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
gitignored. Certification then runs on the frozen `runs/*_logits.npz`.

## Canonical metric schema (do not rename)

```
certified, cert_risk_ucb, cert_coverage_lcb, cert_n, cert_k, prop_coverage,
prop_risk, test_coverage, test_risk, score_name, gamma, alpha, delta, Lambda,
dirichlet_alpha, n_clients
```

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

MIT — see [LICENSE](LICENSE).
