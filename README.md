# Fed-CORE

Experiment code for **Federated Certified Open-Set Recognition (Fed-CORE)**.
Fed-CORE certifies the accepted selective risk of a federated open-set
classifier under client heterogeneity and uncertain deployment mixtures.

The primary reported quantity is `CertifiedCoverage@alpha`. Accuracy and AUROC
are diagnostic metrics, not certification outcomes.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
pip install -e .
```

GPU experiments are Docker-first. The launch scripts use a pinned PyTorch CUDA
image and install the CPU dependencies inside the container.

## CPU validation

These commands do not require PyTorch:

```bash
python -m fedcore.experiments.exp_lemma_L
python -m fedcore.experiments.exp_pooling_fail
python -m fedcore.experiments.run_smoke
```

The smoke experiment validates the proposal, certification, and test pipeline
with synthetic logits. It is a wiring check, not scientific evidence.

## CIFAR experiments

Run one configuration through Docker:

```bash
bash scripts/docker_cifar.sh
```

Or run directly in a CUDA-enabled environment:

```bash
python -m fedcore.experiments.run_cifar \
  --dataset cifar10 \
  --n_known 6 \
  --n_clients 5 \
  --dirichlet_alpha 0.1 \
  --rounds 50 \
  --local_epochs 2 \
  --alpha 0.10 \
  --delta 0.10
```

The primary corruption conditions are clean training, symmetric 35%
client-side label corruption, and asymmetric 20% client-side label corruption.
Evaluate seeds 0, 1, and 2 with `dirichlet_alpha` in `{0.1, 0.5, 5}`.

Additional launchers under `scripts/` reproduce FedPD, FedOSS, FOOGD, and
self-training experiment variants. Importable entry points live under
`fedcore/experiments/`; aggregation and plotting code live under
`fedcore/aggregate/` and `fedcore/plotting/`.

## Experimental protocol

- Proposal, certification, and test folds are disjoint.
- Selector choice uses only the proposal fold.
- Certification uses labeled known and unknown examples.
- Proposal selection enforces `prop_risk <= gamma * alpha`.
- The stratified conditional certificate is the primary result. Pooled
  certification is only valid for matched-mixture i.i.d. calibration.
- Zero accepted coverage is non-deployable, and a vanishing denominator bound is
  infeasible.

The canonical output fields are:

```text
certified, cert_risk_ucb, cert_coverage_lcb, cert_n, cert_k,
prop_coverage, prop_risk, test_coverage, test_risk, score_name,
gamma, alpha, delta, Lambda, dirichlet_alpha, n_clients
```

Generated datasets, logs, frozen logits, result tables, and figures are not
versioned. Experiment outputs are written below `runs/`.
