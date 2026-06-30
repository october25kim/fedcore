# REPRODUCE — Fed-CORE artifacts

Every paper artifact, which tier it is (CPU / GPU), the exact command, and its golden oracle.
The structure-only refactor guarantees the **deterministic** (CPU) paths and certification on the
**existing frozen** `runs/*_logits.npz` are bit-for-bit unchanged (`tests/golden/`, abs diff <= 1e-9).
GPU **training** bit-reproducibility is a separate, documented procedure (see the last section).

Environment: `requirements.lock`; container `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`.
Conventions: `runs/`, `data/`, `third_party/` are gitignored (results + heavy inputs live there).

## 0. Regression gate (run before every commit)
| what | command | oracle |
|---|---|---|
| golden bit-for-bit | `make test`  (= `bash scripts/docker_test.sh` -> `python tests/golden_check.py`) | `tests/golden/*.json` |
| CPU sanity | `make smoke` (= `bash scripts/docker_smoke.sh`) | `tests/golden/*.stdout.txt` |

## 1. CPU-only (no GPU; deterministic; covered by golden)
| artifact | command | golden |
|---|---|---|
| Lemma L | `python experiments/fedcore/exp_lemma_L.py` | `exp_lemma_L.stdout.txt` |
| pooling-fail ablation | `python experiments/fedcore/exp_pooling_fail.py` | `exp_pooling_fail.stdout.txt` |
| smoke (fake logits) | `python experiments/fedcore/run_smoke.py` | `run_smoke.stdout.txt` |
| Table 1 / agg_main (CertCov, all frozen npz) | `python experiments/fedcore/aggregate.py` [`--alpha 0.20`] | `agg_main.golden.csv` |
| covtype breadth | `python experiments/fedcore/aggregate_covtype.py` | `agg_covtype.golden.csv` |
| T8 real FedOSR bases | `python experiments/fedcore/aggregate_T8.py` | `T8_fedosr_bases_agg.golden.csv` |
| self-training agg (pkg/lowlabel) | `python experiments/fedcore/aggregate_selftrain.py --src runs/<csv>` | `*_agg.golden.csv` |
| figures F6 (composite) / F7 / F8 / F9 / gain / diagram | `python experiments/fedcore/make_{composites,figures,F8,corruption_curve,selftrain_gain,problem_diagram}.py` | (figs/ outputs) |

## 2. GPU (Docker-first; TRAINING — not bit-reproducible, see caveat)
| artifact | command |
|---|---|
| CIFAR FedOSR logits export | `bash scripts/docker_cifar.sh` (env-driven; BACKBONE/NORM/NOISE_*) |
| FOOGD base (repr) / full SAG | `bash scripts/run_foogd_all.sh` / `bash scripts/docker_foogd_full.sh` |
| FedPD-PROSER base | `bash scripts/run_fedpd_all.sh` |
| self-training package P1-P4 | `bash scripts/docker_selftrain_pkg.sh` ; `bash scripts/docker_selftrain_fedpd.sh` |
| low-label self-training | `LABELED_FRAC=.. AUDIT=.. bash scripts/docker_selftrain_fedpd.sh` |

After any GPU export, the CPU certify/aggregate step (section 1) certifies the FROZEN logits and
is deterministic. The reports (`REPORT_*.md`) record the exact numbers.

## 3. Run manifest
See `Makefile` — `make test|smoke|agg-main|agg-covtype|agg-t8|agg-selftrain|figs`. Each target maps a
paper Figure/Table to its exact command; CPU targets are diffed against the golden oracle.

## Caveat — GPU training reproducibility
Re-running the GPU TRAINING from scratch is NOT guaranteed bit-identical (cuDNN nondeterminism,
library/driver versions). What IS guaranteed and gated by `make test`: the certificate math,
scores, selector, split-index construction, aggregation, and certification on the EXISTING
`runs/*_logits.npz`. To approximate training determinism, set seeds (already fixed in the runners),
`torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the eval/certify
path; full training determinism across hardware is out of scope for this refactor.
