# HANDOFF.md - Fed-CORE runbook for Claude Code (updated 2026-07-05)

Read `CLAUDE.md` and `AGENTS.md` first. `CLAUDE.md` wins if instructions disagree. Discussion can be Korean; code, configs, metric keys, and command names stay English.

## Current state

The codebase is now organized as a normal importable Python package under `fedcore/`. The previous flat `experiments/fedcore/*.py` code layout is gone; `experiments/fedcore/` is artifact space for figures only. Run all code through module entry points such as `python -m fedcore.experiments.run_smoke`.

This refactor is intended to be structure-only and behavior-preserving. Do not change theorem logic, metric schema, split logic, thresholds, or RNG behavior while doing cleanup. The golden regression suite is the guardrail.

## Package map

| path | purpose |
|---|---|
| `fedcore/certificate/` | CP bounds, Theorem 1/1 conditional certificate, App-C stratified baseline, Proposition 3 pooled certificate, Thm-2 feasibility |
| `fedcore/{scores,selector,certify,config}.py` | score functions, risk-buffered selector, proposal/cert/test glue, canonical schema/config |
| `fedcore/data/` | synthetic clients, FedOSR split, Dirichlet partition, train-label noise |
| `fedcore/models/` | torch models, FedAvg training, logit export |
| `fedcore/experiments/` | runnable experiment modules and GPU runners |
| `fedcore/aggregate/` | seed-aware aggregators for frozen logits/run CSVs |
| `fedcore/plotting/` | figure builders |
| `scripts/` | Docker-first wrappers and batch runners |
| `tests/golden*` | deterministic regression oracle for the structure refactor |

## What changed in the structure pass

- Documentation now points to the real project-root `CLAUDE.md` / `AGENTS.md`, not stale parent-relative paths.
- README module map now matches the actual `fedcore/` package layout.
- `fedcore.__init__` and `pyproject.toml` comments now state that old flat path shims are removed.
- Shared grouped-certification helpers are centralized in `fedcore.grouping` with public names: `make_group_map`, `repartition_trusted_pool`, and `views_from_parts`. The old underscore names remain as compatibility aliases only.
- CSV write helpers now live in `fedcore.io_utils`; `fedcore.atomic_io` is a backward-compatible shim for older commands/imports.
- Aggregators, plotting, and handoff scripts now import the public helper names where possible instead of experiment-private underscore helpers.
- This handoff replaces the stale 2026-06-26 GPU-run handoff with current Claude Code guidance.

## Invariants to preserve

- Main object remains accepted selective risk: `R_sel(lambda) = sum_j lam_j m_j / sum_j lam_j a_j`.
- Theorem 1/1 conditional certificate is primary; pooled certificate is subordinate and matched-mixture only.
- Proposal, certification, and test folds must remain disjoint. The selector is chosen on proposal only.
- Canonical metric keys must not be renamed: `certified`, `cert_risk_ucb`, `cert_coverage_lcb`, `cert_n`, `cert_k`, `prop_coverage`, `prop_risk`, `test_coverage`, `test_risk`, `score_name`, `gamma`, `alpha`, `delta`, `Lambda`, `dirichlet_alpha`, `n_clients`.
- Judge experiments by `cert_*` risk/coverage, not accuracy or AUROC.
- Keep changes surgical; do not reintroduce SRCC / RC-OWPL / pseudo-labeling as the main project.

## Validation commands

```bash
# one-time install
pip install -e .

# deterministic regression
python tests/golden_check.py

# CPU smoke
python -m fedcore.experiments.exp_lemma_L
python -m fedcore.experiments.exp_pooling_fail
python -m fedcore.experiments.run_smoke

# Makefile aliases
make test
make smoke
make repro-check
```

Docker wrappers are available as `bash scripts/docker_test.sh` and `bash scripts/docker_smoke.sh`. Use them when Docker is available; otherwise the direct Python commands above are the canonical local checks.

## Next useful work

1. Run `python tests/golden_check.py` after any structure change.
2. Run the three CPU smoke commands before committing.
3. For GPU work, start with `bash scripts/docker_cifar.sh`, then aggregate frozen logits with the `fedcore.aggregate.*` modules.
4. If a command fails, report the exact failed command and fix before moving on.
