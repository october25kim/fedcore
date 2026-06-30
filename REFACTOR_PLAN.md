# REFACTOR_PLAN — structure-only repro refactor

## STATUS (checkpoint after 5 commits; all golden-green, behaviour-preserving)
DONE:
- efd1760 baseline: fedcore package (was untracked) + Phase-0 golden suite (tests/golden + golden_capture/check.py)
- ed8853b test: covtype/T8/main aggregate golden references
- e6f3e5b refactor: centralize CANONICAL_SCHEMA + SELFTRAIN_MIN_ACC in config.py (debt c; values unchanged)
- 0c6cb62 refactor: atomic_io.py atomic/locked CSV writes; all 5 aggregators + 2 runners routed (debt b)
- 6f53516 docs: REPRODUCE.md + Makefile manifest + requirements.lock (Phase 3)
Verified: `make repro-check` PASS (golden + covtype + T8 + selftrain bit-identical); aggregate.py
OLD-vs-NEW byte-identical. Branch `refactor/structure-repro`; only fedcore/ paths committed (parent
repo's unrelated deletions untouched); nothing pushed.

DONE (resume) — CORE packaged into fedcore/ (golden green at every commit; explicit-re-export shims):
- c78ab63 fix: anchor .gitignore dir patterns to repo root (so the new fedcore/data sub-package is tracked)
- b4878f8 M1: certificates.py -> fedcore.certificate.{cp,theorem1(simplex+box/Thm1'),theorem3(pooled/
  stratified+true_selective_risk),feasibility(Thm2 floor)}; certificate_math.json bit-for-bit.
- 47dd3a5 M2: scores/selector + data/(fedosr_split,clients,noise) + models/(models,fed_train, byte-
  unchanged) -> fedcore; scores_selector.json + split_determinism.json match.
- 939a36c M3a: config + certify -> fedcore (certify imports rewired flat->fedcore.*).
fedcore/ now contains the importable CORE: certificate/ scores selector data/ models/ config certify.
Ten old flat paths are explicit-re-export shims. `make repro-check` PASS.

RESUMING (user, 2nd resume): continue the FULL relocation. Order (golden green before each commit):
- PRECOND: extract shared helpers (_group_map/_repartition/_views_from_parts) -> fedcore/grouping.py
  (behaviour-preserving) so plotting+aggregators import one copy, not experiment scripts. NOTE only the
  exp_feasibility_lever versions are shared via import; make_figures._repartition and aggregate_T8.
  _repartition are LOCAL variants (different formatting/signature) — left untouched.
- M3: move plotting/(make_*) -> fedcore/plotting + shims; figure OUTPUT paths stay experiments/fedcore/
  figs/*.png (verify a regenerated figure writes the same path).
- M4: consolidate aggregate*/T8/covtype/selftrain -> fedcore/aggregate.py; convergence threshold is a
  PER-CALLER PARAM (self-training 0.30; covtype/T8 keep current per-cell values); add a SUB-GUARD golden
  fixture row; keep seed-aware n_seeds, sample SD ddof=1, grid-aware keys. *_agg golden byte-identical.
- M5: move experiments/ runners (exp_*, run_*, selftrain*) -> fedcore/experiments + shims.
- GPU-PARITY GATE (after M5): per GPU entry point import/--help parity + one short real job to valid
  canonical output; bit-parity only if deterministic ref exists. STOP-AND-ASK with results.
- M6 (optional): migrate internal call sites to fedcore.*; prune shims to public-CLI/docker/CLAUDE.md.

Helper-script note: scripts/docker_test.sh + docker_smoke.sh CREATED; git_start_day.sh /
git_end_day.sh referenced by the prompt do NOT exist — git was handled manually (branch + per-commit
fedcore-only staging).

---

(original Phase-0 plan follows)

# REFACTOR_PLAN — structure-only repro refactor (Phase 0 output)

**Scope:** structure + reproducibility ONLY. Zero change to any number, metric value, schema
name, threshold, RNG seed, or split logic. The golden regression suite (Phase 0, below) is the
pass/fail criterion: deterministic outputs must match bit-for-bit (abs diff <= 1e-9) after every
commit. NO retraining, NO feature work, NO big-bang rewrite. One concern per commit.

---

## Phase 0 — golden snapshot (DONE, read/run only)

Regression oracle written under `tests/golden/` and a checker `tests/golden_check.py`
(re-runs `tests/golden_capture.py` into a temp dir and diffs, float tol 1e-9). **Current status:
`python tests/golden_check.py` -> PASS.** Snapshots captured:

| snapshot | pins |
|---|---|
| `ENV.txt` | python 3.13.12, numpy 2.4.6, scipy 1.17.1, sklearn 1.9.0; container torch 2.3.0; seeds=0 |
| `exp_lemma_L.stdout.txt` | Lemma L True/True; worst-case coverage |
| `exp_pooling_fail.stdout.txt` | conditional median U 0.3823 < mass-ratio 0.4727 |
| `run_smoke.stdout.txt` | per-score best-gamma cert_ucb/cov_lcb/test_risk; validity MC 0.000 |
| `certificate_math.json` | cp_upper/lower, Thm1 simplex (U=0.2714), Thm1' box (U=0.2171), stratified, Thm3 pooled (0.0953), Thm2 floor |
| `scores_selector.json` | all 4 scores + selector threshold/mask on a fixed logit fixture |
| `split_determinism.json` | calibration fold sizes + index sums; prop/cert/test disjoint=True |
| `certify_frozen.json` | full canonical schema on cifar10_d5_resnet18_seed0 + cifar100_d5 (for_score simplex/box, best_gamma, best_gamma_grouped G2) |
| `*_agg.golden.csv` + `aggregate_*.stdout.txt` | aggregate rows for selftrain_pkg + selftrain_lowlabel (means, sample-SD, seed-aware n, dropped-row log) |

---

## Environment complications found in Phase 0 (need a decision — see STOP-AND-ASK)

1. **Git root is the PARENT** `/home/sanghoon/projects`, not `fedcore/`. `git status` shows MANY
   unrelated pending deletions (UPLIFT-v1, RC-OWPL, selective_risk_cifar, .DS_Store churn). A
   branch+commit here would entangle fedcore changes with that unrelated churn.
2. **Helper scripts referenced by the prompt do NOT exist:** `scripts/docker_test.sh`,
   `scripts/docker_smoke.sh`, `scripts/git_start_day.sh`, `scripts/git_end_day.sh`. Phase 0 ran the
   underlying python directly instead. These must be created (Phase 3) or the workflow adapted.

---

## Current layout (flat: experiments/fedcore/, ~50 modules, 7759 LOC)

Imports are FLAT (`from certificates import ...`, `from certify import ...`, `from scores import
...`). This is the single biggest refactor risk: any physical move breaks these imports unless
backward-compat shims are left in place.

## Proposed target layout (a real importable `fedcore/` package)

```
fedcore/
  __init__.py
  config.py                 # single source: constants + CANONICAL METRIC SCHEMA + guard thresholds
  io.py                     # NEW: atomic CSV read/append/write (temp + os.replace, unique temp/proc)
  data/                     # fedosr_split, clients, noise (dirichlet partition, calibration folds)
  models/                   # models.py, fed_train.py            (training code path UNCHANGED)
  scores.py  selector.py
  certificate/              # pure functions, no I/O:
    cp.py                   #   cp_upper, cp_lower
    theorem1.py             #   conditional_risk_certificate (simplex)  + Thm1' box (same fn, Lambda arg)
    theorem3.py             #   pooled_cp / stratified_certificate
    feasibility.py          #   Theorem-2 floor helpers
  certify.py                # proposal->cert->test glue (imports certificate/*)
  aggregate.py              # ONE aggregation module (consolidates the 4 below)
  plotting/                 # make_figures, make_composites, make_F8, make_selftrain_gain, make_problem_diagram, make_corruption_curve
  experiments/              # exp_lemma_L, exp_pooling_fail, run_smoke, run_cifar, run_tabular,
                            #   run_foogd*, run_fedpd*, run_selftrain*, selftrain*, self_training, exp_*
  cli.py                    # arg parsing / entry points (thin)
experiments/fedcore/<old>.py  # BACKWARD-COMPAT SHIMS: `from fedcore.<new path> import *`
```

## File move map (current -> target) — representative; full map applied incrementally

| current | target | note |
|---|---|---|
| certificates.py | fedcore/certificate/{cp,theorem1,theorem3,feasibility}.py | split by theorem; `certificates.py` shim re-exports all |
| certify.py | fedcore/certify.py | imports certificate/*; shim left |
| scores.py, selector.py | fedcore/scores.py, fedcore/selector.py | shims |
| config.py | fedcore/config.py | + CANONICAL_SCHEMA, MIN_ACC guard moved here (values UNCHANGED) |
| fedosr_split.py, clients.py, noise.py | fedcore/data/ | shims |
| models.py, fed_train.py | fedcore/models/ | training path byte-unchanged |
| aggregate.py, aggregate_covtype.py, aggregate_selftrain.py, aggregate_T8.py | fedcore/aggregate.py | **CONSOLIDATE** (see below); old names become shims calling the one impl |
| make_*.py | fedcore/plotting/ | shims |
| run_*.py, exp_*.py, selftrain*.py, self_training.py, foogd_score.py | fedcore/experiments/ | public CLIs keep working via shims |

## Debt to retire (behavior-preserving)

a. **Consolidate the 4 aggregators -> `fedcore/aggregate.py`.** Centralize + document the guards
   already in `aggregate_selftrain.py`: convergence guard (drop known_acc < MIN_ACC; keep the
   CURRENT value — read from `config.py`; note the codebase uses 0.30 for self-training and the
   per-cell guards in aggregate.py/T8 unchanged), seed-aware `n_seeds` (distinct seeds, never row
   count), sample SD (ddof=1), grid-aware keys incl labeled_frac/audit_mult/beta. Every caller
   (pkg/fedpd/lowlabel/covtype/T8) imports the one function. **Verify agg rows == golden** for both
   self-training CSVs (and re-snapshot covtype/T8 aggregates first — see RISK).
b. **Atomic CSV writes in `io.py`** (temp + `os.replace`, unique-temp per process) to remove the
   clean+launch race that produced the duplicate smoke rows; consistent `runs/` paths + permissions.
c. **Centralize metric schema + column order in `config.py`** so every writer emits the same header.
d. **Type hints + concise docstrings** on public functions. NO logic change.

## Risk notes (per change)

| change | risk | mitigation |
|---|---|---|
| package move (flat -> fedcore/) | HIGH — breaks `from X import` everywhere | leave backward-compat SHIM modules at old paths re-exporting; golden_check + run_smoke before each commit |
| split certificates.py by theorem | MED — re-export ordering / circular imports | one commit; `certificates.py` shim re-exports identical names; golden certificate_math.json must match |
| consolidate 4 aggregators | MED — covtype/T8 aggregators NOT yet golden-snapshotted | FIRST snapshot covtype + T8 aggregate outputs into tests/golden, THEN consolidate, THEN diff |
| atomic io.py | LOW — write path change | byte-compare a written CSV vs current writer on a fixture |
| schema in config.py | LOW | header string compared to golden |
| docstrings/type hints | LOW | no runtime effect; golden unaffected |
| training code (models/, fed_train.py) | OUT OF SCOPE to verify bit-for-bit | move only; verify smoke passes + certify on FROZEN npz identical; NO retrain |

## Invariants enforced before every commit (by `tests/golden_check.py`)
1. canonical schema names unchanged; 2. certificate math bit-for-bit (<=1e-9); 3. split-index
determinism + disjointness; 4. public CLIs + docker_*.sh run unchanged (shims); 5. CPU sanity green.

## Proposed commit sequence (one concern each; golden green before each)
1. add `tests/` golden suite + (NEW) `scripts/docker_test.sh` wrapping golden_check (no source move).
2. snapshot covtype + T8 aggregates into golden (close the coverage gap before touching aggregators).
3. `config.py`: centralize CANONICAL_SCHEMA + guard constants (values unchanged) + shim.
4. introduce `fedcore/` package skeleton + `__init__`; move certificate/* with shims.
5. move scores/selector/data/models with shims.
6. `io.py` atomic writes; route aggregate writers through it.
7. consolidate aggregators -> `fedcore/aggregate.py`; old names become shims; diff golden.
8. move plotting/ and experiments/ with shims; move certify.py.
9. Phase 3: REPRODUCE.md + run manifest (Makefile/scripts/repro) + pinned requirements + docker_test/smoke + cli.py.

## Out of scope / DO NOT
re-train; change any number/threshold/seed/schema/split; delete or overwrite any `runs/*.csv` or
`*_logits.npz`; rename Fed-CORE concepts; reintroduce SRCC/RC-OWPL/pseudo-labeling as the object.
