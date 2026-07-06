# Claude Code prompt — structure-only refactor with reproducibility lock

Goal. The Fed-CORE experiment code has accumulated ad-hoc runners, several overlapping
aggregation scripts, scattered constants, and CSV-write races (the smoke-row pollution and
root-owned-file issues we already hit). Refactor it into a **clean, well-structured, reproducible
package WITHOUT changing a single number**. The hard constraint: every result already produced
(certificate metrics, the self-training CSVs/figures, the CPU experiments) must reproduce
**bit-for-bit** from the same inputs after the refactor.

**The safety mechanism is a golden / characterization test suite captured BEFORE any change.**
Snapshot the current deterministic outputs first; then every refactor step must reproduce that
snapshot exactly. If a number moves, the refactor is wrong — not the snapshot.

**Use Claude Code** (not a one-shot transform): this needs repo-wide context, Docker test loops,
and incremental verified commits. Paste the fenced block into Claude Code in the FedCORE repo.

```text
READ CLAUDE.md AND AGENTS.md FIRST. Object = certified accepted selective risk (Fed-CORE); canonical
metric schema and split hygiene are sacred. THIS IS A STRUCTURE-ONLY REFACTOR: zero change to any
numerical result, metric definition, schema name, threshold, RNG seed, or split logic. Reproducibility
of already-produced results is the pass/fail criterion. Docker-first. Do NOT do feature work,
re-train models, or rename canonical keys. Report honestly in the fixed format.

PHASE 0 — GOLDEN SNAPSHOT + PLAN (read/run only; NO source changes). STOP-AND-ASK at the end.
  Capture the current deterministic outputs as a regression oracle under tests/golden/:
    - bash scripts/docker_test.sh ; bash scripts/docker_smoke.sh  (record pass + key stdout metrics)
    - python experiments/fedcore/exp_lemma_L.py        -> snapshot its printed/CSV numbers
    - python experiments/fedcore/exp_pooling_fail.py    -> snapshot
    - certification on EXISTING frozen logits: run the certify path on a fixed set of
      runs/*_logits.npz (e.g. cifar10_d5_resnet18_seed0_logits.npz and one cifar100) and snapshot
      every canonical metric (certified, cert_risk_ucb, cert_coverage_lcb, cert_n, cert_k,
      prop_coverage, prop_risk, test_coverage, test_risk, plus Thm1/1'/Thm3 outputs).
    - aggregation on EXISTING csvs: run the current aggregate logic on runs/selftrain_pkg.csv and
      runs/selftrain_lowlabel.csv and snapshot the agg rows (means, sample-SD, seed-aware n).
    - scores.py / selector.py: snapshot outputs on a fixed small logit fixture (all four scores;
      selector threshold + accepted mask) to pin them.
  Record the environment: python/torch/numpy/scipy versions, and the exact seeds used.
  Write REFACTOR_PLAN.md: proposed target package layout, the file moves/consolidations, what each
  current script maps to, and a risk note per change. Then STOP and ask for approval before Phase 2.

INVARIANTS (must hold after EVERY commit; verified by the golden suite):
  1. Canonical metric schema names UNCHANGED: certified, cert_risk_ucb, cert_coverage_lcb, cert_n,
     cert_k, prop_coverage, prop_risk, test_coverage, test_risk, score_name, gamma, alpha, delta,
     Lambda, dirichlet_alpha, n_clients (+ labeled_frac, audit_mult, beta, seed for self-training).
  2. Certificate math identical (CP UCB, Theorem 1 simplex, Theorem 1' box linear-fractional,
     Theorem 2 floor, Prop. 3 pooled): bit-for-bit (abs diff <= 1e-9) on the golden inputs.
  3. Split hygiene preserved: proposal/certification/test disjoint; selector chosen on the proposal
     fold only; identical RNG seed -> identical fold indices (assert index equality on a fixture).
  4. Public entry points keep working: run_cifar.py / run_smoke.py CLI flags and the docker_*.sh
     scripts in CLAUDE.md run unchanged (provide thin shims if you relocate code).
  5. Docker-first: docker_test.sh and docker_smoke.sh stay green.

PHASE 2 — INCREMENTAL REFACTOR (one concern per commit; run the golden suite before each commit).
  Move code, do NOT rewrite logic. Target a clear package layout, e.g.:
    fedcore/
      config.py            single source of constants + the canonical metric schema (defined once)
      data/                fedosr_split, clients, dirichlet partition, calibration folds
      models/              models.py, fed_train.py (training code path UNCHANGED)
      scores.py selector.py
      certificate/         cp primitives, theorem1 (simplex), theorem1p (box), theorem3 (pooled),
                           feasibility (Thm2)  -- pure functions, no I/O
      aggregate.py         ONE aggregation module (see below)
      io.py                atomic CSV read/append/write
      plotting/            figure generators
      experiments/         exp_lemma_L, exp_pooling_fail, run_smoke, run_cifar, self-training runners
      cli.py               argument parsing / entry points
  Specific debt to retire (behavior-preserving):
    a. CONSOLIDATE the multiple aggregate scripts (aggregate_selftrain.py and any inline/ad-hoc
       aggregators) into ONE aggregate.py with the guards centralized and documented:
         - convergence guard (drop rows with known_acc < threshold; keep the SAME threshold currently
           used, 0.30/0.40 — read it from config, do not change the value),
         - seed-aware n_seeds = number of DISTINCT seeds (never row count),
         - sample SD (ddof=1),
         - grid-aware grouping keys including labeled_frac/audit_mult/beta.
       Every caller (pkg, fedpd, lowlabel) imports this one function. Verify agg rows are identical
       to the golden snapshot for both existing CSVs.
    b. ATOMIC CSV writes in io.py (write temp file + os.replace; lock or unique-temp per process) to
       remove the clean+launch race that produced duplicate/smoke rows; consistent output paths and
       file permissions under runs/.
    c. Centralize the metric schema + column order in config.py so every writer emits the same header.
    d. Add type hints + concise docstrings to public functions; NO logic change.
  Remove dead/duplicate code ONLY when the golden suite covers the surviving path. NEVER delete or
  overwrite result CSVs or *_logits.npz under runs/.

PHASE 3 — REPRODUCIBILITY DELIVERABLES.
  - REPRODUCE.md: how to regenerate every paper artifact (which are CPU vs GPU), with exact commands.
  - A run manifest (Makefile or scripts/repro/*.sh) mapping each paper Figure/Table -> exact command
    -> expected golden output (hash or snapshot file). e.g. make fig3, make table5, make selftrain-agg.
  - Pin the environment: a locked requirements file (and pin the Docker base/image tag); set
    deterministic flags where feasible (numpy/torch seeds; torch.use_deterministic_algorithms and
    cudnn deterministic for the certify/eval path). Document that full GPU TRAINING bit-reproducibility
    is a separate, documented procedure, not a pass/fail of this refactor.
  - tests/: the golden regression suite runnable in-container (pytest or via scripts/docker_test.sh).

STOCHASTIC-TRAINING CAVEAT (read carefully).
  Do NOT re-train any model during this refactor. Deterministic components (certificate math, scores,
  selector, split-index construction, aggregation, exp_lemma_L, exp_pooling_fail) MUST match the
  golden snapshot bit-for-bit (<=1e-9). For the torch training code, preserve the exact code path and
  verify only that (i) the smoke run still passes and (ii) certification on the EXISTING frozen
  runs/*_logits.npz is identical. Re-deriving GPU results from scratch is out of scope here.

DO NOT:
  - change any number, metric value, threshold, schema name, RNG seed, or split logic;
  - rename canonical metric/config keys; mix or leak proposal/certification/test folds;
  - delete or overwrite any result CSV or *_logits.npz under runs/ (or data/, checkpoints/);
  - re-train models or add new experiments/features; do a big-bang rewrite;
  - reintroduce SRCC / RC-OWPL / pseudo-labeling as the object, or rename Fed-CORE concepts;
  - hide a failing golden test or a failed command — report them.

GIT WORKFLOW.
  Branch refactor/structure-repro. bash scripts/git_start_day.sh at start. One concern per commit;
  run the golden suite before each commit; commit message in English. Keep runs/, data/, outputs/,
  checkpoints/, logs/, wandb/, *.pt/*.pth/*.npy/*.npz uncommitted (per .gitignore). bash
  scripts/git_end_day.sh "structure-only refactor: <summary>" at the end.

REPORT (fixed format), after Phase 0 and after each phase:
  진단 요약 / 확인한 명령 / 핵심 결과 (무엇을 이동·통합했는지, golden suite pass 여부, 발견된 diff)
  / 판정 (strong go / moderate go / warning / fail) / 다음 행동.
  After Phase 0, output REFACTOR_PLAN.md and STOP for approval before touching source.
```

---

### Notes for Sanghoon
- **안전장치의 핵심은 Phase 0의 golden snapshot**입니다. 리팩토링 전에 결정론적 출력(certificate
  수학·aggregation·split index·CPU 실험·기존 logits certification)을 스냅샷으로 박고, 모든 커밋이
  그것과 1e-9 이내 일치를 강제 → 숫자가 바뀌면 그 즉시 테스트가 깨져서 잡힙니다. "재현성을 망치지
  않는다"가 곧 "golden suite가 항상 green"으로 환원됩니다.
- **GPU training은 재현 대상에서 분리**합니다. torch 학습은 bit-재현이 원래 불가하니, 리팩토링은
  학습 코드 경로를 그대로 두고 *frozen logits 기반 certification*과 *smoke*만 검증합니다. 이미 나온
  결과(metrics·CSV·figure)는 전부 결정론적 후처리라 안전하게 고정됩니다.
- **이번에 실제로 물린 부채를 정조준**: 여러 aggregate 스크립트 → 단일 `aggregate.py`(수렴 가드 +
  seed-aware n + sample-SD 중앙화), CSV race → atomic write(temp+rename), schema 중복 → config 단일
  정의. 전부 behavior-preserving.
- 도구는 **Claude Code 권장**(repo 맥락 + Docker/test 루프 + 점진 커밋 검증). Codex는 고립된 변환엔
  쓸 수 있지만 이 작업의 핵심인 verification loop가 약합니다. 같은 프롬프트를 Codex에 줄 거면
  "Phase 0 STOP 후 승인"·"커밋마다 golden suite 실행"을 사람이 직접 챙겨야 합니다.
- **Phase 0 후 반드시 멈추고 REFACTOR_PLAN.md를 같이 검토**하세요 — 패키지 레이아웃·파일 이동 맵을
  승인한 뒤에만 Phase 2로 보내는 게 안전합니다.
