# Claude Code prompt — confirm the standalone repo october25kim/fedcore (read-only)

Final confirmation that the extracted standalone repo is clean, self-contained, and reproducible.
READ-ONLY: makes no changes; if something is broken it STOPS and reports (does not fix silently).
Paste the fenced block into Claude Code.

```text
READ CLAUDE.md AND AGENTS.md FIRST (they now live at the ROOT of the standalone repo). This is a
READ-ONLY confirmation of october25kim/fedcore — verify it is clean, self-contained, and reproducible.
Make NO file changes, NO commits, NO pushes. Do NOT touch the source monorepo. If anything is broken
or unexpected, STOP-AND-ASK with the diagnosis — do NOT fix silently. Report in the fixed format
(진단 요약 / 확인한 명령 / 핵심 결과 / 판정 / 다음 행동).

STEP 1 — fresh clone + editable install.
  rm -rf /tmp/fedcore-confirm
  git clone git@github.com:october25kim/fedcore.git /tmp/fedcore-confirm
  cd /tmp/fedcore-confirm
  pip install -e .          # must resolve `import fedcore` with NO sys.path hack
  python -c "import fedcore, fedcore.certificate, fedcore.aggregate, fedcore.grouping, fedcore.io_utils; print(fedcore.__file__)"

STEP 2 — structure is clean, self-contained, and correctly documented.
  - repo root has: fedcore/ (package), experiments/, tests/, scripts/, pyproject.toml, Makefile, README.md.
  - NO other monorepo projects present: UPLIFT-v1, RC-OWPL, selective_risk_cifar, fedtold — none should exist.
  - NO leftover flat shims: `ls experiments/fedcore/*.py` -> expect NONE (only figs/ + README.md remain).
  - third_party.zip is NOT tracked (`git ls-files | grep -i third_party` -> empty; it must not have leaked).
  - CLAUDE.md and AGENTS.md at root describe **Fed-CORE** (certified accepted selective risk,
    proposal/certification split, Clopper–Pearson UCB, cert_risk_ucb/cert_coverage_lcb) — NOT a stale
    SRCC / RC-OWPL-as-main-object instruction. Flag if they are stale.

STEP 3 — reproducibility gate (distinguish RAN vs SKIPPED).
  make repro-check    # must PASS
  Report exactly which checks ran FULLY vs were skipped:
   - CPU-only goldens that need NO runs/ artifacts (exp_lemma_L, exp_pooling_fail, run_smoke,
     certificate_math, scores/selector, split-determinism) MUST run and be bit-identical (<=1e-9).
   - checks needing runs/*.npz or *_agg CSV (certify_frozen, aggregate goldens) will SKIP in a bare
     clone (artifacts are gitignored).
  CONFIRM the skip is EXPLICIT AND LOUD (prints e.g. "SKIP: no runs/ artifacts") — a skip must NEVER be
  silently counted as a pass. If a skip is silent, flag it (this was a review note).

STEP 4 — import / CLI parity (catches the class of bug Task A found; no artifacts needed).
  # every entry point + plotting module must IMPORT with no ImportError / missing-module:
  python -c "import fedcore.plotting.make_figures, fedcore.plotting.make_selftrain_gain, fedcore.plotting.make_F8"
  python -m fedcore.experiments.run_cifar --help
  python -m fedcore.experiments.run_foogd_cifar --help
  python -m fedcore.experiments.run_selftrain_pkg --help
  python -m fedcore.experiments.exp_lemma_L
  python -m fedcore.experiments.exp_pooling_fail
  python -m fedcore.experiments.run_smoke
  NOTE: full `make figs` regeneration needs runs/ artifacts, so it is NOT expected to fully run in a
  bare clone — import-checking the plotting modules is the right bare-clone test (the full pixel-identity
  regen was already verified on the source repo in Task A). If `make figs` is attempted and fails only
  on missing runs/ data, that is expected, not a code defect — say so.

STEP 5 (optional, only if GPU available) — short end-to-end in the standalone repo.
  ROUNDS=1 LOCAL_EPOCHS=1 OUT=_tmp/confirm.csv bash scripts/docker_cifar.sh
  Criterion (NOT bit-parity): runs to completion via `python -m fedcore.experiments.*` and exports
  canonical schema (score_name,gamma,alpha,delta,Lambda,dirichlet_alpha,n_clients,certified,
  cert_risk_ucb,cert_coverage_lcb,cert_n,cert_k,prop_coverage,prop_risk,test_coverage,test_risk).
  Then delete _tmp/. Leave everything else untouched.

DO NOT change, commit, or push anything. STOP-AND-ASK on any breakage or unexpected finding.

REPORT (fixed format): 진단 요약 / 확인한 명령 / 핵심 결과 (clone root listing; other-project absence;
shim absence; third_party not tracked; CLAUDE/AGENTS = Fed-CORE?; which goldens ran vs skipped + skip
loudness; import/CLI results; optional GPU schema) / 판정 (strong/warning/fail) / 다음 행동.
```

---

### Notes for Sanghoon
- **read-only 확인 프롬프트**입니다 — 변경/커밋/push 없음, 깨진 게 있으면 고치지 말고 STOP-AND-ASK.
- 핵심 확인: (1) editable install로 `import fedcore` 해결, (2) 루트가 self-contained(다른 프로젝트·shim·third_party.zip 없음, CLAUDE/AGENTS가 Fed-CORE), (3) golden에서 **CPU 결정론 체크는 full bit-identical**·artifact 필요분만 **loud skip**, (4) 모든 entry point/plotting 모듈 **import·--help OK**(Task A가 잡은 버그 부류 방지).
- **bare clone엔 `runs/` artifact가 없어** certify_frozen·aggregate golden은 skip이 정상 — skip이 "명시적(loud)"인지가 이번 확인의 포인트(제가 남긴 리뷰 노트). full 그림 regen(pixel-identity)은 이미 source repo Task A에서 검증됨.
- GPU 있으면 STEP 5(짧은 docker_cifar)로 standalone repo의 GPU 경로까지 확정. 없으면 STEP 1–4로 충분.
- 결과 붙여주시면 제가 최종 판정(standalone repo 완결 여부)을 같이 확인하겠습니다.
