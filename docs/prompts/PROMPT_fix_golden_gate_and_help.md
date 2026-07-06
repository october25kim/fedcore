# Claude Code prompt — close the golden stdout-gate gap (Issue 2) + torch --help verify/record (Issue 1)

Two items from the standalone-repo confirmation. Work in the standalone repo october25kim/fedcore.
TEST-HARNESS + DOCS only — no science change. Paste the fenced block into Claude Code.

```text
READ CLAUDE.md AND AGENTS.md FIRST. Work in the STANDALONE repo october25kim/fedcore (a fresh proper
clone, NOT the /tmp/fedcore-confirm throwaway). This is TEST-HARNESS + DOCS only: NO change to any
number, metric value, schema name, RNG seed, threshold, split, or experiment logic. Report in the
fixed format (진단 요약 / 확인한 명령 / 핵심 결과 / 판정 / 다음 행동). STOP-AND-ASK where marked.

SETUP.
  rm -rf /tmp/fedcore-work
  git clone git@github.com:october25kim/fedcore.git /tmp/fedcore-work
  cd /tmp/fedcore-work && pip install -e .

TASK 1 (main) — close the golden stdout-gate gap (Issue 2).
  The three CPU-script stdout goldens exist (tests/golden/exp_lemma_L.stdout.txt,
  exp_pooling_fail.stdout.txt, run_smoke.stdout.txt) but are NOT diffed by any gate, even though
  golden_check.py's docstring claims they are — so those scripts' output bit-identity is unenforced.

  STEP A — DIAGNOSE FIRST (do NOT wire in yet). Run:
    python -m fedcore.experiments.exp_lemma_L
    python -m fedcore.experiments.exp_pooling_fail
    python -m fedcore.experiments.run_smoke
  Capture each stdout and compare to its pinned golden with a TOLERANT comparison — normalize
  whitespace, extract floats and compare within 1e-6, compare remaining text after normalization —
  NOT a raw byte diff. Report per script: MATCH or MISMATCH (with the diff).
    -> If ANY script MISMATCHES, STOP-AND-ASK and report the drift. Do NOT wire it in and do NOT
       re-snapshot the golden — a mismatch means the gate never caught a real change; surface it for a
       decision.

  STEP B — only if all three MATCH: wire the tolerant stdout comparison INTO golden_check.py so
  `make repro-check` actually enforces the three scripts' stdout (same tolerant method, never raw
  byte). Update golden_check.py's docstring to match the now-true behavior. Re-run `make repro-check`
  -> PASS; confirm the report shows the three JSON goldens AND the three stdout checks as RAN/PASS,
  while artifact-dependent checks (certify_frozen, *_agg) still LOUD-SKIP in a bare clone. TEST-HARNESS
  change only — no experiment numbers change.

TASK 2 (verify + record) — torch entry `--help` (Issue 1). NO code change to entry points.
  run_cifar / run_foogd_cifar / run_selftrain_pkg import torchvision, which the GPU pytorch container
  provides (not `pip install -e .`), so `--help` failing OUTSIDE the container is BY DESIGN, not a
  defect.
  STEP A — verify INSIDE the container: run the three `--help` inside the GPU pytorch container
  (scripts/docker_cifar.sh env / the torch image the docker scripts use). Confirm they parse without
  error there. Report.
  STEP B — RECORD a follow-up (docs only): append to HANDOFF.md (or a KNOWN_GAPS section) exactly:
    "Follow-up: lazy-import torchvision inside the training entry points (run_cifar / run_foogd_cifar /
     run_selftrain_pkg) so `--help` and arg-parsing work outside the GPU container — improves CLI
     ergonomics and makes bare-clone import checks meaningful. Do NOT add torchvision to pip install
     requirements (torch/CUDA version matching); keep it container-provided."
  Do NOT modify run_cifar/foogd/selftrain code now.

COMMIT + PUSH (standalone repo).
  Two scoped commits (explicit pathspecs): (1) Task 1 golden gate + docstring; (2) Task 2 doc note.
  Push to october25kim/fedcore. Normal push / fast-forward expected — if the remote moved, STOP-AND-ASK
  before any force operation.

CLEANUP. rm -rf /tmp/fedcore-confirm  (the earlier throwaway confirmation clone).

DO NOT: change experiment numbers/schema/seed/split/logic; re-snapshot a golden to mask a mismatch;
force torchvision into pip; touch the source monorepo (ssrc).

REPORT (fixed format): 진단 요약 / 확인한 명령 / 핵심 결과 (stdout MATCH/MISMATCH per script; gate
wired + docstring corrected; container `--help` results; follow-up recorded) / 판정 / 다음 행동.
STOP-AND-ASK if any stdout golden mismatches in Task 1 Step A.
```

---

### Notes for Sanghoon
- **Task 1이 핵심**: stdout golden을 게이트에 실제로 강제 — 단 **먼저 현재 stdout == pinned golden 확인**하고, **불일치면 STOP**(게이트가 못 잡은 실제 drift일 수 있으니 몰래 re-snapshot 금지). tolerant 비교(숫자 1e-6, 공백 정규화)라 무해한 포맷 차이론 안 깨집니다. test-harness 변경이라 science 무영향.
- **Task 2는 코드 변경 없음**: `--help`를 컨테이너 안에서 확인(설계상 정상 확인) + lazy-import는 HANDOFF에 follow-up으로 기록만. torchvision을 pip에 강제하지 않음(버전 매칭 지옥).
- **home = standalone**이니 수정은 october25kim/fedcore에만. monorepo(ssrc) 사본은 별도로 deprecate/mirror 결정하세요(이 프롬프트는 안 건드림).
- Task 1 Step A 결과(3개 스크립트 MATCH/MISMATCH)를 붙여주시면 제가 같이 판정하겠습니다 — 특히 mismatch가 나오면 그건 조사할 실제 신호입니다.
