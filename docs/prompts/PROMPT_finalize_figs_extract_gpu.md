# Codex prompt — finalize (a) figs regen check, (b) standalone extract at HEAD, (c) short GPU path check

Three closing checks on the current HEAD of `refactor/hoist-dedup` after the io_utils/grouping helper
extraction. All structure/verify only — no behavior change. Run in a session with GPU available (Task
B). Paste the fenced block into Codex in the fedcore repo.

```text
READ CLAUDE.md AND AGENTS.md FIRST. This finalizes the structure refactor with three checks on the
current HEAD of branch refactor/hoist-dedup. STRUCTURE/VERIFY ONLY: zero change to any number, metric
value, schema name, RNG seed, threshold, or split logic. Do NOT touch the parent monorepo's unrelated
projects (UPLIFT-v1, RC-OWPL, selective_risk_cifar, fedtold, ...) or its pending deletions; NEVER
`git add -A` or a directory add. Report EACH task in the fixed format (진단 요약 / 확인한 명령 /
핵심 결과 / 판정 / 다음 행동). STOP-AND-ASK where marked.

PRE-FLIGHT (no change): `make repro-check` PASS and the fedcore working tree clean. Confirm HEAD
includes the helper-extraction commit (fedcore/io_utils.py, fedcore/grouping.py public helpers
make_group_map / repartition_trusted_pool / views_from_parts).

TASK A — verify figure regeneration is content-unchanged (plotting-refactor safety).
  Regenerate the figures whose generators now import fedcore.grouping / fedcore.io_utils (make_figures,
  make_composites, make_F8, make_selftrain_gain, make_problem_diagram, make_corruption_curve, and any
  other touched make_*), via the current entry points (Makefile figs target or
  `python -m fedcore.plotting.<name>`), overwriting experiments/fedcore/figs/ in place. Then:
    - `git status --porcelain experiments/fedcore/figs/` and `git diff --stat experiments/fedcore/figs/`.
    - If a PNG shows as modified, matplotlib metadata (timestamps) can differ even when the plot is
      identical, so CONFIRM CONTENT-IDENTITY: decode the regenerated PNG and the committed one (HEAD:)
      to pixel arrays (PIL/numpy) and assert pixel-equal; for PDFs compare drawn content. Report, per
      figure, whether it was byte-identical or pixel-identical (metadata-only diff).
    - After confirming, discard any metadata-only changes: `git checkout -- experiments/fedcore/figs/`.
  ACCEPTANCE: every regenerated figure is PIXEL-identical to the committed version. If ANY figure
  differs in content, STOP and report the diff (do NOT commit) — the helper refactor changed a plot.

TASK B — short GPU end-to-end (confirm the helper + `python -m` refactor didn't break the GPU path).
  Run `bash scripts/docker_cifar.sh` with the SMALLEST config (ROUNDS=1 / smoke env), writing to a TEMP
  or parity output dir — do NOT overwrite the real runs/. Criterion (NOT bit-parity; GPU training is
  nondeterministic per REPRODUCE): the pipeline runs to completion through the
  `python -m fedcore.experiments.*` entry points and EXPORTS canonical-schema output (certified,
  cert_risk_ucb, cert_coverage_lcb, cert_n, cert_k, prop_coverage, prop_risk, test_coverage, test_risk,
  score_name, gamma, alpha, delta, ...) with no error. Then delete the temp outputs; leave real runs/,
  data/, checkpoints/ untouched. Report the exported schema (column names) + completion status.

TASK C — extract fedcore/ to the standalone repo october25kim/fedcore at the CURRENT HEAD (history preserved).
  ONLY after Tasks A and B are green.
    1. On branch refactor/hoist-dedup (checkout if needed).
    2. `git branch -D fedcore-export 2>/dev/null; git subtree split --prefix=fedcore -b fedcore-export`
       — extracts only fedcore/-touching commits (fedcore/ becomes the repo root), including the latest
       helper-extraction + shim-removal commits. History PRESERVED (not a fresh init). subtree split
       reads committed history only, so the parent monorepo's dirty/untracked changes are irrelevant.
    3. `git ls-remote https://github.com/october25kim/fedcore.git`:
       - EMPTY (no refs) -> `git push https://github.com/october25kim/fedcore.git fedcore-export:main`
       - HAS refs/heads/main (a stale earlier extract) -> STOP-AND-ASK before force-pushing; on
         approval: `git push --force https://github.com/october25kim/fedcore.git fedcore-export:main`
    4. Post-push sanity in a throwaway clone:
       `git clone https://github.com/october25kim/fedcore.git /tmp/fedcore-verify && cd /tmp/fedcore-verify
        && pip install -e . && make repro-check`  -> must PASS. Confirm the repo root shows fedcore/
       (package) + experiments/ + pyproject.toml and contains NO other monorepo projects.
  Do NOT `git add -A`; use only the subtree/push commands above. Do NOT touch the parent repo's
  unrelated changes.

REPORT each task (fixed format). STOP-AND-ASK: (A) if any figure differs in content; (C) before the
force-push. Output exact numbers/paths so the Mac side can confirm.
```

---

### Notes for Sanghoon (Codex 사용)
- 순서: **(a) figs → (b) GPU → (c) 추출**. HEAD가 a·b로 완전 검증된 뒤 추출해야 standalone repo가 검증된 상태로 안착합니다.
- **(a)는 pixel-identity 기준**입니다 — matplotlib PNG는 메타데이터(타임스탬프)가 매번 달라 byte-diff가 뜰 수 있으니, "플롯 내용이 같은가"를 픽셀로 확인하고 metadata-only diff는 `git checkout`으로 버립니다. logic 무변경이라 pixel-identical이어야 정상.
- **(b)는 bit-parity가 아니라 "완주 + canonical schema 산출"** 기준(GPU 비결정성). 임시 출력 dir 사용, 실제 runs/ 미접촉.
- **(c)는 이력 보존(subtree split)** + 원격 상태 확인 후 push(비어있으면 `:main`, stale면 승인 후 `--force`). 부모 모노레포 무관 변경·`git add -A` 금지.
- Codex는 STOP-AND-ASK를 카드처럼 대화형으로 못 멈출 수 있으니, **force-push 직전엔 보고만 하고 대기**하도록 지시돼 있습니다 — 그 보고가 오면 승인 주세요. 두 게이트(figs 내용 diff, force-push) 결과를 저에게 붙여주시면 검토하겠습니다.
