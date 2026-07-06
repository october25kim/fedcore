# Claude Code prompt — hoist package to project root + conservative dedup (golden-gated)

Goal. Two cleanups on top of the merged/pushed structure-only relocation:
  (1) FIX the triple-nesting: the core package currently lives at
      `fedcore/experiments/fedcore/fedcore/` ("fedcore" ×3, and the core library is awkwardly nested
      under `experiments/`). Hoist it to the project root as `fedcore/fedcore/`, made importable via a
      proper editable install — so `experiments/` imports the library, not the reverse.
  (2) Remove TRUE duplicate / dead files and tidy strays — conservatively, behind a reference audit
      and an explicit approval gate.

Constraints. Same discipline as the relocation: STRUCTURE/CLEANUP ONLY — zero change to any number,
metric, schema, seed, threshold, or split logic; the golden suite (bit-for-bit <=1e-9) + GPU-parity
are the pass/fail criteria. This is HIGHER risk than the relocation (it touches sys.path / docker /
CLAUDE.md paths AND deletes files), so: one concern per commit, golden green before each, reference
audit before any deletion, STOP-AND-ASK at the marked gates, and rely on git (everything is committed
and pushed) as the undo net. **When uncertain, KEEP and report — never delete on a guess.** Run in a
session with GPU available for the parity re-check. Paste the fenced block into Claude Code.

```text
READ CLAUDE.md AND AGENTS.md FIRST. Resumes cleanup on top of the pushed relocation (fedcore package
at fedcore/experiments/fedcore/fedcore/, ~40 shims). STRUCTURE/CLEANUP ONLY: zero change to any
number/metric/schema/seed/threshold/split. golden (<=1e-9) + GPU-parity are pass/fail. One concern
per commit; golden green before each; reference audit before any deletion; STOP-AND-ASK where marked;
git is the undo net. NEVER delete runs/*.csv, *_logits.npz, data/, checkpoints/. Conservative: when
uncertain, KEEP.

PRE-FLIGHT (no change). New branch refactor/hoist-dedup off the pushed relocation. Confirm
`make repro-check` PASS + clean tree. Snapshot the current public run commands (from CLAUDE.md /
docker_*.sh / Makefile / REPRODUCE.md) so we can verify they still work after the hoist. STOP-AND-ASK
with the plan (hoist target + the dedup candidate METHOD) before changing anything.

PHASE A — finish M6 first (import migration + shim pruning), if not already done.
  grep the WHOLE repo for flat-path imports and rewrite internal call-sites to fedcore.*. Build a
  KEEP-LIST of flat paths referenced by PUBLIC entry points (scripts/docker_*.sh, CLAUDE.md,
  REPRODUCE.md, Makefile, notebooks); those shims stay (or update the referencing doc in the same
  commit). Remove only internally-orphaned shims. Preserve run_smoke's __file__-anchored output path.
  golden + import/--help parity green. (Doing M6 first means fewer shims to carry through the hoist.)

PHASE B — hoist the core package to the project root (fix the triple-nesting).
  Move fedcore/experiments/fedcore/fedcore/  ->  fedcore/fedcore/  (project root = the FedCORE project
  dir). Make it a proper installable package: add pyproject.toml (or setup.py) at the project root and
  `pip install -e .` so `import fedcore` resolves with NO sys.path hacks. Then:
    - update experiments/ runners + shims to import the hoisted fedcore.*;
    - update run paths in CLAUDE.md, scripts/docker_*.sh, REPRODUCE.md, Makefile to the new locations;
    - keep public-entry behavior identical (a documented command must still run; update the command
      text if the path changed, in the same commit).
  Verify: golden bit-for-bit + import/--help parity on every public entry point + the representative-3
  GPU end-to-end (run_cifar / run_foogd_cifar / run_selftrain_pkg) — certify on frozen logits must stay
  bit-identical. STOP-AND-ASK with the path-change diff (since this edits CLAUDE.md/docker commands).

PHASE C — conservative dedup (STOP-AND-ASK before ANY deletion).
  DETECT, do not yet delete:
    - exact duplicates: hash every tracked .py; list byte-identical pairs.
    - near-duplicates / suspected dead code: diff the clusters — self-training
      (self_training, selftrain, selftrain_oneshot, exp_self_training, run_selftrain_{cifar,pkg,fedpd,smoke}),
      necessity (exp_necessity vs exp_necessity_real), foogd (run_foogd_cifar vs run_foogd_full_cifar),
      and any others surfaced by the hash/diff.
  CLASSIFY each candidate with EVIDENCE (grep references across repo + git log history + docker/CLAUDE/
  Makefile/REPORT references):
    - EXACT-DUP  : byte-identical and the duplicate is unreferenced -> propose delete.
    - DEAD       : never imported, never invoked by any script/doc/manifest, superseded -> propose delete.
    - PROVENANCE : a script that PRODUCED a paper result (figure/table/REPORT) -> KEEP (or move to a
                   reports/legacy/ only if the user approves); never silently delete.
    - VARIANT    : has real behavioral differences -> KEEP.
  HOUSEKEEPING (low risk): move REPORT_*.md + LEMMA_L_proof.md -> reports/; move stray smoke_results.csv
  -> runs/ (or delete only if regenerable AND unreferenced); ensure __pycache__/ is gitignored.
  GATE: present the FULL candidate list (path, classification, evidence, proposed action) and DELETE/MOVE
  ONLY what the user approves. After approved deletions: golden + import/--help parity + representative
  GPU end-to-end green. git is the undo net.

DO NOT.
  change any number/metric/schema/seed/threshold/split; delete or overwrite runs/*.csv, *_logits.npz,
  data/, checkpoints/; delete a PROVENANCE script without explicit approval; delete on a guess
  (uncertain -> KEEP); reintroduce SRCC/RC-OWPL naming; touch the parent repo's unrelated projects
  (UPLIFT-v1, RC-OWPL, feduno, ...) or its unrelated pending deletions; big-bang; hide a failing golden
  test, parity failure, or failed command.

GIT. branch refactor/hoist-dedup; one concern per commit; EXPLICIT fedcore-source pathspecs only
  (never `git add -A`/dir-add); golden green before each commit; English messages; reversible.

REPORT (fixed format) after pre-flight, Phase A, Phase B, and the Phase-C detection (before deletion):
  진단 요약 / 확인한 명령 / 핵심 결과 (이동·삭제 후보 목록 + 분류 + 근거, golden pass, parity 결과,
  경로 변경 diff) / 판정 (strong/moderate/warning/fail) / 다음 행동. STOP-AND-ASK at: pre-flight plan,
  the Phase-B path-change diff, and the Phase-C deletion list.
```

---

### Notes for Sanghoon
- **순서가 중요**: M6(shim 정리) → Phase B(루트로 hoist + editable install) → Phase C(dedup). hoist 전에
  shim을 줄이면 옮길 게 적습니다.
- **삭제 안전장치 3중**: (1) reference audit(grep+git log+docker/CLAUDE/Makefile/REPORT 참조 확인),
  (2) **삭제 전 후보 목록 STOP-AND-ASK 승인**, (3) 전부 commit/push돼 있어 git revert로 복구 가능.
- **provenance 보호**: 논문 figure/table를 생산한 experiment/runner 스크립트는 "중복처럼 보여도"
  기본 보존(승인 시 reports/legacy/로 이동만). self-training 8파일 클러스터가 최대 후보지만, 바로
  그래서 audit이 필수입니다.
- **Phase B는 CLAUDE.md/docker 명령 경로를 바꿉니다** — 같은 커밋에서 문서·스크립트를 함께 갱신하고
  parity로 검증합니다. pyproject.toml + `pip install -e .`로 import가 sys.path hack 없이 풀리는 게
  핵심 개선입니다.
- GPU 가용 세션에서 돌리세요(parity 재검증). 세 STOP gate에서 중간 결과를 붙여주시면 제가 검토하고
  다음 단계로 넘기겠습니다 — 특히 **Phase C 삭제 목록은 제가 같이 확인**한 뒤 승인하시길 권합니다.
