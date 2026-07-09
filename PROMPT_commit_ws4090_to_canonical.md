# Claude Code prompt — commit ws4090 work to october25kim/fedcore (ssrc deleted; fedcore is sole home)

`october25kim/ssrc` was deleted; `october25kim/fedcore` (standalone, hoisted layout) is now the SOLE
home. This runs ON the ws4090 server: fresh-clones fedcore, brings the ws4090 changes from the local
WORK dir, verifies package changes against the golden, updates the T8 golden to 5 seeds, and commits on
a feature branch + PR. Files are safe on the WORK disk; only the deleted ssrc git history is gone.
Paste the fenced block into Claude Code on the ws4090 server.

```text
READ CLAUDE.md AND AGENTS.md FIRST. Runs ON the ws4090 server. october25kim/ssrc has been DELETED;
october25kim/fedcore (standalone, hoisted layout) is the SOLE home — commit there.
  CANON = a FRESH clone of october25kim/fedcore (created in STEP 0).
  WORK  = /mnt/hdd/workspace/sanghoon/fedcore  (ws4090 working dir, ssrc-structured; SOURCE of files).
STRUCTURE + DATA-SCRIPT commit only — no experiment numbers change. Do NOT commit runs/, data/, weights.
Report in the fixed format; STOP-AND-ASK at the two marked gates.

STEP 0 — token access + fresh clone.
  Confirm the token can reach fedcore (it returned Not Found earlier — it needs october25kim/fedcore
  access on the PAT):
    git ls-remote https://<TOKEN>@github.com/october25kim/fedcore.git   # must succeed; if not, STOP.
  Then:
    git clone https://<TOKEN>@github.com/october25kim/fedcore.git ~/fedcore-canonical
    export CANON=~/fedcore-canonical WORK=/mnt/hdd/workspace/sanghoon/fedcore
    cd "$CANON" && pip install -e . && make repro-check          # baseline must PASS
  NOTE: fedcore predates the ws4090 work, so files that were 'already committed' on the deleted ssrc
  branch may be MISSING here — copy them too. All copies are server-local (WORK -> CANON), no transfer.

STEP 1 — copy the ws4090 files WORK -> CANON (new or updated). After copying, run `git -C "$CANON"
  status` to see which are new vs modified:
    scripts/docker_grid.sh
    scripts/ws4090/          (dispatch.py, gen_manifest*, certify_grid*, certify_client_scaling*,
                              manifest_R3.txt, manifest_R6_J3.txt)
    experiments/fedcore/exp_r4_knob_sensitivity.py
    experiments/fedcore/exp_r6_simplex_positive.py
    experiments/fedcore/exp_r7_covtype_stable.py
    experiments/fedcore/exp_r8_detector_reconcile.py
    experiments/fedcore/exp_alpha20_diagnostics.py

STEP 1b — M-queue additions (2026-07-10; copy WORK -> CANON, same rules as STEP 1;
  place experiment runners at the CANONICAL location for experiment code — check where the
  existing exp_* files live in CANON, hoisted layout = fedcore/experiments/):
    exp_grouped_validity_stress.py   exp_r6_simplex_grid.py
    exp_a4_composition_stress.py     exp_dp_count_release.py
    exp_client_scaling_synth.py      exp_oracle_comparison.py
    exp_unknown_split_robustness.py  exp_m6_cifar100_fedpd.py
    build_selftrain_gain_5seed.py
    scripts/run_selftrain_fedpd_5seed.sh   scripts/run_fedpd_cifar100_m6.sh
    scripts/ws4090/manifest_M2.txt scripts/ws4090/manifest_M4.txt scripts/ws4090/manifest_M4ext.txt
    scripts/ws4090/dispatch.py    (updated since STEP 1's copy)
  EXPLICITLY EXCLUDED — do NOT copy or commit plotting files (make_figures.py,
  make_composites.py): the manuscript-side (laptop) copies supersede them; the laptop
  make_figures.py already contains the Theorem-3 relabel AND the new Figure-4d panel, and
  committing the server copy would regress it. Plotting reconciliation is a follow-up owned
  by the manuscript side.

STEP 2 — PACKAGE changes (GATE 1). Copy WORK -> CANON:
    fedcore/certificate/theorem1.py
    fedcore/aggregate/t8.py
  Also copy the M-era package changes (GATE 1 applies to each):
    fedcore/data/fedosr_split.py        (additive unknown_classes support)
    run_cifar.py at its canonical path   (plumbs --unknown_classes)
    run_selftrain_fedpd.py               (oracle-mode import fix only)
  `git -C "$CANON" diff` each (and exp_alpha20_diagnostics.py from STEP 1) and CONFIRM only the intended
  change is present:
    - theorem1.py: ONLY the box sup replaced by the O(J log J) rbar-sort solver (bit-identical).
    - t8.py: ONLY the detector SEEDS extended 3 -> 5.
    - exp_alpha20_diagnostics.py: ONLY the detector 5-seed extension plus the baseline 10-seed CELLS.
    - fedosr_split.py: ONLY the additive unknown_classes parameter; default None must preserve the
      seed-driven split bit-identically (goldens unchanged).
    - run_cifar.py: ONLY the --unknown_classes flag plumbing.
    - run_selftrain_fedpd.py: ONLY the `from fedcore.experiments.selftrain import ...` import fix.
  If a diff shows unrelated changes (ssrc-era divergence), STOP-AND-ASK — do not commit a divergent file;
  re-apply only the intended change to the fedcore version instead.
  Verify: `make repro-check`. theorem1.py must keep the JSON goldens bit-identical (<=1e-9); the agg-t8
  golden will intentionally diff (handled in STEP 3). If theorem1 breaks any golden, STOP-AND-ASK.

STEP 3 — golden T8 fixture 5-seed update (GATE 2; user-approved). Before regenerating, confirm the seed
  0-2 rows of the T8 aggregate are BIT-IDENTICAL to the old 3-seed golden (purely additive: seeds 3,4
  added, existing seeds unchanged). If seed 0-2 differ, STOP-AND-ASK (that would be a rewrite). Then
  regenerate tests/golden/T8_fedosr_bases_agg* at 5 seeds and re-run `make repro-check` -> PASS. This is
  an intended seed extension, not an oracle rewrite; say so in the commit message.

STEP 4 — SKIP. paper_figs.py is not in this repo; the corruption figure was already regenerated on the
  manuscript side. Do NOT add it or hunt for a figure generator here.

STEP 5 — commit on a FEATURE BRANCH + open a PR (race-free, reviewable). Branch off main:
    git -C "$CANON" checkout -b expansion/ws4090-queue
  Scoped commits (EXPLICIT pathspecs; never `git add -A`):
    c1  scripts/docker_grid.sh scripts/ws4090/*                      "feat: multi-GPU dispatcher + docker_grid + manifests"
    c2  experiments/fedcore/exp_r{4,6,7,8}_*.py                      "feat: R4/R6/R7/R8 experiment runners"
    c3  fedcore/certificate/theorem1.py                             "perf: O(J log J) box sup (bit-identical)"
    c4  fedcore/aggregate/t8.py experiments/fedcore/exp_alpha20_diagnostics.py tests/golden/T8_fedosr_bases_agg*
                                                                     "data: detectors to 5 seeds + baseline 10-seed cells + refresh T8 golden (additive)"
    c5  <M-queue exp runners from STEP 1b> build_selftrain_gain_5seed.py
                                                                     "feat: M-queue runners (grouped validity, simplex grid, A4 composition, DP release, synth scaling, oracle comparison, unknown-split robustness, CIFAR-100 FedPD)"
    c6  scripts/run_selftrain_fedpd_5seed.sh scripts/run_fedpd_cifar100_m6.sh scripts/ws4090/manifest_M2.txt scripts/ws4090/manifest_M4.txt scripts/ws4090/manifest_M4ext.txt scripts/ws4090/dispatch.py
                                                                     "feat: M-queue launchers, manifests, dispatcher update"
    c7  fedcore/data/fedosr_split.py <run_cifar.py> <run_selftrain_fedpd.py>
                                                                     "feat: pre-declared unknown-class splits (additive); fix: selftrain oracle import"
    git push -u origin expansion/ws4090-queue
  Then open a PR into main (gh pr create if available, else print the compare URL). Do NOT commit runs/,
  data/, weights.

DO NOT: overwrite a package file without diff-verifying it (STEP 2); commit runs/data/weights; rewrite
the golden to mask a mismatch; change any experiment number/metric/schema/seed logic; `git add -A`;
force-push main.

REPORT (fixed format): 진단 요약 / 확인한 명령 / 핵심 결과 (token reaches fedcore?, files new/modified,
package diffs = intended-only?, theorem1 golden bit-identical?, seed 0-2 identity for t8, repro-check
PASS, branch pushed + PR URL) / 판정 / 다음 행동. STOP-AND-ASK at GATE 1 (package diff) and GATE 2 (golden).
```

---

### Notes for Sanghoon
- **선행조건**: PAT이 `october25kim/fedcore`를 볼 수 있어야 합니다(아까 Not Found). Fine-grained면 그 repo를
  토큰의 Repository access에 추가(Contents R/W), classic repo-스코프면 이미 됨. STEP 0의 `ls-remote`로 확인.
- **ssrc 삭제로 작업 유실 없음**: 파일은 WORK 디스크에 그대로. fedcore가 ws4090 작업 *이전*에 추출됐으니,
  아까 "ssrc에 이미 있던" 파일도 fedcore엔 없을 수 있어 **전부 copy** 대상입니다.
- **두 GATE가 핵심 안전장치**: theorem1은 diff가 O(JlogJ) 교체만 + golden bit-identical; t8는 seed 0–2
  동일성 확인 후 5-seed. 어느 쪽이든 예상 밖이면 STOP.
- **STEP 4(figs) skip**: 그림은 매니페스트 쪽에서 이미 완결(F7 재생성 + docx 반영).
- feature 브랜치+PR로 올려 race·리뷰 안전. 두 GATE 결과 붙여주시면 검토하겠습니다.
