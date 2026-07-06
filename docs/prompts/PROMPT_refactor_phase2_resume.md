# Claude Code prompt — RESUME the package relocation (Phase 2 remaining, GPU-parity gated)

Context (UPDATED — CORE already packaged). The structure-only refactor reached a clean
**library-core boundary**: the deterministic certification CORE is now packaged under `fedcore/`
(17 modules: `certificate/{cp,theorem1,theorem3,feasibility}`, `scores`, `selector`,
`data/{fedosr_split,clients,noise}`, `models/{models,fed_train}`, `config`, `certify`) with **10 flat
shims** (explicit re-exports, leaf rule respected — `fedcore/*` never imports a shim). All golden
bit-for-bit, `make repro-check` PASS, behavior-preserving. Banked commits include the Phase-0 golden
suite, schema/guard centralization, `atomic_io.py` race fix, covtype/T8/selftrain golden snapshots,
Phase-3 reproducibility deliverables, and the M1/M2/M3a core relocation.

So **M1 (certificate split), M2 (scores/selector/data/models), M3a (config+certify) are DONE.** The
remaining, deliberately deferred work is the higher-risk, golden-uncovered surface:
  - M3 plotting move, M4 aggregator consolidation, M5 experiments + GPU entry-point move, M6 import-migration.

Why it was deferred and what this prompt adds. The golden suite covers deterministic certify /
scores / aggregate / run_smoke, but **NOT** the GPU entry points (run_cifar / run_foogd / run_fedpd),
so a shim mistake there could break silently. This prompt adds an explicit **GPU-parity gate**, the
must-fixes agreed in review, and a **shared-helper extraction precondition** flagged in the last run
(extract `_group_map` / `_repartition` / `_views_from_parts` before consolidating, since the runners
share this grouping logic). **Run this in a session with GPU available** so the moved entry points
can be exercised for real, not just import-smoked. Paste the fenced block into Claude Code in the
FedCORE repo. Note: the library-core boundary is already a coherent, shippable state — the steps
below are optional polish, not required debt (the race bug is already fixed by `atomic_io.py`).

```text
READ CLAUDE.md AND AGENTS.md FIRST. This RESUMES a structure-only refactor (branch
refactor/structure-repro). The CORE is already packaged (M1/M2/M3a done, 10 shims, golden green);
`make repro-check` PASS. STRUCTURE ONLY: zero
change to any number, metric value, schema name, threshold, RNG seed, or split logic. The golden
suite (bit-for-bit <=1e-9) plus a GPU-parity check are the pass/fail criteria. One concern per commit;
golden green before every commit. Docker-first. Report in the fixed format. STOP-AND-ASK where noted.

PRE-FLIGHT (no source change).
  - Confirm `make repro-check` PASS and `fedcore` working tree clean (only gitignored runs/data left).
  - Confirm on branch refactor/structure-repro.
  - Verify the PARENT-root .gitignore excludes FedCORE/runs/**, **/*.npz/*.npy/*.pt/*.pth, figs/*.png,
    data/, checkpoints/. We will `git add` EXPLICIT source pathspecs only — NEVER `git add -A` or a
    directory add — and never touch the parent repo's unrelated pending deletions.
  - Refresh REFACTOR_PLAN.md with a "DONE vs REMAINING" status section. STOP-AND-ASK before moving code.

INVARIANTS (verified before every commit by `tests/golden_check.py`).
  1. canonical schema names unchanged; 2. certificate math bit-for-bit (<=1e-9); 3. split-index
  determinism + disjointness; 4. public CLIs + scripts/docker_*.sh run unchanged (via shims);
  5. CPU sanity (exp_lemma_L, exp_pooling_fail, run_smoke) green.

SHIM RULES (critical — prevents silent breakage and circular imports).
  - Leave a backward-compat shim at each OLD flat path re-exporting from the new fedcore location.
  - `fedcore/*` modules MUST NEVER import an old flat shim path (shims are LEAVES; one-directional).
  - Shims use EXPLICIT re-exports (`from fedcore.x import a, b, c`) or a defined `__all__` — NO bare
    `import *` that could drop underscore/private API or submodule attributes.

DONE (do NOT redo): M1 certificate split (-> fedcore/certificate/{cp,theorem1,theorem3,feasibility}),
  M2 scores/selector/data/models, M3a config+certify. 10 shims in place, golden green.

REMAINING MOVES (incremental; golden green before each commit; move code, do NOT rewrite logic).
  M3. Move plotting/ (make_*). Figure OUTPUT paths MUST stay experiments/fedcore/figs/*.png (the
      manuscript references them) — verify a regenerated figure writes the same path.
  PRECONDITION for M4/M5 — extract shared helpers. The runners share grouping logic
      (`_group_map` / `_repartition` / `_views_from_parts`); extract these into one fedcore module
      FIRST (behavior-preserving, golden green) so the aggregator/runner moves don't duplicate or
      diverge them.
  M4. CO-LOCATE the aggregators -> fedcore/aggregate/ subpackage (one module each:
      main/t8/covtype/selftrain) + flat shims. **Do NOT merge into a single unified function** — that
      was an over-specification: 3 of the 4 aggregators use ddof=0 (population SD) and unifying ddof
      would change numbers and break the byte-identical golden. Instead: move each aggregator as-is,
      and extract ONLY micro-helpers that are byte-for-byte identical across modules (e.g. mean / n_pass);
      if sharing a helper would alter any module's ddof or rounding, leave it duplicated (byte-identical
      > DRY). PRESERVE per-caller behavior exactly: each module keeps its CURRENT ddof (do NOT unify)
      and its CURRENT convergence guard (self-training 0.30; covtype/T8 their per-cell values — read
      each live value and preserve). Keep seed-aware n_seeds (distinct seeds), grid-aware keys
      (labeled_frac/audit_mult/beta). BEFORE the move, add a SUB-GUARD row (known_acc below the drop
      threshold) to the selftrain aggregate golden fixture so the guard path is pinned. All *_agg
      golden CSVs must stay byte-identical. (Authority is the byte-identical golden, not a DRY ideal.)
  M5. Move the experiments/ runners (exp_*, run_smoke, run_cifar, run_foogd*, run_fedpd*,
      run_selftrain*, selftrain*) with shims. Public CLIs keep working. (certify.py is already in
      fedcore from M3a.)

GPU-PARITY GATE (run AFTER M5, before declaring done — this closes the golden-uncovered surface).
  For EACH moved GPU entry point (run_cifar, run_foogd, run_fedpd, and the self-training runners):
    a. import-parity: importing the old shim path and the new fedcore path both succeed; `--help` /
       a dry-run (no training) parses identically.
    b. end-to-end-parity: run ONE short real job (smallest rounds/epochs, fixed seed, deterministic
       flags where available) and confirm it completes and EXPORTS a canonical-schema output without
       error. If the run is configured deterministic, compare the exported logits .npz against a
       reference captured from the SAME short run BEFORE the move within tol (state the tol). If torch
       nondeterminism prevents bit-parity, the criterion is "runs to completion + valid canonical-schema
       output + certify-on-its-logits within tol" — state explicitly which criterion was used per entry
       point. Do NOT claim bit-parity for stochastic training.
  STOP-AND-ASK with the parity results before the final commit.

M6 (separate follow-up PR — shim pruning + import migration; the file-clutter cleanup).
  This is the only step with hidden-dependency risk (a shim removal can break references that
  repo-grep + golden cannot see), so do it as its OWN small PR on top of the merged relocation, with
  an explicit REFERENCE AUDIT first. It is fully reversible (git revert) and golden-gated.
  AUDIT (before removing anything):
    1. grep the WHOLE repo for flat-path imports (`import certificates`, `from scores import ...`,
       etc.) and rewrite internal call-sites to `fedcore.*`.
    2. Build a KEEP-LIST of flat paths referenced by PUBLIC entry points — `scripts/docker_*.sh`,
       `CLAUDE.md`, `REPRODUCE.md`, `Makefile`, and any notebooks — e.g. run_cifar / run_smoke /
       exp_lemma_L / exp_pooling_fail and the docker-invoked runners. These shims STAY (or update the
       referencing doc/script in the same commit). Preserve run_smoke's `__file__`-anchored output
       path.
    3. Remove ONLY the shims that, after step 1, have no internal users AND are not on the keep-list
       (the ~30 internal library/make/aggregator shims).
    4. Re-verify: golden bit-for-bit + import/`--help` parity on every public entry point + the
       representative GPU end-to-end. One commit. If any reference is uncertain, KEEP the shim.

DO NOT.
  - change any number, metric value, threshold, RNG seed, schema name, or split logic;
  - rename canonical metric/config keys or Fed-CORE concepts; reintroduce SRCC/RC-OWPL/pseudo-labeling;
  - delete or overwrite any runs/*.csv, *_logits.npz, data/, or checkpoints/;
  - `git add -A` / directory-add, or commit the parent repo's unrelated deletions;
  - big-bang rewrite; hide a failing golden test, parity failure, or failed command.

GIT. One concern per commit on refactor/structure-repro; EXPLICIT fedcore source pathspecs only;
  golden green before each commit; English commit messages. git_*.sh manual if absent.

REPORT (fixed format) after pre-flight, after each M-step, and at the GPU-parity gate:
  진단 요약 / 확인한 명령 / 핵심 결과 (이동·통합 내역, golden pass, GPU-parity 기준·결과, 발견 diff)
  / 판정 (strong go / moderate go / warning / fail) / 다음 행동. STOP-AND-ASK at the two marked gates.
```

---

### Notes for Sanghoon
- 이 프롬프트의 핵심 추가는 **GPU-parity gate**입니다 — golden이 못 덮는 run_cifar/foogd/fedpd를
  import+--help parity와 짧은 end-to-end 실행으로 실제 검증. torch 비결정성 때문에 bit-parity는
  주장하지 않고 "정상 완료 + canonical schema 산출 + 그 logits certify가 tol 내"를 기준으로 둡니다.
- 합의한 must-fix를 전부 내장: caller별 guard threshold(통합 금지) + sub-guard fixture, fedcore는
  shim을 import하지 않음 + 명시적 re-export/`__all__`, Thm 매핑(stratified→theorem1, pooled→theorem3),
  figure 출력 경로 불변, 끝에 import-migration 커밋으로 50 shim 영구화 방지.
- **GPU 가용 세션에서** 돌리세요. 두 곳(pre-flight 후, GPU-parity gate)에서 STOP-AND-ASK 하니,
  중간 결과를 저에게 붙여주시면 검토하고 다음 단계 보내드리겠습니다.
- M4(aggregator 통합)가 이번에 미뤄둔 마지막 실질 부채입니다 — guard 파라미터화와 sub-guard fixture만
  지키면 golden이 정확히 잡아줍니다.
