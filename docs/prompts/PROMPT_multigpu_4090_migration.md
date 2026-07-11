# Claude Code prompt — migrate the pending experiment queue to a 4×RTX-4090 server (parallel, infra-only)

A 4×RTX-4090 (24 GB each) server is available. Re-target the pending R/E experiment queue from the
single 4070 to parallel multi-GPU execution. **Execution/infrastructure change ONLY** — the R/E task
DEFINITIONS (grids, seeds, metrics, split logic) are unchanged; only how they are dispatched changes.
Run in a clone of the standalone repo october25kim/fedcore on the new server. Paste the fenced block
into Claude Code.

```text
READ CLAUDE.md, AGENTS.md, AND docs/prompts/PROMPT_expansion_experiments_4070.md FIRST. Work in a fresh
clone of the standalone repo october25kim/fedcore on the 4×RTX-4090 server. GOAL: dispatch the pending
R/E queue in parallel across 4 GPUs. This is INFRASTRUCTURE ONLY — the R/E task DEFINITIONS (grids,
seeds, metrics, split logic) in the expansion prompt are UNCHANGED. Zero change to any experiment
number, metric value, schema name, RNG seed policy, or split logic. Reproducibility, canonical schema,
and split hygiene are preserved. Docker-first. Report in the fixed format; STOP-AND-ASK before the full
GPU queue. New scripts are committed; runs/ artifacts are not.

PRE-FLIGHT (no science change).
  - git clone git@github.com:october25kim/fedcore.git ; pip install -e . INSIDE the GPU pytorch
    container (torch+torchvision). `nvidia-smi -L` shows 4 GPUs; nvidia container toolkit runs docker
    with all 4 GPUs. `make repro-check` PASS. Report GPU list + torch/torchvision versions.

TASK 1 — GPU-selectable runners (behavior-preserving).
  Every training entry point (run_cifar, run_fedpd_cifar, run_foogd_cifar / run_foogd_full_cifar,
  run_selftrain_*, run_tabular) must honor CUDA_VISIBLE_DEVICES for GPU pinning and must NOT hardcode
  cuda:0 / a fixed device index. If they already use the default `cuda` device they inherit
  CUDA_VISIBLE_DEVICES automatically — VERIFY with a 2-GPU test (pin two cells to gpu 0 and gpu 1, check
  nvidia-smi). Add a `--gpu` passthrough only if a runner ignores the env var. NO numeric/logic change.

TASK 2 — multi-GPU job dispatcher (new committed script).
  Add scripts/run_grid_multigpu.py (+ a thin scripts/docker_grid.sh wrapper) that:
   - takes a GRID SPEC = a list of cells, where a cell is exactly one runner invocation with its
     dataset/model/dirichlet/alpha/seed args — REUSE the exact grids from the expansion prompt
     (R1/R2/R3/R6/R7). Do NOT invent or alter cells.
   - dispatches cells across NUM_GPUS (default 4) with JOBS_PER_GPU workers, pinning each worker via
     CUDA_VISIBLE_DEVICES=<gpu>. Per-model default packing for the 24 GB budget (tune conservatively):
       CIFAR ResNet-18 (GN/BN) / SimpleCNN: 3 per GPU;  WRN-28-10 FedPD-PROSER: 1–2 per GPU;
       FOOGD-full: 1 per GPU.
   - is IDEMPOTENT / RESUMABLE: skip any cell whose output row already exists in the target runs/ CSV
     (match on the cell's key columns: dataset,model,dirichlet_alpha,alpha,seed,score_name,gamma,...),
     so a crash or restart never re-runs a completed cell.
   - appends results to the shared runs/ CSV THROUGH THE ATOMIC LOCKED APPEND (fedcore.io_utils) —
     parallel writers MUST NOT race or corrupt the CSV (this is exactly what atomic_io was built for;
     verify the runners write via it). Per-cell logs -> runs/logs/<cell>.log.
   - runs inside ONE container with all 4 GPUs visible, spawning one subprocess per cell (each its own
     process => isolated seed/RNG/CUDA context).
   - env knobs: NUM_GPUS, JOBS_PER_GPU (scalar or per-model map), RESUME=1, DRY_RUN=1 (print the plan
     + GPU assignment only, run nothing).

TASK 3 — smoke the dispatcher BEFORE any real spend.
  (a) DRY_RUN the R2 grid: print the full cell plan and confirm the count matches the spec exactly
      (e.g. 3 models x 2 dirichlet x 10 seeds = 60 cells) with a valid GPU assignment.
  (b) Run a TINY real slice (2-3 cells, ROUNDS=1/LOCAL_EPOCHS=1) across 2 GPUs and confirm:
      - each cell lands on its assigned GPU (nvidia-smi during the run),
      - canonical-schema rows are appended with NO race/corruption (row count == cell count, headers
        intact),
      - RESUME=1 re-run skips the already-completed cells (0 re-runs),
      - a cell dispatched in parallel yields the SAME certification result as that cell run standalone
        (identical certify on its frozen logits; GPU-training nondeterminism only where already
        accepted by REPRODUCE).
      Delete the smoke outputs (do not pollute real runs/). STOP-AND-ASK with the smoke report before
      launching the queue.

TASK 4 (only after approval) — launch the pending queue, respecting the ordering dependency.
  DEPENDENCY (from the coordination rule): R8's T9 regeneration MUST run AFTER Task E's rows are present
  in runs/T9_diagnostics.csv. So the order is:
   0. Task E — FedPD-PROSER seeds 3->5 (d in {0.5,5}, alpha in {0.10,0.20}) + self-training 4x-budget
      5-seed -> appends to runs/T9_diagnostics.csv and writes runs/selftrain_gain_5seed.csv. This is the
      heaviest GPU work (WRN-28-10) and benefits most from 4-GPU parallelism. **IF Task E already
      finished on the 4070**, ingest its T9 / selftrain rows into this clone's runs/ and SKIP to step 1;
      otherwise run Task E here as the first dispatcher sweep (via the same runners/grids, unchanged).
   1. R8 (CPU, no GPU) — detector harmonization + simultaneous-T9 regeneration, once, now that Task E's
      rows are in T9. (This closes the P0 T8-vs-T9 detector mismatch and enables the simultaneous
      delta/2 headline transition.)
   2. GPU sweeps via the dispatcher in order R1 > R2 > R3 > R6 > R7 (R4/R5 are CPU/light, run when
      convenient). Target seeds = 10 (floor 5) per the spec.
  Report per task: 핵심 결과 (canonical metrics: cert_coverage_lcb mean±std, n_certified/n_seeds,
  cert_risk_ucb, test_risk) as each completes.

DO NOT: change any experiment number/metric/schema/seed policy/split logic; alter the R/E grids;
promote Thm 3 / Remark 1 over Thm 1/2; judge success by accuracy/AUROC; hide a failed cell; touch the
manuscript. Keep proposal/certification/test split hygiene; corruption on train labels only.

REPORT (fixed format) after pre-flight, Task 2 build, Task 3 smoke, and each Task-4 task:
  진단 요약 / 확인한 명령 / 핵심 결과 / 판정 / 다음 행동. STOP-AND-ASK before Task 4.
```

---

### Notes for Sanghoon
- **과학은 그대로, 실행만 병렬화**입니다 — grid/seed/split/schema 불변, 각 셀은 독립 프로세스(자기 seed·CUDA context)라 GPU 분산이 결과를 바꾸지 않습니다(golden·canonical schema 유지).
- **atomic_io가 여기서 값을 합니다**: 병렬 writer들이 같은 `runs/*.csv`에 append할 때 locked append로 race를 막습니다 — 우리가 리팩토링에서 만든 게 정확히 이 상황을 위한 것이었습니다. 프롬프트가 "반드시 io_utils 경유"를 강제합니다.
- **RESUME/idempotent**: R2(60 runs)처럼 큰 sweep은 중간에 죽어도 완료된 셀을 건너뛰게 했습니다 — 서버 작업엔 필수.
- **packing**: 4090 24GB라 CIFAR ResNet은 GPU당 3잡까지, WRN-28-10 FedPD는 1–2잡, FOOGD-full은 1잡으로 보수적 시작 후 튜닝. 4 GPU × 3잡 ≈ 최대 12 동시 실행.
- **STOP-AND-ASK 2곳**: smoke 후(dispatcher 검증), 그리고 full queue 전. smoke 결과 붙여주시면 packing/우선순위 같이 조정하겠습니다.
- **산출물 반입**: dispatcher는 `runs/`에 평소대로 씁니다 — 기존 "복사완료" 패턴으로 laptop에 반입하시거나, 앞서 논의한 rsync/Syncthing로 자동화하면 60-run 결과도 편하게 받습니다.
- **home 정합**: 이 작업은 standalone `october25kim/fedcore` clone에서 하므로, dispatcher 커밋도 거기로 → 자연스럽게 standalone이 단일 home으로 굳습니다(거버넌스 결정과 일치).
