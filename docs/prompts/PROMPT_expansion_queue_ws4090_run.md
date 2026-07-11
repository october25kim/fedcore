# Claude Code prompt — run the remaining expansion queue on the ws4090 server (reuse dispatcher)

Operational run prompt. The dispatcher is already extended and smoke-verified; this runs the queue.
Incorporates the reconciled coordination (in-flight Task E, R8 split, R8b gated on all T9-feeders).
Paste the fenced block into Claude Code on the ws4090 server.

```text
READ CLAUDE.md AND AGENTS.md FIRST. This RUNS the remaining expansion queue using the ALREADY-EXTENDED
dispatcher (scripts/ws4090/dispatch.py + scripts/docker_grid.sh, smoke-verified: GPU pinning,
per-run-npz idempotency, no silent retry, status CSV). Do NOT build a new dispatcher and do NOT change
the runners or experiment definitions. DATA ARTIFACTS ONLY (CSV/npz under runs/); NO manuscript edits.
Report per task in the fixed format (진단 요약 / 확인한 명령 / 핵심 결과 / 판정 / 다음 행동); report
failures, no silent retries.

INFRASTRUCTURE (reuse — do not reinvent).
  - Dispatch via scripts/docker_grid.sh -> scripts/ws4090/dispatch.py, env NUM_GPUS / JOBS_PER_GPU
    (per-model: resnet18=3, wrn=2, foogd_full=1) / RESUME / DRY_RUN. Idempotency = per-run logits-npz
    existence (runners write per-run CSV+npz via save_csv; they do NOT append to a shared CSV — no
    row-matching, no atomic-append needed here).
  - GRID SOURCE: R1/R2/R5 manifests already exist (gen_manifest.py). For R3/R6/R7, DRAFT manifests from
    the grids below, show the DRY_RUN cell plan for approval, THEN run. Do NOT invent cells beyond these.
  - Hardware: 4x TITAN RTX (24 GB, Turing) — same VRAM as 4090, packing budget holds, no code change.
    torch 2.6.0+cu118 / tv 0.21.0; golden PASS.
  - COMMIT: this working dir is NOT a git repo. Copy the two changed scripts (docker_grid.sh,
    ws4090/dispatch.py) to the canonical october25kim/fedcore clone and commit them THERE — do NOT
    git init here. Non-blocking for the run; back them up regardless.

COORDINATION with the in-flight Task E (FedPD-PROSER seeds 3-4 + self-training 5-seed).
  - Do NOT rerun Task E.
  - R8a (protocol reconciliation) runs NOW (CPU, read-only), parallel to Task E.
  - The GPU queue (R1,R2,R3,R6,R7) starts only AFTER Task E's GPU jobs finish.
  - R8b (simultaneous-split full-T9 regeneration) runs LAST — after ALL T9-appending tasks finish:
    Task E AND R2 AND R3 (these three append to runs/T9_diagnostics.csv). Regenerating it earlier would
    miss R2/R3 rows and not be "full". -> runs/T9_diagnostics_simul.csv (delta_r = delta_c = delta/2).

SEED POLICY: floor 10 per new cell ({0..9}) for R1,R2,R5,R7; detector-retraining exception floor 5 for
R3,R6 (and Task E) — extend toward 10 if budget allows, flag every cell below 10. Per-seed rows always,
never only aggregates; never mix seed counts in one cell without flagging.

=== TASKS (priority R8 > R1 > R2 > R3 > R6 > R7 > R4 > R5; R8a/R4 are CPU — run now, parallel to Task E) ===

R8 (P0, CPU). Detector diagnostics reconciliation. T9_diagnostics.csv disagrees with published detector
  cells: FedPD d=5 a=0.20 mean cert_coverage_lcb 0.4912 vs T8 0.483; FOOGD d=5 0.3498 vs T8 0.071 (5x).
  R8a NOW: identify the protocol difference (gamma grid? fold definition? representative-head vs native
  score? which coverage quantity T8 reports?) and deliver ONE consistent protocol for the detector block
  — either regenerate T9 detector rows under exactly the T8 protocol, or document that T8's number is a
  different quantity and state which is correct for the paper. (Report R8a's finding before R8b.)
  R8b LAST (after Task E + R2 + R3): regenerate the full T9 at delta_r=delta_c=delta/2 ->
  runs/T9_diagnostics_simul.csv (same schema).

R1 (P0, GPU). Client scaling on CIFAR-10. J in {10,20} (J=5 exists), resnet18gn, d=0.5, clean, seeds
  {0..9} (stage {0..4} both J, then {5..9} J=20 then J=10 if budget forces):
    run_cifar.py --dataset cifar10 --n_known 6 --n_clients J --dirichlet_alpha 0.5 --rounds 50
      --local_epochs 2 --alpha 0.10 --delta 0.10 --seed S   (export logits npz)
  Grouped certificates at G in {J, J/2, 5, 2} (contiguous pre-declared), alpha in {0.10,0.20} ->
  runs/client_scaling.csv (J,seed,alpha,G,cert_risk_ucb,cert_coverage_lcb,cert_n_min_group,certified,
  test_risk,test_coverage). Acceptance: per-client (G=J) bounds degrade with J at fixed total data;
  grouping restores certification; report where CertCov@0.20 survives. One partial-participation variant
  (0.5 at J=20, seed 0) ONLY if the trainer natively supports client sampling; else say so and skip
  (do not hack the trainer).

R2 (P0, GPU). CIFAR-100 multi-model, 10 seeds. Backbones {resnet18gn, resnet18bn, simplecnn}, n_known 60,
  d in {0.5,5}, clean, seeds {0..9} (60 runs; order resnet18gn -> resnet18bn -> simplecnn; report any
  incomplete backbone cell as INCOMPLETE, never at fewer seeds):
    run_cifar.py --dataset cifar100 --n_known 60 --n_clients 5 --dirichlet_alpha D --rounds 50
      --local_epochs 2 --alpha 0.10 --delta 0.10 --seed S
  Grouped G in {2, J} at alpha in {0.10,0.20}; append per-seed rows to runs/T9_diagnostics.csv and write
  runs/cifar100_multimodel.csv. Stretch after grid: one FedPD-PROSER CIFAR-100 cell (d=5, 3 seeds,
  detector exception).

R3 (P1, GPU; detector exception floor 5). FOOGD-SM3D seed extension: seeds {3,4} at d=5, alpha=0.20
  (native score, same protocol as before); extend toward {5..9} if budget remains -> extend
  runs/T8_fedosr_bases.csv and runs/T9_diagnostics.csv. Flag the final seed count.

R4 (P1, CPU, anytime). Deployment-knob sensitivity on stored GN d in {5,0.5} logits (all stored seeds),
  alpha in {0.10,0.20}: (1) rho sweep {0.05,0.10,0.15,0.25,simplex} -> runs/rho_sensitivity.csv
  (rho,seed,d,alpha,cert_risk_ucb,cert_coverage_lcb,certified); (2) gamma ablation fixed gamma in
  {0.5,0.7,1.0} (no selection) -> runs/gamma_ablation.csv (gamma,seed,d,alpha,cert_n,cert_risk_ucb,
  cert_coverage_lcb,certified,test_risk).

R5 (P2, GPU). Corruption curve, seeded. Symmetric rates {0.1,0.2,0.35}, d in {0.5,5}, resnet18gn, seeds
  {0..9} -> runs/corruption_curve_seeded.csv (noise_type,rate,d,seed,CertCov@0.10,CertCov@0.20,test_risk).

R6 (P1, GPU; detector exception floor 5 where FedPD retraining is involved). One client-simplex (small-J)
  deep positive. Attempt in order, stop at the first non-vacuous cell: (1) CIFAR-10 / FedPD-PROSER / J=5 /
  full simplex (Theorem 1) / alpha=0.20, enlarged audit budget (cert_frac 0.5 or the 4x protocol);
  (2) same with J=3 (retrain FedPD at J=3, seeds {0,1,2}). -> runs/simplex_positive.csv (J,seed,alpha,
  cert_risk_ucb,cert_coverage_lcb,cert_n_min_client,certified). A clean failure is reportable as the
  measured price of client-simplex robustness.

R7 (P1, GPU/CPU). covtype stable second positive. A2-compliant only: proposal-fold score selection (not
  fixed MSP), enlarged audit budget, optionally a wider MLP. alpha in {0.20,0.25,0.30}, delta=0.10,
  grouped G=2, seeds {0..9} -> runs/covtype_stable.csv (T9 schema). Goal >=8/10 non-vacuous; else report
  the best honest cell.

GUARDRAILS: proposal/certification/test split hygiene inviolable; corruption on TRAIN labels only,
calibration folds clean; contiguous pre-declared grouping stated in every CSV header; never commit
runs/data/weights (DO persist new scripts to the canonical repo); no silent retries; failed commands
reported as failed. Do NOT promote Thm 3 / Remark 1 over Thm 1/2; judge by cert_* not accuracy/AUROC.

STOP-AND-ASK: show the R3/R6/R7 DRY_RUN cell plans for approval before running them; report R8a's
protocol finding before R8b. Otherwise run the queue in priority order and report each task in the fixed
format (canonical metrics: cert_coverage_lcb mean±std, n_certified/n_seeds, cert_risk_ucb, test_risk).
```

---

### Notes for Sanghoon
- **인프라 재사용 프롬프트**입니다 — 새 스크립트 안 만들고 이미 확장·스모크 통과한 `dispatch.py`+`docker_grid.sh`로 큐만 돌립니다. idempotency는 per-run npz(공유 CSV append 아님).
- **R8 분할 + R8b 게이트**: R8a(프로토콜 규명)는 지금 CPU로, R8b(δ/2 full-T9)는 **Task E + R2 + R3 후 맨 끝 1회**(셋 다 T9에 append하므로).
- **R3 = FOOGD만**(FedPD+self-training은 in-flight Task E). GPU 큐는 Task E GPU 후 시작.
- **git**: 이 dir은 git 아님 — 두 스크립트는 canonical `october25kim/fedcore` clone에 별도 커밋(여기선 백업만).
- **TITAN RTX**(4090 아님, 24GB 동일)라 패킹 예산 유효.
- STOP-AND-ASK: R3/R6/R7 DRY_RUN 셀 플랜 승인 + R8a 진단 먼저. R8a(P0 detector 프로토콜) 결과 오면 Table 5(b)·headline 방향 같이 확정하겠습니다.
