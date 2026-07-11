# PROMPT — ws4090 RESUME: finish Task 3 + close out campaign (2026-07-10)

> The previous session completed Tasks 1, 2, 5 and was interrupted during
> Task 3 (asymmetric corruption sweep) around wave 1/8 of 60 docker jobs.
> This brief resumes from the exact on-disk state. Read `CLAUDE.md` /
> `AGENTS.md` first and keep all prior rules (Docker-first, append-only,
> fixed report format, never hide failed commands). Task 4 remains **HOLD** —
> do not start it under any circumstances.

## Step 0 — State assessment (do this before launching anything)

1. `nvidia-smi` and `docker ps` — identify any still-running or orphaned
   containers **belonging to this campaign only** (match by image/name/mount;
   other users' containers are running on this host — do NOT touch them).
   Kill only confirmed orphans of the asym sweep.
2. Reconcile `manifest_R5_asym.txt` (60 jobs) against on-disk artifacts:
   for each job, classify DONE / PARTIAL / MISSING by the presence of its
   expected outputs (per-run csv + `*_logits.npz` per the r5asym naming).
3. **Integrity-check every existing asym npz** (`np.load` + assert the 12-key
   schema and nonzero fold sizes). An interrupted write can leave a truncated
   npz: any file that fails to load or has a wrong schema → move to
   `runs/_quarantine/` and reclassify its job as MISSING.
4. Report the tally (DONE/PARTIAL/MISSING counts) before relaunching.

## Step 1 — Resume the sweep (idempotent)

- Relaunch **only** PARTIAL/MISSING jobs, 2 jobs/GPU across the 4 TITAN RTX,
  same docker image and driver as the previous session (`gen_manifest.py`
  `r5asym_jobs()` → filtered manifest → existing driver). If the driver lacks
  a resume filter, write the filtered manifest to `manifest_R5_asym_resume.txt`
  and launch from that; do not modify completed jobs' outputs.
- Re-verify one relaunched job's outputs (schema + certify wiring) before
  scaling to all waves — the previous smoke passed, but the interruption is a
  new failure mode.

## Step 2 — Certify + append (only after all 60 jobs are DONE)

1. `md5sum runs/corruption_curve_seeded.csv` **before** appending; also save
   a copy of the symmetric-only subset (first 60 rows) for comparison.
2. `certify_grid.py --task r5asym` (append-only path with idempotent dedup).
3. Verify and put in the report:
   - the 60 symmetric rows are byte-identical (md5 of the symmetric subset
     unchanged);
   - exactly 60 asymmetric rows appended (no dupes after dedup);
   - per-cell aggregate: mean CertCov@0.20 (and @0.10) by rate × d.
     Expected pattern — marginal at rate 0.1, ≈0 at rates ≥ 0.2, mirroring
     symmetric; report honestly if it deviates.

## Step 3 — Close out the campaign

1. `runs/SYNC_MANIFEST_ws4090_2026-07-10.txt` with md5sums, covering ALL
   campaign artifacts (Tasks 1, 2, 5 included):
   - `fcp_recast.csv` (server reproduction), `fcp_recast_resampling.csv`,
     `reports/REPORT_fcp_recast_resampling.md`
   - `delta_sensitivity.csv`
   - appended `corruption_curve_seeded.csv` (+ the asym per-run csv/npz list)
   - `fedpd_cifar10_d5_seed{3,4}.npz`, `fedpd_cifar10_d0.5_seed{3,4}.npz`,
     server-side `T8_fedosr_bases_agg.csv`
   - note: BN-d5 clean npz exist without the `none0.0` tag (naming memo);
     `_fedpd_batch.log` is absent and cannot be regenerated (no retraining)
   - every log of this campaign
2. List, per task, the exact files the laptop should commit and a one-line
   English commit message each (the laptop commits; this tree has no git).
3. Final report in the fixed format (진단 요약 / 확인한 명령 / 핵심 결과 /
   판정 / 다음 행동), one 판정 per task. Include the Step-0 tally, the
   Step-2 verification results, and GPU wall-clock actually used.

## Do NOT

- Start Task 4 (CIFAR-100 frontier / FedOSS) — HOLD until laptop decision.
- Touch other users' containers or the symmetric rows of
  `corruption_curve_seeded.csv`.
- Re-run Tasks 1, 2, 5 — their outputs are accepted and already reflected in
  the manuscript.
