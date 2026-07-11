# R8 — Detector diagnostics reconciliation (T9 vs T8)

Read-only CPU analysis. Reconciles `runs/T9_diagnostics.csv` with the published
detector block `runs/T8_fedosr_bases.csv`. No GPU, no manuscript edits.

## 진단 요약

`T9_diagnostics.csv` disagreed with the published detector cells:

| cell (d=5, α=0.20) | T9 mean `cert_coverage_lcb` | T8 mean `cert_coverage_lcb` | ratio |
|---|---|---|---|
| FOOGD | 0.3498 | 0.0706 | **~5×** |
| FedPD | 0.4912 | 0.4828 | ~1.02× |

The disagreement is a **score-head conflation**, not a bug in either certificate.
`T9` certifies **every** backbone (including the detectors) on a **fixed MSP** score
computed from the classifier logits. `T8` certifies each detector on its **native
open-set score** (`-sm` for FOOGD-SM3D / FedPD-PROSER). All other protocol knobs
(G=2 grouped, `cert_frac=0.5`, `test_frac=0.2`, `box=0.15`, δ=0.10, seed-0 fold
repartition, worst-group coverage LCB) are identical between the two tables.

## 확인한 명령

```
python experiments/fedcore/exp_r8_detector_reconcile.py      # -> runs/T9_detector_reconciliation.csv
python experiments/fedcore/exp_alpha20_diagnostics.py --out <scratch>   # patched-generator dry run
```

The reconcile script re-certifies each detector npz under all four
{score head} × {γ-grid} combinations and compares to both recorded tables.

## 핵심 결과

Two candidate axes, isolated:

- **(A) Score head — the whole 5× gap.** Holding the γ-grid fixed, MSP vs native
  at d=5 α=0.20 gives FOOGD 0.3498 vs 0.0706 and FedPD 0.4912 vs 0.4828. T9's
  `foogd` rows are **byte-identical** to T8's controlled `FedAvg+MSP` baseline
  (same FOOGD npz, same MSP score) — i.e. T9's "foogd" cell is *MSP on the FOOGD
  backbone*, **not** the FOOGD-SM3D detector. FedPD's gap is tiny only because
  PROSER's `-sm` ≈ MSP on its logits.
- **(B) γ-grid — negligible.** T8 uses `{0.5,0.7,1.0}`; T9/`main.py` use
  `{0.2,0.3,0.5,0.7,1.0}`. At **α=0.20 it changes nothing** (γ*=0.7 in both). It
  only diverges for a few seeds at α=0.10 where the wider grid selects γ≤0.3.
- **G=2 is the same coverage quantity** in both tables — not a source of the gap.

Reconcile script assertions (both pass):
`msp_wide` reproduces recorded T9 exactly; `native_narrow` reproduces recorded T8 exactly.

## 판정

**T8's number is the correct DETECTOR quantity.** The detector block exists to
certify each detector's *native* open-set score (isolating the score head, per
`t8.py`). T9's MSP-scored detector rows measured a *different* quantity — an
MSP-on-detector-backbone baseline — and must not be presented as the detector's
certificate. The 5× FOOGD figure was an artifact of that swap.

**One consistent protocol for the detector block** (= T8, now also enforced in the
T9 generator):
- Score: **native open-set score** (`-sm`) for FedPD/FOOGD; MSP for MSP-backbones
  (ResNet cells keep MSP — MSP *is* their score head).
- γ-grid: **{0.5,0.7,1.0}** for detector cells (T8 / documented risk-buffer set);
  ResNet cells keep the `main.py` grid, matching their headline.
- G=2 grouped, `cert_frac=0.5`, `box=0.15`, δ=0.10, seed-0 repartition — unchanged.

`exp_alpha20_diagnostics.py` was patched to score detector cells with `-sm` +
`{0.5,0.7,1.0}`. Verified against T8: **0/24 detector rows mismatch**; the 30
ResNet rows are unchanged (0 diffs).

Secondary flag (out of R8 scope, noted for the record): `main.py`/`covtype.py`/T9
use the γ-grid `{0.2,0.3,0.5,0.7,1.0}`, while `CLAUDE.md`/`AGENTS.md` and `t8.py`
document the buffer set as `{0.5,0.7,1.0}`. The docs and the headline code disagree
on the buffer grid; worth a one-line reconciliation decision.

## 다음 행동 (gated on Task E)

The full T9 regeneration **waits until the in-flight Task E** (FedPD-PROSER seeds
3–4) has appended its detector-seed rows, per the coordination rule, then regenerate
**once**:

```
# corrected main table (native detectors, δ=0.10)
python experiments/fedcore/exp_alpha20_diagnostics.py --out runs/T9_diagnostics.csv
# simultaneous risk+coverage split, δ_r = δ_c = δ/2  (Corollary 1)
python experiments/fedcore/exp_alpha20_diagnostics.py --delta 0.05 --out runs/T9_diagnostics_simul.csv
```

The `--delta 0.05` run realizes the simultaneous split exactly: `certify_best_gamma`
applies its `delta` to **both** the conditional risk certificate and the coverage
LCB, so δ=0.05 gives δ_r=δ_c=0.05 jointly ≤ δ=0.10 (same mechanism as the validated
`exp_delta_split.py`).
