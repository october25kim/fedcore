# Low-label self-training (STEP 0-3) — report

Central question: in a label-scarce regime, (a) does the oracle clean-pseudo-label headroom grow
large, and (b) does certified self-training capture a large fraction SAFELY (contam <= alpha) with
a gain statistically separable from zero at n=3? Base = FedPD-PROSER (CE pretrain -> PROSER), one-shot
certification, beta=1.0, alpha=0.20, d=5, CIFAR-10, A5 condition enforced (U unknown rate matched to
0.30). Artifacts: `runs/selftrain_lowlabel.csv` (+ `_agg.csv` via `aggregate_selftrain.py`, convergence
guard known_acc<0.30 so a legit weak low-label base is kept and only chance/smoke runs are excluded).

## Results

| step | labeled_frac | cert pool | mode / audit | Δ vs none | contam | admitted | n |
|---|---|---|---|---|---|---|---|
| 0 | 0.10 | 0.30 | oracle | **+0.155** | 0 | 10118 | 1 |
| 0 | 0.10 | 0.30 | certified 4x | +0.000 (adm 0) | – | 0 | 1 |
| 2 | 0.10 | 0.60 | oracle | +0.147 | 0 | 12042 | 1 |
| 2 | 0.10 | 0.60 | certified 4x | +0.129 | 0.065 | 830 | 1 |
| 2 | 0.10 | 0.60 | certified 8x | +0.115 | 0.065 | 830 | 1 |
| 3 | 0.10 | 0.60 | **oracle (n=3)** | **+0.161 +/- 0.052** | 0 | – | 3 |
| 3 | 0.10 | 0.60 | **certified 4x (n=3)** | **+0.043 +/- 0.075** | 0.065 | 830/0/0 | 3 |

Half-label reference (from REPORT_selftrain_pkg.md): oracle +0.045, certified +0.030 +/- 0.022 (n=3).

## Per-step (fixed format)

### STEP 0 — oracle ceiling probe
- 핵심: lf=0.10에서 oracle headroom = **+0.155** (half-label +0.045의 3.4배). certified=0 인정.
- 판정: **strong (headroom hypothesis holds)** — label이 희소할수록 clean pseudo-label 잠재이득 폭증.

### STEP 2 — audit-budget rescue (single seed)
- 핵심: cert pool을 0.6로 키운 config에서 certified가 **+0.129 (contam 0.065 <= alpha, 88% capture)**.
  audit 8x는 4x 대비 추가 이득 없음(둘 다 830 인정; 4x에서 이미 certify). admission을 푼 것은 8x가
  아니라 partition(cert pool 0.3->0.6) — feasibility는 config-sensitive.
- 판정: **strong (single seed), but confounded** — multi-seed 필요.

### STEP 3 — multi-seed confirmation (DECISIVE)
- 핵심: paired Δ_s, n=3 (lf=0.10, audit 4x):
  - **certified Δ = +0.043 +/- 0.075 (sample SD, ddof=1); 95% t-CI [-0.142, +0.228] → INCLUDES ZERO.**
  - oracle Δ = +0.161 +/- 0.052 (large, consistent, clearly > 0).
  - **1/3 seeds admits** (seed0: +0.129; seeds 1,2: adm=0 → Δ=0). 0/N contamination violations.
  - high base variance (none = 0.448/0.639/0.475); notably seed1 has the STRONGEST base (0.639) yet
    certified admitted nothing -> admission is feasibility-noisy, not monotone in base strength here.
- 판정: **WARNING / NOT a claimable gain.** Per the pre-registered rule (claim only if t-CI excludes 0),
  the label-scarce certified gain is **not statistically separable from zero at n=3**.

## Honest conclusion (the publishable framing)
This is the package's anticipated **feasibility-limited** outcome:
> "In the label-scarce regime the clean-pseudo-label HEADROOM is real and large (oracle +0.161 +/-
>  0.052, separable from zero), but the trusted AUDIT BUDGET / per-seed feasibility — NOT the
>  detector — is the binding constraint (Theorem 2): certified admits on only 1/3 seeds, so the
>  realized gain is +0.043 +/- 0.075 (t-CI includes 0)."

The certified GATE remains the contribution (always safe: 0/N contamination violations across every
step). The accuracy gain is a SUPPORTING result that is robustly positive only in the half-label cell
(certified +0.030 +/- 0.022, n=3, both > 0 in 2/3 seeds there); in the deep label-scarce cell it is
feasibility-limited and not separable from zero.

## Figure / Section 5.6 deltas
- **NO new figure** (t-CI includes zero) — keep the current supporting Figure 9 (half-label gain).
- §5.6: add one honest sentence — "In a deeper label-scarce regime (10% labels) the clean headroom
  grows to +0.16, but certified self-training becomes feasibility-limited (admits on 1/3 seeds; gain
  +0.04 +/- 0.07, 95% CI includes 0); the safe contamination gate (0/N violations) is unaffected."
- Untested lever: audit 8x was only run on the seed that ALREADY admitted (seed0); whether 8x rescues
  the zero-admission seeds (1,2) is not yet established (STEP-2 rescue on the failing seeds).
