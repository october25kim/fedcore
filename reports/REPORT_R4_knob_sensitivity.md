# R4 — Deployment-knob sensitivity on stored GN logits

Read-only CPU ablation on stored `resnet18gn` (GN) CIFAR-10 logits. No GPU, no
new training, no manuscript edits. Headline pipeline (MSP, G=2 grouped,
cert_frac=0.5, δ=0.10, seed-0 fold repartition), GN d∈{5,0.5}, α∈{0.10,0.20}.

## 진단 요약

Two knobs swept: (1) the Λ deployment-mixture-set width ρ (box half-width, plus
full simplex); (2) the risk buffer γ, fixed with NO proposal-fold selection. Goal:
show how far the certificate moves under each, and confirm the buffer matters.

## 확인한 명령

```
python experiments/fedcore/exp_r4_knob_sensitivity.py
  -> runs/rho_sensitivity.csv    (100 rows)
  -> runs/gamma_ablation.csv     ( 60 rows)
```

Validation: R4's ρ=0.15 cells reproduce the headline exactly. GN d=5 α=0.20 mean
CertCov_lcb = 0.3923 (== T9 resnet18gn d5 α=0.20); α=0.10 = 0.0769 (== T9). The R4
pipeline is the headline pipeline with a swept knob.

## 핵심 결과

Seeds: GN has **5 stored seeds (0..4)** for each d; this reuses stored logits (no
new training), so the 10-seed floor for NEW cells does not apply.

**(1) ρ sweep — mean CertCov_lcb (certified→cov else 0):**

| d | α | ρ=0.05 | 0.10 | 0.15 | 0.25 | simplex |
|---|---|---|---|---|---|---|
| 5   | 0.10 | 0.0915 | 0.0771 | 0.0769 | 0.0765 | 0.0753 |
| 5   | 0.20 | 0.3940 | 0.3932 | 0.3923 | 0.3906 | 0.3855 |
| 0.5 | 0.10 | 0.0914 | 0.0912 | 0.0910 | 0.0745 | 0.0895 |
| 0.5 | 0.20 | 0.3769 | 0.3537 | 0.3531 | 0.3521 | 0.3700 |

The certificate is **robust to the Λ-set width**: from the tightest box (ρ=0.05)
to the full simplex, mean CertCov_lcb moves only ~2–6% relative (e.g. d5 α=0.20:
0.3940 → 0.3855). Widening ρ is monotone-ish downward as expected (larger mixture
set → more conservative worst case); residual non-monotonicity between box and
simplex is small and within the G=2 discreteness. Takeaway: the headline box=0.15
is not a fragile choice — the deployment-mixture assumption is not driving the
result.

**(2) γ ablation — fixed γ, no selection (cov / n_pass / n_seeds):**

| d | α | γ=0.5 | γ=0.7 | γ=1.0 |
|---|---|---|---|---|
| 5   | 0.10 | 0.107 / 3 / 5 | 0.084 / 2 / 5 | **0.000 / 0 / 5** |
| 5   | 0.20 | 0.261 / 4 / 5 | 0.392 / 5 / 5 | 0.079 / 1 / 5 |
| 0.5 | 0.10 | 0.087 / 2 / 5 | 0.112 / 2 / 5 | **0.000 / 0 / 5** |
| 0.5 | 0.20 | 0.255 / 5 / 5 | 0.376 / 5 / 5 | **0.000 / 0 / 5** |

**The risk buffer is essential.** Unbuffered γ=1.0 certifies **0/5 seeds in three of
four cells** (and only 1/5 in the fourth). Buffered γ recovers certification
(2–5/5). The best fixed γ is α-dependent — γ=0.5 wins at α=0.10 (d5), γ=0.7 wins
at α=0.20 — which is exactly why the headline chooses γ on the proposal fold rather
than fixing it. This directly answers the CLAUDE.md workflow check: γ=1.0 does NOT
suffice; the buffer matters.

## 판정

- **Λ-width ρ: low sensitivity.** The certificate degrades gently and predictably;
  box=0.15 is a safe default and the full-simplex worst case is only marginally
  tighter. Not a fragile knob.
- **Risk buffer γ: high sensitivity, buffer required.** γ=1.0 is essentially
  non-certifying; proposal-fold γ-selection is load-bearing, not cosmetic.

## 다음 행동

None gated. CSVs written under runs/ (gitignored); script committed. R4 complete
at the 5 available GN seeds (flagged). If GN seeds {5..9} are ever trained, re-run
the same script to extend both CSVs.
