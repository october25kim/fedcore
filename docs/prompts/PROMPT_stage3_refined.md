# Claude Code prompt — STAGE 3 (refined, time-boxed ~2h): stabilize headline + reduced ladder

Paste the fenced block into Claude Code on the 4070. GATE PASSED but BORDERLINE
(worst-group G=2 α=0.1: CertCov 10.1% ± 8.5%, 2/3 seeds; seed2 missed). So do NOT
run the full 27-run ResNet ladder — most α=0.1 cells are predictably 0. Instead
(3a) tighten the headline with more seeds, (3b) a reduced ladder for T1, (3c) one
collapse cell. Time-box ~2h; stop-and-ask before exceeding.

```text
CONTEXT (read CLAUDE.md + AGENTS.md first). GATE=PASS but borderline: d=5 ResNet
(CIFAR stem + GroupNorm), cert_frac=0.5, worst-group G=2 certifies α=0.1 in 2/3
seeds (CertCov 10.1% ± 8.5%; seed2 ucb 0.226). covtype non-vacuous from α=0.20.
Headline will LEAD with the feasibility law + α=0.20 robust positive (CIFAR+covtype);
the α=0.1 worst-group result is a favorable-regime, seed-variable secondary — report
it with full seed variance, never overclaimed. Docker-first; no faked crossings;
proposal/cert/test disjoint; judge by cert_*/matched-risk, not AUROC. TIME-BOX ~2h.

STAGE 3a — STABILIZE the headline (highest value; do first).
Run cifar10 d=5 ResNet (CIFAR stem, GroupNorm), cert_frac=0.5, seeds {3,4} with the
SAME config as the seed0-2 crossover; export npz. Recompute the grouped staircase
(G in {2,3}) at α=0.10 over seeds {0,1,2,3,4}. Report:
  - per-seed G2/G3 cert_ucb and CertCov@0.1,
  - 5-seed mean±std of CertCov@0.1 and cert_ucb for G2 and G3,
  - n_pass/5 (seeds with (G2 or G3) ucb<=0.10 & cov>0).
This converts "2/3" into a credible 5-seed estimate with a tighter CI.

STAGE 3b — REDUCED main ladder for T1 (the law table).
ResNet primary: d in {5, 0.5} × {clean, symmetric-0.35} × seeds {0,1,2}.
certify_best_gamma + box-Λ; export npz; aggregate to runs/agg_main.csv. Record per
cell: cert_risk_ucb, CertCov@α for α in {0.10, 0.20}, test_risk, test_coverage
(mean±std). This shows the law: cert_ucb rises as d→0.1 and as corruption increases.

STAGE 3c — ONE collapse cell (cheap, for completeness).
One ResNet run at cifar10 d=0.1 clean to confirm Mode-1 collapse (test_risk>α). For
the remaining grid cells (asym, other d) DO NOT run full ResNet — reuse the existing
SimpleCNN ladder rows in T1, clearly labeled by backbone. Note asym ResNet as future.

ASSEMBLE.
Update T1 (ResNet + SimpleCNN rows, mean±std), regenerate F6 (staircase, 5-seed
band for d=5) and F7 (collapse vs d). Update README; after draft sync, update
Fed-CORE_draft.md §5.1: (i) 5-seed α=0.1 worst-group result with variance and the
cert_frac=0.5 in the claim; (ii) α=0.20 robust positive on CIFAR AND covtype as the
co-headline; (iii) the feasibility law as the framing.

REPORT (fixed format) after 3a (the gate-tightening), then after 3b+3c. After 3a,
state whether the 5-seed estimate strengthens or weakens the α=0.1 headline. STOP
and ask before exceeding ~2h GPU or before any extra runs beyond the above.

RULES: never use test labels in proposal/cert; grouping public/fixed; report all
failures and the exact cert_frac; G=1 pooled stays a near-IID-only bonus; do not run
the full 27-run grid.
```

---

### Notes for Sanghoon
- **Why not (A) full ladder:** at α=0.1 the crossover is seed-fragile even in the
  *best* regime (d=5), so the harder cells (d≤0.5, corruption) are predictably 0 —
  the 27-run grid would spend hours re-confirming zeros. The law table (T1) only
  needs the reduced grid + reused SimpleCNN rows.
- **3a is the real value:** 2 extra d=5 seeds turn a fragile "2/3" into a 5-seed
  estimate — that single thing most strengthens the paper per GPU-hour.
- If GPU is tighter than ~2h, drop 3b's d=0.5 and keep 3a + d=5 clean/sym only;
  if you'd rather stop now (option C), the paper already stands on the law + α=0.20
  (CIFAR+covtype) + necessity + T4 + Prop4 + Lemma L.
