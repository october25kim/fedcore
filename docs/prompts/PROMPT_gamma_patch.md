# Claude Code prompt — γ-grid expansion + certified-coverage-maximizing threshold

Paste the fenced block into Claude Code on the 4070. Small patch; preserves
validity. Goal: turn the `α=0.1` null in the Mode-2 regime (d=5, where
`test_risk≈0.08<α` but `cert_risk_ucb≈0.185`) into non-vacuous certified coverage.

```text
CONTEXT (read CLAUDE.md + AGENTS.md first). The real CIFAR ladder gives
CertifiedCoverage@alpha=0.1 = 0 in all runs. Diagnosis: at d=5 the model is safe
(test_risk~0.08 < alpha) but the realized risk sits in the (alpha - rhat)^2
explosion band, so the UCB (~0.185) cannot clear alpha at the current operating
points. The gamma grid {0.5,0.7,1.0} is too aggressive for alpha=0.1 at this
calibration size: gamma=0.5 allows prop_risk up to 0.05, leaving too little
margin. Pushing realized risk well below alpha (smaller gamma) should let the UCB
clear alpha at a lower-but-nonzero coverage. Implement this WITHOUT breaking the
proposal/certification split.

PATCH 1 — widen the buffer grid.
config.py: gammas = (0.2, 0.3, 0.5, 0.7, 1.0).

PATCH 2 — certified-coverage-maximizing selector (VALIDITY-PRESERVING).
The invalid shortcut is to certify all gammas on the CERT fold and report the max
(that selects on the cert fold -> union/selection bias). Instead choose gamma on
the PROPOSAL fold, then certify ONE selector at full delta:

  In certify.py add a function certify_best_gamma(prop, cert, test, score, gammas,
  alpha, delta, n_clients, Lambda, lam=None, box=None):
    1. For each gamma in gammas: t_gamma = choose_threshold(prop.score, prop.pred,
       prop.y_open, gamma, alpha)  # existing risk-buffered selector on PROPOSAL.
    2. PROXY on the PROPOSAL fold only: compute per-client (A_j^p, K_j^p) on the
       proposal fold under t_gamma, and U_proxy(gamma) =
       conditional_risk_certificate(A^p,K^p,n^p,delta,Lambda,lam,box).U, plus
       proposal coverage cov_p(gamma).
    3. Pick gamma* = argmax_{gamma : U_proxy(gamma) <= alpha} cov_p(gamma)
       (i.e., the MOST aggressive buffer whose PROPOSAL-side certificate already
       clears alpha). If none clears alpha on proposal, pick the smallest gamma
       (most conservative) as a best-effort.
    4. Certify ONLY t_{gamma*} on the CERTIFICATION fold at the FULL delta
       (single selector -> no union penalty). Evaluate test fold as usual.
    5. Emit the standard metric dict plus the chosen gamma_star and U_proxy.
  This is valid: t_{gamma*} is a function of the proposal fold only (independent
  of the certification fold), so the single CP certification at delta holds.

  Keep certify_grid as-is for the per-gamma sweep (useful for plots/ablation),
  but make run_cifar.py and run_selftrain_cifar.py use certify_best_gamma for the
  headline CertifiedCoverage@alpha. Prefer Lambda='box' (tighter) when client data
  fractions are known; fall back to 'simplex'.

PATCH 3 — real-data alpha-frontier.
Add a flag to run_cifar.py to evaluate alpha in {0.10, 0.15, 0.20, 0.25} from the
SAME exported logits (no retraining), reporting certify_best_gamma per alpha. This
produces the real-data CertifiedCoverage@alpha frontier (d=5 most promising).

CPU SANITY FIRST (no GPU): extend run_smoke.py (or a tiny script) to call
certify_best_gamma and assert (i) it runs, (ii) when it reports certified=True the
empirical test_risk <= alpha, (iii) over many synthetic trials with a fixed
selector at true risk = alpha the certified deploys keep unsafe rate <= delta
(validity is not broken by gamma selection on proposal).

THEN RE-RUN (Docker) the clean ladder rungs that matter:
  cifar10 d=5 clean ; cifar10 d=0.5 clean ; cifar100 d=5 clean
and the alpha-frontier on cifar10 d=5 clean.

EXPECTED / WHAT TO REPORT (fixed format 진단/명령/핵심결과/판정/다음행동):
- Hypothesis: at d=5, certify_best_gamma picks a small gamma* (~0.2-0.3) and yields
  CertifiedCoverage@alpha=0.1 > 0 (non-vacuous) at reduced coverage. Report
  gamma*, cert_risk_ucb, cert_coverage_lcb, test_coverage, test_risk per run.
- If it is STILL 0 at d=5 even with gamma*=0.2 and box-Lambda: that means the
  per-client accepted counts are below the Theorem-2 threshold for r~0.03 -> report
  the required vs available accepted counts, and conclude the lever is calibration
  size / fewer clients / stronger backbone (not gamma). Do NOT fake certification.
- Update README with a 'gamma-grid + best-gamma' result block.

RULES: never certify on the test fold; never select the final selector on the
certification fold; report failures; judge by cert_* not AUROC.
```

---

### Note for Sanghoon
- The crux is PATCH 2's validity argument: gamma is chosen on the **proposal** fold
  (via a proposal-side proxy certificate), so the single certification on the
  cert fold at full δ stays valid. Reporting `max over γ` of cert-fold results
  would be invalid — the prompt explicitly forbids it.
- If d=5 flips to certified>0, that is the paper's positive central result; if not,
  the prompt forces an honest, quantified "calibration/backbone is the lever"
  conclusion rather than a fudge.
