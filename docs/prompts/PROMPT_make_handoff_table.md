# Claude Code prompt — generate the GATE + covtype α-frontier handoff table

Paste the fenced block into Claude Code on the 4070. It builds a small reporting
script that prints the GATE decision and the covtype α-frontier in the EXACT
paste-back format, and also writes CSVs (for sync). No new training — it reads the
already-exported logits/results.

```text
GOAL (read CLAUDE.md + AGENTS.md first). Build experiments/fedcore/make_handoff.py
that produces a copy-paste handoff summary for the seed GATE and the covtype
alpha-frontier, in the EXACT format below, and also writes runs/handoff_*.csv.
No retraining: read exported npz / cached results. Do not fake; if a seed or file
is missing, print "pending" for it and compute over what exists.

INPUTS (use what exists; print the paths you actually read):
- cifar10 d=5 ResNet logits per seed: runs/cifar10_d5_resnet18_seed{0,1,2}_logits.npz
- covtype logits/counts: runs/covtype_*.npz (or runs/tabular.csv from run_tabular.py)
- the crossover cert_frac that produced the seed0 positive (expose --cert_frac,
  default to that value; print it).

WHAT make_handoff.py MUST COMPUTE:
1) GATE (cifar10 d=5 ResNet, at --cert_frac, alpha=0.10, box-Lambda,
   certify_best_gamma + proxy margin), for each available seed in {0,1,2} and for
   G in {2,3}:
     - cert_risk_ucb, CertifiedCoverage@0.10,
     - r_hat = accepted-set test_risk at the certified-coverage-maximizing selector,
     - per-group accepted count (cert_n / G) at the crossover.
   Then aggregate across available seeds: mean +/- std of cert_ucb and CertCov@0.1
   for G=2 and G=3; and count
     n_pass = #seeds where (G2 OR G3) has cert_ucb<=0.10 AND CertCov>0.
   GATE rule: PASS iff n_pass >= 2 of 3 (>= ceil(2/3 of available if <3 ran);
   if <3 seeds available, print "GATE: PROVISIONAL (k/N seeds)").
2) covtype alpha-frontier (post-hoc, no GPU): for alpha in
   {0.10,0.15,0.20,0.25,0.30}, the best valid certified coverage and its cert_ucb
   (grouped/box as used elsewhere); covtype r_hat; the smallest alpha that is
   non-vacuous (CertCov>0).

OUTPUT — print EXACTLY this block (fill the blanks; keep the labels/spacing):

GATE  (cifar10 d=5, ResNet, cert_frac=____)
seed0: G2 ucb=__ cov=__ | G3 ucb=__ cov=__ | r_hat=__
seed1: G2 ucb=__ cov=__ | G3 ucb=__ cov=__ | r_hat=__
seed2: G2 ucb=__ cov=__ | G3 ucb=__ cov=__ | r_hat=__
agg : G2 ucb=__±__ cov=__±__ ; G3 ucb=__±__ cov=__±__
#seeds with (G2 or G3) ucb<=0.10 & cov>0 :  __/3
per-group cert_n at crossover ~ __
GATE: PASS / FAIL / PROVISIONAL

covtype alpha-frontier (post-hoc, no GPU)
a=0.10 cov=__ ucb=__ | 0.15 cov=__ ucb=__ | 0.20 cov=__ ucb=__ | 0.25 cov=__ ucb=__ | 0.30 cov=__ ucb=__
covtype r_hat ~ __ ; first non-vacuous alpha = __

GPU budget remaining ~ ____ hrs   (fill manually)

ALSO WRITE (for the sync option):
- runs/handoff_gate.csv      (seed, G, cert_ucb, CertCov@0.1, r_hat, per_group_n)
- runs/handoff_covtype.csv   (alpha, CertCov, cert_ucb, r_hat)

RULES: reuse existing certify/feasibility-lever code (no reimplementation drift);
proposal/cert/test disjoint; never use test labels in proposal/cert; print the
exact files read and any missing ones; round to 3 decimals; do not fabricate any
value. Run it and paste the printed block back.
```

---

### Note for Sanghoon
- Run `python experiments/fedcore/make_handoff.py --cert_frac <crossover value>`,
  then paste the printed block to me — that is the fastest handoff.
- Or, if you `git push` the two `runs/handoff_*.csv` and pull into the Mac FedCORE
  folder, tell me "synced" and I'll read them directly.
