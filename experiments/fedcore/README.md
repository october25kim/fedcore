# Fed-CORE experiment package

Federated Certified Open-Set Recognition. Every experiment here certifies the
**accepted selective risk** `R_sel(lambda) = sum_j lam_j m_j / sum_j lam_j a_j`
of a federated open-set classifier with a finite-sample distribution-free upper
confidence bound, using secure-aggregatable counts. Judge by `cert_*` risk /
coverage, **never by accuracy / AUROC**. Headline metric: **CertifiedCoverage@alpha**.

See `../../CLAUDE.md` and `../../AGENTS.md` for the theory and hard rules.

## Modules (certification core, numpy + scipy only)

| file | role |
|---|---|
| `certificates.py` | CP primitives; **conditional** cert (Thm 1/1', MAIN); mass-ratio (App C); pooled (Prop 3) |
| `clients.py` | synthetic heterogeneous client populations |
| `scores.py` | MSP / neg-entropy / margin / energy (higher = accept) + `scored_views` |
| `selector.py` | open-set error, risk-buffered threshold (`prop_risk <= gamma*alpha`), per-client counts |
| `certify.py` | proposal -> certification -> test glue; canonical metric schema |
| `config.py` | `FedOSRConfig` |
| `fedosr_split.py` | open-set split + Dirichlet non-IID partition + calibration folds |
| `noise.py` | client-side TRAIN-label corruption (calibration/test stay clean) |
| `self_training.py` | **Proposition 4** gate + synthetic accuracy-dynamics demo |
| `selftrain.py` | **Proposition 4** audit-fold split + certified/naive torch loops |
| `models.py`, `fed_train.py` | SimpleCNN + FedAvg + logit export (torch) |

Proof note: `LEMMA_L_proof.md` (Lemma L reduction + transfer argument + exact certificate).

## Runnable experiments

CPU (no torch):

```bash
python run_smoke.py            # fake-logit end-to-end smoke (full schema)
python exp_lemma_L.py          # Lemma L: binomial CP conservative for Poisson-binomial mean
python exp_pooling_fail.py     # non-reducibility: pooled collapses, conditional valid
python exp_necessity.py        # without a certificate you deploy unsafe configs
# STEP 4 (paper sec 5):
python exp_validity.py         # (a) P(R_sel<=Ubar) >= 1-delta vs heterogeneity
python exp_tightness.py        # (b) conditional vs mass-ratio vs box vs pooled
python exp_frontier.py         # (c) CertifiedCoverage@alpha frontier over alpha
python exp_hetero_collapse.py  # (d) certified-coverage collapse vs Theorem 2
python exp_score_agnostic.py   # (e) 4 scores keep validity, change only coverage
python exp_necessity_real.py   # (f) naive deploy unsafe-rate > delta (real npz or synthetic)
python exp_superiority.py      # (g) price of federation + baseline harness
python exp_utilization.py      # (h) automation rate = CertifiedCoverage@alpha
python exp_self_training.py    # (i) Prop 4: certified safe+improves, naive diverges (synthetic dynamics)
python run_selftrain_smoke.py  # (i) Prop 4: delta/T temporal split keeps simultaneous unsafe <= delta
python exp_lemma_L.py          # also runs the EXACT adversarial certificate (see LEMMA_L_proof.md)
```

GPU self-training (torch):

```bash
bash ../../scripts/docker_selftrain.sh         # certified vs naive vs none (smoke-size)
python run_selftrain_cifar.py --dataset cifar10 --T 4 ...
```

GPU (torch, Docker-first):

```bash
bash ../../scripts/docker_cifar.sh            # one rung (env-driven)
bash ../../scripts/run_ladder.sh              # full ladder -> runs/<tag>.csv
python run_cifar.py --dataset cifar10 ...     # direct (inside a torch container)
```

## Acceptance gate (CPU, validated)

- `exp_lemma_L`: worst-case coverage **0.918** (>= 0.90).
- `exp_pooling_fail`: pooled `matched 0.91 -> shift/all 0.00`; conditional simplex `~1.0` every mixture; conditional median U **0.382** < mass-ratio **0.473**.
- `exp_necessity`: boundary unsafe-deploy naive **0.48** / pooled **0.07** / Fed-CORE **0.00**.
- `run_smoke`: simplex 0/12, box 4/12 certified; certified TEST risk < alpha.
- `exp_validity`: coverage **>= 0.98** across heterogeneity. `exp_tightness`: box < simplex < mass-ratio. `exp_hetero_collapse`: certified% -> 0 as `E[A_bad]` crosses the Theorem-2 floor. `exp_self_training`: certified **0.85->0.92**, naive **0.85->0.58**.

## gamma-grid + best-gamma (validity-preserving coverage maximization)

`config.gammas = (0.2, 0.3, 0.5, 0.7, 1.0)`. `certify.certify_best_gamma` chooses
the buffer `gamma` on the PROPOSAL fold (via a proposal-side proxy certificate),
then certifies the single chosen selector ONCE on the certification fold at full
`delta` -- valid because the selector is independent of the cert fold. `run_cifar.py`
uses it for the headline and `--alpha_frontier` sweeps `alpha`.

CPU sanity (`run_smoke.py::best_gamma_sanity`): certified => `test_risk <= alpha`;
validity Monte-Carlo unsafe-among-certified = 0.000 (<= delta).

Real CIFAR (50 rounds, 5 clients, test-set calibration, delta=0.1):

| run | CertCov@0.1 | min cert_ucb | cert_n | note |
|---|---|---|---|---|
| cifar10 d=5 clean   | 0 | 0.222 (g*=0.2) | 151 (~30/client) | **frontier: CertCov@0.20 = 0.160** (msp/box) |
| cifar10 d=0.5 clean | 0 | 0.315 (g*=0.2) | 136 | more non-IID -> looser |
| cifar100 d=5 clean  | 0 | 0.859 (g*=0.2) | 16  | 60 classes -> severe undersampling |

Finding: smaller `gamma` does NOT help here -- it shrinks the accepted count below
the Theorem-2 floor (`ln(J/delta)/(-ln(1-alpha)) ~ 37/client`), loosening the CP.
The lever is calibration size / fewer (or grouped) clients / stronger backbone.

### Feasibility lever (grouped-stratified, post-hoc, `exp_feasibility_lever.py`)

Post-hoc on the SAME d=5 logits (no retraining), `certify.certify_best_gamma_grouped`
applies the conditional certificate over `G` PUBLIC client groups (sec 4.4 grouped-
stratified = privacy compromise). As `G` shrinks (per-group accepted counts rise),
`cert_risk_ucb` falls **monotonically through alpha=0.1** -- the Theorem-2 staircase:

| G | best cert_ucb | CertCov@0.1 | note |
|---|---|---|---|
| 5 (per-client) | 0.295 | 0 | avg <=26/group |
| 3 | 0.185 | 0 | avg <=43/group |
| 2 | 0.153 | 0 | avg <=64/group |
| 1 (pooled) | 0.078 | **0.126** | matched-mixture bonus only* |

*G=1 is the pooled certificate (valid only under near-IID/matched mixture); the
worst-group `G>=2` results are the legitimate ones and get within ~0.05 of alpha=0.1
at this calibration size. Closing it needs more trusted points per group or a lower
empirical risk `rhat` (stronger backbone): the `(alpha-rhat)^2` count requirement
`z^2 rhat(1-rhat)/(alpha-rhat)^2` shrinks fast as `rhat` drops.

Frontier monotonicity: `certify_best_gamma(..., margin=eps)` adds a proposal-proxy
safety margin (`run_cifar.py --proxy_margin`, default 0.01). On d=5 it makes the
alpha-frontier monotone: CertCov@{0.10,0.15,0.20,0.25} = {0, 0, 0.063, 0.193}. (The
earlier non-monotone 0.16@0.20 was a proxy-optimism artifact, now excluded.)

## C3 superiority -- T4 (matched-risk vs SOTA-style scores)

`exp_superiority.py::build_T4` (post-hoc on exported logits) compares, per base
score (MSP / energy / FOOGD-style `score_norm`) and `d in {5, 0.5}`, at matched
risk `alpha`:

* **oracle(test-peek)** -- threshold tuned on TEST labels to hit risk = alpha
  (upper bound, NO guarantee);
* **naive(no-cert)** -- threshold by empirical risk <= alpha on the PROPOSAL fold
  (no certificate); realized TEST risk often EXCEEDS alpha (unsafe);
* **Fed-CORE(G=J)** and **Fed-CORE(G=2 worst-group)** -- `certify_best_gamma`
  (box, proxy margin); certified, realized risk <= alpha, finite-sample guarantee.

Representative (cifar10 d=5, alpha=0.20): oracle msp 0.444 / energy 0.431 (no
guarantee); Fed-CORE G=2 0.286 / 0.290 (certified, risk <= alpha) -> price of
honesty ~0.15 (Fed-CORE ~65% of the cheating oracle WITH a guarantee). naive
exceeds alpha in several cells (e.g. d=0.5 msp@0.20 risk=0.201). Output: `runs/T4.csv`.

## T8 — certification on a real FedOSR base model (FOOGD)

Closes Risk 2 for one base model: Fed-CORE certifies FOOGD's NATIVE open-set score
(SM3D score-norm `||score_model(latents)||`), not MSP/energy. The FOOGD SM3D detector
(`Energy(MLPScore)` + denoising score matching, faithful to FOOGD-main; `foogd_score.py`)
is trained FEDERATED on penultimate features of a shared FedAvg ResNet-GN backbone;
accept-score = `-||score_model(standardize(feat))||`. Identical CIFAR-10 FedOSR split as
the headline (n_known=6, 5 clients, d in {5,0.5}); worst-group G=2, fixed native score,
cert_frac=0.5, seeds {0,1,2}. Pipeline: `run_foogd_cifar.py` (GPU export) ->
`aggregate_T8.py` (CPU certify). Runners: `scripts/docker_foogd.sh`, `scripts/run_foogd_all.sh`.

| base_model (kind) | d | AUROC | CertCov@0.20 (G=2) | r_hat |
|---|---|---|---|---|
| **FOOGD-SM3D** (representative) | 5   | 0.689 | **0.071+/-0.053 (3/3)** | 0.121 |
| FOOGD-SM3D (representative)     | 0.5 | 0.624 | 0.014+/-0.020 (1/3) | 0.030 |
| FedAvg+MSP (full, same backbone)| 5   | 0.728 | 0.350+/-0.077 (3/3) | 0.111 |
| FedAvg+MSP (full, same backbone)| 0.5 | 0.732 | 0.330+/-0.104 (3/3) | 0.113 |

**0/13 false certificates** (every certified cell has empirical test_risk <= alpha).
Fed-CORE certifies a real FedOSR native score (FOOGD SM3D), 3/3 seeds @0.20 on d=5. The
representative FOOGD-sm (AUROC 0.69) certifies SMALLER coverage than MSP (0.73): certified
COVERAGE tracks the score's ability to form a large low-risk accepted set, while VALIDITY
holds for every score. `base_model_kind=representative` because the backbone is FedAvg, not
FOOGD's full SAG training -- NOT passed off as full FOOGD.

**Full-method reproductions.** We also ran the FULL methods with their real training code.
**FedPD-PROSER** (`run_fedpd_cifar.py`: WideResNet-28-10 + manifold-mixup dummy head, lifted
verbatim) **certifies strongly as a full base model** once its real recipe is honoured —
closed-set CE pretrain THEN PROSER dummy fine-tune (PROSER is a fine-tuning method; from
scratch it diverged). Result d=5 (3 seeds): known-acc 0.999, native PROSER dummy-vs-known
score AUROC **0.80**, **CertCov@0.10 = 0.174±0.125 (2/3), CertCov@0.20 = 0.483±0.100 (3/3)**
(test_risk 0.029/0.112 <= alpha) — the strongest row, certifying even at the hard alpha=0.10.
`base_model_kind=full`. (d=0.5 via `scripts/run_fedpd_all.sh`.) **FOOGD-SM3D-SAG**
(`run_foogd_full_cifar.py`: real ODGClient WideResNet + anneal-DSM + KSD/MMD/Langevin SAG)
stays weak on our semantic-shift split at feasible budget (sm-AUROC 0.467: no feature
standardization + jointly-evolving backbone; FOOGD's published strength is covariate-shift
OOD) — there the representative standardized SM3D head is the better vehicle. **Takeaway:**
full reproduction certifies when the method's training recipe is honoured (CertCov tracks
native-score AUROC: FedPD 0.80 > MSP 0.73 > FOOGD-repr 0.69 > FOOGD-SAG 0.47); Fed-CORE
validity held in every case (0/N false certs). No faked rows. See `REPORT_T8_fedosr_bases.md`.

## Backbone push (worst-group alpha=0.1 lever)

`models.py` adds a CIFAR ResNet-18 (`--backbone resnet18`, optional `--pretrained`).
Rationale: the Theorem-2 sample requirement scales as `(alpha - rhat)^-2`, so a
stronger backbone (lower realized `rhat`) shrinks the per-group accepted count
needed to certify. `run_cifar.py --backbone ... --alpha_frontier`, then
`exp_feasibility_lever.py` on the new logits gives the per-backbone staircase
(CertCov@0.1 vs per-group accepted count).

Result (cifar10 d=5, ResNet-18 80 rounds, seed 0): backbone drops realized
`rhat` ~0.05 -> ~0.01-0.04 and min cert_ucb (G=5) 0.185 -> **0.111**. On the
feasibility staircase (pooled trusted, cert_frac >= 0.33), this is enough to
**certify alpha=0.1**: worst-group G=2 CertCov@0.1 ~ **0.17** (cert_ucb 0.07,
test_risk 0.035), G=3 ~0.09, and even per-client G=5 ~0.06-0.08. At the DEFAULT
0.4/0.3/0.3 split, G=2 sits at cert_ucb ~0.115 (just above alpha) -- the crossover
needs ResNet AND cert_frac >= 0.33 (both Theorem-2 levers), so it is reported as a
combined result, not backbone-alone. Single seed so far (seeds 1,2 = next).

## Main results (seed-aggregated, ResNet-18 **GroupNorm** primary, fixed-MSP, cert_frac=0.5)

GroupNorm is the FL-appropriate normalization (BatchNorm running stats diverge under
non-IID FedAvg) and is the PRIMARY backbone; BatchNorm is an appendix comparison.
Headline LEADS with the feasibility law + the **alpha=0.20 worst-group positive
(robust across norms, 5/5 seeds)**; alpha=0.10 worst-group is a favorable-regime,
seed-variable secondary, reported with full variance. Grouping G=2 (public, fixed)
is the legitimate worst-group certificate; G=1 pooled is a near-IID-only bonus.
cert_ucb summarized by MEDIAN among certified seeds (uncertified -> +inf).

T1 (worst-group G=2 CertifiedCoverage, cifar10, 5 clients, mean+/-std, n_pass/seeds):

| cell (ResNet-GN, clean) | alpha=0.10 | alpha=0.20 |
|---|---|---|
| d=5   | 0.077+/-0.097 (2/5) | **0.392+/-0.097 (5/5)** |
| d=0.5 | 0.091+/-0.104 (3/5) | 0.353+/-0.130 (5/5) |
| d=0.1 | 0 (Mode-1 collapse: empirical test_risk > alpha) | -- |
| d=5/0.5 symmetric-0.35 (BN proxy) | 0 (corruption -> model too poor) | -- |
| covtype (tabular, breadth) | 0 | seed-variable, NOT stable (see below) |

The cifar10 alpha=0.20 row is now a SAVED 5-seed artifact (`runs/agg_alpha20.csv`,
`python aggregate.py --alpha 0.20`): **d=5 G2 = 0.392+/-0.097 (5/5)**, **d=0.5 G2 =
0.353+/-0.130 (5/5)**; both 0/5 false certificates (all cert_ucb <= 0.20 AND all
empirical test_risk <= 0.20). This replaces the earlier single-config T4 ~0.29.

**covtype is NOT a stable second-domain positive (corrected 2026-06-28).** The old
"0.433 @ alpha=0.20" was `make_handoff.py::covtype_frontier` = best-of-4-scores x
best-of-G in {1,2,3} (incl. pooled G=1) on a SINGLE seed. Under the matched CIFAR
protocol (fixed single score, fixed worst-group G=2), 5-seed covtype is
(`runs/agg_covtype.csv`, `python aggregate_covtype.py`): fixed-MSP/G2 = **0/5** at
alpha in {0.20,0.25,0.30}; best honest single score (neg_entropy)/G2 = 0.07+/-0.14
(1/5) @0.20, 0.20+/-0.25 (3/5) @0.30; old selection protocol = 0.10+/-0.17 (2/5)
@0.20 -- all positives concentrated in seed 0 (std > mean). Cause: the federated
LINEAR logreg has realized r_hat ~ 0.14-0.24, too close to alpha (Theorem-2 sample
requirement ~ (alpha - r_hat)^-2). Report covtype as a seed-variable, selection-
optimistic breadth probe, NOT as a positive; a stronger federated tabular model
(lower r_hat) would be needed to certify it honestly.

BN/GN comparison (appendix), d=5 G2: BN alpha=0.10 0.106+/-0.098 (3/5), GN
0.077+/-0.097 (2/5); BN alpha=0.20 0.431+/-0.048 (5/5), GN 0.392+/-0.097 (5/5).
GN lowers realized r_hat (0.022 vs 0.043) but accepts fewer points, so per-group
counts shrink and cert_ucb does not improve -- GN does NOT strengthen alpha=0.10;
alpha=0.20 stays robust. (GN at 80 rounds may be undertrained vs BN; not tuned further.)

Reading of the law: certifiability rises with per-group accepted count (Thm 2),
with backbone quality, and with less heterogeneity; falls under corruption.
alpha=0.20 robustly certified (5/5, both norms); alpha=0.10 at the feasibility edge
(2-3/5), real but seed-variable (std ~ mean) and conditional on cert_frac>=0.5 --
stated in the claim, not hidden.

### Real-data ablations (A4/A5, post-hoc on d=5 ResNet-GN logits, 2026-06-28)

Mirror the synthetic Figs 9/10 on real exported logits (CPU, no retraining).

- **A4-real calibration-budget sweep** (`exp_ablation_calib_budget.py` ->
  `runs/ablation_calib_budget.csv`, `figs/ablation_calib_budget.png`). Model fixed,
  prop/test folds fixed, certification fold GROWN. Worst-group G=2 @ alpha=0.10:
  `cert_ucb` falls monotonically **0.575 -> 0.182** as per-group accepted count rises
  15 -> 185, turning alpha=0.10 non-vacuous (CertCov 0.061+/-0.100, 2/5) at the
  largest budget. Confirms synthetic A4: larger audit budget -> non-vacuous alpha=0.10
  (Theorem-2 floor); full-seed crossover needs a larger trusted pool than CIFAR affords.
- **A5-real unknown-proportion sweep** (`exp_ablation_unknown_prop.py` ->
  `runs/ablation_unknown_prop.csv`, `figs/ablation_unknown_prop.png`). Cert-fold
  unknowns subsampled to rho * deployment-rate (p=0.30); 5 seeds x 40 reps. Empirical
  coverage P(realized test risk <= cert_ucb): **rho=1.0 -> 1.00** (>= 1-delta), but
  **rho=0.75 -> 0.36, rho=0.50 -> 0.01, rho=0.25 -> 0.005** (cert_ucb drops 0.174 ->
  0.070 while realized risk stays 0.141). Confirms synthetic A5: under-representing
  unknowns in calibration is ANTI-CONSERVATIVE.

H2 split-leakage (exp_leakage.py): choosing the threshold on the certification fold
(a leak) raises the unsafe-deploy rate to 18.2% (> delta) vs 2.8% (<= delta) for the
proper proposal/cert split -- split hygiene is load-bearing.

Corruption axis (make_corruption_curve.py, ResNet-GN, F9): worst-group G=2
CertCov@0.20 falls from 0.31 (d=5 clean) / 0.13 (d=0.5 clean) to **0 at any
client-side noise rate >= 0.1** (symmetric AND asymmetric). Even mild training-label
corruption makes the clean-calibrated certificate vacuous (the corrupted model's
accepted-set risk rises) -- the corruption axis of the feasibility law.

F8 certified self-training (Prop 4, ResNet-GN d=5, run_selftrain_cifar): naive
self-training injects pseudo-labels at realized contamination 0.19-0.67 (>> alpha);
the certified loop admits 0 (refuses / hits a Theorem-2 infeasible round and STOPS),
so contamination stays <= alpha by construction. HONEST: certified PREVENTS
catastrophic contamination; the accuracy-improvement half needs a larger feasible
audit-fold regime and is not claimed here.

## Hard rules

- proposal / certification / test folds disjoint; selector chosen on proposal only; never use test labels in proposal/certification.
- Theorem 1/1' (conditional) is the MAIN certificate; mass-ratio is an App-C baseline; do not promote pooled (Prop 3) above stratified.
- Privacy: only pooled is sum-only secure-aggregatable; stratified needs per-client counts.
- Corruption affects TRAIN labels only; calibration/test stay clean.
- `runs/` and `data/` are git-ignored.
