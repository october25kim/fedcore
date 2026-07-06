# Prompt to paste into Claude Code (Ubuntu 4070)

> Copy everything inside the fenced block below into Claude Code as your first
> message. It assumes `CLAUDE.md` and `AGENTS.md` are already at the repo root.
> (Zero-drift alternative: instead of regenerating, `git clone`/`scp` the
> `experiments/fedcore/` and `scripts/` folders from the Mac. The prompt below is
> for the case where you want Claude Code to recreate them.)

---

```text
You are working in the Fed-CORE repo (Federated Certified Open-Set Recognition).

STEP 0 — Read first, then restate.
Read CLAUDE.md and AGENTS.md at the repo root in full. In 5 bullets, restate:
the controlled object R_sel(lambda); Theorem 1/1' (conditional selective-risk
certificate); why naive pooling is invalid; the privacy taxonomy; and the
proposal/certification/test split rule. Do not proceed until this is correct.

STEP 1 — Recreate the experiment package.
Create exactly these files under experiments/fedcore/ and scripts/. Use Python
3.10+, numpy + scipy only for the CPU core; torch + torchvision only in
models.py / fed_train.py / run_cifar.py. Keep everything typed and documented.

experiments/fedcore/
  certificates.py     CP primitives + certificates (SPEC A below)
  clients.py          synthetic heterogeneous clients (SPEC B)
  scores.py           MSP/entropy/margin/energy + scored_views (SPEC C)
  selector.py         open-set error + risk-buffered threshold + counts (SPEC D)
  certify.py          glue -> metric schema (SPEC E)
  config.py           FedOSRConfig dataclass (SPEC F)
  fedosr_split.py     open-set + Dirichlet non-IID + calibration folds (SPEC G)
  noise.py            client-side TRAIN-label corruption (SPEC H)
  models.py           SimpleCNN over known classes (torch) (SPEC I)
  fed_train.py        FedAvg + logit export (torch) (SPEC J)
  run_smoke.py        fake-logit end-to-end (no torch) (SPEC K)
  run_cifar.py        real CIFAR-10/100 (torch) (SPEC L)
  exp_lemma_L.py      Lemma L verification (SPEC M)
  exp_pooling_fail.py pooling-fail ablation (SPEC N)
  exp_necessity.py    necessity-of-certificate (SPEC O)
scripts/
  docker_cifar.sh     CUDA torch container wrapper (SPEC P)

=== SPEC A — certificates.py ===
Clopper-Pearson one-sided limits via scipy.stats.beta:
  cp_upper(k,n,eps) = 1.0 if k>=n else Beta.ppf(1-eps, k+1, n-k); 1.0 if n<=0.
  cp_lower(k,n,eps) = 0.0 if k<=0 else Beta.ppf(eps, k, n-k+1); 0.0 if n<=0.

conditional_risk_certificate(A,K,n,delta,Lambda='simplex',lam=None,box=None,...)
  -- THE MAIN certificate (Theorem 1/1'). Uses K_j|A_j ~ Bin(A_j,r_j).
  Lambda='simplex': eps=delta/J; rbar_j = cp_upper(K_j,A_j,eps) (=1.0 if A_j==0);
                    U = min(max_j rbar_j, 1.0).
  Lambda in {'box','known'}: eps=delta/(3J); rbar_j as above;
                    alow_j=cp_lower(A_j,n_j,eps); ahigh_j=cp_upper(A_j,n_j,eps).
                    Inner sup over a-vertices (each a_j in {alow_j,ahigh_j},
                    enumerate 2^J via itertools.product) of
                    (sum lam*a*rbar)/(sum lam*a) when sum lam*a>0, else +inf.
                    'known': fixed lam. 'box': sample lam in [lo,hi], renormalize,
                    take max over samples. Return U=min(.,1.0) or +inf (infeasible).
  Return a dataclass with U, rbar, alow, ahigh, eps, feasible.

stratified_certificate(A,K,n,delta,Lambda='simplex',lam=None,box=None,...)
  -- MASS-RATIO BASELINE ONLY (Appendix C). eps=delta/(2J);
  mbar_j=cp_upper(K_j,n_j,eps); alow_j=cp_lower(A_j,n_j,eps);
  U=sup_{lam in Lambda}(sum lam*mbar)/(sum lam*alow); simplex closed form
  max_j mbar_j/alow_j (alow_j>0). Cap U at 1.0. Return U, mbar, alow, eps.

pooled_cp(A,K,delta) = cp_upper(sum K, sum A, delta).
true_selective_risk(a,r,lam) = sum(lam*a*r)/sum(lam*a).

=== SPEC B — clients.py ===
@dataclass ClientPopulation(a: np.ndarray, r: np.ndarray); property m=a*r, J.
draw_counts(pop,n,rng) -> (A,K): per client, accept each of n_j points w.p. a_j;
  among accepted, error w.p. r_j (so K_j<=A_j). Returns int arrays.
heterogeneous_population(n_good=4,a_good=0.7,r_good=0.02,a_bad=0.5,r_bad=0.3):
  4 low-risk clients + 1 high-risk client.

=== SPEC C — scores.py ===
softmax; msp; neg_entropy (= -entropy, higher=more confident); margin (top1-top2
of softmax); energy (= T*logsumexp(logits/T), higher=ID). All oriented so HIGHER
=> more likely a known class (accept). compute_score(name,logits).
scored_views(logits,y_open,client,score_names) -> {score_name:{score,pred,y_open,
client}} with pred=argmax(logits).

=== SPEC D — selector.py ===
Open-set convention: y_open is known class in [0,C) or -1 for unknown.
open_set_error(pred,y_open) = (y_open<0) | (pred!=y_open).
@dataclass Selector(threshold,feasible); accept(score)=score>=threshold.
empirical_risk_coverage(score,err,t) -> (coverage, selective_risk among accepted).
choose_threshold(score,pred,y_open,gamma,alpha,n_grid=300): over score quantiles,
  maximize coverage s.t. empirical risk <= gamma*alpha; if none feasible return
  threshold=+inf, feasible=False (accept nothing).
counts_per_client(score,pred,y_open,client,selector,n_clients) -> (A,K,n) per client.

=== SPEC E — certify.py ===
certify_for_score(...): (1) choose_threshold on PROPOSAL fold; (2) counts on
CERT fold; (3) U = conditional_risk_certificate(...).U  [PRIMARY]; coverage LCB
from alow=cp_lower(A_j,n_j,delta/(2J)) via worst-case over Lambda (simplex: min_j
alow_j; known: lam.alow; box: min over sampled lam of lam.alow); (4) empirical on
TEST fold. Emit metric dict with keys: score_name, gamma, alpha, delta, Lambda,
dirichlet_alpha, n_clients, certified (= feasible & U<=alpha), cert_risk_ucb,
cert_coverage_lcb, cert_n=sum A, cert_k=sum K, prop_coverage, prop_risk,
test_coverage, test_risk. certify_grid sweeps score x gamma (x Lambda).

=== SPEC F — config.py ===
FedOSRConfig dataclass: dataset, n_known=6, seed=0, n_clients=5,
dirichlet_alpha=0.1, noise_type='none', noise_rate=0.0, prop/cert/test fracs,
unknown_contamination=0.30, alpha=0.10, delta=0.10, gammas=(0.5,0.7,1.0),
Lambda='simplex', box_radius=0.15, scores=('msp','neg_entropy','margin','energy'),
rounds=50, local_epochs=2, batch_size=64, lr=0.01. folds() normalizes fractions.

=== SPEC G — fedosr_split.py ===
open_set_split(labels,n_known,seed) -> known/unknown classes + remap to [0,n_known).
dirichlet_partition(indices,labels_remapped,n_clients,alpha,seed) -> per-client
  index lists (label Dirichlet(alpha); smaller alpha = more non-IID).
build_calibration(known_idx,known_y_remapped,unknown_idx,n_clients,folds,
  unknown_contamination,seed) -> per-client {prop,cert,test}:{idx,y_open};
  each client gets known points (true remapped label) + injected unknowns (-1).
  CALIBRATION/TEST STAY CLEAN.

=== SPEC H — noise.py ===
make_label_noise(remapped_labels,indices,noise_type,rate,n_known,seed) ->
  {dataset_index: noisy_label} for flipped TRAIN points only. symmetric=uniform
  other class; asymmetric=(y+1)%n_known; none/rate<=0 -> {}. No self-flips.
  NEVER call on calibration.

=== SPEC I/J — models.py / fed_train.py ===
SimpleCNN(n_known): 3 conv blocks (BN+ReLU, maxpool) -> GAP -> Linear. make_model.
local_train(model,loader,epochs,lr,device): SGD+momentum, CE over known classes.
fedavg(make_model_fn,client_datasets,rounds,local_epochs,lr,batch_size,device):
  weighted average of client state_dicts (weights = client sizes; copy non-float
  buffers from client 0). export_logits(model,base_dataset,indices,device,bs)
  -> logits (len(indices), n_known).

=== SPEC K — run_smoke.py ===
Fake-logit end-to-end (no torch): synth J clients with a built-in confidence
deformation (some unknown points get a spurious high-confidence known boost) and
one hardest/most-contaminated client. Run scored_views -> certify_grid for
Lambda in {simplex, box(+-0.10 around uniform)}. Print the metric schema table;
save smoke_results.csv. Must run end-to-end on CPU.

=== SPEC L — run_cifar.py ===
argparse: --dataset cifar10|cifar100 --n_known --n_clients --dirichlet_alpha
--rounds --local_epochs --alpha --delta --noise_type --noise_rate --seed --out.
Load CIFAR (torchvision, normalize). open_set_split on train labels.
known-train -> dirichlet_partition -> per-client _LabelRemapSubset(train, idx,
remap, label_override=make_label_noise(...)). Trusted calibration from TEST set
(known + unknown) via build_calibration. fedavg train. Per fold: gather indices/
y_open/client across clients, export_logits, scored_views, certify_grid for
Lambda in {simplex, box}. Print + save CSV. Identical certification path as smoke.

=== SPEC M/N/O — standalone CPU experiments ===
exp_lemma_L.py: does binomial CP upper limit stay conservative for a Poisson-
  binomial mean? Sweep A in {100,300,1000}, rbar in {0.02,0.05}, profiles
  {homogeneous, mild, strong-bimodal, two-point p in {0.25,0.5,0.75}}; report
  empirical coverage of rbar by cp_upper(K,A,delta); delta=0.1.
exp_pooling_fail.py: 4 good + 1 bad client; deployment mixtures {matched,uniform,
  shift->bad(0.6), all->bad}; compare empirical coverage of R_sel(lambda*) by
  pooled_cp vs conditional simplex vs conditional box; delta=0.1.
exp_necessity.py: at matched budget, unsafe-deploy rate P(deploy | R_sel>alpha)
  for naive-empirical (deploy iff sumK/sumA<=alpha) vs pooled_cp vs Fed-CORE
  conditional(known lam); sweep r_bad so R_true crosses alpha=0.05; delta=0.1.

=== SPEC P — scripts/docker_cifar.sh ===
env-driven (IMAGE, DATASET, N_KNOWN, N_CLIENTS, DIRICHLET_ALPHA, ROUNDS,
LOCAL_EPOCHS, ALPHA, DELTA, NOISE_TYPE, NOISE_RATE, SEED, OUT). Mount repo at
/workspace, run experiments/fedcore/run_cifar.py with --gpus all. Default
IMAGE=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime. Do not commit runs/ data/.

STEP 2 — Self-verify against ACCEPTANCE GATE (do not skip).
Run the three CPU experiments and the smoke; confirm:
  - exp_lemma_L: worst-case coverage >= 0.90 (expect ~0.919); homogeneous near
    nominal, two-point profiles drive coverage toward 1.0.
  - exp_pooling_fail (delta=0.1): pooled coverage ~1.0 at 'matched' but
    COLLAPSES to ~0.0 at 'shift->bad'/'all->bad'; conditional simplex stays ~1.0
    for every mixture; box recovers tightness and is valid for in-box lambda.
  - exp_necessity (alpha=0.05, delta=0.1): at the boundary R_sel=alpha,
    naive-empirical unsafe-deploy ~0.52, pooled-CP ~0.08, Fed-CORE ~0.00.
  - conditional vs mass-ratio (simplex): conditional median U ~0.37 (valid) is
    TIGHTER than mass-ratio ~0.45.
  - run_smoke: runs end-to-end, emits the full schema; box-Lambda certifies a few
    (score,gamma=0.5) combos with cert_risk_ucb just under alpha.
If any number is materially off, debug the implementation (not the gate) until it
matches, and report what was wrong.

STEP 3 — Run the real experiments (Docker-first), report in the fixed format.
Ladder (CLAUDE.md sec 4): run_smoke -> cifar10 clean seed0 -> cifar10 symmetric
0.35 seed0 -> cifar10 asymmetric 0.20 seed0 -> seeds 1,2 -> dirichlet sweep
{0.1,0.5,5} -> cifar100. Use scripts/docker_cifar.sh, e.g.:
  bash scripts/docker_cifar.sh
  NOISE_TYPE=symmetric  NOISE_RATE=0.35 bash scripts/docker_cifar.sh
  NOISE_TYPE=asymmetric NOISE_RATE=0.20 bash scripts/docker_cifar.sh
For each run report: 진단 요약 / 확인한 명령 / 핵심 결과 (alpha,delta,Lambda,
score,gamma,dirichlet_alpha,cert_risk_ucb,cert_coverage_lcb,test_risk,
test_coverage) / 판정 (strong go / moderate go / warning / fail) / 다음 행동.
Headline metric: CertifiedCoverage@alpha.

STEP 4 — Upcoming experiments to implement, then run (paper sec 5).
  (a) Validity plot: empirical P(R_sel<=Ubar) vs heterogeneity, must stay >=1-delta.
  (b) Tightness: conditional vs mass-ratio vs box-Lambda vs pooled.
  (c) CertifiedCoverage@alpha frontier over alpha in {0.01,0.02,0.05,0.1}.
  (d) Heterogeneity sweep -> certified-coverage-collapse curve vs Theorem 2.
  (e) Score-agnostic: 4 scores keep validity, change only coverage.
  (f) Necessity (real data): naive empirical threshold unsafe-deploy rate > delta.
  (g) Superiority: matched-risk coverage vs oracle-tuned FedPD/FedOSS/FOOGD;
      price-of-federation vs centralized oracle as heterogeneity->0.
  (h) Utilization A: automation rate (=CertifiedCoverage@alpha) at guaranteed risk.
  (i) Utilization B / Proposition 4 (NEW CODE): certified federated self-training.
      Partition trusted set into T disjoint audit folds; each round t: train
      FedAvg, choose selector on proposal, certify on fold C^(t) at level delta/T,
      accept pseudo-labels from unlabeled pool, fold them back into training.
      Compare downstream accuracy of certified vs naive (uncertified) vs no
      self-training under non-IID corruption. Naive should diverge; certified
      should stay safe (contamination <=alpha per round) and improve.

HARD RULES (also in CLAUDE.md / AGENTS.md):
- Docker-first, smoke-first. Never judge success by accuracy/AUROC alone.
- proposal/certification/test folds disjoint; NEVER use test labels in
  proposal/certification. Selector chosen on proposal only.
- Theorem 1/1' (conditional) is the MAIN certificate; the mass-ratio version is
  an Appendix-C baseline. Do NOT promote pooled (Proposition 3) above stratified.
- Privacy: only pooled is sum-only; stratified needs per-client (or grouped) counts.
- Corruption affects TRAIN labels only; calibration/test stay clean.
- Report failed commands; do not hide them.
```

---

## Notes for Sanghoon (not part of the prompt)

- If you prefer **zero drift**, skip regeneration and copy the real folders:
  `experiments/fedcore/` and `scripts/` already exist in the synced project
  folder; `scp`/`git add` them to the 4070. The prompt's STEP 2 acceptance gate
  still works as a sanity check after copying.
- The only piece that is **not yet implemented anywhere** is STEP 4(i), the
  certified self-training loop (Proposition 4). Claude Code will build it new.
- All other files and the acceptance numbers come from validated CPU runs on this
  side, so they are reliable regression targets.
