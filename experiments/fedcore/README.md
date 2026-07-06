# Fed-CORE certificate experiments

Reusable scaffold validating the two load-bearing claims of the Fed-CORE draft
(`../../Fed-CORE_draft.md`). Pure `numpy` + `scipy`; no FL infrastructure yet, so
the certificate logic can be validated in isolation before plugging into a real
non-IID CIFAR pipeline.

```
certificates.py     CP primitives; stratified (Thm 1) + pooled (Thm 3 / naive) certificates
clients.py          synthetic heterogeneous client populations and count draws
exp_lemma_L.py      Lemma L numerical verification
exp_pooling_fail.py pooling-fail ablation (non-reducibility of Thm 1)
exp_necessity.py    necessity-of-certificate (naive empirical thresholding fails)
```

Run:

```bash
python3 exp_lemma_L.py
python3 exp_pooling_fail.py     # writes lemma_pooling_results.csv
python3 exp_necessity.py        # unsafe-deployment rates vs naive thresholding
```

## Result 0 — Necessity of a certificate (`exp_necessity.py`)

At matched calibration budget ($J{=}5$, $\alpha{=}0.05$, $\delta{=}0.10$), the
**unsafe-deployment rate** $\Pr(\text{deploy}\mid R_{\rm sel}>\alpha)$ (must be
$\le\delta$ for a valid method):

| true risk region | naive-empirical | pooled-CP | Fed-CORE |
|---|---|---|---|
| at boundary $R\!=\!\alpha$ | **0.52** ⚠ | 0.08 | 0.00 |
| above boundary $R\!=\!0.069$ | 0.002 | 0.00 | 0.00 |

A practitioner who simply thresholds on the empirical accepted-error estimate
deploys **unsafely ~half the time at the boundary** — good AUROC / empirical
tuning does not control the risk. The certificate is necessary. (Under *matched*
mixtures pooled-CP has high power; its failure mode is *mismatch*, see Result 2.)

---

## Result 1 — Lemma L: SUPPORTED

**Question.** Is the binomial Clopper–Pearson upper limit applied to a
*Poisson-binomial* accepted-error count (heterogeneous per-point error
probabilities) still a valid `1 - delta` upper bound for its mean? If yes,
Theorem 3's pooled certificate is internally valid under matched-λ.

**Finding (delta = 0.10, target coverage ≥ 0.90).** Across `A ∈ {100,300,1000}`,
`rbar ∈ {0.02,0.05}`, and homogeneous / mild / strong-bimodal / adversarial
two-point profiles, the **worst observed coverage was 0.919 ≥ 0.90**. The
homogeneous (exact-binomial) case sits closest to nominal; *increasing*
heterogeneity only makes the bound **more** conservative (two-point profiles
push coverage toward 1.0). This matches the theory: at fixed mean the binomial
is the maximally dispersed sum of independent Bernoullis (Hoeffding, 1956), so
its CP limit dominates the Poisson-binomial's.

**Implication.** Theorem 3 (tighter pooled certificate) **survives** — but only
under the matched-λ assumption, as Result 2 shows. A formal proof (binomial-CP
conservativeness for the Poisson-binomial mean) should accompany the theorem.

## Result 2 — Pooling-fail ablation: confirms Theorem 1 is non-reducible

**Setup.** 4 low-risk clients (a=0.70, r=0.02) + 1 high-risk client (a=0.50,
r=0.30); `n_j = 400`; `delta = 0.10`. We measure empirical coverage of the
*deployment* risk `R_sel(λ*)` under several deployment mixtures.

| mixture λ* | R_true | cov **pooled** | cov **stratified (simplex)** | cov **stratified (box)** | medU pooled / simplex / box |
|---|---|---|---|---|---|
| matched (control) | 0.052 | **1.00** | 1.00 | 1.00 | 0.072 / 0.445 / 0.167 |
| uniform | 0.062 | 0.92 | 1.00 | 1.00 | 0.072 / 0.445 / 0.168 |
| shift → bad (0.6) | 0.165 | **0.00** ⚠ | 1.00 | 0.56 | 0.072 / 0.444 / 0.168 |
| all → bad | 0.300 | **0.00** ⚠ | 1.00 | 0.00 | 0.072 / 0.446 / 0.167 |

**Reading.**

1. **Naive pooled CP is valid only under matched-λ.** As soon as the deployment
   mixture overweights the high-risk client, pooled coverage **collapses to 0%**
   (it certifies ≈0.072 while the true deployment risk is 0.165–0.30). This is
   the empirical proof that one *cannot* reduce the federated certificate to
   "centralized Clopper–Pearson on the pooled accepted set."
2. **Stratified simplex (Theorem 1) holds for every mixture** — coverage ≈1.0
   throughout, at the price of a larger, worst-client-dominated certificate
   (U ≈ 0.45). It is honest, not vacuous: it certifies the risk that is actually
   defensible without knowing λ.
3. **Box-Λ is the practical tightening.** Restricting Λ to ±0.15 around the
   (known) client data fractions tightens U from 0.45 → 0.17 and stays valid for
   every λ* **inside** the box (matched, uniform); it correctly *does not* claim
   validity for the out-of-box extremes (shift-0.6, all-bad), where coverage
   drops — exactly the robustness/tightness trade-off Λ is meant to expose.

**Mapping to the draft.** Result 1 ⇒ §4.5 / Theorem 3 (Lemma L). Result 2 ⇒
§4.2 / Theorem 1 closed form and the §5 ablation (iii) "naive-pooled CP violates
the target under heterogeneity while Theorem 1 holds." Together they substantiate
the non-reducibility argument that defeats the "trivial combination" reviewer.

## Caveats (certificate experiments)

Synthetic Bernoulli clients isolate the *certificate* behavior; they do not test
score quality, deep-model confidence deformation, or FL optimization. Coverage is
estimated with 2.5k–8k trials, so reported coverages carry ~±0.5–1.0% Monte-Carlo
error.

---

# FedOSR pipeline (score → risk-buffered selector → certificate)

A full pipeline that turns a (federated, non-IID) open-set classifier into a
certified accept/reject rule. The scoring / selection / certification stages are
pure `numpy` and run in the sandbox; only model + FedAvg training need torch.

```
config.py         experiment configuration dataclass
scores.py         MSP / neg-entropy / margin / energy  (higher = accept)
selector.py       open-set error semantics + risk-buffered threshold choice + per-client counts
certify.py        glue: selector -> per-client (A_j,K_j) -> stratified certificate -> metric schema
fedosr_split.py   open-set class split + Dirichlet non-IID partition + calibration folds (numpy)
models.py         compact CNN over known classes (torch)
fed_train.py      FedAvg loop + logit export (torch)
run_smoke.py      FAKE-LOGIT smoke -- end-to-end in the sandbox, no torch
run_cifar.py      real CIFAR-10/100 run -- for the project's Docker (torch+torchvision)
```

**Open-set error semantics.** A point `(score, pred, y_open)` with `y_open` a known
class in `[0,C)` or `-1` for unknown. Accept iff `score >= t`; an accepted point is
an error iff `y_open == -1` (accepted an unknown) **or** `pred != y_open` (wrong known
class). Per client we report the secure-aggregatable counts `A_j` (accepted) and
`K_j` (accepted-errors), which feed the stratified certificate.

**Metric schema (emitted per score × gamma × Lambda).**
`certified, cert_risk_ucb, cert_coverage_lcb, cert_n, cert_k, prop_coverage,
prop_risk, test_coverage, test_risk, score_name, gamma, alpha, delta, Lambda,
dirichlet_alpha, n_clients`.

### Smoke result (sandbox, `python3 run_smoke.py`)

J=5 clients, C=6 known, α=δ=0.10, with a built-in confidence deformation and one
hardest/most-contaminated client. The pipeline runs end-to-end and emits the full
schema. Representative pattern:

| Lambda | certified | best cert_risk_ucb (gamma=0.5) | note |
|---|---|---|---|
| simplex | 0/12 | ~0.16 | robust-but-conservative; the hardest client caps the worst-case mixture |
| **box** (±0.10) | **3/12** | **~0.092–0.099 ≤ α** | MSP / entropy / margin certify at gamma=0.5; tighter Λ recovers feasibility |

This confirms the `certified` flag flips correctly, the risk buffer `gamma` behaves
(smaller gamma → lower UCB), the bound is score-agnostic, and box-Λ trades
robustness for the tightness needed to certify. (The exact counts depend on the
synthetic regime; the point is the wiring and the qualitative behavior.)

### Real CIFAR run (Docker, `python run_cifar.py ...`)

```bash
python run_cifar.py --dataset cifar10 --n_known 6 --n_clients 5 \
    --dirichlet_alpha 0.1 --rounds 50 --local_epochs 2 --alpha 0.10 --delta 0.10
```

Trains FedAvg on the non-IID known-class partition, holds out the remaining
classes as test-time unknowns, builds the trusted calibration pool from the test
split (known points + injected unknowns), exports logits, and runs the identical
certification path as the smoke. This is the run that answers the central
question — *is certified accepted coverage non-trivial at CIFAR scale under
non-IID corruption?* — and should be launched via the project's Docker harness.

## Status

- numpy core (config/scores/selector/certify/fedosr_split): runs + validated in sandbox.
- torch path (models/fed_train/run_cifar): compiles; to be executed in Docker (no GPU here).
- Next: wire `run_cifar.py` into `scripts/docker_*` and run cifar10 sym/asym noise seeds.
