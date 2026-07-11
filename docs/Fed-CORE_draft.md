# Fed-CORE: Federated Certified Open-Set Recognition via Selective&nbsp;Risk&nbsp;Control

Sanghoon Kim

[Department / University, City, Country]

Corresponding author. E-mail address: october25kim@gmail.com

---

## Abstract

Federated learning is increasingly deployed in safety-sensitive applications where test inputs may belong to classes never seen during training. Existing federated open-set recognition methods evaluate unknown rejection only empirically (AUROC, FPR95), and federated conformal prediction certifies closed-set prediction-set coverage; neither bounds the error rate of the predictions a deployed model accepts. We propose Fed-CORE, a post-hoc certification layer that bounds this accepted selective risk for any federated open-set model, with a finite-sample guarantee that is distribution-free with respect to the audited deployment distribution, under client heterogeneity and an unknown deployment mixture. Pooling calibration data across heterogeneous clients certifies only the calibration mixture and turns anti-conservative whenever deployment overweights a high-risk client, because the pooled accepted-error count is Poisson-binomial rather than binomial; Fed-CORE instead bounds each client's conditional selective risk from released accepted/error counts and certifies the global risk as a mixture-robust worst case, together with a per-stratum feasibility threshold that characterizes when certification is possible at all. Across synthetic and CIFAR-10/100 real-logit experiments, certified deployments showed no held-out risk violation, naive pooling collapsed under client-mixture shift, and certified coverage followed the feasibility law; certifying the native score of a full FedOSR method (FedPD-PROSER) reached risk targets 0.10 and 0.20, and a full-simplex per-client certificate confirmed the positives do not depend on the grouping relaxation.

**Keywords:** Federated learning; Open-set recognition; Selective risk control; Conformal prediction; Distribution-free certification; Uncertainty quantification

---

## 1. Introduction

Consider many hospitals that jointly train one diagnostic model without sharing any patient record. In use, the model will sometimes meet a disease it never saw during training, so it must answer "I do not know" instead of forcing a guess. Two practical questions follow: among the cases where the model *does* commit to an answer, how often is it wrong — and can we *promise*, with statistical confidence, that this error rate stays below a chosen tolerance, although every hospital's data looks different? This paper provides exactly that promise, under one explicitly stated condition: it is computed from a small amount of trusted, correctly labelled audit data that tracks the deployment stream, and it does not change or retrain the model. We call the framework Fed-CORE; in one sentence, *it does not try to make the model better — it certifies which of the model's answers are safe to trust.*

Federated learning (FL) trains a shared model across clients that never disclose their raw data [1]. A broad survey documents how widely such training is deployed and how central client heterogeneity is to its behavior [2]. In practice these models are deployed in *open-world* conditions: a fraud detector meets a new fraud pattern, a clinical model meets a disease subtype absent from every hospital's training data. A deployed model must therefore not only classify known classes but also **abstain** on inputs it cannot safely classify, which is the classical reject option [3]. This is open-set recognition (OSR) [4], and its federated form (FedOSR) is now an active area [5].

The dominant FedOSR methods improved the *quality* of unknown rejection — parameter-disentanglement aggregation in FedPD [5], score-model out-of-distribution (OOD) detection in FOOGD [6], open-set voting in FedOV [7], novel-class discovery in FedNovel [8] — and reported it through ranking metrics such as AUROC and FPR95. None answers the question a deployer actually needs answered: **if I accept and act on this model's confident predictions, what is the worst-case error rate among them, and can I guarantee it stays below a tolerance α?** A model with high AUROC can still have an unacceptable error rate among accepted predictions at the fixed operating threshold that deployment requires.

A second literature does provide finite-sample guarantees in FL: federated conformal prediction (FCP) [9] constructs prediction sets with marginal coverage under a partial-exchangeability relaxation of the i.i.d. assumption, with variants robust to label shift [10] and Byzantine clients [11]. However, FCP is closed-set — it assumes the true label is among the known classes and guarantees that the label lies in the returned set. It does not reject unknowns, and, as its own authors note, its selective-classification demonstrations were heuristics *without* a guarantee; coverage of a prediction set and the *risk of an accepted point prediction* are different functionals.

A third body of work certifies unknown rejection, almost entirely in the **centralized** setting: conformal novelty detection with false-discovery-rate (FDR) control against a clean reference sample [12], conformal open-set classification [13], reject-option conformal classification [14], conformal risk control [15], and selective risk control on a chosen subset [16,17]. The one recent decentralized entrant [18] controls **batch FDR** in a pure novelty-detection framing, with no known-class classifier and no notion of accepted selective risk — again a different object requiring different finite-sample machinery.

The intersection, namely **a federated, heterogeneity-aware, finite-sample certificate on the accepted selective risk of an open-set classifier**, remains unaddressed by existing methods. Filling it is the contribution of this paper.

**Why this is hard, and not a trivial combination.** It is tempting to apply a Clopper–Pearson [19] selective-risk certificate, as in the centralized i.i.d. case, to the *pooled* federated calibration data. Under heterogeneity this is invalid: clients have different conditional error rates, so pooling lets a few reliable clients "average away" the mistakes of an unreliable one, and the guarantee looks safer than it truly is (made precise as a Poisson-binomial failure in Proposition 2). The correct target is the deployment-mixture-weighted accepted error rate $R_{\mathrm{sel}}(\lambda)$, defined formally in Section 3: of all the predictions the model chooses to act on, the fraction that are wrong under the unknown mix $\lambda$ of clients that deployment will actually present. Certifying this ratio, finite-sample, distribution-free, and robustly over unknown $\lambda$, reduces to *neither* the single-binomial certificate of the centralized case *nor* the quantile-coverage certificate of federated conformal prediction.

The purpose of this study is to certify, with a finite-sample distribution-free guarantee, the accepted selective risk of a federated open-set classifier under client heterogeneity and unknown deployment mixture, without retraining. Our hypothesis: a small trusted clean audit set, although too small to repair a heterogeneous global model, suffices to certify which of its predictions can be accepted safely.

The main contribution of this study can be summarized as follows:

- We formalize the federated accepted selective risk $R_{\mathrm{sel}}(\lambda)$ as the certification target for federated open-set recognition, a different functional from prediction-set coverage, ranking metrics, and batch false discovery rate (Section 3).
- We derive a finite-sample, distribution-free certificate for $R_{\mathrm{sel}}(\lambda)$ from the conditional law $K_j\mid A_j\sim\mathrm{Bin}(A_j,r_j)$, valid under client heterogeneity and unknown deployment mixture (Theorems 1–2, Corollary 1), and prove that naive pooling of federated calibration counts is anti-conservative under mixture shift (Proposition 2), so the problem is not reducible to centralized conformal prediction (Section 4).
- We characterize when certified open-set deployment is statistically feasible: a per-stratum sample threshold (Theorem 3) in the risk target and the accepted audit count that turns the apparent null at small risk targets into a quantitative law, traced on real data along the target, heterogeneity, and corruption axes (Sections 4.5 and 5.4–5.5).
- We evaluate validity, non-reducibility, feasibility, and score/base-model agnosticism on synthetic and real federated benchmarks, with no held-out risk violation among certified deployments (Section 5); the certificate variants' privacy/communication trade-offs (Proposition 3) and a certified admission gate for federated self-training with bounded pseudo-label contamination (Proposition 4) complete the deployment picture (Sections 4.6–4.7, 5.6).

Fed-CORE is best read not as a new FedOSR algorithm but as a **certification layer for any FedOSR / open-set FL model**, whose output is *used* for safe automation and certified self-training.

---

## 2. Related Work

**Federated open-set / novel-class / OOD recognition.** FedPD [5] and its extension FedPD++ [20] framed FedOSR around cross-client interference between closed- and open-set objectives and addressed it by parameter disentanglement and divide-and-conquer aggregation. FOOGD [6] jointly targeted OOD generalization and detection; crucially, its only formal result bounded the *estimation error of its score model* (an MMD bound), not the error rate of the rejection decision. FedOV [7] tackled one-shot FL under label skew via open-set voting; FedNovel [8] and federated open-world semi-supervised learning [21] discovered novel classes across clients; FOSTER [22] recast client heterogeneity itself as an OOD-detection signal. Noise-Resistant FedOSR [23] is closest to a corruption setting, using Bayesian uncertainty and label correction; like all of the above, it evaluated rejection empirically (AUROC/FPR95). Centralized OSR continues to refine compact accept/reject regions [24], and federated heterogeneity has been studied beyond images, for example on graphs [25].

Adjacent federated lines addressed individual facets of deployment reliability — personalization and clustering under statistical heterogeneity [26,27,28], fairness-aware objectives and the reject option for discrimination control [29,30], scalable scheduling and aggregation [31,32], and unsupervised outlier and concept-drift detection [33,34] — but none certifies a finite-sample bound on the accepted-prediction error rate.

**Federated conformal / distribution-free uncertainty.** Lu et al. [9] introduce Federated Conformal Prediction, replacing exchangeability with *partial exchangeability* (a test point matches client $k$ with probability $\lambda_k$) and proving marginal prediction-set coverage with a privacy-preserving T-Digest quantile sketch; Plassier et al. [10] handle label shift, and Rob-FCP [11] adds Byzantine robustness. All certify **closed-set prediction-set coverage**, not unknown rejection or selective risk. Beyond federation, non-exchangeable conformal risk control [35] relaxed exchangeability along temporal and covariate axes to control a monotone expected risk under drift, whereas Fed-CORE certifies a client-stratified accepted-risk *ratio* under an unknown deployment mixture — a different relaxation and a different functional. We adopt the partial-exchangeability viewpoint but certify a *risk* (a binomial functional), which behaves differently from a *quantile*.

**Centralized selective-risk certification.** Conformal Risk Control [15] generalized conformal coverage to any monotone risk, building on distribution-free risk-controlling prediction sets [36]. A fast-moving 2025–2026 line then certified *post-selection* risk: two-stage selective conformal risk control [16], e-value risk control among "trusted" cases [17], and a joint finite-sample certificate for risk ratio, acceptance, and utility under an adaptively chosen selector [37]. These are Fed-CORE's closest statistical neighbors, certifying the same *kind* of object — a post-selection risk ratio. However, all of them assume **centralized, exchangeable (i.i.d.) calibration from a single population**: none addresses client-stratified calibration, heterogeneous per-client error rates, an unknown deployment mixture $\lambda$, or the count-release constraints of federation, which are precisely the axes on which pooled calibration becomes invalid (Proposition 2). Fed-CORE is therefore best read as the federated, mixture-robust counterpart of this line, not as a competitor to it; these methods are positioned conceptually and are not drop-in valid baselines in the federated setting (Section 5.1).

**Conformal open-set / novelty detection.** Conformal novelty detection [12], conformal open-set classification via Good–Turing p-values [13], and reject-option conformal classification [14] certified rejection centrally. The sole decentralized entry [18] controlled **global FDR over a test batch** via quantized surrogate scores — pure novelty detection without a known-class classifier, a different functional from accepted selective risk; we treat it as the nearest neighbor and differentiate on object and on certifying a single fixed selector.

**Positioning.** Table 1 organizes the prior work by the object it certifies. The FedOSR line reports empirical rejection quality [5], the federated conformal line certifies closed-set prediction-set coverage [9], and centralized selective risk control certifies the right kind of object in the wrong setting; none certifies accepted point-prediction risk under an unknown client mixture. Fed-CORE fills this intersection with a certificate whose calibration statistic is a conditional-binomial proportion rather than a score quantile: to our knowledge, the first federated, client-stratified, deployment-mixture-robust certificate for the accepted selective risk of open-set point predictions — the setting in which naive pooling is invalid under heterogeneity.

**Table 1. Prior work organized by certified object.**

| method family | fed. | open-set | certified object | unknown $\lambda$ | released statistic | finite-sample |
|---|:-:|:-:|---|:-:|---|:-:|
| FedPD / FedOSS / FOOGD [5,6,20,38] | ✓ | ✓ | none (empirical AUROC/FPR95) | ✗ | model updates | ✗ |
| FCP and variants [9,10,11] | ✓ | ✗ | prediction-set coverage | partial exch. | quantile sketch | ✓ |
| non-exchangeable CRC [35] | ✗ | ✗ | monotone expected risk under drift | ✗ | — (centralized) | ✓ |
| CRC / SCRC / SCoRE / joint cert. [15,16,17,37] | ✗ | optional | monotone risk / selected risk ratio | ✗ | — (centralized) | ✓ |
| decentralized novelty FDR [18] | partial | novelty | batch FDR | limited | quantized scores | ✓ |
| **Fed-CORE (this work)** | ✓ | ✓ | accepted selective risk $R_{\mathrm{sel}}(\lambda)$ | ✓ | count pairs | ✓ |

---

## 3. Problem Setup

Figure 1 locates the certified object among its neighbors before the formal definitions: of the four questions one can ask about a deployment stream partitioned into accepted and rejected points, only the error rate among accepted predictions under the deployment mixture is what Fed-CORE certifies — ranking quality, prediction-set coverage, and batch FDR do not bound it. The figure also marks where the trusted folds enter: the proposal fold selects the selector threshold, the disjoint certification fold certifies $R_{\mathrm{sel}}$ independently of that choice, and the test fold only estimates deployment behavior.

![](experiments/fedcore/figs/fig0_problem_diagram.png){width=88%}

**Figure 1. What Fed-CORE certifies.** A federated open-set model $\hat h$ with a selector $A$ partitions the deployment stream into accepted and rejected points; among the four quantities that can be asked about this stream, Fed-CORE certifies the accepted selective risk $R_{\mathrm{sel}}(\lambda)$.

**Clients and mixture.** There are $J$ clients; client $j$ has data distribution $P_j$ over $\mathcal{X}\times\mathcal{Y}$, where $\mathcal{Y}=\{1,\dots,C\}\cup\{\textsf{unknown}\}$. Deployment data follow a mixture $Q_\lambda=\sum_{j=1}^J \lambda_j P_j$ for some weight vector $\lambda\in\Delta^{J-1}$. We allow $\lambda$ to be **unknown at calibration time**, constrained only to a known convex set $\Lambda\subseteq\Delta^{J-1}$ (e.g., $\Lambda=\Delta^{J-1}$, or a box around client data fractions).

**Open-set decision.** A federated-trained classifier $\hat h$ produces a point prediction $\hat y(x)\in\{1,\dots,C\}$. A *selector* $A:\mathcal{X}\to\{0,1\}$ decides acceptance: $A(x)=1$ means "accept and act on $\hat y(x)$ as a known-class prediction"; $A(x)=0$ means "reject as unknown / abstain." $A$ may threshold any score $s(x)$ — maximum softmax probability (MSP) [39], entropy, margin, energy [40] — and the guarantee will not depend on which; such selectors descend from the classical reject option [3] and its learned form [41].

**Target risk.** The quantity to control is the **accepted selective risk** under deployment, the federated open-set analogue of the selective risk of classical selective classification [42]:
$$
R_{\mathrm{sel}}(\lambda)\;=\;\Pr_{(X,Y)\sim Q_\lambda}\!\big(\hat y(X)\ne Y \,\big|\, A(X)=1\big)
\;=\;\frac{\sum_{j}\lambda_j\, m_j}{\sum_{j}\lambda_j\, a_j},
$$
where $a_j=\Pr_{P_j}(A(X)=1)$ is the per-client acceptance rate and $m_j=\Pr_{P_j}(A(X)=1,\ \hat y(X)\ne Y)$ the accepted-error mass (per-client selective risk $r_j=m_j/a_j$, with $r_j:=0$ when $a_j=0$); the ratio form follows from the law of total probability. The companion quantity is the **accepted coverage** $C(\lambda)=\sum_j\lambda_j a_j$. **Goal:** deploy $A$ only if we can certify $R_{\mathrm{sel}}(\lambda)\le\alpha$ for the unknown deployment $\lambda\in\Lambda$ with confidence $1-\delta$, while maximizing $C(\lambda)$. The certificate outputs two numbers — a risk upper confidence bound $\bar U_\Lambda$ (deploy iff $\bar U_\Lambda\le\alpha$) and a coverage lower confidence bound $\underline C_\Lambda$ — reported as `cert_risk_ucb` and `cert_coverage_lcb`.

**Trusted calibration data and the split.** Each client holds a small *trusted, clean* calibration sample split into a **proposal** fold and a **certification** fold. The selector $A$ is chosen on the proposal fold, hence *fixed and independent* of the certification fold; because certification consumes a single fixed selector, the proposal fold may select the score family and threshold by any data-dependent rule with no multiplicity correction (assumption A2). On the certification fold, client $j$ contributes $n_j$ i.i.d. draws from $P_j$ and reports two integers:
$$
A_j=\sum_{i=1}^{n_j}\mathbf 1\{A(x_i)=1\},\qquad
K_j=\sum_{i=1}^{n_j}\mathbf 1\{A(x_i)=1,\ \hat y(x_i)\ne y_i\}.
$$
By construction $A_j\sim\mathrm{Bin}(n_j,a_j)$, $K_j\sim\mathrm{Bin}(n_j,m_j)$ with $K_j\le A_j$, and conditionally $K_j\mid A_j\sim\mathrm{Bin}(A_j,r_j)$. What leaves the client depends on the certificate (per-client pairs for the stratified one, sums for the pooled; Section 4.6). A held-out **test fold** estimates deployment behavior *after* certification and is never used to select or certify; all `test_*` quantities in Section 5 are deployment estimates, not inputs to the guarantee.

**Calibration must contain labeled unknowns.** To certify *unknown rejection*, the certification fold must include points with $Y=\textsf{unknown}$ **labeled as such**: unknown classes are unseen during training but present and labeled in this small post-training audit fold, so "distribution-free" is *with respect to the calibration distribution $Q_\lambda$*, not the entire unknown universe. In OSR benchmarks this holds by construction; in deployment it corresponds to a small audited monitoring set. Without audit unknowns the certificate degrades to a closed-set selective-risk guarantee. We certify only selectors with positive accepted coverage $a_\lambda=\sum_j\lambda_j a_j>0$.

The need for trusted labels is not an artifact of our construction; it is information-theoretically unavoidable.

**Proposition 1 (necessity of trusted calibration labels).** *No procedure that observes only the model's outputs together with corrupted or unlabeled client data can certify $R_{\mathrm{sel}}(\lambda)\le\alpha$ for $\alpha<1$, distribution-free, while deploying with nontrivial acceptance.*

*Proof sketch.* Fix an instance (World A) on which the procedure deploys with probability $p_0>0$ and whose *clean* labels agree with the model's predictions on all accepted points, so $R_{\mathrm{sel}}(\lambda)=0\le\alpha$. Construct World B by flipping the clean conditional label distribution on the acceptance region while leaving the corrupted/unlabeled observations and all model outputs unchanged — possible because the corruption process is not assumed known or invertible and unlabeled data carry no label information. In World B the accepted predictions are wrong, so $R_{\mathrm{sel}}(\lambda)=1>\alpha$; but the *distribution* of the observable input is identical, so any procedure, randomized or not, deploys in World B with the same probability $p_0$, and validity forces $p_0\le\delta$. The construction applies to every instance on which the procedure deploys, so the procedure is vacuous. Trusted clean certification labels break the indistinguishability. $\square$

**Risk-buffered proposal.** To avoid certifying a selector whose empirical risk already sits at $\alpha$ (which makes the certificate fail), the proposal fold selects $A$ subject to an empirical buffer $\widehat R_{\mathrm{prop}}(A)\le\gamma\alpha$ with $0<\gamma<1$ (candidate grid $\gamma\in\{0.2,0.3,0.5,0.7,1.0\}$), a standard safety-margin device in selective risk control.

**Notation and assumptions (collected).** Table 2 collects the core notation and the validity assumptions; each assumption is referenced where it is used.

**Table 2. Core notation and validity assumptions.**

| item | meaning / requirement |
|---|---|
| $P_j,\ \lambda,\ \Lambda$ | client distribution; deployment mixture; admissible mixture set |
| $\hat h,\ \hat y,\ s,\ A$ | federated classifier; point prediction; score; accept/reject selector |
| $a_j,\ r_j,\ m_j$ | acceptance rate; conditional selective risk; accepted-error mass |
| $n_j,\ A_j,\ K_j$ | certification-fold size; accepted count; accepted-error count |
| `cert_risk_ucb` $\bar U$ | risk upper confidence bound; deploy iff $\le\alpha$ |
| `cert_coverage_lcb` $\underline C_\Lambda$ | certified coverage lower confidence bound (Corollary 1) |
| **A1** | within-client i.i.d. certification draws from $P_j$ |
| **A2** | selector fixed on the proposal fold, independent of the certification fold |
| **A3** | trusted clean certification labels, including labeled unknowns (necessary by Proposition 1) |
| **A4** | the audited $P_j$ *is* the deployment client-conditional distribution |
| **A4′** | unknown over-representation is conservative only under accepted-error stochastic dominance; empirical stress protocol, not a theorem assumption |
| **A5** | deployment mixture lies in the declared $\Lambda$ (trivial for the full simplex) |
| **A6** | a grouped certificate certifies *group* mixtures only; exactness requires i.i.d. group-mixture audit sampling (Proposition 3) |

Two deserve emphasis. **A4/A4′:** the theorems consume an audit fold drawn from the deployment client-conditional distribution; over-representing unknowns retains validity only under stochastic dominance of the accepted-error indicator, so we use it only as an empirical stress protocol (Section 5.3), whereas under-representation is demonstrably anti-conservative. **A6:** grouping (Section 4.6) deliberately relaxes the certified object from client to group mixtures; it does not protect against arbitrary within-group shift.

---

## 4. Fed-CORE Method and Theory

The method has four ingredients: exact binomial confidence bounds (Section 4.1); the core certificates — full-simplex, bounded-$\Lambda$ with a coverage lower bound, and the reason naive pooling fails (Sections 4.2–4.4); a feasibility law for how much audit data each stratum needs before a guarantee is possible (Section 4.5); and the practical consequences — protocol, privacy, certificate variants, score agnosticism, and certified uses (Sections 4.6–4.7). All proofs are given in the body.

### 4.1 Clopper–Pearson primitives

For $K\sim\mathrm{Bin}(n,p)$, the one-sided **upper** Clopper–Pearson limit [19] at level $\varepsilon$ is
$$
U^+(K,n;\varepsilon)=\mathrm{BetaInv}\big(1-\varepsilon;\,K+1,\,n-K\big)\quad(\,=1\text{ if }K=n\,),
$$
and the one-sided **lower** limit is
$$
L^-(K,n;\varepsilon)=\mathrm{BetaInv}\big(\varepsilon;\,K,\,n-K+1\big)\quad(\,=0\text{ if }K=0\,).
$$
These satisfy, for every $p$, $\Pr\!\big(p\le U^+(K,n;\varepsilon)\big)\ge 1-\varepsilon$ and $\Pr\!\big(p\ge L^-(K,n;\varepsilon)\big)\ge 1-\varepsilon$, exactly and distribution-free.

### 4.2 Theorem 1: Conditional selective-risk certificate

In words: for each client we count the errors among its accepted audit points and use an exact binomial bound (Section 4.1) to upper-bound that client's true error rate; the global guarantee is then driven by the worst client, because no client's errors can be hidden behind another's.

The sharpest certificate works directly with the per-client **conditional** selective risk $r_j=\Pr_{P_j}(\hat y(X)\ne Y\mid A(X)=1)=m_j/a_j$. Conditional on the accepted count $A_j$, the accepted-error count is exactly $K_j\mid A_j\sim\mathrm{Bin}(A_j,r_j)$; hence a single Clopper–Pearson upper limit on the *observed accepted sub-sample* bounds $r_j$ with **no acceptance-rate slack**:
$$
\bar r_j=U^+(K_j,A_j;\varepsilon)\qquad(\bar r_j:=1\text{ if }A_j=0).
$$
The object being certified is not a collection of per-client error rates but their *acceptance-reweighted mixture*, which the following decomposition makes precise.

**Lemma 1 (acceptance-reweighted decomposition).** *For every $\lambda$ with $\sum_j\lambda_j a_j>0$,*
$$
R_{\mathrm{sel}}(\lambda)=\sum_j w_j(\lambda)\,r_j,\qquad
w_j(\lambda)=\frac{\lambda_j a_j}{\sum_\ell \lambda_\ell a_\ell},\quad \sum_j w_j(\lambda)=1 .
$$

*Proof.* By the law of total probability under $Q_\lambda$, $\Pr(\text{accept})=\sum_j\lambda_j a_j$ and $\Pr(\text{accept}\wedge\text{error})=\sum_j\lambda_j m_j=\sum_j\lambda_j a_j r_j$; divide. $\square$

Lemma 1 is elementary, but it fixes what a federated certificate must control: a convex combination of the per-client conditional risks with *unknown, acceptance-coupled* weights $w_j(\lambda)$, not a single error rate and not a maximum of unrelated intervals.

**Theorem 1 (full simplex).** *Take $\varepsilon=\delta/J$, assume within-client i.i.d. certification samples and a selector $A$ independent of the certification fold. With $\Lambda=\Delta^{J-1}$ and $\bar U_\Delta^{\,r}=\max_j \bar r_j$,*
$$
\Pr\big(R_{\mathrm{sel}}(\lambda)\le \bar U_\Delta^{\,r}\ \text{ for all }\lambda\in\Lambda\big)\ \ge\ 1-\delta .
$$

*Proof (Theorem 1).* Conditional on $A_j=a$, $K_j\sim\mathrm{Bin}(a,r_j)$, so Clopper–Pearson gives $\Pr(r_j\le\bar r_j\mid A_j{=}a)\ge1-\varepsilon$ for every $a$ (including $a{=}0$, where $\bar r_j:=1$ trivially covers $r_j$); holding for every conditioning value, the bound holds marginally by the tower property. A union over the $J$ clients with $\varepsilon=\delta/J$ gives $E=\{\forall j:\ r_j\le\bar r_j\}$ with $\Pr(E)\ge1-\delta$. On $E$, for any admissible $\lambda$, Lemma 1 gives $R_{\mathrm{sel}}(\lambda)=\sum_j w_j(\lambda)\,r_j\le\max_j\bar r_j=\bar U_\Delta^{\,r}$, a bound independent of $\lambda$ and hence simultaneous over $\Lambda$. $\square$

Theorem 1 uses **one** event per client and **no** acceptance lower bound; a mass-ratio variant that separately bounds $m_j$ and $a_j$ is valid but uniformly looser (it pays both numerator and denominator slack) and is kept only as a diagnostic baseline (Table 3). The interpretation: the **worst client sets the bar** — no client's error can be averaged away.

### 4.3 Bounded-$\Lambda$ certificate and coverage lower bound

**Theorem 2 (bounded $\Lambda$, robust certificate recommended for deployment).** *When $\Lambda$ is a known strict subset (e.g. a box around public client data fractions), worst-client domination is avoided to the extent that $\Lambda$ excludes the adversarial vertices; with $\Lambda=\Delta^{J-1}$ the bound reduces to Theorem 1. At level $\varepsilon=\delta/3J$ bound $r_j\le\bar r_j$ and $a_j\in[\underline a_j,\bar a_j]$ with $\underline a_j=L^-(A_j,n_j;\varepsilon)$, $\bar a_j=U^+(A_j,n_j;\varepsilon)$, and set*
$$
\bar U_{\Lambda}^{\,r,a}=\sup_{\lambda\in\Lambda,\ a_j\in[\underline a_j,\bar a_j]}\ \frac{\sum_j \lambda_j a_j \bar r_j}{\sum_j \lambda_j a_j}.
$$
*Then $\Pr(R_{\mathrm{sel}}(\lambda^\star)\le \bar U_\Lambda^{\,r,a})\ge1-\delta$ for the true $\lambda^\star\in\Lambda$.*

*Proof.* The $3J$ events $\{r_j\le\bar r_j\}$, $\{a_j\ge\underline a_j\}$, $\{a_j\le\bar a_j\}$ each hold with probability $\ge1-\delta/3J$ (the conditional argument of Theorem 1 for the first; Clopper–Pearson applied to $A_j\sim\mathrm{Bin}(n_j,a_j)$ for the others), so their intersection $E'$ has probability $\ge1-\delta$. On $E'$ the pair $(\lambda^\star,a^\star)$ is feasible for the supremum and, using $r_j\le\bar r_j$ in the numerator, $R_{\mathrm{sel}}(\lambda^\star)\le\bar U_\Lambda^{\,r,a}$. For fixed $\lambda$ the objective is linear-fractional in $a$ over a compact box, hence pseudolinear with its supremum at an extreme point ($a_j\in\{\underline a_j,\bar a_j\}$); the outer supremum is a small linear-fractional program (Charnes–Cooper / Dinkelbach). $\square$

*Choosing $\Lambda$ in practice.* The deployer, not the certificate, owns the declared $\Lambda$ (assumption A5). A concrete protocol: take the publicly known client data or traffic fractions $\hat\lambda$ (already exchanged for FedAvg weighting), declare the box $\Lambda=\prod_j[\hat\lambda_j(1-\rho),\ \hat\lambda_j(1+\rho)]\cap\Delta^{J-1}$ with $\rho$ set by domain knowledge about traffic drift (we use $\rho=0.15$), and fall back to the full simplex when no defensible $\hat\lambda$ exists; declaring $\Lambda$ too narrow voids the guarantee outside it. Section 5.5 sweeps both deployment knobs ($\rho$ and the proposal buffer $\gamma$) on real data.

The certificate's second output, the coverage lower bound, admits the same treatment; we state it as a corollary so that the two headline quantities are formally on equal footing.

**Corollary 1 (simultaneous coverage lower confidence bound).** *Split the budget as $\delta_r+\delta_c\le\delta$: run the risk certificate (Theorem 1 or 2) at level $\delta_r$ and bound each acceptance rate from below, $\underline a_j=L^-(A_j,n_j;\delta_c/J)$. With*
$$
\underline C_\Lambda=\inf_{\lambda\in\Lambda}\ \sum_j \lambda_j\,\underline a_j,
$$
*the events $\{\sup_{\lambda\in\Lambda}R_{\mathrm{sel}}(\lambda)\le\bar U\}$ and $\{\inf_{\lambda\in\Lambda}C(\lambda)\ge\underline C_\Lambda\}$ hold simultaneously with probability at least $1-\delta$.*

*Proof.* The $J$ additional events $\{a_j\ge\underline a_j\}$ each hold with probability $\ge1-\delta_c/J$; a union bound over all risk and coverage events costs $\delta_r+\delta_c\le\delta$. On the intersection, $C(\lambda)\ge\sum_j\lambda_j\underline a_j\ge\underline C_\Lambda$ for every $\lambda\in\Lambda$; the infimum is linear in $\lambda$, attained at a vertex. $\square$

*Implementation note.* Section 5 reports throughout at the simultaneous split $\delta_r=\delta_c=\delta/2$ of Corollary 1, so risk and coverage bounds hold *jointly* at $1-\delta$; the risk-only full-$\delta$ budget raises coverage marginally and is used only where noted.

**Edge cases.** (i) If $a_\lambda=0$ the selector accepts nothing and is **non-deployable** ($R_{\mathrm{sel}}$ undefined). (ii) If $\inf_{\lambda\in\Lambda}\sum_j\lambda_j\underline a_j=0$ (possible when some $A_j$ is small) the robust certificate is declared **infeasible** ($\bar U=+\infty$); we do *not* silently drop such a client, which would break the worst-case guarantee.

### 4.4 Why pooling is invalid

The certificate is not a corollary of existing constructions, for three separate reasons. The first is a formal statement; the failure it describes is the technical crux of the federated setting.

**Proposition 2 (naive pooling is anti-conservative under mixture shift).** *Let $U_{\mathrm{pool}}=U^+(\sum_j K_j,\sum_j A_j;\delta)$ be the single Clopper–Pearson bound applied to the pooled accepted calibration points. There exist client populations $\{(a_j,r_j)\}_{j\le J}$ and deployment mixtures $\lambda\in\Delta^{J-1}$ such that $\Pr\big(R_{\mathrm{sel}}(\lambda)>U_{\mathrm{pool}}\big)\to1$ as the per-client calibration sizes $n_j\to\infty$.*

*Proof (constructive).* Conditioned on the accepted counts, the pooled accepted-error count is a sum of independent binomials with unequal success probabilities — a Poisson-binomial — so the single-binomial Clopper–Pearson calibration does not apply. Quantitatively, take equal calibration sizes and $J{=}5$ with four low-risk clients ($a{=}0.7$, $r{=}0.02$) and one high-risk client ($a{=}0.5$, $r{=}0.3$). By the strong law, $\sum_j K_j/\sum_j A_j\to\bar r_{\mathrm{cal}}=\sum_j a_j r_j/\sum_j a_j\approx0.074$ and $U_{\mathrm{pool}}\to\bar r_{\mathrm{cal}}$, since the Clopper–Pearson width shrinks as $O(\sqrt{\log(1/\delta)/n})$. A $\lambda$ putting all deployment mass on the high-risk client has $R_{\mathrm{sel}}(\lambda)=0.3>\bar r_{\mathrm{cal}}$, so $\Pr(R_{\mathrm{sel}}(\lambda)>U_{\mathrm{pool}})\to1$; the same holds for every mixture overweighting above-average-risk clients. (Figure 2 is the finite-sample counterpart.) $\square$

The failure is one of certified target rather than of the binomial bound alone: pooled Clopper–Pearson consistently certifies the *calibration-mixture* risk — the wrong quantity when deployment overweights a high-risk client — and the Poisson-binomial structure explains why no single-binomial correction can repair it. Two further reductions fail: **(b)** federated conformal prediction controls a *quantile/coverage* of nonconformity scores under partial exchangeability, whereas Fed-CORE controls a post-selection conditional error ratio whose calibration statistic is $K_j\mid A_j$, not a score quantile; **(c)** the certificate is not a Bonferroni union of unrelated intervals, since Lemma 1 couples the per-client bounds through acceptance-reweighted convex weights and Theorem 2 is a robust linear-fractional program over $(\lambda,a)$.

### 4.5 Feasibility law

In the zero-accepted-error regime $K_j=0$ with $\Lambda=\Delta^{J-1}$, the deploy condition $\max_j\bar r_j\le\alpha$ reduces, for each binding client, to $U^+(0,A_j;\delta/J)\le\alpha$.

**Theorem 3 (feasibility law).** *In the zero-accepted-error regime, certification at level $\alpha$ over the simplex holds for a client if and only if its **observed accepted count** satisfies*
$$
U^+(0,A_j;\delta/J)\le\alpha
\quad\Longleftrightarrow\quad
A_j\ \ge\ \frac{\ln(J/\delta)}{-\ln(1-\alpha)} .
$$

*Proof.* For $K=0$, $U^+(0,A_j;\varepsilon)=1-\varepsilon^{1/A_j}$ with $\varepsilon=\delta/J$. The deploy condition $1-(\delta/J)^{1/A_j}\le\alpha$ is equivalent to $\tfrac1{A_j}\ln(\delta/J)\ge\ln(1-\alpha)$; both logarithms are negative, so dividing and flipping gives $A_j\ge\ln(J/\delta)/(-\ln(1-\alpha))$. $\square$

Because $-\ln(1-\alpha)\ge\alpha$, the threshold is of order $\ln(J/\delta)/\alpha$ (with the expected-count form $n_j a_j\gtrsim \ln(J/\delta)/\alpha$ following from concentration of $A_j$); we state it on the **observed $A_j$**, which is what the certificate consumes. This is the federated analog of the centralized $N_{\min}=\lceil\log\delta/\log(1-\alpha)\rceil$: the condition must hold *per stratum*, with a $\log J$ federation penalty, and it predicts **certified-coverage collapse** when a client is simultaneously small and high-risk. Beyond the zero-error floor, the interval-width extension (normal/Bernstein, required count $\propto(\alpha-\hat r)^{-2}$) adds that the audit count explodes when the model's own error rate $\hat r$ sits close to the target. Certification is possible only when the model is comfortably below the target and the audit set is large enough; otherwise the correct output is "cannot certify." Section 5.4 traces this law on real logits.

*The floor is fundamental, not an artifact of Clopper–Pearson (minimax sketch).* Could a less conservative certificate deploy with fewer accepted points? No: the count in Theorem 3 is a minimax lower bound for any valid procedure whose deploy decision is measurable with respect to the released counts. Fix a single stratum and a borderline-unsafe client with conditional risk $r=\alpha+\epsilon$, so that $R_{\mathrm{sel}}>\alpha$. A procedure that never deploys on $\{K=0\}$ is vacuous even for arbitrarily safe clients (its deploy probability $1-(1-r)^A\to0$ as $r\to0$); hence any non-vacuous procedure must deploy on $\{K=0\}$. But under $r=\alpha+\epsilon$, $\Pr(K{=}0\mid A)=(1-\alpha-\epsilon)^{A}$, so validity forces $(1-\alpha-\epsilon)^{A}\le\delta$, and letting $\epsilon\downarrow0$ gives $A\ge\ln(1/\delta)/(-\ln(1-\alpha))$ — Theorem 3 up to the $\ln J$ union factor. No valid certificate, however tight, escapes the $\Omega(\ln(1/\delta)/\alpha)$ per-stratum floor: the collapse of certified coverage under small, high-risk clients is information-theoretic, not a looseness of the Clopper–Pearson bound.

### 4.6 Algorithm, privacy, and certificate variants

The full protocol is summarized as Algorithm 1.

**Algorithm 1 (Fed-CORE certification).**

1. **Train.** Obtain the federated open-set model $\hat h$ and any score $s$; Fed-CORE does not modify training.
2. **Propose.** On the proposal fold, select the selector threshold subject to the risk buffer $\widehat R_{\mathrm{prop}}\le\gamma\alpha$; fix $A$.
3. **Count.** Each client evaluates $A$ on its certification fold and reports $(A_j,K_j)$ — per client for the stratified certificate, secure-aggregated within groups for the grouped variant.
4. **Certify.** The server computes $\bar r_j=U^+(K_j,A_j;\varepsilon)$ and the certificate — $\bar U_\Delta^{\,r}=\max_j\bar r_j$ (Theorem 1) or $\bar U_\Lambda^{\,r,a}$ (Theorem 2) — plus the coverage LCB $\underline C_\Lambda$ (Corollary 1).
5. **Decide.** Deploy iff $\bar U\le\alpha$ and $\underline C_\Lambda>0$; otherwise report "cannot certify," with the Theorem-3 diagnosis of whether the failure is risk- or count-driven.

The privacy footprint depends on **which** certificate is deployed: sum-only secure aggregation suffices **only for the pooled diagnostic**, while the stratified certificate requires per-client count pairs. Table 3 summarizes, for each variant, the certified target, the counts released, and the key assumption; in the table, "released" is what leaves each client, and "sec. agg." marks compatibility with secure aggregation.

**Table 3. Certificate variants: certified target, privacy, and role.**

| variant | certified target | released | sec. agg. | assumption | role |
|---|---|---|:-:|---|---|
| simplex (Thm 1) | any client mixture | client pairs | ✗ | A1–A4 | most robust |
| box-$\Lambda$ (Thm 2) | $\lambda\in\Lambda$ | client pairs | ✗ | A1–A5 | recommended |
| grouped-stratified | group mixtures | group pairs | within groups | A1–A4, A6 | privacy/feasibility |
| pooled (Rem. 1) | matched mixture | sums | ✓ | matched calibration | diagnostic only |
| mass-ratio | any client mixture | client pairs | ✗ | A1–A4 | loose diagnostic |

Because Theorem 1 needs **per-client** counts, it is *not* compatible with sum-only secure aggregation. The recommended compromise is a **grouped-stratified certificate**: partition clients into $G$ pre-declared public groups, secure-aggregate counts *within* each group, and run the certificate over the $G$ groups. The certified object weakens accordingly (A6) — robust to shift *across* groups, not within — a declared relaxation, qualitatively distinct from naive pooling (Proposition 2), stating its coarser target up front. The released statistic is lower-dimensional than the per-client score *distributions* (T-Digest) of federated conformal prediction or the quantized score *functions* of decentralized novelty detection; this is an information-flow statement, not a formal privacy guarantee. A differentially private count-release variant (calibrated noise plus correspondingly widened Clopper–Pearson levels) is a natural extension: an empirical Laplace-noise ablation (ten seeds, $G{=}2$) found the robust $\alpha{=}0.20$ certificate nearly free (certified fraction $0.90$ both without noise and at $\varepsilon{=}3$; $0.77$ at $\varepsilon{=}1$), whereas at $\alpha{=}0.10$ even $\varepsilon{=}1$ destroyed certification ($0.40\to0.001$) — the feasibility law reappearing as the binding constraint on private release. No formal differential-privacy theorem is claimed.

The grouped certificate deserves the same formal treatment as the client-level one, because merging heterogeneous clients raises exactly the Poisson-binomial concern of Proposition 2 *inside* each group. The following statement records when the group-level counts satisfy an exact conditional binomial law.

**Proposition 3 (exact validity of the grouped certificate).** *Fix a pre-declared partition of the clients into groups $g=1,\dots,G$ with within-group composition $\pi_{j\mid g}$, and let $P_g=\sum_{j\in g}\pi_{j\mid g}P_j$. If each group's certification fold is drawn i.i.d. from $P_g$, then, with $A_g$ and $K_g$ the group's accepted and accepted-error counts, $K_g\mid A_g\sim\mathrm{Bin}(A_g,r_g)$ exactly, where $r_g$ is the conditional selective risk of $P_g$. Consequently Theorems 1–2, Corollary 1, and Theorem 3 hold verbatim with groups as strata, and the certified object is $R_{\mathrm{sel}}$ over mixtures of the group distributions $\{P_g\}$ at the fixed within-group composition.*

*Proof.* A group whose audit fold is drawn i.i.d. from $P_g$ is formally a single client with distribution $P_g$: each audit point independently accepts with probability $a_g$ and, conditionally on acceptance, errs with probability $r_g$. The conditioning argument of Theorem 1 applies unchanged at the group level. $\square$

*Scope of Proposition 3.* The statement is exact only for i.i.d. group-mixture audit sampling; this exact stratum law is what separates the grouped certificate from hidden pooling. Our implementation draws per-client audit quotas, under which $K_g\mid A_g$ is conditionally Poisson-binomial and exactness is not claimed, so the grouped empirical results of Section 5 are read under A6. The gap is real when within-group risks diverge — in a controlled synthetic stress ($5{,}000$ trials per cell, $\delta{=}0.10$), quota sampling grew anti-conservative as the within-group risk spread widened (coverage $0.23$ at $n_j{=}150$ falling to $0.0002$ at $n_j{=}600$) while i.i.d.-group sampling stayed $\ge0.95$ — but three pieces of evidence bound it in our implementation: the two sampling modes differed by less than $0.002$ in coverage on the real CIFAR-10 logits ($3{,}000$ redraws each); the resampling study of Section 5.2 observed a violation rate of $8.7\times10^{-4}$ against the nominal $\delta{=}0.10$; and the full-simplex per-client positive of Section 5.5 shows that the real-data conclusions do not depend on grouping at all.

**Remark 1 (matched-mixture pooled diagnostic).** When calibration samples are drawn i.i.d. from the deployment mixture itself, the pooled bound $U^+(\sum_j K_j,\sum_j A_j;\delta)$ can serve as a diagnostic tightness reference. It is not used for any guarantee or headline metric: its validity would require Poisson-binomial-to-binomial comparison arguments (cf. Hoeffding [43]) plus the matched-mixture assumption, so we do not elevate it to a theorem.

### 4.7 Score agnosticism and certified uses

Because the guarantee in Theorems 1–2 is produced entirely by the certification split (the conditional-binomial structure of $K_j\mid A_j$ under a *fixed* selector), it holds for **any** score $s(\cdot)$: score quality affects *how much coverage* is certified, never *whether the risk is controlled*. This matters because federated heterogeneity, like label corruption, **deforms the confidence–correctness ranking** — at clients holding only minority known classes, those classes are easily confused with genuine unknowns, so any single global score is miscalibrated somewhere. Fed-CORE does not repair this deformation with the small trusted set; it *certifies around it* (made concrete on real detectors in Section 5.5).

The certificate licenses two downstream uses of the accepted set. **Use case A (safe automation/triage).** Accept $=$ act automatically; reject $=$ defer to a human. CertifiedCoverage@$\alpha$ is then exactly the **fraction of the workload safely automated at guaranteed error $\le\alpha$**, and $1-\text{coverage}$ is the human-review load. **Use case B (certified federated self-training).** Accepted predictions on unlabeled client data become pseudo-labels folded back into FedAvg; the certificate bounds their **contamination** (pseudo-label error rate $\le\alpha$ with respect to the calibration distribution), replacing the unbounded contamination of naive self-training with provably-bounded noise.

Self-training makes the round-$t$ model depend on what was accepted at round $t-1$, so **reusing one certification fold across rounds would break the required independence**. We sidestep the closed-loop problem by **data-splitting in time**: partition the trusted set into $T$ disjoint audit folds and certify the round-$t$ selector on the fresh fold $\mathcal C^{(t)}$ at level $\delta/T$.

**Proposition 4 (round-wise self-training validity).** *If $\mathcal C^{(t)}$ is independent of $(f_t,A_t)$ (guaranteed by forming $f_t,A_t$ only from folds indexed $<t$ and from unlabeled data) and each round is certified at level $\delta/T$, then*
$$
\Pr\big(\forall t\le T:\ R_{\mathrm{sel}}(A_t)\le \bar U^{(t)}\big)\ \ge\ 1-\delta .
$$

*Proof.* By construction $A_t$ is a fixed selector with respect to the fresh fold, so Theorem 1 (or 2) applies on that fold at level $\delta/T$; a union bound over the $T$ rounds gives simultaneous validity at level $\delta$. $\square$

Every injected pseudo-label batch therefore has certified contamination $\le\alpha$ simultaneously across all $T$ rounds, without a closed-loop adaptivity argument. The price is feasibility: each round's fold must clear the Theorem-3 threshold, so $T$ is bounded by the trusted-set size. Section 5.6 verifies the contract empirically.

---

## 5. Experiments

The experiments evaluate the *operating characteristics of a statistical procedure*, not open-set accuracy. Four certification claims are tested in order: **(i) validity** — certified deployments showed no held-out risk violation and the decision rule's unsafe-deployment rate stayed below $\delta$ (Sections 5.2–5.3); **(ii) non-reducibility** — pooled Clopper–Pearson failed under client-mixture shift exactly as Proposition 2 predicts (Section 5.2); **(iii) feasibility** — certified coverage appeared only past the Theorem-3 floor and collapsed along the risk-target, heterogeneity, and corruption axes (Sections 5.4–5.5); **(iv) score- and base-model agnosticism** — validity held for every score and detector while certified coverage tracked detector quality (Section 5.5). Section 5.6 evaluates the downstream self-training gate. Full per-run logs and scripts are released with the reproducibility package.

### 5.1 Experimental setup

*Data and training.* We train federated models with FedAvg [1] under non-IID Dirichlet partitions ($d\in\{0.1,0.5,5\}$; smaller $d$ is more heterogeneous), hold out classes as test-time unknowns (the standard FedOSR open-set split), and optionally corrupt client labels to connect to the corrupted-training setting. The datasets are CIFAR-10 (primary), CIFAR-100 (a second real-data positive at $\alpha{=}0.20$; Section 5.5), and a tabular FL benchmark (covtype, a feasibility-edge domain). Backbones are a CIFAR-stem ResNet-18 (GroupNorm/BatchNorm) and SimpleCNN, with WideResNet for the FedPD/FOOGD reproductions; FedAvg runs $50$ rounds, two local epochs, lr $0.01$, batch $64$. Corruption is applied to train labels only (symmetric $0.35$ / asymmetric $0.20$); calibration folds stay clean.

*Metric.* The headline metric is CertifiedCoverage@$\alpha$: the mean `cert_coverage_lcb` across seeds, credited only when the certificate deploys ($\bar U_\Lambda\le\alpha$; uncertified runs contribute zero), reported at the simultaneous budget of Corollary 1 ($\delta_r=\delta_c=\delta/2$) so that `cert_risk_ucb` and `cert_coverage_lcb` hold *jointly* at $1-\delta$. Held-out `test_coverage` is the separate deployment estimate. Four post-hoc scores — maximum softmax probability (MSP) [39], entropy, margin, and energy [40] — test the score-agnostic claim.

*Baselines and base models.* FedOSR detectors are treated as **base models**, not competitors: they carry no certificate, and Fed-CORE certifies their scores post hoc (primary FedPD [5]; second native score FOOGD [6]; FedOSS deferred as the highest-cost reproduction [38]). The nearest guarantee-bearing methods certify different functionals — prediction-set coverage for federated CP [9], batch FDR for decentralized novelty detection [18] — and are contrasted by object in Table 1; a coverage-rule recast of federated CP and the operative invalid alternative for *our* functional, pooled Clopper–Pearson, are both evaluated directly (Section 5.2), alongside our own variants as ablations. Centralized selective-risk certificates are positioned conceptually (Section 2), not run as baselines, because they assume a single exchangeable population and are not drop-in valid here [37].

*Certificate parameters.* These are parameters of the guarantee, not tuning details: $J{=}5$ clients, six known CIFAR-10 classes (the rest unknown); disjoint trusted-pool folds of $0.34/0.33/0.33$ (proposal/certification/test) with audit unknown fraction $0.30$; $\alpha\in\{0.10,0.20,0.25\}$ ($\alpha{=}0.05$ appears only in the synthetic boundary study of Table 4, where it places the true risk at the decision boundary), $\delta{=}0.10$, $\gamma\in\{0.2,0.3,0.5,0.7,1.0\}$, $\Lambda\in\{\text{simplex},\ \text{box}(\rho{=}0.15)\}$.

### 5.2 Validity and non-reducibility

This subsection establishes both validity and non-reducibility by contrasting four decision rules: a naive empirical threshold (ignores finite-sample noise), a leaked split (re-searching the threshold on the certification fold voids independence), pooled Clopper–Pearson (certifies the wrong mixture under heterogeneity), and Fed-CORE (independent split plus per-client stratification).

*Controlled synthetic study.* On synthetic clients with independently varied ground-truth risks and deployment mixtures, every property holds: empirical coverage $\ge0.98$ across heterogeneity; the tightness order box $<$ simplex $<$ mass-ratio (all valid; pooled tightest but invalid off the matched mixture); a monotone CertifiedCoverage@$\alpha$ frontier; the heterogeneity-collapse curve crossing the Theorem-3 floor ($\approx37$ accepted/client); and all four scores valid.

*Pooling is anti-conservative (non-reducibility).* The natural shortcut, pooling every client's accepted audit points and applying one Clopper–Pearson bound, failed under heterogeneity.

![](experiments/fedcore/figs/fig1_pooling_collapse.png){width=78%}

**Figure 2. Pooling certifies the wrong mixture under heterogeneity.** Empirical coverage of each certificate as the deployment mixture shifts from the calibration-matched mixture toward the high-risk client ($700$ Monte Carlo draws per mixture; $80$ for the box variant).

The population comprises four low-risk clients ($a{=}0.7$, $r{=}0.02$) and one high-risk client ($a{=}0.5$, $r{=}0.3$) at $\delta{=}0.1$. Pooled CP was valid only at the matched mixture and collapsed to $0$ under shift — certifying $\approx0.072$ while the true risk reached $0.165$–$0.30$ — because the pooled accepted-error count is Poisson-binomial. The stratified certificate (Theorem 1) stayed $\ge1-\delta$ for every mixture; the box-$\Lambda$ certificate (Theorem 2) is by design valid only inside its declared box (shaded region), so its drop outside the box does not contradict its guarantee.

*Uncertified rules are unsafe (necessity).* Table 4 collects every validity check in one place: the unsafe-deployment rate $\Pr(\text{deploy}\mid R_{\mathrm{sel}}>\alpha)$ of each alternative — which any valid method must keep $\le\delta=0.1$ — over $2{,}000$ Monte Carlo calibration draws per configuration ($J{=}5$, $n_j{=}300$, $\alpha{=}0.05$; the boundary regime places the true $R_{\mathrm{sel}}$ just above $\alpha$), together with the pooling collapse of Figure 2 and the real-logit resampling study described next.

**Table 4. Validity checks and invalid alternatives** (unsafe rates over $2{,}000$ trials per synthetic cell; FCP-recast row over $18{,}000$ audit-fold redraws on the $18$ clean ResNet runs; resampling row over $526{,}000$ evaluations on stored CIFAR logits).

| check | setting | outcome | conclusion |
|---|---|---|---|
| naive empirical threshold (deploy iff $\hat r\le\alpha$) | boundary synthetic | unsafe rate $0.49$ | invalid |
| leaked split (threshold chosen on the certification fold) | boundary synthetic | unsafe rate $0.18$ | invalid |
| pooled CP under mixture shift | synthetic, Figure 2 | coverage collapses to $0$ | invalid (Prop. 2) |
| FCP recast (accept iff singleton set, $90\%$ coverage) | real logits, $18$ runs $\times$ $1{,}000$ redraws | realized risk $>\alpha$ in $\ge 0.9997$ of redraws | wrong functional |
| **Fed-CORE (proper split)** | boundary synthetic | unsafe rate $0.00$–$0.03$ | **valid** |
| **Fed-CORE, resampling on real logits** | $526$ configs $\times$ $1{,}000$ redraws | violation rate $8.7\times10^{-4}$ (CP95 UCB $1.1\times10^{-3}$) | **valid** |

The naive threshold ignores finite-sample noise (unsafe about half the time at the boundary), and the split is load-bearing: re-searching the threshold on the certification fold inflates the unsafe rate to $18.2\%\gg\delta$, because the certificate cannot correct multiple-testing on the same fold. Only the proper split keeps the unsafe rate $\le\delta$.

*A coverage rule controls the wrong functional.* The closest guarantee-bearing alternative, federated CP, is recast as a selector by accepting exactly the points whose conformal prediction set at the standard $90\%$ closed-set coverage level is a singleton — the selective heuristic its own authors flag as guarantee-free. On the $18$ clean CIFAR-10 ResNet runs (GN/BN, $d\in\{0.5,5\}$, split-conformal on the certification fold's known-class points), this rule accepted $84\%$ of the deployment stream on average with realized accepted risk $0.225$–$0.382$, exceeding both $\alpha{=}0.10$ and $\alpha{=}0.20$ in $18/18$ runs (Table 4); under the same audit-fold resampling protocol as the validity study ($B{=}1{,}000$ redraws per run, $18{,}000$ evaluations), its realized risk exceeded $\alpha{=}0.10$ in every redraw and $\alpha{=}0.20$ in a fraction $0.9999$ of them (CP95 lower bounds $0.9998$ and $0.9997$). Nothing ties closed-set coverage to open-set accepted risk: confidently misclassified unknowns produce singleton sets and are accepted wholesale.

*Resampling validity on real logits.* Because the guarantee is a probability over the draw of the certification fold at a *fixed* model, training-seed counts alone cannot resolve a false-certificate rate; we therefore resample. For each of the 55 stored CIFAR runs, the held-out pool is the ground-truth population, per-client audit folds of the original sizes are re-drawn $B{=}1{,}000$ times, and the certificate is recomputed across $\alpha\in\{0.10,0.20\}$, $G\in\{J,2\}$, and $\gamma\in\{0.5,0.7,1.0\}$ — $526{,}000$ evaluations. Of $70{,}025$ deployments, $61$ violated the population worst-stratum risk: a violation rate of $8.7\times10^{-4}$ (CP95 upper bound $1.1\times10^{-3}$) against $\delta{=}0.10$. Tellingly, $56$ of the $61$ violations occurred at $\gamma{=}1.0$ (no risk buffer) versus $5$ across $69{,}293$ buffered deployments: the risk buffer of Section 3 is what keeps boundary configurations from consuming the failure budget. This finite-population check at fixed models does not replace the theorem or quantify training variability (Section 6).

### 5.3 Audit representativeness is a condition of validity (A4 stress test)

A4 is a condition of validity, so we test what its violation costs before reporting any certified coverage. The practically dangerous violation is an audit fold that carries unknown-class points at *less* than their deployment rate. (Over-representation behaved conservatively in all our benchmarks, consistent with the dominance heuristic of A4′, but only the matched case is covered by the theorem.)

![](experiments/fedcore/figs/ablation_unknown_prop.png){width=92%}

**Figure 3. Audit representativeness is required for validity.** (a) Synthetic study: coverage of the true deployment risk as the calibration unknown fraction varies around the deployment fraction of $0.06$ ($3{,}000$ Monte Carlo draws per point). (b) Real CIFAR-10 logits: the same collapse as the certification-fold unknown fraction is subsampled below the deployment rate of $0.30$ (five model seeds $\times$ $40$ redraws per point).

Under-representation was anti-conservative in both panels. Synthetically (Figure 3a), coverage fell to $0.522$ at a calibration fraction of $0.04$ and to $0.057$ at $0.02$; matching preserved coverage ($0.913$ at $0.06$), which is exactly the theorem condition A4, and over-representation was empirically conservative ($0.992$ at $0.08$) though theorem-covered only under the A4′ dominance condition. On the real logits (Figure 3b) the collapse reproduced: coverage $1.00$ at the matched rate down to $0.005$ at one quarter of it. It is therefore not enough that the audit fold *contains* labeled unknowns: it must carry them at no less than the deployment rate. In practice the monitoring set should track, not under-sample, the unknown-class incidence of the live stream; every validity statement below is read under A4 (with over-representation as the A4′ empirical stress protocol).

Composition matters beyond the fraction. Holding the audit unknown fraction fixed, we partitioned the unknown pool by a pre-declared difficulty proxy (MSP-score quartiles) and mismatched audit against deployment on the real logits ($2{,}000$ redraws per cell, $d\in\{0.5,5\}$). Matched composition covered at $\ge0.9996$; an easy-half audit collapsed coverage to $\approx0$ against hard-half and full-pool deployments (certified bound $0.07$–$0.08$ against true risks $0.20$–$0.32$), while a hard-half audit was conservative (coverage $1.0$). A4 therefore requires the audit to track the *composition*, not merely the rate, of the unknown stream.

### 5.4 The finite-sample feasibility law

This subsection asks why certification becomes vacuous and whether Theorem 3 predicts the transition, separating its two failure modes: *risk-driven* (the model itself is unsafe, $\hat r\ge\alpha$) and *count-driven* (the model is safe but the accepted audit count sits below the floor). *The apparent null, and its cause.* On a real CIFAR-10 ladder (12 runs, $\alpha=\delta=0.1$, five clients, small CNN), CertifiedCoverage@$0.1$ was $0$ in every run, for two distinct reasons: at extreme non-IID ($d{=}0.1$) the empirical accepted risk already exceeds $\alpha$, so no method can deploy safely and the certificate correctly declines (Mode 1); near-IID ($d{=}5$, test_risk $\approx0.08<\alpha$) the model is safe but the thin per-client accepted counts cannot drive the bound below $\alpha$ (Mode 2, the Theorem-3 collapse). Tellingly, shrinking the risk buffer $\gamma$ to lower realized risk *also* starved the accepted set (cert_n $500\to151$, $\approx30<37$/client), which *raised* the bound ($0.185\to0.222$): the binding lever is calibration budget, not the operating point. A single-run dissection confirmed the mechanism: per-client bounds $\bar r_j\in[0.139,0.190]$ all exceeded $\alpha{=}0.10$, whereas $G{=}2$ grouping roughly doubled per-stratum counts and lowered the bounds to $0.128$–$0.131$ — driven by counts, not certificate looseness.

*The staircase.* Re-aggregating the $J{=}5$ clients into $G$ public groups (the grouped-stratified certificate, Section 4.6) raises the per-group accepted count and drives the bound monotonically through $\alpha$.

![](experiments/fedcore/figs/F6_feasibility_law.png)

**Figure 4. The feasibility law (Theorem 3)** on ResNet-GN: (a) per-seed grouped bound versus minimum per-group accepted count for $G\in\{5,3,2,1\}$ at $d{=}5$; (b) CertifiedCoverage@$0.10$ by grouping; (c) coverage and pass rate versus audit budget; (d) client scaling at $J\in\{10,20\}$, CertifiedCoverage@$0.20$ versus grouping $G$ (ten seeds).

Panels (a) and (c) make one point from two directions: *merging clients into groups* and *enlarging the audit budget* are the same Theorem-3 sample-size lever. On real CIFAR-10 logits, growing the audit fold drove the mean worst-group bound from $0.58$ to $0.18$, with $\alpha{=}0.10$ becoming non-vacuous ($2/5$ seeds; CertifiedCoverage $0.061\pm0.100$) only at the largest budget (Figure 4c) — the seed variability that keeps $\alpha{=}0.10$ at the feasibility edge for this baseline detector. The risk-target axis behaves identically: under per-client grouping ($G{=}5$, box-$\Lambda$, $d{=}5$), certification became non-vacuous only at larger targets (SimpleCNN $0.063$ at $\alpha{=}0.20$, $0.193$ at $\alpha{=}0.25$; ResNet-18 $0.316$ at $\alpha{=}0.25$) — failing below not because the certificate is loose but because the accepted count is below the floor.

*Client scaling.* The same lever governs the certificate as the federation grows. On real CIFAR-10 (ResNet-GN, $d{=}0.5$, ten seeds) scaled to $J\in\{10,20\}$ (Figure 4d), the per-client certificate ($G{=}J$) at $\alpha{=}0.20$ was vacuous on every seed at both scales, because splitting a fixed audit budget across more clients drives each per-client accepted count below the floor; coarsening the grouping restored certification monotonically (at $J{=}20$: $0\to0.045$ at $G{=}5$, $3/10$, $\to0.158$ at $G{=}2$, $6/10$; at $J{=}10$: $0\to0.089$ at $G{=}5$, $4/10$, $\to0.177$ at $G{=}2$, $6/10$). This is the feasibility law at federated scale, with the declared grouping of Section 4.6 — not hidden pooling — recovering a non-vacuous guarantee; $\alpha{=}0.10$ stayed at the feasibility edge throughout ($\le 2/10$). A synthetic extension to $J{=}50$ reproduced the pattern beyond the trained range: at $G{=}J$ the expected per-group accepted count fell below the floor ($14.4$ versus $59$) and nothing certified, whereas $G{=}2$ restored coverage to $\approx0.52$ essentially independently of $J$.

### 5.5 Real detectors and certified coverage

Table 5 collects the real-data certification results: the FedAvg+MSP baseline, the full FedOSR detectors (FedPD-PROSER, FOOGD), and the edge/negative settings. The strongest positive is FedPD-PROSER, which certifies at the hard target $\alpha{=}0.10$ where the baseline cannot.

*Purpose.* These experiments ask three questions of the certificate on real federated logits: is certified accepted coverage non-vacuous at all; does it track detector quality at fixed validity (the score-agnostic claim made concrete); and is the grouped headline anything more than hidden pooling. One reading rule, stated once: $\alpha{=}0.20$ is a finite-sample feasibility-demonstration target, not a recommended safety threshold, and the certified coverage is the largest fraction a label-honest procedure can *prove* safe — the alternative to a certified $0.27$ is not a higher guaranteed number but *zero* guaranteed coverage. Where coverage collapses ($\alpha{=}0.10$, extreme non-IID, corruption), the information-theoretic converse of Section 4.5 shows the collapse is fundamental to any valid certificate, not a looseness of ours.

*Headline.* Unless noted, this subsection uses the grouped ($G{=}2$) certificate, whose certified object is the **group-mixture** $R_{\mathrm{sel}}$ over two public groups under assumption A6 and Proposition 3 — a declared relaxation of the client-simplex guarantee (Section 4.6), not a rediscovery of pooling; the full-simplex per-client certificate reported at the end of this subsection certifies $5/5$ seeds with no grouping, so the grouped form is an audit-budget device, not the source of the positives. At $\alpha{=}0.20$ the GroupNorm certificate was non-vacuous over ten seeds — CertifiedCoverage $0.269\pm0.200$ at $d{=}5$ ($8/10$) and $0.273\pm0.171$ at $d{=}0.5$ ($9/10$), with no held-out violation among certified runs (Table 5); at $\alpha{=}0.10$ it was seed-variable ($2/10$ at both levels, secondary). BatchNorm attained the highest baseline coverage ($0.393\pm0.088$, $10/10$), but GroupNorm remains the headline as the principled FL normalization (BatchNorm's running statistics diverge under non-IID FedAvg); validity holds under both. The edge and negative cells (lower Table 5) are where the feasibility law predicts collapse: extreme non-IID, corruption, or a backbone whose $\hat r$ is near $\alpha$.

**Table 5. Real-data certification diagnostics** (CIFAR-10 unless noted; grouped $G{=}2$ certificate under A6): (a) FedAvg+MSP baseline (ten seeds), (b) real FedOSR detectors (five seeds; FOOGD-SAG single seed), (c) edge and negative settings. *Notes.* CertCov = CertifiedCoverage@$\alpha$; "cert." = certified seeds; GN/BN = ResNet-18 GroupNorm/BatchNorm. In block (a) (cert\_frac $0.5$), $\bar U$, $A_g$, $K_g$ (worst-group risk bound, min per-group accepted count, worst-group accepted-error count) are medians over certified seeds; test risk/coverage are means; $\gamma$ reports proposal-fold selections. The $\alpha{=}0.20$ median $A_g$ of $539$–$696$ sits far above the Theorem-3 floor, whereas $\alpha{=}0.10$ certifies with roughly half the accepted mass. Block (c) tags each vacuous cell as risk-driven ($\hat r$ near or above $\alpha$) or count-driven (below the floor). Coverage uses the simultaneous $\delta/2$ budget; a risk-only full-$\delta$ budget moves the boundary cell (GN $d{=}5$, $\alpha{=}0.20$) from $0.269$ ($8/10$) to $0.305$ ($9/10$), all other cells by $\le0.003$, and drops one marginal FOOGD seed ($0.089\to0.067$). Per-run values are released with the code.

(a) FedAvg+MSP baseline

| model | $d$ | $\alpha$ | cert. | CertCov | med. $\bar U$ | med. $A_g$ | med. $K_g$ | $\gamma$ | test risk | test cov. |
|---|:-:|:-:|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| GN | 5 | 0.20 | 8/10 | $\mathbf{0.269\pm0.200}$ | 0.158 | 677 | 123 | 0.5–0.7 | 0.114 | 0.373 |
| GN | 5 | 0.10 | 2/10 | $0.037\pm0.079$ | 0.086 | 358.5 | 32.5 | 0.3–0.5 | 0.043 | 0.214 |
| GN | 0.5 | 0.20 | 9/10 | $\mathbf{0.273\pm0.171}$ | 0.160 | 539 | 87 | 0.5–0.7 | 0.103 | 0.339 |
| GN | 0.5 | 0.10 | 2/10 | $0.027\pm0.057$ | 0.067 | 259 | 14 | 0.3 | 0.020 | 0.164 |
| BN | 5 | 0.20 | 10/10 | $0.393\pm0.088$ | 0.165 | 695.5 | 138.5 | 0.5–0.7 | 0.117 | 0.431 |
| BN | 5 | 0.10 | 6/10 | $0.086\pm0.103$ | 0.081 | 281 | 13 | 0.2–0.5 | 0.034 | 0.171 |

(b) Real FedOSR detectors, native scores

| base model | $d$ | AUROC | $\alpha$ | cert. | CertCov | test risk |
|---|:-:|:-:|:-:|:-:|---|:-:|
| FedPD–PROSER | 5 | 0.81 | 0.20 | 5/5 | $\mathbf{0.532\pm0.099}$ | 0.117 |
| FedPD–PROSER | 5 | 0.81 | 0.10 | 4/5 | $0.250\pm0.134$ | 0.039 |
| FedPD–PROSER | 0.5 | 0.79 | 0.20 | 5/5 | $0.464\pm0.084$ | 0.114 |
| FedPD–PROSER | 0.5 | 0.79 | 0.10 | 5/5 | $\mathbf{0.240\pm0.082}$ | 0.040 |
| FedAvg+MSP (control) | 5 | 0.74 | 0.20 | 5/5 | $0.404\pm0.090$ | 0.119 |
| FOOGD–SM3D | 5 | 0.68 | 0.20 | 4/5 | $0.089\pm0.088$ | 0.118 |
| FOOGD–SAG | 5 | 0.47 | 0.20 | 0/1 | $0$ | — |

(c) Edge and negative settings

| setting | CertCov | failure mode | reason it does not certify |
|---|:-:|:-:|---|
| ResNet, $d{=}0.1$, clean | $0$ | count-driven | extreme non-IID; below Theorem-3 floor |
| ResNet, symmetric $0.35$ | $0$ | risk-driven | corruption: $\hat r>\alpha$ (test risk $0.167$ at $d{=}0.5$) |
| CIFAR-10 SimpleCNN, $d{=}5$ ($\alpha{=}0.20$) | $0.063$ | risk-driven | looser backbone, higher $\hat r$ |
| covtype MLP, fixed MSP | $0$ (0/5) | risk-driven | $\hat r$ near $\alpha$; feasibility edge, see text |

covtype is reported as a second domain at the feasibility edge, **not** a positive: fixed-MSP certified $0/5$ at $\alpha{=}0.20$, and a procedurally valid protocol (proposal-fold score selection, enlarged audit budget, $G{=}2$, ten seeds) crossed from vacuous toward non-vacuous along the risk-target axis ($0.036$ at $\alpha{=}0.20$, $0.075$ at $0.25$, $0.154$ at $0.30$) without reaching a stable positive — exactly the feasibility-law behavior for a backbone whose risk sits near the target.

*A second positive dataset: CIFAR-100.* A multi-model CIFAR-100 grid (sixty known classes; GN/BN/CNN; $d\in\{0.5,5\}$; ten seeds each; Table 6) tested the certificate beyond CIFAR-10 and a single backbone. At $\alpha{=}0.20$ the grouped certificate was non-vacuous in all six backbone-by-heterogeneity cells ($0.05$–$0.09$, $7$–$8/10$ seeds) with test risk below target throughout — a second real-data positive, with coverage small in absolute terms because a sixty-class model's accepted-error rate leaves little safe margin. As on CIFAR-10, $\alpha{=}0.10$ stayed at the feasibility edge ($3/60$). The strong-detector effect transferred: FedPD-PROSER's native score reached $0.167\pm0.145$ ($2/3$) at $\alpha{=}0.20$, roughly double the best baseline cell, while $\alpha{=}0.10$ remained infeasible.

**Table 6. CIFAR-100 multi-model certification** (grouped $G{=}2$; ten seeds per baseline cell, three seeds for the FedPD-PROSER row; per-seed rows are released with the reproducibility package).

| backbone | $d$ | cert. @0.20 | CertCov @0.20 | test risk @0.20 | cert. @0.10 | CertCov @0.10 |
|---|:-:|:-:|---|:-:|:-:|---|
| GN | 0.5 | 8/10 | $0.046\pm0.033$ | 0.084 | 0/10 | $0$ |
| GN | 5 | 8/10 | $0.056\pm0.040$ | 0.094 | 0/10 | $0$ |
| BN | 0.5 | 7/10 | $0.074\pm0.055$ | 0.101 | 1/10 | $0.003\pm0.011$ |
| BN | 5 | 8/10 | $0.093\pm0.059$ | 0.103 | 1/10 | $0.003\pm0.010$ |
| CNN | 0.5 | 7/10 | $0.052\pm0.046$ | 0.090 | 0/10 | $0$ |
| CNN | 5 | 8/10 | $0.070\pm0.050$ | 0.106 | 1/10 | $0.003\pm0.010$ |
| FedPD-PROSER | 5 | 2/3 | $\mathbf{0.167\pm0.145}$ | 0.110 | 0/3 | $0$ |

*Unknown-split robustness.* The primary protocol draws a fresh known/unknown partition per training seed, so Table 5's seed variability already contains open-set-split variability. Fixing the partition isolated its effect (GN, $d{=}0.5$, six pre-declared splits, ten seeds each, $G{=}2$): CertifiedCoverage@$0.20$ ranged from $0.077$ ($5/10$) to $0.388\pm0.033$ ($10/10$), and the cross-split standard deviation of cell means ($0.115$) exceeded the mean within-split seed variability ($0.095$). Which classes play the unknowns therefore drives certified coverage at least as strongly as training randomness, and the wide per-seed range of the primary cell is largely split variance; the hard split's failures were risk-driven (median bound $0.196$ against $\alpha{=}0.20$). The certificate reports this honestly — certifying less on harder partitions rather than violating the target.

*Deployment knobs.* The two choices a deployer owns — box radius $\rho$ (Section 4.3) and proposal buffer $\gamma$ (Section 3) — were swept on the ResNet-GN $d{=}5$ logits at $\alpha{=}0.20$. Widening the box from $\rho{=}0.05$ to the full simplex barely moved certified coverage ($0.394\to0.386$, $5/5$ seeds throughout), whereas removing the buffer ($\gamma{=}1.0$) collapsed certification to $1/5$ seeds where $\gamma{=}0.7$ kept $5/5$, with validity holding at every setting. The buffer, not the box, is the operative deployment choice — consistent with the resampling finding that $56$ of $61$ violations sat at $\gamma{=}1.0$ (Section 5.2). The guarantee level itself is likewise not a hidden tuning knob: sweeping $\delta\in\{0.05,0.10,0.20\}$ on the same logits degraded coverage gracefully (GN, $d{=}5$, $\alpha{=}0.20$: $0.319/0.341/0.392$), with validity holding throughout.

*Cost of mixture robustness (diagnostic).* With no prior method certifying this object, two references bracket what is achievable (GN $d{=}5$ logits, $\alpha{=}0.20$, five feasible seeds; we read the *gaps*, not the absolute levels): Fed-CORE $0.392$ (no test labels), the invalid matched-mixture pooled diagnostic $0.403$, and a test-peeking oracle $0.626$ (no guarantee). Dropping the invalid pooling assumption cost only $0.011$ of coverage; the remaining gap to the oracle is the price of label honesty. Fed-CORE was the only label-honest, safe, finite-sample option.

*Stress axes.* Beyond the risk target $\alpha$, the feasibility law has two further axes, heterogeneity and corruption, shown together in Figure 5; each pushes $\hat r$ past $\alpha$ or starves the per-group count below the Theorem-3 floor.

![](experiments/fedcore/figs/F7_hetero_collapse.png){width=92%}

**Figure 5. Stress axes of the feasibility law.** (a) Best grouped certified-risk bound ($G{=}2$, $\alpha{=}0.10$) per seed versus Dirichlet concentration $d$ (ResNet-18, stored logits; horizontal marks are medians). (b) Ten-seed mean grouped CertifiedCoverage@$0.20$ versus training-label noise rate at $d\in\{0.5,5\}$, symmetric (solid) and asymmetric (dashed) noise (bands: $\pm1$ sd); the rate-$0$ anchors are the clean Table-5 headline cells.

As $d$ fell (more non-IID), per-group accepted counts thinned and the grouped bound rose monotonically, sitting near $\alpha$ at $d{=}5$, straddling it at $d{=}0.5$, and moving far above it ($\approx0.30$) at the extreme $d{=}0.1$; the ResNet-GroupNorm headline cells of Table 5 correspond to the feasible end of this axis. On the corruption axis, a ten-seed sweep confirmed the collapse: relative to the clean headline (Table 5), symmetric training-label noise drove the grouped CertifiedCoverage@$0.20$ to a marginal $\approx0.03$ at rate $0.1$ and to $0$ at rates $\ge0.2$, at both $d\in\{0.5,5\}$, *although the calibration fold stayed clean*; a matched ten-seed asymmetric sweep collapsed identically ($0.014$ at rate $0.1$, $\le0.003$ at $\ge0.2$ marginally over $d$, with a single certified seed at rate $0.2$, $d{=}0.5$; Figure 5b). Corruption raises the model's $\hat r$ above $\alpha$, so a clean audit set has nothing safe left to certify.

*Score-agnosticism.* The guarantee held for **every** score: across MSP, entropy, margin, and energy the realized test_risk stayed $\le\alpha$ ($0.042$–$0.046$) while certified coverage varied — the score changed only *how much* was certified, confirming that validity comes from the certification split (Section 4.7), a point made concrete on real detectors next.

*Two real FedOSR detectors.* To answer the concern that the baseline uses MSP on a FedAvg backbone rather than a genuine FedOSR detector, we certify the native open-set scores of two real FedOSR methods: FedPD's PROSER dummy-vs-known score (a full reproduction, pretrain-then-fine-tune on WideResNet-28-10, AUROC $0.81$ over five seeds) and FOOGD's SM3D score (a representative head on the shared backbone); the middle block of Table 5 reports both, alongside the FedAvg+MSP control and a faithfully reproduced full FOOGD-SAG. FedPD-PROSER was the only detector that certified robustly at the hard target $\alpha{=}0.10$, and the result *strengthened* under heterogeneity ($5/5$ seeds at both targets at $d{=}0.5$): a strong detector plus the certificate reached risk targets the baseline could not.

*The thesis across base models.* Certified coverage tracked the native-score AUROC (FedPD $0.81$ > MSP $0.74$ > FOOGD-representative $0.68$ > FOOGD-SAG $0.47$) while **validity held in every certified cell**: coverage follows detector quality, validity is independent of it. Figure 6 makes both halves of the thesis visual: panel (b) plots the monotone AUROC-to-certified-coverage relation across all base models, and panel (a) shows that the strong detector dominates the baselines across the *entire* risk-target frontier — certifying non-trivially already at $\alpha{=}0.05$, where the baselines are vacuous.

![](experiments/fedcore/figs/F8_frontier_detectors.png){width=92%}

**Figure 6. Certified coverage across risk targets and detectors** ($d{=}5$, grouped $G{=}2$, best-$\gamma$; default audit folds, so absolute levels are not directly comparable to the cert\_frac-$0.5$ block of Table 5). (a) CertifiedCoverage@$\alpha$ versus the risk target (mean $\pm$ sd over seeds): the MSP baselines become non-vacuous between $\alpha{=}0.05$ and $0.10$, while FedPD-PROSER's native score certifies already at $\alpha{=}0.05$ and dominates at every target. (b) CertifiedCoverage@$0.20$ versus native-score AUROC (Table 5b aggregates; filled $=d{=}5$, open $=d{=}0.5$): coverage rises monotonically with detector quality while validity is fixed by the certificate.

The FedAvg+MSP control ($0.404$ at $\alpha{=}0.20$) sat between the GN and BN baseline cells, validating the harness. On reproducibility: FedPD-PROSER required the pretrain-then-fine-tune recipe (from scratch it did not converge), full FOOGD-SAG reached only chance-level AUROC $0.467$ on our split at a single-GPU budget (single-seed negative), and FedOSS (a medical-imaging codebase without a CIFAR loader) was deferred; we therefore claim full compatibility with FedPD-PROSER and representative compatibility with FOOGD-SM3D, not broad coverage over all FedOSR methods.

*A full-simplex positive rules out hidden pooling.* The full-simplex certificate (Theorem 1, per-client, $G{=}J$) settles the worry that grouping is pooling by another name: on FedPD-PROSER at $J{=}5$, $\alpha{=}0.20$ with an enlarged audit budget (cert\_frac $0.5$), it certified $5/5$ seeds at $d{=}5$ (mean coverage $0.390$, per-client bounds $\le0.18$, accepted counts $156$–$455$, all past the floor) and $5/5$ at $d{=}0.5$ (mean coverage $0.348$), whereas $\alpha{=}0.10$ certified $2/5$ and $1/5$ seeds — the measured price of worst-client robustness at a small per-client audit budget. This is a genuine client-simplex guarantee with no grouping relaxation; the real-data positives are not an artifact of assumption A6.

### 5.6 Downstream use: certified pseudo-label admission

The first downstream use, safe automation, is CertifiedCoverage@$\alpha$ itself (Section 4.7). The second is **a certified admission gate, not an accuracy booster**: accepted predictions are folded back into FedAvg as pseudo-labels only when their contamination certifies below the target. On real CIFAR self-training (ResNet-GN, $d{=}5$), naive self-training kept injecting pseudo-labels with realized error far above target ($0.19$–$0.67$ and growing), whereas the certified procedure found the first round Theorem-3-infeasible and **halted, admitting nothing** — the safe outcome the feasibility law predicts. The Proposition-4 contract was verified (Table 7): over $20{,}000$ Monte Carlo trials, the simultaneous unsafe rate was $0.086\le\delta$ with the round-wise fresh-fold split versus $0.386>\delta$ when reusing one fold across adaptive rounds — round-wise certification is necessary.

**Table 7. Certified pseudo-label admission.**

| scheme | fresh fold / round | contaminated batches admitted | simult. unsafe rate | valid? |
|---|:-:|---|:-:|:-:|
| Fed-CORE (round-wise split, $\delta/T$) | yes | $0$ | 0.086 | ✓ |
| reused fold ($\delta$ per round) | no | — | 0.386 | ✗ |
| naive self-training (no certificate) | — | every round (contam. $0.19$–$0.67$) | — | no guarantee |

*A supporting descriptive result.* With the stronger FedPD-PROSER base and a $4\times$ audit budget, certified admission additionally yielded a descriptive known-accuracy gain ($+0.031\pm0.019$, $4/5$ seeds positive, against a clean-pseudo-label oracle of $+0.043$; $0/5$ contamination violations, maximum realized contamination $0.149\le\alpha$). The gain is seed-variable and is not the guarantee — the guarantee is the contamination bound on each admitted batch. Self-training is therefore not evidence that Fed-CORE improves accuracy; it is evidence that the certificate prevents unsafe pseudo-label ingestion.

## 6. Limitations

**Conservatism.** The stratified certificate is conservative (Clopper–Pearson exactness, union bound, worst-case mixture); box-$\Lambda$ and the pooled diagnostic recover tightness but require, respectively, knowledge of $\Lambda$ and matched-mixture calibration. The guarantee is marginal over $Q_\lambda$, not conditional per client.

**Trusted calibration (A3–A4).** Certifying unknown rejection requires labeled unknown-class points in an audit fold drawn from the deployment client-conditional distribution (Proposition 1; A4). Under-representing unknowns is anti-conservative (Section 5.3); over-representation behaved conservatively in our benchmarks but is theorem-covered only under the A4′ dominance condition. "Distribution-free" therefore means with respect to the calibration distribution $Q_\lambda$, not the entire unknown universe; in deployment the audit fold is a small audited monitoring set that must track the live unknown incidence, and Theorem 3 quantifies how scarce it may be before certification becomes infeasible.

**Privacy and grouping.** Only the pooled diagnostic is sum-only; the stratified certificate needs per-client (or per-group) counts, and the grouped variant weakens the certified target to group mixtures (A6) — a one-group certificate is not a client-simplex guarantee. Exactness of the grouped certificate additionally requires the sampling condition of Proposition 3; under our per-client audit quotas the group counts are conditionally Poisson-binomial, and the grouped results rest on the resampling evidence of Section 5.2 and the full-simplex positive of Section 5.5 rather than an exactness theorem. The privacy claim is an information-flow statement; a differentially private count-release variant is left as future work.

**Self-training use case (B).** The certificate guarantees the *contamination* of each injected pseudo-label batch, **not** that self-training improves accuracy; the gain is an empirical, seed-variable claim (Section 5.6). Round-wise fold splitting trades feasibility for adaptivity ($T$ capped by the trusted-set size via Theorem 3); a reused-fold scheme with formal closed-loop validity is left as future work.

**External validity.** The positive evidence is confined to cross-silo scales ($J\le20$ trained, $J{=}50$ synthetic), vision benchmarks plus one tabular domain, and pre-declared contiguous groupings. Certified coverage also depends strongly on which classes play the unknowns (the unknown-split study of Section 5.5: cross-split variation exceeded seed variation), so absolute coverage numbers should not be expected to transfer across open-set splits; only the validity guarantee transfers.

**Statistical resolution of the validity evidence.** With five training seeds per cell, "0 held-out violations" alone bounds the per-cell violation rate only loosely; the validity evidence in this paper instead rests on the resampling study of Section 5.2, which evaluates the certificate over $526{,}000$ audit-fold redraws on the real logits (violation rate $8.7\times10^{-4}$, CP95 upper bound $1.1\times10^{-3}$ against $\delta{=}0.10$), together with the synthetic Monte Carlo coverage ($\ge0.98$ against the $0.90$ target). What resampling at a fixed model cannot capture is variability over training itself; the per-cell seed counts (Table 5) remain the evidence at that level.

---

## 7. Conclusion

Fed-CORE certifies the accepted selective risk of a federated open-set classifier with a finite-sample, distribution-free guarantee under heterogeneity and unknown deployment mixtures. Its core is a **stratified conditional selective-risk certificate** — per-client conditional-binomial CP limits with a robust bounded-$\Lambda$ form and a coverage lower bound — valid where naive pooling fails, accompanied by a per-stratum feasibility law. The framework recasts federated open-set recognition from a ranking problem into a *certification* problem: a small trusted audit set certifies which predictions are safe to accept, and those predictions are then used for guaranteed-risk automation and contamination-bounded self-training.

Empirically, under the stated assumptions (A1–A6) no certified deployment exhibited a held-out risk violation in any seed, dataset, or normalization. Fed-CORE obtained non-vacuous grouped certificates at $\alpha{=}0.20$ on CIFAR-10 and CIFAR-100, a full-simplex per-client positive on FedPD-PROSER ($5/5$ seeds, mean coverage $0.39$) that rules out grouping artifacts, and pooled-CP collapse under mixture shift exactly as Proposition 2 predicts, while covtype marked the feasibility edge. Certified coverage was governed throughout by the **feasibility law** along its three axes — risk target, heterogeneity, and corruption — and certifying a strong detector's native score converted detector quality directly into certified coverage at fixed validity. The contribution is therefore the *object and its finite-sample certificate*, the exposure of pooled invalidity under heterogeneity, and the characterization of *when* certified open-set deployment is feasible; it is not a new FedOSR algorithm and claims no raw-accuracy gain.

Three directions follow from the limitations. First, reducing the conservatism of the stratified certificate without losing mixture robustness remains open; the matched-mixture pooled diagnostic (Remark 1) marks the available tightness, but a valid general tightening requires new arguments. Second, the $(\alpha-\hat r)^{-2}$ growth of the Theorem-3 threshold keeps $\alpha=0.10$ at the feasibility edge; variance-adaptive bounds of the Bernstein or betting type should bring small risk targets into the feasible regime at realistic audit budgets. Third, the grouped certificate releases only per-group counts, making a differentially private count-release certificate a concrete next step toward a formal privacy guarantee.

---

## CRediT authorship contribution statement

Sanghoon Kim: Conceptualization, Methodology, Formal analysis, Software, Investigation, Writing – original draft, Writing – review and editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

This work was supported by [funding source to be completed].

## Data availability

The CIFAR-10 and CIFAR-100 datasets are publicly available. The code and the derived calibration counts used to reproduce the certificates will be made available on publication.

---

## References

[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas, Communication-efficient learning of deep networks from decentralized data, in: AISTATS, 2017.

[2] P. Kairouz, H.B. McMahan, B. Avent, et al., Advances and open problems in federated learning, Found. Trends Mach. Learn. 14 (1–2) (2021) 1–210.

[3] C.K. Chow, On optimum recognition error and reject tradeoff, IEEE Trans. Inf. Theory 16 (1) (1970) 41–46.

[4] W.J. Scheirer, A. de Rezende Rocha, A. Sapkota, T.E. Boult, Toward open set recognition, IEEE Trans. Pattern Anal. Mach. Intell. 35 (7) (2013) 1757–1772.

[5] C. Yang, M. Zhu, Y. Liu, Y. Yuan, FedPD: Federated open set recognition with parameter disentanglement, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[6] X. Liao, W. Liu, P. Zhou, F. Yu, J. Xu, J. Wang, W. Wang, C. Chen, X. Zheng, FOOGD: Federated collaboration for both out-of-distribution generalization and detection, in: Advances in Neural Information Processing Systems (NeurIPS), 2024. arXiv:2410.11397.

[7] Y. Diao, Q. Li, B. He, Towards addressing label skews in one-shot federated learning, in: International Conference on Learning Representations (ICLR), 2023.

[8] L. Wang, C. Liu, J. Guo, J. Dong, X. Wang, H. Huang, Q. Zhu, Federated continual novel class learning, arXiv:2312.13500, 2023.

[9] C. Lu, Y. Yu, S.P. Karimireddy, M.I. Jordan, R. Raskar, Federated conformal predictors for distributed uncertainty quantification, in: International Conference on Machine Learning (ICML), 2023. arXiv:2305.17564.

[10] V. Plassier, M. Makni, A. Rubashevskii, E. Moulines, M. Panov, Conformal prediction for federated uncertainty quantification under label shift, in: International Conference on Machine Learning (ICML), 2023. arXiv:2306.05131.

[11] M. Kang, Z. Lin, J. Sun, C. Xiao, B. Li, Certifiably Byzantine-robust federated conformal prediction, in: International Conference on Machine Learning (ICML), 2024. arXiv:2406.01960.

[12] S. Bates, E. Candès, L. Lei, Y. Romano, M. Sesia, Testing for outliers with conformal p-values, Ann. Statist. 51 (1) (2023) 149–178.

[13] T. Xie, Y. Zhou, Z. Liang, S. Favaro, M. Sesia, Conformal inference for open-set and imbalanced classification, arXiv:2510.13037, 2025.

[14] J. Hallberg Szabadváry, T. Löfström, U. Johansson, C. Sönströd, E. Ahlberg, L. Carlsson, Classification with reject option: Distribution-free error guarantees via conformal prediction, Mach. Learn. Appl. 20 (2025).

[15] A.N. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster, Conformal risk control, in: ICLR, 2024, arXiv:2208.02814.

[16] Y. Xu, W. Guo, Z. Wei, Selective conformal risk control, arXiv:2512.12844, 2025.

[17] T. Bai, Y. Jin, Conformal selective prediction with general risk control, arXiv:2603.24704, 2026.

[18] K. Loh, Y. Xiang, Decentralized conformal novelty detection via quantized model exchange, arXiv:2605.08263, 2026.

[19] C.J. Clopper, E.S. Pearson, The use of confidence or fiducial limits illustrated in the case of the binomial, Biometrika 26 (1934) 404–413.

[20] C. Yang, M. Zhu, Y. Liu, Y. Yuan, FedPD++: Enhanced federated open-set recognition with parameter disentanglement, Int. J. Comput. Vis. (2026). https://doi.org/10.1007/s11263-026-02861-9

[21] J. Zhang, X. Ma, S. Guo, W. Xu, Towards unbiased training in federated open-world semi-supervised learning, in: International Conference on Machine Learning (ICML), 2023.

[22] S. Yu, J. Hong, H. Wang, Z. Wang, J. Zhou, Turning the curse of heterogeneity in federated learning into a blessing for out-of-distribution detection, in: International Conference on Learning Representations (ICLR), 2023.

[23] H. Gao, Y. Liu, Z. Qin, W. Ou, Noise-resistant federated open set recognition, in: Knowledge Science, Engineering and Management (KSEM 2025), Lecture Notes in Computer Science, vol. 15920, Springer, Singapore, 2026, pp. 1–13.

[24] L. Zhang, M. Wan, P. Huang, G. Yang, Adversarial compact wrapping classifier learning for open set recognition, Inf. Sci. 680 (2024).

[25] Z. Dai, G. Shen, H. Yuan, S. Zheng, Y. Hu, J. Du, X. Kong, F. Xia, Towards heterogeneous federated graph learning via structural entropy and prototype aggregation, Inf. Sci. 718 (2025) 122338.

[26] M. Ren, Z. Wang, X. Yu, Personalized federated learning: A clustered distributed co-meta-learning approach, Inf. Sci. 647 (2023) 119499.

[27] Z. Pan, C. Li, F. Yu, S. Wang, X. Tang, J. Zhao, Balancing the trade-off between global and personalized performance in federated learning, Inf. Sci. 712 (2025) 122154.

[28] X. Yu, Z. Liu, W. Wang, Y. Sun, Clustered federated learning based on nonconvex pairwise fusion, Inf. Sci. 678 (2024) 120956.

[29] X. Li, S. Zhao, C. Chen, Z. Zheng, Heterogeneity-aware fair federated learning, Inf. Sci. 619 (2023) 968–986.

[30] F. Kamiran, S. Mansha, A. Karim, X. Zhang, Exploiting reject option in classification for social discrimination control, Inf. Sci. 425 (2018) 18–33.

[31] H. Yang, W. Xi, Z. Wang, Y. Shen, X. Ji, C. Sun, J. Zhao, FedRich: Towards efficient federated learning for heterogeneous clients using heuristic scheduling, Inf. Sci. 645 (2023) 119360.

[32] X. Zhou, G. Yang, Communication-efficient and privacy-preserving large-scale federated learning counteracting heterogeneity, Inf. Sci. 661 (2024) 120167.

[33] X. Du, J. Yu, Z. Chu, L. Jin, J. Chen, Graph autoencoder-based unsupervised outlier detection, Inf. Sci. 608 (2022) 532–550.

[34] R.A. Coelho, L.C.B. Torres, C.L. de Castro, Concept drift detection with quadtree-based spatial mapping of streaming data, Inf. Sci. 625 (2023) 578–592.

[35] A. Farinhas, C. Zerva, D. Ulmer, A.F.T. Martins, Non-exchangeable conformal risk control, in: International Conference on Learning Representations (ICLR), 2024. arXiv:2310.01262.

[36] S. Bates, A.N. Angelopoulos, L. Lei, J. Malik, M.I. Jordan, Distribution-free, risk-controlling prediction sets, J. ACM 68 (2021) 1–34.

[37] X. Yu, J. Liu, A joint finite-sample certificate for adaptive selective conformal risk control, arXiv:2606.08517, 2026.

[38] M. Zhu, J. Liao, J. Liu, Y. Yuan, FedOSS: Federated open set recognition via inter-client discrepancy and collaboration, IEEE Trans. Med. Imaging 43 (1) (2024) 190–202.

[39] D. Hendrycks, K. Gimpel, A baseline for detecting misclassified and out-of-distribution examples in neural networks, in: ICLR, 2017.

[40] W. Liu, X. Wang, J.D. Owens, Y. Li, Energy-based out-of-distribution detection, in: NeurIPS, 2020.

[41] Y. Geifman, R. El-Yaniv, SelectiveNet: A deep neural network with an integrated reject option, in: ICML, 2019.

[42] R. El-Yaniv, Y. Wiener, On the foundations of noise-free selective classification, J. Mach. Learn. Res. 11 (2010) 1605–1641.

[43] W. Hoeffding, On the distribution of the number of successes in independent trials, Ann. Math. Statist. 27 (1956) 713–721.
