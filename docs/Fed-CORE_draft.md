# Fed-CORE: Federated Certified Open-Set Recognition via Selective&nbsp;Risk&nbsp;Control

Sanghoon Kim

[Department / University, City, Country]

Corresponding author. E-mail address: october25kim@gmail.com

---

## Abstract

Federated learning is increasingly deployed in safety-sensitive applications where test inputs may belong to classes never seen during training. Existing federated open-set recognition methods evaluate unknown rejection only empirically (AUROC, FPR95), and federated conformal prediction certifies closed-set prediction-set coverage; neither bounds the error rate of the predictions a deployed model accepts. The purpose of this study is to certify the accepted selective risk of a federated open-set classifier, that is, the probability that an accepted prediction is wrong, with a finite-sample guarantee that is distribution-free with respect to the audited deployment distribution, under client heterogeneity and an unknown deployment mixture. We propose Fed-CORE, a post-hoc certification layer for any federated open-set model. Pooling accepted calibration points across heterogeneous clients certifies only the calibration mixture and becomes anti-conservative whenever deployment overweights a high-risk client, because the pooled accepted-error count is Poisson-binomial rather than binomial under stratified calibration; Fed-CORE instead bounds each client's conditional selective risk from low-dimensional accepted/error counts and certifies the global risk as a mixture-robust worst case, together with a per-client feasibility threshold that characterizes when certification is possible at all. The theorems are finite-sample valid under client-conditional audit representativeness. Empirically, certified deployments showed no held-out risk violation across our synthetic studies and CIFAR-10/100 real-logit experiments, whereas naive pooling collapsed under client-mixture shift and certified coverage followed the feasibility law. CIFAR-100 certified as a second real-data positive at a risk target of 0.20 across three backbones, while a tabular task remained a feasibility-edge stress domain. On CIFAR-10, Fed-CORE certified the native score of a full FedOSR method (FedPD-PROSER) at risk targets 0.10 and 0.20, and a five-seed FedAvg baseline at 0.20; these are grouped certificates of group mixtures under an explicit within-group sampling assumption, and a full-simplex per-client certificate on FedPD-PROSER additionally certified all five seeds at 0.20 without the grouping relaxation. All results are reported as finite-sample feasibility demonstrations rather than safety recommendations.

**Keywords:** Federated learning; Open-set recognition; Selective risk control; Conformal prediction; Distribution-free certification; Uncertainty quantification

---

## 1. Introduction

Consider many hospitals that jointly train one diagnostic model without sharing any patient record. Once the model is in use it will sometimes meet a disease it never saw during training; hence it must be honest enough to answer "I do not know" instead of forcing a guess. Two practical questions then follow. First, among the cases where the model *does* commit to an answer, how often is it wrong? Second, can we *promise*, with statistical confidence, that this error rate stays below a chosen tolerance, although every hospital's data looks different? This paper provides exactly that promise. It is computed from a small amount of trusted, correctly labelled audit data, and it does not change or retrain the model; we call the framework Fed-CORE. In one sentence: *Fed-CORE does not try to make the model better; it certifies which of the model's answers are safe to trust.*

Federated learning (FL) trains a shared model across clients that never disclose their raw data [1]. A broad survey documents how widely such training is deployed and how central client heterogeneity is to its behavior [2]. In practice these models are deployed in *open-world* conditions: a fraud detector meets a new fraud pattern, a clinical model meets a disease subtype absent from every hospital's training data. A deployed model must therefore not only classify known classes but also **abstain** on inputs it cannot safely classify, which is the classical reject option [3]. This is open-set recognition (OSR) [4], and its federated form (FedOSR) is now an active area [5].

The dominant FedOSR methods improved the *quality* of unknown rejection and reported it through ranking metrics such as the area under the receiver operating characteristic curve (AUROC) and FPR95. Parameter-disentanglement aggregation is the design of FedPD [5]. Score-model out-of-distribution (OOD) detection is the design of FOOGD [6]. Open-set voting drives FedOV [7]. Novel-class discovery drives FedNovel [8]. None of them answers the question a deployer actually needs answered: **if I accept and act on this model's confident predictions, what is the worst-case error rate among them, and can I guarantee it stays below a tolerance α?** Ranking metrics do not answer this; a model with high AUROC can still have an unacceptable error rate among accepted predictions at any fixed operating threshold, and that threshold is exactly what deployment fixes.

A second, separate literature does provide finite-sample guarantees in FL: federated conformal prediction (FCP) [9] constructs prediction sets with marginal coverage under a partial-exchangeability relaxation of the i.i.d. assumption. One recent variant added robustness to label shift [10]. Another added robustness to Byzantine clients [11]. However, FCP is closed-set: it assumes the true label is among the known classes and guarantees that the label lies in the returned set. It does not reject unknowns, and, as its own authors note, its selective-classification demonstrations were heuristics *without* a guarantee. Coverage of a prediction set and the *risk of an accepted point prediction* are different functionals; one does not imply the other.

A third body of work certifies unknown rejection, almost entirely in the **centralized** setting. Conformal novelty detection controls the false discovery rate (FDR) against a clean reference sample [12]. Conformal open-set classification extends prediction sets to unseen classes [13]. Reject-option conformal classification bounds the error of accepted singleton predictions [14]. Conformal risk control generalizes coverage to monotone risks [15]. Selective conformal risk control certifies risk on a selected subset [16]. An e-value framework extends the controlled risks further [17]. The one recent decentralized entrant [18] controls **batch FDR** over a set of test points in a pure novelty-detection (inlier/outlier) framing, with no known-class classifier and no notion of accepted selective risk. FDR over a test batch and the selective risk of a classifier's accepted predictions are, again, different objects requiring different finite-sample machinery.

The intersection, namely **a federated, heterogeneity-aware, finite-sample certificate on the accepted selective risk of an open-set classifier**, remains unaddressed by existing methods. Filling it is the contribution of this paper.

**Why this is hard, and not a trivial combination.** It is tempting to view the problem as "apply a Clopper–Pearson [19] selective-risk certificate (as in the centralized i.i.d. case) to the pooled federated calibration data." This is *invalid*. Under heterogeneity, accepted calibration points from different clients have different conditional error probabilities $r_j$. Consequently the pooled accepted-error count, conditioned on the pooled accepted count, is a sum of binomials with unequal success probabilities (a Poisson-binomial [20]), not a single binomial. Hence the standard Clopper–Pearson construction does not apply, and using it can be anti-conservative when the deployment mixture overweights high-error clients. In plain terms: if we lump every client's audit data into one pile, a few reliable clients can mathematically "average away" the mistakes of an unreliable one; hence the guarantee looks safer than it truly is. The correct object is the deployment-mixture-weighted ratio
$$ R_{\mathrm{sel}}(\lambda)=\frac{\sum_j \lambda_j\, m_j}{\sum_j \lambda_j\, a_j},\qquad m_j=\Pr_{P_j}(\text{accept}\wedge\text{error}),\ a_j=\Pr_{P_j}(\text{accept}), $$
with deployment weights $\lambda$ that are unknown at calibration time. Intuitively, $R_{\mathrm{sel}}(\lambda)$ is just this: of all the predictions the model chooses to act on, what fraction are wrong, measured under the (unknown) mix of clients that will actually appear at deployment. Certifying this ratio, finite-sample, distribution-free, and robustly over unknown $\lambda$, is a genuinely new statistical problem that reduces to *neither* the single-binomial certificate of the centralized case *nor* the quantile-coverage certificate of federated conformal prediction.

The purpose of this study is to certify, with a finite-sample distribution-free guarantee, the accepted selective risk of a federated open-set classifier under client heterogeneity and unknown deployment mixture, without retraining the model. Our hypothesis is that a small trusted clean audit set, although too small to repair a heterogeneous global model, is sufficient to certify which of its predictions can be accepted safely; validity comes from the independent certification fold, non-vacuity is constrained by a finite-sample feasibility law, and, once feasible, certified coverage tracks the quality of the base detector's score. We test this hypothesis through a conditional-binomial certificate, a controlled study of when the certificate is non-vacuous, and two downstream uses evaluated on synthetic and real federated benchmarks.

The main contribution of this study can be summarized as follows:

- We formalize the federated accepted selective risk $R_{\mathrm{sel}}(\lambda)$ as the certification target for federated open-set recognition, and show that it is a different functional from prediction-set coverage, ranking metrics, and batch false discovery rate (Section 3).
- We derive a finite-sample, distribution-free certificate for $R_{\mathrm{sel}}(\lambda)$ from the conditional law $K_j\mid A_j\sim\mathrm{Bin}(A_j,r_j)$, valid under client heterogeneity and unknown deployment mixture (Theorems 1–2, with a coverage lower bound in Corollary 1), together with a per-stratum feasibility threshold (Theorem 3) and a privacy/communication characterization of the certificate variants, including a grouped certificate whose target is deliberately coarsened to group mixtures and whose exactness condition we state formally (Proposition 3); we prove (Proposition 2) that naive pooling of federated calibration counts is anti-conservative under mixture shift and is therefore not reducible to centralized conformal prediction (Section 4).
- We characterize when certified open-set deployment is statistically feasible through a feasibility law in the risk target, the per-group audit count, heterogeneity, and corruption, turning the apparent null at small risk targets into a quantitative phenomenon (Sections 4.5 and 5).
- We evaluate the certificate's operating characteristics (validity, non-reducibility, feasibility, and score/base-model agnosticism) on synthetic and real federated benchmarks, with no held-out risk violation among certified deployments under the audit-representativeness assumption; as a secondary use, the certificate yields an admission gate for federated self-training with bounded pseudo-label contamination (Proposition 4; Sections 4.7 and 5).

Positioning note: Fed-CORE is best read not as a new FedOSR algorithm but as a **certification layer for any FedOSR / open-set FL model**, whose output is *used* for safe automation and certified self-training.

Section 2 positions the work against the four nearest literatures, Sections 3–4 develop the problem and the certificates with all proofs in the body, Section 5 reports the experiments, and Sections 6–7 discuss limitations and conclude.

---

## 2. Related Work

**Federated open-set / novel-class / OOD recognition.** FedPD [5] and its extension FedPD++ [21] framed FedOSR around two pathologies, cross-client *inter-set interference* between closed- and open-set objectives and cross-client *intra-set inconsistency* from heterogeneity, and addressed them by parameter disentanglement and divide-and-conquer aggregation. FOOGD [6] jointly targeted OOD generalization and detection: its SM³D module learned a feature-space score model (detection rule: threshold on the score norm), and its SAG module regularized feature invariance via Stein's identity. Crucially, FOOGD's only formal result bounded the *estimation error of the score model* (an MMD bound), not the error rate of the rejection decision; detection was reported via AUROC/FPR95. FedOV [7] tackled one-shot FL under label skew by training each client to emit an "unknown" class and ensembling via open-set voting. FedNovel [8] and federated open-world semi-supervised learning [22] discovered and learned novel classes across clients, and client heterogeneity itself was recast as a signal for OOD detection (FOSTER) [23]. Noise-Resistant FedOSR [24] is the closest to a corruption setting, using Bayesian uncertainty and label correction; however, like all of the above, it evaluated rejection empirically. Centralized OSR has continued to refine compact accept/reject regions [25], and federated heterogeneity has been studied beyond images, for example on graphs [26].

Adjacent lines addressed individual facets of deployment reliability. Clustered co-meta-learning personalized models under statistical heterogeneity [27]. Fairness-aware objectives balanced heterogeneous clients [28]. The global-versus-personalized trade-off was tuned explicitly [29]. Nonconvex pairwise fusion clustered clients [30]. Heuristic scheduling served heterogeneous clients efficiently [31]. Communication-efficient aggregation preserved privacy at scale [32]. The reject option was exploited for discrimination control [33]. Graph autoencoders detected outliers without supervision [34]. Quadtree spatial mapping detected concept drift in streams [35]. **None of these certifies a finite-sample bound on the accepted-prediction error rate.**

**Federated conformal / distribution-free uncertainty.** Lu et al. [9] introduce Federated Conformal Prediction, replacing exchangeability with *partial exchangeability* (a test point matches client $k$ with probability $\lambda_k$) and proving marginal coverage $1-\alpha \le \Pr(Y\in C) \le 1-\alpha + K/(N+K)$ with a privacy-preserving T-Digest quantile sketch. Plassier et al. [10] handle label shift via importance weighting; Rob-FCP [11] adds Byzantine robustness; generative and conditional variants refine conditional coverage. All certify **closed-set prediction-set coverage**, not unknown rejection or selective risk. Beyond federation, non-exchangeable conformal risk control [36] weighted calibration points to control a monotone expected risk under distribution drift; it relaxed exchangeability along the temporal and covariate axes, whereas Fed-CORE certifies a client-stratified accepted-risk *ratio* under an unknown deployment mixture, which is a different relaxation and a different functional. We adopt the partial-exchangeability viewpoint but certify a *risk* (a binomial functional), which behaves differently from a *quantile* and is not addressed by these works.

**Centralized selective-risk certification.** Conformal Risk Control [15] generalized conformal coverage to any monotone risk, building on distribution-free risk-controlling prediction sets [37]. A fast-moving 2025–2026 line then certified *post-selection* risk: Selective Conformal Risk Control [16] selected in two stages and then controlled risk on the selected subset; SCoRE [17] controlled risk among "trusted/positive" cases via e-values; and a recent joint finite-sample certificate [38] simultaneously bounded the selected risk ratio, the acceptance probability, and a deployment utility for an adaptively chosen selector. These are the closest statistical neighbors to Fed-CORE, and they certify the same *kind* of object, a post-selection risk ratio. However, all of them assume **centralized, exchangeable (i.i.d.) calibration from a single population**: none addresses client-stratified calibration, heterogeneous per-client error rates, an unknown deployment mixture $\lambda$, or the count-release constraints of federation, which are precisely the axes on which pooled calibration becomes invalid (Proposition 2). Fed-CORE is therefore best read as the federated, mixture-robust counterpart of this line, not as a competitor to it; we use these methods as **centralized oracles** (upper bounds), not as drop-in baselines.

**Conformal open-set / novelty detection.** Conformal novelty detection tests new points against a clean reference sample with finite-sample FDR control [12]; conformal open-set classification via Good–Turing p-values [13] and reject-option conformal classification [14] certified rejection centrally. The sole decentralized entry, *Decentralized Conformal Novelty Detection* [18], controlled **global FDR over a test batch** via quantized surrogate scores; this is pure novelty detection without a known-class classifier, a different functional from accepted selective risk. We used it as the nearest neighbor and differentiate on object and on certifying a single fixed selector.

**Positioning.** Existing selective conformal/risk-control methods certify post-selection risk in centralized/exchangeable settings. FedOSR methods such as FedPD report empirical rejection quality [5]. Federated conformal methods such as FCP certify closed-set prediction-set coverage [9]. Neither line certifies accepted point-prediction risk under an unknown client mixture. Table 1 organizes the prior work by the object it certifies. Fed-CORE fills the missing intersection: **federated open-set accepted-risk certification under client heterogeneity and deployment-mixture uncertainty.** It contributes the conditional selective-risk certificate (and its mixture-robust form) that the federated setting newly requires; the calibration statistic is a conditional-binomial proportion, not a score quantile. In one sentence: **Fed-CORE is not the first selective-risk certificate; to our knowledge, it is the first federated, client-stratified, deployment-mixture-robust certificate for the accepted selective risk of open-set point predictions, the setting in which naive pooling is invalid under heterogeneity.**

**Table 1. Prior work organized by certified object.**

| method family | fed. | open-set | certified object | unknown $\lambda$ | released statistic | finite-sample |
|---|:-:|:-:|---|:-:|---|:-:|
| FedPD / FedOSS / FOOGD [5,6,21,39] | ✓ | ✓ | none (empirical AUROC/FPR95) | ✗ | model updates | ✗ |
| FCP and variants [9,10,11] | ✓ | ✗ | prediction-set coverage | partial exch. | quantile sketch | ✓ |
| non-exchangeable CRC [36] | ✗ | ✗ | monotone expected risk under drift | ✗ | — (centralized) | ✓ |
| CRC / SCRC / SCoRE / joint cert. [15,16,17,38] | ✗ | optional | monotone risk / selected risk ratio | ✗ | — (centralized) | ✓ |
| decentralized novelty FDR [18] | partial | novelty | batch FDR | limited | quantized scores | ✓ |
| **Fed-CORE (this work)** | ✓ | ✓ | accepted selective risk $R_{\mathrm{sel}}(\lambda)$ | ✓ | count pairs | ✓ |

---

## 3. Problem Setup

Figure 1 locates the certified object among its neighboring quantities before the formal definitions. A deployment stream is partitioned by the model and a selector into accepted and rejected points; ranking metrics, prediction-set coverage, and batch FDR each answer a different question about this stream, and only the fourth quantity, the error rate among the accepted predictions under the deployment mixture, is the object that Fed-CORE certifies. Coverage of a prediction set, ranking quality, and batch FDR do not bound this accepted-prediction error rate; Fed-CORE does, in the federated setting, without pooling calibration data across heterogeneous clients. The figure also marks where the trusted folds enter the pipeline: the proposal fold selects the threshold $t$ of the selector, the disjoint certification fold certifies $R_{\mathrm{sel}}$ independently of that choice, and the test fold only estimates deployment behavior. The remainder of this section defines each ingredient.

![](experiments/fedcore/figs/fig0_problem_diagram.png)

**Figure 1. What Fed-CORE certifies.** A federated open-set model $\hat h$ with a selector $A$ partitions the deployment stream into accepted and rejected points; among the four quantities that can be asked about this stream, Fed-CORE certifies the accepted selective risk $R_{\mathrm{sel}}(\lambda)$.

**Clients and mixture.** There are $J$ clients; client $j$ has data distribution $P_j$ over $\mathcal{X}\times\mathcal{Y}$, where $\mathcal{Y}=\{1,\dots,C\}\cup\{\textsf{unknown}\}$. Deployment data follow a mixture $Q_\lambda=\sum_{j=1}^J \lambda_j P_j$ for some weight vector $\lambda\in\Delta^{J-1}$. We allow $\lambda$ to be **unknown at calibration time**, constrained only to a known convex set $\Lambda\subseteq\Delta^{J-1}$ (e.g., $\Lambda=\Delta^{J-1}$, or a box around client data fractions).

**Open-set decision.** A federated-trained classifier $\hat h$ produces a point prediction $\hat y(x)\in\{1,\dots,C\}$. A *selector* $A:\mathcal{X}\to\{0,1\}$ decides acceptance: $A(x)=1$ means "accept and act on $\hat y(x)$ as a known-class prediction"; $A(x)=0$ means "reject as unknown / abstain." $A$ may threshold any score $s(x)$ (maximum softmax probability (MSP) [40], entropy, margin, energy [41]); the guarantee will not depend on which. Such a selector descends from the classical reject option [3]. Its learned form follows selective classification with an integrated reject network [42].

**Target risk.** The quantity to control is the **accepted selective risk** under deployment, the federated open-set analogue of the selective risk of classical selective classification [43]:
$$
R_{\mathrm{sel}}(\lambda)\;=\;\Pr_{(X,Y)\sim Q_\lambda}\!\big(\hat y(X)\ne Y \,\big|\, A(X)=1\big)
\;=\;\frac{\sum_{j}\lambda_j\, m_j}{\sum_{j}\lambda_j\, a_j},
$$
where $a_j=\Pr_{P_j}(A(X)=1)$ is the per-client acceptance rate and $m_j=\Pr_{P_j}(A(X)=1,\ \hat y(X)\ne Y)$ is the per-client accepted-error mass (so the per-client selective risk is $r_j=m_j/a_j$; if $a_j=0$, client $j$ carries zero acceptance weight and we set $r_j:=0$ by convention). The ratio form follows from the law of total probability under the mixture. The companion quantity is the **accepted coverage** $C(\lambda)=\sum_j\lambda_j a_j$, the fraction of the deployment stream that is accepted. **Goal:** deploy $A$ only if we can certify $R_{\mathrm{sel}}(\lambda)\le\alpha$ for the (unknown) deployment $\lambda\in\Lambda$ with confidence $1-\delta$, while maximizing $C(\lambda)$. Accordingly, the certificate outputs two numbers: a risk upper confidence bound $\bar U_\Lambda$ (deploy iff $\bar U_\Lambda\le\alpha$) and a coverage lower confidence bound $\underline C_\Lambda$; the experimental metrics `cert_risk_ucb` and `cert_coverage_lcb` report exactly these two quantities.

**Trusted calibration data and the split.** Each client holds a small *trusted, clean* calibration sample, partitioned into a **proposal** fold and a **certification** fold. The selector $A$ is chosen on the proposal fold (across clients) and is therefore *fixed and independent* of the certification fold. The proposal fold may select both the score family and its threshold from any finite candidate class by any data-dependent rule; because certification consumes a single fixed selector, no multiplicity correction over the proposal search is needed on the certification side (assumption A2). On the certification fold, client $j$ contributes $n_j$ i.i.d. draws from $P_j$ and reports two integers:
$$
A_j=\sum_{i=1}^{n_j}\mathbf 1\{A(x_i)=1\},\qquad
K_j=\sum_{i=1}^{n_j}\mathbf 1\{A(x_i)=1,\ \hat y(x_i)\ne y_i\}.
$$
By construction $A_j\sim\mathrm{Bin}(n_j,a_j)$ and $K_j\sim\mathrm{Bin}(n_j,m_j)$ with $K_j\le A_j$; conditionally, $K_j\mid A_j\sim\mathrm{Bin}(A_j,r_j)$. What must leave the client depends on the certificate (per-client pairs for the stratified one, sums for the pooled one); see the privacy characterization in Section 4.6. A third, held-out **test fold** estimates deployment behavior *after* certification; it is never used to select or certify a selector, and all `test_*` quantities in Section 5 are deployment estimates, not inputs to the guarantee.

**Calibration must contain labeled unknowns.** To certify *unknown rejection*, the certification fold must include points with $Y=\textsf{unknown}$ that are **labeled as such**: unknown classes are unseen during training but present and labeled in this small post-training **audit/calibration fold**. Hence "distribution-free" is *with respect to the calibration distribution $Q_\lambda$*, not the entire unknown universe. In OSR benchmarks this holds by construction (held-out classes excluded from training, used as unknown-labeled calibration/test examples); in deployment it corresponds to a small audited monitoring set. If the audit fold has no unknowns, the certificate degrades to a closed-set selective-risk guarantee. We certify only selectors with positive accepted coverage $a_\lambda=\sum_j\lambda_j a_j>0$; a zero-coverage selector is non-deployable.

The need for trusted labels is not an artifact of our construction; it is information-theoretically unavoidable.

**Proposition 1 (necessity of trusted calibration labels).** *No procedure that observes only the model's outputs together with corrupted or unlabeled client data can certify $R_{\mathrm{sel}}(\lambda)\le\alpha$ for $\alpha<1$, distribution-free, while deploying with nontrivial acceptance.*

*Proof.* Fix an instance (World A) on which the procedure deploys with probability $p_0>0$: a model, a selector-eligible score, and a calibration sample whose *clean* labels agree with the model's predictions on all accepted points, and thus $R_{\mathrm{sel}}(\lambda)=0\le\alpha$. Construct World B by flipping the clean labels of the accepted calibration points (and, for the population statement, the conditional label distribution on the acceptance region) while leaving the corrupted/unlabeled observations and all model outputs unchanged; this is possible because the corruption process is not assumed known or invertible and unlabeled data carry no label information. In World B the accepted predictions are wrong; hence $R_{\mathrm{sel}}(\lambda)=1>\alpha$. The observable input is identical in both worlds; therefore the procedure deploys in World B with the same probability $p_0$, and every such deployment violates its certificate; validity therefore forces $p_0\le\delta$. Because the construction applies to *every* instance on which the procedure deploys, the procedure deploys with probability $\le\delta$ everywhere: it is vacuous. Trusted clean labels on the certification fold break the indistinguishability, since the two worlds then differ observably. $\square$

**Risk-buffered proposal.** To avoid certifying a selector whose empirical risk already sits at $\alpha$ (which makes the certificate fail), the proposal fold selects $A$ subject to an empirical buffer $\widehat R_{\mathrm{prop}}(A)\le\gamma\alpha$ with $0<\gamma<1$ (candidate grid $\gamma\in\{0.2,0.3,0.5,0.7,1.0\}$), inherited from the centralized framework.

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
| **A4′** | deliberate unknown over-representation is conservative only under accepted-error stochastic dominance; empirical stress protocol, not a theorem assumption |
| **A5** | deployment mixture lies in the declared $\Lambda$ (trivial for the full simplex) |
| **A6** | a grouped certificate certifies *group* mixtures only; exactness requires i.i.d. group-mixture audit sampling (Proposition 3) |

Two of these deserve emphasis. **A4/A4′:** the theorems consume an audit fold drawn from the deployment client-conditional distribution; if the audit instead intentionally over-represents unknowns, validity is retained only under stochastic dominance of the accepted-error indicator; a higher unknown fraction alone is not sufficient in general, because accepted error also depends on known-class composition and difficulty. We use over-representation only as an empirical conservative stress protocol (Section 5.3); under-representation is demonstrably anti-conservative. **A6:** merging clients into public groups (Section 4.6) deliberately relaxes the certified object from client mixtures to group mixtures; it does not protect against arbitrary within-group client-mixture shift.

---

## 4. Fed-CORE Method and Theory

The method has four plain ingredients, built up in this section. (1) A classical, exact way to turn a count of errors into a high-confidence upper bound on an error rate (Section 4.1). (2) The core certificates, full-simplex (Section 4.2) and bounded-$\Lambda$ with a coverage lower bound (Section 4.3), together with the reason naive pooling fails (Section 4.4). (3) A feasibility law that says how much audit data each stratum needs before a guarantee is even possible (Section 4.5). (4) The practical consequences: the protocol, what information must leave each client, and the certificate variants (Section 4.6); and why the guarantee does not depend on the chosen score, plus what the certified predictions are used for (Section 4.7). All proofs are given in the body.

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

In words, the certificate does the following. For each client we look only at the audit points that the model actually accepted, count how many of those were wrong, and use an exact binomial confidence bound (Section 4.1) to obtain an upper bound on that client's true error rate. The global guarantee is then driven by the worst client, because, as explained above, no client's errors can be hidden behind another's. The rest of this subsection makes this precise.

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

*Proof (Theorem 1).* Conditional on $A_j=a$, the accepted-error count is $K_j\mid A_j{=}a\sim\mathrm{Bin}(a,r_j)$; thus the Clopper–Pearson guarantee gives $\Pr(r_j\le\bar r_j\mid A_j{=}a)\ge1-\varepsilon$ for every value $a$ (including $a{=}0$, where $\bar r_j:=1$ trivially covers $r_j$); because the bound holds for every conditioning value, it holds marginally by the tower property. A union over the $J$ clients with $\varepsilon=\delta/J$ gives the event $E=\{\forall j:\ r_j\le\bar r_j\}$ with $\Pr(E)\ge1-\delta$. On $E$, for **any** $\lambda$ with $\sum_j\lambda_j a_j>0$, Lemma 1 gives $R_{\mathrm{sel}}(\lambda)=\sum_j w_j(\lambda)\,r_j\le\sum_j w_j(\lambda)\,\bar r_j\le\max_j\bar r_j=\bar U_\Delta^{\,r}$. The bound does not depend on $\lambda$; thus, on $E$ it holds simultaneously for all admissible $\lambda$. $\square$

Theorem 1 uses **one** event per client (not two) and **no** acceptance lower bound. A mass-ratio variant that separately bounds $m_j$ and $a_j$ is valid by the same argument but uniformly looser, because it pays both numerator and denominator confidence slack; we keep it only as a diagnostic baseline (Table 3). The interpretation of Theorem 1 is that the **worst client sets the bar**: no client's error can be averaged away.

### 4.3 Bounded-$\Lambda$ certificate and coverage lower bound

**Theorem 2 (bounded $\Lambda$, robust certificate recommended for deployment).** *When $\Lambda$ is a known strict subset (e.g. a box around public client data fractions), the worst-client domination is avoided. At level $\varepsilon=\delta/3J$ bound $r_j\le\bar r_j$ and $a_j\in[\underline a_j,\bar a_j]$ with $\underline a_j=L^-(A_j,n_j;\varepsilon)$, $\bar a_j=U^+(A_j,n_j;\varepsilon)$, and set*
$$
\bar U_{\Lambda}^{\,r,a}=\sup_{\lambda\in\Lambda,\ a_j\in[\underline a_j,\bar a_j]}\ \frac{\sum_j \lambda_j a_j \bar r_j}{\sum_j \lambda_j a_j}.
$$
*Then $\Pr(R_{\mathrm{sel}}(\lambda^\star)\le \bar U_\Lambda^{\,r,a})\ge1-\delta$ for the true $\lambda^\star\in\Lambda$.*

*Proof.* Define the $3J$ events $\{r_j\le\bar r_j\}$, $\{a_j\ge\underline a_j\}$, $\{a_j\le\bar a_j\}$, each of probability $\ge1-\delta/3J$ (the conditional argument of Theorem 1 for the first; the Clopper–Pearson lower/upper guarantees applied to $A_j\sim\mathrm{Bin}(n_j,a_j)$ for the others). By the union bound their intersection $E'$ has probability $\ge1-\delta$. On $E'$ the true parameters satisfy $r_j\le\bar r_j$ and $a_j^\star\in[\underline a_j,\bar a_j]$; thus the pair $(\lambda^\star,a^\star)$ is feasible for the supremum and, using $r_j\le\bar r_j$ in the numerator, $R_{\mathrm{sel}}(\lambda^\star)\le\bar U_\Lambda^{\,r,a}$. For fixed $\lambda$ the objective is linear-fractional in $a$ over the compact box, hence pseudolinear, and thus its supremum is attained at an extreme point ($a_j\in\{\underline a_j,\bar a_j\}$); equivalently, for any threshold $t$ the condition $\sum_j\lambda_j a_j(\bar r_j-t)\ge0$ is linear in $a$; the outer supremum is a small linear-fractional program (Charnes–Cooper / Dinkelbach). $\square$

*Choosing $\Lambda$ in practice.* Theorem 2 requires a declared $\Lambda$ (assumption A5), and the deployer, not the certificate, owns this choice. A concrete protocol: take the publicly known client data or traffic fractions $\hat\lambda$ (already exchanged for FedAvg weighting), declare the box $\Lambda=\prod_j[\hat\lambda_j(1-\rho),\ \hat\lambda_j(1+\rho)]\cap\Delta^{J-1}$ with a margin $\rho$ chosen from domain knowledge about how far deployment traffic can drift (we use $\rho=0.15$), and fall back to the full simplex (Theorem 1) when no defensible $\hat\lambda$ exists. Declaring $\Lambda$ too narrow voids the guarantee outside it; the simplex certificate remains the assumption-free default. Section 5.5 reports a real-data sweep over both deployment knobs, the box radius $\rho$ and the proposal buffer $\gamma$.

The certificate's second output, the coverage lower bound, admits the same treatment; we state it as a corollary so that the two headline quantities are formally on equal footing.

**Corollary 1 (simultaneous coverage lower confidence bound).** *Split the budget as $\delta_r+\delta_c\le\delta$: run the risk certificate (Theorem 1 or 2) at level $\delta_r$ and bound each acceptance rate from below, $\underline a_j=L^-(A_j,n_j;\delta_c/J)$. With*
$$
\underline C_\Lambda=\inf_{\lambda\in\Lambda}\ \sum_j \lambda_j\,\underline a_j,
$$
*the events $\{\sup_{\lambda\in\Lambda}R_{\mathrm{sel}}(\lambda)\le\bar U\}$ and $\{\inf_{\lambda\in\Lambda}C(\lambda)\ge\underline C_\Lambda\}$ hold simultaneously with probability at least $1-\delta$.*

*Proof.* The $J$ additional events $\{a_j\ge\underline a_j\}$ each hold with probability $\ge1-\delta_c/J$ by the Clopper–Pearson lower guarantee; a union bound over all risk and coverage events costs $\delta_r+\delta_c\le\delta$. On the intersection, $C(\lambda)=\sum_j\lambda_j a_j\ge\sum_j\lambda_j\underline a_j\ge\underline C_\Lambda$ for every $\lambda\in\Lambda$; the infimum is linear in $\lambda$ and attained at a vertex of $\Lambda$. $\square$

*Implementation note.* Our pipeline reports `cert_risk_ucb` at level $\delta$ and `cert_coverage_lcb` at auxiliary level $\delta/2$; when a simultaneous statement is required, both should be run at the split levels of Corollary 1 (e.g., $\delta_r=\delta_c=\delta/2$), which widens each bound only marginally.

**Edge cases (in the statement, not the proof).** (i) *Zero accepted coverage.* We certify only selectors with $a_\lambda=\sum_j\lambda_j a_j>0$; if $a_\lambda=0$ the selector accepts nothing and is **non-deployable** ($R_{\mathrm{sel}}$ undefined). (ii) *Vanishing denominator bound.* If $\inf_{\lambda\in\Lambda}\sum_j\lambda_j\underline a_j=0$ (a client may accept yet $\underline a_j=0$ when $A_j$ is small) the robust certificate is declared **infeasible** ($\bar U=+\infty$); we do *not* silently drop such a client, which would break the worst-case guarantee.

### 4.4 Why pooling is invalid

The certificate is not a corollary of existing constructions, for three separate reasons. The first is a formal statement; the failure it describes is the technical crux of the federated setting.

**Proposition 2 (naive pooling is anti-conservative under mixture shift).** *Let $U_{\mathrm{pool}}=U^+(\sum_j K_j,\sum_j A_j;\delta)$ be the single Clopper–Pearson bound applied to the pooled accepted calibration points. There exist client populations $\{(a_j,r_j)\}_{j\le J}$ and deployment mixtures $\lambda\in\Delta^{J-1}$ such that $\Pr\big(R_{\mathrm{sel}}(\lambda)>U_{\mathrm{pool}}\big)\to1$ as the per-client calibration sizes $n_j\to\infty$.*

*Proof (constructive).* Conditioned on the accepted counts, the pooled accepted-error count is a sum of independent binomials with unequal success probabilities $\{r_j\}$, that is, a Poisson-binomial; hence the single-binomial Clopper–Pearson calibration does not apply to it. Quantitatively, take equal calibration sizes and the population $J{=}5$ with four low-risk clients ($a{=}0.7$, $r{=}0.02$) and one high-risk client ($a{=}0.5$, $r{=}0.3$). By the strong law of large numbers, $\sum_j K_j/\sum_j A_j\to\bar r_{\mathrm{cal}}=\sum_j a_j r_j/\sum_j a_j\approx0.074$ almost surely, and $U_{\mathrm{pool}}\to\bar r_{\mathrm{cal}}$ as well, because the Clopper–Pearson width shrinks as $O(\sqrt{\log(1/\delta)/n})$. Now let $\lambda$ put all deployment mass on the high-risk client: $R_{\mathrm{sel}}(\lambda)=0.3>\bar r_{\mathrm{cal}}$; thus, for any fixed margin $0<c<0.3-\bar r_{\mathrm{cal}}$, $\Pr(U_{\mathrm{pool}}<\bar r_{\mathrm{cal}}+c)\to1$ and hence $\Pr(R_{\mathrm{sel}}(\lambda)>U_{\mathrm{pool}})\to1$. The same argument applies to every mixture that overweights above-average-risk clients. (Figure 2 reports the finite-sample counterpart, with coverage collapsing to $0$.) $\square$

The failure is one of certified target rather than of the binomial bound alone: pooled Clopper–Pearson consistently certifies the *calibration-mixture* risk, which is simply the wrong quantity when deployment overweights a high-risk client, and the Poisson-binomial structure explains why no single-binomial correction can repair it. In plain terms, pooling lets reliable clients average away an unreliable one: the guarantee looks safer than it is exactly when deployment overweights the unsafe client. Two further reductions fail as well: **(b)** Lu et al.'s federated-conformal certificate controls a *quantile/coverage* of nonconformity scores under partial exchangeability, whereas Fed-CORE controls a **post-selection conditional error ratio** under client-mixture uncertainty; the calibration statistic is a conditional-binomial proportion $K_j\mid A_j$, not a score quantile. **(c)** The certificate is not a Bonferroni union of unrelated intervals: by Lemma 1 the certified object couples the per-client bounds through acceptance-reweighted convex weights, and the deployment certificate (Theorem 2) is a robust linear-fractional program over $(\lambda,a)$, not a maximum of independent intervals.

### 4.5 Feasibility law

In the zero-accepted-error regime $K_j=0$ with $\Lambda=\Delta^{J-1}$, the deploy condition $\max_j\bar r_j\le\alpha$ reduces, for each binding client, to $U^+(0,A_j;\delta/J)\le\alpha$.

**Theorem 3 (feasibility law).** *In the zero-accepted-error regime, certification at level $\alpha$ over the simplex holds for a client if and only if its **observed accepted count** satisfies*
$$
U^+(0,A_j;\delta/J)\le\alpha
\quad\Longleftrightarrow\quad
A_j\ \ge\ \frac{\ln(J/\delta)}{-\ln(1-\alpha)} .
$$

*Proof.* For $K=0$ the upper Clopper–Pearson limit is $U^+(0,A_j;\varepsilon)=1-\varepsilon^{1/A_j}$ with $\varepsilon=\delta/J$ (the solution of $(1-p)^{A_j}=\varepsilon$). The deploy condition $1-(\delta/J)^{1/A_j}\le\alpha$ is equivalent to $(\delta/J)^{1/A_j}\ge1-\alpha$, i.e. $\tfrac1{A_j}\ln(\delta/J)\ge\ln(1-\alpha)$; both logarithms are negative; thus dividing and flipping gives $A_j\ge\ln(J/\delta)/(-\ln(1-\alpha))$. $\square$

Because $-\ln(1-\alpha)\ge\alpha$, the threshold is of order $\ln(J/\delta)/\alpha$; the expected-count form $n_j a_j\gtrsim \ln(J/\delta)/\alpha$ follows from $\mathbb E[A_j]=n_j a_j$ and concentration of $A_j$. We state the bound on the **observed $A_j$**, which is exactly what the certificate consumes. This is the federated analog of the centralized $N_{\min}=\lceil\log\delta/\log(1-\alpha)\rceil$: the condition must hold *per stratum*, with a $\log J$ federation penalty, and it predicts **certified-coverage collapse** when a client is simultaneously small and high-risk. Theorem 3 is the zero-accepted-error floor; its finite-sample *width* extension (the normal/Bernstein approximation to the Clopper–Pearson interval, for which the required count scales like $(\alpha-\hat r)^{-2}$) adds that if the model's own error rate $\hat r$ already sits close to the target $\alpha$, the required audit count explodes. Certification is therefore possible only when the model is comfortably below the target and the audit set is large enough; otherwise the correct output is "cannot certify." Section 5.4 traces this law on real logits.

*The floor is fundamental, not an artifact of Clopper–Pearson.* One might ask whether a less conservative certificate could deploy with fewer accepted points; it cannot, because the count in Theorem 3 is a minimax lower bound for *any* valid procedure. Fix a single stratum and, for $\epsilon>0$, a borderline-unsafe client with conditional risk $r=\alpha+\epsilon$, so that $R_{\mathrm{sel}}>\alpha$. Validity requires $\Pr(\text{deploy}\wedge R_{\mathrm{sel}}>\alpha)\le\delta$. A procedure that never deploys on the event $\{K=0\}$ deploys with probability at most $\Pr(K\ge1)=1-(1-r)^A\to0$ as $r\to0$, and is therefore vacuous even for arbitrarily safe clients; hence any non-vacuous procedure must deploy on $\{K=0\}$. Under $r=\alpha+\epsilon$ one has $\Pr(K=0\mid A)=(1-\alpha-\epsilon)^{A}$, hence $\Pr(\text{deploy}\mid r=\alpha+\epsilon)\ge(1-\alpha-\epsilon)^{A}$, and validity forces $(1-\alpha-\epsilon)^{A}\le\delta$. Letting $\epsilon\downarrow0$ gives $A\ge\ln(1/\delta)/(-\ln(1-\alpha))$, which matches Theorem 3 up to the $\ln J$ union factor across strata. No valid certificate, however tight, escapes the $\Omega(\ln(1/\delta)/\alpha)$ per-stratum floor: the collapse of certified coverage under small, high-risk clients is information-theoretic, not a looseness of the Clopper–Pearson bound.

### 4.6 Algorithm, privacy, and certificate variants

The full protocol is summarized as Algorithm 1.

**Algorithm 1 (Fed-CORE certification).**

1. **Train.** Obtain the federated open-set model $\hat h$ and any score $s$ (FedAvg or any FedOSR method; Fed-CORE does not modify training).
2. **Propose.** On the proposal fold, select the threshold $t$ of the selector $A$ subject to the risk buffer $\widehat R_{\mathrm{prop}}\le\gamma\alpha$ (Section 3); fix $A$.
3. **Count.** Each client $j$ evaluates $A$ on its certification fold and reports the pair $(A_j,K_j)$: per client for the stratified certificate, secure-aggregated within groups for the grouped variant.
4. **Certify.** The server computes $\bar r_j=U^+(K_j,A_j;\varepsilon)$ and the certificate: $\bar U_\Delta^{\,r}=\max_j\bar r_j$ (Theorem 1) or the robust linear-fractional bound $\bar U_\Lambda^{\,r,a}$ (Theorem 2), plus the coverage LCB $\underline C_\Lambda$ (Corollary 1).
5. **Decide.** Deploy $A$ iff $\bar U\le\alpha$ and $\underline C_\Lambda>0$; otherwise report "cannot certify" (with the Theorem-3 diagnosis of whether the failure is risk- or feasibility-driven).

The privacy footprint depends on **which** certificate is deployed: sum-only secure aggregation suffices **only for the pooled diagnostic**, while the stratified certificate requires per-client count pairs. Table 3 summarizes, for each variant, the certified target, the counts released, and the key assumption; in the table, "released" is what leaves each client, and "sec. agg." marks compatibility with secure aggregation.

**Table 3. Certificate variants: certified target, privacy, and role.**

| variant | certified target | released | sec. agg. | assumption | role |
|---|---|---|:-:|---|---|
| simplex (Thm 1) | any client mixture | client pairs | ✗ | A1–A4 | most robust |
| box-$\Lambda$ (Thm 2) | $\lambda\in\Lambda$ | client pairs | ✗ | A1–A5 | recommended |
| grouped-stratified | group mixtures | group pairs | within groups | A1–A4, A6 | privacy/feasibility |
| pooled (Remark 1) | matched mixture only | sums | ✓ | matched calibration | diagnostic only |
| mass-ratio | any client mixture | client pairs | ✗ | A1–A4 | loose diagnostic |

Because Theorem 1 needs **per-client** counts, it is *not* compatible with sum-only secure aggregation. The recommended compromise is a **grouped-stratified certificate**: partition clients into $G$ pre-declared public groups (approximately balanced when equal sizes are impossible), secure-aggregate counts *within* each group, and run the certificate over the $G$ groups. This keeps a group-mixture guarantee while releasing only $G$ aggregated pairs. However, the certified object weakens accordingly (assumption A6): the guarantee is robust to mixture shift *across* groups, not to arbitrary client-mixture shift *within* a group. Grouping is therefore a declared relaxation, qualitatively distinct from naive pooling (which certifies only the matched mixture and silently fails off it, Proposition 2); the grouped certificate states its coarser target up front. The released statistic is lower-dimensional than the per-client score *distributions* (T-Digest) of federated conformal prediction or the quantized score *functions* of decentralized conformal novelty detection; this is an information-flow statement, not a formal privacy guarantee. A differentially private count-release variant (calibrated noise on the counts with correspondingly widened Clopper–Pearson levels) is a natural extension but is left as future work; no formal DP guarantee is claimed here.

The grouped certificate deserves the same formal treatment as the client-level one, because merging heterogeneous clients raises exactly the Poisson-binomial concern of Proposition 2 *inside* each group. The following statement records when the group-level counts satisfy an exact conditional binomial law.

**Proposition 3 (exact validity of the grouped certificate).** *Fix a pre-declared partition of the clients into groups $g=1,\dots,G$ with within-group composition $\pi_{j\mid g}$, and let $P_g=\sum_{j\in g}\pi_{j\mid g}P_j$. If each group's certification fold is drawn i.i.d. from $P_g$, then, with $A_g$ and $K_g$ the group's accepted and accepted-error counts, $K_g\mid A_g\sim\mathrm{Bin}(A_g,r_g)$ exactly, where $r_g$ is the conditional selective risk of $P_g$. Consequently Theorems 1–2, Corollary 1, and Theorem 3 hold verbatim with groups as strata, and the certified object is $R_{\mathrm{sel}}$ over mixtures of the group distributions $\{P_g\}$ at the fixed within-group composition.*

*Proof.* A group whose audit fold is drawn i.i.d. from $P_g$ is formally a single client with distribution $P_g$: because the client membership of an i.i.d. draw from the mixture is itself random with the same law for every point, each audit point independently accepts with probability $a_g=\Pr_{P_g}(A(X)=1)$ and, conditionally on acceptance, errs with probability $r_g=\Pr_{P_g}(\hat y(X)\ne Y\mid A(X)=1)$. The conditioning argument of Theorem 1 then applies unchanged at the group level. $\square$

The sampling condition is the substance of assumption A6, and it is what separates the grouped certificate from hidden pooling: under i.i.d. group-mixture sampling the group is a genuine stratum with an exact binomial law, whereas fixed per-client audit quotas inside a group make $K_g$ conditionally Poisson-binomial, for which exactness is not claimed. Our implementation draws per-client folds (quota sampling); thus the grouped results of Section 5 are read under A6 with this caveat stated openly. Two pieces of evidence bound the practical gap: the resampling study of Section 5.2, which redraws audit folds and evaluates the grouped ($G{=}2$) certificate directly, observed a violation rate of $8.7\times10^{-4}$ against the nominal $\delta{=}0.10$, and the full-simplex per-client positive of Section 5.5 shows that the real-data conclusions do not depend on grouping at all.

**Remark 1 (matched-mixture pooled diagnostic).** When calibration samples are drawn i.i.d. from the deployment mixture itself, the pooled bound $U^+(\sum_j K_j,\sum_j A_j;\delta)$ can serve as a diagnostic tightness reference. It is not used for any main guarantee, headline metric, or deployment recommendation: its validity would require additional Poisson-binomial-to-binomial comparison arguments and the matched-mixture assumption (cf. Hoeffding's classical comparison of Poisson-binomial and binomial tails [20]); therefore we do not elevate it to a theorem in this paper.

### 4.7 Score agnosticism and certified uses

Because the guarantee in Theorems 1–2 is produced entirely by the certification split (the conditional-binomial structure of $K_j\mid A_j$ under a *fixed* selector), it holds for **any** score $s(\cdot)$ used to define $A$. Score quality affects *how much coverage* is certified (a better score accepts more at the same risk), never *whether the risk is controlled*. This matters because federated heterogeneity, like label corruption, **deforms the confidence–correctness ranking**: at clients holding only minority known classes, those classes are easily confused with genuine unknowns; hence any single global score is miscalibrated somewhere. Fed-CORE does not attempt to repair this deformation with the small trusted set; it *certifies around it*. Section 5.5 makes this concrete on real detectors: validity holds for every score while certified coverage tracks score quality.

The certificate licenses two downstream uses of the accepted set. **Use case A (safe automation/triage, no retraining).** Accept $=$ act automatically; reject $=$ defer to a human. CertifiedCoverage@$\alpha$ is then exactly the **fraction of the workload safely automated at guaranteed error $\le\alpha$**, and $1-\text{coverage}$ is the human-review load; an uncertified operating point either breaches $\alpha$ or automates less at the same guaranteed risk. **Use case B (certified federated self-training).** Accepted predictions on unlabeled client data become pseudo-labels folded back into FedAvg; the certificate bounds their **contamination** (pseudo-label error rate $\le\alpha$ with respect to the calibration distribution), replacing the unbounded contamination of naive self-training with provably-bounded noise.

Self-training makes the round-$t$ model depend on what was accepted at round $t-1$; hence **reusing one certification fold across rounds would break the independence** the certificate needs. We sidestep the closed-loop concentration problem by **data-splitting in time**: partition the trusted set into $T$ disjoint audit folds $\mathcal C^{(1)},\dots,\mathcal C^{(T)}$ and certify the round-$t$ selector on the fresh fold $\mathcal C^{(t)}$ at level $\delta/T$.

**Proposition 4 (round-wise self-training validity).** *If $\mathcal C^{(t)}$ is independent of $(f_t,A_t)$ (guaranteed by forming $f_t,A_t$ only from folds indexed $<t$ and from unlabeled data) and each round is certified at level $\delta/T$, then*
$$
\Pr\big(\forall t\le T:\ R_{\mathrm{sel}}(A_t)\le \bar U^{(t)}\big)\ \ge\ 1-\delta .
$$

*Proof.* By construction $\mathcal C^{(t)}$ is independent of $(f_t,A_t)$; thus $A_t$ is a fixed selector with respect to the fresh fold and Theorem 1 (or 2) applies on that fold at level $\delta/T$: $\Pr(R_{\mathrm{sel}}(A_t)>\bar U^{(t)})\le\delta/T$ for each $t$. A union bound over the $T$ rounds gives simultaneous validity at level $\delta$. $\square$

Every injected pseudo-label batch therefore has certified contamination $\le\alpha$ simultaneously across all $T$ rounds, without a closed-loop adaptivity argument. The price is feasibility: each round's fold must clear the Theorem-3 threshold; hence $T$ is bounded by the trusted-set size, an explicit budget/utility trade-off. Section 5.6 verifies the contract empirically.

---

## 5. Experiments

The experiments are not designed to show that Fed-CORE improves open-set recognition accuracy; they evaluate the *operating characteristics of a statistical procedure*. Four certification claims are tested, in order: **(i) validity:** the theorems guarantee risk control under A1–A6; empirically, certified deployments showed no held-out risk violation and the unsafe-deployment rate of the decision rule stayed below $\delta$ (Sections 5.2–5.3); **(ii) non-reducibility:** pooled Clopper–Pearson failed under client-mixture shift exactly as Proposition 2 predicts (Section 5.2); **(iii) feasibility:** certified coverage appeared only when the per-group accepted audit count clears the Theorem-3 floor, and collapsed along the risk-target, heterogeneity, and corruption axes as the feasibility law predicts (Sections 5.4–5.5); **(iv) score- and base-model agnosticism:** validity held for every score and detector while certified coverage tracked detector quality (Section 5.5). Section 5.6 evaluates the downstream self-training gate. Full per-run logs and scripts are released with the reproducibility package.

### 5.1 Experimental setup

We train federated models with FedAvg [1] under non-IID Dirichlet partitions ($d\in\{0.1,0.5,5\}$; smaller $d$ is more heterogeneous), hold out classes as test-time unknowns (the standard FedOSR open-set split), and optionally corrupt client labels (symmetric/asymmetric) to connect to the corrupted-training setting. The datasets are CIFAR-10 (primary), CIFAR-100 (a second real-data positive at $\alpha{=}0.20$; Section 5.5), and a tabular FL benchmark (covtype, a feasibility-edge domain). Four post-hoc scores, namely maximum softmax probability (MSP) [40], entropy, margin, and energy [41], test the score-agnostic claim. The headline metric is CertifiedCoverage@$\alpha$: the mean certified coverage lower bound `cert_coverage_lcb` across seeds, credited only when the certificate deploys ($\bar U_\Lambda\le\alpha$; uncertified runs contribute zero). Its confidence statement is the simultaneous budget of Corollary 1 ($\delta_r=\delta_c=\delta/2$): the risk upper bound and the coverage lower bound hold *jointly* with confidence $1-\delta$, so `cert_risk_ucb` and `cert_coverage_lcb` are reported as a matched pair. Reporting the coverage at the full budget $\delta$ instead (a risk-only guarantee) raises the boundary headline cell by about $0.05$ and the robust cells by at most $0.003$. Held-out `test_coverage` is reported separately as the deployment estimate. FedOSR detectors are treated as **base models**, not competitors: they carry no certificate, and Fed-CORE certifies their scores post hoc. The primary base detector is FedPD [5]. The second native score comes from FOOGD [6]. FedOSS is deferred as the highest-cost reproduction [39]. As, to our knowledge, the first method to certify this object, we compare against re-cast nearest methods (federated CP [9], decentralized novelty FDR [18]) that are invalid or control a different functional, and against our own variants as ablations; centralized selective-risk certificates are positioned conceptually in Section 2 rather than run as baselines, because they assume a single exchangeable population and are not drop-in valid here [38]. The configuration, whose values are parameters of the guarantee itself rather than tuning details, is the following: $J{=}5$ clients, six known CIFAR-10 classes (the rest unknown); backbones CIFAR-stem ResNet-18 (GroupNorm/BatchNorm) and SimpleCNN, with WideResNet for the FedPD/FOOGD reproductions; FedAvg for $50$ rounds, two local epochs, lr $0.01$, batch $64$; disjoint trusted-pool folds of $0.34/0.33/0.33$ (proposal/certification/test) with audit unknown fraction $0.30$; certificate parameters $\alpha\in\{0.10,0.20,0.25\}$ ($\alpha{=}0.05$ is used only in the synthetic boundary study of Table 4, where it places the true risk at the decision boundary), $\delta{=}0.10$, $\gamma\in\{0.2,0.3,0.5,0.7,1.0\}$, $\Lambda\in\{\text{simplex},\ \text{box}(\rho{=}0.15)\}$; corruption applied to train labels only (symmetric $0.35$ / asymmetric $0.20$), calibration folds always clean.

### 5.2 Validity and non-reducibility

We first establish that the certificate is valid and that the obvious alternative, pooling, is not.

*Controlled synthetic study.* On synthetic clients whose ground-truth risks and deployment mixtures are varied independently, every property holds: empirical coverage $\ge0.98$ across heterogeneity; the tightness order box $<$ simplex $<$ mass-ratio (all valid; the pooled certificate is tightest but invalid off the matched mixture); a monotone CertifiedCoverage@$\alpha$ frontier; the heterogeneity-collapse curve crossing the Theorem-3 floor ($\approx37$ accepted/client); and all four scores valid (test_risk $\approx0.044\le\alpha$).

*Pooling is anti-conservative (non-reducibility).* The natural shortcut, pooling every client's accepted audit points and applying one Clopper–Pearson bound, failed under heterogeneity.

![](experiments/fedcore/figs/fig1_pooling_collapse.png)

**Figure 2. Pooling certifies the wrong mixture under heterogeneity.** Empirical coverage of each certificate as the deployment mixture shifts from the calibration-matched mixture toward the high-risk client ($700$ Monte Carlo draws per mixture; $80$ for the box variant).

The population comprises four low-risk clients ($a{=}0.7$, $r{=}0.02$) and one high-risk client ($a{=}0.5$, $r{=}0.3$) at $\delta{=}0.1$. Naive pooled CP was valid only at the matched mixture and collapsed to $0$ under shift: it certified $\approx0.072$ while the true risk reached $0.165$–$0.30$, because the pooled accepted-error count is Poisson-binomial rather than binomial. The stratified certificate (Theorem 1) stayed $\ge1-\delta$ for every mixture. The box-$\Lambda$ certificate (Theorem 2) is by design valid only inside its declared box (shaded region); its drop outside the box was expected and does not contradict its guarantee.

*Uncertified rules are unsafe (necessity).* A practitioner without the certificate has two options, both unsafe. Table 4 collects every validity check in one place: the unsafe-deployment rate $\Pr(\text{deploy}\mid R_{\mathrm{sel}}>\alpha)$ of the alternatives, which any valid method must keep $\le\delta=0.1$, estimated over $2{,}000$ Monte Carlo calibration draws per configuration ($J{=}5$, $n_j{=}300$, $\alpha{=}0.05$; the boundary regime places the true $R_{\mathrm{sel}}$ just above $\alpha$ by sweeping the high-risk client's $r_j$), together with the pooling collapse of Figure 2 and the real-logit resampling study described next.

**Table 4. Validity checks and invalid alternatives** (unsafe rates over $2{,}000$ trials per synthetic cell; resampling row over $526{,}000$ evaluations on stored CIFAR logits).

| check | setting | outcome | conclusion |
|---|---|---|---|
| naive empirical threshold (deploy iff $\hat r\le\alpha$) | boundary synthetic | unsafe rate $0.49$ | invalid |
| leaked split (threshold chosen on the certification fold) | boundary synthetic | unsafe rate $0.18$ | invalid |
| pooled CP under mixture shift | synthetic, Figure 2 | coverage collapses to $0$ | invalid (Prop. 2) |
| **Fed-CORE (proper split)** | boundary synthetic | unsafe rate $0.00$–$0.03$ | **valid** |
| **Fed-CORE, resampling on real logits** | $526$ configs $\times$ $1{,}000$ redraws | violation rate $8.7\times10^{-4}$ (CP95 UCB $1.1\times10^{-3}$) | **valid** |

The naive threshold ignores finite-sample noise and deploys unsafely about half the time at the boundary. The proposal/certification split is load-bearing: re-searching the threshold on the certification fold inflates the deploy rate to $99.8\%$ but the unsafe rate to $18.2\%\gg\delta$, because the certificate cannot correct the multiple-testing of thresholds chosen on the same fold. Only the proper split with the conditional certificate keeps the unsafe rate $\le\delta$. (Federated-CP rules control a quantile, not the accepted risk, so they do not appear here; naive pooled CP is anti-conservative under shift, Figure 2.)

*Resampling validity on real logits.* Training-seed counts alone cannot resolve a false-certificate rate, because the guarantee is a probability over the draw of the certification fold at a *fixed* model. We therefore quantify validity by resampling: for each of the 55 stored CIFAR runs, the held-out pool (certification $\cup$ test folds) is treated as the ground-truth population, per-client audit folds of the original sizes are re-drawn $B{=}1{,}000$ times, and the certificate is recomputed for each draw across $\alpha\in\{0.10,0.20\}$, grouping $G\in\{J,2\}$ (contiguous public groups, fixed a priori), and $\gamma\in\{0.5,0.7,1.0\}$, for $526{,}000$ certificate evaluations in total (CPU-only; script released with the code). The result (Table 4, last row): $70{,}025$ evaluations deployed and $61$ violated the population worst-stratum risk, a violation rate of $8.7\times10^{-4}$ with Clopper–Pearson $95\%$ upper bound $1.1\times10^{-3}$ against the guarantee level $\delta{=}0.10$; the largest per-configuration violation probability over all $526$ configurations is $0.008$. Moreover, $56$ of the $61$ violations occurred at $\gamma{=}1.0$ (proposals selected with no risk buffer, at configurations whose true worst-group risk sits marginally above $\alpha$), whereas the buffered proposals ($\gamma\le0.7$) produced $5$ violations across $69{,}293$ deployments. The risk buffer of Section 3 is therefore not cosmetic: it is what keeps boundary configurations from consuming the entire failure budget. This resampling study is a finite-population empirical check at fixed trained models; it does not replace the theorem, and it does not quantify variability over training itself (Section 6).

### 5.3 Audit representativeness is a condition of validity (A4 stress test)

Assumption A4 requires the audit fold to measure the deployment client-conditional distribution; the practically dangerous way to break it is an audit fold that carries unknown-class points at *less* than their deployment rate. This is a *condition of the guarantee*, so we stress-test it before reporting any certified coverage. (Over-representation behaved conservatively in all our benchmarks, consistent with the dominance heuristic of A4′, but only the matched case is covered by the theorem.)

![](experiments/fedcore/figs/ablation_unknown_prop.png)

**Figure 3. Audit representativeness is required for validity.** Coverage of the true deployment risk as the calibration unknown fraction varies around the deployment fraction of $0.06$ ($3{,}000$ Monte Carlo draws per point).

Under-representation was anti-conservative: coverage fell to $0.522$ at a calibration fraction of $0.04$ and to $0.057$ at $0.02$. Matching preserved coverage ($0.913$ at $0.06$), which is exactly the theorem condition A4. Over-representation was empirically conservative here ($0.992$ at $0.08$), but it is theorem-covered only under the A4′ dominance condition.

The real-data counterpart on CIFAR-10 logits reproduced this collapse (coverage $1.00$ at $\rho{=}1$ down to $0.005$ as unknowns are under-represented). It is therefore not enough that the audit fold *contains* labeled unknowns: it must carry them at no less than the deployment rate. In practice the monitoring set should track, not under-sample, the unknown-class incidence of the live stream; every validity statement below is read under A4 (with over-representation as the A4′ empirical stress protocol).

### 5.4 The finite-sample feasibility law

*The apparent null, and its cause.* On a real CIFAR-10 ladder (12 runs, $\alpha=\delta=0.1$, five clients, small CNN), CertifiedCoverage@$0.1$ was $0$ in every run. Two distinct modes explain this. **Mode 1** (extreme non-IID, $d{=}0.1$): the empirical accepted risk already exceeds $\alpha$, so no method can deploy safely and the certificate correctly declines. **Mode 2** (near-IID, $d{=}5$, test_risk $\approx0.08<\alpha$): the model is safe, but the thin per-client accepted counts cannot drive the upper bound below $\alpha$, which is the Theorem-3 feasibility collapse rather than certificate looseness. Tellingly, shrinking the risk buffer $\gamma$ to lower realized risk *also* starved the accepted set (cert_n $500\to151$, $\approx30<37$/client), which *raised* the bound ($0.185\to0.222$): the binding lever is calibration budget, not the operating point. Dissecting a single run confirmed the mechanism: per-client bounds $\bar r_j\in[0.139,0.190]$ all exceeded $\alpha{=}0.10$ (thus the simplex certificate correctly declined), whereas $G{=}2$ grouping roughly doubled per-stratum counts and lowered the bounds to $0.128$–$0.131$, driven by counts rather than by certificate looseness.

*The staircase.* Re-aggregating the $J{=}5$ clients into $G$ public groups (the grouped-stratified certificate, Section 4.6) raises the per-group accepted count and drives the bound monotonically through $\alpha$.

![](experiments/fedcore/figs/F6_feasibility_law.png)

**Figure 4. The feasibility law (Theorem 3)** on ResNet-GN: (a) per-seed grouped bound versus minimum per-group accepted count for $G\in\{5,3,2,1\}$ at $d{=}5$; (b) CertifiedCoverage@$0.10$ by grouping; (c) coverage and pass rate versus audit budget; (d) client scaling at $J\in\{10,20\}$, CertifiedCoverage@$0.20$ versus grouping $G$ (ten seeds).

Panels (a) and (c) make one point from two directions: *merging clients into groups* and *enlarging the audit budget* are the same Theorem-3 sample-size lever, governed by the $(\alpha-\hat r)^{-2}$ requirement rather than by the operating point. The bound crossed $\alpha{=}0.10$ around several hundred accepted points per group, and CertifiedCoverage@$0.10$ rose from $0$ to $\approx0.21$; on real CIFAR-10 logits, growing the audit fold drove the worst-group bound from $0.58$ to $0.18$, with $\alpha{=}0.10$ becoming non-vacuous ($2/5$ seeds) only at the largest budget. This reflects the usual gap between a mean and a pass rate and the seed variability that keeps $\alpha{=}0.10$ at the feasibility edge for this baseline detector. The risk-target axis behaves the same way: under the most conservative per-client grouping ($G{=}5$, box-$\Lambda$, $d{=}5$), certification became non-vacuous only at larger targets (SimpleCNN $0.063$ at $\alpha{=}0.20$ and $0.193$ at $\alpha{=}0.25$; ResNet-18 $0.316$ at $\alpha{=}0.25$), with $\alpha{=}0.20$ serving as the finite-sample feasibility demonstration point, chosen to exhibit the vacuous-to-non-vacuous transition under realistic audit budgets rather than as a recommended safety threshold. Certification failed below these thresholds not because the certificate is loose but because the accepted audit count is below the finite-sample floor.

*Client scaling.* The same lever governs how the certificate behaves as the federation grows. On real CIFAR-10 (ResNet-GN, $d{=}0.5$, ten seeds) we scaled the client count to $J\in\{10,20\}$ and certified at grouping levels $G\in\{J,J/2,5,2\}$ (Figure 4d). At $\alpha{=}0.20$ the per-client certificate ($G{=}J$) was vacuous on every seed at both scales ($0/10$ for $J{=}10$ and $J{=}20$), because splitting a fixed audit budget across more clients drove each per-client accepted count below the Theorem-3 floor. Coarsening the grouping restored certification monotonically: at $J{=}20$ CertifiedCoverage@$0.20$ rose from $0$ ($G\in\{20,10\}$, $0/10$) through $0.045$ ($G{=}5$, $3/10$) to $0.158$ ($G{=}2$, $6/10$), and at $J{=}10$ it rose from $0$ ($G{=}10$, $0/10$) through $0.089$ ($G{=}5$, $4/10$) to $0.177$ ($G{=}2$, $6/10$). This is the feasibility law on real data at federated scale: per-client certification degrades with $J$ at fixed total audit data exactly as the floor predicts, and the declared grouping of Section 4.6 is the lever that recovers a non-vacuous guarantee rather than a hidden pooling of clients. The hard target $\alpha{=}0.10$ stayed at the feasibility edge ($\le 2/10$ at every grouping), consistent with the baseline-detector budget.

### 5.5 Real detectors and certified coverage

This subsection reports the real-data certification results in one place (Table 5): the FedAvg+MSP baseline, the full FedOSR detectors (FedPD-PROSER, FOOGD), and the edge/negative settings. The strongest real-data positive is FedPD-PROSER, which certifies at the hard target $\alpha{=}0.10$ where the baseline cannot.

*Backbone.* We adopt a CIFAR-stem ResNet-18 with **GroupNorm** as the principled FL normalization (BatchNorm's running statistics diverge under non-IID FedAvg). GroupNorm lowered $\hat r$; however, the accepted set shrank in step, and thus it did not strengthen the seed-variable $\alpha{=}0.10$ certificate. What matters is robust: no held-out risk violation occurred under either normalization.

*Headline.* Unless noted, the results in this subsection use the grouped ($G{=}2$) certificate: the certified object is the mixture over two public groups under assumption A6 and Proposition 3, a declared relaxation of the client-simplex guarantee (Section 4.6), not a rediscovery of pooling, which certifies only the matched mixture and fails silently off it (Proposition 2). A full-simplex per-client certificate on the strongest detector, reported at the end of this subsection, certified five of five seeds with no grouping relaxation; the grouped form is therefore an audit-budget device, not the source of the positives. At $\alpha{=}0.20$ the GroupNorm certificate was non-vacuous with CertifiedCoverage $0.341\pm0.182$ at $d{=}5$ ($4/5$ seeds) and $0.350\pm0.130$ at $d{=}0.5$ (all five seeds), reported at the simultaneous $\delta/2$ budget so that risk and coverage hold jointly, with no held-out violation among certified runs (Table 5). This is a finite-sample feasibility demonstration under the current audit budget, not a safety target. At $\alpha{=}0.10$ the grouped result was seed-variable ($1$–$2/5$), reported as a secondary result. Table 5 carries the certification diagnostics directly: at $\alpha{=}0.20$ the median certified bound sat at $0.156$–$0.163$ with median per-group accepted counts of $783$–$796$ (comfortably past the Theorem-3 floor) and realized test risk of $0.098$–$0.119$, quantifying the finite-sample margin between bound, target, and realization. BatchNorm attained the highest certified coverage at $\alpha{=}0.20$ ($0.428\pm0.048$); however, GroupNorm remains the headline as the principled FL normalization; validity holds under both. The edge and negative cells (lower block of Table 5) are exactly where the feasibility law predicts collapse: extreme non-IID, corruption, or a backbone whose $\hat r$ is near $\alpha$.

**Table 5. Real-data certification diagnostics** (CIFAR-10 unless noted; grouped $G{=}2$ certificate under A6): (a) FedAvg+MSP baseline (five seeds), (b) real FedOSR detectors (five seeds; FOOGD-SAG single seed), (c) edge and negative settings ($\dagger$ single seed).

In Table 5, CertCov abbreviates CertifiedCoverage@$\alpha$ as defined in Section 5.1, "cert." counts certified seeds, and GN/BN abbreviate ResNet-18 with GroupNorm/BatchNorm. In the baseline block (cert\_frac $0.5$), the diagnostics $\bar U$ (worst-group risk bound), $A_g$ (minimum per-group accepted count), and $K_g$ (worst-group accepted-error count) are medians over certified seeds, test risk and test coverage are means over certified seeds, and the $\gamma$ column reports the proposal-fold selections among certified seeds. The accepted counts read directly against the feasibility law: at $\alpha{=}0.20$ the median $A_g$ of $783$–$796$ sits far above the Theorem-3 floor, whereas the $\alpha{=}0.10$ runs certify with roughly half the accepted mass and correspondingly thinner margins. The edge block tags each vacuous cell by failure mode: risk-driven means $\hat r$ sits near or above $\alpha$, whereas count-driven means the accepted counts fall below the Theorem-3 floor. The certified coverage is reported at the simultaneous $\delta/2$ budget (Section 5.1), so risk and coverage hold jointly at $1-\delta$; the diagnostics $\bar U$, $A_g$, and $K_g$ are medians over the certified seeds at that budget. Reporting the coverage at the full budget $\delta$ (a risk-only statement) instead would raise the boundary cell (GN $d{=}5$, $\alpha{=}0.20$) from $0.341$ ($4/5$) to $0.392$ ($5/5$) and the two robust $\alpha{=}0.20$ cells by $0.003$; we take the joint bound as the headline. The detector cells of block (b) are reported at the detector-native protocol and, being robust cells well past the feasibility floor, change by at most $0.01$ under the same $\delta/2$ recompute. Per-run values follow the canonical metric schema and are released with the code.

(a) FedAvg+MSP baseline

| model | $d$ | $\alpha$ | cert. | CertCov | med. $\bar U$ | med. $A_g$ | med. $K_g$ | $\gamma$ | test risk | test cov. |
|---|:-:|:-:|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| GN | 5 | 0.20 | 4/5 | $\mathbf{0.341\pm0.182}$ | 0.160 | 783 | 145 | 0.7 | 0.119 | 0.432 |
| GN | 5 | 0.10 | 1/5 | $0.030\pm0.060$ | 0.088 | 366.5 | 27.5 | 0.3–0.5 | 0.056 | 0.226 |
| GN | 0.5 | 0.20 | 5/5 | $\mathbf{0.350\pm0.130}$ | 0.156 | 789 | 160 | 0.5–0.7 | 0.098 | 0.392 |
| GN | 0.5 | 0.10 | 2/5 | $0.054\pm0.066$ | 0.080 | 387 | 33 | 0.2–0.5 | 0.024 | 0.180 |
| BN | 5 | 0.20 | 5/5 | $0.428\pm0.048$ | 0.163 | 796 | 179 | 0.7 | 0.119 | 0.469 |
| BN | 5 | 0.10 | 3/5 | $0.105\pm0.097$ | 0.075 | 380 | 33 | 0.3–0.5 | 0.040 | 0.208 |

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

covtype is reported as a second domain that exhibited the same feasibility law at its edge, **not** as a positive. With the fixed-MSP protocol it certified $0/5$ at $\alpha{=}0.20$. A *procedurally valid* protocol that selects the score on the proposal fold (A2-compliant), with an enlarged audit budget and grouped $G{=}2$, over ten seeds reached $0.036\pm0.106$ ($1/10$) at $\alpha{=}0.20$, $0.075\pm0.160$ ($3/10$) at $\alpha{=}0.25$, and $0.154\pm0.204$ ($4/10$) at $\alpha{=}0.30$ (per-seed values are released with the reproducibility package): the domain crossed from vacuous toward non-vacuous along the risk-target axis, as the feasibility law predicts for a backbone whose risk sits near the target, but it did not reach a stable positive (the $8/10$ goal was not met), so covtype remains a feasibility-edge domain.

*A second positive dataset: CIFAR-100.* To test whether the certificate holds beyond CIFAR-10 and beyond a single backbone, we ran a multi-model CIFAR-100 grid (sixty known classes; backbones ResNet-GN, ResNet-BN, and SimpleCNN, abbreviated GN/BN/CNN; $d\in\{0.5,5\}$; ten seeds each), summarized in Table 6. At $\alpha{=}0.20$ the grouped ($G{=}2$) certificate was non-vacuous in all six backbone-by-heterogeneity cells, with certified coverage of $0.05$–$0.09$ and $7$–$8/10$ certified seeds per cell, and the realized test risk of $0.08$–$0.11$ stayed below the target in every cell. CIFAR-100 is therefore a second real-data positive rather than a stress-domain negative, and the effect held across three backbones rather than resting on one; the certified coverage is small in absolute terms, so the result is read as a validity and feasibility demonstration on a sixty-class problem, not as a deployable utility claim. As on CIFAR-10, the hard target $\alpha{=}0.10$ stayed at the feasibility edge ($3/60$ certified overall), exactly the risk-target behavior the feasibility law predicts at this audit budget.

**Table 6. CIFAR-100 multi-model certification** (grouped $G{=}2$; ten seeds per cell; per-seed rows are released with the reproducibility package).

| backbone | $d$ | cert. @0.20 | CertCov @0.20 | test risk @0.20 | cert. @0.10 | CertCov @0.10 |
|---|:-:|:-:|---|:-:|:-:|---|
| GN | 0.5 | 8/10 | $0.046\pm0.033$ | 0.084 | 0/10 | $0$ |
| GN | 5 | 8/10 | $0.056\pm0.040$ | 0.094 | 0/10 | $0$ |
| BN | 0.5 | 7/10 | $0.074\pm0.055$ | 0.101 | 1/10 | $0.003\pm0.011$ |
| BN | 5 | 8/10 | $\mathbf{0.093\pm0.059}$ | 0.103 | 1/10 | $0.003\pm0.010$ |
| CNN | 0.5 | 7/10 | $0.052\pm0.046$ | 0.090 | 0/10 | $0$ |
| CNN | 5 | 8/10 | $0.070\pm0.050$ | 0.106 | 1/10 | $0.003\pm0.010$ |

*Deployment knobs.* The two choices a deployer owns, the box radius $\rho$ (Section 4.3) and the proposal buffer $\gamma$ (Section 3), were swept on the ResNet-GN $d{=}5$ logits at $\alpha{=}0.20$. Widening the declared box from $\rho{=}0.05$ to the full simplex changed the certified coverage only marginally ($0.394\to0.386$, all five seeds certified throughout); thus the $\Lambda$ declaration was not the binding lever at this scale. Fixing the proposal buffer to $\gamma{=}1.0$ (no buffer) collapsed certification to $1/5$ seeds where $\gamma{=}0.7$ kept $5/5$, with validity holding at every setting; the buffer, not the box, was the operative deployment choice, consistent with the resampling finding that $56$ of the $61$ observed violations sat at $\gamma{=}1.0$ (Section 5.2).

*Superiority.* At $d{=}5$, $\alpha{=}0.20$ a test-peeking oracle reached accepted coverage of $0.444$ (MSP) and $0.431$ (energy); however, it uses test labels and carries no guarantee, and a no-peek naive threshold attained high coverage near-IID but breached $\alpha$ wherever proposal and test diverged. Fed-CORE retained $0.341\pm0.182$ of that coverage while being the only option that is simultaneously label-honest, safe, and finite-sample-guaranteed.

*Stress axes.* Beyond the risk target $\alpha$, the feasibility law has two further axes, heterogeneity and corruption, shown together in Figure 5; each pushes $\hat r$ past $\alpha$ or starves the per-group count below the Theorem-3 floor.

![](experiments/fedcore/figs/F7_hetero_collapse.png)

**Figure 5. Stress axes of the feasibility law.** (a) Best grouped certified-risk bound ($G{=}2$, $\alpha{=}0.10$) per seed versus Dirichlet concentration $d$ (ResNet-18, stored logits; horizontal marks are medians). (b) Ten-seed mean grouped CertifiedCoverage@$0.20$ versus symmetric training-label noise rate at $d\in\{0.5,5\}$.

As $d$ fell (more non-IID), per-group accepted counts thinned and the grouped bound rose monotonically, sitting near $\alpha$ at $d{=}5$, straddling it at $d{=}0.5$, and moving far above it ($\approx0.30$) at the extreme $d{=}0.1$; the ResNet-GroupNorm headline cells of Table 5 correspond to the feasible end of this axis. On the corruption axis, a ten-seed sweep confirmed the collapse: relative to the clean headline (Table 5), symmetric training-label noise drove the grouped CertifiedCoverage@$0.20$ to a marginal $\approx0.03$ at rate $0.1$ and to $0$ at rates $\ge0.2$, on every seed and at both $d\in\{0.5,5\}$, *although the calibration fold stayed clean*. Corruption raises the model's $\hat r$ above $\alpha$, so a clean audit set has nothing safe left to certify.

*Score-agnosticism.* The guarantee held for **every** score: across MSP, entropy, margin, and energy the realized test_risk stayed $\le\alpha$ ($0.042$–$0.046$) while certified coverage varied ($0.64$–$0.66$ on the synthetic study). The score changed only *how much* coverage was certified, confirming that validity comes from the certification split rather than from score quality (Section 4.7), a point made concrete on real detectors next.

*Two real FedOSR detectors.* To answer the concern that the baseline uses MSP on a FedAvg backbone rather than a genuine FedOSR detector, we certify the native open-set scores of two real FedOSR methods: FedPD's PROSER dummy-vs-known score (a full reproduction) and FOOGD's SM3D score (a representative head on the shared backbone); the middle block of Table 5 reports both, alongside the FedAvg+MSP control and a faithfully reproduced full FOOGD-SAG. *FedPD-PROSER, the strongest base model and a full method.* Trained with the standard recipe, namely a closed-set cross-entropy pretraining stage (known-class accuracy $0.39\to0.65$ over eight rounds, versus $0.26$ stalled from scratch) followed by PROSER fine-tuning on the exact WideResNet-28-10 architecture, its native score reached an AUROC of $0.81$ over five seeds. It was the only detector that certified robustly at the hard target $\alpha{=}0.10$, where FOOGD could not and the baseline was only seed-variable, and the result *strengthened* under heterogeneity: at the more non-IID $d{=}0.5$ it certified $5/5$ seeds at both targets (Table 5). This is a genuine **full** FedOSR method, not a representative head, and it shows that Fed-CORE is a certification layer for real FedOSR detectors: a strong detector plus the certificate reached risk targets that the baseline could not.

*The thesis across base models.* Certified coverage tracked the native-score AUROC (FedPD $0.81$ > MSP $0.74$ > FOOGD-representative $0.68$ > FOOGD-SAG $0.47$), while **validity held in every cell** (no held-out violation in any certified cell across all seeds and base models): exactly the score-agnostic guarantee, with coverage following detector quality and validity independent of it. The FedAvg+MSP control reproduced the baseline headline ($\approx0.40$ at $\alpha{=}0.20$), validating the harness. On reproducibility, FedPD-PROSER required the closed-set-pretrain-then-fine-tune recipe (from scratch it did not converge), whereas full FOOGD-SAG reached only a chance-level AUROC of $0.467$ on our six-known semantic-shift split at a single-GPU budget and is reported as a single-seed negative; FedOSS (a medical-imaging codebase without a CIFAR loader) was deferred. Fed-CORE's validity was unaffected throughout: certified coverage simply followed the base model's score quality. The scope of this evidence should be read precisely: we demonstrate full compatibility with FedPD-PROSER and representative compatibility with FOOGD-SM3D, and we do not claim broad empirical coverage over all FedOSR methods.

*A full-simplex positive rules out hidden pooling.* Because the headline uses the grouped ($G{=}2$) certificate, one might worry that grouping is pooling by another name. The full-simplex certificate (Theorem 1, per-client, $G{=}J$) settles this directly: on FedPD-PROSER at $J{=}5$, $d{=}5$, $\alpha{=}0.20$ with an enlarged audit budget (cert\_frac $0.5$), it certified $5/5$ seeds with mean certified coverage $0.390$ and per-client worst-group risk bounds $\le0.18<\alpha$ (per-client accepted counts $156$–$455$, all past the Theorem-3 floor). This is a genuine client-simplex guarantee, robust to any client mixture in $\Delta^{J-1}$ with no grouping relaxation; the real-data positives are therefore not an artifact of the group-mixture assumption A6.

### 5.6 Downstream use: certified pseudo-label admission

The certificate's first downstream use, safe automation, is CertifiedCoverage@$\alpha$ itself (Section 4.7). The second use is **a certified admission gate, not an accuracy booster**: accepted predictions are folded back into FedAvg as pseudo-labels only when their contamination can be certified below the target. On real CIFAR self-training (ResNet-GN, $d{=}5$), naive self-training kept injecting pseudo-labels whose realized error rate was far above the target ($0.19$–$0.67$ and growing, reaching $0.59$–$0.98$ over longer runs while the certificate correctly output $\bar U{=}1.0$), whereas the certified procedure found the very first round Theorem-3-infeasible (the accepted set was too thin to certify below $\alpha$) and **halted, admitting nothing** rather than an uncertified batch. This is the safe outcome the feasibility law predicts: Fed-CORE never injects a contaminated batch, even when that means admitting none. Over the rounds the Proposition-4 contract ($\delta/T$) was verified (Table 7): over $20{,}000$ Monte Carlo trials (five rounds, $1{,}000$ accepted points per fold), the simultaneous unsafe rate was $0.086\le\delta$ with the round-wise fresh-fold split, whereas reusing one audit fold across adaptive rounds inflated it to $0.386>\delta$, and naive self-training admitted contaminated batches in every round. Round-wise certification is therefore necessary because reusing audit evidence across adaptive rounds breaks the simultaneous guarantee.

**Table 7. Certified pseudo-label admission.**

| scheme | fresh audit fold per round? | admitted contaminated batches | simultaneous unsafe rate | valid? |
|---|---|---|---|---|
| Fed-CORE (round-wise split, $\delta/T$) | yes | $0$ | 0.086 | ✓ |
| reused fold ($\delta$ per round) | no | — | 0.386 | ✗ |
| naive self-training (no certificate) | — | every round (contamination $0.19$–$0.67$) | — | no guarantee |

*A supporting descriptive result.* With the stronger FedPD-PROSER base and a $4\times$ audit budget, certified admission additionally yielded a descriptive known-accuracy gain ($+0.030$, sample standard deviation $0.027$, $2/3$ seeds positive against a clean-pseudo-label oracle of $+0.045$; $0/3$ contamination violations, maximum realized contamination $0.137\le\alpha$). This gain is seed-variable and is not the guarantee; the guarantee is the contamination bound on each admitted batch. We report it only as supporting evidence that the admission gate becomes useful when detector quality and audit budget are sufficient (both levers, plus A4 matching of the unlabeled pool, are required). Self-training is therefore not evidence that Fed-CORE improves accuracy; it is evidence that the certificate prevents unsafe pseudo-label ingestion.

## 6. Limitations

**Conservatism.** The stratified certificate is conservative (Clopper–Pearson exactness + union bound + worst-case mixture); box-$\Lambda$ (Theorem 2) and the pooled diagnostic (Remark 1) recover tightness but require, respectively, knowledge of $\Lambda$ and matched-mixture calibration. The guarantee is marginal over $Q_\lambda$, not conditional per client.

**Trusted calibration (A3–A4).** Certifying unknown rejection requires labeled unknown-class points in the audit fold (necessary by Proposition 1), and the theorems consume an audit fold drawn from the deployment client-conditional distribution (A4). Under-representing unknowns is anti-conservative, as the stress test of Section 5.3 shows; deliberate over-representation behaved conservatively in our benchmarks but is covered by the theorem only under the explicit dominance condition of A4′, not automatically. "Distribution-free" therefore means with respect to the calibration distribution $Q_\lambda$, not the entire unknown universe; in deployment the audit fold corresponds to a small audited monitoring set that must track the live unknown incidence, and Theorem 2 quantifies how scarce it may be before certification becomes infeasible.

**Privacy and grouping.** As characterized in Section 4.6, only the pooled diagnostic is sum-only; the stratified certificate needs per-client (or per-group) counts, and the grouped variant weakens the certified target to group mixtures (A6). In particular, a one-group grouped certificate should not be interpreted as a client-simplex guarantee: it certifies the deployment mixture only at the declared group granularity and under the within-group composition assumption A6. Exactness of the grouped certificate additionally requires the group-mixture sampling condition of Proposition 3; under the per-client audit quotas of our implementation the group counts are conditionally Poisson-binomial, and the grouped results are supported by the resampling evidence of Section 5.2 and by the full-simplex positive of Section 5.5 rather than by an exactness theorem. The privacy claim is an information-flow statement; a differentially private count-release variant is left as future work.

**Self-training use case (B).** What the certificate guarantees is the *contamination* of each injected pseudo-label batch ($\le\alpha$ per round, simultaneously over $T$ rounds by Proposition 4), **not** that self-training necessarily improves accuracy; the accuracy gain is an empirical claim (bounded-noise training is well-behaved but not monotone-guaranteed). Empirically the gain is conditional and seed-variable (Section 5.6: $+0.030$ over three seeds, $2/3$ positive, on a strong detector at $4\times$ audit budget, with one seed feasibility-limited to zero admissions); we therefore frame the contamination gate, not the accuracy gain, as the guarantee. Round-wise audit-fold splitting trades feasibility for adaptivity: $T$ is capped by the trusted-set size via the Theorem-3 per-fold threshold. A reused-fold scheme with formal closed-loop validity (avoiding the $\delta/T$ split) is left as future work.

**Statistical resolution of the validity evidence.** With five training seeds per cell, "0 held-out violations" alone bounds the per-cell violation rate only loosely; the validity evidence in this paper instead rests on the resampling study of Section 5.2, which evaluates the certificate over $526{,}000$ audit-fold redraws on the real logits (violation rate $8.7\times10^{-4}$, CP95 upper bound $1.1\times10^{-3}$ against $\delta{=}0.10$), together with the synthetic Monte Carlo coverage ($\ge0.98$ against the $0.90$ target). What resampling at a fixed model cannot capture is variability over training itself; the per-cell seed counts (Table 5) remain the evidence at that level.

---

## 7. Conclusion

Fed-CORE certifies the accepted selective risk of a federated open-set classifier with a finite-sample, distribution-free guarantee that holds under heterogeneity and unknown deployment mixtures. Its core is a **stratified conditional selective-risk certificate** ($\max_j$ of per-client conditional-binomial CP limits, with a robust bounded-$\Lambda$ form and a certified coverage lower bound) that is valid where naive pooling fails, accompanied by a per-stratum feasibility law. The framework recasts federated open-set recognition from a ranking problem into a *certification* problem: a small trusted audit set is used not to repair the model but to certify which of its predictions are safe to accept; those certified-safe predictions are then *used* for guaranteed-risk automation and for certified federated self-training that expands training with provably-bounded contamination.

Empirically, the theorems' guarantee is read under the stated assumptions (A1–A6), and no certified deployment exhibited a held-out risk violation in any seed, dataset, or normalization. On CIFAR-10 and CIFAR-100, Fed-CORE obtained non-vacuous grouped certificates at $\alpha{=}0.20$ (CIFAR-100 across three backbones and two heterogeneity levels), while the tabular domain (covtype) sat at the feasibility edge and corroborated the sample-size law; naive pooling collapsed under mixture shift exactly as Proposition 2 predicts. Certified coverage is governed by a **feasibility law**: a per-group sample threshold scaling as $(\alpha-\hat r)^{-2}$, traced out by a monotone grouped-stratified staircase, by client scaling to $J{=}20$, and by the heterogeneity and corruption axes. Both positives held under the grouped certificate (a group-mixture guarantee, assumptions A6 and Proposition 3), and a full-simplex certificate (Theorem 1, per-client) on FedPD-PROSER at $J{=}5$ further certified $5/5$ seeds at $\alpha{=}0.20$ with mean coverage $0.39$, confirming that the positives are not an artifact of grouping. Certifying the native score of a full FedOSR method (FedPD-PROSER) reached grouped certified coverage at both risk targets, including $5/5$ seeds at the hard target $\alpha{=}0.10$ under strong heterogeneity, showing that detector quality converted directly into certified coverage at fixed validity. The contribution is therefore the *object and its finite-sample certificate*, the exposure of pooled invalidity under heterogeneity, and the characterization of *when* certified open-set deployment is feasible; it is not a new FedOSR algorithm and does not claim a raw-accuracy gain.

Three directions follow from the limitations of this study. First, reducing the conservatism of the stratified certificate without losing mixture robustness remains open; the matched-mixture pooled diagnostic (Remark 1) marks the tightness that is available when calibration and deployment mixtures coincide, but a valid general tightening requires new arguments. Second, the per-group sample threshold of Theorem 3 grows as $(\alpha-\hat r)^{-2}$ near the boundary, which keeps the $\alpha=0.10$ regime at the feasibility edge; variance-adaptive bounds of the Bernstein or betting type, which replace the worst-case Clopper–Pearson width by an empirical-variance width, are expected to bring small risk targets into the feasible regime at realistic audit budgets. Third, the grouped-stratified certificate releases only per-group counts, which makes a differentially private count-release certificate a concrete next step toward a formal privacy guarantee rather than the information-flow statement given here.

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

[20] W. Hoeffding, On the distribution of the number of successes in independent trials, Ann. Math. Statist. 27 (1956) 713–721.

[21] C. Yang, M. Zhu, Y. Liu, Y. Yuan, FedPD++: Enhanced federated open-set recognition with parameter disentanglement, Int. J. Comput. Vis. (2026). https://doi.org/10.1007/s11263-026-02861-9

[22] J. Zhang, X. Ma, S. Guo, W. Xu, Towards unbiased training in federated open-world semi-supervised learning, in: International Conference on Machine Learning (ICML), 2023.

[23] S. Yu, J. Hong, H. Wang, Z. Wang, J. Zhou, Turning the curse of heterogeneity in federated learning into a blessing for out-of-distribution detection, in: International Conference on Learning Representations (ICLR), 2023.

[24] H. Gao, Y. Liu, Z. Qin, W. Ou, Noise-resistant federated open set recognition, in: Knowledge Science, Engineering and Management (KSEM 2025), Lecture Notes in Computer Science, vol. 15920, Springer, Singapore, 2026, pp. 1–13.

[25] L. Zhang, M. Wan, P. Huang, G. Yang, Adversarial compact wrapping classifier learning for open set recognition, Inf. Sci. 680 (2024).

[26] Z. Dai, G. Shen, H. Yuan, S. Zheng, Y. Hu, J. Du, X. Kong, F. Xia, Towards heterogeneous federated graph learning via structural entropy and prototype aggregation, Inf. Sci. 718 (2025) 122338.

[27] M. Ren, Z. Wang, X. Yu, Personalized federated learning: A clustered distributed co-meta-learning approach, Inf. Sci. 647 (2023) 119499.

[28] X. Li, S. Zhao, C. Chen, Z. Zheng, Heterogeneity-aware fair federated learning, Inf. Sci. 619 (2023) 968–986.

[29] Z. Pan, C. Li, F. Yu, S. Wang, X. Tang, J. Zhao, Balancing the trade-off between global and personalized performance in federated learning, Inf. Sci. 712 (2025) 122154.

[30] X. Yu, Z. Liu, W. Wang, Y. Sun, Clustered federated learning based on nonconvex pairwise fusion, Inf. Sci. 678 (2024) 120956.

[31] H. Yang, W. Xi, Z. Wang, Y. Shen, X. Ji, C. Sun, J. Zhao, FedRich: Towards efficient federated learning for heterogeneous clients using heuristic scheduling, Inf. Sci. 645 (2023) 119360.

[32] X. Zhou, G. Yang, Communication-efficient and privacy-preserving large-scale federated learning counteracting heterogeneity, Inf. Sci. 661 (2024) 120167.

[33] F. Kamiran, S. Mansha, A. Karim, X. Zhang, Exploiting reject option in classification for social discrimination control, Inf. Sci. 425 (2018) 18–33.

[34] X. Du, J. Yu, Z. Chu, L. Jin, J. Chen, Graph autoencoder-based unsupervised outlier detection, Inf. Sci. 608 (2022) 532–550.

[35] R.A. Coelho, L.C.B. Torres, C.L. de Castro, Concept drift detection with quadtree-based spatial mapping of streaming data, Inf. Sci. 625 (2023) 578–592.

[36] A. Farinhas, C. Zerva, D. Ulmer, A.F.T. Martins, Non-exchangeable conformal risk control, in: International Conference on Learning Representations (ICLR), 2024. arXiv:2310.01262.

[37] S. Bates, A.N. Angelopoulos, L. Lei, J. Malik, M.I. Jordan, Distribution-free, risk-controlling prediction sets, J. ACM 68 (2021) 1–34.

[38] X. Yu, J. Liu, A joint finite-sample certificate for adaptive selective conformal risk control, arXiv:2606.08517, 2026.

[39] M. Zhu, J. Liao, J. Liu, Y. Yuan, FedOSS: Federated open set recognition via inter-client discrepancy and collaboration, IEEE Trans. Med. Imaging 43 (1) (2024) 190–202.

[40] D. Hendrycks, K. Gimpel, A baseline for detecting misclassified and out-of-distribution examples in neural networks, in: ICLR, 2017.

[41] W. Liu, X. Wang, J.D. Owens, Y. Li, Energy-based out-of-distribution detection, in: NeurIPS, 2020.

[42] Y. Geifman, R. El-Yaniv, SelectiveNet: A deep neural network with an integrated reject option, in: ICML, 2019.

[43] R. El-Yaniv, Y. Wiener, On the foundations of noise-free selective classification, J. Mach. Learn. Res. 11 (2010) 1605–1641.
