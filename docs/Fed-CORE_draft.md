# Fed-CORE: Federated Certified Open-Set Recognition via Selective Risk Control

Sanghoon Kim

[Department / University, City, Country]

Corresponding author. E-mail address: october25kim@gmail.com

---

## Abstract

Federated learning is increasingly deployed in safety-sensitive applications where test inputs may belong to classes never seen during training. Existing federated open-set recognition methods evaluate unknown rejection only empirically (AUROC, FPR95), and federated conformal prediction certifies closed-set prediction-set coverage; neither bounds the error rate of the predictions a deployed model accepts. The purpose of this study is to certify the accepted selective risk of a federated open-set classifier — the probability that an accepted prediction is wrong — with a finite-sample, distribution-free guarantee under client heterogeneity and an unknown deployment mixture. We propose Fed-CORE, a post-hoc certification layer for any federated open-set model. Pooling accepted calibration points across heterogeneous clients is anti-conservative, because the pooled accepted-error count is Poisson-binomial rather than binomial; Fed-CORE instead bounds each client's conditional selective risk from low-dimensional accepted/error counts and certifies the global risk as a mixture-robust worst case, together with a per-client feasibility threshold that characterizes when certification is possible at all. Across synthetic and real federated benchmarks, certified deployments exhibit no held-out risk violation under an audit-representativeness condition, naive pooling collapses under mixture shift as predicted, and certified coverage follows the feasibility law. On CIFAR-10, Fed-CORE certifies the native score of a full FedOSR method (FedPD-PROSER) at risk targets 0.10 and 0.20, and a five-seed FedAvg baseline at 0.20.

**Keywords:** Federated learning; Open-set recognition; Selective risk control; Conformal prediction; Distribution-free certification; Uncertainty quantification

---

## 1. Introduction

Consider many hospitals that jointly train one diagnostic model without sharing any patient record. Once the model is in use it will sometimes meet a disease it never saw during training, so it must be honest enough to answer "I do not know" instead of forcing a guess. Two practical questions then follow. First, among the cases where the model *does* commit to an answer, how often is it wrong? Second, can we *promise* — with statistical confidence — that this error rate stays below a chosen tolerance, even though every hospital's data looks different? This paper provides exactly that promise. It is computed from a small amount of trusted, correctly labelled audit data, and it does not change or retrain the model; we call the framework Fed-CORE. The rest of this introduction makes each of these ideas precise, but the reader can keep this plain picture in mind throughout: *Fed-CORE does not try to make the model better — it certifies which of the model's answers are safe to trust.*

Federated learning trains a shared model across clients that never disclose their raw data. In practice these models are deployed in *open-world* conditions: a fraud detector meets a new fraud pattern, a clinical model meets a disease subtype absent from every hospital's training data. A deployed model must therefore not only classify known classes but also **abstain** on inputs it cannot safely classify. This is open-set recognition (OSR), and its federated form (FedOSR) is now an active area.

The dominant FedOSR methods — parameter-disentanglement aggregation (FedPD, FedPD++) [1,2], score-model OOD detection (FOOGD) [3], open-set voting (FedOV) [4], and novel-class discovery (FedNovel) [5] — improve the *quality* of unknown rejection and report it through ranking metrics such as AUROC and FPR95. None of them answers the question a deployer actually needs answered: **if I accept and act on this model's confident predictions, what is the worst-case error rate among them, and can I guarantee it stays below a tolerance α?** Ranking metrics do not answer this; a model with high AUROC can still have an unacceptable error rate among accepted predictions at any fixed operating threshold, and that threshold is exactly what deployment fixes.

A second, separate literature does provide finite-sample guarantees in FL: federated conformal prediction (FCP) [6] constructs prediction sets with marginal coverage under a partial-exchangeability relaxation of the i.i.d. assumption, and recent variants add robustness to label shift [7] and to Byzantine clients [8]. However, FCP is closed-set: it assumes the true label is among the known classes and guarantees that the label lies in the returned set. It does not reject unknowns, and — as its own authors note — its selective-classification demonstrations are heuristics *without* a guarantee. Coverage of a prediction set and the *risk of an accepted point prediction* are different functionals; one does not imply the other.

A third body of work certifies unknown rejection — conformal novelty detection with false-discovery-rate (FDR) control, conformal open-set classification, and selective conformal risk control — but almost entirely in the **centralized** setting. The one recent decentralized entrant controls **batch FDR** over a set of test points in a pure novelty-detection (inlier/outlier) framing, with no known-class classifier and no notion of accepted selective risk. FDR over a test batch and the selective risk of a classifier's accepted predictions are, again, different objects requiring different finite-sample machinery.

The intersection — **a federated, heterogeneity-aware, finite-sample certificate on the accepted selective risk of an open-set classifier** — is empty. Filling it is the contribution of this paper.

**Why this is hard, and not a trivial combination.** It is tempting to view the problem as "apply a Clopper–Pearson selective-risk certificate (as in the centralized i.i.d. case) to the pooled federated calibration data." This is *invalid*. Under heterogeneity, accepted calibration points from different clients have different conditional error probabilities $r_j$. Consequently the pooled accepted-error count, conditioned on the pooled accepted count, is a sum of binomials with unequal success probabilities (a Poisson-binomial), not a single binomial — so the standard Clopper–Pearson construction does not apply, and using it can be anti-conservative when the deployment mixture overweights high-error clients. In plain terms: if we lump every client's audit data into one pile, a few reliable clients can mathematically "average away" the mistakes of an unreliable one, so the guarantee looks safer than it truly is. The correct object is the deployment-mixture-weighted ratio
$$ R_{\mathrm{sel}}(\lambda)=\frac{\sum_j \lambda_j\, m_j}{\sum_j \lambda_j\, a_j},\qquad m_j=\Pr_{P_j}(\text{accept}\wedge\text{error}),\ a_j=\Pr_{P_j}(\text{accept}), $$
with deployment weights $\lambda$ that are unknown at calibration time. Intuitively, $R_{\mathrm{sel}}(\lambda)$ is just this: of all the predictions the model chooses to act on, what fraction are wrong — measured under the (unknown) mix of clients that will actually appear at deployment. Certifying this ratio, finite-sample, distribution-free, and robustly over unknown $\lambda$, is a genuinely new statistical problem that reduces to *neither* the single-binomial certificate of the centralized case *nor* the quantile-coverage certificate of federated conformal prediction.

The purpose of this study is to certify, with a finite-sample distribution-free guarantee, the accepted selective risk of a federated open-set classifier under client heterogeneity and unknown deployment mixture, without retraining the model. Our hypothesis is that a small trusted clean audit set, although too small to repair a heterogeneous global model, is sufficient to certify which of its predictions can be accepted safely, and that the achievable certified coverage is governed by a finite-sample feasibility law rather than by the model ranking quality. We test this hypothesis through a conditional-binomial certificate, a controlled study of when the certificate is non-vacuous, and two downstream uses evaluated on synthetic and real federated benchmarks.

The main contribution of this study can be summarized as follows:

- We formalize the federated accepted selective risk $R_{\mathrm{sel}}(\lambda)$ as the certification target for federated open-set recognition, and show that it is a different functional from prediction-set coverage, ranking metrics, and batch false discovery rate (Section 3).
- We derive a finite-sample, distribution-free certificate for $R_{\mathrm{sel}}(\lambda)$ from the conditional law $K_j\mid A_j\sim\mathrm{Bin}(A_j,r_j)$, valid under client heterogeneity and unknown deployment mixture (Theorem 1/1′), together with a per-client feasibility threshold (Theorem 2) and a privacy/communication characterization of the certificate variants; we prove (Proposition 2) that naive pooling of federated calibration counts is anti-conservative under mixture shift and is therefore not reducible to centralized conformal prediction (Section 4).
- We characterize when certified open-set deployment is statistically feasible through a feasibility law in the risk target, the per-group audit count, heterogeneity, and corruption, turning the apparent null at small risk targets into a quantitative phenomenon (Sections 4.3 and 5).
- We present two downstream uses of the certificate, namely safe automation at guaranteed risk and certified federated self-training with bounded contamination (Proposition 4), and evaluate them on synthetic and real federated benchmarks, where no certified deployment exhibits a held-out risk violation under the audit-representativeness assumption (Sections 4.7 and 5).

Positioning note: Fed-CORE is best read not as a new FedOSR algorithm but as a **certification layer for any FedOSR / open-set FL model**, whose output is *used* for safe automation and certified self-training.

**Findings (preview).** Across synthetic and real federated benchmarks (CIFAR-10/100 and a tabular FL task), no certified deployment exhibits a held-out risk violation in any seed, data set, or normalization — under the audit-representativeness assumption (A4, Section 3) — and naive pooling collapses under mixture shift as predicted. Certified coverage obeys a **feasibility law** ($(\alpha-\hat r)^{-2}$ per-group sample threshold). Two positives result. First, certifying the native score of a full FedOSR method, FedPD-PROSER, yields worst-group certified coverage at *both* risk targets — $0.483\pm0.100$ at $\alpha{=}0.20$ and $0.210\pm0.089$ ($3/3$ seeds) at the harder $\alpha{=}0.10$ under strong heterogeneity ($d{=}0.5$) — showing that a strong detector plus the certificate reaches practically relevant targets. Second, a five-seed FedAvg+MSP baseline certifies at $\alpha{=}0.20$ on CIFAR-10 ResNet-GroupNorm ($0.392\pm0.097$ at $d{=}5$, $0.353\pm0.130$ at $d{=}0.5$, both $5/5$). A second federated domain (tabular FL) sits at the feasibility edge, corroborating the law. The contribution is a *characterized, valid certificate*, not a headline accuracy number.

The remainder of this paper is organized as follows. Section 2 reviews related work in federated open-set recognition, federated conformal prediction, and conformal risk control. Section 3 formalizes the problem and the accepted selective risk. Section 4 develops the certificate, its feasibility threshold, the privacy taxonomy, and the two downstream uses. Section 5 reports the experiments, and Section 6 discusses limitations. Section 7 concludes.

---

## 2. Related Work

**Federated open-set / novel-class / OOD recognition.** FedPD [1] and its extension FedPD++ [2] frame FedOSR around two pathologies — cross-client *inter-set interference* between closed- and open-set objectives, and cross-client *intra-set inconsistency* from heterogeneity — and address them by parameter disentanglement and divide-and-conquer aggregation. FOOGD [3] jointly targets OOD generalization and detection: its SM³D module learns a feature-space score model (detection rule: threshold on the score norm), and its SAG module regularizes feature invariance via Stein's identity. Crucially, FOOGD's only formal result bounds the *estimation error of the score model* (an MMD bound), not the error rate of the rejection decision; detection is reported via AUROC/FPR95. FedOV [4] tackles one-shot FL under label skew by training each client to emit an "unknown" class and ensembling via open-set voting. FedNovel [5] and federated open-world semi-supervised learning [9] discover and learn novel classes across clients, and client heterogeneity itself has been recast as a signal for OOD detection (FOSTER) [10]. Noise-Resistant FedOSR [11] is the closest to a corruption setting, using Bayesian uncertainty and label correction — but, like all of the above, evaluates rejection empirically. Centralized OSR continues to refine compact accept/reject regions [12], and federated heterogeneity is studied beyond images, for example on graphs [13]. Adjacent lines address individual facets of deployment reliability — personalization and fairness under statistical heterogeneity [14,15,16], client clustering and scheduling [17,18], communication-efficient private aggregation [19], reject-option classification [20], outlier detection [21], and concept-drift detection [22]. **None of these certifies a finite-sample bound on the accepted-prediction error rate.**

**Federated conformal / distribution-free uncertainty.** Lu et al. [6] introduce Federated Conformal Prediction, replacing exchangeability with *partial exchangeability* (a test point matches client $k$ with probability $\lambda_k$) and proving marginal coverage $1-\alpha \le \Pr(Y\in C) \le 1-\alpha + K/(N+K)$ with a privacy-preserving T-Digest quantile sketch. Plassier et al. [7] handle label shift via importance weighting; Rob-FCP [8] adds Byzantine robustness; generative and conditional variants refine conditional coverage. All certify **closed-set prediction-set coverage**, not unknown rejection or selective risk. We adopt the partial-exchangeability viewpoint but certify a *risk* (a binomial functional), which behaves differently from a *quantile* and is not addressed by these works.

**Centralized selective-risk certification.** Conformal Risk Control [23] generalizes conformal coverage to any monotone risk, building on distribution-free risk-controlling prediction sets [24]. A fast-moving 2025–2026 line then certifies *post-selection* risk: Selective Conformal Risk Control [25] two-stage selects then controls risk on the selected subset; SCoRE [26] controls risk among "trusted/positive" cases via e-values; and a recent joint finite-sample certificate [37] simultaneously bounds the selected risk ratio, the acceptance probability, and a deployment utility for an adaptively chosen selector. These are the closest statistical neighbors to Fed-CORE, and they certify the same *kind* of object — a post-selection risk ratio. But all of them assume **centralized, exchangeable (i.i.d.) calibration from a single population**: none addresses client-stratified calibration, heterogeneous per-client error rates, an unknown deployment mixture $\lambda$, or the count-release constraints of federation — precisely the axes on which pooled calibration becomes invalid (Proposition 2). Fed-CORE is therefore best read as the federated, mixture-robust counterpart of this line, not as a competitor to it; we use these methods as **centralized oracles** (upper bounds), not as drop-in baselines.

**Conformal open-set / novelty detection.** Conformal open-set classification via Good–Turing p-values [27] and reject-option conformal classification [28] certify rejection centrally. The sole decentralized entry, *Decentralized Conformal Novelty Detection* [29], controls **global FDR over a test batch** via quantized surrogate scores — pure novelty detection (no known-class classifier), a different functional than accepted selective risk; we use it as the nearest neighbor and differentiate on object and on certifying a single fixed selector.

**Positioning.** Existing selective conformal/risk-control methods certify post-selection risk in centralized/exchangeable settings; federated conformal methods (including FedOSS-style FedOSR [30] and FCP [6]) certify either empirical rejection quality or closed-set prediction-set coverage. Table 1 organizes the prior work by the object it certifies. Fed-CORE fills the missing intersection: **federated open-set accepted-risk certification under client heterogeneity and deployment-mixture uncertainty.** It contributes the conditional selective-risk certificate (and its mixture-robust form) that the federated setting newly requires; the calibration statistic is a conditional-binomial proportion, not a score quantile.

**Table 1. Prior work organized by certified object.** Fed-CORE occupies the previously empty intersection of the first four columns.

| method family | federated? | open-set? | accepted point-prediction risk? | unknown deployment mixture $\lambda$? | calibration statistic released | finite-sample? |
|---|---|---|---|---|---|---|
| FedPD / FedOSS / FOOGD [1,2,3,30] | ✓ | ✓ | ✗ (AUROC/FPR95 only) | ✗ | n/a | ✗ |
| FCP and variants [6,7,8] | ✓ | ✗ (closed-set coverage) | ✗ | partial exchangeability | score quantile sketch | ✓ |
| CRC / SCRC / SCoRE / joint certificate [23,25,26,37] | ✗ | optional | ✓ | ✗ | n/a (centralized) | ✓ |
| decentralized novelty FDR [29] | decentralized | novelty only | ✗ (batch FDR) | limited | quantized score functions | ✓ |
| **Fed-CORE (this work)** | ✓ | ✓ | ✓ | ✓ ($\lambda\in\Lambda$) | per-client/per-group counts | ✓ |

---

## 3. Problem Setup

![Figure 1](experiments/fedcore/figs/fig0_problem_diagram.png)

**Figure 1. What Fed-CORE certifies — and what it is not.** A federated open-set model $\hat h$ plus a selector $A$ partitions a deployment stream into *accepted* (predict a known class) and *rejected* (unknown). Four different quantities can be asked about this stream, and they are *not* interchangeable: AUROC/FPR95 measure the *ranking* of the unknown score over all points; federated conformal prediction guarantees *prediction-set coverage* for known classes (closed-set); batch FDR controls false novelties over a test batch; **Fed-CORE controls $R_{\mathrm{sel}}(\lambda)=\Pr(\hat y\ne Y\mid \text{accept})$ — the error rate among the predictions one actually acts on — under the deployment mixture $\lambda$.** Coverage of a set, ranking quality, and batch FDR do not bound this accepted-prediction error rate; Fed-CORE does, in the federated setting, without pooling calibration data across heterogeneous clients. The diagram also shows where the trusted folds enter: the **proposal fold** selects the threshold $t$ of the selector $A$, and a disjoint **certification fold** certifies $R_{\mathrm{sel}}$ independently of that choice.

**Clients and mixture.** There are $J$ clients; client $j$ has data distribution $P_j$ over $\mathcal{X}\times\mathcal{Y}$, where $\mathcal{Y}=\{1,\dots,C\}\cup\{\textsf{unknown}\}$. Deployment data follow a mixture $Q_\lambda=\sum_{j=1}^J \lambda_j P_j$ for some weight vector $\lambda\in\Delta^{J-1}$. We allow $\lambda$ to be **unknown at calibration time**, constrained only to a known convex set $\Lambda\subseteq\Delta^{J-1}$ (e.g., $\Lambda=\Delta^{J-1}$, or a box around client data fractions).

**Open-set decision.** A federated-trained classifier $\hat h$ produces a point prediction $\hat y(x)\in\{1,\dots,C\}$. A *selector* $A:\mathcal{X}\to\{0,1\}$ decides acceptance: $A(x)=1$ means "accept and act on $\hat y(x)$ as a known-class prediction"; $A(x)=0$ means "reject as unknown / abstain." $A$ may threshold any score $s(x)$ (maximum softmax probability (MSP) [31], entropy, margin, energy [32]); the guarantee will not depend on which. Such a selector is the open-set analogue of selective classification with a learned reject option [33].

**Target risk.** The quantity to control is the **accepted selective risk** under deployment:
$$
R_{\mathrm{sel}}(\lambda)\;=\;\Pr_{(X,Y)\sim Q_\lambda}\!\big(\hat y(X)\ne Y \,\big|\, A(X)=1\big)
\;=\;\frac{\sum_{j}\lambda_j\, m_j}{\sum_{j}\lambda_j\, a_j},
$$
where $a_j=\Pr_{P_j}(A(X)=1)$ is the per-client acceptance rate and $m_j=\Pr_{P_j}(A(X)=1,\ \hat y(X)\ne Y)$ is the per-client accepted-error mass (so the per-client selective risk is $r_j=m_j/a_j$). The ratio form follows from the law of total probability under the mixture. The companion quantity is the **accepted coverage** $C(\lambda)=\sum_j\lambda_j a_j$, the fraction of the deployment stream that is accepted. **Goal:** deploy $A$ only if we can certify $R_{\mathrm{sel}}(\lambda)\le\alpha$ for the (unknown) deployment $\lambda\in\Lambda$ with confidence $1-\delta$, while maximizing $C(\lambda)$. Accordingly, the certificate outputs two numbers: a risk upper confidence bound $\bar U_\Lambda$ (deploy iff $\bar U_\Lambda\le\alpha$) and a coverage lower confidence bound $\underline C_\Lambda$; the experimental metrics `cert_risk_ucb` and `cert_coverage_lcb` report exactly these two quantities.

**Trusted calibration data and the split.** Each client holds a small *trusted, clean* calibration sample, partitioned into a **proposal** fold and a **certification** fold. The selector $A$ is chosen on the proposal fold (across clients) and is therefore *fixed and independent* of the certification fold. On the certification fold, client $j$ contributes $n_j$ i.i.d. draws from $P_j$ and reports two integers:
$$
A_j=\sum_{i=1}^{n_j}\mathbf 1\{A(x_i)=1\},\qquad
K_j=\sum_{i=1}^{n_j}\mathbf 1\{A(x_i)=1,\ \hat y(x_i)\ne y_i\}.
$$
By construction $A_j\sim\mathrm{Bin}(n_j,a_j)$ and $K_j\sim\mathrm{Bin}(n_j,m_j)$ with $K_j\le A_j$; conditionally, $K_j\mid A_j\sim\mathrm{Bin}(A_j,r_j)$. What must leave the client depends on the certificate (per-client pairs for the stratified one, sums for the pooled one); see the privacy characterization in Section 4.4. A third, held-out **test fold** estimates deployment behavior *after* certification; it is never used to select or certify a selector, and all `test_*` quantities in Section 5 are deployment estimates, not inputs to the guarantee.

**Calibration must contain labeled unknowns.** To certify *unknown rejection*, the certification fold must include points with $Y=\textsf{unknown}$ that are **labeled as such**: unknown classes are unseen during training but present and labeled in this small post-training **audit/calibration fold**. Hence "distribution-free" is *with respect to the calibration distribution $Q_\lambda$*, not the entire unknown universe. In OSR benchmarks this holds by construction (held-out classes excluded from training, used as unknown-labeled calibration/test examples); in deployment it corresponds to a small audited monitoring set. If the audit fold has no unknowns, the certificate degrades to a closed-set selective-risk guarantee. We certify only selectors with positive accepted coverage $a_\lambda=\sum_j\lambda_j a_j>0$; a zero-coverage selector is non-deployable.

The need for trusted labels is not an artifact of our construction; it is information-theoretically unavoidable.

**Proposition 1 (necessity of trusted calibration labels).** *No procedure that observes only the model's outputs together with corrupted or unlabeled client data can certify $R_{\mathrm{sel}}(\lambda)\le\alpha$ for $\alpha<1$, distribution-free, while deploying with nontrivial acceptance.*

*Proof sketch.* Construct two worlds with identical observable data: in World A the clean labels of the accepted points agree with the model's predictions; in World B they are adversarially flipped, while the corrupted/unlabeled observations and all model outputs are unchanged. Any procedure sees the same input in both worlds and must behave identically; if it deploys with acceptance bounded away from zero, its certificate fails in World B with probability one. Hence either the procedure abstains vacuously or its guarantee is violated. Trusted clean labels break the indistinguishability. $\square$ (Full proof in Appendix D.)

**Risk-buffered proposal.** To avoid certifying a selector whose empirical risk already sits at $\alpha$ (which makes the certificate fail), the proposal fold selects $A$ subject to an empirical buffer $\widehat R_{\mathrm{prop}}(A)\le\gamma\alpha$ with $0<\gamma<1$ (default candidates $\gamma\in\{0.5,0.7,1.0\}$), inherited from the centralized framework.

**Assumptions (collected).** All guarantees in this paper are read under the following assumptions, each referenced where it is used.

- **A1 (within-client i.i.d.).** Certification draws on client $j$ are i.i.d. from $P_j$.
- **A2 (selector independence).** The selector $A$ is fixed on the proposal fold, independently of the certification fold.
- **A3 (trusted labels).** Certification-fold labels are clean, including unknown-class points labeled as such (necessary by Proposition 1).
- **A4 (audit representativeness).** The audit distribution matches the deployment client-conditional distribution, or is conservative for the unknown incidence: the audit fold carries unknowns at no less than their deployment rate. Under-representation makes the certificate anti-conservative — this is a validity condition, stress-tested in Section 5.3.
- **A5 (mixture set).** The deployment mixture satisfies $\lambda\in\Lambda$ for the declared $\Lambda$ (trivial when $\Lambda=\Delta^{J-1}$).
- **A6 (grouped certificates).** When clients are merged into public groups (Section 4.4), the certified object is the mixture over *groups*; the group-level audit sample must represent the within-group deployment composition. A grouped certificate does not protect against arbitrary *within-group* client-mixture shift — it is a deliberate relaxation of the client-level guarantee, not a free tightening.

---

## 4. Method

The method has four plain ingredients, built up in this section. (1) A classical, exact way to turn a count of errors into a high-confidence upper bound on an error rate (Section 4.1). (2) The core certificate, which applies that bound to each client's accepted audit points and combines the per-client bounds into a single guarantee that is robust to the unknown deployment mix (Section 4.2). (3) A feasibility condition that says how much audit data each client needs before a guarantee is even possible (Section 4.3). (4) The practical consequences — what information must leave each client (privacy, Section 4.4), an optional tighter variant (Section 4.5), why the guarantee does not depend on the chosen confidence score (Section 4.6), and what the certified predictions are then used for (Section 4.7).

### 4.1 Clopper–Pearson primitives

For $K\sim\mathrm{Bin}(n,p)$, the one-sided **upper** Clopper–Pearson limit [34] at level $\varepsilon$ is
$$
U^+(K,n;\varepsilon)=\mathrm{BetaInv}\big(1-\varepsilon;\,K+1,\,n-K\big)\quad(\,=1\text{ if }K=n\,),
$$
and the one-sided **lower** limit is
$$
L^-(K,n;\varepsilon)=\mathrm{BetaInv}\big(\varepsilon;\,K,\,n-K+1\big)\quad(\,=0\text{ if }K=0\,).
$$
These satisfy, for every $p$, $\Pr\!\big(p\le U^+(K,n;\varepsilon)\big)\ge 1-\varepsilon$ and $\Pr\!\big(p\ge L^-(K,n;\varepsilon)\big)\ge 1-\varepsilon$, exactly and distribution-free.

### 4.2 Theorem 1 — Conditional selective-risk certificate

In words, the certificate does the following. For each client we look only at the audit points that the model actually accepted, count how many of those were wrong, and use an exact binomial confidence bound (Section 4.1) to obtain an upper bound on that client's true error rate. The global guarantee is then driven by the worst client, because — as explained above — no client's errors can be hidden behind another's. The rest of this subsection makes this precise.

The sharpest certificate works directly with the per-client **conditional** selective risk $r_j=\Pr_{P_j}(\hat y(X)\ne Y\mid A(X)=1)=m_j/a_j$. Conditional on the accepted count $A_j$, the accepted-error count is exactly $K_j\mid A_j\sim\mathrm{Bin}(A_j,r_j)$, so a single Clopper–Pearson upper limit on the *observed accepted sub-sample* bounds $r_j$ with **no acceptance-rate slack**:
$$
\bar r_j=U^+(K_j,A_j;\varepsilon)\qquad(\bar r_j:=1\text{ if }A_j=0).
$$
The object being certified is not a collection of per-client error rates but their *acceptance-reweighted mixture*, which the following decomposition makes precise.

**Lemma 1 (acceptance-reweighted decomposition).** *For every $\lambda$ with $\sum_j\lambda_j a_j>0$,*
$$
R_{\mathrm{sel}}(\lambda)=\sum_j w_j(\lambda)\,r_j,\qquad
w_j(\lambda)=\frac{\lambda_j a_j}{\sum_\ell \lambda_\ell a_\ell},\quad \sum_j w_j(\lambda)=1 .
$$

Lemma 1 is elementary, but it fixes what a federated certificate must control: a convex combination of the per-client conditional risks with *unknown, acceptance-coupled* weights $w_j(\lambda)$ — not a single error rate, and not a max of unrelated intervals.

**Theorem 1 (full simplex).** *Take $\varepsilon=\delta/J$, assume within-client i.i.d. certification samples and a selector $A$ independent of the certification fold. With $\Lambda=\Delta^{J-1}$ and $\bar U_\Delta^{\,r}=\max_j \bar r_j$,*
$$
\Pr\big(R_{\mathrm{sel}}(\lambda)\le \bar U_\Delta^{\,r}\ \text{ for all }\lambda\in\Lambda\big)\ \ge\ 1-\delta .
$$

This uses **one** event per client (not two) and **no** acceptance lower bound, so it is uniformly tighter than the mass-ratio bound $\max_j \bar m_j/\underline a_j$ (kept as a valid but looser baseline in Appendix B). The interpretation is unchanged — the **worst client sets the bar**, no client's error can be averaged away — but the constant is smaller.

**Theorem 1′ (bounded $\Lambda$, robust certificate — recommended for deployment).** *When $\Lambda$ is a known strict subset (e.g. a box around public client data fractions), the worst-client domination is avoided. At level $\varepsilon=\delta/3J$ bound $r_j\le\bar r_j$ and $a_j\in[\underline a_j,\bar a_j]$ with $\underline a_j=L^-(A_j,n_j;\varepsilon)$, $\bar a_j=U^+(A_j,n_j;\varepsilon)$, and set*
$$
\bar U_{\Lambda}^{\,r,a}=\sup_{\lambda\in\Lambda,\ a_j\in[\underline a_j,\bar a_j]}\ \frac{\sum_j \lambda_j a_j \bar r_j}{\sum_j \lambda_j a_j}.
$$
*Then $\Pr(R_{\mathrm{sel}}(\lambda^\star)\le \bar U_\Lambda^{\,r,a})\ge1-\delta$ for the true $\lambda^\star\in\Lambda$.* The inner supremum over $a$ is attained at a box vertex ($a_j\in\{\underline a_j,\bar a_j\}$); the outer supremum is a small linear-fractional program (Charnes–Cooper / Dinkelbach).

*Choosing $\Lambda$ in practice.* Theorem 1′ requires a declared $\Lambda$ (assumption A5), and the deployer — not the certificate — owns this choice. A concrete protocol: take the publicly known client data or traffic fractions $\hat\lambda$ (already exchanged for FedAvg weighting), declare the box $\Lambda=\prod_j[\hat\lambda_j(1-\rho),\ \hat\lambda_j(1+\rho)]\cap\Delta^{J-1}$ with a margin $\rho$ chosen from domain knowledge about how far deployment traffic can drift (we use $\rho=0.15$), and fall back to the full simplex (Theorem 1) when no defensible $\hat\lambda$ exists. Declaring $\Lambda$ too narrow voids the guarantee outside it; the simplex certificate remains the assumption-free default.

**Edge cases (in the statement, not the proof).** (i) *Zero accepted coverage.* We certify only selectors with $a_\lambda=\sum_j\lambda_j a_j>0$; if $a_\lambda=0$ the selector accepts nothing and is **non-deployable** ($R_{\mathrm{sel}}$ undefined). (ii) *Vanishing denominator bound.* If $\inf_{\lambda\in\Lambda}\sum_j\lambda_j\underline a_j=0$ (a client may accept yet $\underline a_j=0$ when $A_j$ is small) the robust certificate is declared **infeasible** ($\bar U=+\infty$); we do *not* silently drop such a client, which would break the worst-case guarantee.

**Proof sketch (Theorem 1).** Conditional on $A_j$, $K_j\sim\mathrm{Bin}(A_j,r_j)$, so $\Pr(r_j\le\bar r_j)\ge1-\varepsilon$ for each $j$ (and hence marginally, as it holds for every value of $A_j$). A union over the $J$ clients with $\varepsilon=\delta/J$ gives the event $E=\{\forall j:\ r_j\le\bar r_j\}$ with $\Pr(E)\ge1-\delta$. On $E$, for **any** $\lambda$, the convex weights $w_j(\lambda)\ge0$, $\sum_j w_j=1$ give $R_{\mathrm{sel}}(\lambda)=\sum_j w_j r_j\le\sum_j w_j\bar r_j\le\max_j\bar r_j=\bar U_\Delta^{\,r}$. For Theorem 1′, additionally $a_j\in[\underline a_j,\bar a_j]$ on the corresponding event (now $3J$ events at $\delta/3J$), and $\sup$ over the admissible $(\lambda,a)$ dominates the true value. $\square$

**Non-reducibility.** The certificate is not a corollary of existing constructions, for three separate reasons. The first is a formal statement; the failure it describes is the technical crux of the federated setting.

**Proposition 2 (naive pooling is anti-conservative under mixture shift).** *Let $U_{\mathrm{pool}}=U^+(\sum_j K_j,\sum_j A_j;\delta)$ be the single Clopper–Pearson bound applied to the pooled accepted calibration points. There exist client populations $\{(a_j,r_j)\}_{j\le J}$ and deployment mixtures $\lambda\in\Delta^{J-1}$ such that $\Pr\big(R_{\mathrm{sel}}(\lambda)>U_{\mathrm{pool}}\big)\to1$ as the per-client calibration sizes $n_j\to\infty$.*

*Proof sketch (constructive).* Conditioned on the accepted counts, the pooled accepted-error count is a sum of binomials with unequal success probabilities $\{r_j\}$ — a Poisson-binomial — so the single-binomial CP calibration does not apply. Concretely, take $J{=}5$ with four low-risk clients ($a{=}0.7$, $r{=}0.02$) and one high-risk client ($a{=}0.5$, $r{=}0.3$), calibrated under equal weights. By the law of large numbers $U_{\mathrm{pool}}$ concentrates near the calibration-mixture risk $\approx0.07$, while the deployment mixture concentrated on the high-risk client has $R_{\mathrm{sel}}=0.3>0.07$; the violation probability tends to one. (Full proof in Appendix D; Figure 2 reports the finite-sample counterpart, with coverage collapsing to $0$.) $\square$

In plain terms, pooling lets reliable clients average away an unreliable one — the guarantee looks safer than it is exactly when deployment overweights the unsafe client. Two further reductions also fail: **(b)** Lu et al.'s federated-conformal certificate controls a *quantile/coverage* of nonconformity scores under partial exchangeability, whereas Fed-CORE controls a **post-selection conditional error ratio** under client-mixture uncertainty — the calibration statistic is a conditional-binomial proportion $K_j\mid A_j$, not a score quantile. **(c)** The certificate is not a Bonferroni union of unrelated intervals: by Lemma 1 the certified object couples the per-client bounds through acceptance-reweighted convex weights, and the deployment certificate (Thm 1′) is a robust linear-fractional program over $(\lambda,a)$, not a maximum of independent intervals.

*Empirical check ($J=5$, $\delta=0.1$).* The conditional certificate is valid (coverage $0.98$–$1.00\ge0.90$) and **tighter than the mass-ratio baseline** (median simplex $\bar U^{\,r}\approx 0.37$ vs. $0.45$). The bounded-$\Lambda$ box version (Thm 1′) tightens further to $\bar U^{\,r,a}\approx 0.13$ while staying valid for in-box $\lambda^\star$, confirming both validity and the worst-client domination that motivates restricting $\Lambda$.

The full protocol is summarized as Algorithm 1.

**Algorithm 1 (Fed-CORE certification).**

1. **Train.** Obtain the federated open-set model $\hat h$ and any score $s$ (FedAvg or any FedOSR method; Fed-CORE does not modify training).
2. **Propose.** On the proposal fold, select the threshold $t$ of the selector $A$ subject to the risk buffer $\widehat R_{\mathrm{prop}}\le\gamma\alpha$ (Section 3); fix $A$.
3. **Count.** Each client $j$ evaluates $A$ on its certification fold and reports the pair $(A_j,K_j)$ — per client for the stratified certificate, secure-aggregated within groups for the grouped variant.
4. **Certify.** The server computes $\bar r_j=U^+(K_j,A_j;\varepsilon)$ and the certificate: $\bar U_\Delta^{\,r}=\max_j\bar r_j$ (Theorem 1) or the robust linear-fractional bound $\bar U_\Lambda^{\,r,a}$ (Theorem 1′), plus the coverage LCB $\underline C_\Lambda$.
5. **Decide.** Deploy $A$ iff $\bar U\le\alpha$ and $\underline C_\Lambda>0$; otherwise report "cannot certify" (with the Theorem-2 diagnosis of whether the failure is risk- or feasibility-driven).

### 4.3 Theorem 2 — Federated feasibility (per-client accepted-sample threshold)

In the zero-accepted-error regime $K_j=0$ with $\Lambda=\Delta^{J-1}$, the deploy condition $\max_j\bar r_j\le\alpha$ reduces, for each binding client, to $U^+(0,A_j;\delta/J)\le\alpha$, i.e. $1-(\delta/J)^{1/A_j}\le\alpha$.

**Theorem 2.** *Certification at level $\alpha$ over the simplex requires, for every client with non-negligible deployment mass, an **observed accepted count***
$$
A_j\ \ge\ \frac{\ln(J/\delta)}{-\ln(1-\alpha)}\ =\ \Omega\!\Big(\tfrac{\ln(J/\delta)}{\alpha}\Big).
$$
*The expected-count form $n_j a_j\gtrsim \ln(J/\delta)/\alpha$ follows as a corollary.*

We state the bound on the **observed $A_j$** — exactly what the certificate consumes — rather than on $n_j a_j$; the expected-count version is the corollary. This is the federated analog of the centralized $N_{\min}=\lceil\log\delta/\log(1-\alpha)\rceil$: the condition must hold *per client*, with a $\log J$ federation penalty. It predicts **certified-coverage collapse** when a client is simultaneously small and high-risk; restricting $\Lambda$ to a box relaxes the worst-client domination. Plainly, this is the heart of what we later call the *feasibility law*: to prove the error rate is below $\alpha$ you need enough accepted-and-audited examples per client. Theorem 2 is the zero-accepted-error floor; its finite-sample *width* extension — the normal/Bernstein approximation to the Clopper–Pearson interval, for which the required count scales like $(\alpha-\hat r)^{-2}$ — adds that if the model's own error rate $\hat r$ already sits close to the target $\alpha$, the number of audit examples needed explodes. Certification is therefore possible only when the model is comfortably below the target and the audit set is large enough; otherwise the correct output is "cannot certify."

*Empirical check.* The predicted $\ln(J/\delta)/(-\ln(1-\alpha))\approx 78$ accepted/client ($J{=}5,\delta{=}0.1,\alpha{=}0.05$) is consistent with the simulated crossover (median $\bar U$ below $\alpha$ near $\approx 100$ accepted/client). Real CIFAR confirms the mechanism is binding: shrinking the risk buffer $\gamma$ to lower realized risk *also* starves the accepted set below the floor (`cert_n` $500\to151$, $\approx30<37$/client at $d{=}5$), which *raises* the UCB ($0.185\to0.222$) — so feasibility, not operating-point tuning, controls certifiability (Section 5.4).

### 4.4 Privacy and communication

The privacy footprint depends on **which** certificate is deployed: sum-only secure aggregation suffices **only for the pooled certificate**, while the stratified certificate requires per-client count pairs. Table 2 summarizes, for each variant, the certified target, the counts released, and the key assumption — so the privacy/validity/tightness trade-off can be read in one place.

**Table 2. Certificate variants — certified target, privacy, and tightness.** Median upper bound $\bar U$ is from the synthetic study ($r_{\mathrm{bad}}{=}0.3$; lower is tighter).

| variant | certified target | counts released | secure aggregation | key assumption | median $\bar U$ | role |
|---|---|---|---|---|---|---|
| stratified simplex (Thm 1) | any client mixture $\lambda\in\Delta^{J-1}$ | per-client pairs | no | A1–A4 | 0.383 | worst-case guarantee |
| bounded-$\Lambda$ (Thm 1′) | client mixtures $\lambda\in\Lambda$ | per-client pairs | no | A1–A5 (correct $\Lambda$) | 0.158 | recommended deployment |
| grouped-stratified | any mixture **over groups** | per-group pairs | within groups | A1–A4, **A6** (within-group composition) | — | privacy/feasibility compromise |
| pooled (Prop. 3) | matched deployment mixture only | sums only | yes (sum-only) | calibration mixture $=$ deployment | 0.073 | optional tightening |
| mass-ratio (App. B) | any client mixture | per-client pairs | no | A1–A4 | 0.473 | loose baseline |

Because Theorem 1 needs **per-client** counts, it is *not* compatible with sum-only secure aggregation. The recommended compromise is a **grouped-stratified certificate**: partition clients into $G$ public strata of $\ge k$ clients each, secure-aggregate counts *within* each stratum, and run the certificate over the $G$ groups. This keeps a worst-*group* guarantee while releasing only $G$ aggregated pairs — but the certified object weakens accordingly (assumption A6): the guarantee is robust to mixture shift *across* groups, not to arbitrary client-mixture shift *within* a group. Grouping is therefore a declared relaxation, qualitatively distinct from naive pooling (which certifies only the matched mixture and silently fails off it, Proposition 2); the grouped certificate states its coarser target up front. Even per-client counts leak far less than the per-client score *distributions* (T-Digest) of federated conformal prediction or the quantized score *functions* of decentralized conformal novelty detection. A differentially private variant adds calibrated noise to the counts and widens the Clopper–Pearson levels to absorb it.

### 4.5 Proposition 3 — Pooled certificate under matched-mixture calibration

When the calibration data are themselves drawn i.i.d. from the deployment mixture $Q_{\lambda^\star}$ (rather than fixed per-client strata of size $n_j$), pooling avoids the worst-client domination. Let $A=\sum_j A_j$, $K=\sum_j K_j$, $U_{\mathrm{pool}}=U^+(K,A;\delta)$.

**Proposition 3 (matched-mixture).** *Under (a) i.i.d. mixture calibration and (b) Lemma L, $\Pr(R_{\mathrm{sel}}(\lambda^\star)\le U_{\mathrm{pool}})\ge1-\delta$ and $U_{\mathrm{pool}}\le\bar U_\Delta^{\,r}$.*

Proposition 3 is an **optional matched-mixture tightening**; it is not used for the main guarantee or for any headline claim, and Theorems 1/1′ stand without it. It rests on two ingredients. **Lemma L** states that the binomial Clopper–Pearson upper limit remains conservative for the mean of a Poisson-binomial accepted-error count at the operative CP threshold (cf. Hoeffding's comparison of Poisson-binomial and binomial tails [35]); a self-contained transfer argument and an adversarial search over 990 configurations (minimum coverage $0.902\ge1-\delta$), together with an assumption-free Bernstein fallback, are given in the supplementary material. The remaining **roster-composition gap** — under fixed-stratum sampling the pooled accepted-error mean need not equal $R_{\mathrm{sel}}(\lambda^\star)$ — is why Proposition 3 requires i.i.d. mixture calibration (its assumption (a)) and why it is stated as a proposition rather than a theorem.

### 4.6 Score-agnostic guarantee and the deformation phenomenon

Because the guarantee in Theorems 1, 1′ (and Proposition 3) is produced entirely by the certification split — the conditional-binomial structure of $K_j\mid A_j$ under a *fixed* selector — it holds for **any** score $s(\cdot)$ used to define $A$. Score quality affects *how much coverage* is certified (a better score accepts more at the same risk), never *whether the risk is controlled*. This matters because federated heterogeneity, like label corruption, **deforms the confidence–correctness ranking**: at clients holding only minority known classes, those classes are easily confused with genuine unknowns, so any single global score is miscalibrated somewhere. Fed-CORE does not attempt to repair this deformation with the small trusted set; it *certifies around it*. Section 5.6 makes this concrete on real detectors: validity holds for every score while certified coverage tracks score quality.

### 4.7 What accepted predictions are *for*: two certified uses

The certificate is not an end in itself — it licenses two downstream uses of the accepted set, which are the paper's "so what."

**Use case A — safe automation / triage (no retraining).** Accept $=$ act automatically; reject $=$ defer to a human. Then CertifiedCoverage@$\alpha$ is exactly the **fraction of the workload safely automated at guaranteed error $\le\alpha$**, and $1-\text{coverage}$ is the human-review load. This reads directly off Theorems 1/1′ and is the primary deployment value: an uncertified operating point (a fixed confidence threshold, or a FedOSR model's default) either *breaches* $\alpha$ or automates *less* at the same guaranteed risk.

**Use case B — certified federated self-training (consumes the accepted set).** Accepted predictions on unlabeled client data become pseudo-labels folded back into FedAvg; the certificate bounds their **contamination** (pseudo-label error rate $\le\alpha$ w.r.t. the calibration distribution), so the model is expanded with *provably-bounded* noise rather than the unbounded contamination of naive self-training.

*Preserving validity across the self-training loop (the subtle part).* Self-training makes the round-$t$ model depend on what was accepted at round $t-1$, so **reusing one certification fold across rounds would break the independence** the certificate needs. We sidestep the hard closed-loop concentration problem by **data-splitting in time**: partition the trusted set into $T$ disjoint audit folds $\mathcal C^{(1)},\dots,\mathcal C^{(T)}$ and certify the round-$t$ selector on the fresh fold $\mathcal C^{(t)}$ at level $\delta/T$.

**Proposition 4 (round-wise self-training validity).** *If $\mathcal C^{(t)}$ is independent of $(f_t,A_t)$ — guaranteed by forming $f_t,A_t$ only from folds indexed $<t$ and from unlabeled data — and each round is certified at level $\delta/T$, then*
$$
\Pr\big(\forall t\le T:\ R_{\mathrm{sel}}(A_t)\le \bar U^{(t)}\big)\ \ge\ 1-\delta .
$$
*Every injected pseudo-label batch therefore has certified contamination $\le\alpha$ simultaneously across all $T$ rounds.* Unlike a reused-fold scheme (which would require a closed-loop adaptivity argument), data-splitting makes each round a clean, independent application of Theorem 1/1′. The price is feasibility: each round's fold must clear the Theorem 2 threshold, so $T$ is bounded by the trusted-set size — an explicit budget/utility trade-off.

**Algorithm 2 (certified federated self-training).**

1. Partition the trusted set into $T$ disjoint audit folds $\mathcal C^{(1)},\dots,\mathcal C^{(T)}$.
2. For round $t=1,\dots,T$: form $(f_t,A_t)$ from folds $<t$ and unlabeled data only; run Algorithm 1 on the fresh fold $\mathcal C^{(t)}$ at level $\delta/T$.
3. If certified ($\bar U^{(t)}\le\alpha$), admit the accepted pseudo-label batch into FedAvg; otherwise **halt admission** for this round (admit nothing).

*Empirical verification (synthetic smoke test and real CIFAR).* The contract is checked directly: with the $\delta/T$ split the **simultaneous** unsafe rate is $0.086\le\delta$, whereas using $\delta$ per round (no split) inflates it to $0.386>\delta$ — confirming the split is necessary, not cosmetic. On real CIFAR self-training, **naive** (uncertified) self-training injects pseudo-label contamination of $0.59$–$0.98\gg\alpha$ (the certificate correctly recommends rejection with $U=1.0$, which the naive rule ignores), while **certified** self-training keeps contamination $\approx 0$ (rejecting / halting on Theorem-2-infeasible rounds); clean-data accuracy is comparable across certified, naive, and none. The safety contract thus holds on real data. We stress that the claimed benefit of this use case is **contamination control, not an accuracy gain**: with the present small backbone there is little safe-acceptance headroom, so accuracy is not the point (see Section 5).

---

## 5. Experiments

The experiments are not designed to show that Fed-CORE improves open-set recognition accuracy; they evaluate the *operating characteristics of a statistical procedure*. Four certification claims are tested, in order: **(i) validity** — certified deployments do not violate the target risk on held-out data, and the unsafe-deployment rate of the decision rule stays below $\delta$ (Sections 5.2–5.3); **(ii) non-reducibility** — pooled Clopper–Pearson fails under client-mixture shift exactly as Proposition 2 predicts (Section 5.2); **(iii) feasibility** — certified coverage appears only when the per-group accepted audit count clears the Theorem-2 floor, and collapses along the risk-target, heterogeneity, and corruption axes as the feasibility law predicts (Sections 5.4–5.5); **(iv) score- and base-model-agnosticism** — validity holds for every score and detector while certified coverage tracks detector quality (Section 5.6). Section 5.7 evaluates the downstream self-training gate, and Section 5.8 collects ablations.

### 5.1 Experimental setup

We train federated models with FedAvg [36] under non-IID Dirichlet partitions ($d\in\{0.1,0.5,5\}$; smaller $d$ is more heterogeneous), hold out classes as test-time unknowns (the standard FedOSR open-set split), and optionally corrupt client labels (symmetric/asymmetric) to connect to the corrupted-training setting. Data sets are CIFAR-10/100 (primary), TinyImageNet, and a tabular FL benchmark (covtype). Four post-hoc scores — maximum softmax probability (MSP) [31], entropy, margin, and energy [32] — test the score-agnostic claim. The headline metric is CertifiedCoverage@$\alpha$: the accepted coverage among runs that certify $\bar U_\Lambda\le\alpha$. FedOSR detectors (FedPD/FedPD++ [1,2], FedOSS [30], FOOGD [3]) are treated as **base models**, not competitors — they carry no certificate, and Fed-CORE certifies their scores post-hoc. As the first method to certify this object, we compare against re-cast nearest methods (federated CP [6], decentralized novelty FDR [29]) that are invalid or control a different functional, against our own variants as ablations, and against centralized oracles [25,26,37] as upper bounds. Table 3 fixes the configuration; these values are the parameters of the guarantee itself, not tuning details.

**Table 3. Experimental configuration.** Fractions are of each client's trusted clean pool; all folds are disjoint.

| component | value |
|---|---|
| clients $J$ / known classes | 5 / 6 (CIFAR-10; rest unknown) |
| heterogeneity | Dirichlet $d\in\{0.1,0.5,5\}$ |
| backbones | CIFAR-stem ResNet-18 (GroupNorm / BatchNorm), SimpleCNN; WideResNet for FedPD/FOOGD reproductions |
| FedAvg | 50 rounds, 2 local epochs, lr 0.01, batch 64 |
| trusted-pool folds | proposal 0.34 / certification 0.33 / test 0.33 |
| audit unknown fraction | 0.30 (A4 stress test in Section 5.3) |
| certificate | $\alpha\in\{0.05,0.10,0.20,0.25\}$, $\delta=0.10$, $\gamma\in\{0.5,0.7,1.0\}$, $\Lambda\in\{\text{simplex},\text{box}(\rho{=}0.15)\}$ |
| corruption | train labels only (symmetric 0.35 / asymmetric 0.20); calibration folds stay clean |
| reported metrics | `certified`, `cert_risk_ucb`, `cert_coverage_lcb`, `cert_n`, `cert_k`, `test_risk`, `test_coverage` |

### 5.2 Validity and non-reducibility

We first establish that the certificate is valid and that the obvious alternative — pooling — is not.

*Controlled synthetic study.* On synthetic clients whose ground-truth risks and deployment mixtures are varied independently, every property holds: empirical coverage $\ge0.98$ across heterogeneity; the tightness order box $<$ simplex $<$ mass-ratio (all valid; the pooled certificate is tightest but invalid off the matched mixture); a monotone CertifiedCoverage@$\alpha$ frontier; the heterogeneity-collapse curve crossing the Theorem-2 floor ($\approx37$ accepted/client); and all four scores valid (test_risk $\approx0.044\le\alpha$).

*Pooling is anti-conservative (non-reducibility).* The natural shortcut — pool every client's accepted audit points and apply one Clopper–Pearson bound — fails under heterogeneity.

![Figure 2](experiments/fedcore/figs/fig1_pooling_collapse.png)

**Figure 2. Why federated calibration cannot be pooled.** Empirical coverage of the certificate as the deployment mixture shifts from the calibration-matched $\lambda$ toward one high-risk client (four low-risk clients $a{=}0.7,r{=}0.02$ plus one high-risk $a{=}0.5,r{=}0.3$; $\delta{=}0.1$). Naive pooled CP is valid only at the matched mixture and collapses to $0$ under shift (it certifies $\approx0.072$ while the true risk reaches $0.165$–$0.30$); the stratified conditional certificate (Theorem 1) stays $\ge1-\delta$ for every mixture, and the box-$\Lambda$ certificate (Theorem 1′) is by design valid only for mixtures inside its assumed box — its drop outside the box is expected and does not contradict its guarantee. The pooled accepted-error count is Poisson-binomial, not binomial, so one Clopper–Pearson bound is anti-conservative.

*Uncertified rules are unsafe (necessity).* A practitioner without the certificate has two options, both unsafe. Table 4 reports the unsafe-deployment rate $\Pr(\text{deploy}\mid R_{\mathrm{sel}}>\alpha)$, which any valid method must keep $\le\delta=0.1$, estimated over $2{,}000$ Monte Carlo calibration draws per configuration ($J{=}5$, $n_j{=}300$, $\alpha{=}0.05$; the boundary regime places the true $R_{\mathrm{sel}}$ just above $\alpha$ by sweeping the high-risk client's $r_j$).

**Table 4. Necessity — unsafe-deployment rate (must be $\le\delta=0.1$; $2{,}000$ trials per cell).** (Naive pooling is omitted here because it controls coverage, not the accepted-risk metric; its collapse is shown in Figure 2.)

| rule | uses the split correctly? | stratified? | unsafe-deploy rate | valid? |
|---|---|---|---|---|
| naive empirical threshold (deploy iff $\hat r\le\alpha$) | yes | no | 0.49 | no |
| leaked split (threshold chosen on the certification fold) | no | yes | 0.18 | no |
| **Fed-CORE (proper split)** | yes | yes | **0.00–0.03** | yes |

The naive threshold ignores finite-sample noise and deploys unsafely about half the time at the boundary. The proposal/certification split is load-bearing: re-searching the threshold on the certification fold inflates the deploy rate to $99.8\%$ but the unsafe rate to $18.2\%\gg\delta$, because the certificate cannot correct the multiple-testing of thresholds chosen on the same fold. Only the proper split with the conditional certificate keeps the unsafe rate $\le\delta$. (Federated-CP rules control a quantile, not the accepted risk, so they do not appear here; naive pooled CP is anti-conservative under shift, Figure 2.)

### 5.3 Audit representativeness is a condition of validity (A4 stress test)

Assumption A4 requires the audit fold to carry unknown-class points at no less than their deployment rate. This is a *condition of the guarantee*, so we stress-test it before reporting any certified coverage.

![Figure 3](experiments/fedcore/figs/FA5_unknown_proportion.png)

**Figure 3. Audit representativeness (A4) stress test.** Holding the deployment unknown-among-accepted fraction at $0.06$, we vary the *calibration* unknown fraction. Coverage of the true deployment risk is $\ge1-\delta$ only when the calibration fraction matches or exceeds the deployment one ($0.913$ at $0.06$, $0.992$ at $0.08$); under-representation is anti-conservative ($0.057$ at $0.02$, $0.522$ at $0.04$).

The real-data counterpart on CIFAR-10 logits (Section 5.8) reproduces this collapse (coverage $1.00$ at $\rho{=}1$ down to $0.005$ as unknowns are under-represented). It is therefore not enough that the audit fold *contains* labeled unknowns — it must carry them at no less than the deployment rate. In practice the monitoring set should track, not under-sample, the unknown-class incidence of the live stream; every validity statement below is read under A4.

### 5.4 The finite-sample feasibility law

*The apparent null, and its cause.* On a real CIFAR-10 ladder (12 runs, $\alpha=\delta=0.1$, 5 clients, small CNN), CertifiedCoverage@$0.1$ is $0$ in every run. Two distinct modes explain this. **Mode 1** (extreme non-IID, $d{=}0.1$): the empirical accepted risk already exceeds $\alpha$, so no method can deploy safely and the certificate correctly declines. **Mode 2** (near-IID, $d{=}5$, test_risk $\approx0.08<\alpha$): the model is safe, but the thin per-client accepted counts cannot drive the upper bound below $\alpha$ — the Theorem-2 feasibility collapse, not certificate looseness. Tellingly, shrinking the risk buffer $\gamma$ to lower realized risk *also* starves the accepted set (cert_n $500\to151$, $\approx30<37$/client), which *raises* the bound ($0.185\to0.222$): the binding lever is calibration budget, not the operating point.

*The staircase.* Re-aggregating the $J{=}5$ clients into $G$ public groups (the grouped-stratified certificate, Section 4.4) raises the per-group accepted count and drives the bound monotonically through $\alpha$.

![Figure 4](experiments/fedcore/figs/F6_feasibility_law.png)

**Figure 4. The feasibility law (Theorem 2).** (a) Worst-group certified-risk upper bound versus per-group accepted count (log axis), as clients merge into $G\in\{5,3,2,1\}$ groups (ResNet, 5-seed $\pm1$ std band); the vertical marker is the Theorem-2 floor ($\approx37$/client). (b) The corresponding CertifiedCoverage@$0.10$. (c) Calibration-budget sweep: mean certified-risk bound (left axis) and probability of certifying (right axis) versus per-client audit count.

Panels (a) and (c) make one point from two directions: *merging clients into groups* and *enlarging the audit budget* are the same Theorem-2 sample-size lever, governed by the $(\alpha-\hat r)^{-2}$ requirement rather than by the operating point. The bound crosses $\alpha{=}0.10$ around several hundred accepted points per group, and CertifiedCoverage@$0.10$ rises from $0$ to $\approx0.21$; at the largest budget the seed-mean bound sits just above $\alpha$ while $2/5$ seeds individually certify — the usual gap between a mean and a pass rate, and the seed-variability that keeps $\alpha{=}0.10$ at the feasibility edge for this baseline detector. Certification fails here not because the certificate is loose, but because the accepted audit count is below the finite-sample floor.

![Figure 5](experiments/fedcore/figs/F5_alpha_frontier.png)

**Figure 5. Risk–coverage frontier on real CIFAR-10 (per-client $G{=}5$, box-$\Lambda$; $d{=}5$).** Certified coverage versus risk target $\alpha$ under the most conservative per-client grouping (SimpleCNN $0.063$ at $\alpha{=}0.20$, $0.193$ at $\alpha{=}0.25$; ResNet-18 $0.316$ at $\alpha{=}0.25$).

The frontier is monotone because the proposal-side buffer enforces a small safety margin; an aggressive operating point is corrected rather than allowed to pass by luck (validity is preserved by the independent certification fold). Merging into worst-group $G{=}2$ raises the achievable coverage at the same $\alpha$ (Section 5.5). The marker at $\alpha{=}0.20$ is the finite-sample feasibility demonstration point — chosen to exhibit the vacuous-to-non-vacuous transition under realistic audit budgets, not as a recommended safety threshold — while $\alpha{=}0.10$ sits at the feasibility edge for this baseline detector (the stronger FedPD-PROSER detector certifies there; Section 5.6).

### 5.5 Real-data certified coverage

*Backbone.* We adopt a CIFAR-stem ResNet-18 with **GroupNorm** as the principled FL normalization (BatchNorm's running statistics diverge under non-IID FedAvg). GroupNorm lowers $\hat r$ but the accepted set shrinks in step, so it does not strengthen the seed-variable $\alpha{=}0.10$ certificate; what matters is robust — no held-out risk violation under either normalization.

*Headline.* At $\alpha{=}0.20$ the GroupNorm certificate is non-vacuous on **all five seeds**: CertifiedCoverage $0.392\pm0.097$ ($d{=}5$) and $0.353\pm0.130$ ($d{=}0.5$), with $0/10$ held-out violations among certified runs (Table 5). This is a finite-sample feasibility demonstration under the current audit budget, not a safety target. At $\alpha{=}0.10$ the worst-group result is seed-variable ($2$–$3/5$), reported as a secondary result; Table 5 also reports the certificate diagnostics for the $\alpha{=}0.10$ runs — the median worst-group `cert_risk_ucb` sits just below $\alpha$ where seeds certify, and the mean `test_risk` is an order of magnitude below the bound, quantifying the finite-sample conservatism. The edge and negative cells (Table 6) are exactly where the feasibility law predicts collapse — extreme non-IID, corruption, or a backbone whose $\hat r$ is near $\alpha$.

**Table 5. Real-data certified coverage — clean-data main results** (CIFAR-10, worst-group $G{=}2$ CertifiedCoverage@$\alpha$, fixed score MSP, mean$\pm$std over seeds; certified-seed counts in parentheses; ResNet-GroupNorm, cert_frac$=0.5$). Diagnostics (median worst-group $\bar U$ and mean test risk) are from the $\alpha{=}0.10$ runs; per-run values for every cell follow the canonical metric schema and accompany the code release.

| backbone / cell | $\alpha{=}0.10$ (5-seed) | $\alpha{=}0.20$ (5-seed) | median $\bar U_{G2}$ | mean `test_risk` | held-out violations |
|---|---|---|---|---|---|
| ResNet-GN $d{=}5$ clean | $0.077\pm0.097$ (2/5) | $\mathbf{0.392\pm0.097}$ (5/5) | 0.088 | 0.007 | 0 |
| ResNet-GN $d{=}0.5$ clean | $0.091\pm0.103$ (3/5) | $\mathbf{0.353\pm0.130}$ (5/5) | 0.080 | 0.007 | 0 |
| ResNet-BN $d{=}5$ clean | $0.106\pm0.098$ (3/5) | $\approx0.29$–$0.31$ (single-config) | 0.075 | 0.041 | 0 |

**Table 6. Edge and negative settings** (where certification is correctly vacuous, with the feasibility-law reason).

| setting | CertCov@$\alpha$ | reason it does not certify |
|---|---|---|
| ResNet $d{=}0.1$ clean | $0$ | extreme non-IID; accepted set too small (Theorem-2 infeasible) |
| ResNet sym-0.35 | $0$ | corruption raises $\hat r>\alpha$ (e.g. `test_risk` $0.167$ at $d{=}0.5$) |
| SimpleCNN | $0.063$ at $d{=}5$ ($\alpha{=}0.20$) | looser backbone; higher $\hat r$ |
| covtype (tabular FL) | $0$ at fixed MSP | linear backbone $\hat r\approx0.14$–$0.24$ near $\alpha$ |

covtype is reported as a second domain that exhibits the same feasibility law at its edge, **not** as a positive: with the procedurally valid fixed-MSP protocol it certifies $0/5$. (A best-of-four-scores rule reaches $0.10\pm0.17$ on $2/5$ seeds, but selecting the score after seeing certification results breaks assumption A2, so this number carries no guarantee; we use it only to locate the feasibility edge and exclude it from all validity claims. A valid multi-score certificate would Bonferroni-correct to $\delta/4$ per score or select the score on the proposal fold.)

*Superiority.* Table 7 compares Fed-CORE against the two uncertified alternatives at the feasibility demonstration point.

**Table 7. Superiority — matched-risk comparison at $d{=}5$, $\alpha{=}0.20$** (post-hoc on real logits). Fed-CORE is the only method that is simultaneously safe and finite-sample-guaranteed.

| method | uses test labels? | accepted coverage | realized risk $\le\alpha$? | finite-sample guarantee? |
|---|---|---|---|---|
| test-peeking oracle (MSP / energy) | yes | $0.444$ / $0.431$ | yes (by construction) | ✗ |
| no-peek naive threshold | no | high near-IID | sometimes no (breaches off-IID) | ✗ |
| **Fed-CORE** (worst-group $G{=}2$) | no | $0.392\pm0.097$ (5-seed) | yes | ✓ |

The test-peeking oracle reaches higher coverage but uses test labels and carries no guarantee; the no-peek naive threshold breaches $\alpha$ wherever proposal and test diverge. Fed-CORE retains a large fraction of the oracle's coverage while being the only safe, guaranteed option.

*Stress axes.* Beyond the risk target $\alpha$ (Figure 5), the feasibility law has two further axes — heterogeneity and corruption — shown together in Figure 6; each pushes $\hat r$ past $\alpha$ or starves the per-group count below the Theorem-2 floor.

![Figure 6](experiments/fedcore/figs/F7_hetero_collapse.png)

**Figure 6. Stress axes of the feasibility law.** (a) Heterogeneity: worst-group certificate versus Dirichlet concentration $d$ (SimpleCNN stress configuration). (b) Corruption: worst-group CertifiedCoverage@$0.20$ versus client-side training-label noise rate, at $d\in\{0.5,5\}$; the trusted calibration fold stays clean throughout. Panel (b) uses single fixed configurations and is not directly comparable to the five-seed headline of Table 5.

As $d$ falls (more non-IID), per-group accepted counts thin and the worst-group certificate degrades, collapsing at the extreme ($d{=}0.1$); the looser SimpleCNN backbone is used in panel (a) only to isolate the mechanism, with the ResNet-GroupNorm headline in Tables 5–6. On the corruption axis, certification is non-vacuous on clean data ($0.31$ at $d{=}5$, $0.13$ at $d{=}0.5$) but collapses to $0$ once the noise rate exceeds $\approx0.1$ *even though the calibration fold stays clean*: corruption raises the model's $\hat r$ above $\alpha$, so a clean audit set has nothing safe left to certify.

### 5.6 Score and base-model dependence

*Score-agnosticism.* The tightness ordering of the certificate variants (conditional simplex tighter than the mass-ratio baseline, bounded-$\Lambda$ tighter still) is reported together with their privacy footprint in Table 2. The guarantee holds for **every** score: across MSP, entropy, margin, and energy the realized test_risk stays $\le\alpha$ (0.042–0.046) while certified coverage varies (0.64–0.66; Appendix C, Table C1). The score changes only *how much* coverage is certified, confirming that validity comes from the certification split, not from score quality (Section 4.6) — a point made concrete on real detectors next.

*Two real FedOSR detectors.* To answer the concern that the above uses MSP on a FedAvg backbone rather than a genuine FedOSR detector, we certify the native open-set scores of two real FedOSR methods: FedPD's PROSER dummy-vs-known score (a full reproduction) and FOOGD's SM3D score (a representative head on the shared backbone). Table 8 reports both, alongside the FedAvg+MSP control and a faithfully reproduced full FOOGD-SAG.

**Table 8. Certification on real FedOSR base models** (native open-set scores; worst-group $G{=}2$, three seeds; no held-out violation in any certified cell). $\ddagger$ single-seed negative reproduction (Appendix C discusses the configuration sensitivity).

| base model (kind) | $d$ | AUROC | CertCov@$0.20$ (3 seeds) | CertCov@$0.10$ (3 seeds) | mean `test_risk` ($\alpha{=}0.20/0.10$) |
|---|---|---|---|---|---|
| FedPD–PROSER (full) | $5$ | $0.80$ | $\mathbf{0.483\pm0.100}$ (3/3) | $0.174\pm0.125$ (2/3) | $0.112$ / $0.029$ |
| FedPD–PROSER (full) | $0.5$ | $0.79$ | $0.455\pm0.090$ (3/3) | $\mathbf{0.210\pm0.089}$ (3/3) | $0.105$ / $0.036$ |
| FedAvg+MSP (full, same backbone) | $5$ | $0.73$ | $0.350\pm0.077$ (3/3) | — | — |
| FOOGD–SM3D (representative) | $5$ | $0.69$ | $0.071\pm0.053$ (3/3) | — | — |
| FOOGD–SM3D–SAG (full) | $5$ | $0.47$ | $0$ (0/1)$^{\ddagger}$ | — | — |

*FedPD-PROSER — the strongest base model, and a full method.* Trained with the standard recipe — a closed-set CE pretraining stage (known-class accuracy $0.39\to0.65$ over eight rounds, versus $0.26$ stalled from scratch) followed by PROSER fine-tuning on the exact WideResNet-28-10 architecture — its native PROSER dummy-vs-known score reaches AUROC $0.80$. It is the only detector that certifies at the hard target $\alpha{=}0.10$, where MSP and FOOGD cannot, and the result *strengthens* under heterogeneity: at the more non-IID $d{=}0.5$ it certifies $3/3$ seeds at both targets (Table 8). This is a genuine **full** FedOSR method, not a representative head, and it shows that Fed-CORE is a certification layer for real FedOSR detectors — a strong detector plus the certificate reaches risk targets the baseline cannot.

*The thesis across base models.* Certified coverage tracks the native-score AUROC — FedPD $0.80$ > MSP $0.73$ > FOOGD-representative $0.69$ > FOOGD-SAG $0.47$ — while **validity holds in every cell** ($0/18$ held-out violations across all seeds and base models): exactly the score-agnostic guarantee, with coverage following detector quality and validity independent of it. The FedAvg+MSP control reproduces the Section 5.5 headline ($\approx0.35$ at $\alpha{=}0.20$), validating the harness.

*A reproducibility note.* The full pipelines are configuration-sensitive: FedPD-PROSER requires the closed-set-pretrain-then-fine-tune recipe (from scratch it does not converge, as PROSER is designed to fine-tune a pretrained model), while full FOOGD-SAG reaches only chance-level AUROC $0.467$ on our six-known semantic-shift split at a single-GPU budget; we report it as a single-seed negative, with the configuration analysis in Appendix C. FedOSS (a medical-imaging codebase without a CIFAR loader) is deferred as the highest-cost option. Fed-CORE's validity is unaffected throughout — certified coverage simply follows the base model's score quality.

### 5.7 Downstream use: certified pseudo-label admission

The certificate's first downstream use, safe automation, is CertifiedCoverage@$\alpha$ itself (Section 4.7). The second use is **first a certified admission gate, not primarily an accuracy booster**: accepted predictions are folded back into FedAvg as pseudo-labels only when their contamination can be certified below the target — though, as a supporting result below shows, a sufficiently strong base detector with an adequate audit budget can additionally turn safe admission into a safe accuracy gain.

![Figure 7](experiments/fedcore/figs/F8_selftraining.png)

**Figure 7. Certified pseudo-label admission prevents unsafe self-training (Proposition 4, ResNet-GN $d{=}5$).** (a) Realized pseudo-label contamination per round (naive self-training; $\alpha{=}0.1$ dashed). (b) Admission/halt behavior (points annotated with batch size $n$). (c) Round-wise validity: simultaneous unsafe rate with and without the $\delta/T$ split.

In Figure 7, naive self-training keeps injecting pseudo-labels whose realized error rate is far above the target ($0.19$–$0.67$ and growing), whereas the certified procedure finds the very first round Theorem-2-infeasible — the accepted set is too thin to certify below $\alpha$ — and **halts, admitting nothing** rather than an uncertified batch. This is the safe outcome the feasibility law predicts: Fed-CORE never injects a contaminated batch, even when that means admitting none. Over the rounds the Proposition-4 contract ($\delta/T$) is verified (Table 9): round-wise certification is necessary — reusing one audit fold across adaptive rounds breaks the simultaneous guarantee.

**Table 9. Round-wise certification is necessary for valid self-training.**

| scheme | fresh audit fold per round? | per-round level | simultaneous unsafe rate | valid at $\delta{=}0.10$? | interpretation |
|---|---|---|---|---|---|
| Fed-CORE (round-wise split) | yes | $\delta/T$ | 0.086 | ✓ | every admitted batch satisfies the simultaneous contamination certificate |
| reused / no split | no | $\delta$ per round | 0.386 | ✗ | reusing the same audit evidence across adaptive rounds breaks the finite-sample guarantee |

*A supporting result: safe admission can also yield an accuracy gain on a strong detector.* The gate's purpose is safety, but when the base detector is strong and the audit budget large, admitted pseudo-labels can additionally raise accuracy without breaching the contamination target. Replacing the FedAvg+MSP base with FedPD-PROSER (Section 5.6) and using one-shot certification at level $\delta$, the certified procedure yields a known-accuracy gain of $+0.030$ (sample SD $0.027$; $2/3$ seeds positive, the third feasibility-limited to zero admissions; with $n{=}3$ seeds we report descriptive statistics only, not a confidence interval) at $\alpha{=}0.20$ and audit budget $4\times$, against a clean-pseudo-label oracle upper bound of $+0.045$ (Appendix C, Figure 9); every admitted batch stays certified ($0/3$ contamination violations, maximum realized contamination $0.137\le\alpha$). The gain is governed by the certificate's own two levers. *Detector strength:* at identical budget the weaker FedAvg+MSP base admits nothing (certified $\Delta{=}0$, oracle headroom $+0.023$). *Audit budget:* at $1\times$ and $2\times$ the procedure admits nothing and halts; only at $4\times$ does it admit — the Theorem-2 feasibility law applied to self-training. A necessary validity condition is that the unlabeled pool's unknown rate be matched to the certification rate (assumption A4, Section 3); without it the procedure becomes anti-conservative and admits batches with contamination $0.285>\alpha$. A deeper label-scarce regime ($10\%$ labels) confirms that feasibility, not headroom, bounds the gain: the oracle headroom grows to $+0.16\pm0.05$ while certified admission occurs on only $1/3$ seeds ($0$ contamination violations throughout); details in Appendix C. We therefore report the admission gate as the primary result and the accuracy gain as a conditional, supporting one.

### 5.8 Ablations

Three controlled ablations probe the certificate's knobs — calibration budget, unknown-class proportion, and client count. The calibration-budget lever is panel (c) of Figure 4; the unknown-proportion condition is the A4 stress test of Figure 3 (Section 5.3); the client-count ablation is in Appendix C (Figure 8). On real CIFAR-10 logits, the audit-budget sweep drives the worst-group bound from $0.58$ to $0.18$ as the calibration fold grows ($\alpha{=}0.10$ becoming non-vacuous, $2/5$ seeds, only at the largest budget), and the unknown-proportion sweep reproduces the audit-representativeness collapse (coverage $1.00$ at $\rho{=}1$ down to $0.005$ as unknowns are under-represented) — confirming the feasibility law and the A4 requirement on real federated logits.

## 6. Limitations

**Conservatism.** The stratified certificate is conservative (Clopper–Pearson exactness + union bound + worst-case mixture); box-$\Lambda$ (Thm 1′) and pooled (Prop. 3) recover tightness but require, respectively, knowledge of $\Lambda$ and the assumptions of Section 4.5. The guarantee is marginal over $Q_\lambda$, not conditional per client.

**Trusted calibration (A3–A4).** Certifying unknown rejection requires labeled unknown-class points in the audit fold (necessary by Proposition 1), and the audit fold must carry unknowns at no less than the deployment rate (A4) — under-representation is anti-conservative, as the stress test of Section 5.3 shows. "Distribution-free" therefore means with respect to the calibration distribution $Q_\lambda$, not the entire unknown universe; in deployment the audit fold corresponds to a small audited monitoring set that must track the live unknown incidence, and Theorem 2 quantifies how scarce it may be before certification becomes infeasible.

**Privacy.** As characterized in Section 4.4, only the pooled certificate is sum-only; the stratified certificate needs per-client (or per-group) counts, and the grouped variant weakens the certified target to group mixtures (A6). The privacy claim is an information-flow statement, not a differential-privacy guarantee unless the noised variant is used.

**Self-training use case (B).** What the certificate guarantees is the *contamination* of each injected pseudo-label batch ($\le\alpha$ per round, simultaneously over $T$ rounds by Prop. 4) — **not** that self-training necessarily improves accuracy; the accuracy gain is an empirical claim (bounded-noise training is well-behaved but not monotone-guaranteed). Empirically the gain is conditional and seed-variable (Section 5.7: $+0.030$ over three seeds, $2/3$ positive, on a strong detector at $4\times$ audit budget, with one seed feasibility-limited to zero admissions); we therefore frame the contamination gate, not the accuracy gain, as the guarantee. Round-wise audit-fold splitting trades feasibility for adaptivity: $T$ is capped by the trusted-set size via the Theorem 2 per-fold threshold. A reused-fold scheme with formal closed-loop validity (avoiding the $\delta/T$ split) is left as future work.

**Statistical resolution of the validity evidence.** The real-data evidence for validity is necessarily coarse: with five training seeds per cell, "0 held-out violations" bounds the per-cell violation rate only loosely. The aggregate count across all certified cells in this paper ($0$ violations over all certified runs, seeds, scores, and base models) is more informative, and the synthetic studies — where thousands of Monte Carlo calibration draws are feasible — estimate the coverage of the certificate directly ($\ge0.98$ observed against the $0.90$ target, Section 5.2). Resampling-based validity quantification on the real logits (re-drawing audit folds from the held-out pool at fixed model) is a natural strengthening left for the final version.

---

## 7. Conclusion

Fed-CORE certifies the accepted selective risk of a federated open-set classifier with a finite-sample, distribution-free guarantee that holds under heterogeneity and unknown deployment mixtures. Its core is a **stratified conditional selective-risk certificate** ($\max_j$ of per-client conditional-binomial CP limits, with a robust bounded-$\Lambda$ form) that is valid where naive pooling fails, accompanied by a per-client feasibility threshold and an optional matched-mixture pooled tightening. The framework recasts federated open-set recognition from a ranking problem into a *certification* problem: a small trusted audit set is used not to repair the model but to certify which of its predictions are safe to accept — and those certified-safe predictions are then *used*, for guaranteed-risk automation and for certified federated self-training that expands training with provably-bounded contamination.

Empirically, across synthetic clients and real federated benchmarks (CIFAR-10/100 and a tabular FL task), the theorems' guarantee holds under the stated assumptions (A1–A6), and no certified deployment exhibits a held-out risk violation in any seed, data set, or normalization; naive pooling collapses under mixture shift exactly as Proposition 2 predicts. Certified coverage is governed by a **feasibility law**: a per-group sample threshold scaling as $(\alpha-\hat r)^{-2}$, traced out by a monotone grouped-stratified staircase and by the heterogeneity and corruption axes. Two positives result. Certifying the native score of a full FedOSR method (FedPD-PROSER) reaches worst-group certified coverage at both risk targets — including $3/3$ seeds at the hard target $\alpha{=}0.10$ under strong heterogeneity — showing that detector quality converts directly into certified coverage at fixed validity. The five-seed FedAvg+MSP baseline certifies at $\alpha{=}0.20$ ($0.392\pm0.097$ at $d{=}5$, $0.353\pm0.130$ at $d{=}0.5$, both $5/5$), with $\alpha{=}0.10$ seed-variable ($2$–$3/5$) for this weaker detector; a second federated domain (tabular FL) sits at the feasibility edge ($0/5$ at fixed MSP) and corroborates the law. The contribution is therefore the *object and its finite-sample certificate*, the exposure of pooled invalidity under heterogeneity, and the characterization of *when* certified open-set deployment is feasible — not a new FedOSR algorithm or a raw-accuracy gain.

Three directions follow from the limitations of this study. First, the matched-mixture pooled result (Proposition 3) remains subordinate because of the roster-composition coupling between the pooled accepted-error mean and the deployment risk ratio; closing this coupling with a stratified finite-population correction would promote the pooled tightening to a theorem and reduce the per-group audit requirement. Second, the per-group sample threshold of Theorem 2 grows as $(\alpha-\hat r)^{-2}$ near the boundary, which keeps the $\alpha=0.10$ regime at the feasibility edge; variance-adaptive bounds of the Bernstein or betting type, which replace the worst-case Clopper–Pearson width by an empirical-variance width, are expected to bring small risk targets into the feasible regime at realistic audit budgets. Third, the grouped-stratified certificate releases only per-group counts, which makes a differentially private count-release certificate a concrete next step toward a formal privacy guarantee rather than the information-flow statement given here.

---

## CRediT authorship contribution statement

Sanghoon Kim: Conceptualization, Methodology, Formal analysis, Software, Investigation, Writing – original draft, Writing – review and editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

This work was supported by [funding source to be completed].

## Data availability

The CIFAR-10 and CIFAR-100 data sets are publicly available. The code and the derived calibration counts used to reproduce the certificates will be made available on publication.

---

## Appendix A. Notation

| Symbol | Meaning |
|---|---|
| $J$, $j$ | number of clients; client index |
| $P_j$, $\lambda$, $\Lambda$ | client distribution; deployment mixture weights; admissible weight set |
| $\hat h,\ \hat y(x)$ | federated classifier; its known-class point prediction |
| $A(\cdot),\ s(\cdot)$ | accept/reject selector; underlying score |
| $a_j,\ r_j,\ m_j$ | per-client acceptance rate; conditional error; accepted-error mass $m_j=a_jr_j$ |
| $R_{\mathrm{sel}}(\lambda)$ | global accepted selective risk $=\sum\lambda_j m_j/\sum\lambda_j a_j$ |
| $n_j,\ A_j,\ K_j$ | per-client certification size; accepted count; accepted-error count |
| $U^+,L^-$ | one-sided upper/lower Clopper–Pearson limits |
| $\bar r_j$ | per-client upper CP bound on conditional risk $r_j$ (main certificate) |
| $\underline a_j,\bar a_j$ | per-client lower/upper CP bounds on $a_j$ (bounded-$\Lambda$ certificate) |
| $\bar m_j$ | per-client upper bound on $m_j$ (mass-ratio baseline, App. B) |
| $\bar U_\Delta^{\,r},\ \bar U_\Lambda^{\,r,a}$ | conditional certificate (simplex; bounded-$\Lambda$); deploy iff $\le\alpha$ |
| $w_j(\lambda)$ | acceptance-reweighted mixture weight $\lambda_j a_j/\sum_\ell\lambda_\ell a_\ell$ |
| $\alpha,\delta,\gamma$ | risk tolerance; certificate confidence; proposal risk buffer |

## Appendix B. Mass-ratio baseline certificate (valid but looser)

The original certificate bounds $m_j$ and $a_j$ separately: $\bar m_j=U^+(K_j,n_j;\delta/2J)$, $\underline a_j=L^-(A_j,n_j;\delta/2J)$, $\bar U_\Lambda=\sup_{\lambda\in\Lambda}(\sum\lambda\bar m)/(\sum\lambda\underline a)$, simplex closed form $\max_j\bar m_j/\underline a_j$. It is valid (same monotone-transport proof) but uniformly looser than the conditional certificate of Section 4.2 because it pays a separate acceptance-lower-bound slack and a $2J$-way (vs. $J$-way) union. It is retained only as a sanity baseline; the conditional certificate is the main result.

## Appendix C. Supplementary synthetic ablations

**Table C1. Score-agnostic validity (four scores; all valid, coverage differs).**

| score | realized test_risk | CertifiedCoverage@$\alpha$ |
|---|---|---|
| MSP | 0.042–0.046 ($\le\alpha$) | 0.64–0.66 |
| entropy | 0.042–0.046 ($\le\alpha$) | 0.64–0.66 |
| margin | 0.042–0.046 ($\le\alpha$) | 0.64–0.66 |
| energy | 0.042–0.046 ($\le\alpha$) | 0.64–0.66 |

All four scores keep the realized risk $\le\alpha$; only the certified coverage differs, confirming that validity is produced by the certification split rather than by the score (Section 4.6).

Two of the three synthetic method-knob ablations are main figures — the calibration-budget sweep is panel (c) of Figure 4, and the unknown-proportion (A4) sweep is Figure 3 — because both are central (a feasibility lever and a validity condition). The remaining supplementary ablation, client scaling, is kept here.

![Figure 8](experiments/fedcore/figs/FJ_client_scaling.png)

**Figure 8 (J). Per-client starvation / $\log J$ penalty.** With the *total* calibration budget fixed and split across $J$ clients, the worst-group cert_ucb rises ($0.064\to0.276$ as $J:2\to20$), crossing $\alpha{=}0.10$ around $J{=}5$ — the per-client feasibility threshold of Theorem 2 in action: more clients means thinner per-client folds and a looser worst-group bound. Client-participation subsampling behaves identically (it lowers per-client counts).

*FOOGD-SAG reproduction analysis (supporting Section 5.6).* The full FOOGD-SAG pipeline (WideResNet-40-2 with annealed denoising score matching and a KSD/MMD Stein regularizer, trained jointly) reaches only AUROC $0.467$ on our six-known semantic-shift split at a single-GPU budget — plausibly because the pipeline does not standardize the score-model input (a decisive fix in the representative protocol), trains the backbone jointly, and targets covariate-shift OOD over long sweeps rather than the semantic-shift split used here. Reproducing a SOTA FedOSR method at published strength on a new split is itself expensive and configuration-sensitive; the certificate's validity is unaffected.

*Certified self-training gain (supporting Section 5.7).* The accuracy-gain result discussed in Section 5.7 is summarized in Figure 9. We keep it supplementary because, while safe, the gain is seed-variable and at $n{=}3$ not statistically separable from zero; the primary self-training result remains the contamination gate of Figure 7. In the deeper label-scarce regime ($10\%$ labels) the clean-pseudo-label oracle headroom grows to $+0.16\pm0.05$ while certified admission occurs on only $1/3$ seeds (realized certified gain $+0.04\pm0.07$; $0$ contamination violations at every step, all admitted batches $\le\alpha$). Admission is not monotone in detector strength at this budget — the strongest-base seed admits nothing — which is Theorem-2 feasibility noise, not a detector deficiency. We verified the audit-budget rescue only on a seed that already admitted; whether a larger budget rescues the zero-admission seeds is left open.

![Figure 9](experiments/fedcore/figs/F_selftrain_gain.png)

**Figure 9 (supplementary). Certified one-shot self-training: a safe accuracy gain on a strong base detector** ($\alpha{=}0.20$, audit budget $4\times$, one-shot certification). **(a)** Certified known-accuracy gain over the no-pseudo-label baseline, per base model: the certified (Fed-CORE) bar and the clean-pseudo-label oracle upper bound, with $\pm1$ SD and per-seed points. The weaker FedAvg+MSP base ($n{=}1$) gains $+0.012$ (oracle $+0.043$); the stronger FedPD-PROSER base ($n{=}3$) gains $+0.030$ (oracle $+0.045$), with $2/3$ seeds positive and the third feasibility-limited to zero admissions — a stronger detector yields a larger safe gain. The statistical claim is the $t$-interval reported in Section 5.6 ($95\%$ CI $[-0.04,+0.10]$, positive but not separable from zero at $n{=}3$); the error bars are illustrative. **(b)** Safety: the realized contamination of every admitted batch stays $\le\alpha{=}0.20$ (FedAvg certified $0.136$, FedPD certified $0.124$), whereas naive self-training sits at the contamination budget ($0.200$). Across all runs there are $0/N$ certified contamination violations.

## Appendix D. Full proofs

Throughout, the selector $A$ is fixed on the proposal fold and is therefore independent of the certification fold; within each client the certification draws are i.i.d. from $P_j$. We write $U^+(\cdot,\cdot;\varepsilon)$ and $L^-(\cdot,\cdot;\varepsilon)$ for the one-sided Clopper–Pearson limits of Section 4.1, which satisfy, for $K\sim\mathrm{Bin}(n,p)$ and every $p$, $\Pr(p\le U^+(K,n;\varepsilon))\ge1-\varepsilon$ and $\Pr(p\ge L^-(K,n;\varepsilon))\ge1-\varepsilon$.

**Proposition 1 (necessity of trusted calibration labels).** *Consider any procedure $\Pi$ that observes the model's outputs on the calibration data together with corrupted or unlabeled calibration data (but no trusted clean labels), and outputs either "deploy" with a certificate $R_{\mathrm{sel}}(\lambda)\le\alpha$, $\alpha<1$, at confidence $1-\delta$, $\delta<1/2$, or "abstain." If $\Pi$ deploys with probability bounded away from zero on some instance with acceptance $a_\lambda>0$, then $\Pi$'s certificate is violated with probability $>\delta$ on another instance that $\Pi$ cannot distinguish from the first.*

*Proof.* Fix an instance $\omega_A$ (World A) on which $\Pi$ deploys with probability $p_0>0$: a model $\hat h$, a selector-eligible score, and a calibration sample whose *clean* labels agree with $\hat h$'s predictions on all accepted points, so $R_{\mathrm{sel}}(\lambda)=0\le\alpha$. Construct World B by flipping the clean labels of the accepted calibration points (and, for the population statement, the conditional label distribution on the acceptance region) while leaving the corrupted/unlabeled observations and all model outputs unchanged — possible because the corruption process is not assumed known or invertible and unlabeled data carry no label information. In World B the accepted predictions are wrong, so $R_{\mathrm{sel}}(\lambda)=1>\alpha$. The observable input to $\Pi$ is identical in both worlds, hence $\Pi$ deploys in World B with the same probability $p_0$, and each such deployment violates the certificate. For $\Pi$ to be valid, $p_0\le\delta$ must hold — but $\delta<1/2$ and the construction applies to *every* instance on which $\Pi$ deploys, so $\Pi$ deploys with probability $\le\delta$ everywhere: it is vacuous. Trusted clean labels on the certification fold break the indistinguishability, since the two worlds then differ observably. $\square$

**Proposition 2 (naive pooling is anti-conservative under mixture shift).** *Let $U_{\mathrm{pool}}=U^+\big(\sum_j K_j,\sum_j A_j;\delta\big)$. There exist client populations $\{(a_j,r_j)\}_{j\le J}$ and deployment mixtures $\lambda\in\Delta^{J-1}$ such that $\Pr\big(R_{\mathrm{sel}}(\lambda)>U_{\mathrm{pool}}\big)\to1$ as $\min_j n_j\to\infty$.*

*Proof.* Take equal calibration sizes $n_j=n$ and the population of Section 4.2: $J{=}5$, four clients with $(a,r)=(0.7,0.02)$ and one with $(a,r)=(0.5,0.3)$. Conditioned on the accepted counts, $\sum_j K_j$ is a sum of independent binomials with unequal success probabilities — a Poisson-binomial — so the single-binomial Clopper–Pearson calibration does not apply to it; quantitatively, by the strong law of large numbers, $\sum_j K_j/\sum_j A_j\to\bar r_{\mathrm{cal}}=\sum_j a_j r_j/\sum_j a_j\approx0.0738$ a.s., and $U_{\mathrm{pool}}\to\bar r_{\mathrm{cal}}$ as well, since the Clopper–Pearson width shrinks as $O(\sqrt{\log(1/\delta)/n})$. Now let $\lambda$ put all deployment mass on the high-risk client: $R_{\mathrm{sel}}(\lambda)=0.3$. For any fixed margin $0<c<0.3-\bar r_{\mathrm{cal}}$, $\Pr(U_{\mathrm{pool}}<\bar r_{\mathrm{cal}}+c)\to1$, hence $\Pr(R_{\mathrm{sel}}(\lambda)>U_{\mathrm{pool}})\to1$. The same argument applies to any $\lambda$ with $R_{\mathrm{sel}}(\lambda)>\bar r_{\mathrm{cal}}$, i.e., to every mixture that overweights above-average-risk clients. $\square$

**Lemma A (conditional coverage of the per-client bound).** *For each client $j$, $\Pr\big(r_j\le U^+(K_j,A_j;\varepsilon)\big)\ge1-\varepsilon$.*

*Proof.* Conditional on $A_j=a$, the accepted-error count is $K_j\mid A_j{=}a\sim\mathrm{Bin}(a,r_j)$, a binomial with $a$ trials and success probability $r_j$. The Clopper–Pearson guarantee gives $\Pr(r_j\le U^+(K_j,a;\varepsilon)\mid A_j{=}a)\ge1-\varepsilon$ for every value $a$ (including $a{=}0$, where $U^+:=1$ trivially covers $r_j$). Since the bound holds for every conditioning value, it holds marginally by the tower property: $\Pr(r_j\le U^+(K_j,A_j;\varepsilon))=\mathbb E_{A_j}\big[\Pr(r_j\le U^+\mid A_j)\big]\ge1-\varepsilon$. $\square$

**Theorem 1 (full simplex).** *With $\varepsilon=\delta/J$ and $\bar U_\Delta^{\,r}=\max_j U^+(K_j,A_j;\delta/J)$, $\Pr\big(R_{\mathrm{sel}}(\lambda)\le\bar U_\Delta^{\,r}\ \text{for all }\lambda\in\Delta^{J-1}\big)\ge1-\delta$.*

*Proof.* Let $\bar r_j=U^+(K_j,A_j;\delta/J)$ and $E=\{\,\forall j:\ r_j\le\bar r_j\,\}$. By Lemma A each event $\{r_j\le\bar r_j\}$ has probability $\ge1-\delta/J$, so by the union bound $\Pr(E)\ge1-J\cdot(\delta/J)=1-\delta$. Fix any $\lambda\in\Delta^{J-1}$ with $\sum_j\lambda_j a_j>0$. Writing the deployment risk as the acceptance-reweighted convex combination $R_{\mathrm{sel}}(\lambda)=\sum_j w_j(\lambda)\,r_j$ with weights $w_j(\lambda)=\lambda_j a_j/\sum_\ell\lambda_\ell a_\ell\ge0$ and $\sum_j w_j(\lambda)=1$, we have on $E$:
$$
R_{\mathrm{sel}}(\lambda)=\sum_j w_j(\lambda)\,r_j\ \le\ \sum_j w_j(\lambda)\,\bar r_j\ \le\ \Big(\max_j\bar r_j\Big)\sum_j w_j(\lambda)=\max_j\bar r_j=\bar U_\Delta^{\,r}.
$$
The bound $\max_j\bar r_j$ does not depend on $\lambda$, so on $E$ it holds simultaneously for all admissible $\lambda$. Hence the displayed event has probability $\ge\Pr(E)\ge1-\delta$. $\square$

**Theorem 1′ (bounded $\Lambda$).** *With $\varepsilon=\delta/3J$, $\bar r_j=U^+(K_j,A_j;\varepsilon)$, $\underline a_j=L^-(A_j,n_j;\varepsilon)$, $\bar a_j=U^+(A_j,n_j;\varepsilon)$, and $\bar U_\Lambda^{\,r,a}=\sup_{\lambda\in\Lambda,\,a_j\in[\underline a_j,\bar a_j]}\big(\sum_j\lambda_j a_j\bar r_j\big)/\big(\sum_j\lambda_j a_j\big)$, the true $\lambda^\star\in\Lambda$ satisfies $\Pr\big(R_{\mathrm{sel}}(\lambda^\star)\le\bar U_\Lambda^{\,r,a}\big)\ge1-\delta$.*

*Proof.* Define the $3J$ events $\{r_j\le\bar r_j\}$, $\{a_j\ge\underline a_j\}$, $\{a_j\le\bar a_j\}$, each of probability $\ge1-\delta/3J$ (Lemma A for the first; the Clopper–Pearson lower/upper guarantees applied to $A_j\sim\mathrm{Bin}(n_j,a_j)$ for the others). By the union bound their intersection $E'$ has probability $\ge1-3J\cdot(\delta/3J)=1-\delta$. On $E'$ the true parameters satisfy $r_j\le\bar r_j$ and $a_j^\star\in[\underline a_j,\bar a_j]$, so the pair $(\lambda^\star,a^\star)$ is feasible for the supremum and, using $r_j\le\bar r_j$ in the numerator,
$$
R_{\mathrm{sel}}(\lambda^\star)=\frac{\sum_j\lambda_j^\star a_j^\star r_j}{\sum_j\lambda_j^\star a_j^\star}\ \le\ \frac{\sum_j\lambda_j^\star a_j^\star \bar r_j}{\sum_j\lambda_j^\star a_j^\star}\ \le\ \bar U_\Lambda^{\,r,a}.
$$
The inner supremum over $a$ is a linear-fractional objective that is monotone in each $a_j$, hence attained at a box vertex $a_j\in\{\underline a_j,\bar a_j\}$; the outer supremum over $\lambda\in\Lambda$ is a linear-fractional program solved by the Charnes–Cooper transformation. $\square$

**Theorem 2 (feasibility threshold).** *In the zero-accepted-error regime $K_j=0$ on the simplex, the deploy condition $\max_j U^+(0,A_j;\delta/J)\le\alpha$ requires $A_j\ge \ln(J/\delta)/(-\ln(1-\alpha))=\Omega\big(\ln(J/\delta)/\alpha\big)$ for every binding client.*

*Proof.* For $K=0$ the upper Clopper–Pearson limit is $U^+(0,A_j;\varepsilon)=1-\varepsilon^{1/A_j}$ with $\varepsilon=\delta/J$ (the solution of $(1-p)^{A_j}=\varepsilon$). The per-client deploy condition $U^+(0,A_j;\delta/J)\le\alpha$ is therefore
$$
1-(\delta/J)^{1/A_j}\le\alpha\ \Longleftrightarrow\ (\delta/J)^{1/A_j}\ge 1-\alpha\ \Longleftrightarrow\ \tfrac{1}{A_j}\ln(\delta/J)\ge\ln(1-\alpha).
$$
Both $\ln(\delta/J)$ and $\ln(1-\alpha)$ are negative; dividing and flipping the inequality gives $A_j\ge \ln(\delta/J)/\ln(1-\alpha)=\ln(J/\delta)/(-\ln(1-\alpha))$. Since $-\ln(1-\alpha)=\alpha+\alpha^2/2+\cdots\ge\alpha$, this lower bound is $\Omega(\ln(J/\delta)/\alpha)$. The expected-count form $n_j a_j\gtrsim\ln(J/\delta)/\alpha$ follows from $\mathbb E[A_j]=n_j a_j$ and a concentration of $A_j$ around its mean. $\square$

**Proposition 4 (round-wise self-training validity).** *Partition the trusted set into $T$ disjoint audit folds $\mathcal C^{(1)},\dots,\mathcal C^{(T)}$. If the round-$t$ model and selector $(f_t,A_t)$ are formed only from folds indexed $<t$ and from unlabeled data — so that $(f_t,A_t)$ is independent of $\mathcal C^{(t)}$ — and each round is certified at level $\delta/T$, then $\Pr\big(\forall t\le T:\ R_{\mathrm{sel}}(A_t)\le\bar U^{(t)}\big)\ge1-\delta$.*

*Proof.* By construction $\mathcal C^{(t)}$ is independent of $(f_t,A_t)$, so $A_t$ is a fixed selector with respect to the fresh fold $\mathcal C^{(t)}$ and the i.i.d. assumption of Theorem 1/1′ applies on that fold. Certifying round $t$ at level $\delta/T$ therefore gives $\Pr\big(R_{\mathrm{sel}}(A_t)\le\bar U^{(t)}\big)\ge1-\delta/T$, where the bound $\bar U^{(t)}$ is computed from $\mathcal C^{(t)}$. Let $B_t=\{R_{\mathrm{sel}}(A_t)>\bar U^{(t)}\}$; then $\Pr(B_t)\le\delta/T$, and by the union bound $\Pr\big(\bigcup_{t\le T}B_t\big)\le T\cdot(\delta/T)=\delta$. Taking complements gives $\Pr\big(\forall t\le T:\ R_{\mathrm{sel}}(A_t)\le\bar U^{(t)}\big)\ge1-\delta$. Each admitted pseudo-label batch thus has certified contamination $\le\alpha$ simultaneously across all $T$ rounds. $\square$

## References

[1] Yang et al., FedPD: Federated open set recognition with parameter disentanglement, in: ICCV, 2023.

[2] C. Yang, M. Zhu, Y. Liu, Y. Yuan, FedPD++: Enhanced federated open-set recognition with parameter disentanglement, Int. J. Comput. Vis. (2026).

[3] FOOGD: Federated collaboration for both OOD generalization and detection, in: NeurIPS, 2024, arXiv:2410.11397.

[4] E. Diao, J. Li, Z. He, Towards addressing label skews in one-shot federated learning (FedOV), in: ICLR, 2023.

[5] Wang, Liu, Guo, Dong, Wang, Huang, Zhu, Federated continual novel class learning (FedNovel), arXiv:2312.13500, 2023.

[6] K. Lu, Y. Yu, S.P. Karimireddy, M. Jordan, R. Raskar, Federated conformal predictors for distributed uncertainty quantification, in: ICML, 2023, arXiv:2305.17564.

[7] V. Plassier, M. Makni, A. Rubashevskii, E. Moulines, M. Panov, Conformal prediction for federated uncertainty quantification under label shift, in: ICML, 2023, arXiv:2306.05131.

[8] Certifiably Byzantine-robust federated conformal prediction (Rob-FCP), arXiv:2406.01960, 2024.

[9] Zhang et al., Towards unbiased training in federated open-world semi-supervised learning, in: ICML, 2023.

[10] Turning the curse of heterogeneity in FL into a blessing for OOD detection (FOSTER), in: ICLR, 2023.

[11] Gao, Liu, Qin, Ou, Noise-resistant federated open set recognition, in: KSEM, 2025.

[12] Adversarial compact wrapping classifier learning for open set recognition, Inf. Sci. (2024).

[13] Towards heterogeneous federated graph learning via structural entropy and prototype aggregation, Inf. Sci. 718 (2025) 122338.

[14] Personalized federated learning: A clustered distributed co-meta-learning approach, Inf. Sci. (2023).

[15] X. Li, S. Zhao, C. Chen, Z. Zheng, Heterogeneity-aware fair federated learning, Inf. Sci. 619 (2023) 968–986.

[16] Z. Pan, C. Li, F. Yu, S. Wang, X. Tang, J. Zhao, Balancing the trade-off between global and personalized performance in federated learning, Inf. Sci. 712 (2025) 122154.

[17] X. Yu, Z. Liu, W. Wang, Y. Sun, Clustered federated learning based on nonconvex pairwise fusion, Inf. Sci. 678 (2024) 120956.

[18] H. Yang, W. Xi, Z. Wang, Y. Shen, X. Ji, C. Sun, J. Zhao, FedRich: Towards efficient federated learning for heterogeneous clients using heuristic scheduling, Inf. Sci. 645 (2023) 119360.

[19] X. Zhou, G. Yang, Communication-efficient and privacy-preserving large-scale federated learning counteracting heterogeneity, Inf. Sci. 661 (2024) 120167.

[20] Exploiting reject option in classification for social discrimination control, Inf. Sci. (2017).

[21] Graph autoencoder-based unsupervised outlier detection, Inf. Sci. (2022).

[22] Concept drift detection with quadtree-based spatial mapping of streaming data, Inf. Sci. (2023).

[23] A.N. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster, Conformal risk control, in: ICLR, 2024, arXiv:2208.02814.

[24] S. Bates, A.N. Angelopoulos, L. Lei, J. Malik, M.I. Jordan, Distribution-free, risk-controlling prediction sets, J. ACM 68 (2021) 1–34.

[25] Y. Xu, W. Guo, Z. Wei, Selective conformal risk control, arXiv:2512.12844, 2025.

[26] Conformal selective prediction with general risk control (SCoRE), arXiv:2603.24704 (e-value selective risk; centralized).

[27] Y. Xie, Y. Zhou, T. Liang, S. Favaro, M. Sesia, Conformal inference for open-set and imbalanced classification, arXiv:2510.13037, 2025.

[28] Classification with reject option: Distribution-free error guarantees via conformal prediction, Mach. Learn. Appl. (2025).

[29] Decentralized conformal novelty detection via quantized model exchange, arXiv:2605.08263, 2026.

[30] Zhu, Liao, Liu, Yuan, FedOSS: Federated open set recognition via inter-client discrepancy and collaboration, IEEE Trans. Med. Imaging (2023).

[31] D. Hendrycks, K. Gimpel, A baseline for detecting misclassified and out-of-distribution examples in neural networks, in: ICLR, 2017.

[32] W. Liu, X. Wang, J.D. Owens, Y. Li, Energy-based out-of-distribution detection, in: NeurIPS, 2020.

[33] Y. Geifman, R. El-Yaniv, SelectiveNet: A deep neural network with an integrated reject option, in: ICML, 2019.

[34] C.J. Clopper, E.S. Pearson, The use of confidence or fiducial limits illustrated in the case of the binomial, Biometrika 26 (1934) 404–413.

[35] W. Hoeffding, On the distribution of the number of successes in independent trials, Ann. Math. Statist. 27 (1956) 713–721.

[36] B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas, Communication-efficient learning of deep networks from decentralized data, in: AISTATS, 2017.

[37] X. Yu, J. Liu, A joint finite-sample certificate for adaptive selective conformal risk control, arXiv:2606.08517, 2026.

