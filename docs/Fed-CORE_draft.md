# Fed-CORE: Federated Certified Open-Set Recognition via Selective Risk Control

Sanghoon Kim

[Department / University, City, Country]

Corresponding author. E-mail address: october25kim@gmail.com

---

## Abstract

Federated learning is increasingly deployed in safety-sensitive applications where, at test time, inputs may belong to classes never seen during training. Existing federated open-set recognition methods reject unknowns but evaluate rejection only empirically (AUROC, FPR95), with no guarantee on the error rate of the predictions they accept; federated conformal prediction provides finite-sample guarantees, but only for closed-set prediction-set coverage. The purpose of this study is to certify the accepted selective risk of a federated open-set classifier, namely the probability that an accepted prediction is wrong, with a finite-sample distribution-free guarantee under client heterogeneity and an unknown deployment mixture. We propose Fed-CORE, a post-hoc certification layer for any federated open-set model. The key obstacle is that naively pooling accepted calibration points across heterogeneous clients is anti-conservative, because the pooled accepted-error count is Poisson-binomial rather than binomial; we instead bound each client's conditional selective risk directly — from the binomial law of its accepted errors given its accepted count — and certify the global risk as a mixture-robust worst case. We derive a per-client feasibility threshold and connect the certificate to safe automation and certified federated self-training. Across synthetic and real federated benchmarks (CIFAR-10/100 and a tabular task) the certificate is valid in all tested settings under a stated audit-representativeness condition (the audit fold must represent, or over-represent, the deployment unknown incidence), naive pooling collapses as predicted, and certified coverage follows a feasibility law. On CIFAR-10 the method attains a five-seed worst-group certified coverage of 0.39 at a 0.20 risk target with no false certificate — reported as a finite-sample feasibility demonstration rather than a safety recommendation — while smaller risk targets and a second domain sit at the feasibility edge.

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
- We derive a finite-sample, distribution-free certificate for $R_{\mathrm{sel}}(\lambda)$ from the conditional law $K_j\mid A_j\sim\mathrm{Bin}(A_j,r_j)$, valid under client heterogeneity and unknown deployment mixture (Theorem 1/1′), together with a per-client feasibility threshold (Theorem 2) and an honest privacy taxonomy; we prove that naive pooling of federated calibration counts is anti-conservative and is therefore not reducible to centralized conformal prediction (Section 4).
- We characterize when certified open-set deployment is statistically feasible through a feasibility law in the risk target, the per-group audit count, heterogeneity, and corruption, turning the apparent null at small risk targets into a quantitative phenomenon (Sections 4.3 and 5).
- We present two downstream uses of the certificate, namely safe automation at guaranteed risk and certified federated self-training with bounded contamination (Proposition 4), and evaluate them on synthetic and real federated benchmarks, where the certificate is valid in all tested settings under the stated audit-representativeness condition (Sections 4.7 and 5).

Positioning note: Fed-CORE is best read not as a new FedOSR algorithm but as a **certification layer for any FedOSR / open-set FL model**, whose output is *used* for safe automation and certified self-training.

**Findings (preview).** Across synthetic and real federated benchmarks (CIFAR-10/100 and a tabular FL task) the certificate is valid in all tested settings under the stated audit-representativeness condition — no false certificate in any seed, data set, or normalization — and naive pooling collapses under mixture shift as predicted. Certified coverage obeys a **feasibility law** ($(\alpha-\hat r)^{-2}$ per-group sample threshold); we report a 5-seed worst-group $\alpha{=}0.20$ certified-coverage positive on CIFAR-10 ResNet-GroupNorm ($0.392\pm0.097$ at $d{=}5$, $0.353\pm0.130$ at $d{=}0.5$, both $5/5$, $0/10$ false certificates), with a second federated domain (tabular FL) exhibiting the same feasibility law at its edge (seed-variable, non-vacuous only under selection-optimistic scoring) and the $\alpha{=}0.10$ regime a seed-variable feasibility edge. The honest message is a *characterized, valid certificate*, not a headline accuracy number.

The remainder of this paper is organized as follows. Section 2 reviews related work in federated open-set recognition, federated conformal prediction, and conformal risk control. Section 3 formalizes the problem and the accepted selective risk. Section 4 develops the certificate, its feasibility threshold, the privacy taxonomy, and the two downstream uses. Section 5 reports the experiments, and Section 6 discusses limitations. Section 7 concludes.

---

## 2. Related Work

**Federated open-set / novel-class / OOD recognition.** FedPD [1] and its extension FedPD++ [2] frame FedOSR around two pathologies — cross-client *inter-set interference* between closed- and open-set objectives, and cross-client *intra-set inconsistency* from heterogeneity — and address them by parameter disentanglement and divide-and-conquer aggregation. FOOGD [3] jointly targets OOD generalization and detection: its SM³D module learns a feature-space score model (detection rule: threshold on the score norm), and its SAG module regularizes feature invariance via Stein's identity. Crucially, FOOGD's only formal result bounds the *estimation error of the score model* (an MMD bound), not the error rate of the rejection decision; detection is reported via AUROC/FPR95. FedOV [4] tackles one-shot FL under label skew by training each client to emit an "unknown" class and ensembling via open-set voting. FedNovel [5] and federated open-world semi-supervised learning [9] discover and learn novel classes across clients, and client heterogeneity itself has been recast as a signal for OOD detection (FOSTER) [10]. Noise-Resistant FedOSR [11] is the closest to a corruption setting, using Bayesian uncertainty and label correction — but, like all of the above, evaluates rejection empirically. Centralized OSR continues to refine compact accept/reject regions, for example adversarial compact wrapping classifiers [12], and federated heterogeneity is studied well beyond images, for example on graphs via structural-entropy prototype aggregation [13]; both report empirical accuracy or detection quality rather than a guaranteed accepted-error rate. Several adjacent lines each address one facet of deployment reliability — federated personalization and fairness under statistical heterogeneity [14,15,16], client clustering and heuristic scheduling for heterogeneous federations [17,18], communication-efficient privacy-preserving aggregation under heterogeneity [19], reject-option classification [20], unsupervised outlier detection [21], and concept-drift detection in data streams [22] — covering heterogeneity, abstention, novelty, and distribution shift, respectively, yet none certifies a finite-sample bound on the accepted-prediction error rate. **None provides a finite-sample certificate on accepted predictions.**

**Federated conformal / distribution-free uncertainty.** Lu et al. [6] introduce Federated Conformal Prediction, replacing exchangeability with *partial exchangeability* (a test point matches client $k$ with probability $\lambda_k$) and proving marginal coverage $1-\alpha \le \Pr(Y\in C) \le 1-\alpha + K/(N+K)$ with a privacy-preserving T-Digest quantile sketch. Plassier et al. [7] handle label shift via importance weighting; Rob-FCP [8] adds Byzantine robustness; generative and conditional variants refine conditional coverage. All certify **closed-set prediction-set coverage**, not unknown rejection or selective risk. We adopt the partial-exchangeability viewpoint but certify a *risk* (a binomial functional), which behaves differently from a *quantile* and is not addressed by these works.

**Conformal risk control and selective conformal prediction (centralized).** Conformal Risk Control [23] generalizes conformal coverage to any monotone risk, building on distribution-free risk-controlling prediction sets [24]. A fast-moving 2025–2026 line then certifies *post-selection* risk: Selective Conformal Risk Control [25] two-stage selects then controls risk on the selected subset, and SCoRE [26] controls risk among "trusted/positive" cases via e-values. These are powerful but **centralized / exchangeable**: none addresses federated heterogeneity, unknown client mixture, or count-only aggregation — the axes Fed-CORE is built on. We compare against them as **centralized oracles** (upper bounds), not as drop-in baselines.

**Conformal open-set / novelty detection.** Conformal open-set classification via Good–Turing p-values [27] and reject-option conformal classification [28] certify rejection centrally. The sole decentralized entry, *Decentralized Conformal Novelty Detection* [29], controls **global FDR over a test batch** via quantized surrogate scores — pure novelty detection (no known-class classifier), a different functional than accepted selective risk; we use it as the nearest neighbor and differentiate on object and on certifying a single fixed selector.

**Positioning.** Existing selective conformal/risk-control methods certify post-selection risk in centralized/exchangeable settings; federated conformal methods (including FedOSS-style FedOSR [30] and FCP [6]) certify either empirical rejection quality or closed-set prediction-set coverage. Fed-CORE fills the missing intersection: **federated open-set accepted-risk certification under client heterogeneity and deployment-mixture uncertainty.** It contributes the conditional selective-risk certificate (and its mixture-robust form) that the federated setting newly requires; the calibration statistic is a conditional-binomial proportion, not a score quantile.

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
where $a_j=\Pr_{P_j}(A(X)=1)$ is the per-client acceptance rate and $m_j=\Pr_{P_j}(A(X)=1,\ \hat y(X)\ne Y)$ is the per-client accepted-error mass (so the per-client selective risk is $r_j=m_j/a_j$). The ratio form follows from the law of total probability under the mixture. **Goal:** deploy $A$ only if we can certify $R_{\mathrm{sel}}(\lambda)\le\alpha$ for the (unknown) deployment $\lambda\in\Lambda$ with confidence $1-\delta$, while maximizing accepted coverage $\mathrm{cov}(\lambda)=\sum_j\lambda_j a_j$.

**Trusted calibration data and the split.** Each client holds a small *trusted, clean* calibration sample, partitioned into a **proposal** fold and a **certification** fold. The selector $A$ is chosen on the proposal fold (across clients) and is therefore *fixed and independent* of the certification fold. On the certification fold, client $j$ contributes $n_j$ i.i.d. draws from $P_j$ and reports two integers:
$$
A_j=\sum_{i=1}^{n_j}\mathbf 1\{A(x_i)=1\},\qquad
K_j=\sum_{i=1}^{n_j}\mathbf 1\{A(x_i)=1,\ \hat y(x_i)\ne y_i\}.
$$
By construction $A_j\sim\mathrm{Bin}(n_j,a_j)$ and $K_j\sim\mathrm{Bin}(n_j,m_j)$ with $K_j\le A_j$; conditionally, $K_j\mid A_j\sim\mathrm{Bin}(A_j,r_j)$. What must leave the client depends on the certificate (per-client pairs for the stratified one, sums for the pooled one); see the corrected privacy taxonomy in Section 4.4.

**Calibration must contain labeled unknowns (stated openly).** To certify *unknown rejection*, the certification fold must include points with $Y=\textsf{unknown}$ that are **labeled as such**. Unknown classes are unseen during training but present and labeled in this small post-training **audit/calibration fold**. Hence "distribution-free" is *with respect to the calibration distribution $Q_\lambda$*, not the entire unknown universe: the guarantee protects against the unknowns represented in the audit fold. In OSR benchmarks this holds by construction (held-out classes excluded from training, used as unknown-labeled calibration/test examples); in deployment it corresponds to a small audited monitoring set. If the audit fold has no unknowns, the certificate degrades to a closed-set selective-risk guarantee. We further assume the audit distribution is **representative of, or conservative with respect to, the deployment unknown incidence**: the audit fold should carry unknowns at no less than their deployment rate. This assumption is necessary — under-representing unknowns makes the certificate anti-conservative (Section 5.6 and Appendix C) — and we state it openly here so that every later validity claim is read under it. We certify only selectors with positive accepted coverage $a_\lambda=\sum_j\lambda_j a_j>0$; a zero-coverage selector is non-deployable.

**Risk-buffered proposal.** To avoid certifying a selector whose empirical risk already sits at $\alpha$ (which makes the certificate fail), the proposal fold selects $A$ subject to an empirical buffer $\widehat R_{\mathrm{prop}}(A)\le\gamma\alpha$ with $0<\gamma<1$ (default candidates $\gamma\in\{0.5,0.7,1.0\}$), inherited from the centralized framework.

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

### 4.2 Theorem 1 — Conditional selective-risk certificate (the core, N1)

In words, the certificate does the following. For each client we look only at the audit points that the model actually accepted, count how many of those were wrong, and use an exact binomial confidence bound (Section 4.1) to obtain an upper bound on that client's true error rate. The global guarantee is then driven by the worst client, because — as explained above — no client's errors can be hidden behind another's. The rest of this subsection makes this precise.

The sharpest certificate works directly with the per-client **conditional** selective risk $r_j=\Pr_{P_j}(\hat y(X)\ne Y\mid A(X)=1)=m_j/a_j$. Conditional on the accepted count $A_j$, the accepted-error count is exactly $K_j\mid A_j\sim\mathrm{Bin}(A_j,r_j)$, so a single Clopper–Pearson upper limit on the *observed accepted sub-sample* bounds $r_j$ with **no acceptance-rate slack**:
$$
\bar r_j=U^+(K_j,A_j;\varepsilon)\qquad(\bar r_j:=1\text{ if }A_j=0).
$$
Writing the global risk as an **acceptance-reweighted convex combination** makes the certificate transparent:
$$
R_{\mathrm{sel}}(\lambda)=\sum_j w_j(\lambda)\,r_j,\qquad
w_j(\lambda)=\frac{\lambda_j a_j}{\sum_\ell \lambda_\ell a_\ell},\quad \sum_j w_j(\lambda)=1 .
$$

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

**Edge cases (in the statement, not the proof).** (i) *Zero accepted coverage.* We certify only selectors with $a_\lambda=\sum_j\lambda_j a_j>0$; if $a_\lambda=0$ the selector accepts nothing and is **non-deployable** ($R_{\mathrm{sel}}$ undefined). (ii) *Vanishing denominator bound.* If $\inf_{\lambda\in\Lambda}\sum_j\lambda_j\underline a_j=0$ (a client may accept yet $\underline a_j=0$ when $A_j$ is small) the robust certificate is declared **infeasible** ($\bar U=+\infty$); we do *not* silently drop such a client, which would break the worst-case guarantee.

**Proof sketch (Theorem 1).** Conditional on $A_j$, $K_j\sim\mathrm{Bin}(A_j,r_j)$, so $\Pr(r_j\le\bar r_j)\ge1-\varepsilon$ for each $j$ (and hence marginally, as it holds for every value of $A_j$). A union over the $J$ clients with $\varepsilon=\delta/J$ gives the event $E=\{\forall j:\ r_j\le\bar r_j\}$ with $\Pr(E)\ge1-\delta$. On $E$, for **any** $\lambda$, the convex weights $w_j(\lambda)\ge0$, $\sum_j w_j=1$ give $R_{\mathrm{sel}}(\lambda)=\sum_j w_j r_j\le\sum_j w_j\bar r_j\le\max_j\bar r_j=\bar U_\Delta^{\,r}$. For Theorem 1′, additionally $a_j\in[\underline a_j,\bar a_j]$ on the corresponding event (now $3J$ events at $\delta/3J$), and $\sup$ over the admissible $(\lambda,a)$ dominates the true value. $\square$

**Non-reducibility (why this is N1, not a corollary).** (a) Not centralized CP on pooled data: pooling is invalid because the $\{r_j\}$ differ, so the pooled accepted-error count is Poisson-binomial, not binomial (Section 4.5; ablation in Section 5). (b) Not Lu's federated-conformal certificate: that controls a *quantile/coverage* of nonconformity scores under partial exchangeability; Fed-CORE controls a **post-selection conditional error ratio** under client-mixture uncertainty. The calibration statistic is not a quantile of scores but a **conditional-binomial proportion** $K_j\mid A_j$ — a different functional with a different finite-sample construction. (c) *Not "merely a Bonferroni union of per-client binomial CIs."* The certified object is the post-selection accepted-risk **ratio** $R_{\mathrm{sel}}(\lambda)$ under an *unknown* deployment mixture — the per-client $r_j$ are combined through acceptance-reweighted convex weights, and the deployment certificate (Thm 1′) is a *robust linear-fractional program* over $(\lambda,a)$, not a maximum of independent intervals. The contribution is formalizing this object and exposing that the *intuitive* fix (pool, then one CP) is anti-conservative — not the Clopper–Pearson arithmetic itself.

*Empirical check ($J=5$, $\delta=0.1$).* The conditional certificate is valid (coverage $0.98$–$1.00\ge0.90$) and **tighter than the mass-ratio baseline** (median simplex $\bar U^{\,r}\approx 0.37$ vs. $0.45$). The bounded-$\Lambda$ box version (Thm 1′) tightens further to $\bar U^{\,r,a}\approx 0.13$ while staying valid for in-box $\lambda^\star$, confirming both validity and the worst-client domination that motivates restricting $\Lambda$.

### 4.3 Theorem 2 — Federated feasibility (per-client accepted-sample threshold)

In the zero-accepted-error regime $K_j=0$ with $\Lambda=\Delta^{J-1}$, the deploy condition $\max_j\bar r_j\le\alpha$ reduces, for each binding client, to $U^+(0,A_j;\delta/J)\le\alpha$, i.e. $1-(\delta/J)^{1/A_j}\le\alpha$.

**Theorem 2.** *Certification at level $\alpha$ over the simplex requires, for every client with non-negligible deployment mass, an **observed accepted count***
$$
A_j\ \ge\ \frac{\ln(J/\delta)}{-\ln(1-\alpha)}\ =\ \Omega\!\Big(\tfrac{\ln(J/\delta)}{\alpha}\Big).
$$
*The expected-count form $n_j a_j\gtrsim \ln(J/\delta)/\alpha$ follows as a corollary.*

We state the bound on the **observed $A_j$** — exactly what the certificate consumes — rather than on $n_j a_j$; the expected-count version is the corollary. This is the federated analog of the centralized $N_{\min}=\lceil\log\delta/\log(1-\alpha)\rceil$: the condition must hold *per client*, with a $\log J$ federation penalty. It predicts **certified-coverage collapse** when a client is simultaneously small and high-risk; restricting $\Lambda$ to a box relaxes the worst-client domination. Plainly, this is the heart of what we later call the *feasibility law*: to prove the error rate is below $\alpha$ you need enough accepted-and-audited examples per client. Theorem 2 is the zero-accepted-error floor; its finite-sample *width* extension — the normal/Bernstein approximation to the Clopper–Pearson interval, for which the required count scales like $(\alpha-\hat r)^{-2}$ — adds that if the model's own error rate $\hat r$ already sits close to the target $\alpha$, the number of audit examples needed explodes. Certification is therefore possible only when the model is comfortably below the target and the audit set is large enough; otherwise the honest answer is "cannot certify."

*Empirical check.* The predicted $\ln(J/\delta)/(-\ln(1-\alpha))\approx 78$ accepted/client ($J{=}5,\delta{=}0.1,\alpha{=}0.05$) is consistent with the simulated crossover (median $\bar U$ below $\alpha$ near $\approx 100$ accepted/client). Real CIFAR confirms the mechanism is binding: shrinking the risk buffer $\gamma$ to lower realized risk *also* starves the accepted set below the floor (`cert_n` $500\to151$, $\approx30<37$/client at $d{=}5$), which *raises* the UCB ($0.185\to0.222$) — so feasibility, not operating-point tuning, controls certifiability (Section 5.1).

### 4.4 Privacy and communication (corrected)

The privacy footprint depends on **which** certificate is deployed; the "two secure-aggregated counts only" claim holds **only for the pooled certificate**, not the stratified one. Table 1 summarizes each variant's privacy footprint together with its validity and tightness, so the privacy/efficiency trade-off can be read in one place.

**Table 1. Certificate variants — privacy, validity, and tightness.** Median upper bound $\bar U$ is from the synthetic study ($r_{\mathrm{bad}}{=}0.3$; lower is tighter).

| variant | counts needed | secure aggregation | valid under shift? | median $\bar U$ | role |
|---|---|---|---|---|---|
| pooled (Prop. 3) | sums only | yes (sum-only) | no (matched only) | 0.073 | optional tightening |
| bounded-$\Lambda$ (Thm 1′) | per-client + intervals | no | inside $\Lambda$ | 0.158 | recommended deployment |
| stratified simplex (Thm 1) | per-client | no | yes | 0.383 | worst-case guarantee |
| grouped-stratified | per-group | within groups | yes (worst-group) | — | privacy/feasibility compromise |
| mass-ratio (App. B) | per-client | no | yes | 0.473 | loose baseline |

Because Theorem 1 needs **per-client** counts, it is *not* compatible with sum-only secure aggregation — a correction to the original claim. The recommended compromise is a **grouped-stratified certificate**: partition clients into $G$ public strata of $\ge k$ clients each, secure-aggregate counts *within* each stratum, and run the certificate over the $G$ groups. This keeps a worst-group guarantee while releasing only $G$ aggregated pairs. Even per-client counts leak far less than the per-client score *distributions* (T-Digest) of federated conformal prediction or the quantized score *functions* of decentralized conformal novelty detection. A differentially private variant adds calibrated noise to the counts and widens the Clopper–Pearson levels to absorb it.

### 4.5 Proposition 3 — Pooled certificate under matched-mixture calibration

When the calibration data are themselves drawn i.i.d. from the deployment mixture $Q_{\lambda^\star}$ (rather than fixed per-client strata of size $n_j$), pooling avoids the worst-client domination. Let $A=\sum_j A_j$, $K=\sum_j K_j$, $U_{\mathrm{pool}}=U^+(K,A;\delta)$.

**Proposition 3 (matched-mixture).** *Under (a) i.i.d. mixture calibration and (b) Lemma L, $\Pr(R_{\mathrm{sel}}(\lambda^\star)\le U_{\mathrm{pool}})\ge1-\delta$ and $U_{\mathrm{pool}}\le\bar U_\Delta^{\,r}$.*

This is a **tightening, not a core result**, and is deliberately stated as a Proposition because two gaps must close before it can be a theorem.

*Gap 1 — Lemma L (resolved).* Does the binomial CP upper limit applied to a Poisson-binomial accepted-error count stay conservative for its mean (cf. Hoeffding's classical comparison of Poisson-binomial and binomial tails [35])? The **naive global** domination $P_{\mathrm{PB}}(S\le b)\le P_{\mathrm{Bin}(A,\bar r)}(S\le b)$ *for all* $b$ is **false** (explicit counterexample), but it holds **at the operative CP threshold** $b=k_\delta\le\mu$ — which is all the certificate uses — so **Lemma L holds**. A self-contained two-coordinate transfer argument establishes the threshold domination, and an adversarial search over 990 configurations gives minimum coverage $0.902/0.952/0.990\ge1-\delta$ (the worst case attained by the binomial itself, as expected). An unconditional **Bernstein fallback** (using $\sum_i r_i(1-r_i)\le A\bar r(1-\bar r)$, so the binomial variance dominates) is provided as a slightly looser but assumption-free alternative. Details and the adversarial certificate are provided in the supplementary material.

*Gap 2 — roster-composition coupling.* Under fixed-stratum (non-i.i.d.-mixture) sampling, the random client composition of the accepted roster makes the pooled mean $\bar r_A=\tfrac1A\sum_{\text{accepted }i}r_{c(i)}$ differ from the target $R_{\mathrm{sel}}(\lambda^\star)$. Closing this needs either assumption (a), a stratified finite-population correction, or an explicit conditioning on the roster. The partial exchangeability of Lu et al. [6] operates on score ranks/quantiles, not on this accepted-risk ratio, so it does **not** transfer for free.

**Theorems 1 and 1′ stand without either gap.** Proposition 3 is the optional matched-mixture tightening and will not be promoted to a theorem until the remaining roster-composition gap (Gap 2) is formally closed; Gap 1 (Lemma L) is already resolved.

### 4.6 Score-agnostic guarantee and the deformation phenomenon (N3)

Because the guarantee in Theorems 1, 1′ (and Proposition 3) is produced entirely by the certification split — the conditional-binomial structure of $K_j\mid A_j$ under a *fixed* selector — it holds for **any** score $s(\cdot)$ used to define $A$. Score quality affects *how much coverage* is certified (a better score accepts more at the same risk), never *whether the risk is controlled*. This matters because federated heterogeneity, like label corruption, **deforms the confidence–correctness ranking**: at clients holding only minority known classes, those classes are easily confused with genuine unknowns, so any single global score is miscalibrated somewhere. Fed-CORE does not attempt to repair this deformation with the small trusted set; it *certifies around it*. Demonstrating that certification beats repair at equal trusted-data budget is a central empirical claim (Section 5).

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

*Empirical verification (synthetic smoke test and real CIFAR).* The contract is checked directly: with the $\delta/T$ split the **simultaneous** unsafe rate is $0.086\le\delta$, whereas using $\delta$ per round (no split) inflates it to $0.386>\delta$ — confirming the split is necessary, not cosmetic. On real CIFAR self-training, **naive** (uncertified) self-training injects pseudo-label contamination of $0.59$–$0.98\gg\alpha$ (the certificate correctly recommends rejection with $U=1.0$, which the naive rule ignores), while **certified** self-training keeps contamination $\approx 0$ (rejecting / halting on Theorem-2-infeasible rounds); clean-data accuracy is comparable across certified, naive, and none. The safety contract thus holds on real data. We stress that the claimed benefit of this use case is **contamination control, not an accuracy gain**: with the present small backbone there is little safe-acceptance headroom, so accuracy is not the point (see Section 5).

---

## 5. Experiments

### 5.1 Experimental setup

We train federated models with FedAvg [36] under non-IID Dirichlet partitions ($d\in\{0.1,0.5,5\}$; smaller $d$ is more heterogeneous), hold out classes as test-time unknowns (the standard FedOSR open-set split), and optionally corrupt client labels (symmetric/asymmetric) to connect to the corrupted-training setting. Data sets are CIFAR-10/100 (primary), TinyImageNet, and a tabular FL benchmark (covtype). Four post-hoc scores — maximum softmax probability (MSP) [31], entropy, margin, and energy [32] — test the score-agnostic claim. The headline metric is CertifiedCoverage@$\alpha$: the accepted coverage among runs that certify $\bar U_\Lambda\le\alpha$. FedOSR detectors (FedPD/FedPD++ [1,2], FedOSS [30], FOOGD [3]) are treated as **base models**, not competitors — they carry no certificate, and Fed-CORE certifies their scores post-hoc. As the first method to certify this object, we compare against re-cast nearest methods (federated CP [6], decentralized novelty FDR [29]) that are invalid or control a different functional, against our own variants as ablations, and against centralized oracles [25,26] as upper bounds.

### 5.2 Validity and non-reducibility

We first establish that the certificate is valid and that the obvious alternative — pooling — is not.

*Controlled synthetic study.* On synthetic clients whose ground-truth risks and deployment mixtures are varied independently, every property holds: empirical coverage $\ge0.98$ across heterogeneity; the tightness order box $<$ simplex $<$ mass-ratio (all valid; the pooled certificate is tightest but invalid off the matched mixture); a monotone CertifiedCoverage@$\alpha$ frontier; the heterogeneity-collapse curve crossing the Theorem-2 floor ($\approx37$ accepted/client); and all four scores valid (test_risk $\approx0.044\le\alpha$).

*Pooling is anti-conservative (non-reducibility).* The natural shortcut — pool every client's accepted audit points and apply one Clopper–Pearson bound — fails under heterogeneity.

![Figure 2](experiments/fedcore/figs/fig1_pooling_collapse.png)

**Figure 2. Why federated calibration cannot be pooled.** Empirical coverage of the certificate as the deployment mixture shifts from the calibration-matched $\lambda$ toward one high-risk client (four low-risk clients $a{=}0.7,r{=}0.02$ plus one high-risk $a{=}0.5,r{=}0.3$; $\delta{=}0.1$). Naive pooled CP is valid only at the matched mixture and collapses to $0$ under shift (it certifies $\approx0.072$ while the true risk reaches $0.165$–$0.30$); the stratified conditional certificate (Theorem 1) stays $\ge1-\delta$ for every mixture, and the box-$\Lambda$ certificate (Theorem 1′) is by design valid only for mixtures inside its assumed box — its drop outside the box is expected and does not contradict its guarantee. The pooled accepted-error count is Poisson-binomial, not binomial, so one Clopper–Pearson bound is anti-conservative.

*Uncertified rules are unsafe (necessity).* A practitioner without the certificate has two options, both unsafe. Table 2 reports the unsafe-deployment rate $\Pr(\text{deploy}\mid R_{\mathrm{sel}}>\alpha)$, which any valid method must keep $\le\delta=0.1$.

**Table 2. Necessity — unsafe-deployment rate (must be $\le\delta=0.1$).** (Naive pooling is omitted here because it controls coverage, not the accepted-risk metric; its collapse is shown in Figure 2.)

| rule | uses the split correctly? | stratified? | unsafe-deploy rate | valid? |
|---|---|---|---|---|
| naive empirical threshold (deploy iff $\hat r\le\alpha$) | yes | no | 0.49 | no |
| leaked split (threshold chosen on the certification fold) | no | yes | 0.18 | no |
| **Fed-CORE (proper split)** | yes | yes | **0.00–0.03** | yes |

The naive threshold ignores finite-sample noise and deploys unsafely about half the time at the boundary. The proposal/certification split is load-bearing: re-searching the threshold on the certification fold inflates the deploy rate to $99.8\%$ but the unsafe rate to $18.2\%\gg\delta$, because the certificate cannot correct the multiple-testing of thresholds chosen on the same fold. Only the proper split with the conditional certificate keeps the unsafe rate $\le\delta$. (Federated-CP rules control a quantile, not the accepted risk, so they do not appear here; naive pooled CP is anti-conservative under shift, Figure 2.)

### 5.3 The finite-sample feasibility law

*The apparent null, and its cause.* On a real CIFAR-10 ladder (12 runs, $\alpha=\delta=0.1$, 5 clients, small CNN), CertifiedCoverage@$0.1$ is $0$ in every run — reported honestly. Two distinct modes explain this. **Mode 1** (extreme non-IID, $d{=}0.1$): the empirical accepted risk already exceeds $\alpha$, so no method can deploy safely and the certificate correctly declines. **Mode 2** (near-IID, $d{=}5$, test_risk $\approx0.08<\alpha$): the model is safe, but the thin per-client accepted counts cannot drive the upper bound below $\alpha$ — the Theorem-2 feasibility collapse, not certificate looseness. Tellingly, shrinking the risk buffer $\gamma$ to lower realized risk *also* starves the accepted set (cert_n $500\to151$, $\approx30<37$/client), which *raises* the bound ($0.185\to0.222$): the binding lever is calibration budget, not the operating point.

*The staircase.* Re-aggregating the $J{=}5$ clients into $G$ public groups (the grouped-stratified certificate, Section 4.4) raises the per-group accepted count and drives the bound monotonically through $\alpha$.

![Figure 3](experiments/fedcore/figs/F6_feasibility_law.png)

**Figure 3. The feasibility law (Theorem 2): grouping and audit budget are the same sample-size lever.** (a) Worst-group certified-risk upper bound versus per-group accepted count (log axis), as the clients merge into $G\in\{5,3,2,1\}$ groups (ResNet 5-seed band): the bound falls as the count grows and crosses $\alpha{=}0.10$ around several-hundred accepted points, with the Theorem-2 floor ($\approx37$/client) marking where certification first becomes possible. (b) The corresponding CertifiedCoverage@$0.10$ rises from $0$ to $\approx0.21$. (c) Calibration-budget sweep: as the per-client audit count grows, the mean certified-risk bound falls (left axis) and the probability of certifying rises (right axis); at the largest budget the seed-mean bound sits just above $\alpha$ while $2/5$ seeds individually certify — the usual gap between a mean and a pass rate. Panels (a) and (c) make the same point from two directions — *merging clients into groups* and *enlarging the audit budget* are the same Theorem-2 sample-size lever, governed by the $(\alpha-\hat r)^{-2}$ requirement, not by the operating point. (The shaded band in (a)–(b) is $\pm1$ std over five seeds; the wide band near $\alpha{=}0.10$ is the seed-variability we report honestly.)

![Figure 4](experiments/fedcore/figs/F5_alpha_frontier.png)

**Figure 4. Risk–coverage frontier on real CIFAR-10 (per-client $G{=}5$, box-$\Lambda$; $d{=}5$).** Under the most conservative per-client grouping, certification becomes non-vacuous only at larger risk targets (SimpleCNN $0.063$ at $\alpha{=}0.20$, $0.193$ at $\alpha{=}0.25$; ResNet-18 $0.316$ at $\alpha{=}0.25$). The frontier is monotone because the proposal-side proxy enforces a small safety margin; an aggressive operating point is corrected rather than allowed to pass by luck (validity is preserved by the independent certification fold). Merging into worst-group $G{=}2$ raises the achievable coverage at the same $\alpha$ (Section 5.4). The vertical marker at $\alpha{=}0.20$ is the practical operating point; $\alpha{=}0.10$ sits at the feasibility edge.

### 5.4 Real-data certified coverage

*Backbone.* We adopt a CIFAR-stem ResNet-18 with **GroupNorm** as the principled FL normalization (BatchNorm's running statistics diverge under non-IID FedAvg). GroupNorm lowers $\hat r$ but the accepted set shrinks in step, so it does not strengthen the seed-variable $\alpha{=}0.10$ certificate; what matters is robust — no false certificate under either normalization.

*Headline.* At $\alpha{=}0.20$ the GroupNorm certificate is non-vacuous on **all five seeds**: CertifiedCoverage $0.392\pm0.097$ ($d{=}5$) and $0.353\pm0.130$ ($d{=}0.5$), with $0/10$ false certificates (Table 3a). This is a finite-sample feasibility demonstration under the current audit budget, not a safety target. At $\alpha{=}0.10$ the worst-group result is seed-variable ($2$–$3/5$), reported as a secondary result. The edge and negative cells (Table 3b) are exactly where the feasibility law predicts collapse — extreme non-IID, corruption, or a backbone whose $\hat r$ is near $\alpha$.

**Table 3a. Real-data certified coverage — clean-data main results** (CIFAR-10, worst-group $G{=}2$ CertifiedCoverage@$\alpha$, fixed score MSP, mean$\pm$std over seeds; $n_{\mathrm{cert}}/N$ in parentheses; ResNet-GroupNorm, cert_frac$=0.5$).

| backbone / cell | $\alpha{=}0.10$ (5-seed) | $\alpha{=}0.20$ (5-seed) | false certs |
|---|---|---|---|
| ResNet-GN $d{=}5$ clean | $0.077\pm0.097$ (2/5) | $\mathbf{0.392\pm0.097}$ (5/5) | 0 |
| ResNet-GN $d{=}0.5$ clean | $0.091\pm0.103$ (3/5) | $\mathbf{0.353\pm0.130}$ (5/5) | 0 |
| ResNet-BN $d{=}5$ clean | $0.106\pm0.098$ (3/5) | $\approx0.29$–$0.31$ (single-config) | 0 |

**Table 3b. Edge and negative settings** (where certification is correctly vacuous, with the feasibility-law reason).

| setting | CertCov@$\alpha$ | reason it does not certify |
|---|---|---|
| ResNet $d{=}0.1$ clean | $0$ | extreme non-IID; accepted set too small (Theorem-2 infeasible) |
| ResNet sym-0.35 | $0$ | corruption raises $\hat r>\alpha$ (e.g. $0.167$ at $d{=}0.5$) |
| SimpleCNN | $0.063$ at $d{=}5$ ($\alpha{=}0.20$) | looser backbone; higher $\hat r$ |
| covtype (tabular FL) | $0$ at fixed MSP; $0.10\pm0.17$ (2/5) selection-optimistic | linear backbone $\hat r\approx0.14$–$0.24$ near $\alpha$ |

covtype is reported as a second domain that exhibits the same feasibility law at its edge, **not** as a stable positive: at fixed MSP it certifies $0/5$, reaching a non-vacuous point only under a selection-optimistic best-of-scores rule.

*Superiority.* Table 4 compares Fed-CORE against the two uncertified alternatives at the practical operating point.

**Table 4. Superiority — matched-risk comparison at $d{=}5$, $\alpha{=}0.20$** (post-hoc on real logits). Fed-CORE is the only method that is simultaneously safe and finite-sample-guaranteed.

| method | uses test labels? | accepted coverage | realized risk $\le\alpha$? | finite-sample guarantee? |
|---|---|---|---|---|
| test-peeking oracle (MSP / energy) | yes | $0.444$ / $0.431$ | yes (by construction) | ✗ |
| no-peek naive threshold | no | high near-IID | sometimes no (breaches off-IID) | ✗ |
| **Fed-CORE** (worst-group $G{=}2$) | no | $0.392\pm0.097$ (5-seed) | yes | ✓ |

The test-peeking oracle reaches higher coverage but uses test labels and carries no guarantee; the no-peek naive threshold breaches $\alpha$ wherever proposal and test diverge. Fed-CORE retains a large fraction of the oracle's coverage while being the only safe, guaranteed option.

*Stress axes.* Beyond the risk target $\alpha$ (Figure 4), the feasibility law has two further axes — heterogeneity and corruption — shown together in Figure 5; each pushes $\hat r$ past $\alpha$ or starves the per-group count below the Theorem-2 floor.

![Figure 5](experiments/fedcore/figs/F7_hetero_collapse.png)

**Figure 5. Stress axes of the feasibility law: heterogeneity and corruption.** **(a) Heterogeneity (SimpleCNN stress configuration).** As the Dirichlet concentration $d$ falls (more non-IID), per-group accepted counts thin and the worst-group certificate degrades, collapsing at the extreme ($d{=}0.1$); the looser SimpleCNN backbone is used only to isolate the mechanism, with the ResNet-GroupNorm headline in Table 3a/3b. **(b) Corruption.** Worst-group CertifiedCoverage versus the client-side training-label noise rate, at $d\in\{0.5,5\}$: non-vacuous on clean data ($0.31$ at $d{=}5$, $0.13$ at $d{=}0.5$) but collapsing to $0$ once the noise rate exceeds $\approx0.1$, although the trusted calibration fold stays clean — corruption raises the model's $\hat r$ above $\alpha$, so a clean audit set has nothing safe left to certify. (Panel (b) uses single fixed configurations and is not directly comparable to the five-seed clean-data headline in Table 3a.)

### 5.5 Score and base-model dependence

*Score-agnosticism.* The tightness ordering of the certificate variants (conditional simplex tighter than the mass-ratio baseline, bounded-$\Lambda$ tighter still) is reported together with their privacy footprint in Table 1. The guarantee holds for **every** score: across MSP, entropy, margin, and energy the realized test_risk stays $\le\alpha$ (0.042–0.046) while certified coverage varies (0.64–0.66; Appendix C, Table C1). The score changes only *how much* coverage is certified, confirming that validity comes from the certification split, not from score quality (Section 4.6) — a point made concrete on real detectors next.

*Two real FedOSR detectors.* To answer the concern that the above uses MSP on a FedAvg backbone rather than a genuine FedOSR detector, we certify the native open-set scores of two real FedOSR methods: FedPD's PROSER dummy-vs-known score (a full reproduction) and FOOGD's SM3D score (a representative head on the shared backbone). Table 5 reports both, alongside the FedAvg+MSP control and a faithfully reproduced full FOOGD-SAG.

**Table 5. Certification on real FedOSR base models** (native open-set scores of FedPD-PROSER and FOOGD, with the FedAvg+MSP control and a full FOOGD-SAG attempt; worst-group $G{=}2$, three seeds; no false certificate in any certified cell). $\ddagger$ FOOGD-SAG is a single-seed honest negative.

| base model (kind) | $d$ | AUROC | CertCov@$0.20$ (3 seeds) | status |
|---|---|---|---|---|
| FedPD–PROSER (full) | $5$ | $0.80$ | $\mathbf{0.483\pm0.100}$ (3/3) | strongest; certified — also $0.174\pm0.125$ (2/3) at $\alpha{=}0.10$ |
| FedPD–PROSER (full) | $0.5$ | $0.79$ | $0.455\pm0.090$ (3/3) | robust under heterogeneity — $0.210\pm0.089$ (**3/3**) at $\alpha{=}0.10$ |
| FedAvg+MSP (full, same backbone) | $5$ | $0.73$ | $0.350\pm0.077$ (3/3) | control (harness validation) |
| FOOGD–SM3D (representative) | $5$ | $0.69$ | $0.071\pm0.053$ (3/3) | certified, multi-seed |
| FOOGD–SM3D–SAG (full) | $5$ | $0.47$ | $0$ (0/1)$^{\ddagger}$ | faithfully reproduced; weak at feasible budget |

*FedPD-PROSER — the strongest base model, and a full method.* Trained with the standard recipe — a closed-set CE pretraining stage (known-class accuracy $0.39\to0.65$ over eight rounds, versus $0.26$ stalled from scratch) followed by PROSER fine-tuning on the exact WideResNet-28-10 architecture — its native PROSER dummy-vs-known score reaches AUROC $0.80$. Over three seeds at $d{=}5$, Fed-CORE certifies it with worst-group CertifiedCoverage $0.483\pm0.100$ ($3/3$) at $\alpha{=}0.20$ and $0.174\pm0.125$ ($2/3$) at $\alpha{=}0.10$ (mean test_risk $0.112$ / $0.029\le\alpha$) — the strongest base-model row, and the only detector that certifies at the hard target $\alpha{=}0.10$, where MSP and FOOGD cannot. This is a genuine **full** FedOSR method (not a representative head), and it shows that Fed-CORE is a certification layer for real FedOSR detectors, not only for the MSP baseline. The result holds under stronger heterogeneity: at the more non-IID $d{=}0.5$, FedPD-PROSER certifies $3/3$ at *both* targets — $0.455\pm0.090$ at $\alpha{=}0.20$ and $0.210\pm0.089$ at $\alpha{=}0.10$ (AUROC $0.79$, mean test_risk $0.105$ / $0.036\le\alpha$) — so the only detector that certifies at the hard target $\alpha{=}0.10$ does so even as client heterogeneity increases. The paper's headline remains the five-seed MSP result of Table 3a.

*The thesis across base models.* Certified coverage tracks the native-score AUROC — FedPD $0.80$ > MSP $0.73$ > FOOGD-representative $0.69$ > FOOGD-SAG $0.47$ — while **validity holds in every cell** ($0/18$ false certificates across all seeds and base models): exactly the score-agnostic guarantee, with coverage following detector quality and validity independent of it. The FedAvg+MSP control reproduces the Section 5.4 headline ($\approx0.35$ at $\alpha{=}0.20$), validating the harness.

*A reproducibility note.* The full pipelines are configuration-sensitive. Full FedPD-PROSER succeeds with the closed-set-pretrain-then-fine-tune recipe above (from scratch it does not converge, since PROSER is designed to fine-tune a closed-set-pretrained model). In contrast, **full FOOGD-SAG** (WideResNet-40-2 with annealed denoising score matching and a KSD/MMD Stein regularizer, trained jointly) reaches only AUROC $0.467$ ($\approx$ chance) on our six-known split at a single-GPU budget — plausibly because its pipeline does not standardize the score-model input (a decisive fix in the representative protocol), trains the backbone jointly, and targets covariate-shift OOD over long sweeps rather than the semantic-shift split here; we report it as a single-seed honest negative. **FedOSS** (a medical-imaging codebase without a CIFAR loader) is deferred as the highest-cost option. Reproducing a SOTA FedOSR method at published strength on a new split is itself expensive and configuration-sensitive; Fed-CORE's validity is unaffected throughout, and certified coverage simply follows the base model's score quality.

### 5.6 Downstream use: certified pseudo-label admission

The certificate's first downstream use, safe automation, is CertifiedCoverage@$\alpha$ itself (Section 4.7). The second use is **first a certified admission gate, not primarily an accuracy booster**: accepted predictions are folded back into FedAvg as pseudo-labels only when their contamination can be certified below the target — though, as a supporting result below shows, a sufficiently strong base detector with an adequate audit budget can additionally turn safe admission into a safe accuracy gain.

![Figure 6](experiments/fedcore/figs/F8_selftraining.png)

**Figure 6. Certified pseudo-label admission prevents unsafe self-training (Proposition 4, ResNet-GN $d{=}5$).** (a) Realized pseudo-label contamination per round: naive self-training keeps injecting pseudo-labels even when their error rate is far above the target $\alpha$ ($0.19$–$0.67$ and growing), whereas Fed-CORE injects none (so it has no contamination line; $\alpha{=}0.1$ dashed). (b) Admission/halt behavior: naive admits every batch (each point annotated with the batch size $n$), whereas Fed-CORE finds the first round Theorem-2-infeasible and **halts**, admitting nothing rather than an uncertified batch. (c) Round-wise validity (inset): the simultaneous unsafe rate is $0.086\le\delta$ with the $\delta/T$ split versus $0.386>\delta$ without. The message is contamination control via a certified admission gate, not an accuracy gain.

This particular round-wise run should **not** be read as a general accuracy improvement. Rather, it shows that Fed-CORE acts as a *certified admission gate*: it prevents the model from consuming pseudo-label batches whose contamination cannot be certified below the target risk. In this run the certified procedure found the very first round Theorem-2-infeasible — the accepted set was too thin to certify below $\alpha$ — and halted, admitting nothing, whereas naive self-training admitted contaminated batches throughout. This is precisely the safe outcome predicted by the feasibility law: Fed-CORE never injects a contaminated batch, even when that means admitting none. Over the rounds the Proposition-4 contract ($\delta/T$) is verified (Table 6): round-wise certification is necessary — reusing one audit fold across adaptive rounds breaks the simultaneous guarantee.

**Table 6. Round-wise certification is necessary for valid self-training.**

| scheme | fresh audit fold per round? | per-round level | simultaneous unsafe rate | valid at $\delta{=}0.10$? | interpretation |
|---|---|---|---|---|---|
| Fed-CORE (round-wise split) | yes | $\delta/T$ | 0.086 | ✓ | every admitted batch satisfies the simultaneous contamination certificate |
| reused / no split | no | $\delta$ per round | 0.386 | ✗ | reusing the same audit evidence across adaptive rounds breaks the finite-sample guarantee |

*A supporting result: safe admission can also yield an accuracy gain on a strong detector.* The gate's purpose is safety, but when the base detector is strong and the audit budget large, the admitted pseudo-labels can additionally raise accuracy without ever breaching the contamination target. Replacing the FedAvg+MSP base with the stronger FedPD-PROSER detector (Section 5.5) and using a single one-shot certification at level $\delta$ (rather than the round-wise $\delta/T$ split), the certified procedure admits a thicker low-contamination pseudo-label set and yields a certified known-accuracy gain of $+0.030$ (sample SD $0.027$; $2/3$ seeds positive, the third feasibility-limited to zero admissions; at $n{=}3$ the gain is positive but not statistically separable from zero, 95% $t$-CI $[-0.04,+0.10]$) at $\alpha{=}0.20$ and audit budget $4\times$, against a clean-pseudo-label oracle upper bound of $+0.045$ (supplementary Figure 9) — the certified gate captures roughly two-thirds of the achievable headroom while every admitted batch stays certified ($0/3$ contamination violations, maximum realized contamination $0.137\le\alpha$). The gain is **conditional and seed-variable**, governed by the same two levers as the certificate itself. First, detector strength: at the identical budget the weaker FedAvg+MSP base admits nothing under certification (certified $\Delta{=}0$, oracle headroom only $+0.023$), so the realized gain rises from $+0.012$ to $+0.030$ as the detector strengthens — a stronger detector produces more low-contamination pseudo-labels to admit. Second, audit budget: at $1\times$ and $2\times$ the certified procedure admits nothing and halts, and only at $4\times$ does it admit — exactly the Theorem-2 feasibility law applied to self-training. A necessary validity condition is that the unlabeled pool's unknown rate be matched to the deployment/certification rate (the A5 condition of Section 6); without this matching the certified procedure becomes anti-conservative and admits batches with contamination $0.285{>}\alpha$. We therefore report the safe admission gate as the primary result and this accuracy gain as a supporting one: certified self-training is first a contamination gate, and — under a strong detector with adequate feasibility budget — additionally a safe source of accuracy.

Pushing the label-scarcity lever further confirms that feasibility, not headroom, is what bounds the gain. In a deeper label-scarce regime ($10\%$ labels) the clean-pseudo-label headroom grows sharply — the oracle gain rises to $+0.16\pm0.05$ ($n{=}3$, sample SD; clearly above zero) — yet certified self-training there becomes **feasibility-limited**, admitting on only $1/3$ seeds, so the realized certified gain is $+0.04\pm0.07$ ($95\%$ $t$-CI includes zero) even though the safe contamination gate is untouched ($0/N$ violations across every step, all admitted batches $\le\alpha$). Admission is not even monotone in detector strength at this budget — the strongest-base seed still admits nothing — which is feasibility noise (Theorem 2), not a detector deficiency. The robust supporting accuracy gain therefore remains the half-label cell above; deeper label scarcity enlarges the prize but tightens the Theorem-2 feasibility constraint that governs whether it can be claimed, so the binding constraint on the certified gain is the trusted audit budget, not the detector. (We verified the audit-budget rescue only on a seed that already admitted; whether a larger budget rescues the zero-admission seeds is left open.)

*Method-knob ablations.* Three controlled ablations probe the certificate's knobs — calibration budget, unknown-class proportion, and client count. The calibration-budget lever is panel (c) of Figure 3, the unknown-proportion condition is Figure 7 (Section 6), and the client-count ablation is in Appendix C. On real CIFAR-10 logits the audit-budget sweep drives the worst-group bound from $0.58$ to $0.18$ as the calibration fold grows ($\alpha{=}0.10$ becoming non-vacuous, $2/5$, only at the largest budget), and the unknown-proportion sweep reproduces the audit-representativeness collapse (coverage $1.00$ at $\rho{=}1$ down to $0.005$ as unknowns are under-represented), confirming both the feasibility law and the audit-representativeness requirement on real federated logits.

## 6. Limitations

The stratified certificate is conservative (Clopper–Pearson exactness + union bound + worst-case mixture); box-$\Lambda$ (Thm 1′) and pooled (Prop. 3) recover tightness but require, respectively, knowledge of $\Lambda$ and the two gaps of Section 4.5. The guarantee is marginal over $Q_\lambda$, not conditional per client.

**Trusted calibration assumption (stated openly).** Certifying *unknown rejection* requires that the certification fold contain **labeled unknown-class points** ($Y=\textsf{unknown}$): unknown classes are unseen during training but must be present and labeled in a small post-training **audit/calibration fold**. Consequently "distribution-free" means *with respect to the calibration distribution $Q_\lambda$* — not with respect to the entire unknown universe. If the audit fold contains no unknowns, the guarantee is a closed-set selective-risk guarantee, not an OSR one. In OSR benchmarks this is satisfied by construction (held-out classes excluded from training, available as unknown-labeled examples for calibration/test); in deployment it corresponds to a small audited monitoring set. Theorem 2 quantifies how scarce such audit data may be before certification becomes infeasible. **A sharper requirement (ablation A5, Figure 7):** the audit fold must carry unknowns at $\ge$ the deployment rate — under-representing unknowns makes the certificate *anti-conservative* (coverage falls below $1-\delta$). In practice this means the monitoring set should track, not under-sample, the unknown-class incidence of the live stream. We promote this from a supplementary check to a main figure because it is a **condition of the guarantee**, not a peripheral ablation.

![Figure 7](experiments/fedcore/figs/FA5_unknown_proportion.png)

**Figure 7 (A5). Audit representativeness is a condition of open-set risk certification.** Holding the deployment unknown-among-accepted fraction at $0.06$, we vary the *calibration* unknown fraction. Coverage of the true deployment risk is $\ge1-\delta$ only when the calibration fraction **matches or exceeds** the deployment one ($0.913$ at $0.06$, $0.992$ at $0.08$); **under-representation is anti-conservative** ($0.057$ at $0.02$, $0.522$ at $0.04$). The real-data counterpart on CIFAR-10 logits (Section 5.6) reproduces this collapse ($1.00$ at $\rho{=}1$ down to $0.005$ as unknowns are under-represented). It is therefore not enough that the audit fold *contains* labeled unknowns — it must carry them at $\ge$ the deployment rate.

**Privacy.** As corrected in Section 4.4, only the pooled certificate is sum-only; the stratified certificate needs per-client (or per-group) counts. The privacy claim is an information-flow statement, not a differential-privacy guarantee unless the noised variant is used.

**Self-training use case (B).** What the certificate guarantees is the *contamination* of each injected pseudo-label batch ($\le\alpha$ per round, simultaneously over $T$ rounds by Prop. 4) — **not** that self-training necessarily improves accuracy; the accuracy gain is an empirical claim (bounded-noise training is well-behaved but not monotone-guaranteed). Empirically the gain is real but conditional and seed-variable (Section 5.6: $+0.030$ over three seeds, $2/3$ positive, on a strong detector at $4\times$ audit budget, with one seed feasibility-limited to zero admissions); we therefore frame the contamination gate, not the accuracy gain, as the guarantee. Round-wise audit-fold splitting trades feasibility for adaptivity: $T$ is capped by the trusted-set size via the Theorem 2 per-fold threshold. A reused-fold scheme with formal closed-loop validity (avoiding the $\delta/T$ split) is left as future work.

---

## 7. Conclusion

Fed-CORE certifies the accepted selective risk of a federated open-set classifier with a finite-sample, distribution-free guarantee that holds under heterogeneity and unknown deployment mixtures. Its core is a **stratified conditional selective-risk certificate** ($\max_j$ of per-client conditional-binomial CP limits, with a robust bounded-$\Lambda$ form) that is valid where naive pooling fails, accompanied by a per-client feasibility threshold and an optional matched-mixture pooled tightening. The framework recasts federated open-set recognition from a ranking problem into a *certification* problem: a small trusted audit set is used not to repair the model but to certify which of its predictions are safe to accept — and those certified-safe predictions are then *used*, for guaranteed-risk automation and for certified federated self-training that expands training with provably-bounded contamination.

Empirically, across synthetic clients and real federated benchmarks (CIFAR-10/100 and a tabular FL task), the certificate is **valid in all tested settings under the stated audit-representativeness condition — no false certificate in any seed, data set, or normalization** — and naive pooling collapses under mixture shift exactly as the theory predicts. Certified coverage is governed by a **feasibility law**: a per-group sample threshold scaling as $(\alpha-\hat r)^{-2}$, traced out by a monotone grouped-stratified staircase and by the heterogeneity and corruption axes. The positive is a 5-seed worst-group $\alpha{=}0.20$ certified coverage on CIFAR-10 ResNet-GroupNorm ($0.392\pm0.097$ at $d{=}5$ and $0.353\pm0.130$ at $d{=}0.5$, both $5/5$, with $0/10$ false certificates); a second federated domain (tabular FL) exhibits the same feasibility law at its edge but is seed-variable and non-vacuous only under selection-optimistic scoring ($0/5$ at fixed MSP), so it corroborates the law rather than adding a second stable positive. The $\alpha{=}0.10$ worst-group result is achievable but seed-variable ($2$–$3/5$) and reported as such. The contribution is therefore the *object and its finite-sample certificate*, the exposure of pooled invalidity under heterogeneity, and the characterization of *when* certified open-set deployment is feasible — not a new FedOSR algorithm or a raw-accuracy gain.

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

Two of the three synthetic method-knob ablations are now main figures — the calibration-budget sweep is panel (c) of Figure 3, and the unknown-proportion sweep is Figure 7 — because both turned out to be central (a feasibility lever and a validity condition). The remaining supplementary ablation, client scaling, is kept here.

![Figure 8](experiments/fedcore/figs/FJ_client_scaling.png)

**Figure 8 (J). Per-client starvation / $\log J$ penalty.** With the *total* calibration budget fixed and split across $J$ clients, the worst-group cert_ucb rises ($0.064\to0.276$ as $J:2\to20$), crossing $\alpha{=}0.10$ around $J{=}5$ — the per-client feasibility threshold of Theorem 2 in action: more clients means thinner per-client folds and a looser worst-group bound. Client-participation subsampling behaves identically (it lowers per-client counts).

*Certified self-training gain (supporting Section 5.6).* The accuracy-gain result discussed in Section 5.6 is summarized in Figure 9. We keep it supplementary because, while safe, the gain is seed-variable and at $n{=}3$ not statistically separable from zero; the primary self-training result remains the contamination gate of Figure 6.

![Figure 9](experiments/fedcore/figs/F_selftrain_gain.png)

**Figure 9 (supplementary). Certified one-shot self-training: a safe accuracy gain on a strong base detector** ($\alpha{=}0.20$, audit budget $4\times$, one-shot certification). **(a)** Certified known-accuracy gain over the no-pseudo-label baseline, per base model: the certified (Fed-CORE) bar and the clean-pseudo-label oracle upper bound, with $\pm1$ SD and per-seed points. The weaker FedAvg+MSP base ($n{=}1$) gains $+0.012$ (oracle $+0.043$); the stronger FedPD-PROSER base ($n{=}3$) gains $+0.030$ (oracle $+0.045$), with $2/3$ seeds positive and the third feasibility-limited to zero admissions — a stronger detector yields a larger safe gain. The statistical claim is the $t$-interval reported in Section 5.6 ($95\%$ CI $[-0.04,+0.10]$, positive but not separable from zero at $n{=}3$); the error bars are illustrative. **(b)** Safety: the realized contamination of every admitted batch stays $\le\alpha{=}0.20$ (FedAvg certified $0.136$, FedPD certified $0.124$), whereas naive self-training sits at the contamination budget ($0.200$). Across all runs there are $0/N$ certified contamination violations.

## Appendix D. Full proofs

Throughout, the selector $A$ is fixed on the proposal fold and is therefore independent of the certification fold; within each client the certification draws are i.i.d. from $P_j$. We write $U^+(\cdot,\cdot;\varepsilon)$ and $L^-(\cdot,\cdot;\varepsilon)$ for the one-sided Clopper–Pearson limits of Section 4.1, which satisfy, for $K\sim\mathrm{Bin}(n,p)$ and every $p$, $\Pr(p\le U^+(K,n;\varepsilon))\ge1-\varepsilon$ and $\Pr(p\ge L^-(K,n;\varepsilon))\ge1-\varepsilon$.

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

