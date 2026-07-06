# Federated Open-Set Recognition: Meta-Analysis and Novelty-Validation Brief

**Prepared for:** Sanghoon Kim
**Date:** 2026-06-26
**Purpose:** Decide whether a journal-targeted paper on open-set recognition (OSR) in federated learning (FL) is defensibly novel, and if so, lock the angle.
**Deliverable type:** Meta-analysis + adversarial novelty verification (no full draft yet).

---

## 한국어 요약 (진단 / 판정 / 다음 행동)

**진단 요약**

- FL+OSR은 이미 형성된 영역이다. FedPD(ICCV'23), FedPD++(IJCV'26), FOOGD(NeurIPS'24), FedOV(ICLR'23), FedNovel(2023), Noise-Resistant FedOSR(KSEM'25)이 핵심 좌표를 차지한다. 그러나 이들은 unknown rejection을 **경험적 지표(AUROC/FPR95/accuracy)로만** 평가한다. **finite-sample certificate가 없다.**
- Federated Conformal Prediction(Lu et al. ICML'23; Plassier et al. label-shift; Rob-FCP Byzantine)은 finite-sample 보장을 주지만 **prediction-set의 marginal coverage**일 뿐, open-set도 unknown rejection도 selective risk도 아니다. Lu의 selective-classification demo는 저자 스스로 "heuristic, no guarantee"라고 명시한다.
- 가장 가까운 위협 2건: (1) **Decentralized Conformal Novelty Detection**(arXiv 2605.08263, 2026-05) — decentralized + conformal + novelty + privacy까지 닿았으나, 통제 대상이 **test batch의 FDR**(다중검정)이지 classifier accepted prediction의 selective risk가 아니다. (2) **Conformal Inference for Open-Set Classification**(2510.13037, 2025) — open-set + finite-sample이지만 **centralized**이고 통제 대상이 **coverage**다.
- 즉 "**federated 환경에서 corrupted/heterogeneous하게 학습된 classifier의 accepted selective risk를 finite-sample certify**"하는 객체는 **현재 비어 있다.** 이것은 Paper 1(SRCC)의 객체를 FL로 이식하되, i.i.d. CP가 깨지는 partial exchangeability + secure aggregation이라는 **새 이론 문제**를 만든다.

**판정**

- **moderate-to-strong go.** Gap은 실재하며 방어 가능하다. 단, "SRCC(Paper 1) + Federated CP(Lu) 단순 결합 아니냐"는 reviewer 공격이 가장 큰 위험이다. 이를 무력화하려면 **단순 결합으로 환원되지 않는 진짜 이론 기여**(아래 N1)가 반드시 main theorem이어야 한다.

**다음 행동**

1. Main theorem을 **partial exchangeability 하에서의 risk(이항/Clopper-Pearson) UCB**로 잡는다 — Lu의 quantile-coverage가 아니라 **weighted/stratified accepted-error count의 finite-sample UCB**. 이것이 novelty의 심장이다.
2. DCND(2605.08263)를 related work에서 **객체 차이(batch FDR vs accepted selective risk)**로 명시 차별화.
3. Baseline은 FOOGD, FedPD(++), FedOV + (certification 측) Federated CP, Rob-FCP를 OSR risk-control로 각색해 비교.
4. 실험은 Paper 1의 CIFAR 파이프라인을 FL(Dirichlet non-IID)로 확장 — 재사용 가능.

---

## 1. Scope and Method of This Meta-Analysis

I surveyed three adjacent literatures that the proposed paper must dominate or differentiate from:

1. **Federated Open-Set / Novel-Class / OOD Recognition** — methods that, in the FL setting, classify known classes while rejecting or discovering unknowns.
2. **Federated Conformal / Distribution-Free Uncertainty** — methods that give finite-sample statistical guarantees in FL, but for *closed-set* prediction sets.
3. **Conformal Open-Set / Novelty Detection with Guarantees** — methods that give finite-sample guarantees for unknown rejection, but in the *centralized* setting.

The thesis under test is: **no existing work provides a finite-sample, distribution-free certificate on the accepted selective risk of an open-set classifier in the federated, heterogeneous (and optionally label-corrupted) setting.** Each source below was checked for the specific guarantee it provides (coverage vs. FDR vs. selective risk) and for whether it is federated. Two pivotal sources (FOOGD, Lu et al.) were read in full to verify their guarantees verbatim; the rest were verified at the abstract/method level. Citation-verification status is in §9.

---

## 2. Landscape Map

The field decomposes into three non-overlapping cells along two axes — **Federated?** and **Guarantee type** (empirical / coverage / risk-or-FDR):

| | Empirical only (AUROC/FPR/acc) | Coverage guarantee | Risk / FDR guarantee on rejection |
|---|---|---|---|
| **Centralized** | classic OSR (OpenMax, etc.) | CGTC open-set conformal (2025) | Conformal novelty/FDR (Bates'23, Marandon'24); reject-option CP (2025); Selective Conformal Risk Control (2025) |
| **Federated / decentralized** | **FedPD, FedPD++, FOOGD, FedOV, FedNovel, Noise-Resistant FedOSR** | Federated CP (Lu'23), label-shift FCP (Plassier'23), Rob-FCP (2024) | **Decentralized Conformal Novelty Detection — FDR only (2026)** ← nearest |
| **↳ TARGET CELL** | | | **Federated certified *selective risk* of accepted open-set predictions — EMPTY** |

The empty cell is the contribution target. The nearest occupant (decentralized novelty detection, 2026) sits in the FDR sub-column, not the selective-risk sub-column — a different statistical object (see §5).

---

## 3. Meta-Analysis Comparison Table (Primary FL+OSR / Federated-Guarantee Papers)

| # | Method | Venue / Year | Task | Federated? | Handles non-IID | Privacy beyond FedAvg | Guarantee on unknown rejection | Eval metric | Key limitation for our purposes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **FedPD** (Yang et al.) | ICCV 2023 | FedOSR (classify known + reject unknown) | Yes | Yes (parameter disentanglement LPD + global divide-conquer aggregation GDCA) | No | **None** (empirical) | AUROC, closed-set acc | No finite-sample certificate; aggregation-centric |
| 2 | **FedPD++** | IJCV 2026 | FedOSR (extended) | Yes | Yes | No | **None** (empirical) | AUROC, acc | Same — still empirical only |
| 3 | **FOOGD** (SM³D + SAG) | NeurIPS 2024 | FL OOD **generalization + detection** | Yes | Yes (Dirichlet α∈{0.1,0.5,5}) | No (FedAvg of score models) | **None** — only Thm 4.1 MMD error bound on *score model*; detection is score-norm threshold | AUROC, FPR95 | Bound is on estimation, **not** on rejection error rate |
| 4 | **FedOV** (open-set voting) | ICLR 2023 (one-shot FL, label skew) | One-shot FL with "unknown" voting | Yes (one-shot) | Yes (label skew) | One-shot (single round) | **None** (empirical) | Accuracy | GAN outliers; no calibration guarantee |
| 5 | **FedNovel / Fed. Continual Novel Class Learning** | 2023 (preprint) | Discover + learn novel classes | Yes | Yes (global alignment GAL, bi-level clustering) | Yes (no raw data) | **None** (empirical) | Novel-class acc | Discovery, not certified rejection |
| 6 | **Fed. Open-world Semi-Sup. (FedoSSL)** (Zhang et al.) | ICML 2023 | Open-world SSL in FL | Yes | Yes (unbiased training) | No | **None** (empirical) | Acc on seen/novel | No risk control |
| 7 | **Noise-Resistant FedOSR (NRFed)** (Gao et al.) | KSEM 2025 | FedOSR **under label noise** | Yes | Partial | Yes (BNN uncertainty) | **None** (empirical) | OSR metrics | Closest to corruption angle, but no certificate |
| 8 | **FOSTER** ("Curse→Blessing") | ICLR 2023 | FL OOD detection via synthesized OOD | Yes | Yes (uses other clients as virtual OOD) | Yes (class-conditional generator) | **None** (empirical) | AUROC | Generative; no finite-sample bound |
| 9 | **Federated CP** (Lu et al.) | ICML 2023 | Distributed UQ — **prediction sets** | Yes | Yes (**partial exchangeability**, Assump. 4.1) | Yes (T-Digest sketch, no raw scores) | **Coverage** only (Thm 4.3: 1−α ≤ cov ≤ 1−α+K/(N+K)); **no open-set, no rejection** (authors: selective-classification demo is "heuristic ... does not have a coverage guarantee") | Set coverage/size | Closed-set; coverage ≠ selective risk |
| 10 | **FCP under label shift** (Plassier et al.) | ICML 2023 | Federated CP, label shift | Yes | Yes (importance weighting) | Yes (quantile regression) | **Coverage** only | Coverage | Closed-set |
| 11 | **Rob-FCP** (Byzantine-robust FCP) | arXiv 2024 | Federated CP vs malicious clients | Yes | Yes | Yes | **Coverage** (Byzantine setting) | Coverage | Closed-set; robustness, not OSR |
| 12 | **Decentralized Conformal Novelty Detection** | arXiv 2026-05 | Decentralized **novelty detection** | Yes (quantized model exchange) | Yes (heterogeneous composite nulls; conditional exchangeability) | Yes (quantized surrogate scores, no raw data) | **Global FDR** on a test batch | FDR, power | **Object = batch FDR, not classifier accepted selective risk**; pure novelty (no known-class classification) |
| 13 | **CGTC (Conformal Open-Set)** (Xie et al.) | arXiv 2025-10 | Open-set conformal classification | **No (centralized)** | n/a | n/a | **Coverage** of open-set sets (Good–Turing p-values) | Coverage | Not federated |
| 14 | **Selective Conformal Risk Control** | arXiv 2025-12 | Selective classification + CRC | **No (centralized)** | n/a | n/a | **Selective risk** | Risk | Not federated; closed-set base |

**Reading of the table.** The two properties our paper needs to hold *simultaneously* — (A) federated/heterogeneous + privacy-aware, and (B) a finite-sample certificate on the **selective risk of accepted, classified open-set predictions** — are never both present. Rows 1–8 have (A) but no certificate. Rows 9–12 have (A) + a certificate, but on the *wrong object* (coverage, or batch FDR — not classifier selective risk). Rows 13–14 have (B)-like guarantees but lack (A).

---

## 4. Gap Analysis (Which Sub-Axes Are Saturated vs. Open)

| Sub-axis | Saturation | Notes |
|---|---|---|
| FedOSR *architecture / aggregation* (disentanglement, voting, alignment) | **Saturated** | FedPD/FedPD++/FedOV/FedNovel cover this well. Hard to beat on pure empirical OSR. |
| Non-IID → OOD detection *as a method* | **Saturated** | FOOGD, FOSTER mature; "heterogeneity as a blessing" already mined. |
| Federated *coverage* guarantees (closed-set CP) | **Maturing** | Lu'23, Plassier'23, Rob-FCP'24, generative/conditional FCP'25. |
| Federated **certified selective risk for OSR** (accepted-prediction error control) | **OPEN** ← target | No paper controls R_sel = P(ŷ≠Y \| accept ∧ predicted-known) ≤ α with finite-sample federated certificate. |
| **Label noise / corruption × FedOSR × certification** | **OPEN** | NRFed (KSEM'25) touches noise but empirically. Certified version is unclaimed — and is the natural FL extension of your Paper 1. |
| **Heterogeneity-induced minority-known vs. unknown confusion**, treated as the certified object | **OPEN** | Acknowledged as a pathology (FedPD's "intra-set inconsistency") but never *certified against*. |

The convergence of the three open rows defines the contribution.

---

## 5. Proposed Contribution

**Working title:** *Fed-CORE — Federated Certified Open-set Recognition.*
**One-sentence thesis (parallel to your Paper 1):**

> In federated learning, data heterogeneity (and client-side label corruption) deforms the confidence–correctness ranking of the trained classifier; a small trusted clean calibration set, held across clients, should not be used primarily to repair the global model, but to **certify, with a finite-sample distribution-free guarantee, which open-set predictions can be safely accepted** — under partial exchangeability and without pooling raw calibration scores.

**Controlled object (the differentiator).** Let A(x)=1 denote "accept and classify x as a known class." We certify the **accepted selective risk**

  R_sel(h) = Pr( ŷ(X) ≠ Y | A(X)=1 ) ≤ α,

maximizing accepted coverage, where the certificate is a **one-sided Clopper–Pearson-type upper confidence bound** computed over a *federated* trusted calibration set. This is identical in spirit to your SRCC object but is, in FL, a genuinely new statistical problem because the calibration draws are **not i.i.d.** — they are partially exchangeable across heterogeneous clients with unknown mixture weights.

**Pipeline (mirrors and extends your Paper 1):**

  heterogeneous/corrupted FL-trained classifier
   → score-agnostic accept/reject proposal (risk-buffered, γα)
   → **federated independent certification under partial exchangeability**
   → **certified accepted coverage** with secure-aggregation-only leakage.

---

## 6. Novelty Verification — Adversarial Prior-Art Collision Analysis

Each plausible "this already exists" attack, with the rebuttal grounded in the verified sources.

| Attack ("X already did this") | Source | Why it does **not** collide |
|---|---|---|
| "Federated CP already gives finite-sample guarantees in FL." | Lu'23, Plassier'23, Rob-FCP | They guarantee **marginal coverage of closed-set prediction sets**, not selective risk, and explicitly **do not** handle unknown classes/rejection. Lu's own selective-classification demo is stated to have **no guarantee**. Object mismatch: coverage ≠ R_sel; closed-set ≠ open-set. |
| "Decentralized conformal novelty detection (2026) already certifies rejection in a federated way." | arXiv 2605.08263 | It controls **global FDR over a batch of test points** in a pure novelty-detection (inlier/outlier) framing — there is **no known-class classifier and no accepted-prediction selective risk**. FDR-over-a-test-batch and R_sel-of-accepted-classifications are different functionals with different finite-sample machinery (BH/e-values vs. binomial UCB). We cite it as the closest neighbor and differentiate on object + on certifying a *single fixed selector* (proposal/certification split) rather than batch multiple-testing. |
| "Open-set conformal classification (2025) already certifies open-set decisions." | CGTC, 2510.13037 | **Centralized.** No federation, no client, no heterogeneity/privacy. Also controls **coverage**, not selective risk. |
| "Selective Conformal Risk Control (2025) already controls selective risk." | arXiv 2512.12844 | **Centralized**, i.i.d. calibration. Does not address partial exchangeability, secure aggregation, or open-set/unknown structure under non-IID clients. |
| "FedPD/FOOGD already do FedOSR." | FedPD, FOOGD | **Empirical only** — AUROC/FPR95. FOOGD's sole theorem bounds **score-model estimation error (MMD)**, not the rejection error rate. No certificate on deployed accept decisions. |
| "It's just SRCC (Paper 1) ported to FL." | (reviewer's strongest attack) | This is the attack to kill with theory. The port is **non-trivial**: your Paper 1's Clopper–Pearson UCB assumes i.i.d. calibration. In FL the accepted-error events are **stratified across clients with unknown mixture weights λ∈Λ**; a valid UCB must be **robust over the mixture simplex** (agnostic-FL style) and computable from **secure-aggregated sufficient statistics only**. That theorem (N1 below) exists in neither Paper 1 nor Lu'23. |

**Net novelty claims (ranked by defensibility):**

- **N1 (core, theoretical).** A finite-sample, distribution-free **upper confidence bound on accepted selective risk under partial exchangeability** across heterogeneous clients with unknown mixture weights — a *binomial/Clopper–Pearson* certificate (controlling a risk), not a *quantile/coverage* certificate (Lu) and not a *batch FDR* certificate (DCND). This is the heart and must be the main theorem.
- **N2 (systems/privacy).** The certificate depends only on two aggregate sufficient statistics (Σ accepted, Σ errors-among-accepted), so it is **computable under secure aggregation with no leakage beyond the count itself** — strictly lighter than DCND's quantized score-function exchange and Lu's T-Digest sketches. Includes a stratified guard so no single client's accepted-error mass breaks the bound.
- **N3 (phenomenon).** The federated analog of your "confidence deformation": **heterogeneity (and corruption) jointly deform the confidence–correctness ranking**, making minority known classes statistically indistinguishable from unknowns at some clients; the **score-agnostic** certificate survives this because the guarantee comes from the certification split, not score quality.

**Honesty flag.** N1 is the only claim that single-handedly defeats the "trivial combination" reviewer. If N1's theorem reduces (after analysis) to a direct corollary of Lu'23 Thm 4.3 or of standard CP, the paper weakens to N2+N3 (a *systems/empirical* contribution) and should retarget accordingly. **Proving N1 is non-reducible is the first task before committing.**

---

## 7. Feasibility and Theoretical Soundness

**Known (verified):**
- Clopper–Pearson UCB for selective risk in the i.i.d. case is sound and is already your Paper 1 machinery (CP-UCB, risk-buffered proposal γα, feasibility N_min = ⌈log δ / log(1−α)⌉).
- Partial exchangeability across clients is an established, accepted relaxation (Lu'23, De Finetti/Diaconis lineage) and yields valid finite-sample statements with a K/(N+K) degradation for quantiles.
- Secure aggregation reveals exactly sums — compatible with a count-only certificate.

**Assumed (must be proven/validated):**
- That a *binomial-tail* (not quantile) UCB remains valid under partial exchangeability with the proposal/certification split. **Plausible** but requires its own proof; the quantile degradation in Lu does not transfer mechanically to a one-sided binomial bound. This is the main risk and the main prize.
- That the mixture-robust (worst-case over Λ) version is not so conservative that certified coverage collapses — analogous to your Paper 1's certified-coverage-collapse regime under extreme noise.

**Needs verification (empirical):**
- Whether certified accepted coverage stays **non-trivial** at CIFAR-scale FL with strong non-IID (Dirichlet α=0.1) — the exact open question that mirrors your Paper 1's central question. Synthetic + smoke tests first, per your Docker-first discipline.

---

## 8. Counterarguments, Weaknesses, and Mitigations

1. **"Incremental over your own Paper 1."** Mitigation: lead with N1 (partial-exchangeability risk UCB) as a standalone statistical contribution; frame Paper 1 as the i.i.d. special case (K=1). Position as a *journal* paper precisely because the theory + systems + empirical breadth justify length.
2. **"DCND (2026) is too close."** Mitigation: it is very recent and adjacent — cite prominently, run it (or its FDR rule) as a baseline re-cast to OSR, and differentiate on object (selective risk vs FDR) and on certifying a fixed deployed selector vs. batch testing.
3. **"Conservativeness kills coverage."** Mitigation: report `CertifiedCoverage@α` as the headline metric (as in Paper 1), characterize the collapse regime honestly, and show γ-buffer + split recovers it.
4. **"Why not just repair the global model?"** Mitigation: this is the thesis — the trusted set is too small to repair a heterogeneous global model but sufficient to *certify*; demonstrate empirically that certification beats repair at equal trusted-data budget.
5. **Privacy claim overreach.** Mitigation: state precisely what leaks (two integers per round under secure aggregation) and prove nothing else is needed; do not claim DP unless a DP variant is added.

---

## 9. Proposed Methodology and Experiments (for the eventual paper)

**Method skeleton.** (i) Train global model under FL with non-IID partition (Dirichlet α). (ii) Each client holds a slice of a *trusted clean* calibration set, split into proposal and certification folds. (iii) On the proposal fold, select a score-agnostic accept rule with risk-buffered target γα (γ∈{0.5,0.7,1.0}). (iv) On the certification fold, compute the **federated mixture-robust CP-UCB** on accepted selective risk via secure-aggregated (n, k). (iv) Deploy only if `cert_risk_ucb ≤ α`; report `cert_coverage_lcb`.

**Metrics (reuse Paper 1 schema):** `certified`, `cert_risk_ucb`, `cert_coverage_lcb`, `cert_n`, `cert_k`, `test_coverage`, `test_risk`, `prop_coverage`, `prop_risk`, `score_name`, `gamma`, plus `dirichlet_alpha`, `n_clients`.

**Scores (score-agnostic claim):** MSP, entropy, margin, energy.

**Datasets / setups:** CIFAR-10/100 with Dirichlet non-IID (α∈{0.1,0.5,5}); held-out classes as unknowns (standard FedOSR open-set split); optional symmetric/asymmetric client-side label noise to connect to Paper 1; later TinyImageNet, and a tabular/medical FL set for journal breadth.

**Baselines:** FedPD(++), FOOGD (score-norm), FedOV — re-evaluated under risk control; Federated CP (Lu) and Rob-FCP re-cast to OSR rejection; DCND FDR rule re-cast to selective risk; centralized CGTC / Selective-CRC as upper-bound oracles.

**Ablations:** with/without risk buffer γ; proposal/certification split vs. single-split union penalty; i.i.d. CP vs. mixture-robust CP (shows N1 matters); heterogeneity sweep (α) → certified-coverage-collapse curve.

---

## 10. Target Venue

Given a finite-sample theory contribution + privacy/systems framing + empirical FL breadth, the strongest journal fits are **IEEE TIFS** (privacy + security framing), **IEEE TNNLS** / **Pattern Recognition** (method + OSR), or **Information Fusion** (FL + reliability). If the theory (N1) is strong and self-contained, a top ML venue (NeurIPS/ICML) is viable; for a *journal*, TIFS or TNNLS best reward the certificate-under-privacy story. Decide after N1's proof is in hand.

---

## 11. Limitations of This Brief

- The 2026 decentralized-novelty paper (2605.08263) and FedPD++ (IJCV'26) are very recent; the field is moving fast, so a fresh prior-art sweep is warranted immediately before submission.
- Author lists marked "needs verification" in §12 were not individually confirmed against the official PDF; confirm before citing.
- N1's non-reducibility is argued, not yet proven. Treat this brief as a go/no-go gate, not as established theory.

---

## 12. References (with verification status)

Verified = confirmed via fetched full text or multiple consistent sources. NV = author list/venue not individually confirmed; verify before citing.

1. **FedPD: Federated Open Set Recognition with Parameter Disentanglement.** Yang et al. ICCV 2023. *Verified* (CVF open access; official repo CityU-AIM-Group/FedPD). https://openaccess.thecvf.com/content/ICCV2023/html/Yang_FedPD_Federated_Open_Set_Recognition_with_Parameter_Disentanglement_ICCV_2023_paper.html
2. **FedPD++: Enhanced Federated Open-Set Recognition with Parameter Disentanglement.** IJCV 2026. *Verified (venue)*; author list **NV**. https://link.springer.com/article/10.1007/s11263-026-02861-9
3. **FOOGD: Federated Collaboration for Both Out-of-distribution Generalization and Detection.** NeurIPS 2024. arXiv:2410.11397. *Verified (full text read; SM³D+SAG, score-norm detector, MMD error bound, baselines FedAvg/FedRoD/FOSTER/FedLN/FedATOL/FedT3A/FedIIR/FedTHE/FedICON).* https://arxiv.org/abs/2410.11397
4. **Towards Addressing Label Skews in One-Shot Federated Learning (FedOV).** ICLR 2023. *Verified (venue/OpenReview rzrqh85f4Sc)*; author list **NV**. https://openreview.net/forum?id=rzrqh85f4Sc
5. **Federated Continual Novel Class Learning (FedNovel / GAL).** 2023. arXiv:2312.13500. *Verified (arXiv)*; author list **NV**. https://arxiv.org/abs/2312.13500
6. **Towards Unbiased Training in Federated Open-world Semi-supervised Learning (FedoSSL).** Zhang et al. ICML 2023, PMLR v202. *Verified (PMLR)*. https://proceedings.mlr.press/v202/zhang23af/zhang23af.pdf
7. **Noise-Resistant Federated Open Set Recognition (NRFed).** Gao, Liu, Qin, Ou. KSEM 2025. *Verified (Springer chapter)*; confirm author order. https://link.springer.com/chapter/10.1007/978-981-95-3052-6_1
8. **Turning the Curse of Heterogeneity in FL into a Blessing for OOD Detection (FOSTER).** ICLR 2023. OpenReview mMNimwRb7Gr; repo illidanlab/FOSTER. *Verified (venue/repo)*; author list **NV**. https://openreview.net/forum?id=mMNimwRb7Gr
9. **Federated Conformal Predictors for Distributed Uncertainty Quantification.** Lu, Yu, Karimireddy, Jordan, Raskar. ICML 2023, PMLR v202. arXiv:2305.17564. *Verified (full text read; partial exchangeability Assump. 4.1, Thm 4.3 coverage 1−α ≤ · ≤ 1−α+K/(N+K), T-Digest; no open-set/rejection; selective-classification demo stated as heuristic without guarantee).* https://proceedings.mlr.press/v202/lu23i/lu23i.pdf
10. **Conformal Prediction for Federated Uncertainty Quantification Under Label Shift (Plassier et al.).** ICML 2023. *Verified (venue)*; author list **NV**. https://dl.acm.org/doi/10.5555/3618408.3619568
11. **Certifiably Byzantine-Robust Federated Conformal Prediction (Rob-FCP).** arXiv 2024. arXiv:2406.01960. *Verified (arXiv)*. https://arxiv.org/abs/2406.01960
12. **Decentralized Conformal Novelty Detection via Quantized Model Exchange.** arXiv 2026 (2026-05). arXiv:2605.08263. *Verified (arXiv abstract; decentralized, quantized surrogate scores, global FDR control, conditional exchangeability).* https://arxiv.org/abs/2605.08263
13. **Conformal Inference for Open-Set and Imbalanced Classification (CGTC).** Xie, Zhou, Liang, Favaro, Sesia. arXiv 2025-10. arXiv:2510.13037. *Verified (full text read; centralized; Good–Turing conformal p-values; coverage guarantee; no federation).* https://arxiv.org/abs/2510.13037
14. **Selective Conformal Risk Control.** arXiv 2025-12. arXiv:2512.12844. *Verified (arXiv); centralized.* https://arxiv.org/abs/2512.12844
15. **Classification with reject option: Distribution-free error guarantees via conformal prediction.** arXiv:2506.21802 / ScienceDirect 2025. *Verified (arXiv); centralized.* https://arxiv.org/pdf/2506.21802

*Cross-link to your project:* this proposal is the federated extension of **Selective Risk Control after Corrupted Training (Paper 1)**; the metric schema, risk-buffered proposal, and proposal/certification split are inherited directly.
