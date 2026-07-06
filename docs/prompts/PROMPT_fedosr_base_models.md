# Claude Code prompt — certify Fed-CORE on REAL FedOSR base models (FedPD / FedOSS / FOOGD)

Purpose. Close reviewer Risk 2: "you claim a certification layer for FedOSR methods, but
your experiments use FedAvg/CNN/ResNet scores, not real FedOSR models." We put Fed-CORE on
top of one or more **actual FedOSR base models** and certify each model's **native**
open-set score. The headline becomes: *Fed-CORE certifies accepted selective risk on top of
real FedOSR backbones, and the certified coverage tracks the base model's accepted risk
$\hat r$ exactly as the feasibility law predicts.*

Verified base models (official code exists):
- **FedPD** — ICCV 2023, Parameter Disentanglement (LPD + GDCA).
  repo: https://github.com/CityU-AIM-Group/FedPD  (mirror: CUHK-AIM-Group/FedPD)
- **FedOSS** — IEEE TMI 2023, DUSS + FOSS virtual-unknown synthesis.
  repo: https://github.com/CityU-AIM-Group/FedOSS
- **FOOGD** — NeurIPS 2024, SM3D score model + SAG. CIFAR-native (easiest).
  repo: https://github.com/XeniaLLL/FOOGD-main

Decision (Sanghoon): commit additional GPU budget to **reproduce all three base models in
full**. Order by ramp-up cost = FOOGD → FedPD → FedOSS. The immediate carry-over from the
2026-06-28 session (PRIORITY 3) is the **full feature-space FOOGD**: export penultimate
features and run the real SM3D score-norm detector, replacing the current logit-space proxy
(correctly labeled "representative FedOSR-style score" in the draft until then). Reproduction
risk: FedPD/FedOSS repos are medical-imaging oriented, so CIFAR adaptation is the main cost.
Never fabricate: if a repo will not reproduce within budget, report exactly where it failed
and fall back to a faithful "representative FedOSR-style score head" on shared features,
clearly marked `base_model_kind=representative` — do not pass it off as the full method.

Paste the fenced block into Claude Code in the FedCORE repo.

```text
READ CLAUDE.md AND AGENTS.md FIRST. Fed-CORE = post-hoc certification layer for federated
open-set models. Object = certified accepted selective risk (cert_risk_ucb,
cert_coverage_lcb, test_risk, test_coverage; headline CertifiedCoverage@alpha). Docker-first.
NEVER use proposal/certification/test labels in training; the base model must be trained ONLY
on training clients, with the audit (proposal/cert/test) folds HELD OUT. Judge by cert_* and
matched-risk, not accuracy. Report every failure. Stop-and-ask before large GPU jobs.

GOAL. For each FedOSR base model B in {FOOGD, FedPD, FedOSS} (in that priority order),
train B on a CIFAR-10 FedOSR split, export B's NATIVE open-set score on the held-out
proposal/cert/test folds, and run the Fed-CORE certificate on that score. Produce one
comparison table (T8) showing Fed-CORE certifies on top of each real base model.

SPLIT PROTOCOL (critical — identical across all base models, reuse fedosr_split.py).
  - CIFAR-10, n_known=6 known classes, 4 held-out as `unknown`.
  - non-IID over known classes via Dirichlet, dirichlet_alpha in {5, 0.5}.
  - n_clients = 5 (match the existing CIFAR runs).
  - Partition each client's data into TRAIN (for the base model) and a disjoint AUDIT pool;
    the AUDIT pool is split into proposal / certification / test folds (cert_frac=0.5).
    Held-out unknown-class points appear ONLY in the audit folds (never in TRAIN).
  - SPLIT IDENTITY = "identical audit, matched train" (Sanghoon's decision):
      * AUDIT folds are BYTE-IDENTICAL across all base models — the same CIFAR-test indices and
        the same y_open feed every model's proposal/cert/test. All cert_*/test_* numbers are
        therefore computed on exactly the same points (apples-to-apples; this is what T8
        comparability and certificate validity require).
      * TRAIN need NOT be byte-identical: each base model draws from the SAME known-class pool
        with a MATCHED Dirichlet(dirichlet_alpha) and the SAME seed, so heterogeneity is
        equalized, but each method may partition/preprocess its own way.
      * NON-NEGOTIABLE invariant: the audit indices are EXCLUDED from every base model's TRAIN
        (disjointness preserved). Do not rewrite each repo's loader to consume our exact TRAIN
        partition (the rejected "fully identical split" — high glue/reproduction risk for no
        comparability gain); only the audit scoring must be on identical indices.

PER BASE MODEL.
  1. FOOGD (do first; CIFAR-native). DECISION = "sure-win first, then full SM3D".
     STEP 1a (smoke, ~10 min, de-risk deps): clone XeniaLLL/FOOGD-main, install its env in a
       torch container, run a few training steps to confirm it actually trains. Report pass/fail
       BEFORE spending real GPU. Do not skip this.
     STEP 1b (SURE-WIN — the real SM3D detector on our backbone): export penultimate features
       from OUR federated backbone (the existing FedAvg/GroupNorm model) for every audit-fold
       point, train the FOOGD SM3D score model on those features, and use the SM3D score as the
       NATIVE open-set score. Export per audit-fold point: SM3D score, closed-set logits/pred
       yhat, true label, client id. Certify with Fed-CORE. Label base_model_kind = "sm3d_on_
       fedavg" (a genuine FOOGD detector on a FedAvg backbone — honest, and it closes Risk 2).
     STEP 1c (FULL, after 1b lands): run FOOGD's own main.py (full federated SM3D + SAG), then
       adapt to our split. This is the highest-fidelity version; label base_model_kind = "full".
       If main.py will not reproduce in budget, STOP, report where, and keep the 1b sure-win as
       the FOOGD row.
  2. FedPD (second).
     - clone CityU-AIM-Group/FedPD; adapt its data loader to the CIFAR FedOSR split above
       (the repo targets medical data — wire in CIFAR-10 with the same known/unknown split).
     - native score = the open-set subnetwork's unknown score. Export the same triple.
     - if CIFAR wiring is infeasible in budget: STOP, report exactly where it failed, and
       fall back to a faithful re-implementation of FedPD's open-set score head on the shared
       FedAvg features, labeled "FedPD-style (representative)".
  3. FedOSS (third; same pattern).
     - clone CityU-AIM-Group/FedOSS; adapt loader; native score = its unknown probability
       from DUSS/FOSS. Same export + same representative fallback rule if needed.

CERTIFY (reuse certify.py / certificates.py; identical settings for all rows).
  - score = each base model's NATIVE score (FIXED per model; this is the honest protocol).
  - worst-group G=2, cert_frac=0.5, delta=0.10, gamma in {0.5,0.7,1.0}, seeds {0,1,2}.
  - alpha in {0.10, 0.20}.

TABLE T8 (the deliverable). Rows = {FedAvg+MSP (existing baseline), FOOGD, FedPD, FedOSS};
columns per (dirichlet d, alpha):
  - base model's own open-set metric: AUROC and closed-set accuracy (context only)
  - accepted empirical risk r_hat
  - CertifiedCoverage@alpha (worst-group G=2, mean+/-std over seeds, n_pass/3)
  - median cert_risk_ucb among certified seeds
  - test_risk, test_coverage
Save runs/T8_fedosr_bases.csv with the canonical schema (certified, cert_risk_ucb,
cert_coverage_lcb, cert_n, cert_k, prop_coverage, prop_risk, test_coverage, test_risk,
score_name, gamma, alpha, delta, dirichlet_alpha, n_clients, base_model) and an aggregate
runs/T8_fedosr_bases_agg.csv. Add a column flag base_model_kind in
{full, sm3d_on_fedavg, representative} so the FOOGD sure-win (real SM3D on a FedAvg backbone)
is distinguished from the full SM3D+SAG pipeline and from any fallback score head.

REPORT (fixed format) per base model:
  진단 요약 / 확인한 명령 / 핵심 결과 (alpha, gamma, score_name, cert_risk_ucb,
  cert_coverage_lcb, test_risk, test_coverage, base_model, base_model_kind) /
  판정 (strong/moderate/warning/fail) / 다음 행동.

EXPECTED STORY (state honestly, do not force):
  - Fed-CORE should certify a non-trivial CertCov@0.20 on top of base models whose r_hat is
    comfortably below alpha, and degrade/abstain as r_hat -> alpha — the SAME feasibility law,
    now on real FedOSR scores. The ranking of CertCov across base models should follow their
    r_hat, NOT their AUROC (this is the score-agnostic point: the certificate's coverage is
    governed by accepted risk, not ranking quality).
  - If a base model has high AUROC but high accepted r_hat, Fed-CORE will certify little —
    report this; it is evidence FOR the paper's thesis (ranking != certifiable risk), not
    against it.

CONSTRAINTS / HONESTY:
  - one fixed native score per base model; no best-of-scores selection in the headline.
  - never leak audit-fold labels into training or score selection.
  - label any representative reimplementation as base_model_kind=representative.
  - report partial success (e.g. only FOOGD reproduced) plainly; one real base model already
    answers Risk 2, three is the stretch.
```

---

### Notes for Sanghoon
- **FOOGD가 가장 빠른 진짜 base model**입니다(CIFAR-native + SM3D score). 최소 이거 하나만
  certify돼도 Risk 2의 핵심 공격("진짜 FedOSR 아님")은 막힙니다.
- FedPD/FedOSS는 의료영상 repo라 CIFAR 데이터로더 wiring이 관문입니다. 안 되면 정직하게
  "representative score head"로 낮춰 라벨(`base_model_kind=representative`)하고 보고하게
  설계했습니다 — 절대 full로 위장하지 않음.
- 결과가 오면 manuscript에 **§5.7 "Certification on real FedOSR base models" + Table 8**을
  추가하고, line 291의 "FedPD/FedOSS full training is deferred" 문장을 실제 결과로 교체하겠습니다.
- 핵심 메시지: CertCov 순위가 AUROC가 아니라 **r̂(accepted risk)** 를 따른다는 점 — 이게
  논문의 score-agnostic thesis를 real FedOSR 위에서 다시 입증합니다.

### Sources (base-model verification)
- FedPD (ICCV 2023): https://openaccess.thecvf.com/content/ICCV2023/html/Yang_FedPD_Federated_Open_Set_Recognition_with_Parameter_Disentanglement_ICCV_2023_paper.html · code https://github.com/CityU-AIM-Group/FedPD
- FedOSS (IEEE TMI 2023): https://ieeexplore.ieee.org/document/10177875/ · code https://github.com/CityU-AIM-Group/FedOSS
- FOOGD (NeurIPS 2024): https://arxiv.org/abs/2410.11397 · code https://github.com/XeniaLLL/FOOGD-main
