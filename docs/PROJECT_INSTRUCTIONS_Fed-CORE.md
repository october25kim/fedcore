# Project Instructions — Fed-CORE (Federated Certified Open-Set Recognition)

> 이 텍스트를 프로젝트 설정의 Instructions 칸에 **통째로 붙여넣어** 기존
> SRCC/Paper 1 지침을 교체한다. (프로젝트 이름은 유지해도 되지만, 원하면
> "Fed-CORE — Federated Certified Open-Set Recognition"으로 바꾸는 것을 권장.)

이 프로젝트는 논문 **"Fed-CORE: Federated Certified Open-Set Recognition"**
관리를 위한 전용 프로젝트다. 목표는 저널 투고 가능한 flagship 논문을 완성하는
것이다. **SRCC(Paper 1, centralized selective risk control after corrupted
training)는 novelty 사유로 폐기**되었고, 그 CP 기계장치는 Fed-CORE의 특수경우
(K=1)로 흡수된다.

## 1. 핵심 연구 객체

이 논문은 FedOSR-accuracy 논문도, noisy-label robust training 논문도 아니다.
핵심 객체는 다음이다.

> 이질적(non-IID)·corrupted 환경에서 federated로 학습된 open-set classifier를
> 재학습하지 않고, deployment 단계에서 어떤 prediction을 accept할 수 있는지를,
> 클라이언트에 분산된 작은 trusted clean calibration set으로, **secure-
> aggregatable counts만 사용해** finite-sample certify하는 문제.

중심 risk는 deployment mixture 하의 **accepted selective risk**다.

$[
R_{\mathrm{sel}}(\lambda)=\frac{\sum_j \lambda_j m_j}{\sum_j \lambda_j a_j},\quad
a_j=\Pr_{P_j}(\text{accept}),\ m_j=\Pr_{P_j}(\text{accept}\wedge\text{error})
]$

목표는 미지의 deployment $\lambda\in\Lambda$에 대해

$[
R_{\mathrm{sel}}(\lambda)\le \alpha
]$

를 보증하면서 accepted coverage를 최대화하는 것이다.

핵심 metric은 accuracy/AUROC가 아니라 다음이다.

* `certified`
* `cert_risk_ucb`
* `cert_coverage_lcb`
* `cert_n`
* `cert_k`
* `test_coverage`
* `test_risk`
* `prop_coverage`
* `prop_risk`
* `score_name`
* `gamma`
* `alpha`, `delta`, `Lambda`, `dirichlet_alpha`, `n_clients`

Headline metric은 **CertifiedCoverage@alpha**.

## 2. 논문 thesis

> Heterogeneity(및 client-side label corruption)는 confidence–correctness
> ranking을 변형시킨다. 클라이언트에 분산된 작은 trusted clean set은 모델을
> 수리하는 데 쓰는 것이 아니라, corrupted/heterogeneous하게 학습된 모델의 어떤
> prediction을 안전하게 accept할 수 있는지 certify하는 데 써야 한다.

핵심 방법 구조:

$[
\text{heterogeneous/corrupted FL-trained classifier}
\rightarrow
\text{score-agnostic risk-buffered proposal}
\rightarrow
\text{federated independent certification (partial exchangeability)}
\rightarrow
\text{certified accepted coverage (count-only leakage; see privacy taxonomy)}
]$

## 3. 핵심 이론 (반드시 유지)

1. **Impossibility.** corrupted/unlabeled 정보만으로는 clean selective risk를
   distribution-free하게 control할 수 없다. trusted clean calibration set이
   필요하다.

2. **Confidence deformation motivation.** non-IID(및 class-conditional
   corruption)에서 CE-optimal 모델은 minority known class를 unknown과
   통계적으로 구분 못 하게 되어 ranking이 변형된다. 단 이건 motivation이지 main
   guarantee가 아니다.

3. **Theorem 1/1′ (core, 비환원) — conditional selective-risk certificate.**
   conditional law $K_j\mid A_j\sim\mathrm{Bin}(A_j,r_j)$로 $r_j$를 직접 bound:
   $\bar r_j=U^+(K_j,A_j;\delta/J)$. 전 단순체 $\bar U_\Delta^{r}=\max_j\bar r_j$.
   bounded $\Lambda$(권장, Thm 1′): $a_j\in[\underline a_j,\bar a_j]$도 bound 후
   $\sup_{\lambda\in\Lambda,a}(\sum\lambda a\bar r)/(\sum\lambda a)$. 기존 mass-ratio
   $\max_j\bar m_j/\underline a_j$보다 uniformly tight (그건 App-C baseline).
   Edge case: zero coverage→non-deployable, denom bound 0→infeasible(client 제외 금지).

4. **Non-reducibility crux.** 이질 클라이언트의 accepted 점을 단순 pool하면
   per-client $r_j$가 달라 pooled count가 Poisson-binomial(binomial 아님)→CP 무효.
   Theorem 1은 centralized CP도, federated conformal(Lu, quantile/coverage)도 아니다.

5. **Theorem 2 (feasibility).** per-client **observed** accepted count
   $A_j\ge \ln(J/\delta)/(-\ln(1-\alpha))=\Omega(\ln(J/\delta)/\alpha)$.
   expected-count $n_j a_j$ 형태는 corollary.

6. **Proposition 3 (pooled, subordinate).** matched-mixture i.i.d. calibration에서만
   유효. 두 gap: (L) Lemma L(수치 SUPPORTED 0.919≥0.90, 형식증명 TODO),
   (C) roster-composition coupling. **Theorem 1/1′보다 격상 금지.**

7. **Privacy taxonomy (정정).** pooled만 sum-only secure-agg. stratified는 per-client
   counts 필요; grouped-stratified(G strata, ≥k clients)가 compromise.
   "two counts only"를 stratified에 주장 금지.

8. **Calibration 가정 (명시).** unknown rejection certify엔 certification fold에
   **labeled unknown** 필요. "distribution-free"는 calibration 분포 기준.

9. **Score-agnostic certificate.** MSP/entropy/margin/energy 등 어떤 score를
   쓰든 최종 guarantee는 score 품질이 아니라 certification split의 conditional CP에서 나온다.

10. **Proposal / certification / test split.** proposal split에서 selector를
    고르고 independent certification split에서 단일 selector를 certify한다. test
    split은 deployment 추정용이며 절대 proposal/certification에 누설 금지.

11. **Risk-buffered proposal.** proposal에서 $\widehat R_{\mathrm{prop}}\le\gamma\alpha$,
    $0<\gamma<1$. 기본 후보 `{0.5, 0.7, 1.0}`.

## 4. 실험 우선순위

최우선은 Fed-CORE의 CIFAR-level deep reproduction이다.

1. fake-logit smoke 인증 확인 (`run_smoke.py`)
2. CIFAR-10 clean, seed 0, dirichlet_alpha=0.1
3. CIFAR-10 symmetric 35% (client-side, train labels only), seed 0
4. CIFAR-10 asymmetric 20%, seed 0
5. seed 1, 2 반복
6. dirichlet_alpha sweep {0.1, 0.5, 5} → certified-coverage-collapse curve (Thm 2)
7. CIFAR-100으로 확장

corruption은 학습 라벨에만 주입하고 **trusted calibration fold는 clean 유지**.

## 5. Docker-first 원칙

모든 GPU 실험은 Docker 내부에서 실행한다. host Python으로 직접 학습하지 않는다.

```bash
# CPU sanity (no torch)
python experiments/fedcore/exp_lemma_L.py
python experiments/fedcore/exp_pooling_fail.py
python experiments/fedcore/run_smoke.py

# GPU, real CIFAR
bash scripts/docker_cifar.sh
# or:
python experiments/fedcore/run_cifar.py --dataset cifar10 --n_known 6 \
  --dirichlet_alpha 0.1 --rounds 50 --local_epochs 2 --alpha 0.10 --delta 0.10
```

## 6. Git workflow

`runs/`, `data/`, `outputs/`, `checkpoints/`, `logs/`, `wandb/`, `*.pt`, `*.pth`,
`*.npy`, `*.npz`는 commit하지 않는다.

## 7. Agent instruction rule

Claude Code 또는 Codex 사용 시 항상 repo root의 `CLAUDE.md`, `AGENTS.md`를 먼저
읽게 한다. 이 파일들은 반드시 **Fed-CORE 전용**이어야 한다.

올바른 instruction은 다음 용어를 포함해야 한다.

* Federated Certified Open-Set Recognition (Fed-CORE)
* accepted selective risk / `R_sel(lambda)`
* stratified worst-case-mixture certificate / partial exchangeability
* proposal/certification/test split
* risk-buffered proposal
* Clopper–Pearson UCB / ratio-of-binomials
* `cert_risk_ucb`, `cert_coverage_lcb`, `test_risk`, `test_coverage`

다음이 **main object로 남아 있으면 잘못된 instruction**이며 즉시 교체한다:
SRCC/Paper 1을 flagship으로 두기, RC-OWPL, pseudo-labeling, accept/defer/reject
triage, open-world pseudo-label contamination. (federated learning은 이제
올바른 핵심 setting이다.)

## 8. 응답 방식

일반 토론·진단·계획은 한국어. 코드·파일명·변수명·config key·commit message·최종
deliverable(paper text)은 영어.

실험 결과 해석은 반드시 다음 형식:

```text
진단 요약:
- ...

확인한 명령:
- ...

핵심 결과:
- alpha=... / delta=... / Lambda=...
- score=... / gamma=... / dirichlet_alpha=...
- cert_risk_ucb=... / cert_coverage_lcb=...
- test_risk=... / test_coverage=...

판정:
- strong go / moderate go / warning / fail

다음 행동:
- ...
```

## 9. 금지 사항

* smoke run 결과를 manuscript claim으로 과장하기
* accuracy/AUROC만 보고 성공 판단하기
* proposal/certification/test split 섞거나 test label을 proposal/certification에 사용
* `gamma=1.0`만 보고 risk-buffer 불필요 결론
* **이질 환경에서 accepted 점을 pool해 단일 binomial CP 적용하기** (무효)
* Theorem 3(pooled)을 Theorem 1(stratified)보다 격상하기
* SRCC/RC-OWPL/pseudo-labeling을 main object로 재도입
* certified-coverage collapse를 숨기기 (정직하게 보고; 그 자체가 논문 결과)
* failed command 숨기기 / unrelated refactor

## 10. 현재 연구 상태 요약

확인된 것:

* 메타분석 + adversarial novelty 검증 완료 → gap(federated certified selective
  risk)이 실재. 최근접 prior art(decentralized conformal novelty 2026 = batch
  FDR; federated CP = coverage; centralized open-set conformal)와 객체로 차별화.
* 논문 draft(Intro/RW/Method + Theorem 1–3 + proof sketch + 실험계획) 작성.
* **Lemma L: 수치적으로 SUPPORTED** (worst-case coverage 0.919 ≥ 0.90).
* **Pooling-fail ablation: 비환원성 확인** — mixture shift에서 naive pooled CP
  coverage가 0%로 붕괴, stratified Theorem 1은 모든 mixture에서 유지.
* FedOSR 파이프라인 scaffold + fake-logit smoke 인증 통과(box-Λ 3/12, simplex 0/12).

미해결 (최우선 질문):

$[
\text{CIFAR-level non-IID corrupted training에서도 certified accepted coverage가 nontrivial한가?}
]$

이 질문에 답하는 것이 이 프로젝트의 1차 목표다. (`run_cifar.py` GPU 실행 필요.)
