# Guideline Edits Report (Academic Paper Guidelines ver. 2.2) — 2026-07-11

대상: `docs/Fed-CORE_draft.md` (canonical source) → 양쪽 빌드(Elsevier 판, 랩 템플릿 판) 모두 반영.
전체 변경의 정확한 diff는 `git diff`로 확인 가능. 아래는 규칙별 요약 + 대표 예시 (변경부 **bold**).

## 1. Em-dash(—) 문장 제거 — 70건 전면 재작성

본문 프로즈의 em-dash 97건 중 91건을 콜론·세미콜론·괄호·관계절로 재작성 (의미 불변).
표 안의 빈칸 표시 "—" 6건은 표준 조판 관행이라 유지.

대표 예시:

- how often is it wrong — and can we promise → how often is it wrong**, and** can we promise
- it does not try to make the model better — it certifies → better**: it** certifies
- methods improved the quality of unknown rejection — A in FedPD [5], … — and reported
  → rejection **(**A in FedPD [5], …, **and** novel-class discovery in FedNovel [8]**)** and reported
- the worst client sets the bar — no client's error… → the worst client sets the bar**: no** client's error…
- a Poisson-binomial — so the single-binomial… → **(a Poisson-binomial distribution); hence** the single-binomial…
- **(i) validity** — certified… → **(i) validity:** certified… (실험 claim (i)–(iv) 동일 패턴)
- empirically — as a sanity check, not the source of validity — no certified…
  → empirically**, as a sanity check rather than** the source of validity**,** no certified…

## 2. Connective words — 49건

| 규칙 | 건수 | 처리 |
|---|---|---|
| so → hence/thus/therefore | 30 | ", so " → "**; hence** " (전건). "so that"(목적)은 규칙 대상 아님·유지 |
| but → however 등 | 14 | ", but " → "**; however,**" / 소절 대조는 "**yet**"·"**although**"·"**while**"로 재구성. 상관구문 "not only…but also", "not X but Y" 5건은 문법 필수라 유지 |
| also → additionally | 3 | "The figure **additionally** marks…", "…risk ***additionally*** starved…", "coverage **additionally** depends…". "not only…but **also**"는 유지 |
| since → because | 2 | 인과 용법 2건 모두 "**because**"로 |

대표 예시:

- Lemma 1 is elementary, but it fixes… → Lemma 1 is elementary**; however,** it fixes…
- unseen during training but present and labeled → unseen during training **yet** present and labeled
- is valid but uniformly looser → is valid**, although** uniformly looser
- …identical, so any procedure… deploys → …identical**; hence** any procedure… deploys
- …intervals, since Lemma 1 couples… → …intervals **because** Lemma 1 couples…

## 3. 기타 규칙 점검 결과 (변경 불필요 — 이미 준수)

- **Contractions**: 0건 (n't / it's 등 없음)
- **Extreme words**: "novel" 3건은 전부 기술용어(novel-class discovery, novel classes) — 유지.
  very / really / excellent: 0건
- **Numbers 0–10 in words**: 프로즈 위반 0건 (two integers, six known classes, ten seeds 등
  이미 단어 표기; 잔여 숫자는 인용 [n]·정리 번호·통계 표기 8/10·수식으로 규칙 예외)
- **Capitalization**: Table/Figure/Section/Theorem 소문자 사용 0건
- **Oxford comma**: FedNovel 나열부 1건 수정(위 예시), 나머지 준수
- **dataset**: "data set" 0건, dataset 일관
- **following + colon**: "summarized as follows:" 준수
- **Tense**: Related Work 과거(framed/targeted/tackled…), 방법 현재, 결과 과거 — 구조 준수
- **a/an**: "an MMD bound", "a one-group certificate" 등 발음 기준 준수 확인

## 4. Table/Figure 본문 설명 커버리지 점검 (요청 2)

전 표·그림에 본문 walkthrough 존재 확인 (신규 추가 없이 충족):

| 항목 | 본문 설명 위치 |
|---|---|
| Figure 1 | §3 도입부 (네 가지 질문 구분 + fold 진입 지점) |
| Table 1 | §2 Positioning ("Table 1 organizes prior work by the object it certifies") |
| Table 2 | §3 ("Two deserve emphasis. A4/A4′ … A6 …") |
| Table 3 | §4.6 (variant별 target/released/가정 설명 문단) |
| Figure 2 | §5.2 (population 구성 + 세 인증서의 거동 해석 문단) |
| Table 4 | §5.2 (lead-in + naive/leaked 해석 + FCP-recast 문단) |
| Figure 3 | §5.3 (양 패널 수치 해석 + composition 스터디) |
| Figure 4 | §5.4 (staircase / panels (a)(c) 통합 해석 / (d) client scaling 문단) |
| Table 5 | §5.5 Headline 문단 + 캡션 Notes (열 정의 포함) |
| Table 6 | §5.5 CIFAR-100 문단 (셀 판독 + 절대값이 작은 이유) |
| Figure 5 | §5.5 Stress axes 문단 (양 축 메커니즘) |
| Figure 6 | §5.5 thesis 문단 + 캡션 (패널별 판독법) |
| Table 7 | §5.6 (세 행 비교 해석) |

## 5. 빌드 결과

- Elsevier 판: 50pp (Fed-CORE_draft.docx / .pdf)
- 랩 템플릿 판: 59pp (Fed-CORE_draft_labtemplate.docx / .pdf)
- 검증: 본문 em-dash 0건(표 placeholder 6건만), ", so " / " But " / " since " 잔존 0건,
  인용 무결성(등장순 단조·1–43·미인용 0) 유지
