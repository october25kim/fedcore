# FedCORE S1 v2 — Preregistered Protocol

- **protocol_id:** `FEDCORE-S1-DOMAINNET6-PACSCLS-20260925-v2`
- **Sealed:** 2026-09-25 20:47 KST (see `PROTOCOL.json` `seal_kst`). Sealed **before** any DomainNet image was downloaded or decoded and before any DomainNet model output existed.
- **Machine-readable spec:** `PROTOCOL.json` is authoritative. If this file and the JSON disagree, the JSON wins.
- **Bootstrap seed (new):** `2026092501`

## 0. This is a separate experiment
This protocol is a **new, separately registered experiment**. It is **not** S1 v1 and must never be presented as S1 v1.

S1 v1 (`FEDCORE-S1-DOMAINNET-PACSCLS-20260923-v1`, sha256 `e7dc3918…3173`) was declared **INVALID_BY_PROTOCOL** on 2026-09-25 because the DomainNet 345-class taxonomy has no `person` class. That failure was found from label lists alone: no image was decoded and no model output was computed. v2 was registered after that class-availability failure. The S1 v1 protocol and its INVALID record stay as they are, permanently.

Information available when v2 was sealed:
- DomainNet cleaned label lists (class names and counts only)
- HEAD metadata for the official zips
- V1 checkpoint and code hashes
- the locked V1 PACS result (HOLD)

No DomainNet pixels and no DomainNet predictions were available.

## 1. Question
Take the 30 frozen V1 PACS predictors, with no retraining, fine-tuning, or recalibration. On DomainNet (the 6 official domains, restricted to the 6 PACS classes that DomainNet has), does the V1 selection rule `V1_SCALE_CP` keep its ECA advantage over the 7 fixed procedures that have equal information and equal budget?

This is descriptive of transfer only. It is not a rewrite of V1. It makes no claim that RATIO is superior in general, and no claim that geometry alone causes the effect.

## 2. Data (decided before download)
- **Version: CLEANED** (the version the official page calls "recommended"). Reasons:
  - The authors recommend it.
  - It removes duplicate images, which matches V1's duplicate-collapse rule.
  - Its label lists are exactly the lists already hash-bound in the S1 v1 evidence.
  - V1-style `path label` lists apply directly.
- **Zip URLs** (official page https://ai.bu.edu/M3SDA/, snapshot sha256 `4e98f2fb…3141`):

| domain | URL | HEAD Content-Length |
|---|---|---:|
| clipart | https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip | 1,270,909,081 |
| infograph | https://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip | 4,338,659,804 |
| painting | https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip | 3,679,366,174 |
| quickdraw | https://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip | 459,920,701 |
| real | https://csr.bu.edu/ftp/visda/2019/multi-source/real.zip | 6,062,039,589 |
| sketch | https://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip | 2,633,992,926 |

  For infograph, quickdraw, real, and sketch, the official page links the same zip under both "cleaned" and "original". Cleaning is applied through the cleaned lists.
- **Label lists:** `https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/{domain}_{train,test}.txt`. Their sha256 values are in `refs/official_cleaned_lists.sha256` and `PROTOCOL.json`.
- **Zip sha256 binding:** each zip's sha256 is recorded at download in a **DATA_BINDING** addendum in the run directory. That addendum is not an analysis choice. A size mismatch is a transport error and is fixed by re-downloading. No other source may be substituted, and that includes other users' DomainNet copies.
- **Which lists are used:** train ∪ test (both). The predictors never train on DomainNet, so the DomainNet train/test partition plays no role.
- **Which images are selected:** every list entry in the 6 classes across all 6 domains. There is no subsampling. The class is the second path component, matched case-insensitively and exactly after strip.

Listed counts (train+test, cleaned). These are label counts, not outcomes:

| class | clipart | infograph | painting | quickdraw | real | sketch |
|---|---:|---:|---:|---:|---:|---:|
| dog | 70 | 220 | 718 | 500 | 779 | 310 |
| elephant | 115 | 187 | 424 | 500 | 783 | 266 |
| giraffe | 133 | 168 | 105 | 500 | 591 | 186 |
| guitar | 100 | 201 | 202 | 500 | 626 | 181 |
| horse | 200 | 210 | 521 | 500 | 595 | 101 |
| house | 107 | 305 | 105 | 500 | 374 | 144 |
| **total** | 725 | 1291 | 2075 | 3000 | 3748 | 1188 |

Total: 12,027 listed entries.

- **Identity and duplicates:** V1's rules, unchanged.
- **Roles:** `H = int(SHA256(pid|role|identity))`. An image goes to proposal if `2H < 2^256`, otherwise to audit. There is **no training role**: the predictors are frozen, so the V1 training share (60%) would only throw away scarce support. V1's proposal:audit ratio of 1:1 is kept. Roles are global across splits.

## 3. Classes, splits, and person asymmetry
- **Classes:** dog, elephant, giraffe, guitar, horse, house. `person` is excluded and is **not** replaced by face, hand, or any other class.
- **Splits:** the 5 V1 splits are copied verbatim, because the frozen 4-way heads fix them. Head order is alphabetical, as in V1.

| split | frozen head order | DomainNet-supported known | DomainNet unknown | person |
|---|---|---|---|---|
| 0 | dog, elephant, giraffe, guitar | all 4 | horse, house | unknown (absent) |
| 1 | dog, elephant, house, **person** | dog, elephant, house | giraffe, guitar, horse | known head idx 3, no inputs |
| 2 | elephant, giraffe, horse, **person** | elephant, giraffe, horse | dog, guitar, house | known head idx 3, no inputs |
| 3 | dog, guitar, horse, house | all 4 | elephant, giraffe | unknown (absent) |
| 4 | dog, elephant, guitar, house | all 4 | giraffe, horse | unknown (absent) |

- **Person is a V1 known class (splits 1, 2):** the predictor is used exactly as frozen. That means the full head and all 4 logits in msp, energy, margin, and argmax, with no removal, masking, or renormalization. `argmax=person` on a known image counts as a wrong-known error. Any acceptance of an unknown image counts as an error. This is V1's own rule, `err=(label<0)|(argmax!=label)`, so no structural adaptation was needed.
- **Person is a V1 unknown class (splits 0, 3, 4):** the unknown set has 2 classes instead of 3.
- **PRIMARY analysis:** all 5 splits, 30 models, V1 aggregation.
- **Pre-specified SECONDARY (sensitivity) analysis:** splits {0,3,4}, where all known heads are supported. That is 9 models per architecture. It uses the same 16 contrasts, the same bootstrap seed and quantiles, and applies the V1 decision function **descriptively**. It never overrides the primary verdict. Splits {1,2} are reported descriptively. After the decision, a person-prediction-rate diagnostic is reported for splits 1 and 2.

## 4. Budget, allocation, and support
- **J = 6** clients, one per domain. The V1 certifier is generic in J, and every tail keeps its V1 formula evaluated at J=6. For example, the common-acceptance tail becomes `.05/(6·12) = .05/72`.
- **Total budgets are the same as V1 and S1 v1:** B = 512 (primary), 256, and 1024 (secondary), counted as with-replacement draws.
- **Allocation:** equal per client, `n_B = floor(B/6)` = 42 / 85 / 170 per domain. The realized totals are 252 / 510 / 1020.
- **Remainder rule:** the leftover 4 / 2 / 4 draws are not taken by any domain. This keeps V1's equal-per-client contract: every domain and every method gets the same n, and the registered total is never exceeded.
- **Small cells:** draws are taken with replacement from each domain's identity-sorted audit frame, with the 6 classes pooled. So a cell smaller than the budget needs no special handling. Examples: infograph giraffe 168, clipart dog 70, sketch horse 101. Cells matter only through the support gate.
- **Draws:** the V1 stream `audit_indices(pid_v2, split, domain, r, size)[:n_B]`, 100 replicates, coupled across architectures, seeds, methods, t values, and budgets, as in V1.
- **Minimum-support gate (G4):** each of the 36 (class, domain) cells needs **at least 10 unique images in proposal AND at least 10 in audit**. If **any** cell fails, the **whole run is INVALID**. No cell or domain is ever dropped, because every cell is shared by the J=6 equal-budget structure and by all splits, so dropping one would silently change the domain or class list. The expected minimum from the lists is about 35 per role.

## 5. Methods, metrics, gates, bootstrap
- **Taken from V1 unchanged:**
  - candidate family (12)
  - thresholds
  - proposal rule (pooled over 6 domains)
  - α=.2, δ_r=δ_c=.05, t∈{.5,1}
  - 7 fixed controls
  - V1_SCALE_CP rule
  - GEOMETRY_DIRECT_CP diagnostic
  - ECA metric
  - 14 primary + 2 diagnostic contrasts
  - bootstrap: 20000 resamples, quantiles .05/32
  - HOLD_VALIDITY (1260 groups, .05/1260)
  - precedence
- **Decision rules (V1 thresholds, no loosening):**
  - **CONFIRM → TRANSFER:** in each architecture, min over the seven fixed controls of Δ ≥ .01, and all 14 CI lower bounds > 0.
  - **KILL → NOT_TRANSFER:** any fixed-control CI upper bound ≤ 0, or min Δ < .002 in both architectures.
  - Otherwise **HOLD**.
- **Pre-GPU gates** (details in JSON):
  - **G1 download:** all 6 official zips, sha256 recorded, sizes matched. If a zip cannot be obtained, **STOP / INCOMPLETE_DATA** with no substitution.
  - **G2 list/file check:** missing plus undecodable files may be at most 0.5% per domain, otherwise **INVALID**. Labels must be consistent.
  - **G3 duplicates:** duplicates with conflicting labels mean **INVALID**.
  - **G4 support:** see section 4.
  - **G5 checkpoint/code binding:** missing means **INCOMPLETE**, a mismatch means **INVALID**.
  - **G6 PACS exact-identity overlap:** any overlap means **HOLD_VALIDITY**, with no silent deletion (V1 rule).
  - **G7 dry-check** on 64 images per domain or fewer. It checks shapes, label map, head order, and cached transform = V1 transform. It computes no ECA and no certificate.
- **After the run, INVALID checks:** V1 `audit_checks` with `n = floor(B/6)`, 162000 rows, 54000 primary rows, 1,512,000 flags.

## 6. DIFF vs S1 v1 (every difference)

| item | S1 v1 (INVALID) | S1 v2 | why |
|---|---|---|---|
| protocol_id | FEDCORE-S1-DOMAINNET-PACSCLS-20260923-v1 | FEDCORE-S1-DOMAINNET6-PACSCLS-20260925-v2 | new experiment |
| bootstrap seed | 2026092301 | 2026092501 | new registration |
| classes | 7 incl. person | 6, person excluded | person absent from DomainNet |
| unknown set size | 3 in every split | 3 in splits 1, 2; 2 in splits 0, 3, 4 | follows from the class change |
| known support | 4 known classes supported | splits 1, 2: person head gets no inputs (head kept intact) | frozen heads |
| secondary analysis | none specific | person-unknown splits {0,3,4} sensitivity; {1,2} descriptive; person-prediction diagnostic | asymmetry transparency |
| data version | unspecified | cleaned (official recommended), exact URLs, HEAD sizes | pre-download commitment |
| lists used | unspecified | train ∪ test | frozen predictors |
| archive binding | archive sha at gate | zip sha256 in DATA_BINDING addendum | same intent, made explicit |
| role rule | 60/20/20 with training unused ("may be excluded") | 50/50 proposal/audit, no training role | frozen predictors; keeps the 1:1 proposal:audit ratio; more support |
| J | 4 in text; S1 v1 left 6-domain allocation to "document before GPU" | J=6, V1 tail formulas at J=6 | 6 clients |
| per-domain draws | "equal allocation, document floor/remainder" | floor(B/6) = 42/85/170; remainder not drawn | explicit rule |
| support gate | nonempty per cell | ≥10 proposal and ≥10 audit per (class, domain) cell; any failure means whole-run INVALID | explicit minimum |
| data-failure tolerance | none stated | ≤0.5% missing/undecodable per domain | explicit |
| PACS overlap | not stated | exact-identity check; any overlap means HOLD_VALIDITY | V1 rule carried over |
| dry-check | "runner wiring dry-check" | G7 with explicit checks, no outcome computation | explicit |
| GPU hosts | ubuntu preferred | ubuntu GPU0, plus workstation4 GPU0 only if idle | resources |
| transfer labels | CONFIRM/KILL/HOLD | same, plus TRANSFER/NOT_TRANSFER/HOLD names | readability; thresholds identical |

Everything else (candidates, audit constants, procedures, metrics, CI, thresholds, precedence, denominators, transform) is **identical** to S1 v1 and V1.

## 7. Execution binding
- A thin v2 runner will import the unmodified V1 adapter, certifier, and analysis files. The V1 code sha256 values are checked, and the runner's sha256 is recorded in `RUN_BINDING.json` before full inference.
- Any divergence from this spec makes the run INVALID.
- GPU cap: 8 h across both hosts. Wall-clock GPU time is logged.
