# Reference reinforcement for Information Sciences submission

Target: Elsevier *Information Sciences* (INS) recommends that a meaningful share of the
reference list (~20%) come from the target journal. The paper currently has 21 references;
~20% means about 4 should be INS papers. **Two are already added and cited in the draft;**
this list gives those plus a vetted recipe to add ~2 more. Per the project honesty rule,
only papers confirmed in a real search are listed as confirmed; everything else is a search
recipe, not a fabricated citation. Confirm authors/volume/pages on ScienceDirect before
camera-ready.

INS papers are identifiable by the PII prefix **S0020025** (ISSN 0020-0255).

## Already added to the draft (cited in Related Work)

| Ref | Title | Venue | PII / locator | Status |
|---|---|---|---|---|
| [19] | Adversarial compact wrapping classifier learning for open set recognition | Information Sciences (2024) | S0020025524010284 | Confirmed INS; verify authors/vol/pages |
| [20] | Towards heterogeneous federated graph learning via structural entropy and prototype aggregation | Information Sciences 718 (2025) 122338 | art. 122338 | Confirmed INS; verify authors |

Adjacent Elsevier (not INS, but directly on-topic — keep if useful, does not count toward the INS share):

| Ref | Title | Venue | PII |
|---|---|---|---|
| [21] | Classification with reject option: Distribution-free error guarantees via conformal prediction | Machine Learning with Applications (2025) | S2666827025000477 |

## Add ~2 more INS papers — ScienceDirect search recipe

Search ScienceDirect with **Journal = "Information Sciences"** and the queries below; pick the
2 most relevant recent (≤3 years) hits, one from each cluster, and confirm the PII starts with
S0020025.

1. Federated learning under heterogeneity (to pair with the certificate's non-IID motivation):
   - "Information Sciences" federated learning non-IID client heterogeneity
   - "Information Sciences" personalized federated learning aggregation
   - "Information Sciences" federated learning privacy secure aggregation
2. Uncertainty / selective prediction / reject option (to pair with the certificate's object):
   - "Information Sciences" selective classification reject option
   - "Information Sciences" classifier confidence calibration reliability
   - "Information Sciences" conformal prediction distribution-free
   - "Information Sciences" out-of-distribution detection rejection

Practical filter on the results page: ScienceDirect → refine by "Publication title: Information
Sciences" + "Years: 2023–2026". Take the top relevant article from cluster 1 and cluster 2.

## How to wire them in

- Add each as [22], [23] in the reference list (same Elsevier format as [19]/[20]).
- Cite [22] (federated heterogeneity) in Section 2 next to the existing FedOSR/heterogeneity
  discussion, and [23] (selective prediction / reject option) in Section 2 next to the
  centralized selective-risk paragraph — so they are genuinely used, not list padding.
- After adding, recount: 4 INS refs / ~23 total ≈ 17–20% — on target.

## If the venue switches to Pattern Recognition instead

PR also wants ~20% PR references; these were confirmed in search and are PR (PII S0031320):

| Title | Venue | PII |
|---|---|---|
| Verifiably robust conformal prediction for probabilistic guarantees under adversarial attacks | Pattern Recognition (2025) | S0031320325007113 |
| (PR Special Issue) Conformal Prediction and Distribution-Free Uncertainty Quantification | Pattern Recognition (special issue) | — |

PR has an active conformal-prediction / OSR footprint, so reaching the 20% PR share is easy;
INS reaching 20% needs the two extra pulls above.

## Status (updated)

- In-text citations across the whole draft are now numeric **[n]** (Information Sciences style);
  the reference list is bracketed [1]–[21]. The unverifiable "Joint Certificate" entry was
  dropped and the duplicate SCRC entry merged.
- **Confirmed INS papers in the draft and cited in Section 2:** [19] (OSR, S0020025524010284),
  [20] (federated graph learning, Inf. Sci. 718 (2025) 122338). Adjacent Elsevier: [21]
  (Mach. Learn. Appl.).
- **Two more INS papers still needed** to reach ~20%. I re-ran multiple web searches and tried
  to pull ScienceDirect/dblp listings directly, but ScienceDirect search and the dblp API are
  JavaScript-gated / returned empty to the automated fetch, so I could **not** machine-confirm
  two additional specific INS papers in-session without guessing. Per the no-fabrication rule I
  did not invent them.

## How to finish (you have ScienceDirect access)

1. ScienceDirect → search box, then refine **Publication title = "Information Sciences"**,
   **Years 2023–2026**. Run two queries and take the top relevant hit from each:
   - cluster 1 (federated): `federated learning non-IID heterogeneity`
   - cluster 2 (selective/uncertainty): `selective classification reject option` or
     `conformal prediction distribution-free`
2. Confirm each chosen article's URL contains **/pii/S0020025...** (that is the INS fingerprint).
3. Paste the two titles + PIIs back to me; I will insert them as [22], [23] in the reference
   list (same Elsevier format) and cite [22] in the federated-heterogeneity sentence and [23] in
   the selective-risk sentence of Section 2, then rebuild the docx. Target after: 4 INS / ~23 ≈ 17–20%.

A concrete lead worth checking first: a survey cited an INS federated-learning paper as
"Information Sciences 667 (2024) 120482" — confirm its title/PII on ScienceDirect; if relevant it
is a ready cluster-1 candidate.

## Honesty note
I confirmed [19], [20] (INS) and [21] (Mach. Learn. Appl.) directly in web search. I did **not**
invent any further INS citation — the remaining two slots are a search recipe for you to fill
from ScienceDirect, because ScienceDirect/dblp are JS-gated for the automated tools in-session.
