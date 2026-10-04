# FedCORE IJAR v38 evidence-upgrade release

This release candidate binds the IJAR v38 manuscript to the exact aggregate,
count-level, external fixed-frame, and PACS execution evidence used by seven
Figures, six numbered Tables, four Abstract evidence statements, and ten
quantitative prose claims. It supersedes `paper/ijar-v37` as the governing
manuscript binding when tag `v0.8.0` is published. Historical releases and the
execution-time PACS protocol identifiers remain unchanged.

## Binding

- Manuscript version: v38
- Manuscript file: `manuscript/01_FedCORE_IJAR_Manuscript_v38_EvidenceUpgrade.docx`
- Manuscript SHA-256: `5f7762730782a44664a331d24fce76e6efd5a8f13d5fc63dd7124bf4b5d6b2de`
- Rendered PDF file: `manuscript/01_FedCORE_IJAR_Manuscript_v38_EvidenceUpgrade.pdf`
- Rendered PDF SHA-256: `10f0509b984586226506b5238feeb81a6bc00c6781e64fdcca991c11bf5b6312`
- Intended release tag: `v0.8.0`
- PACS execution source commit: `fa016c199a7c846bbe20b64d54ab34164adf7647`
- Evidence asset: `FedCORE_IJAR_v38_evidence_release.zip`
- Ledger: 27 unique manuscript elements in `CLAIM_ARTIFACT.csv`

The manuscript and rendered PDF hashes above are the final v38 identities.
The verifier also requires the tagged checkout to resolve to `v0.8.0`.

## Scientific delta from v37

The v38 release retains the credal selective-risk decision as the
methodological spine and makes four evidence upgrades.

1. It reports a same-target valid comparison under the identical full-simplex
   credal set, predictors, 12-member proposal-frozen selector family, realized
   audit counts, confidence budgets, Holm family correction, and coverage
   rule. At alpha 0.20, the exact clientwise procedure H certified 177 of 450
   conditions with mean effective certified acceptance of 8.3424%, while the
   registered fixed-prior betting comparator certified 81 with 3.9676%.
   Hoeffding and empirical Bernstein certified 29 and 12 conditions,
   respectively. This comparison concerns the registered fixed-prior betting
   construction and does not establish dominance over every valid betting
   procedure.
2. It retains two adverse external fixed-frame evaluations. VLCS returned
   `KILL`; its full-simplex ConvNeXt-Tiny contrast was 0.164 percentage points,
   below the preregistered 1.00-point minimum. DomainNet was sealed before
   image download, decoding, and prediction, and returned `KILL` with transfer
   label `NOT_TRANSFER`. These complete-denominator results are evidence
   against a broad positive transport claim, not failed files to omit.
3. It adds a hash-bound PACS training package for all 30 models: exact source,
   immutable scientific image, five input identities, 30 checkpoints,
   execution receipts, complete denominators, and outcomes. The package
   enables exact-identity audit and a portable rerun, but this release does not
   claim that an independent clean-machine rerun has already been completed.
4. It states the uncertainty object explicitly: the client-weight credal set
   `Lambda` induces the deployment-distribution set
   `P_Lambda = {sum_j lambda_j P_j : lambda in Lambda}`. Sampling uncertainty
   in the audit and epistemic uncertainty about future traffic remain separate.

The Office-Home change from 74 to 92 certified conditions is classified as
target and information sensitivity. The protected set changed from the full
simplex to a traffic-informed credal set. It is not a same-target method gain.

## Evidence boundary

The release reproduces the registered count-to-decision calculations and the
aggregate sources named by `CLAIM_ARTIFACT.csv`. The external evaluations are
fixed-frame studies and do not prove source-population transport. VLCS was
locally frozen before training and prediction but was not externally
timestamped. DomainNet was sealed before its images were downloaded, decoded,
or scored, but its primary result was adverse.

The public PACS assets include exact source, the immutable scientific image,
and all 30 checkpoints. Raw PACS, VLCS, and DomainNet images are not
redistributed. Exact PACS replay therefore still requires the byte-identical
PACS input archive and the two pinned TorchVision initialization files. See
`reproduction/pacs_v3/README_E2E.md`, `RELEASE_ASSETS.sha256`, and
`THIRD_PARTY_NOTICES.md` before downloading or redistributing third-party
material. `RELEASE_ASSETS.json` and `THIRD_PARTY_RELEASE_BOUNDARY.md` provide
the release-level asset inventory and redistribution boundary.

The PACS campaign remains a prospectively sealed post-incident fresh-seed
rerun on the same hashed empirical frame. It is not an untouched-dataset
confirmation or an independent source-population replication. Its registered
result is adverse: H, S, and B each certified three of 30 models.

## Retained non-governing records

Historical, exploratory, secondary, and source-audit files are retained to
preserve lineage and adverse evidence. They do not expand the 27-element v38
governing ledger unless they are named by `CLAIM_ARTIFACT.csv`. The exact
categories are listed in `RETAINED_NON_GOVERNING.md`.

## Verification

From the repository root after the final tag is published, run:

```bash
python paper/ijar-v38/verify_release.py
make unit
```

The verifier checks the complete checksum manifest, the 7F+6T+4H+10C ledger,
manuscript/PDF binding, the 443-condition validity denominator, 450-condition
same-target analyses, the matched fixed-prior betting comparison, the adverse
VLCS and DomainNet decisions, the complete PACS denominator, independent
count-to-decision replay, and source/reproduction bindings. In a tagged
checkout it must require `HEAD` to resolve to `v0.8.0`.
