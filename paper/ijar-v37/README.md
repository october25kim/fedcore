# FedCORE IJAR v37 public release

This release binds the IJAR v37 manuscript and rendered PDF to the exact
aggregate and count-level evidence used by seven Figures, six numbered Tables,
three Abstract evidence statements, and eight quantitative prose claims. It
supersedes `paper/ijar-v34` as the governing manuscript binding. Historical
releases and the execution-time PACS protocol identifiers remain unchanged.

## Binding

- Manuscript version: v37
- Manuscript SHA-256: `3cdf07988c670016e00be135bd93ee8e5d8477811b6f40284e0e945c0f703c4d`
- Rendered PDF SHA-256: `ef585347079765f70990312326a7c9e4107ecd29363b99e65d0a8fe8cd903699`
- Release tag: `v0.7.0`
- PACS source tag: `pacs-local-v1-fa016c1`
- Release asset: `FedCORE_IJAR_v37_evidence_release.zip`
- Ledger: 24 unique manuscript elements in `CLAIM_ARTIFACT.csv`

## Scientific delta from v34

The v37 manuscript makes the credal selective-risk decision the sole
methodological spine and removes the secondary finite-frame allocation case
study from the governing paper. It adds the complete registered PACS rerun:
30/30 terminal models, 90 H/S/B decisions, 1,080 independently replayed
candidate decisions with zero mismatch, and all adverse cells retained. Each
procedure certified three models and mean effective certified acceptance stayed
below 4%, so the PACS result is reported as a risk-gate-limited utility boundary
rather than procedure superiority.

## Evidence boundary

The release reproduces the registered count-to-decision analyses. It does not
reproduce licensed datasets, source image arrays, pretrained initialization
archives, checkpoints, full model training, or every record-level analysis.
The PACS rerun is a prospectively sealed post-incident fresh-seed rerun on the
same hashed empirical frame. It is not an untouched-dataset confirmation or an
independent source-population replication. The terminal receipt supports a
zero process return code; it does not independently prove a stopped-container
state.

## Retained non-governing records

Several historical analysis and source-audit files are retained to preserve
lineage and adverse evidence. They are not additional v37 manuscript claims
and do not expand the 24-element governing ledger. Their status and categories
are stated in `RETAINED_NON_GOVERNING.md`. Only artifacts named by
`CLAIM_ARTIFACT.csv` govern the quantitative claims in the v37 manuscript.

## Verification

From the repository root, run:

```bash
python paper/ijar-v37/verify_release.py
make unit
```

The release verifier checks the complete checksum manifest, the 7F+6T+3H+8C
ledger, manuscript/PDF binding, 443-condition validity evidence, 450-condition
same-count H/S/B analyses, the complete PACS denominator and candidate gates,
independent replay, final outcome gates, and exact source-tag binding.
