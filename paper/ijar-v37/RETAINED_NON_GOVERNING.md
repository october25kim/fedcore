# Retained non-governing records

This file distinguishes preserved lineage material from the evidence that
governs the IJAR v37 manuscript.

The sole claim-to-evidence authority for v37 is `CLAIM_ARTIFACT.csv`. Its 24
rows bind seven Figures, six Tables, three Abstract evidence statements, and
eight quantitative prose claims. A retained file that is not named by that
ledger does not support an additional manuscript claim.

The following categories are retained for transparency but are non-governing:

- exploratory and secondary analysis files under `artifacts/analysis/` that
  are absent from `CLAIM_ARTIFACT.csv`;
- source-audit records under `artifacts/source_audit/`;
- CIFAR-10.1 aggregate and paired-comparison tables retained from the earlier
  release lineage;
- superseded allocation-policy and finite-frame case-study artifacts;
- PACS execution records that establish provenance but are not themselves
  quantitative manuscript results.

Retention does not upgrade exploratory analysis to confirmatory evidence. It
also does not change the registered PACS denominator, recover refused truth,
or broaden the reproduction boundary. `SUPERSESSION.json` remains the explicit
record for the corrected PathMNIST result.
