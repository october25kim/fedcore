# Retained non-governing records

This file distinguishes preserved lineage material from the evidence that
governs the IJAR v38 manuscript.

The sole claim-to-evidence authority for v38 is `CLAIM_ARTIFACT.csv`. Its 27
rows bind seven Figures, six Tables, four Abstract evidence statements, and ten
quantitative prose claims. A retained file that is not named by that ledger
does not support an additional manuscript claim.

The following categories are retained for transparency but are non-governing:

- exploratory and secondary files under `artifacts/analysis/` that are absent
  from `CLAIM_ARTIFACT.csv`;
- source-audit records under `artifacts/source_audit/`;
- CIFAR-10.1 aggregate and paired-comparison tables retained from earlier
  release lineage unless a v38 ledger row names them;
- superseded allocation-policy and finite-frame case-study artifacts;
- invalid diagnostic comparator rows, including Wilson and betting-oracle
  variants, which are never promoted to valid same-target baselines;
- the favorable DomainNet 1,024-draw analysis, which remains secondary and
  descriptive and cannot replace the registered 512-draw `KILL` decision;
- VLCS or DomainNet files beyond those named by the ledger, including
  descriptive contrasts and source-fidelity diagnostics;
- PACS execution records that establish provenance but are not themselves
  quantitative manuscript results; and
- test, build, and packaging evidence that establishes software integrity but
  does not create a scientific result.

Retention does not upgrade exploratory analysis to confirmatory evidence. It
does not turn a fixed empirical frame into a source-population sample, convert
the PACS fresh-seed rerun into untouched-dataset confirmation, or transform
released checkpoints into a completed independent clean-machine rerun.

Adverse evidence is not classified as non-governing merely because it is
unfavorable. The registered VLCS and DomainNet primary decisions and the PACS
`VALID_NEGATIVE` outcome govern whenever `CLAIM_ARTIFACT.csv` names them.

`SUPERSESSION.json` remains the explicit record for the corrected PathMNIST
result. The Office-Home 74-to-92 change is a change in the protected credal
set and information contract, not a same-target method comparison.
