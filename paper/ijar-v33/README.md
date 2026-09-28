# FedCORE IJAR v33 public release

This release binds the IJAR v33 manuscript to the exact standalone figures and
aggregate/count-level sources used by its seven Figures, seven numbered Tables,
three Abstract evidence statements, and seven retained quantitative prose
claims. It supersedes `paper/ijar-v32` only as the current manuscript binding.
Historical releases remain unchanged.

## Binding

- Manuscript version: v33
- Manuscript SHA-256: `2f6afdb7b22a78ed0ba38a9920789c84482cd140a272c7ea0ce47402a5094617`
- Rendered PDF SHA-256: `345a869ddf8e99992c347dfa7c733593961bc8124601d2d30bf60b29857654af`
- Release tag: `v0.6.3`
- Ledger: 24 unique manuscript elements in `CLAIM_ARTIFACT.csv`

## Scope

The release reproduces count-to-decision calculations and publishes compact
aggregate sources for the claims listed in the ledger. The v33 evidence contract
adds the AUROC diagnostic, hierarchical-bootstrap records, retrospective audit
planning frontier, matched Office-Home comparator audit, and CIFAR-10.1 source
audit. It does not reproduce model training, licensed datasets, checkpoints,
record-level logits, or every record-level analysis. Planning and bootstrap
artifacts retain their retrospective/descriptive boundaries. The CIFAR-10.1
study reuses three checkpoints and is not an independently trained-model
replication or a prospectively sealed confirmation.

## Denominator contract

Figure 2 evaluates positive-acceptance validity on 443 proposal-feasible
conditions. Seven proposal-infeasible reject-all placeholders are excluded from
that validity denominator but remain in the fixed 450-condition utility roster.

## Verification

From the repository root, run:

```bash
python paper/ijar-v33/verify_release.py
make unit
```

The verifier checks the v33 binding, all package checksums, all 24
ledger IDs, every governing artifact hash, every standalone Figure hash, the
443/7 validity split, the explicit supersession of the stale v18 PathMNIST
source, and all numerical/evidence claims retained in the v33 manuscript.
