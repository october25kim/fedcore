# FedCORE IJAR v32 public release

This release binds the IJAR v32 manuscript to the exact standalone figures and
aggregate/count-level sources used by its seven Figures, seven numbered Tables,
three Abstract evidence statements, and two retained quantitative prose claims.
It supersedes `paper/ijar-v31` only as the current manuscript binding.
Historical releases remain unchanged.

## Binding

- Manuscript version: v32
- Manuscript SHA-256: `d8087bd4b3d422d52a9d8475c374dfc973be8bd053ce4bf52a2bbf87cd9824c8`
- Rendered PDF SHA-256: `11628ff72104e44f894d4cdb412ed93eb673628b40f861fbf641ba00de23aad5`
- Release tag: `v0.6.2`
- Ledger: 19 unique manuscript elements in `CLAIM_ARTIFACT.csv`

## Scope

The release reproduces count-to-decision calculations and publishes compact
aggregate sources for the claims listed in the ledger. It does not reproduce
model training, licensed datasets, checkpoints, record-level logits, or every
record-level analysis. Table 1 is a literature-synthesis matrix and is pinned as
a textual artifact rather than as an empirical result. The CIFAR-10.1 study uses
source-disjoint images with three reused checkpoints. It is not an independently
trained-model replication or a prospectively sealed confirmation.

## Verification

From the repository root, run:

```bash
python paper/ijar-v32/verify_release.py
make unit
```

The verifier checks the v32 binding, all package checksums, all 19 ledger IDs,
every governing artifact hash, every standalone Figure hash, the explicit
supersession of the stale v18 PathMNIST source, and the numerical claims retained
in the v32 manuscript.
