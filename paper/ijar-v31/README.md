# FedCORE IJAR v31 public release

This release binds the IJAR v31 manuscript to the exact standalone figures and
aggregate/count-level sources used by its eight Figures, ten numbered Tables,
and two Abstract headline claims. It supersedes `paper/ijar-v28` only as the
current manuscript binding. Historical releases remain unchanged.

## Binding

- Manuscript version: v31
- Manuscript SHA-256: `e36c9cf9bfe30f4feb887ffb3ba2f6165d628404a99619ab2cf53f3ceafffdcb`
- Rendered PDF SHA-256: `7fe828ce760e9df1058267e66795a142e911d7e9d07d482010197654bd368897`
- Release tag: `v0.6.1`
- Ledger: 20 unique manuscript elements in `CLAIM_ARTIFACT.csv`

## Scope

The release reproduces count-to-decision calculations and publishes compact
aggregate sources for the claims listed in the ledger. It does not reproduce
model training, licensed datasets, checkpoints, record-level logits, or every
record-level analysis. The CIFAR-10.1 study uses source-disjoint images with
three reused checkpoints. It is not an independently trained-model replication
or a prospectively sealed confirmation.

## Verification

From the repository root, run:

```bash
python paper/ijar-v31/verify_release.py
make unit
```

The verifier checks the v31 binding, all package checksums, all 20 ledger IDs,
every governing artifact hash, every standalone Figure hash, and the explicit
supersession of the stale v18 PathMNIST source.
