# Fed-CORE v18 theorem-aligned numerical release

This directory binds the numerical claims in the clean v18 manuscript to a
versioned, hash-checked evidence package. The release supports independent
verification from frozen client-level counts to certification decisions and
the reported Tables and Figures.

The central question is simple. If future client traffic can change, which
predictions may be accepted while the accepted error rate remains below the
declared target? Fed-CORE answers this question with client-stratified audit
counts. This release makes those counts, the theorem contract, and the derived
results inspectable without releasing machine-readable observation-level image,
logit, label, or sample-identifier arrays. Figure 7 contains nine illustrative
PathMNIST patches from the public benchmark under its attribution license.

## Scope

The release verifies the following numerical chain:

1. A frozen proposal-defined family contains 12 selectors per cell. Each
   selector has a canonical metadata hash, and each cell-alpha family has an
   ordered hash over its 12 selector hashes.
2. Each selector and client contributes only an audit size `n`, an accepted
   count `A`, and an accepted-error count `K`.
3. Full-simplex family risk is tested by a clientwise intersection-union test
   followed by Holm correction over the 12 frozen members.
4. Coverage uses the theorem-aligned family allocation `delta_c / M`, without
   an additional client divisor.
5. The released counts regenerate the 450-cell headline comparison at
   `alpha = 0.20`, `delta_r = 0.05`, and `delta_c = 0.05`.

The headline procedures are:

- `H`: Holm/intersection-union risk decision with theorem-aligned coverage.
- `S`: simple simultaneous family decision with theorem-aligned coverage.
- `B`: the older clientwise `delta / (M J)` implementation, retained only as a
  conservative comparator.

At the headline operating point, `H`, `S`, and `B` certified 198, 198, and 148
of 450 cells. Their zero-imputed mean coverage lower bounds were 0.10049,
0.09695, and 0.07182, respectively.

## What this release does not claim

This package does not reproduce model training, checkpoints, raw `.npz`
exports, or the construction of the frozen empirical reservoirs. Those inputs
are not part of this public numerical release. The public claim is limited to
count-to-decision reproduction, validity summaries from frozen simulations,
and the numerical sources used by the v18 Tables and Figures.

The empirical-reservoir results are conditional on the frozen, source-level
deduplicated reservoirs. They are not claims about an unobserved deployment
population.

## Layout

- `contract/` defines the theorem and reporting contracts used by v18.
- `artifacts/counts/` contains the frozen 450-cell count tensor, the ordered
  selector-family manifest and hashes, and the theorem-aligned recertification
  export.
- `artifacts/noJ/` contains the three-procedure comparison and paired effects.
- `artifacts/validity/` contains the reservoir-validity and precision audits.
- `artifacts/audit_size/` contains the nested audit-size sensitivity results.
- `artifacts/phase_map/` and `artifacts/pathmnist/` provide the numerical
  sources for Figures 5 and 7.
- `artifacts/registry/` records the frozen training metadata and the actual
  450-cell analysis-input registry. The registry records inputs but does not
  make the unavailable raw artifacts part of this release.
- `reference/` contains the seven manuscript Figures and machine-readable
  copies of the four Tables.
- `PROVENANCE_MAP.json` maps every released numerical or reference artifact to
  its evidence role.
- `SHA256SUMS` binds every file in this directory except the manifest itself.

## Verification

From the repository root, run:

```sh
make reproduce-v18
```

The command fails closed on a missing or extra file, a hash mismatch, a count
invariant violation, a mismatch in the theorem-aligned headline results, an
invalid validity or precision summary, an inconsistent Figure 5 or PathMNIST
source, or a local absolute path embedded in a public text file.

See `REPRODUCE.md` for the complete acceptance criteria and `PRIVACY.md` for the
public-data boundary.
