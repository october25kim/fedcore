# Reproducing the v18 numerical evidence

## One-command check

Use a Python environment with SciPy available. From the repository root, run:

```sh
make reproduce-v18
```

The command is read-only. It does not train a model, download a Dataset, or
modify any released artifact.

## Fail-closed acceptance criteria

The release is accepted only when every check below passes:

1. `SHA256SUMS` lists every release file except `SHA256SUMS` itself, lists no
   file outside `paper/v18`, and matches every byte.
2. The deterministic gzip count archive expands to the frozen CSV with SHA-256
   `3074c2d8039b64707bbf27524ebd8baa60548825758c89c77454efc9f0cb3f45`.
3. The count tensor contains exactly 154,800 rows, 450 semantic cells, six
   alpha values, 12 family members per cell and alpha, complete client support,
   and only counts satisfying `0 <= K <= A <= n`.
4. Selector metadata must be identical across all client rows of a candidate.
   The deduplicated family manifest must contain 32,400 ordered members. Every
   selector hash and all 2,700 cell-alpha ordered-family hashes must match an
   independent canonical reconstruction.
5. The verifier independently recomputes the `H`, `S`, and `B` decisions at
   `alpha = 0.20`, `delta_r = 0.05`, and `delta_c = 0.05`. The certified totals
   must be 198, 198, and 148. The corresponding zero-imputed mean coverage
   lower bounds must be 0.10049009894485754, 0.09694642128834831, and
   0.07182488653286874.
6. The dataset-level certified counts must be `66, 11, 88, 33` for `H` and
   `54, 1, 69, 24` for `B`, in the order CIFAR-10, CIFAR-100, Office-Home, and
   PathMNIST.
7. The released per-cell comparison must match the independently recomputed
   decisions and coverage lower bounds.
8. The reservoir-validity summary must contain 450 cells and three methods per
   cell. Fed-CORE must have no empirical validity value below 0.95 over 1,000
   repeats, while the intentionally invalid pooled comparator must have 443
   cells below 0.95.
9. The higher-precision audit must contain 20 summary cells and 200,000
   replicate rows. Its minimum empirical validity must be 0.9543, with no point
   estimate or exact 95% lower bound below 0.95.
10. The nested audit-size analysis must contain 240,000 replicate rows, preserve
   its registered count and nesting invariants, and reproduce the reported
   CIFAR-10 and CIFAR-100 endpoint values at 128 and 1,028 audits per client.
11. Every Figure 5 failure-anatomy row must account for 50 cells, and its
    certified component must agree with the theorem-aligned phase-map source.
    At `alpha = 0.20`, the CIFAR grid must contain 77 certified cells and 223
    refusals, of which 217 are attributed to count or interval width.
12. The PathMNIST source must contain ten split-primary rows and 12
    alpha-frontier rows. At `alpha = 0.20`, the two heterogeneity conditions
    must certify 16 and 17 cells, for a total of 33.
13. No public text file may contain a local absolute path.

The successful terminal line is:

```text
FEDCORE V18 THEOREM-ALIGNED RELEASE: PASS
```

Any exception, skipped headline check, or nonzero exit status means that the
release is not verified.

## Interpretation boundary

This command verifies frozen numerical evidence. It does not establish that
the model-training runs or raw reservoir construction can be reproduced from
this package. Training and raw-data reproduction require separate, licensed
inputs and are outside this release contract.
