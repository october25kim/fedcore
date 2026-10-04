# Fed-CORE Theorem-Aligned Frozen-Reservoir Sampling Contract

Contract ID `fedcore-headline-wr-v3`

## Protected target

The primary analysis protects the empirical distribution of each frozen client certification reservoir. It does not claim validity for an unseen deployment population. One reservoir atom is one source-unique image record. For PathMNIST, this identity is an image-record identity rather than a patient identity.

## Frozen design

- The campaign contains 450 cells. It includes 150 CIFAR-10 cells, 150 CIFAR-100 cells, 100 Office-Home cells, and 50 PathMNIST cells.
- Each cell retains its proposal-defined family of 12 selectors. Certification and evaluation outcomes do not redefine, reorder, or tune this family.
- The primary target is an accepted-error rate of 0.20. The risk-tail and coverage-tail budgets are both 0.05.
- The declared traffic set is the full simplex over clients.
- The complete alpha grid is 0.05, 0.10, 0.15, 0.20, 0.25, and 0.30.

## Primary draw

For client (j), let (N_j) be the number of source-unique records in its frozen certification reservoir. The primary audit draws exactly (n_j=N_j) indices independently and uniformly with replacement from that reservoir.

The generator is NumPy PCG64. The client stream is initialized by `SeedSequence([frozen_primary_audit_seed, client_id, 0])`. CIFAR and Office-Home use `seed_primary_audit_draw` from the frozen Confirmatory-400R matrix. PathMNIST uses `seed_audit` from its frozen matrix.

Duplicate indices are retained with their draw multiplicity. They are independent draws from the frozen empirical law, but they are not additional source-level labels. The same realized client index vector is used for every selector, every alpha value, and all H, S, and B procedures. This common-random-number design makes the procedure contrasts paired.

The headline uses one frozen primary draw per cell. An average over repeated draws is not a finite-sample certificate and is not used as the headline.

## Procedures

H uses the full-simplex intersection-union risk test followed by Holm adjustment over the 12 frozen family members. Its coverage lower bound uses a tail budget of (delta_c/M). H reports a risk decision and an adjusted p value rather than a numerical risk upper confidence bound.

S uses simple simultaneous family bounds. Its risk and coverage tail budgets are (delta_r/M) and (delta_c/M). It does not divide either budget by the number of clients.

B is a deliberately conservative allocation ablation. It divides the risk and coverage tail budgets by both the family size and the number of clients. It is not presented as a prior method.

Every procedure receives the same accepted counts, accepted-error counts, and audit sizes. A B-certified cell must therefore also be S-certified.

## Validation gates

- SHA-256 pins cover all 450 raw NPZ files and all 450 terminal sidecars.
- The record-level reconstruction reproduced all 154,800 archived as-is count entries before the WR headline was accepted.
- The release contains 154,800 WR candidate-count rows, 8,100 cell-alpha-procedure rows, and 2,150 client-reservoir accounting rows.
- H, S, and B used one identical count-tensor hash within every cell and alpha value.
- No evaluation or test fold was accessed.
- The complete output checksum set and an independent local validator passed.

## Interpretation of uncertainty

The certificate confidence is supplied by the finite-sample procedure for the single frozen primary draw. Separately, paired hierarchical bootstrap intervals use 20,000 replicates to describe variation across frozen class-split and training-repetition blocks. Those intervals are not theorem-confidence intervals and do not extend the protected target beyond the frozen empirical reservoirs.

