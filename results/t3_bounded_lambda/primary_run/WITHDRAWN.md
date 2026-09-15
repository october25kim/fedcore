# WITHDRAWN — do not cite the numbers in this directory

This run is the **original** T3 bounded-lambda replay. Its bounded endpoint allocation
omitted the client factor J, so it did not satisfy the joint accounting its own theorem
requires. Its headline values — 248 of 450 certified at alpha = 0.20, and 1168 of 2700
across the alpha grid — are **withdrawn** and are not the values reported in the
manuscript.

The valid replacement is the corrected fixed-traffic replay:

    results/fk_t3_uncertainty_e6/t3_corrected_fixed_traffic.py
    results/fk_t3_uncertainty_e6/imported_corrected/

which allocates delta_r / (J * M) and delta_c / (2 * J * M) and gives

| alpha | simplex | bounded |
|---|---|---|
| 0.05 | 0 | 0 |
| 0.10 | 21 | 21 |
| 0.15 | 81 | 104 |
| 0.20 | 177 | **199** |
| 0.25 | 256 | 288 |
| 0.30 | 343 | 358 |

The corrected replay reuses the traffic realisation recorded in this directory's
`t3_cells.csv` (sha256 fa0fef6032562e819e485a0766df9079af4bafd666248f5a9784165c1bfbab01),
which is why this directory is retained rather than deleted: it is the pinned input to
the correction, not a competing result.

Reproduced on this server on 2026-09-15; all six alphas matched exactly.
