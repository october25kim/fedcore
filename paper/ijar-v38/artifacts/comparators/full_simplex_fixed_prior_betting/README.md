# Full-Simplex Fixed-Prior Betting Comparator

This directory reports a same-target comparison. The predictor, full-simplex protection set, 12 proposal-frozen selectors, realized audit counts, risk and acceptance budgets, Holm family correction, and acceptance lower-confidence rule are held fixed.

At alpha 0.20 over the fixed 450-condition roster:

| Procedure | Certified | Mean ECA |
| --- | ---: | ---: |
| H, exact clientwise binomial test | 177/450 | 8.3424% |
| Fixed-prior betting | 81/450 | 3.9676% |
| Hoeffding | 29/450 | 1.7965% |
| Empirical Bernstein | 12/450 | 0.8546% |

H has higher ECA in 160 conditions, lower ECA in none, and the same ECA in 290. These are paired transformations of a fixed roster, not independent population draws. The result concerns this registered fixed-prior betting construction and does not establish dominance over every valid betting procedure.

`wilson` and `betting-oracle` are retained in the source tables as explicitly invalid diagnostic comparators and are not used as valid baselines.

Files:

- `contract.json`: common comparison contract and validity labels.
- `per_cell_eca.csv`: paired condition-level outputs.
- `summary.csv`: aggregate counts and ECA by dataset, alpha, and procedure.
