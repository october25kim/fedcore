# IJAR v34 theory contract

## Protected object

For a frozen selector, client acceptance probabilities `a_j`, accepted-error
masses `m_j`, and traffic mixture `lambda`, the protected quantity is

`R_sel(lambda) = sum_j lambda_j m_j / sum_j lambda_j a_j`,

with a strictly positive denominator. The credal target is the supremum over
the declared set `Lambda`. Empty positive-denominator domains fail closed.

## Representation boundary

Proposition 1 assumes positive client acceptance and a nonempty closed credal
set. A fixed traffic vector can represent the robust accepted-error target for
every client-risk profile if and only if the credal set is that singleton.
Consequently, a singleton contract reduces to its matched-mixture risk and the
full simplex reduces exactly to the largest client selective risk. This is an
estimand statement, not a license to apply pooled binomial inference.

## Inference contracts

- Theorem 1 gives the full-simplex fixed-selector decision. Its risk tail is
  not divided by the client count.
- Theorem 2 gives the positive-denominator robust program for a strict credal
  polytope using simultaneous client endpoints.
- Theorem 3 permits exact pooling only for independent observations drawn from
  one fixed known client mixture. Fixed heterogeneous client quotas generally
  produce Poisson-binomial accepted errors and do not satisfy this contract.
- Theorems 5 and 6 provide simple simultaneous and Holm/IUT family decisions
  for a proposal-frozen finite family. H and S are these valid full-simplex
  family procedures. B is a deliberately conservative same-target reference
  that additionally divides tails across clients.
- Theorem 7 applies only to the registered finite frame with preassigned atom
  permutations and registered prefixes. It does not imply population transport.

## Empirical interpretation

The target-matched replay assesses empirical coverage of the stated
fixed-selector contract. The clientwise Bonferroni curve is a valid conservative
same-target reference. Fixed-quota pooled curves and mismatch simulations are
mechanism diagnostics only; they are not competing full-simplex certificates
and do not support a procedure-superiority claim.
