# Sampling contracts

1. Fixed-size population or empirical-reservoir certification freezes the
   model, selector family, audit sizes, mixture set, draw law, and tie rules
   before certification.
2. Adaptive finite-frame certification assigns every atom an independent
   uniform random permutation before acquisition and registers all permitted
   prefixes, endpoint tails, and the frozen selector family. Predictable
   interleaving reveals only pre-existing prefixes.
3. Selection after adaptive stopping is restricted to the predeclared frozen
   family on the same finite frame and requires a positive exact acceptance
   denominator.
4. With-replacement reservoir draws target the frozen empirical reservoir and
   do not create new source labels. Finite-frame results do not transport to an
   unseen population.
5. Proposal, certification, and diagnostic evaluation folds remain separate.
6. Empty positive-denominator domains, invalid endpoints, and failed numerical
   solver checks return `cannot certify`.
