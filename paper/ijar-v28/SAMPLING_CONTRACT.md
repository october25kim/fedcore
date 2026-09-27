# Sampling contracts

1. Fixed-size population or empirical-reservoir certification freezes the model,
   selector family, audit sizes, mixture set, and tie rules before certification.
2. Adaptive finite-frame certification freezes atom definitions, independent
   within-atom permutations, registered looks, endpoint tails, and the selector
   family. Only the next atom and stopping time are predictable.
3. With-replacement reservoir draws target the frozen empirical reservoir and do
   not create new source labels. Finite-frame results do not transport to unseen
   populations.
4. The proposal, certification, and diagnostic evaluation folds remain separate.
5. An empty positive-denominator domain, failed endpoint, or failed solver check
   returns `cannot certify`.
