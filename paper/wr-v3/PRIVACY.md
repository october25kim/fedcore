# Privacy and release scope

Fed-CORE's statistical certificate can be computed from selector-indexed
stratum count triples `(A, K, n)`. Releasing these aggregates generally reveals
less information than releasing images, labels, logits, scores, or checkpoints.
It does not by itself provide cryptographic secure aggregation, differential
privacy, or immunity to inference attacks.

This numerical release contains the exact selector-indexed counts required to
reproduce the manuscript decisions. It excludes raw datasets, sample identifiers,
images, labels, logits, checkpoints, and audit index vectors. Selector and count
tensors are bound by SHA-256 values. The reservoir-accounting file reports only
non-sensitive draw size, uniqueness, duplication, multiplicity, and sequence-hash
fields.

The public verifier performs local count-to-decision recomputation. No artifact
is transmitted to a server by the verifier. In a deployment, any secure
aggregation, access control, retention policy, or differential-privacy mechanism
must be specified separately from the statistical guarantee.
