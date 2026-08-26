"""Retired invalid experiment retained only as an explicit fail-closed shim.

The former script treated a finite numerical search as a proof that ordinary
binomial Clopper--Pearson calibration covers a heterogeneous
Poisson--binomial mean.  That conclusion is not established and must not be
used to justify pooled certification.  The supported pooled certificate in
``fedcore.certificate.theorem3`` is restricted to a matched-mixture i.i.d.
audit, where the pooled accepted-error count is genuinely binomial.

This module deliberately exits non-zero so old automation cannot silently
recreate or report the withdrawn claim.
"""

from __future__ import annotations


MESSAGE = (
    "RETIRED: exp_lemma_L did not establish its claimed theorem. "
    "Use exp_pooling_fail for the heterogeneity counterexample and apply pooled "
    "CP only under a matched-mixture i.i.d. audit."
)


def main() -> None:
    raise SystemExit(MESSAGE)


if __name__ == "__main__":
    main()
