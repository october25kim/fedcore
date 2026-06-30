"""Backward-compat shim.

``certificates.py`` was split into the ``fedcore.certificate`` package during the
structure-only refactor (cp / theorem1 / theorem3 / feasibility). This module re-exports the
identical names so existing ``from certificates import ...`` call sites keep working
unchanged. Shim is a LEAF: it imports only from fedcore.*, never the reverse.

Resolution: callers run with ``experiments/fedcore`` on ``sys.path`` (the script dir, or the
golden harness insert), so ``fedcore`` -> ``experiments/fedcore/fedcore``.
"""

from fedcore.certificate.cp import (  # noqa: F401
    _resolve_box_radius,
    _sample_lambdas,
    cp_lower,
    cp_upper,
)
from fedcore.certificate.theorem1 import (  # noqa: F401
    ConditionalCertificate,
    StratifiedCertificate,
    _inner_sup_over_a,
    conditional_risk_certificate,
    stratified_certificate,
)
from fedcore.certificate.theorem3 import pooled_cp, true_selective_risk  # noqa: F401
from fedcore.certificate.feasibility import thm2_floor  # noqa: F401

__all__ = [
    "cp_upper", "cp_lower", "_sample_lambdas", "_resolve_box_radius",
    "ConditionalCertificate", "_inner_sup_over_a", "conditional_risk_certificate",
    "StratifiedCertificate", "stratified_certificate",
    "pooled_cp", "true_selective_risk", "thm2_floor",
]
