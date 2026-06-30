"""Fed-CORE certificate core (CP primitives + Theorem 1/1', stratified, Theorem 3, Thm-2 floor).

Re-exports the public + (used-elsewhere) private names so callers can
``from fedcore.certificate import conditional_risk_certificate`` etc.
"""

from .cp import _resolve_box_radius, _sample_lambdas, cp_lower, cp_upper
from .theorem1 import (
    ConditionalCertificate,
    StratifiedCertificate,
    _inner_sup_over_a,
    conditional_risk_certificate,
    stratified_certificate,
)
from .theorem3 import pooled_cp, true_selective_risk
from .feasibility import thm2_floor

__all__ = [
    "cp_upper", "cp_lower", "_sample_lambdas", "_resolve_box_radius",
    "ConditionalCertificate", "_inner_sup_over_a", "conditional_risk_certificate",
    "StratifiedCertificate", "stratified_certificate",
    "pooled_cp", "true_selective_risk", "thm2_floor",
]
