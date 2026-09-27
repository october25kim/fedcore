"""Theorem-facing FedCORE certification API.

Re-exports the public + (used-elsewhere) private names so callers can
``from fedcore.certificate import conditional_risk_certificate`` etc.
"""

from .cp import _resolve_box_radius, _sample_lambdas, cp_lower, cp_upper
from .theorem1 import (
    ConditionalCertificate,
    FullSimplexCertificate,
    StratifiedCertificate,
    _inner_sup_over_a,
    conditional_risk_certificate,
    full_simplex_fixed_member_certificate,
    stratified_certificate,
)
from .theorem3 import pooled_cp, pooled_cp_diagnostic, true_selective_risk
from .feasibility import thm2_floor
from .joint import JointCertificate, joint_conditional_certificate
from .lambda_sets import (
    ConservativeExtremumResult,
    NormalizedBox,
    solve_normalized_box_coverage,
    solve_normalized_box_risk,
)
from .family import (
    SimpleFamilyCertificate,
    SimpleFamilyMemberCertificate,
    select_simple_family_member,
    simple_simultaneous_family_certificate,
)
from .allocation import zero_error_floor


def zero_error_count_threshold(alpha: float, delta_r: float) -> int:
    """Minimum zero-error accepted count for a one-sided CP risk decision."""
    return zero_error_floor(delta_r, alpha)

__all__ = [
    "cp_upper",
    "cp_lower",
    "_sample_lambdas",
    "_resolve_box_radius",
    "ConditionalCertificate",
    "FullSimplexCertificate",
    "_inner_sup_over_a",
    "conditional_risk_certificate",
    "full_simplex_fixed_member_certificate",
    "StratifiedCertificate",
    "stratified_certificate",
    "pooled_cp",
    "pooled_cp_diagnostic",
    "true_selective_risk",
    "thm2_floor",
    "JointCertificate",
    "joint_conditional_certificate",
    "ConservativeExtremumResult",
    "NormalizedBox",
    "solve_normalized_box_coverage",
    "solve_normalized_box_risk",
    "SimpleFamilyCertificate",
    "SimpleFamilyMemberCertificate",
    "select_simple_family_member",
    "simple_simultaneous_family_certificate",
    "zero_error_count_threshold",
]
