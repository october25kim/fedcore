"""Fail-closed registry for one-shot certificate implementations."""

from __future__ import annotations


JOINT_CONDITIONAL_CERTIFICATE_VARIANT = "joint_conditional_v1"

# A plan may name a variant only after an implementation is registered here and
# wired by the post-hoc runner. Mixture geometry is a separate plan dimension;
# the joint conditional implementation supports all three exact mixture modes.
CERTIFICATE_VARIANT_MIXTURE_MODES = {
    JOINT_CONDITIONAL_CERTIFICATE_VARIANT: frozenset({"simplex", "rho", "traffic"})
}
SUPPORTED_CERTIFICATE_VARIANTS = frozenset(CERTIFICATE_VARIANT_MIXTURE_MODES)


def validate_certificate_variant(
    variant: object, mixture_mode: str | None = None
) -> str:
    """Return a registered variant or fail before an artifact can be created."""

    if not isinstance(variant, str) or variant not in SUPPORTED_CERTIFICATE_VARIANTS:
        supported = ", ".join(sorted(SUPPORTED_CERTIFICATE_VARIANTS))
        raise ValueError(
            f"unsupported certificate_variant {variant!r}; "
            f"implemented variants: {supported}"
        )
    value = str(variant)
    if (
        mixture_mode is not None
        and mixture_mode not in CERTIFICATE_VARIANT_MIXTURE_MODES[value]
    ):
        raise ValueError(
            f"certificate_variant {value!r} does not support "
            f"mixture_mode {mixture_mode!r}"
        )
    return value


__all__ = [
    "CERTIFICATE_VARIANT_MIXTURE_MODES",
    "JOINT_CONDITIONAL_CERTIFICATE_VARIANT",
    "SUPPORTED_CERTIFICATE_VARIANTS",
    "validate_certificate_variant",
]
