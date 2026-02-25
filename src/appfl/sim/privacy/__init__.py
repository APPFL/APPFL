"""
This package implements differential privacy techniques.
"""

from .dp import laplace_mechanism_output_perturb, gaussian_mechanism_output_perturb
from .secure_aggregator import SecureAggregator

try:
    from .opacus_dp import make_private_with_opacus
except Exception:  # pragma: no cover
    make_private_with_opacus = None

__all__ = [
    "laplace_mechanism_output_perturb",
    "gaussian_mechanism_output_perturb",
    "make_private_with_opacus",
    "SecureAggregator",
]
