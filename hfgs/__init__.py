"""hfgs -- half-shifted Fourier inversion for Gerber-Shiu functionals.

Reference implementation accompanying

    "Half-shifted Fourier inversion for finite-horizon Gerber-Shiu
     functionals: fourth-order accuracy at first-order cost, exact
     sensitivities, and applications to credit, capacity planning and
     differentiable design."

    https://github.com/NoEdgeAtLife/hfgs
"""
from .core import (          # noqa: F401
    SNLP,
    grid_scheme,
    midpoints,
    finite_horizon,
    euler_invert,
    gaver_digits,
    boundary_from,
    threshold,
    dLxi,
    dLxi_dc,
    dLxi_dlam,
)

__version__ = "1.0.0"
__all__ = [
    "SNLP", "grid_scheme", "midpoints", "finite_horizon", "euler_invert",
    "gaver_digits", "boundary_from", "threshold", "dLxi", "dLxi_dc",
    "dLxi_dlam",
]
