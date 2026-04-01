"""
wavetank — Differentiable water caustic simulator in JAX.

Physics
-------
Wave field expanded in cosine eigenmodes of a rectangular tank.
Steady-state amplitudes computed analytically in frequency domain.
Caustic rendered via paraxial refraction + bilinear splatting + Gaussian blur.

Usage
-----
>>> from wavetank import Tank, Actuator, build_propagator
>>> from wavetank import steady_state_amplitudes, caustic_image
>>> from wavetank import optimize_caustic, analytical_solve, setup_from_target
"""

from .physics import (
    Tank,
    Actuator,
    Propagator,
    build_propagator,
    transfer_matrix,
    steady_state_amplitudes,
    pack_complex,
    unpack_complex,
)
from .render import (
    reconstruct_surface,
    caustic_image,
    snell_landing,
)
from .loss import (
    cosine_loss,
    ssim_loss,
    load_target_image,
)
from .optimize import (
    Stage,
    make_loss,
    optimize_caustic,
)
from .analytical import (
    analyze_target,
    analytical_solve,
    setup_from_target,
)

__all__ = [
    "Tank", "Actuator", "Propagator", "build_propagator",
    "transfer_matrix", "steady_state_amplitudes",
    "pack_complex", "unpack_complex",
    "reconstruct_surface", "caustic_image", "snell_landing",
    "cosine_loss", "ssim_loss", "load_target_image",
    "Stage", "make_loss", "optimize_caustic",
    "analyze_target", "analytical_solve", "setup_from_target",
]
