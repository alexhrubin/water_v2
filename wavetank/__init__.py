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
    pearson_loss,
    ssim_loss,
    load_target_image,
)
from .optimize import (
    Stage,
    make_loss,
    optimize_caustic,
    surface_validity_report,
    make_hos_forward,
)
from .analytical import (
    analyze_target,
    analytical_solve,
    setup_from_target,
)
from .feasibility import (
    feasibility_report,
    print_feasibility_report,
)
from .surface_solver import (
    solve_target_surface,
    solve_target_surface_mesh,
    project_to_modes,
)
from .nonideal import (
    NonIdealHyperparams,
    NonIdealConfig,
    sample_nonideal_config,
    make_nonideal_propagator,
    caustic_image_nonideal,
    sample_random_phasors,
    generate_training_pair,
)
from .correction import (
    CorrectionUNet,
    apply_correction,
    generate_training_data,
    train_correction,
    make_corrected_loss,
    save_model,
    load_model,
)
from .hos import (
    HOSConfig,
    hos_forward,
    steady_state_initial,
)
from .inverse_render import (
    caustic_from_eta,
    optimize_eta_for_target,
    optimize_modal_eta_for_target,
)

__all__ = [
    "Tank", "Actuator", "Propagator", "build_propagator",
    "transfer_matrix", "steady_state_amplitudes",
    "pack_complex", "unpack_complex",
    "reconstruct_surface", "caustic_image", "snell_landing",
    "cosine_loss", "pearson_loss", "ssim_loss", "load_target_image",
    "Stage", "make_loss", "optimize_caustic", "surface_validity_report",
    "make_hos_forward",
    "analyze_target", "analytical_solve", "setup_from_target",
    "feasibility_report", "print_feasibility_report",
    "solve_target_surface", "solve_target_surface_mesh", "project_to_modes",
    "NonIdealHyperparams", "NonIdealConfig",
    "sample_nonideal_config", "make_nonideal_propagator",
    "caustic_image_nonideal",
    "sample_random_phasors", "generate_training_pair",
    "CorrectionUNet", "apply_correction",
    "generate_training_data", "train_correction",
    "make_corrected_loss", "save_model", "load_model",
    "HOSConfig", "hos_forward", "steady_state_initial",
    "caustic_from_eta", "optimize_eta_for_target",
    "optimize_modal_eta_for_target",
]
