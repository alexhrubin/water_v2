"""
Non-ideal simulator: physically motivated perturbations of the ideal tank model.

Models four real-world deviations from the ideal rectangular-tank physics:

  1. Mode-dependent damping:  γ_j = γ₀ · (1 + α · ω_j / ω_max)
  2. Eigenfrequency jitter:   ω_j → ω_j · (1 + ε_j),  ε ~ N(0, σ²)
  3. Coupling noise:           C_ji → C_ji · (1 + δ_ji), δ ~ N(0, σ²)
  4. Radial camera distortion: Brown-Conrady model on the rendered image

Usage:
    prop = build_propagator(tank, actuators, n_modes, nx, ny)
    hyper = NonIdealHyperparams()
    config = sample_nonideal_config(prop, hyper, jax.random.PRNGKey(42))
    xs, ys, I = caustic_image_nonideal(prop, config, P, Omega, T)
"""

from dataclasses import dataclass, replace
import numpy as np
import jax
import jax.numpy as jnp

from .physics import Propagator, unpack_complex, steady_state_amplitudes
from .render import caustic_image


# ── Configuration ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class NonIdealHyperparams:
    """Distribution parameters controlling the severity of non-ideal perturbations."""
    damping_alpha: float = 0.01       # mode-dependent damping: γ_j = γ₀(1 + α·ω_j/ω_max)
    omega_sigma: float = 0.005        # eigenfrequency jitter (relative std)
    coupling_sigma: float = 0.02      # coupling noise (relative std)
    distortion_k1_range: float = 0.1  # barrel/pincushion distortion k1 ~ U(-r, r)
    distortion_k2_range: float = 0.01 # higher-order radial distortion k2 ~ U(-r, r)
    distortion_center_sigma: float = 0.02  # distortion center offset (relative to tank size)


@dataclass(frozen=True)
class NonIdealConfig:
    """A frozen, deterministic realization of non-ideal perturbations.

    All arrays are numpy — static with respect to JAX tracing.
    """
    gamma: np.ndarray     # per-mode damping γ_j           [n_total]
    omega: np.ndarray     # perturbed eigenfrequencies ω_j  [n_total]
    C: np.ndarray         # perturbed coupling matrix       [n_total, n_act]
    k1: float             # radial distortion coefficient
    k2: float             # higher-order radial distortion
    cx: float             # distortion center x (meters)
    cy: float             # distortion center y (meters)


# ── Sampling ──────────────────────────────────────────────────────────

def sample_nonideal_config(
    prop: Propagator,
    hyper: NonIdealHyperparams,
    key: jax.Array,
) -> NonIdealConfig:
    """Sample a random non-ideal configuration from hyperparameters.

    Parameters
    ----------
    prop  : Propagator (provides baseline omega, C, tank geometry)
    hyper : distribution parameters
    key   : JAX PRNG key (all randomness consumed here)

    Returns
    -------
    NonIdealConfig with deterministic numpy arrays.
    """
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    tank = prop.tank
    omega = prop.omega
    omega_max = omega.max()

    # 1. Mode-dependent damping: γ_j = γ₀ · (1 + α · ω_j / ω_max)
    gamma = tank.damping * (1.0 + hyper.damping_alpha * omega / omega_max)

    # 2. Eigenfrequency jitter: ω_j → ω_j · (1 + ε_j)
    eps = np.asarray(jax.random.normal(k1, shape=omega.shape)) * hyper.omega_sigma
    omega_perturbed = omega * (1.0 + eps)

    # 3. Coupling noise: C_ji → C_ji · (1 + δ_ji)
    delta = np.asarray(jax.random.normal(k2, shape=prop.C.shape)) * hyper.coupling_sigma
    C_perturbed = prop.C * (1.0 + delta)

    # 4. Radial camera distortion
    k1_val = float(jax.random.uniform(k3, minval=-hyper.distortion_k1_range,
                                       maxval=hyper.distortion_k1_range))
    k2_val = float(jax.random.uniform(k4, minval=-hyper.distortion_k2_range,
                                       maxval=hyper.distortion_k2_range))
    center_offset = np.asarray(jax.random.normal(k5, shape=(2,))) * hyper.distortion_center_sigma
    cx = tank.Lx / 2 + center_offset[0] * tank.Lx
    cy = tank.Ly / 2 + center_offset[1] * tank.Ly

    return NonIdealConfig(
        gamma=gamma,
        omega=omega_perturbed,
        C=C_perturbed,
        k1=k1_val,
        k2=k2_val,
        cx=float(cx),
        cy=float(cy),
    )


# ── Perturbed physics ────────────────────────────────────────────────

def make_nonideal_propagator(
    prop: Propagator,
    config: NonIdealConfig,
) -> Propagator:
    """Shallow copy of Propagator with omega and C replaced from config."""
    return replace(prop, omega=config.omega, C=config.C)


def nonideal_transfer_matrix(
    omega_modes: np.ndarray,
    Omega_drive: jnp.ndarray,
    gamma: np.ndarray,
) -> jnp.ndarray:
    """Transfer matrix with per-mode damping.

    H[j,k] = 1 / (ω_j² - Ω_k² + 2i·γ_j·ω_j·Ω_k)

    Parameters
    ----------
    omega_modes : natural frequencies ω_j  [n_total]
    Omega_drive : driving frequencies Ω_k  [n_freq]
    gamma       : per-mode damping ratios  [n_total]
    """
    w = jnp.asarray(omega_modes)[:, None]    # [n_total, 1]
    O = jnp.asarray(Omega_drive)[None, :]    # [1, n_freq]
    g = jnp.asarray(gamma)[:, None]          # [n_total, 1]
    return 1.0 / (w**2 - O**2 + 2j * g * w * O)


def nonideal_steady_state_amplitudes(
    prop: Propagator,
    config: NonIdealConfig,
    P: jnp.ndarray,
    Omega_freqs: jnp.ndarray,
    T: float,
) -> jnp.ndarray:
    """Steady-state modal amplitudes with perturbed omega, C, and damping.

    a = Im( H ⊙ (C_perturbed @ P) @ exp(iΩT) )

    Parameters
    ----------
    prop        : Propagator (used only for structure, not omega/C/damping)
    config      : NonIdealConfig with perturbed arrays
    P           : complex phasor matrix [n_act, n_freq]
    Omega_freqs : driving angular frequencies [n_freq]
    T           : evaluation time (s)
    """
    H = nonideal_transfer_matrix(config.omega, Omega_freqs, config.gamma)
    alpha = H * (jnp.asarray(config.C) @ P)
    E = jnp.exp(1j * Omega_freqs * T)
    return jnp.imag(alpha @ E)


# ── Camera distortion ────────────────────────────────────────────────

def radial_distort(
    I: jnp.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    config: NonIdealConfig,
) -> jnp.ndarray:
    """Apply radial camera distortion (Brown-Conrady model) to a caustic image.

    For each output pixel (x, y), compute the undistorted source coordinate:
        r² = (x - cx)² + (y - cy)²
        x_src = x + (x - cx)(k1·r² + k2·r⁴)
        y_src = y + (y - cy)(k1·r² + k2·r⁴)
    then bilinearly interpolate from the original image.

    Differentiable w.r.t. I via standard JAX autodiff.
    """
    # If no distortion, skip
    if config.k1 == 0.0 and config.k2 == 0.0:
        return I

    nx, ny = I.shape
    # Physical coordinate grids
    X, Y = jnp.meshgrid(jnp.asarray(xs), jnp.asarray(ys), indexing='ij')  # [nx, ny]

    # Radial distance from distortion center
    dx = X - config.cx
    dy = Y - config.cy
    r2 = dx**2 + dy**2
    radial_factor = config.k1 * r2 + config.k2 * r2**2

    # Undistorted source coordinates (in physical space)
    x_src = X + dx * radial_factor
    y_src = Y + dy * radial_factor

    # Convert physical coordinates to pixel indices (fractional)
    dx_grid = xs[1] - xs[0] if len(xs) > 1 else 1.0
    dy_grid = ys[1] - ys[0] if len(ys) > 1 else 1.0
    ix = (x_src - xs[0]) / dx_grid
    iy = (y_src - ys[0]) / dy_grid

    # Bilinear interpolation via map_coordinates
    # map_coordinates expects [ndim, ...] coordinates
    coords = jnp.stack([ix, iy], axis=0)
    return jax.scipy.ndimage.map_coordinates(I, coords, order=1, mode='nearest')


# ── Full non-ideal forward model ─────────────────────────────────────

def caustic_image_nonideal(
    prop: Propagator,
    config: NonIdealConfig,
    P: jnp.ndarray,
    Omega_freqs: jnp.ndarray,
    T: float,
    *,
    n_water: float = 1.33,
    sigma: float = 0.0,
    cutoff_sigmas: float = 4.0,
    full_snell: bool = False,
) -> tuple[np.ndarray, np.ndarray, jnp.ndarray]:
    """Full non-ideal caustic rendering: perturbed physics + camera distortion.

    Pipeline:
      P → nonideal_steady_state → a → caustic_image(perturbed_prop) → I → radial_distort → I_ni

    Differentiable w.r.t. P.
    """
    ni_prop = make_nonideal_propagator(prop, config)
    a = nonideal_steady_state_amplitudes(prop, config, P, Omega_freqs, T)
    xs, ys, I = caustic_image(ni_prop, a, n_water=n_water, sigma=sigma,
                               cutoff_sigmas=cutoff_sigmas, full_snell=full_snell)
    I_distorted = radial_distort(I, xs, ys, config)
    return xs, ys, I_distorted


# ── Training data utilities ──────────────────────────────────────────

def sample_random_phasors(
    key: jax.Array,
    n_act: int,
    n_freq: int,
    scale: float = 1.0,
) -> jnp.ndarray:
    """Sample random phasors as a flat real parameter vector.

    Returns params = [vec(X); vec(Y)] with X, Y ~ N(0, scale²).
    """
    return scale * jax.random.normal(key, shape=(2 * n_act * n_freq,))


def generate_training_pair(
    prop: Propagator,
    config: NonIdealConfig,
    params: jnp.ndarray,
    Omega_freqs: jnp.ndarray,
    T: float,
    *,
    n_water: float = 1.33,
    sigma: float = 0.0,
    cutoff_sigmas: float = 4.0,
    full_snell: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate an (I_ideal, I_nonideal) training pair for correction NN.

    Both images are rendered from the same phasor parameters.

    Returns
    -------
    I_ideal     : caustic image from ideal simulator [nx, ny]
    I_nonideal  : caustic image from non-ideal simulator [nx, ny]
    """
    n_freq = len(Omega_freqs)
    X, Y = unpack_complex(params, prop.n_act, n_freq)
    P = X + 1j * Y

    render_kw = dict(n_water=n_water, sigma=sigma,
                     cutoff_sigmas=cutoff_sigmas, full_snell=full_snell)

    # Ideal
    a_ideal = steady_state_amplitudes(prop, P, Omega_freqs, T)
    _, _, I_ideal = caustic_image(prop, a_ideal, **render_kw)

    # Non-ideal
    _, _, I_nonideal = caustic_image_nonideal(
        prop, config, P, Omega_freqs, T, **render_kw)

    return I_ideal, I_nonideal
