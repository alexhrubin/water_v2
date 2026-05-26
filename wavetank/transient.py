"""Transient caustic optimization.

Replaces the steady-state phasor parametrization with piecewise-constant
time-varying actuator drive. Each actuator i has a real-valued amplitude
``theta[i, b]`` in each time bin ``b``; the apparatus is integrated
forward from rest under this drive using the same HOS physics as the
steady-state pipeline. The loss is evaluated on the caustic rendered
from the surface at ``T_eval = n_bins * dt_bin``.

Adjoint comes free from ``jax.lax.scan`` autodiff — no hand-coded backward
pass needed.

The optimization target is the same as the steady-state pipeline (caustic
intensity matched to a target image), but the reachable surface manifold
is larger because every transient configuration is admissible — not just
configurations holdable indefinitely.
"""

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from .physics import Propagator
from .render import caustic_image, reconstruct_surface, _gaussian_blur_separable
from .hos import HOSConfig, _scatter_2d, _project_to_modes


# ── Forward integrator (transient drive) ───────────────────────────────


def _bin_index(t: float, dt_bin: float, n_bins: int) -> jnp.ndarray:
    """Map time t to bin index, clipped to [0, n_bins-1]."""
    return jnp.clip(
        jnp.floor(t / dt_bin).astype(jnp.int32),
        0, n_bins - 1,
    )


def _rhs_M1_transient(a, b, t, omega, sigma_mode, gamma, g, C, theta, dt_bin, n_bins):
    """M=1 RHS with piecewise-constant time-varying drive.

    ȧ = σ b
    ḃ = -g a - 2γω b + R(t)
    R_j(t) = (g/ω_j²) · Σ_i C[j,i] · θ[i, bin(t)]
    """
    bin_idx = _bin_index(t, dt_bin, n_bins)
    f_t = theta[:, bin_idx]                  # [n_act] — current displacement
    F = C @ f_t                              # [n_total]
    R = (g / omega**2) * F                   # [n_total]

    da = sigma_mode * b
    db = -g * a - 2.0 * gamma * omega * b + R
    return da, db


def _rhs_M2_transient(
    a, b, t,
    omega, sigma_mode, gamma, g, C, theta, dt_bin, n_bins,
    prop, k2_flat, inv_N_flat, dx, dy, dealias_mask,
):
    """M=2 RHS — M=1 linear part + Dommermuth–Yue quadratic corrections.

    Mirrors ``wavetank.hos._rhs_M2`` exactly but with the transient
    forcing replacing the steady-state phasor sum.
    """
    da_lin, db_lin = _rhs_M1_transient(
        a, b, t, omega, sigma_mode, gamma, g, C, theta, dt_bin, n_bins,
    )

    a_in = jnp.where(dealias_mask, a, 0.0)
    b_in = jnp.where(dealias_mask, b, 0.0)

    a_2d   = _scatter_2d(a_in,                 prop)
    b_2d   = _scatter_2d(b_in,                 prop)
    sb_2d  = _scatter_2d(sigma_mode * b_in,    prop)        # for W₁
    k2b_2d = _scatter_2d(k2_flat * b_in,       prop)        # for W₂

    cx, cy   = jnp.asarray(prop.cos_x),  jnp.asarray(prop.cos_y)
    dcx, dcy = jnp.asarray(prop.dcos_x), jnp.asarray(prop.dcos_y)

    eta   = cx  @ a_2d   @ cy.T
    eta_x = dcx @ a_2d   @ cy.T
    eta_y = cx  @ a_2d   @ dcy.T
    psi_x = dcx @ b_2d   @ cy.T
    psi_y = cx  @ b_2d   @ dcy.T
    W1    = cx  @ sb_2d  @ cy.T
    W2    = cx  @ k2b_2d @ cy.T

    eta_W2      = eta * W2
    grad_dot    = psi_x * eta_x + psi_y * eta_y
    grad_psi_sq = psi_x ** 2 + psi_y ** 2
    W1_sq       = W1 ** 2

    proj = lambda f: _project_to_modes(f, prop, inv_N_flat, dx, dy)
    da_nl =  proj(eta_W2) -  proj(grad_dot)
    db_nl = -0.5 * proj(grad_psi_sq) + 0.5 * proj(W1_sq)

    return da_lin + da_nl, db_lin + db_nl


def hos_forward_transient(
    prop: Propagator,
    theta: jnp.ndarray,
    dt_bin: float,
    *,
    M: int = 2,
    config: HOSConfig | None = None,
    dt: float | None = None,
) -> jnp.ndarray:
    """Integrate HOS forward from rest under piecewise-constant drive.

    Parameters
    ----------
    prop   : Propagator
    theta  : actuator drive [n_act, n_bins], real-valued displacement per bin
    dt_bin : width of each time bin (s)
    M      : HOS order (1 or 2; 3 not implemented yet)
    config : optional HOSConfig (defaults to HOSConfig(M=M))
    dt     : integration timestep (auto-from-spectrum if None)

    Returns
    -------
    a_final : modal amplitudes [n_total] at T_eval = n_bins * dt_bin.
    """
    if M not in (1, 2):
        raise NotImplementedError(f"HOS M={M} not implemented (M=3 pending)")
    if config is None:
        config = HOSConfig(M=M)

    n_bins = theta.shape[1]
    T_eval = float(n_bins) * dt_bin

    omega = jnp.asarray(prop.omega)
    sigma_mode = omega ** 2 / prop.tank.g      # σ_j = k_j tanh(k_j d) = ω²/g
    gamma = prop.tank.damping
    g = prop.tank.g
    C = jnp.asarray(prop.C)

    # M=2 precomputations
    if M >= 2:
        Lx, Ly = prop.tank.Lx, prop.tank.Ly
        kx = prop.mode_m * np.pi / Lx
        ky = prop.mode_n * np.pi / Ly
        k2_flat = jnp.asarray(kx ** 2 + ky ** 2)
        alpha_m = np.where(prop.mode_m == 0, 1.0, 2.0)
        alpha_n = np.where(prop.mode_n == 0, 1.0, 2.0)
        N_flat = (Lx / alpha_m) * (Ly / alpha_n)
        inv_N_flat = jnp.asarray(1.0 / N_flat)
        dx_grid = Lx / prop.nx
        dy_grid = Ly / prop.ny
        if config.dealias_max_modes is None:
            dealias_mask = jnp.ones(len(prop.mode_m), dtype=bool)
        else:
            N_eff = int(config.dealias_max_modes)
            dealias_mask = jnp.asarray(
                (prop.mode_m < N_eff) & (prop.mode_n < N_eff)
            )

    # Auto-pick dt from highest natural frequency in the basis
    omega_max = float(prop.omega.max())
    if dt is None:
        dt = 2.0 * np.pi / (config.steps_per_period * omega_max)
    n_steps = int(math.ceil(T_eval / dt))
    dt_actual = T_eval / n_steps  # adjust to land exactly on T_eval

    def rhs(a, b, t):
        if M == 1:
            return _rhs_M1_transient(
                a, b, t, omega, sigma_mode, gamma, g, C, theta, dt_bin, n_bins,
            )
        return _rhs_M2_transient(
            a, b, t, omega, sigma_mode, gamma, g, C, theta, dt_bin, n_bins,
            prop, k2_flat, inv_N_flat, dx_grid, dy_grid, dealias_mask,
        )

    def step(state, _):
        a, b, t = state
        d1a, d1b = rhs(a,                   b,                   t)
        d2a, d2b = rhs(a + 0.5*dt_actual*d1a, b + 0.5*dt_actual*d1b, t + 0.5*dt_actual)
        d3a, d3b = rhs(a + 0.5*dt_actual*d2a, b + 0.5*dt_actual*d2b, t + 0.5*dt_actual)
        d4a, d4b = rhs(a +     dt_actual*d3a, b +     dt_actual*d3b, t +     dt_actual)
        a_new = a + (dt_actual / 6.0) * (d1a + 2*d2a + 2*d3a + d4a)
        b_new = b + (dt_actual / 6.0) * (d1b + 2*d2b + 2*d3b + d4b)
        return (a_new, b_new, t + dt_actual), None

    init_state = (jnp.zeros_like(omega), jnp.zeros_like(omega), 0.0)
    (a_final, _, _), _ = jax.lax.scan(step, init_state, xs=None, length=n_steps)
    return a_final


# ── Warm-start from steady-state phasors ───────────────────────────────


def warm_start_from_steady(
    P: np.ndarray,
    Omega_freqs: np.ndarray,
    dt_bin: float,
    n_bins: int,
) -> jnp.ndarray:
    """Convert steady-state complex phasors into piecewise-constant drive.

    The steady-state actuator displacement is
        f_i(t) = Σ_k Im(P[i,k] · e^{iΩ_k t}).
    Discretize at bin centers to get the warm-start ``theta[i, b]``.

    Parameters
    ----------
    P           : [n_act, n_freq] complex phasors
    Omega_freqs : [n_freq] driving angular frequencies (rad/s)
    dt_bin      : bin width (s)
    n_bins      : number of bins

    Returns
    -------
    theta : [n_act, n_bins], real
    """
    Omega = np.asarray(Omega_freqs)
    P_arr = np.asarray(P)
    t_centers = (np.arange(n_bins) + 0.5) * dt_bin            # [n_bins]
    phase = np.exp(1j * Omega[None, :] * t_centers[:, None])    # [n_bins, n_freq]
    theta = np.imag(P_arr @ phase.T)                            # [n_act, n_bins]
    return jnp.asarray(theta)


# ── Loss function ──────────────────────────────────────────────────────


def make_transient_loss(
    prop: Propagator,
    target: np.ndarray,
    dt_bin: float,
    n_bins: int,
    *,
    M: int = 2,
    config: HOSConfig | None = None,
    sigma: float = 0.02,
    sigma_blur: float = 0.02,
    loss_type: str = 'cosine',
    lambda_eta: float = 100.0,
    lambda_slope: float = 100.0,
    lambda_energy: float = 1e-5,
    n_water: float = 1.33,
    full_snell: bool = False,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build scalar loss(theta) for the transient pipeline.

    Same loss structure as steady-state ``make_loss``: caustic-match term
    (cosine/pearson/ssim) plus L2 penalties on η, slope, and drive amplitude.
    Single-frame evaluation at T_eval = n_bins * dt_bin (no time-averaging
    window in v1 — add later if static optima are too unstable visually).
    """
    dx = float(prop.xs[1] - prop.xs[0])
    dy = float(prop.ys[1] - prop.ys[0])

    if sigma_blur > 0:
        w_blur = int(math.ceil(4.0 * sigma_blur / max(dx, dy)))
        T_b = jnp.asarray(_gaussian_blur_separable(
            jnp.asarray(target), dx, dy, sigma_blur, w_blur))
    else:
        T_b = jnp.asarray(target)

    T_b_centered = T_b - jnp.mean(T_b)
    norm_T = float(jnp.sqrt(jnp.sum(T_b ** 2) + 1e-12))
    norm_T_centered = float(jnp.sqrt(jnp.sum(T_b_centered ** 2) + 1e-12))

    def _frame_loss(I):
        if loss_type == 'cosine':
            dot = jnp.sum(I * T_b)
            norm_I = jnp.sqrt(jnp.sum(I ** 2) + 1e-12)
            return 1.0 - dot / (norm_I * norm_T)
        elif loss_type == 'pearson':
            I_c = I - jnp.mean(I)
            dot = jnp.sum(I_c * T_b_centered)
            norm_I = jnp.sqrt(jnp.sum(I_c ** 2) + 1e-12)
            return 1.0 - dot / (norm_I * norm_T_centered)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type!r}")

    def loss_fn(theta: jnp.ndarray) -> jnp.ndarray:
        a = hos_forward_transient(prop, theta, dt_bin, M=M, config=config)
        _, _, I = caustic_image(prop, a, n_water=n_water, sigma=sigma,
                                full_snell=full_snell)
        eta, deta_dx, deta_dy = reconstruct_surface(prop, a)
        L_match = _frame_loss(I)
        L_eta = jnp.mean(eta ** 2)
        L_slope = jnp.mean(deta_dx ** 2 + deta_dy ** 2)
        L_energy = jnp.mean(theta ** 2)
        return (L_match
                + lambda_eta * L_eta
                + lambda_slope * L_slope
                + lambda_energy * L_energy)

    return loss_fn


# ── Optimizer ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TransientStage:
    sigma: float
    sigma_blur: float
    iters: int
    lr: float = 1e-3


def optimize_transient(
    prop: Propagator,
    target: np.ndarray,
    dt_bin: float,
    n_bins: int,
    *,
    M: int = 2,
    config: HOSConfig | None = None,
    stages: Sequence[TransientStage] = (
        TransientStage(sigma=0.04, sigma_blur=0.04, iters=300, lr=1e-3),
        TransientStage(sigma=0.02, sigma_blur=0.02, iters=300, lr=5e-4),
        TransientStage(sigma=0.01, sigma_blur=0.01, iters=300, lr=2e-4),
    ),
    theta0: np.ndarray | None = None,
    lambda_eta: float = 100.0,
    lambda_slope: float = 100.0,
    lambda_energy: float = 1e-5,
    loss_type: str = 'cosine',
    n_water: float = 1.33,
    full_snell: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """Run sigma-annealed Adam on the transient drive parameters.

    Parameters mirror ``optimize_caustic`` in shape; differences are:
      - ``theta0`` is [n_act, n_bins] real (not packed phasors).
      - ``forward_fn`` is fixed to ``hos_forward_transient``.
      - No multi-frame ('movie') T_eval; single snapshot only.
    """
    n_act = prop.n_act

    if theta0 is None:
        theta = jnp.zeros((n_act, n_bins))
    else:
        theta = jnp.asarray(theta0)

    loss_history: list[float] = []

    for stage in stages:
        loss_fn = make_transient_loss(
            prop, target, dt_bin, n_bins,
            M=M, config=config,
            sigma=stage.sigma, sigma_blur=stage.sigma_blur,
            loss_type=loss_type,
            lambda_eta=lambda_eta, lambda_slope=lambda_slope,
            lambda_energy=lambda_energy,
            n_water=n_water, full_snell=full_snell,
        )

        optimizer = optax.adam(stage.lr)
        opt_state = optimizer.init(theta)

        @jax.jit
        def step(theta, opt_state, _loss_fn=loss_fn):
            L, g = jax.value_and_grad(_loss_fn)(theta)
            updates, opt_state = optimizer.update(g, opt_state)
            theta = optax.apply_updates(theta, updates)
            return theta, opt_state, L

        desc = f"σ={stage.sigma:.3f} [transient]"
        with tqdm(range(stage.iters), desc=desc, leave=True) as pbar:
            for _ in pbar:
                theta, opt_state, L = step(theta, opt_state)
                L_val = float(L)
                loss_history.append(L_val)
                pbar.set_postfix(loss=f"{L_val:.4f}")

    return np.asarray(theta), loss_history
