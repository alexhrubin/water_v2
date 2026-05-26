"""
Higher-Order Spectral (HOS) forward solver for finite-amplitude water waves.

Extends the linear-wave model in ``physics.py`` with finite-amplitude
surface dynamics via the Dommermuth–Yue (1987) perturbation expansion.

The state is the pair of modal amplitude vectors ``(a, b)`` where

    η(x,y,t) = Σ_j a_j(t) · φ_j(x,y)        (surface elevation)
    ψ(x,y,t) = Σ_j b_j(t) · φ_j(x,y)        (surface velocity potential)

evolved by the Zakharov free-surface BCs. At order ``M``:

    M = 1   ⇒  ȧ = σ·b
              ḃ = -g·a - 2γω·b + R(t)        (linear, equivalent to physics.py)

    M = 2   ⇒  + quadratic state products (η·W₂, ∇ψ·∇η, |∇ψ|², W₁²)
    M = 3   ⇒  + cubic state products (not yet)

The forcing ``R(t)`` is chosen so the M=1 steady state matches
``steady_state_amplitudes()`` exactly — see ``docs/hos_math.md`` §"Damping
and forcing".

Nonlinear coupling at M=2 is evaluated via grid-product projection (per
``docs/hos_math.md`` §"Efficient nonlinear evaluation via DCT-convolution"):
reconstruct η, ψ, ∇η, ∇ψ, W₁, W₂ on the spatial grid; form pointwise
products; project back onto cosine modes. Cost scales like
O(nx·ny·n_modes) per timestep — about 1000× faster than the explicit
triple-tensor sum for the portrait setup.
"""

from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp

from .physics import Propagator


# ── State and config ───────────────────────────────────────────────────

@dataclass(frozen=True)
class HOSConfig:
    """
    Time-stepping configuration for the HOS forward solver.

    The defaults follow the dt / settle-time heuristics in
    ``docs/hos_math.md`` §"Time integration".
    """
    M: int = 1                  # perturbation order (1, 2, or 3)
    dt: float | None = None     # timestep; None → auto from max ω
    t_settle: float | None = None  # transient decay time; None → auto
    steps_per_period: int = 20  # for auto-dt: resolve highest mode at this rate
    settle_decays: float = 5.0  # for auto-t_settle: 5 e-foldings of slowest mode
    dealias_max_modes: int | None = None
        # Cap on the per-side mode index that participates in M=2 quadratic
        # nonlinear products. Only modes with (m, n) both < this value
        # contribute. None disables dealiasing (use the full basis), which
        # is fine for small apparatus (n_modes_per_side ≲ 30) but unstable
        # above that — high-mode coupling through W₁=σ·b and W₂=k²·b drives
        # NaN within ~1s. Set this to ~n_modes_per_side / 4 for large bases.


# ── Initial conditions ─────────────────────────────────────────────────

def steady_state_initial(
    prop: Propagator,
    P: jnp.ndarray,
    Omega_freqs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute (a, b) at t=0 on the linear-theory steady-state orbit.

    Useful for validating the M=1 integrator: starting on the steady-state
    orbit, the M=1 integrator must stay on it. Also useful as a warm
    initial condition for M≥2 (skips transient).

    Returns
    -------
    a0, b0 : modal amplitudes [n_total]
    """
    omega = jnp.asarray(prop.omega)
    # σ_j = k_j·tanh(k_j·d). Compute from wavenumber so the capillary case
    # is handled correctly (ω²/g is only valid for pure gravity).
    kx_flat = prop.mode_m * np.pi / prop.tank.Lx
    ky_flat = prop.mode_n * np.pi / prop.tank.Ly
    k_per_mode = np.sqrt(kx_flat**2 + ky_flat**2)
    sigma = jnp.asarray(k_per_mode * np.tanh(k_per_mode * prop.tank.depth))
    gamma = prop.tank.damping
    Omega = jnp.asarray(Omega_freqs)

    H = 1.0 / (omega[:, None]**2 - Omega[None, :]**2
               + 2j * gamma * omega[:, None] * Omega[None, :])   # [n_total, n_freq]
    CP_H = (jnp.asarray(prop.C) @ P) * H                          # [n_total, n_freq]

    # a(t) = Im(Σ_k CP_H[j,k] · e^{iΩ_k t})
    # ȧ(t) = Im(Σ_k iΩ_k · CP_H[j,k] · e^{iΩ_k t})
    a0  = jnp.imag(jnp.sum(CP_H, axis=1))
    ad0 = jnp.imag(jnp.sum(1j * Omega[None, :] * CP_H, axis=1))
    b0 = ad0 / sigma
    return a0, b0


# ── RHS for each order ─────────────────────────────────────────────────

def _rhs_M1(
    a: jnp.ndarray, b: jnp.ndarray, t: float,
    omega: jnp.ndarray, sigma: jnp.ndarray, g_eff: jnp.ndarray, gamma: float,
    CP: jnp.ndarray, Omega: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    M=1 right-hand side. No mode coupling, no nonlinearity.

        ȧ_j = σ_j · b_j
        ḃ_j = -g_eff_j · a_j - 2γω_j · b_j + R_j(t)

    where σ_j = k_j·tanh(k_j·d) is the kinematic coefficient,
    g_eff_j = g + (σ/ρ)·k_j² is the per-mode restoring acceleration
    (reduces to g without capillarity), and
    R_j(t) = (g_eff_j/ω_j²) · Σ_k Im((CP)[j,k] · exp(iΩ_k t))
    is the time-domain forcing whose steady-state matches the linear
    theory (ω² = g_eff · σ).
    """
    # R_j(t) = (g_eff_j/ω_j²) · Im(Σ_k (CP)[j,k] · e^{iΩ_k t})
    phase = jnp.exp(1j * Omega * t)                        # [n_freq]
    F = jnp.imag(CP @ phase)                               # [n_total]
    R = (g_eff / omega**2) * F                             # [n_total]

    da = sigma * b
    db = -g_eff * a - 2.0 * gamma * omega * b + R
    return da, db


# ── Grid helpers for M ≥ 2 ─────────────────────────────────────────────

def _scatter_2d(a_flat: jnp.ndarray, prop: Propagator) -> jnp.ndarray:
    """Flat mode vector [n_total] → 2D mode grid [n_modes, n_modes]."""
    return jnp.zeros((prop.n_modes, prop.n_modes), dtype=a_flat.dtype) \
        .at[prop.mode_m, prop.mode_n].set(a_flat)


def _project_to_modes(
    field: jnp.ndarray,
    prop: Propagator,
    inv_N_flat: jnp.ndarray,
    dx: float, dy: float,
) -> jnp.ndarray:
    """
    Project a spatial field [nx, ny] back onto the cosine modes.

    Returns a flat mode vector [n_total] with the DC (0,0) component
    naturally excluded (it's not in mode_m/mode_n).
    """
    raw = jnp.asarray(prop.cos_x).T @ field @ jnp.asarray(prop.cos_y)   # [n_modes, n_modes]
    flat_raw = raw[prop.mode_m, prop.mode_n]                             # [n_total]
    return flat_raw * (dx * dy) * inv_N_flat


def _rhs_M2(
    a: jnp.ndarray, b: jnp.ndarray, t: float,
    omega: jnp.ndarray, sigma: jnp.ndarray, g_eff: jnp.ndarray, gamma: float,
    CP: jnp.ndarray, Omega: jnp.ndarray,
    prop: Propagator,
    k2_flat: jnp.ndarray,
    inv_N_flat: jnp.ndarray,
    dx: float, dy: float,
    dealias_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    M=2 right-hand side. M=1 linear part + quadratic nonlinear terms:

        η_t = W₁  +  η·W₂  -  ∇ψ·∇η                (kinematic, M=2)
        ψ_t = -gη  -  (1/2)|∇ψ|²  +  (1/2) W₁²    (dynamic, M=2)
              - 2γω·b  +  R(t)                        (damping + forcing as M=1)

    The nonlinear products are formed on the grid (size nx · ny) and
    projected back onto modes. Linear part is reused from ``_rhs_M1``.

    Dealiasing: only modes inside ``dealias_mask`` (a [n_total] boolean
    array) contribute to the nonlinear inputs. See HOSConfig.dealias_ratio.
    """
    da_lin, db_lin = _rhs_M1(a, b, t, omega, sigma, g_eff, gamma, CP, Omega)

    # Dealias the inputs to nonlinear products (1/2-rule for cosine basis).
    a_in = jnp.where(dealias_mask, a, 0.0)
    b_in = jnp.where(dealias_mask, b, 0.0)

    # Reconstruct η, ψ, ∇η, ∇ψ, W₁, W₂ on the grid.
    a_2d  = _scatter_2d(a_in,             prop)
    b_2d  = _scatter_2d(b_in,             prop)
    sb_2d = _scatter_2d(sigma * b_in,     prop)        # for W₁
    k2b_2d = _scatter_2d(k2_flat * b_in,  prop)        # for W₂

    cx, cy   = jnp.asarray(prop.cos_x),  jnp.asarray(prop.cos_y)
    dcx, dcy = jnp.asarray(prop.dcos_x), jnp.asarray(prop.dcos_y)

    eta   = cx  @ a_2d   @ cy.T            # [nx, ny]
    eta_x = dcx @ a_2d   @ cy.T
    eta_y = cx  @ a_2d   @ dcy.T
    psi_x = dcx @ b_2d   @ cy.T
    psi_y = cx  @ b_2d   @ dcy.T
    W1    = cx  @ sb_2d  @ cy.T
    W2    = cx  @ k2b_2d @ cy.T

    # Nonlinear products on the grid.
    eta_W2     = eta * W2
    grad_dot   = psi_x * eta_x + psi_y * eta_y
    grad_psi_sq = psi_x**2 + psi_y**2
    W1_sq      = W1**2

    # Project corrections back onto modes.
    proj = lambda f: _project_to_modes(f, prop, inv_N_flat, dx, dy)
    da_nl =  proj(eta_W2) -  proj(grad_dot)
    db_nl = -0.5 * proj(grad_psi_sq) + 0.5 * proj(W1_sq)

    return da_lin + da_nl, db_lin + db_nl


# ── Forward solver ─────────────────────────────────────────────────────

def hos_forward(
    prop: Propagator,
    P: jnp.ndarray,
    Omega_freqs: jnp.ndarray,
    T_eval: float = 1.0,
    *,
    config: HOSConfig = HOSConfig(),
    initial: tuple[jnp.ndarray, jnp.ndarray] | str = "rest",
) -> jnp.ndarray:
    """
    Integrate the HOS free-surface BCs and return modal amplitudes at T_eval.

    Parameters
    ----------
    prop        : Propagator (same one the linear pipeline uses)
    P           : complex phasor matrix [n_act, n_freq]
    Omega_freqs : driving angular frequencies [n_freq]
    T_eval      : evaluation time on the steady-state orbit (s).
                  Same role as ``T_eval`` in ``steady_state_amplitudes()``.
    config      : HOSConfig — order M and time-stepping params
    initial     : "rest"  → start from (a, b) = (0, 0), integrate through
                            the transient. Use when validating M=1 vs the
                            linear theory.
                  "steady" → start on the linear-theory steady-state orbit.
                            Use for M≥2 to skip the transient.
                  (a0, b0) tuple → explicit initial state.

    Returns
    -------
    a : modal amplitudes [n_total] at ``t_settle + T_eval`` (when starting
        from rest) or at ``T_eval`` (when starting on steady state).
    """
    if config.M not in (1, 2):
        raise NotImplementedError(
            f"HOS M={config.M} not implemented yet. M=3 lands in a follow-up."
        )

    omega = jnp.asarray(prop.omega)
    gamma = prop.tank.damping
    g     = prop.tank.g
    st    = prop.tank.surface_tension
    Omega = jnp.asarray(Omega_freqs)
    CP    = jnp.asarray(prop.C) @ P                # [n_total, n_freq], complex

    # Per-mode kinematics. Compute σ_j = k_j·tanh(k_j·d) from wavenumber
    # directly so the capillary case is handled (where ω² ≠ g·k·tanh(kd)).
    # g_eff_j = g + (σ/ρ)·k_j² is the dispersion-consistent restoring
    # acceleration. Both reduce to the gravity-only forms when st=0.
    Lx_t, Ly_t = prop.tank.Lx, prop.tank.Ly
    kx_flat = prop.mode_m * np.pi / Lx_t
    ky_flat = prop.mode_n * np.pi / Ly_t
    k2_per_mode = kx_flat**2 + ky_flat**2
    k_per_mode = np.sqrt(k2_per_mode)
    sigma = jnp.asarray(k_per_mode * np.tanh(k_per_mode * prop.tank.depth))
    g_eff = jnp.asarray(g + st * k2_per_mode)

    # M=2 precomputations: k_j² per flat mode + per-mode (1/N_j).
    if config.M >= 2:
        Lx, Ly = prop.tank.Lx, prop.tank.Ly
        k2_flat = jnp.asarray(k2_per_mode)
        # N_j = (Lx / α_m)(Ly / α_n), α=1 if index==0 else 2
        alpha_m = np.where(prop.mode_m == 0, 1.0, 2.0)
        alpha_n = np.where(prop.mode_n == 0, 1.0, 2.0)
        N_flat = (Lx / alpha_m) * (Ly / alpha_n)
        inv_N_flat = jnp.asarray(1.0 / N_flat)
        dx = Lx / prop.nx
        dy = Ly / prop.ny

        # Dealiasing mask: limit which modes participate in nonlinear
        # products. None → no dealiasing (use the full basis).
        if config.dealias_max_modes is None:
            dealias_mask = jnp.ones(len(prop.mode_m), dtype=bool)
        else:
            N_eff = int(config.dealias_max_modes)
            dealias_mask = jnp.asarray(
                (prop.mode_m < N_eff) & (prop.mode_n < N_eff)
            )

    # ── Auto-pick dt and t_settle from mode spectrum ──────────────────
    omega_max = float(prop.omega.max())
    omega_min = float(prop.omega.min())
    dt = config.dt
    if dt is None:
        dt = 2.0 * np.pi / (config.steps_per_period * omega_max)
    t_settle = config.t_settle
    if t_settle is None:
        t_settle = config.settle_decays / (gamma * omega_min)

    # Decide integration window
    if initial == "rest":
        a0 = jnp.zeros_like(omega)
        b0 = jnp.zeros_like(omega)
        t0 = 0.0
        t_end = t_settle + T_eval
    elif initial == "steady":
        a0, b0 = steady_state_initial(prop, P, Omega_freqs)
        t0 = 0.0
        t_end = T_eval
    else:
        a0, b0 = initial
        t0 = 0.0
        t_end = T_eval

    n_steps = int(np.ceil((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps    # adjust dt to land exactly on t_end

    # ── RK4 inner step ────────────────────────────────────────────────
    def rhs(a, b, t):
        if config.M == 1:
            return _rhs_M1(a, b, t, omega, sigma, g_eff, gamma, CP, Omega)
        return _rhs_M2(a, b, t, omega, sigma, g_eff, gamma, CP, Omega,
                       prop, k2_flat, inv_N_flat, dx, dy, dealias_mask)

    def step(state, _):
        a, b, t = state
        d1a, d1b = rhs(a,              b,              t)
        d2a, d2b = rhs(a + 0.5*dt*d1a, b + 0.5*dt*d1b, t + 0.5*dt)
        d3a, d3b = rhs(a + 0.5*dt*d2a, b + 0.5*dt*d2b, t + 0.5*dt)
        d4a, d4b = rhs(a +     dt*d3a, b +     dt*d3b, t +     dt)
        a_new = a + (dt / 6.0) * (d1a + 2*d2a + 2*d3a + d4a)
        b_new = b + (dt / 6.0) * (d1b + 2*d2b + 2*d3b + d4b)
        return (a_new, b_new, t + dt), None

    init_state = (a0, b0, t0)
    (a_final, _, _), _ = jax.lax.scan(step, init_state, xs=None, length=n_steps)
    return a_final
