"""
Analytical initial guess via Poisson inversion.

The paraxial caustic intensity satisfies:

    I(x,y) ≈ 1 - (depth/n_water) · ∇²η

Since ∇²φ_{m,n} = -k²_{m,n} · φ_{m,n}, the required modal amplitudes are:

    a_{m,n} = c_{m,n} · n_water / (depth · k²_{m,n})

where c_{m,n} are the cosine expansion coefficients of (I_target - 1).

The steady-state linear system a = Im(H ⊙ (C @ P) @ exp(iΩT)) is then
solved for the phasor matrix P via regularized least-squares.

This gives a physically grounded warm-start for gradient descent.
"""

import numpy as np
from scipy.linalg import lstsq

from .physics import Propagator, Tank, Actuator, build_propagator, pack_complex
from .loss import load_target_image


# ── Target analysis ────────────────────────────────────────────────────

def analyze_target(
    target: np.ndarray,
    tank: Tank,
    *,
    n_modes_max: int = 50,
    energy_fraction: float = 0.95,
) -> dict:
    """
    Analyze a target image's spatial frequency content.

    Projects the target onto the cosine eigenmode basis and identifies
    which modes carry 95% of the energy. Returns suggested n_modes,
    n_freq, frequency range, and actuator count.

    Returns a dict with keys:
      coeffs, energy, freq_map, important_modes,
      n_modes, n_freq, n_act, freq_min, freq_max, m_max, n_max
    """
    Lx, Ly, depth, g = tank.Lx, tank.Ly, tank.depth, tank.g
    nx, ny = target.shape

    xs = np.linspace(0, Lx, nx)
    ys = np.linspace(0, Ly, ny)
    ms = np.arange(n_modes_max + 1)
    ns = np.arange(n_modes_max + 1)

    cos_x = np.cos(np.outer(xs, ms * np.pi / Lx))   # [nx, n_modes_max+1]
    cos_y = np.cos(np.outer(ys, ns * np.pi / Ly))   # [ny, n_modes_max+1]

    dx, dy = Lx / nx, Ly / ny
    raw = cos_x.T @ target.astype(float) @ cos_y    # [n_modes+1, n_modes+1]

    coeffs = np.zeros_like(raw)
    for mi, m in enumerate(ms):
        for ni, n in enumerate(ns):
            Ix = Lx if m == 0 else Lx / 2
            Iy = Ly if n == 0 else Ly / 2
            coeffs[mi, ni] = raw[mi, ni] * dx * dy / (Ix * Iy)
    coeffs[0, 0] = 0.0  # exclude DC

    energy = coeffs**2
    E_total = energy.sum()

    # Natural frequency for each mode
    freq_map = np.zeros_like(raw)
    for mi, m in enumerate(ms):
        for ni, n in enumerate(ns):
            if m == 0 and n == 0:
                continue
            k = np.sqrt((m * np.pi / Lx)**2 + (n * np.pi / Ly)**2)
            freq_map[mi, ni] = np.sqrt(g * k * np.tanh(k * depth)) / (2 * np.pi)

    # Sort modes by energy, find set capturing energy_fraction
    mode_list = [
        dict(m=int(ms[mi]), n=int(ns[ni]),
             E=float(energy[mi, ni]), f=float(freq_map[mi, ni]))
        for mi in range(len(ms)) for ni in range(len(ns))
        if not (mi == 0 and ni == 0)
    ]
    mode_list.sort(key=lambda x: -x['E'])

    cum_E = np.cumsum([m['E'] for m in mode_list])
    n_needed = int(np.searchsorted(cum_E, energy_fraction * E_total)) + 1
    important = mode_list[:n_needed]

    m_max = max(m['m'] for m in important)
    n_max = max(m['n'] for m in important)
    freqs_pos = [m['f'] for m in important if m['f'] > 0]
    freq_min = min(freqs_pos)
    freq_max = max(freqs_pos)

    suggested_n_modes = max(m_max, n_max) + 1
    n_act_per_side = max(m_max, n_max) + 1
    suggested_n_act = 4 * n_act_per_side
    suggested_n_freq = max(4, round(2 * (freq_max - freq_min) + 1))

    print(f"{'='*60}")
    print(f"  Target Analysis")
    print(f"{'='*60}")
    print(f"  Grid: {nx}×{ny}")
    print(f"  Modes capturing {energy_fraction*100:.0f}% energy: {n_needed}")
    print(f"  Highest mode indices: m_max={m_max}, n_max={n_max}")
    print(f"  Frequency range: {freq_min:.2f} – {freq_max:.2f} Hz")
    print(f"  Suggested: n_modes={suggested_n_modes}, n_freq={suggested_n_freq}, "
          f"n_act={suggested_n_act} ({n_act_per_side}/side)")
    print(f"  Top 5 modes by energy:")
    for md in mode_list[:5]:
        print(f"    ({md['m']},{md['n']})  f={md['f']:.2f} Hz  "
              f"energy={md['E']/E_total*100:.1f}%")
    print(f"{'='*60}")

    return dict(
        coeffs=coeffs, energy=energy, freq_map=freq_map,
        important_modes=important, E_total=E_total,
        n_modes=suggested_n_modes, n_freq=suggested_n_freq,
        n_act=suggested_n_act, freq_min=freq_min, freq_max=freq_max,
        m_max=m_max, n_max=n_max,
    )


# ── Analytical solve ───────────────────────────────────────────────────

def analytical_solve(
    prop: Propagator,
    target: np.ndarray,
    Omega_freqs: np.ndarray,
    T_eval: float,
    *,
    n_water: float = 1.33,
    max_contrast: float = 0.5,
    coupling_rtol: float = 1e-3,
    pinv_rcond: float = 1e-3,
) -> dict:
    """
    Compute actuator phasors analytically via Poisson inversion.

    Steps
    -----
    1. Project (target/mean(target) - 1) onto cosine eigenmodes → c_{m,n}
    2. Poisson inversion: a_{m,n} = c_{m,n} · n_water / (depth · k²_{m,n})
    3. Mask modes with negligible actuator coupling
    4. Scale a_desired so caustic contrast ≤ max_contrast
    5. Build linear system M·θ = a_desired; solve via regularized pinv

    Parameters
    ----------
    prop          : Propagator
    target        : target image [nx, ny] in [0, 1]
    Omega_freqs   : driving angular frequencies [n_freq]
    T_eval        : evaluation time (s)
    n_water       : refractive index
    max_contrast  : maximum |I - 1| in the implied caustic (controls amplitude)
    coupling_rtol : modes with coupling < rtol * max_coupling are masked
    pinv_rcond    : rcond for regularized pseudoinverse

    Returns
    -------
    dict with keys:
      p0        : parameter vector [2 * n_act * n_freq]
      a_desired : target modal amplitudes [n_total]
    """
    Lx, Ly = prop.tank.Lx, prop.tank.Ly
    # The Poisson inversion derives from the *paraxial* caustic model
    # `I ≈ 1 - (throw/n_water)·∇²η`, which depends on optical throw, not
    # water depth. With a flat-bottom tank these are equal; with an
    # elevated glass-bottom + air gap they differ.
    throw = prop.tank.throw
    damping = prop.tank.damping
    n_modes = prop.n_modes
    n_total = len(prop.omega)
    n_act = prop.n_act
    n_freq = len(Omega_freqs)
    nx, ny = prop.nx, prop.ny
    assert target.shape == (nx, ny), f"Target shape {target.shape} != propagator grid ({nx}, {ny})"

    # ── 1. Project (I_target - 1) onto cosine basis ───────────────────
    dx, dy = Lx / nx, Ly / ny
    t_mean = max(float(target.mean()), 1e-6)
    residual = target.astype(float) / t_mean - 1.0    # I_target - 1, mean ≈ 0

    raw = prop.cos_x.T @ residual @ prop.cos_y        # [n_modes, n_modes]

    coeffs = np.zeros_like(raw)
    for mi in range(n_modes):
        for ni in range(n_modes):
            m, n = mi, ni
            Ix = Lx if m == 0 else Lx / 2
            Iy = Ly if n == 0 else Ly / 2
            coeffs[mi, ni] = raw[mi, ni] * dx * dy / (Ix * Iy)
    coeffs[0, 0] = 0.0

    # ── 2. Poisson inversion ──────────────────────────────────────────
    a_desired = np.zeros(n_total)
    for j in range(n_total):
        m, n = prop.mode_m[j], prop.mode_n[j]
        k2 = (m * np.pi / Lx)**2 + (n * np.pi / Ly)**2
        a_desired[j] = coeffs[m, n] * n_water / (throw * k2)

    # ── 3. Mask modes with negligible actuator coupling ───────────────
    max_coupling = np.abs(prop.C).max(axis=1)          # [n_total]
    threshold = coupling_rtol * max_coupling.max()
    achievable = max_coupling >= threshold
    a_desired *= achievable
    print(f"  Analytical solve: {achievable.sum()}/{n_total} modes achievable")

    # ── 4. Scale to target caustic contrast ──────────────────────────
    # Implied caustic deviation: δI = -(depth/n_water) · ∇²η
    # ∇²φ_{m,n} = -k²φ_{m,n}, so rebuild ∇²η from a_desired
    k2_2d = np.zeros((n_modes, n_modes))
    a_2d = np.zeros((n_modes, n_modes))
    for j in range(n_total):
        m, n = prop.mode_m[j], prop.mode_n[j]
        k2_2d[m, n] = (m * np.pi / Lx)**2 + (n * np.pi / Ly)**2
        a_2d[m, n] = a_desired[j]

    lap_field = prop.cos_x @ (-k2_2d * a_2d) @ prop.cos_y.T   # ∇²η  [nx, ny]
    caustic_dev = (throw / n_water) * lap_field
    caustic_range = float(np.abs(caustic_dev).max())

    if caustic_range > max_contrast:
        scale = max_contrast / caustic_range
        a_desired *= scale
        print(f"  Caustic contrast {caustic_range:.3f} → {max_contrast} "
              f"(scale={scale:.3g})")
    else:
        print(f"  Caustic contrast {caustic_range:.3f} (within target {max_contrast})")

    # ── 5. Build and solve linear system M·θ = a_desired ─────────────
    # a = Im(H ⊙ (C @ P) @ exp(iΩT))  is linear in θ = [vec(X); vec(Y)]
    # where P = X + iY
    #
    # M_X[:,k*n_act + i] = C[j,i] · Im(β[j,k])
    # M_Y[:,k*n_act + i] = C[j,i] · Re(β[j,k])
    H = 1.0 / (prop.omega[:, None]**2 - Omega_freqs[None, :]**2
               + 2j * damping * prop.omega[:, None] * Omega_freqs[None, :])  # [n_total, n_freq]
    E = np.exp(1j * Omega_freqs * T_eval)              # [n_freq]
    beta = H * E[None, :]                               # [n_total, n_freq]

    # Build M with actuator-major column ordering to match unpack_complex:
    #   θ[i*n_freq + k]           = X[i,k]  (X block, first n_act*n_freq cols)
    #   θ[n_act*n_freq + i*n_freq + k] = Y[i,k]  (Y block, last n_act*n_freq cols)
    #
    # M[j, i*n_freq + k] = C[j,i] * Im(β[j,k])   (X columns)
    # M[j, n_act*n_freq + i*n_freq + k] = C[j,i] * Re(β[j,k])   (Y columns)
    #
    # Vectorised: C[:,i,None] * Im(β)[:,None,:] → [n_total, n_act, n_freq]
    # then reshape to [n_total, n_act*n_freq] (row-major = actuator-major).
    imag_beta = np.imag(beta)   # [n_total, n_freq]
    real_beta = np.real(beta)   # [n_total, n_freq]
    M = np.zeros((n_total, 2 * n_act * n_freq))
    M[:, :n_act * n_freq] = (
        prop.C[:, :, None] * imag_beta[:, None, :]
    ).reshape(n_total, n_act * n_freq)
    M[:, n_act * n_freq:] = (
        prop.C[:, :, None] * real_beta[:, None, :]
    ).reshape(n_total, n_act * n_freq)
    M *= achievable[:, None]   # zero out rows for unachievable modes

    # Regularized least-squares (min-norm if underdetermined)
    theta, *_ = lstsq(M, a_desired, cond=pinv_rcond)

    print(f"  ‖a_desired‖ = {np.linalg.norm(a_desired):.4g},  "
          f"‖p0‖ = {np.linalg.norm(theta):.4g}")

    return dict(p0=theta, a_desired=a_desired)


# ── Auto-setup from target ─────────────────────────────────────────────

def setup_from_target(
    target: np.ndarray,
    tank: Tank,
    *,
    energy_fraction: float = 0.95,
    nx: int = 100,
    ny: int = 100,
    n_modes_max: int = 60,
    actuator_width: float = 0.05,
    T_eval: float = 1.0,
    n_water: float = 1.33,
) -> dict:
    """
    Automatically configure a propagator and compute an analytical warm-start
    from a target image.

    For sharp targets (small bright spots, high-contrast edges), increase
    `energy_fraction` toward 0.99 and `n_modes_max` toward 80. Sharp features
    require short-wavelength modes; if the analyzer caps out at the suggested
    n_modes_max, the optimizer will compensate with large amplitudes that
    may break the linear-wave assumptions.

    Returns a dict with keys:
      prop, Omega_freqs, target_bl, analysis, freqs, actuators, p0
    """
    Lx, Ly = tank.Lx, tank.Ly

    analysis = analyze_target(target, tank, n_modes_max=n_modes_max,
                               energy_fraction=energy_fraction)
    n_modes  = analysis['n_modes']
    n_freq   = analysis['n_freq']
    freq_min = analysis['freq_min']
    freq_max = analysis['freq_max']
    n_act_per_side = max(analysis['m_max'], analysis['n_max']) + 1

    freqs       = np.linspace(freq_min, freq_max, n_freq)
    Omega_freqs = 2 * np.pi * freqs

    # Uniform perimeter actuator layout
    positions: list[tuple[float, float]] = []
    for x in np.linspace(0, Lx, n_act_per_side + 2)[1:-1]:
        positions += [(x, 0.0), (x, Ly)]
    for y in np.linspace(0, Ly, n_act_per_side + 2)[1:-1]:
        positions += [(0.0, y), (Lx, y)]

    actuators = [Actuator(x, y, width=actuator_width) for x, y in positions]

    prop = build_propagator(tank, actuators, n_modes, nx, ny)

    # Band-limit target: reconstruct from achievable cosine modes only
    c = analysis['coeffs'][:n_modes, :n_modes]
    target_bl = prop.cos_x @ c @ prop.cos_y.T
    target_bl = np.clip(target_bl, 0.0, None)
    if target_bl.max() > 0:
        target_bl /= target_bl.max()

    n_act = len(actuators)
    print(f"\nSetup: {n_act} actuators, {n_freq} frequencies "
          f"({freq_min:.2f}–{freq_max:.2f} Hz), {len(prop.omega)} modes")
    print(f"Grid: {nx}×{ny},  Parameters: {2 * n_act * n_freq}")

    sol = analytical_solve(prop, target_bl, Omega_freqs, T_eval,
                           n_water=n_water)

    return dict(
        prop=prop,
        Omega_freqs=Omega_freqs,
        target_bl=target_bl,
        analysis=analysis,
        freqs=freqs,
        actuators=actuators,
        p0=sol['p0'],
    )
