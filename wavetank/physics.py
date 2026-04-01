"""
Core physics: tank geometry, eigenmodes, coupling matrix, steady-state amplitudes.

The wave field is expanded in the eigenmodes of the rectangular tank:

    φ_{m,n}(x,y) = cos(mπx/Lx) · cos(nπy/Ly)

with natural frequencies:

    ω_{m,n} = sqrt(g · k_{m,n} · tanh(k_{m,n} · depth))
    k_{m,n} = sqrt((mπ/Lx)² + (nπ/Ly)²)

Under sinusoidal forcing P[i,k] (complex phasor at actuator i, frequency k),
the steady-state modal amplitudes are:

    a = Im( H ⊙ (C @ P) @ exp(iΩT) )

where H[j,k] = 1 / (ω_j² - Ω_k² + 2iγω_jΩ_k) is the transfer matrix
and C[j,i] is the eigenmode coupling matrix.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import jax.numpy as jnp


# ── Data structures ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Tank:
    """Rectangular wave tank geometry and physical parameters."""
    Lx: float           # length in x (m)
    Ly: float           # length in y (m)
    depth: float        # water depth (m)
    damping: float = 0.02   # modal damping ratio γ
    g: float = 9.81     # gravitational acceleration (m/s²)


@dataclass(frozen=True)
class Actuator:
    """A boundary actuator with a Gaussian spatial footprint."""
    x: float
    y: float
    width: float = 0.05   # Gaussian half-width σ (m); 0 = point source


@dataclass
class Propagator:
    """
    All precomputed quantities needed for forward simulation and optimization.
    Built once by build_propagator(); passed as a closure into JIT-compiled fns.

    Arrays are stored as numpy (not JAX) for easy serialization; JAX functions
    convert them via jnp.asarray() on first use.
    """
    tank: Tank

    # Mode index vectors (length n_total = n_modes² - 1)
    mode_m: np.ndarray    # m-index for each flat mode
    mode_n: np.ndarray    # n-index for each flat mode
    lin_2d: np.ndarray    # scatter index: mode j → position in [n_modes, n_modes] grid

    # Physics
    omega: np.ndarray     # natural frequencies ω_j  [n_total]
    C: np.ndarray         # coupling matrix C[j, i]  [n_total, n_act]

    # Separable 1D basis matrices evaluated on the spatial grid
    cos_x: np.ndarray     # cos_x[i, m] = cos(mπ·xs[i]/Lx)   [nx, n_modes]
    cos_y: np.ndarray     # cos_y[j, n] = cos(nπ·ys[j]/Ly)   [ny, n_modes]
    dcos_x: np.ndarray    # ∂/∂x of cos_x                      [nx, n_modes]
    dcos_y: np.ndarray    # ∂/∂y of cos_y                      [ny, n_modes]

    # Spatial grid
    xs: np.ndarray        # grid x-coordinates  [nx]
    ys: np.ndarray        # grid y-coordinates  [ny]
    X_src: np.ndarray     # meshgrid X_src[i,j] = xs[i]  [nx, ny]
    Y_src: np.ndarray     # meshgrid Y_src[i,j] = ys[j]  [nx, ny]

    nx: int
    ny: int
    n_modes: int
    n_act: int


# ── Build propagator ───────────────────────────────────────────────────

def build_propagator(
    tank: Tank,
    actuators: Sequence[Actuator],
    n_modes: int,
    nx: int,
    ny: int,
) -> Propagator:
    """
    Precompute all quantities needed for simulation and optimization.

    Parameters
    ----------
    tank      : Tank geometry and parameters
    actuators : list of Actuator objects (positions + widths)
    n_modes   : number of modes per direction (total modes = n_modes² - 1)
    nx, ny    : spatial grid resolution
    """
    Lx, Ly, depth, g = tank.Lx, tank.Ly, tank.depth, tank.g
    n_act = len(actuators)

    # ── Collect mode indices, skip (0,0) ──────────────────────────────
    ms, ns = [], []
    for m in range(n_modes):
        for n in range(n_modes):
            if m == 0 and n == 0:
                continue
            ms.append(m)
            ns.append(n)
    mode_m = np.array(ms, dtype=np.int32)
    mode_n = np.array(ns, dtype=np.int32)
    n_total = len(mode_m)

    # ── Natural frequencies (gravity-wave dispersion relation) ─────────
    k = np.sqrt((mode_m * np.pi / Lx)**2 + (mode_n * np.pi / Ly)**2)
    omega = np.sqrt(g * k * np.tanh(k * depth))

    # ── Mode normalization: N_{m,n} = ∫∫ φ_{m,n}² dx dy ─────────────
    def norm_mn(m, n):
        Ix = Lx if m == 0 else Lx / 2
        Iy = Ly if n == 0 else Ly / 2
        return Ix * Iy

    # ── Coupling matrix C[j, i] = φ_j(x_i, y_i) · blob / N_j ────────
    # blob = exp(-σ²k²/2) accounts for finite actuator width
    C = np.zeros((n_total, n_act))
    for i, act in enumerate(actuators):
        for j in range(n_total):
            m, n = mode_m[j], mode_n[j]
            phi_val = (np.cos(m * np.pi * act.x / Lx)
                       * np.cos(n * np.pi * act.y / Ly))
            kx = m * np.pi / Lx
            ky = n * np.pi / Ly
            blob = np.exp(-0.5 * act.width**2 * (kx**2 + ky**2)) if act.width > 0 else 1.0
            C[j, i] = phi_val * blob / norm_mn(m, n)

    # ── Spatial grid and 1D basis matrices ────────────────────────────
    xs = np.linspace(0, Lx, nx)
    ys = np.linspace(0, Ly, ny)
    ms_arr = np.arange(n_modes)  # [n_modes]
    ns_arr = np.arange(n_modes)

    # cos_x[i, m] = cos(m·π·xs[i]/Lx)
    cos_x  =  np.cos(np.outer(xs, ms_arr * np.pi / Lx))          # [nx, n_modes]
    cos_y  =  np.cos(np.outer(ys, ns_arr * np.pi / Ly))          # [ny, n_modes]
    dcos_x = -np.outer(np.ones(nx), ms_arr * np.pi / Lx) * np.sin(np.outer(xs, ms_arr * np.pi / Lx))
    dcos_y = -np.outer(np.ones(ny), ns_arr * np.pi / Ly) * np.sin(np.outer(ys, ns_arr * np.pi / Ly))

    # ── Scatter index: flat mode j → (m, n) position in 2D grid ──────
    # lin_2d[j] = mode_m[j] * n_modes + mode_n[j]  (row-major)
    # So a_2d_flat.at[lin_2d].add(a).reshape(n_modes, n_modes)[m, n] = a[j]
    lin_2d = mode_m * n_modes + mode_n  # 0-indexed, row-major

    # ── Source coordinate grids ────────────────────────────────────────
    X_src, Y_src = np.meshgrid(xs, ys, indexing='ij')  # [nx, ny]

    return Propagator(
        tank=tank,
        mode_m=mode_m, mode_n=mode_n, lin_2d=lin_2d,
        omega=omega, C=C,
        cos_x=cos_x, cos_y=cos_y, dcos_x=dcos_x, dcos_y=dcos_y,
        xs=xs, ys=ys, X_src=X_src, Y_src=Y_src,
        nx=nx, ny=ny, n_modes=n_modes, n_act=n_act,
    )


# ── Transfer matrix and steady-state amplitudes ────────────────────────

def transfer_matrix(
    omega_modes: np.ndarray,
    Omega_drive: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Frequency-domain transfer matrix H[j, k].

    H[j,k] = 1 / (ω_j² - Ω_k² + 2iγω_jΩ_k)

    Parameters
    ----------
    omega_modes : natural frequencies ω_j  [n_total]
    Omega_drive : driving frequencies Ω_k  [n_freq]
    gamma       : modal damping ratio

    Returns
    -------
    H : complex array [n_total, n_freq]
    """
    ω = jnp.asarray(omega_modes)[:, None]   # [n_total, 1]
    Ω = jnp.asarray(Omega_drive)[None, :]   # [1, n_freq]
    return 1.0 / (ω**2 - Ω**2 + 2j * gamma * ω * Ω)


def steady_state_amplitudes(
    prop: Propagator,
    P: jnp.ndarray,
    Omega_freqs: jnp.ndarray,
    T: float,
) -> jnp.ndarray:
    """
    Steady-state modal amplitudes given complex phasor matrix P.

    a = Im( H ⊙ (C @ P) @ exp(iΩT) )

    Parameters
    ----------
    prop        : Propagator
    P           : complex phasor matrix [n_act, n_freq]
    Omega_freqs : driving angular frequencies [n_freq]
    T           : evaluation time (s)

    Returns
    -------
    a : real modal amplitudes [n_total]
    """
    H = transfer_matrix(prop.omega, Omega_freqs, prop.tank.damping)   # [n_total, n_freq]
    alpha = H * (jnp.asarray(prop.C) @ P)                             # [n_total, n_freq]
    E = jnp.exp(1j * Omega_freqs * T)                                  # [n_freq]
    return jnp.imag(alpha @ E)                                         # [n_total]


# ── Parameter packing helpers ──────────────────────────────────────────

def pack_complex(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pack Re and Im parts into a flat real parameter vector [vec(X); vec(Y)]."""
    return np.concatenate([X.ravel(), Y.ravel()])


def unpack_complex(
    params: jnp.ndarray,
    n_act: int,
    n_freq: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Inverse of pack_complex. Returns (X, Y) as [n_act, n_freq] matrices."""
    n = n_act * n_freq
    X = params[:n].reshape(n_act, n_freq)
    Y = params[n:].reshape(n_act, n_freq)
    return X, Y
