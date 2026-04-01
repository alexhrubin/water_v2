"""
Caustic rendering pipeline.

Given modal amplitudes a [n_total], computes the caustic intensity image I [nx, ny]:

  1. Reconstruct surface: a → η, ∂η/∂x, ∂η/∂y  (separable matrix products)
  2. Refraction:  (x,y) → (x_land, y_land)      (paraxial or full Snell's law)
  3. Bilinear splatting: scatter ray densities   (JAX .at[].add(), differentiable)
  4. Separable Gaussian blur                     (smooths the loss landscape)

Phase 1: fully differentiable via JAX autodiff.
Phase 2: caustic_image uses @jax.custom_vjp for an analytical backward pass that
         avoids tape overhead and enables efficient L-BFGS optimization.
"""

import math
from functools import partial
import numpy as np
import jax.numpy as jnp
import jax

from .physics import Propagator


# ── Surface reconstruction ─────────────────────────────────────────────

def reconstruct_surface(
    prop: Propagator,
    a: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Reconstruct surface height and gradients from modal amplitudes.

    η(x,y)      = Σ_{m,n} a_{m,n} cos(mπx/Lx) cos(nπy/Ly)
    ∂η/∂x(x,y) = Σ_{m,n} a_{m,n} ∂cos_x[i,m]/∂x · cos_y[j,n]
    ∂η/∂y(x,y) = Σ_{m,n} a_{m,n} cos_x[i,m] · ∂cos_y[j,n]/∂y

    Uses separable matrix products: O(n_modes · (nx + ny)) instead of O(nx · ny · n_modes²).

    Returns
    -------
    eta, deta_dx, deta_dy : each [nx, ny]
    """
    n_modes = prop.n_modes

    # Scatter flat amplitudes a[j] → 2D grid a_2d[m, n]
    # lin_2d[j] = m * n_modes + n, so a_2d[m, n] = a[j] for mode (m,n)
    a_2d = (jnp.zeros(n_modes * n_modes)
            .at[prop.lin_2d].add(a)
            .reshape(n_modes, n_modes))

    cos_x  = jnp.asarray(prop.cos_x)   # [nx, n_modes]
    cos_y  = jnp.asarray(prop.cos_y)   # [ny, n_modes]
    dcos_x = jnp.asarray(prop.dcos_x)  # [nx, n_modes]
    dcos_y = jnp.asarray(prop.dcos_y)  # [ny, n_modes]

    # η[i,j] = Σ_m Σ_n a_2d[m,n] · cos_x[i,m] · cos_y[j,n]
    eta     = cos_x  @ a_2d @ cos_y.T   # [nx, ny]
    deta_dx = dcos_x @ a_2d @ cos_y.T   # [nx, ny]
    deta_dy = cos_x  @ a_2d @ dcos_y.T  # [nx, ny]

    return eta, deta_dx, deta_dy


def _reconstruct_adjoint(
    prop: Propagator,
    dL_deta: jnp.ndarray,
    dL_detax: jnp.ndarray,
    dL_detay: jnp.ndarray,
) -> jnp.ndarray:
    """
    Adjoint of reconstruct_surface: (∂L/∂η, ∂L/∂ηx, ∂L/∂ηy) → ∂L/∂a.

    From docs/gradient_derivation.md §Step 3 adjoint:
        ∂L/∂a_2d = cos_x.T @ ∂L/∂η  @ cos_y
                 + dcos_x.T @ ∂L/∂ηx @ cos_y
                 + cos_x.T  @ ∂L/∂ηy @ dcos_y
    Then gather: ∂L/∂a[j] = ∂L/∂a_2d[mode_m[j], mode_n[j]]
    """
    cos_x  = jnp.asarray(prop.cos_x)
    cos_y  = jnp.asarray(prop.cos_y)
    dcos_x = jnp.asarray(prop.dcos_x)
    dcos_y = jnp.asarray(prop.dcos_y)

    da_2d = (cos_x.T  @ dL_deta  @ cos_y
           + dcos_x.T @ dL_detax @ cos_y
           + cos_x.T  @ dL_detay @ dcos_y)   # [n_modes, n_modes]

    return da_2d.ravel()[prop.lin_2d]   # gather [n_total]


# ── Refraction ────────────────────────────────────────────────────────

def snell_landing(
    X_src: jnp.ndarray,
    Y_src: jnp.ndarray,
    eta: jnp.ndarray,
    deta_dx: jnp.ndarray,
    deta_dy: jnp.ndarray,
    depth: float,
    n_water: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Full vector Snell's law refraction.

    Incident ray: d_i = (0, 0, -1) (straight down).
    Surface normal: n̂ = normalize(-∂η/∂x, -∂η/∂y, 1).
    Refracted ray traced from the surface to z=0.

    Returns x_land, y_land : each [nx, ny]
    """
    inv_norm = 1.0 / jnp.sqrt(deta_dx**2 + deta_dy**2 + 1.0)
    nx_s = -deta_dx * inv_norm
    ny_s = -deta_dy * inv_norm
    nz_s = inv_norm

    ratio = 1.0 / n_water
    cos_i = nz_s                                       # cos(θ_i) = n̂ · ẑ
    sin2_t = ratio**2 * (1.0 - cos_i**2)
    cos_t = jnp.sqrt(jnp.clip(1.0 - sin2_t, 0.0))    # clip: no total internal reflection

    coeff = ratio * cos_i - cos_t
    dt_x = coeff * nx_s
    dt_y = coeff * ny_s
    dt_z = -ratio + coeff * nz_s

    t_hit = -depth / dt_z
    return X_src + dt_x * t_hit, Y_src + dt_y * t_hit


def _paraxial_landing(
    X_src: jnp.ndarray,
    Y_src: jnp.ndarray,
    eta: jnp.ndarray,
    deta_dx: jnp.ndarray,
    deta_dy: jnp.ndarray,
    depth: float,
    n_water: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Paraxial (small-angle) refraction approximation.

    x_land ≈ x + (depth - η) · (∂η/∂x) / n_water
    """
    ratio = 1.0 / n_water
    x_land = X_src + (depth - eta) * deta_dx * ratio
    y_land = Y_src + (depth - eta) * deta_dy * ratio
    return x_land, y_land


# ── Gaussian blur ─────────────────────────────────────────────────────

def _gaussian_blur_separable(
    M: jnp.ndarray,
    dx: float,
    dy: float,
    sigma: float,
    w: int,
) -> jnp.ndarray:
    """
    Separable 2D Gaussian blur: two 1D passes, O(2 · (2w+1) · nx · ny).

    Parameters
    ----------
    M     : input image [nx, ny]
    dx,dy : grid spacing
    sigma : Gaussian standard deviation
    w     : half-width of kernel in pixels (kernel has 2w+1 taps)
    """
    nx, ny = M.shape
    inv_2s2 = 1.0 / (2.0 * sigma**2)

    # 1D kernel weights (computed outside JAX trace — pure Python/numpy)
    kx = np.array([math.exp(-(i * dx)**2 * inv_2s2) for i in range(-w, w + 1)])
    ky = np.array([math.exp(-(j * dy)**2 * inv_2s2) for j in range(-w, w + 1)])
    kx = jnp.asarray(kx)
    ky = jnp.asarray(ky)

    # Pass 1: blur along axis 0 (x direction)
    pad = jnp.zeros((w, ny))
    M_pad = jnp.concatenate([pad, M, pad], axis=0)        # [nx+2w, ny]
    tmp = sum(kx[i] * M_pad[i : nx + i, :] for i in range(2 * w + 1))

    # Pass 2: blur along axis 1 (y direction)
    pad = jnp.zeros((nx, w))
    tmp_pad = jnp.concatenate([pad, tmp, pad], axis=1)    # [nx, ny+2w]
    return sum(ky[j] * tmp_pad[:, j : ny + j] for j in range(2 * w + 1))


# ── Bilinear splatting helpers ────────────────────────────────────────

def _bilinear_splat(
    ix0: jnp.ndarray,
    iy0: jnp.ndarray,
    wx: jnp.ndarray,
    wy: jnp.ndarray,
    mask: jnp.ndarray,
    nx: int,
    ny: int,
) -> jnp.ndarray:
    """Scatter bilinear weights into a [nx, ny] density image."""
    w00 = mask * (1.0 - wx) * (1.0 - wy)
    w10 = mask * wx          * (1.0 - wy)
    w01 = mask * (1.0 - wx) * wy
    w11 = mask * wx          * wy

    lin00 = ix0       * ny + iy0
    lin10 = (ix0 + 1) * ny + iy0
    lin01 = ix0       * ny + (iy0 + 1)
    lin11 = (ix0 + 1) * ny + (iy0 + 1)

    all_vals = jnp.concatenate([w00.ravel(), w10.ravel(), w01.ravel(), w11.ravel()])
    all_idx  = jnp.concatenate([lin00.ravel(), lin10.ravel(), lin01.ravel(), lin11.ravel()])

    return (jnp.zeros(nx * ny)
            .at[all_idx].add(all_vals)
            .reshape(nx, ny))


def _bilinear_splat_adjoint(
    dL_dD: jnp.ndarray,
    ix0: jnp.ndarray,
    iy0: jnp.ndarray,
    wx: jnp.ndarray,
    wy: jnp.ndarray,
    mask: jnp.ndarray,
    dx: float,
    dy: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Adjoint of bilinear splatting: ∂L/∂D → ∂L/∂(xl), ∂L/∂(yl).

    See docs/gradient_derivation.md §Step 5 adjoint for derivation.

    For each source pixel (i,j), the gradient w.r.t. the fractional position fi is:

        ∂L/∂fi = −G[ix0,iy0]·(1−wy) + G[ix0+1,iy0]·(1−wy)
               − G[ix0,iy0+1]·wy    + G[ix0+1,iy0+1]·wy

    where G = ∂L/∂D, and ∂L/∂(xl) = ∂L/∂fi / dx.
    """
    nx, ny = dL_dD.shape
    G_flat = dL_dD.ravel()   # [nx*ny]

    # Gather G at the four corners for each source pixel
    G00 = G_flat[ix0 * ny + iy0]
    G10 = G_flat[(ix0 + 1) * ny + iy0]
    G01 = G_flat[ix0 * ny + (iy0 + 1)]
    G11 = G_flat[(ix0 + 1) * ny + (iy0 + 1)]

    # ∂L/∂fi (gradient w.r.t. fractional x index)
    dL_dfi = (-G00 * (1.0 - wy) + G10 * (1.0 - wy)
              - G01 * wy         + G11 * wy) * mask

    # ∂L/∂fj (gradient w.r.t. fractional y index)
    dL_dfj = (-G00 * (1.0 - wx) - G10 * wx
              + G01 * (1.0 - wx) + G11 * wx) * mask

    return dL_dfi / dx, dL_dfj / dy


# ── Caustic image with custom VJP ─────────────────────────────────────
#
# JAX's custom_vjp requires all args to be JAX types, OR be declared via
# nondiff_argnums. Propagator contains numpy arrays and is not a JAX type,
# so we mark args (0,2,3,4,5) = (prop, n_water, sigma, cutoff_sigmas,
# full_snell) as nondiff. Only arg 1 (a) is differentiable.
#
# With nondiff_argnums, the backward function receives nondiff args as
# leading positional arguments before the residuals and cotangent.

@partial(jax.custom_vjp, nondiff_argnums=(0, 2, 3, 4, 5))
def _caustic_image_inner(
    prop: Propagator,
    a: jnp.ndarray,
    n_water: float,
    sigma: float,
    cutoff_sigmas: float,
    full_snell: bool,
) -> tuple[np.ndarray, np.ndarray, jnp.ndarray]:
    xs, ys, I, _ = _caustic_image_fwd_core(prop, a, n_water, sigma,
                                            cutoff_sigmas, full_snell)
    return xs, ys, I


def _caustic_image_fwd_core(prop, a, n_water, sigma, cutoff_sigmas, full_snell):
    """Forward computation shared by public API and custom_vjp forward."""
    xs, ys = prop.xs, prop.ys
    nx, ny = prop.nx, prop.ny
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    depth = prop.tank.depth

    sigma = sigma if sigma > 0 else 1.5 * max(dx, dy)
    w = int(math.ceil(cutoff_sigmas * sigma / max(dx, dy)))

    # ── Phase A: surface reconstruction ──────────────────────────────
    eta, deta_dx, deta_dy = reconstruct_surface(prop, a)

    # ── Phase B: ray landing positions ────────────────────────────────
    X_src = jnp.asarray(prop.X_src)
    Y_src = jnp.asarray(prop.Y_src)

    if full_snell:
        x_land, y_land = snell_landing(X_src, Y_src, eta, deta_dx, deta_dy,
                                        depth, n_water)
    else:
        x_land, y_land = _paraxial_landing(X_src, Y_src, eta, deta_dx, deta_dy,
                                            depth, n_water)

    # ── Phase C: bilinear splatting ───────────────────────────────────
    fi_raw = (x_land - xs[0]) / dx
    fj_raw = (y_land - ys[0]) / dy

    in_bounds = ((fi_raw >= 0) & (fi_raw <= nx - 1) &
                 (fj_raw >= 0) & (fj_raw <= ny - 1))
    mask = in_bounds.astype(jnp.float64)

    fi = jnp.clip(fi_raw, 0.0, nx - 1.0)
    fj = jnp.clip(fj_raw, 0.0, ny - 1.0)
    ix0 = jnp.clip(jnp.floor(fi).astype(jnp.int32), 0, nx - 2)
    iy0 = jnp.clip(jnp.floor(fj).astype(jnp.int32), 0, ny - 2)

    wx = fi - ix0.astype(jnp.float64)
    wy = fj - iy0.astype(jnp.float64)

    D = _bilinear_splat(ix0, iy0, wx, wy, mask, nx, ny)

    # ── Phase D: Gaussian blur ────────────────────────────────────────
    I = _gaussian_blur_separable(D, dx, dy, sigma, w)

    residuals = (eta, deta_dx, deta_dy, ix0, iy0, wx, wy, mask, dx, dy, sigma, w,
                 depth, n_water)
    return xs, ys, I, residuals


def _caustic_fwd(prop, a, n_water, sigma, cutoff_sigmas, full_snell):
    # nondiff args (prop, n_water, sigma, cutoff_sigmas, full_snell) are
    # implicit context here; the only traced arg is a.
    xs, ys, I, residuals = _caustic_image_fwd_core(prop, a, n_water, sigma,
                                                    cutoff_sigmas, full_snell)
    return (xs, ys, I), residuals


def _caustic_bwd(prop, n_water, sigma, cutoff_sigmas, full_snell, residuals, g):
    # With nondiff_argnums=(0,2,3,4,5), JAX prepends the nondiff args.
    # residuals: saved from fwd; g: cotangent tuple (dL/dxs, dL/dys, dL/dI)
    (eta, deta_dx, deta_dy,
     ix0, iy0, wx, wy, mask,
     dx, dy, sigma_r, w, depth, n_water_r) = residuals

    dL_dI = g[2]   # gradient w.r.t. I (xs, ys are non-differentiable)

    # ── Step 6 adjoint: Gaussian blur is self-adjoint ─────────────────
    dL_dD = _gaussian_blur_separable(dL_dI, dx, dy, sigma_r, w)

    # ── Step 5 adjoint: bilinear splat ───────────────────────────────
    dL_dxl, dL_dyl = _bilinear_splat_adjoint(dL_dD, ix0, iy0, wx, wy, mask,
                                              dx, dy)

    # ── Step 4 adjoint: paraxial refraction ──────────────────────────
    inv_n = 1.0 / n_water_r
    scale = (depth - eta) * inv_n
    dL_detax = dL_dxl * scale
    dL_detay = dL_dyl * scale
    dL_deta  = -dL_dxl * deta_dx * inv_n - dL_dyl * deta_dy * inv_n

    # ── Step 3 adjoint: surface reconstruction ───────────────────────
    dL_da = _reconstruct_adjoint(prop, dL_deta, dL_detax, dL_detay)

    return (dL_da,)   # gradient for the single diff arg (a)


_caustic_image_inner.defvjp(_caustic_fwd, _caustic_bwd)


def caustic_image(
    prop: Propagator,
    a: jnp.ndarray,
    *,
    n_water: float = 1.33,
    sigma: float = 0.0,
    cutoff_sigmas: float = 4.0,
    full_snell: bool = False,
) -> tuple[np.ndarray, np.ndarray, jnp.ndarray]:
    """
    Render caustic intensity image from modal amplitudes a.

    Pipeline:
      a → (η, ∂η/∂x, ∂η/∂y) → (x_land, y_land) → bilinear splat D → Gaussian blur → I

    Differentiable w.r.t. a via a custom analytical VJP (Phase 2).
    The custom VJP avoids tape overhead and computes the exact adjoint through
    blur → splat → paraxial → reconstruct. See docs/gradient_derivation.md.

    Parameters
    ----------
    prop         : Propagator
    a            : modal amplitudes [n_total]
    n_water      : refractive index of water (default 1.33)
    sigma        : Gaussian blur std (m); default = 1.5 × max grid spacing
    cutoff_sigmas: blur kernel half-width in units of sigma
    full_snell   : if True, use full vector Snell's law; else paraxial approx

    Returns
    -------
    xs, ys : grid coordinates (numpy)
    I      : caustic intensity image [nx, ny] (JAX array)
    """
    return _caustic_image_inner(prop, a, n_water, sigma, cutoff_sigmas, full_snell)
