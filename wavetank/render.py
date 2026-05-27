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
    throw: float,
    n_water: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Full vector Snell's law refraction.

    Incident ray: d_i = (0, 0, -1) (straight down).
    Surface normal: n̂ = normalize(-∂η/∂x, -∂η/∂y, 1).
    Refracted ray traced from the surface to z=0.

    ``throw`` is the optical projection distance from the mean water
    surface to the screen (= ``tank.depth`` for a flat-bottom tank, or
    ``depth + d_air`` for an elevated glass-bottom tank).

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

    t_hit = -(throw - eta) / dt_z
    return X_src + dt_x * t_hit, Y_src + dt_y * t_hit


def _paraxial_landing(
    X_src: jnp.ndarray,
    Y_src: jnp.ndarray,
    eta: jnp.ndarray,
    deta_dx: jnp.ndarray,
    deta_dy: jnp.ndarray,
    throw: float,
    n_water: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Paraxial (small-angle) refraction approximation.

    x_land ≈ x + (throw - η) · (∂η/∂x) · (1 - 1/n_water)

    The (1 - 1/n_water) factor is the standard paraxial Snell deflection
    coefficient for an air-to-water interface (taking small angles of the
    full vector formula `d_t · ẑ_perp = (1-1/n)·∇η`). This matches
    ``snell_landing`` in the small-slope limit.

    ``throw`` is the optical projection distance (= tank.throw); see
    ``snell_landing`` for the physical interpretation.
    """
    ratio = 1.0 - 1.0 / n_water        # paraxial Snell deflection coefficient
    x_land = X_src + (throw - eta) * deta_dx * ratio
    y_land = Y_src + (throw - eta) * deta_dy * ratio
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
    throw = prop.tank.throw

    sigma = sigma if sigma > 0 else 1.5 * max(dx, dy)
    w = int(math.ceil(cutoff_sigmas * sigma / max(dx, dy)))

    # ── Phase A: surface reconstruction ──────────────────────────────
    eta, deta_dx, deta_dy = reconstruct_surface(prop, a)

    # ── Phase B: ray landing positions ────────────────────────────────
    X_src = jnp.asarray(prop.X_src)
    Y_src = jnp.asarray(prop.Y_src)

    if full_snell:
        x_land, y_land = snell_landing(X_src, Y_src, eta, deta_dx, deta_dy,
                                        throw, n_water)
    else:
        x_land, y_land = _paraxial_landing(X_src, Y_src, eta, deta_dx, deta_dy,
                                            throw, n_water)

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
                 throw, n_water)
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
     dx, dy, sigma_r, w, throw, n_water_r) = residuals

    dL_dI = g[2]   # gradient w.r.t. I (xs, ys are non-differentiable)

    # ── Step 6 adjoint: Gaussian blur is self-adjoint ─────────────────
    dL_dD = _gaussian_blur_separable(dL_dI, dx, dy, sigma_r, w)

    # ── Step 5 adjoint: bilinear splat ───────────────────────────────
    dL_dxl, dL_dyl = _bilinear_splat_adjoint(dL_dD, ix0, iy0, wx, wy, mask,
                                              dx, dy)

    # ── Step 4 adjoint: refraction ──────────────────────────────────
    if full_snell:
        # Use JAX autodiff through full vector Snell's law.
        # This avoids hand-deriving the complex Snell adjoint while
        # keeping the rest of the backward pass analytical.
        X_src = jnp.asarray(prop.X_src)
        Y_src = jnp.asarray(prop.Y_src)
        def _snell_fn(eta_, dx_, dy_):
            return snell_landing(X_src, Y_src, eta_, dx_, dy_, throw, n_water_r)
        _, vjp_fn = jax.vjp(_snell_fn, eta, deta_dx, deta_dy)
        dL_deta, dL_detax, dL_detay = vjp_fn((dL_dxl, dL_dyl))
    else:
        # Paraxial Snell deflection coefficient (matches _paraxial_landing).
        coeff = 1.0 - 1.0 / n_water_r
        scale = (throw - eta) * coeff
        dL_detax = dL_dxl * scale
        dL_detay = dL_dyl * scale
        dL_deta  = -dL_dxl * deta_dx * coeff - dL_dyl * deta_dy * coeff

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


# ── Jacobian-based caustic renderer (Wallace-style) ───────────────────
#
# Mirrors the per-fragment area-ratio shading from Wallace's WebGL water
# demo (https://madebyevan.com/webgl-water/, the article describing the
# technique is https://medium.com/@evanwallace/rendering-realtime-caustics-in-webgl-2a99a29a0b2c).
#
# Wallace's algorithm:
#   1. Vertex shader maps each source-grid vertex to its post-refraction
#      landing position (x_land, y_land) on the floor. The mesh is then
#      rasterized in floor-space — triangles are drawn at their deformed
#      positions on the floor.
#   2. Fragment shader uses screen-space derivatives (dFdx, dFdy) on the
#      interpolated source position varying to compute, per floor pixel,
#      the local Jacobian determinant of the (X_src, Y_src) → (x_land,
#      y_land) map. Intensity = source_area / floor_area = 1/|det J|.
#   3. Triangles that fold over on the floor (caustic regions) accumulate
#      additively via GPU blending — the caustic line emerges naturally
#      as the sum of overlapping 1/|det J| contributions.
#
# This JAX implementation captures the same mathematical content:
#   1. Compute J per source point via finite differences on the regular
#      source grid (same numerical character as Wallace's dFdx/dFdy).
#   2. Build the covariance of the deformed source cell on the floor:
#      Σ = J · diag(src_dx²/12, src_dy²/12) · J^T
#      (variance of a uniform-distribution rectangle of size (src_dx,
#      src_dy) mapped through the local linear approximation J).
#   3. Splat per source point an anisotropic Gaussian of total mass
#      src_dx · src_dy at the landing position, with covariance Σ.
#   4. Floor intensity = sum of splats. Folds and caustic accumulation
#      emerge from overlapping splats.
#
# In the continuum limit this produces the same intensity field as the
# splat renderer above (both equal 1/|det J| per the change-of-variables
# theorem). At finite resolution they differ: the J-renderer's per-point
# Gaussian shape carries the local cell-deformation information, so
# caustic lines emerge sharply (small Σ → tight Gaussian) and defocus
# regions are smooth (large Σ → broad Gaussian). The splat renderer
# uses a fixed-size bilinear point splat that doesn't see the local J.


def _aniso_gaussian_splat(
    x_land: jnp.ndarray,        # [nx, ny] landing x-positions
    y_land: jnp.ndarray,        # [nx, ny] landing y-positions
    cov_xx: jnp.ndarray,        # [nx, ny] Σ_xx
    cov_xy: jnp.ndarray,        # [nx, ny] Σ_xy
    cov_yy: jnp.ndarray,        # [nx, ny] Σ_yy
    mass:   jnp.ndarray,        # [nx, ny] total mass per source point
    xs: jnp.ndarray,            # [nx_floor] floor x-coords
    ys: jnp.ndarray,            # [ny_floor] floor y-coords
    kernel_half: int,           # half-width of splat kernel in floor pixels
) -> jnp.ndarray:
    """Splat per-source-point anisotropic Gaussians onto a floor grid.

    For each source point (i, j), evaluates the 2-D Gaussian
        G(p) = (mass/(2π√|Σ|)) · exp(-½ (p-μ)^T Σ^{-1} (p-μ))
    at a (2K+1)×(2K+1) neighborhood of floor pixels around the landing
    position, and accumulates the values into the floor grid via
    scatter-add. JAX-compatible; autodiff flows through the Gaussian
    weights and the scatter.
    """
    nx_floor = xs.shape[0]
    ny_floor = ys.shape[0]
    dx_floor = xs[1] - xs[0]
    dy_floor = ys[1] - ys[0]

    # Nearest floor pixel index per source point
    ix_base = jnp.clip(
        jnp.round((x_land - xs[0]) / dx_floor).astype(jnp.int32),
        0, nx_floor - 1,
    )                                                     # [nx, ny]
    iy_base = jnp.clip(
        jnp.round((y_land - ys[0]) / dy_floor).astype(jnp.int32),
        0, ny_floor - 1,
    )                                                     # [nx, ny]

    # Local kernel grid of relative offsets
    K = 2 * kernel_half + 1
    offs = jnp.arange(-kernel_half, kernel_half + 1)
    ox, oy = jnp.meshgrid(offs, offs, indexing='ij')      # [K, K]

    # Absolute pixel indices for each (src, kernel) pair: [nx, ny, K, K]
    ix = ix_base[..., None, None] + ox[None, None, :, :]
    iy = iy_base[..., None, None] + oy[None, None, :, :]

    # Pixel center world coords
    px = xs[0] + ix.astype(x_land.dtype) * dx_floor
    py = ys[0] + iy.astype(y_land.dtype) * dy_floor

    # Δ from landing position
    dxp = px - x_land[..., None, None]
    dyp = py - y_land[..., None, None]

    # Gaussian via Σ^{-1} = (1/|Σ|)·[[Σ_yy, -Σ_xy], [-Σ_xy, Σ_xx]]
    det_cov = cov_xx * cov_yy - cov_xy * cov_xy           # [nx, ny]
    det_cov = jnp.maximum(det_cov, 1e-24)                  # safety
    inv_norm = 1.0 / (2 * jnp.pi * jnp.sqrt(det_cov))      # [nx, ny]

    quadform = (
        cov_yy[..., None, None] * dxp * dxp
        - 2.0 * cov_xy[..., None, None] * dxp * dyp
        + cov_xx[..., None, None] * dyp * dyp
    ) / det_cov[..., None, None]
    g = jnp.exp(-0.5 * quadform) * inv_norm[..., None, None]   # [nx, ny, K, K]

    # Each pixel collects (mass · G · pixel_area)
    contribution = mass[..., None, None] * g * (dx_floor * dy_floor)

    # Mask out-of-bounds pixels (kept index clipped but contribution zeroed)
    in_bounds = (
        (ix >= 0) & (ix < nx_floor) &
        (iy >= 0) & (iy < ny_floor)
    )
    contribution = jnp.where(in_bounds, contribution, 0.0)
    ix = jnp.clip(ix, 0, nx_floor - 1)
    iy = jnp.clip(iy, 0, ny_floor - 1)

    # Scatter-add: flatten to 1D linear floor index
    linear_idx = ix * ny_floor + iy                       # [nx, ny, K, K]
    floor = (
        jnp.zeros(nx_floor * ny_floor, dtype=contribution.dtype)
        .at[linear_idx.ravel()]
        .add(contribution.ravel())
        .reshape(nx_floor, ny_floor)
    )
    return floor


def caustic_image_jacobian(
    prop: Propagator,
    a: jnp.ndarray,
    *,
    n_water: float = 1.33,
    full_snell: bool = False,
    kernel_half: int = 4,
    sigma_floor_pixels: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, jnp.ndarray]:
    """Jacobian-based caustic renderer mirroring Wallace's GPU shader.

    Parameters
    ----------
    prop, a            : as in caustic_image
    n_water            : refractive index
    full_snell         : if True, use full vector Snell's law (slower).
                         The Jacobian is finite-differenced from the same
                         landing-position grid regardless.
    kernel_half        : per-source-point splat kernel half-extent in
                         floor pixels (so kernel is (2K+1)×(2K+1)).
                         Larger captures more of the Gaussian tail in
                         defocus regions but costs more compute. Default
                         4 gives a 9×9 kernel and is sufficient for
                         typical caustic regimes.
    sigma_floor_pixels : regularization floor on the splat sigma, in
                         floor-pixel units. Prevents the caustic
                         singularity (|det J| → 0) from collapsing the
                         Gaussian below sub-pixel scale where the
                         discretized splat loses mass conservation.
                         Default 0.5 = half a floor pixel.

    Returns
    -------
    xs, ys : floor grid coordinates (numpy)
    I      : caustic intensity image [nx, ny] (JAX array)
    """
    xs, ys = prop.xs, prop.ys
    nx, ny = prop.nx, prop.ny
    dx_floor = float(xs[1] - xs[0])
    dy_floor = float(ys[1] - ys[0])
    throw = prop.tank.throw

    eta, deta_dx, deta_dy = reconstruct_surface(prop, a)

    X_src = jnp.asarray(prop.X_src)
    Y_src = jnp.asarray(prop.Y_src)

    if full_snell:
        x_land, y_land = snell_landing(X_src, Y_src, eta, deta_dx, deta_dy,
                                        throw, n_water)
    else:
        x_land, y_land = _paraxial_landing(X_src, Y_src, eta, deta_dx, deta_dy,
                                            throw, n_water)

    # Jacobian via centered finite differences on the regular source grid
    # (mirrors Wallace's dFdx/dFdy). Source grid spacing:
    src_dx = float(prop.tank.Lx / (nx - 1))
    src_dy = float(prop.tank.Ly / (ny - 1))

    # Centered diffs in interior; one-sided at edges (jnp.gradient handles this)
    dxl_dXs, dxl_dYs = jnp.gradient(x_land, src_dx, src_dy)
    dyl_dXs, dyl_dYs = jnp.gradient(y_land, src_dx, src_dy)

    # Deformed-cell covariance: Σ = J · diag(src_dx²/12, src_dy²/12) · J^T
    s_xx = src_dx * src_dx / 12.0
    s_yy = src_dy * src_dy / 12.0
    cov_xx = dxl_dXs * dxl_dXs * s_xx + dxl_dYs * dxl_dYs * s_yy
    cov_xy = dxl_dXs * dyl_dXs * s_xx + dxl_dYs * dyl_dYs * s_yy
    cov_yy = dyl_dXs * dyl_dXs * s_xx + dyl_dYs * dyl_dYs * s_yy

    # Regularize: don't let the Gaussian collapse below a sub-pixel floor
    # (otherwise mass is concentrated in <1 pixel and discretization
    # loses conservation). Inflate the covariance isotropically:
    cov_min = (sigma_floor_pixels * max(dx_floor, dy_floor)) ** 2
    cov_xx = cov_xx + cov_min
    cov_yy = cov_yy + cov_min

    # Total mass per source point = source-cell area (constant)
    mass = jnp.full_like(x_land, src_dx * src_dy)

    I = _aniso_gaussian_splat(
        x_land, y_land, cov_xx, cov_xy, cov_yy, mass,
        jnp.asarray(xs), jnp.asarray(ys),
        kernel_half=kernel_half,
    )

    return xs, ys, I
