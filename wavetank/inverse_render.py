"""
Inverse rendering of caustics: find a refractive surface η(x, y) that, when
ray-traced through the Snell + splat + blur pipeline, produces a target
caustic image.

Approach: gradient descent on η directly using JAX autodiff through the
existing differentiable renderer (`wavetank/render.py`).  This is the
modern alternative to OT-based caustic engineering — it directly minimises
the discrepancy between the rendered caustic and the target, with no
proxy quantities or source-vs-destination ambiguity.

The η produced here is the *unconstrained* optical-inverse ceiling for
the given apparatus. To bring it under the water-wave constraint, project
η onto the wave eigenspace with `wavetank.surface_solver.project_to_modes`
and use it as the target for the wave optimisation.
"""

from __future__ import annotations
import math
import numpy as np
import jax
import jax.numpy as jnp
import jaxopt

from .render import (
    snell_landing, _paraxial_landing,
    _bilinear_splat, _gaussian_blur_separable,
    reconstruct_surface, caustic_image,
)
from .physics import Propagator
from .loss import ssim_loss


# ── Forward renderer that takes η directly ────────────────────────────

def caustic_from_eta(
    eta: jnp.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    throw: float,
    *,
    n_water: float = 1.33,
    sigma: float = 0.01,
    full_snell: bool = True,
) -> jnp.ndarray:
    """
    Render a caustic intensity image from a raw surface-height field η.

    The η gradients are computed via central differences on the grid (with
    one-sided differences at the boundary).  Everything stays in JAX so
    that ``jax.grad(caustic_from_eta)`` differentiates correctly w.r.t. η.

    Parameters
    ----------
    eta        : (nx, ny) JAX array, surface height in metres (zero-mean)
    xs, ys     : 1D coordinate arrays (length nx, ny)
    throw      : optical projection distance (m)
    n_water    : refractive index
    sigma      : Gaussian blur std (m) applied to the splatted density
    full_snell : if True, use full vector Snell's law; else paraxial

    Returns
    -------
    I : (nx, ny) caustic intensity image (JAX array)
    """
    nx, ny = eta.shape
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])

    # Central differences for gradients of η. One-sided at boundaries.
    eta_x = jnp.zeros_like(eta)
    eta_x = eta_x.at[1:-1, :].set((eta[2:, :] - eta[:-2, :]) / (2.0 * dx))
    eta_x = eta_x.at[0,    :].set((eta[1, :] - eta[0, :]) / dx)
    eta_x = eta_x.at[-1,   :].set((eta[-1, :] - eta[-2, :]) / dx)

    eta_y = jnp.zeros_like(eta)
    eta_y = eta_y.at[:, 1:-1].set((eta[:, 2:] - eta[:, :-2]) / (2.0 * dy))
    eta_y = eta_y.at[:,    0].set((eta[:, 1] - eta[:, 0]) / dy)
    eta_y = eta_y.at[:,   -1].set((eta[:, -1] - eta[:, -2]) / dy)

    X_src, Y_src = jnp.meshgrid(jnp.asarray(xs), jnp.asarray(ys), indexing='ij')

    if full_snell:
        x_land, y_land = snell_landing(X_src, Y_src, eta, eta_x, eta_y,
                                        throw, n_water)
    else:
        x_land, y_land = _paraxial_landing(X_src, Y_src, eta, eta_x, eta_y,
                                            throw, n_water)

    # Bilinear splat (matches caustic_image's renderer)
    fi = (x_land - xs[0]) / dx
    fj = (y_land - ys[0]) / dy
    in_bounds = ((fi >= 0) & (fi <= nx - 1) &
                 (fj >= 0) & (fj <= ny - 1))
    mask = in_bounds.astype(jnp.float64)
    fi_c = jnp.clip(fi, 0.0, nx - 1.0)
    fj_c = jnp.clip(fj, 0.0, ny - 1.0)
    ix0 = jnp.clip(jnp.floor(fi_c).astype(jnp.int32), 0, nx - 2)
    iy0 = jnp.clip(jnp.floor(fj_c).astype(jnp.int32), 0, ny - 2)
    wx = fi_c - ix0.astype(jnp.float64)
    wy = fj_c - iy0.astype(jnp.float64)

    D = _bilinear_splat(ix0, iy0, wx, wy, mask, nx, ny)

    sigma = sigma if sigma > 0 else 1.5 * max(dx, dy)
    w = int(math.ceil(4.0 * sigma / max(dx, dy)))
    I = _gaussian_blur_separable(D, dx, dy, sigma, w)
    return I


# ── η optimiser ────────────────────────────────────────────────────────

def optimize_eta_for_target(
    target: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    throw: float,
    *,
    n_water: float = 1.33,
    sigma: float = 0.01,
    full_snell: bool = True,
    n_iter: int = 300,
    lambda_smooth: float = 0.0,
    eta_init: np.ndarray | None = None,
    loss_type: str = 'cosine',
    verbose: bool = False,
) -> dict:
    """
    Find η(x, y) such that ``caustic_from_eta(η)`` matches ``target``.

    L-BFGS on a flattened η field, with autodiff gradients through the
    rendering pipeline.  Optional Tikhonov-style smoothness penalty on
    ``∇²η`` discourages high-frequency speckle in the solution (useful
    when the target has dim continuous-tone regions where the optimum
    is otherwise under-determined).

    Parameters
    ----------
    target        : (nx, ny) target caustic image, non-negative.
    xs, ys        : grid coordinates.
    throw         : optical projection distance (m).
    n_water       : refractive index.
    sigma         : Gaussian blur in the renderer (m).
    full_snell    : if True, use full Snell; else paraxial.
    n_iter        : L-BFGS max iterations.
    lambda_smooth : strength of the ∇²η smoothness penalty (0 = off).
    eta_init      : optional warm-start η (e.g. from the OT solver).
    loss_type     : 'cosine' (1 - cos_sim, robust to brightness) or
                    'mse'   (||I/sum - target/sum||²).
    verbose       : print L-BFGS state at the end.

    Returns
    -------
    dict with keys:
      'eta'          : optimised surface (nx, ny), m
      'caustic'      : rendered caustic from eta
      'final_loss'   : data + smoothness loss at the optimum
      'iter_num'     : iterations actually run
    """
    nx, ny = target.shape
    target = np.asarray(target, dtype=np.float64)
    target = target / target.sum()                  # normalise to unit mass
    target_j = jnp.asarray(target)

    dx_v = float(xs[1] - xs[0])
    dy_v = float(ys[1] - ys[0])

    def render(eta):
        return caustic_from_eta(eta, xs, ys, throw,
                                 n_water=n_water, sigma=sigma,
                                 full_snell=full_snell)

    if loss_type == 'cosine':
        def data_loss(eta):
            I = render(eta)
            num = jnp.sum(I * target_j)
            den = jnp.sqrt(jnp.sum(I * I) + 1e-30) * jnp.sqrt(jnp.sum(target_j * target_j) + 1e-30)
            return 1.0 - num / den
    elif loss_type == 'mse':
        def data_loss(eta):
            I = render(eta)
            I_norm = I / (jnp.sum(I) + 1e-30)
            return jnp.sum((I_norm - target_j) ** 2)
    elif loss_type == 'ssim':
        # SSIM compares structure / luminance / contrast in local windows.
        # Much less gameable than cosine — bright-but-featureless caustics
        # score low because they lack local structure agreement.
        target_unnorm = jnp.asarray(np.asarray(target) * float(np.asarray(target).sum()))
        def data_loss(eta):
            I = render(eta)
            return ssim_loss(I, target_unnorm, dx_v, dy_v)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

    def smooth_loss(eta):
        lap = (
            (eta[:-2, 1:-1] + eta[2:, 1:-1] - 2.0 * eta[1:-1, 1:-1]) / (dx_v * dx_v) +
            (eta[1:-1, :-2] + eta[1:-1, 2:] - 2.0 * eta[1:-1, 1:-1]) / (dy_v * dy_v)
        )
        return jnp.sum(lap * lap)

    def total_loss(eta_flat):
        eta = eta_flat.reshape(nx, ny)
        L = data_loss(eta)
        if lambda_smooth > 0.0:
            L = L + lambda_smooth * smooth_loss(eta)
        return L

    if eta_init is None:
        eta0 = jnp.zeros(nx * ny, dtype=jnp.float64)
    else:
        eta0 = jnp.asarray(eta_init, dtype=jnp.float64).reshape(-1)

    solver = jaxopt.LBFGS(fun=total_loss, maxiter=n_iter, tol=1e-12)
    eta_opt_flat, state = solver.run(eta0)
    eta_opt = np.asarray(eta_opt_flat.reshape(nx, ny))

    I_opt = np.asarray(render(jnp.asarray(eta_opt)))

    if verbose:
        print(f"  L-BFGS: {int(state.iter_num)} iters, loss = {float(state.value):.4g}")

    return {
        'eta': eta_opt,
        'caustic': I_opt,
        'final_loss': float(state.value),
        'iter_num': int(state.iter_num),
    }


# ── Mode-basis variant: η lives in the wave eigenspace ─────────────────

def optimize_modal_eta_for_target(
    target: np.ndarray,
    prop: Propagator,
    *,
    n_water: float = 1.33,
    sigma: float = 0.01,
    full_snell: bool = True,
    n_iter: int = 300,
    a_init: np.ndarray | None = None,
    loss_type: str = 'cosine',
    slope_cap: float | None = None,
    lambda_slope: float = 1.0,
    verbose: bool = False,
) -> dict:
    """
    Find modal amplitudes ``a`` such that the caustic rendered from
    ``η = reconstruct_surface(prop, a)`` matches ``target``.

    Same paradigm as ``optimize_eta_for_target``, but the surface is
    constrained to the cosine eigenmode basis of ``prop``. This is the
    achievable ceiling under the *mode-truncation* constraint that water
    waves impose — the surface cannot have content beyond ``prop.n_modes``
    cosines per side. (The further constraint that ``a`` be producible by
    actuator phasors is handled separately by ``optimize_caustic``.)

    Throw is taken from ``prop.tank.throw``. The renderer is the existing
    ``caustic_image`` with its analytical custom VJP, so each L-BFGS step
    is fast.

    Parameters
    ----------
    target     : (nx, ny) target caustic image, non-negative.
    prop       : Propagator (defines the wave eigenspace and throw).
    n_water    : refractive index.
    sigma      : Gaussian blur (m) in the renderer.
    full_snell : if True, full Snell; else paraxial.
    n_iter     : L-BFGS max iterations.
    a_init     : optional warm-start modal amplitudes (e.g. from
                 ``project_to_modes`` of an OT or free-η solution).
    loss_type  : 'cosine' or 'mse'.
    slope_cap  : if set, add a soft hinge penalty discouraging
                 ``max |∇η| > slope_cap``. Use 0.1 for the linear-wave
                 regime, ~0.25 for M=2, ~0.4 for M=3.
    lambda_slope : strength of the slope-penalty hinge. The penalty is
                 ``λ · sum( max(|∇η|² − cap², 0) )`` summed over pixels.
    verbose    : print L-BFGS state at the end.

    Returns
    -------
    dict with keys:
      'a'          : optimised modal amplitudes (n_total,)
      'eta'        : reconstructed surface (nx, ny), m
      'caustic'    : rendered caustic
      'final_loss' : loss at the optimum
      'iter_num'   : iterations actually run
    """
    target = np.asarray(target, dtype=np.float64)
    target_unnorm = jnp.asarray(target)            # for SSIM (wants raw scale)
    target = target / target.sum()
    target_j = jnp.asarray(target)
    n_total = len(prop.omega)

    dx_v = float(prop.xs[1] - prop.xs[0])
    dy_v = float(prop.ys[1] - prop.ys[0])

    def render(a):
        _, _, I = caustic_image(prop, a, n_water=n_water,
                                 sigma=sigma, full_snell=full_snell)
        return I

    def slope_penalty(a):
        """Sum of squared excess slope over the slope cap. Zero when
        every pixel has |∇η| ≤ slope_cap."""
        if slope_cap is None:
            return jnp.float64(0.0)
        _, eta_x, eta_y = reconstruct_surface(prop, a)
        slope2 = eta_x ** 2 + eta_y ** 2
        excess = jnp.maximum(slope2 - slope_cap ** 2, 0.0)
        return jnp.sum(excess)

    if loss_type == 'cosine':
        def data_loss(a):
            I = render(a)
            num = jnp.sum(I * target_j)
            den = (jnp.sqrt(jnp.sum(I * I) + 1e-30)
                   * jnp.sqrt(jnp.sum(target_j * target_j) + 1e-30))
            return 1.0 - num / den
    elif loss_type == 'mse':
        def data_loss(a):
            I = render(a)
            I_norm = I / (jnp.sum(I) + 1e-30)
            return jnp.sum((I_norm - target_j) ** 2)
    elif loss_type == 'ssim':
        def data_loss(a):
            I = render(a)
            return ssim_loss(I, target_unnorm, dx_v, dy_v)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

    def loss(a):
        L = data_loss(a)
        if slope_cap is not None:
            L = L + lambda_slope * slope_penalty(a)
        return L

    if a_init is None:
        a0 = jnp.zeros(n_total, dtype=jnp.float64)
    else:
        a0 = jnp.asarray(a_init, dtype=jnp.float64)

    solver = jaxopt.LBFGS(fun=loss, maxiter=n_iter, tol=1e-12)
    a_opt, state = solver.run(a0)
    a_opt_np = np.asarray(a_opt)

    eta_opt, _, _ = reconstruct_surface(prop, a_opt)
    eta_opt = np.asarray(eta_opt)
    I_opt = np.asarray(render(a_opt))

    if verbose:
        print(f"  L-BFGS modal: {int(state.iter_num)} iters, "
              f"loss = {float(state.value):.4g}")

    return {
        'a': a_opt_np,
        'eta': eta_opt,
        'caustic': I_opt,
        'final_loss': float(state.value),
        'iter_num': int(state.iter_num),
    }
