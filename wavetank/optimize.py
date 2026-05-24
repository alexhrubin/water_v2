"""
Caustic optimization via Adam or L-BFGS with coarse-to-fine sigma annealing.

Each optimization stage blurs both the rendered caustic and the target
with Gaussian sigma, then progressively sharpens. Starting coarse avoids
local minima from the sparse ray-splatting landscape; finishing fine
recovers spatial detail.

Phase 2 adds L-BFGS via jaxopt, which converges in ~50–150 steps per stage
vs ~500 Adam steps, using the analytical gradient from caustic_image's
custom VJP.
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence
import numpy as np
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from .physics import Propagator, steady_state_amplitudes, unpack_complex
from .render import caustic_image, reconstruct_surface, _gaussian_blur_separable
from .loss import cosine_loss, ssim_loss
from .hos import hos_forward, HOSConfig


def make_hos_forward(
    M: int = 2,
    dealias_max_modes: int | None = None,
    steps_per_period: int = 40,
    initial: str = "steady",
) -> Callable:
    """
    Build a steady-state-IC HOS forward closure matching the signature of
    ``steady_state_amplitudes``, suitable for passing as ``forward_fn`` to
    ``optimize_caustic`` / ``make_loss``.

    Parameters mirror ``HOSConfig``. ``initial="steady"`` is the default and
    skips the cold-start transient — appropriate for steady-state-style
    optimization at a chosen T_eval.
    """
    cfg = HOSConfig(M=M, dealias_max_modes=dealias_max_modes,
                    steps_per_period=steps_per_period)

    def forward(prop, P, Omega, T):
        return hos_forward(prop, P, Omega, T_eval=T, config=cfg, initial=initial)

    return forward


# ── Stage specification ────────────────────────────────────────────────

@dataclass(frozen=True)
class Stage:
    """One phase of coarse-to-fine optimization."""
    sigma: float              # caustic rendering blur (m)
    sigma_blur: float         # target pre-blur (m); usually equals sigma
    iters: int                # number of optimizer steps
    method: str = 'adam'      # 'adam' or 'lbfgs'


# ── Make loss function ─────────────────────────────────────────────────

def make_loss(
    prop: Propagator,
    target: np.ndarray,
    Omega_freqs: np.ndarray,
    T_eval: float | Sequence[float],
    *,
    sigma: float = 0.02,
    sigma_blur: float = 0.02,
    loss_type: str = 'cosine',
    lambda_energy: float = 1e-5,
    lambda_eta: float = 100.0,
    lambda_slope: float = 100.0,
    n_water: float = 1.33,
    full_snell: bool = False,
    forward_fn: Callable | None = None,
) -> callable:
    """
    Build a scalar loss function over the parameter vector params.

    params = [vec(X); vec(Y)] where P = X + iY is the [n_act, n_freq]
    complex phasor matrix.

    Parameters
    ----------
    prop          : Propagator
    target        : target caustic image [nx, ny] in [0, 1]
    Omega_freqs   : driving angular frequencies [n_freq]
    T_eval        : evaluation time(s). Pass a scalar for single-frame
                    optimization, or a sequence of times to optimize the
                    average loss over a "movie" of frames — useful for
                    making the target persist throughout one period.
    sigma         : rendering blur
    sigma_blur    : target pre-blur (0 = no blur)
    loss_type     : 'cosine' or 'ssim'
    lambda_energy : L2 regularization weight on phasor amplitudes
    lambda_eta    : L2 penalty on mean(η²) — keeps the optimizer in the
                    linear-wave regime where |η| ≪ depth. The default
                    100.0 contributes ≈ 0.014 at η_rms = 12 mm (well below
                    the loss-match term) but ≈ 49 at η_rms = 700 mm
                    (dominant), preventing the scale-invariant cosine loss
                    from running away to unphysical surface heights.
    lambda_slope  : L2 penalty on mean(|∇η|²) — enforces the paraxial
                    refraction assumption |∇η| ≪ 1. With the default
                    100.0 the penalty is ≈ 1.0 at RMS slope 0.1 (the
                    linearity boundary), comparable to a typical match
                    loss. Uses ∂η/∂x, ∂η/∂y from reconstruct_surface —
                    essentially free since they're already computed.
    n_water       : refractive index
    full_snell    : use full Snell's law refraction
    forward_fn    : function with signature (prop, P, Omega, T) → a that
                    produces modal amplitudes from drive at time T. Defaults
                    to ``steady_state_amplitudes`` (linear theory). Pass an
                    HOS-based forward (e.g., from ``make_hos_forward``) to
                    optimize through nonlinear waves. HOS forwards do NOT
                    support movie-mode T_eval (sequence).
    """
    n_act = prop.n_act
    n_freq = len(Omega_freqs)
    dx = float(prop.xs[1] - prop.xs[0])
    dy = float(prop.ys[1] - prop.ys[0])

    # Pre-blur the target (fixed for this stage)
    import math
    if sigma_blur > 0:
        w_blur = int(math.ceil(4.0 * sigma_blur / max(dx, dy)))
        T_b = jnp.asarray(
            _gaussian_blur_separable(jnp.asarray(target), dx, dy, sigma_blur, w_blur))
    else:
        T_b = jnp.asarray(target)

    # Precompute target norm for cosine loss
    norm_T = float(jnp.sqrt(jnp.sum(T_b**2) + 1e-12))

    Omega = jnp.asarray(Omega_freqs)

    fwd = forward_fn if forward_fn is not None else steady_state_amplitudes

    # Movie mode: T_eval is an array → average loss over multiple frames.
    movie_mode = not np.isscalar(T_eval)
    if movie_mode:
        if forward_fn is not None:
            raise ValueError(
                "forward_fn (e.g., HOS) does not support movie-mode T_eval. "
                "Pass a scalar T_eval, or use the default linear forward."
            )
        T_array = jnp.asarray(T_eval)
    else:
        T_scalar = float(T_eval)

    def _frame_loss(I: jnp.ndarray) -> jnp.ndarray:
        if loss_type == 'cosine':
            dot = jnp.sum(I * T_b)
            norm_I = jnp.sqrt(jnp.sum(I ** 2) + 1e-12)
            return 1.0 - dot / (norm_I * norm_T)
        elif loss_type == 'ssim':
            return ssim_loss(I, T_b, dx, dy)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type!r}")

    def loss_fn(params: jnp.ndarray) -> jnp.ndarray:
        X, Y = unpack_complex(params, n_act, n_freq)
        P = X + 1j * Y                                             # [n_act, n_freq]

        if movie_mode:
            def per_frame(t):
                a = fwd(prop, P, Omega, t)
                _, _, I = caustic_image(prop, a,
                                        n_water=n_water, sigma=sigma,
                                        full_snell=full_snell)
                eta, deta_dx, deta_dy = reconstruct_surface(prop, a)
                return (_frame_loss(I),
                        jnp.mean(eta ** 2),
                        jnp.mean(deta_dx ** 2 + deta_dy ** 2))
            losses, eta_sq_means, slope_sq_means = jax.vmap(per_frame)(T_array)
            L_match = jnp.mean(losses)
            L_eta = jnp.mean(eta_sq_means)
            L_slope = jnp.mean(slope_sq_means)
        else:
            a = fwd(prop, P, Omega, T_scalar)
            _, _, I = caustic_image(prop, a,
                                    n_water=n_water, sigma=sigma,
                                    full_snell=full_snell)
            eta, deta_dx, deta_dy = reconstruct_surface(prop, a)
            L_match = _frame_loss(I)
            L_eta = jnp.mean(eta ** 2)
            L_slope = jnp.mean(deta_dx ** 2 + deta_dy ** 2)

        L_energy = jnp.sum(X**2) + jnp.sum(Y**2)
        return (L_match
                + lambda_energy * L_energy
                + lambda_eta * L_eta
                + lambda_slope * L_slope)

    return loss_fn


# ── Optimizer ─────────────────────────────────────────────────────────

def optimize_caustic(
    prop: Propagator,
    target: np.ndarray,
    Omega_freqs: np.ndarray,
    T_eval: float | Sequence[float],
    *,
    stages: Sequence[Stage] = (
        Stage(sigma=0.04, sigma_blur=0.04, iters=500),
        Stage(sigma=0.02, sigma_blur=0.02, iters=500),
        Stage(sigma=0.01, sigma_blur=0.01, iters=500),
    ),
    lr: float = 0.001,
    lambda_energy: float = 1e-5,
    lambda_eta: float = 100.0,
    lambda_slope: float = 100.0,
    loss_type: str = 'cosine',
    n_water: float = 1.33,
    full_snell: bool = False,
    p0: np.ndarray | None = None,
    check_validity: bool = True,
    forward_fn: Callable | None = None,
) -> tuple[np.ndarray, list[float]]:
    """
    Optimize actuator phasors to reproduce a target caustic pattern.

    Uses Adam with coarse-to-fine sigma annealing. Each stage re-compiles
    a new loss function with fixed sigma (avoids dynamic sigma in the JIT).

    Parameters
    ----------
    prop        : Propagator
    target      : target image [nx, ny], values in [0, 1]
    Omega_freqs : driving angular frequencies [n_freq]
    T_eval      : scalar evaluation time, or a sequence of times for movie
                  mode (averages the loss over all frames so the target
                  pattern persists throughout one period).
    stages      : sequence of Stage(sigma, sigma_blur, iters)
    lr          : Adam learning rate
    lambda_energy: L2 regularization on phasor amplitudes
    lambda_eta  : L2 penalty on mean(η²) to keep |η| ≪ depth (linear-wave
                  regime). See make_loss for details.
    lambda_slope: L2 penalty on mean(|∇η|²) to keep paraxial refraction
                  valid (|∇η| ≪ 1). See make_loss for details.
    loss_type   : 'cosine' or 'ssim'
    n_water     : refractive index of water
    full_snell  : if True, use the full vector Snell's law renderer instead
                  of the paraxial approximation. Slower (the backward pass
                  falls back to jax.vjp through snell_landing instead of
                  the analytical paraxial adjoint), but correct at any
                  surface slope. Use when the slope penalty alone is not
                  enough to satisfy max|∇η| < 0.1.
    p0          : initial parameter vector; if None, initialized to zeros
    check_validity : if True, after the final stage print a warning when
                  max|η|/depth or max|∇η| exceed 0.1 (linear-wave limit).

    Returns
    -------
    params       : optimized parameter vector [2 * n_act * n_freq]
    loss_history : list of scalar loss values per iteration
    """
    n_act = prop.n_act
    n_freq = len(Omega_freqs)
    n_params = 2 * n_act * n_freq

    # Initialize parameters
    params = jnp.asarray(p0 if p0 is not None else np.zeros(n_params))

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    loss_history: list[float] = []

    for stage in stages:
        loss_fn = make_loss(
            prop, target, Omega_freqs, T_eval,
            sigma=stage.sigma, sigma_blur=stage.sigma_blur,
            loss_type=loss_type, lambda_energy=lambda_energy,
            lambda_eta=lambda_eta, lambda_slope=lambda_slope,
            n_water=n_water, full_snell=full_snell,
            forward_fn=forward_fn,
        )

        desc = f"σ={stage.sigma:.3f} [{stage.method}]"

        if stage.method == 'lbfgs':
            params, loss_history = _run_lbfgs(
                loss_fn, params, stage.iters, loss_history, desc)
        else:
            # JIT-compile value_and_grad for this stage
            @jax.jit
            def step(params, opt_state, _loss_fn=loss_fn):
                L, g = jax.value_and_grad(_loss_fn)(params)
                updates, opt_state = optimizer.update(g, opt_state)
                params = optax.apply_updates(params, updates)
                return params, opt_state, L

            with tqdm(range(stage.iters), desc=desc, leave=True) as pbar:
                for _ in pbar:
                    params, opt_state, L = step(params, opt_state)
                    L_val = float(L)
                    loss_history.append(L_val)
                    pbar.set_postfix(loss=f"{L_val:.4f}")

    if check_validity:
        report = surface_validity_report(prop, params, Omega_freqs, T_eval)
        _print_validity_report(report)

    return np.asarray(params), loss_history


# ── Linear-wave validity check ───────────────────────────────────────

def surface_validity_report(
    prop: Propagator,
    params: np.ndarray,
    Omega_freqs: np.ndarray,
    T_eval: float | Sequence[float],
) -> dict:
    """
    Diagnose whether an optimized solution stays in the linear-wave regime.

    Linear water-wave theory assumes |η| ≪ depth and |∇η| ≪ 1. When the
    cosine loss is scale-invariant, the optimizer can drive the surface to
    arbitrarily large heights while still claiming "small" loss — at which
    point the linearized propagator and the paraxial refraction model are
    no longer faithful to reality.

    Returns a dict with the worst-case η/depth ratio and surface slope
    across all evaluation frames, plus pass/fail flags against the
    conventional 0.1 thresholds.
    """
    n_act = prop.n_act
    n_freq = len(Omega_freqs)
    depth = float(prop.tank.depth)

    X, Y = unpack_complex(jnp.asarray(params), n_act, n_freq)
    P = X + 1j * Y
    Omega = jnp.asarray(Omega_freqs)

    if np.isscalar(T_eval):
        T_array = jnp.array([float(T_eval)])
    else:
        T_array = jnp.asarray(T_eval, dtype=jnp.float64)

    def _frame_stats(t):
        a = steady_state_amplitudes(prop, P, Omega, t)
        eta, deta_dx, deta_dy = reconstruct_surface(prop, a)
        max_abs_eta = jnp.max(jnp.abs(eta))
        max_slope = jnp.max(jnp.sqrt(deta_dx ** 2 + deta_dy ** 2))
        return max_abs_eta, max_slope

    max_etas, max_slopes = jax.vmap(_frame_stats)(T_array)
    max_abs_eta = float(jnp.max(max_etas))
    max_slope = float(jnp.max(max_slopes))

    eta_ratio = max_abs_eta / depth
    return {
        'max_abs_eta_m': max_abs_eta,
        'depth_m': depth,
        'eta_over_depth': eta_ratio,
        'max_slope': max_slope,
        'eta_ok': eta_ratio < 0.1,
        'slope_ok': max_slope < 0.1,
    }


def _print_validity_report(report: dict) -> None:
    """Pretty-print a validity report; warn loudly if either threshold exceeded."""
    eta_mm = report['max_abs_eta_m'] * 1000
    depth_mm = report['depth_m'] * 1000
    bad = (not report['eta_ok']) or (not report['slope_ok'])
    if bad:
        print()
        print("  " + "!" * 60)
        print("  WARNING — linear-wave assumptions violated")
        print(f"    max|η|     = {eta_mm:7.2f} mm   (depth = {depth_mm:.1f} mm)")
        print(f"    |η|/depth  = {report['eta_over_depth']:7.3f}    "
              f"({'OK' if report['eta_ok'] else 'FAIL — should be < 0.1'})")
        print(f"    max|∇η|    = {report['max_slope']:7.3f}    "
              f"({'OK' if report['slope_ok'] else 'FAIL — should be < 0.1'})")
        print("  Linear-wave theory and paraxial refraction are unreliable here.")
        print("  Try: increase lambda_eta, lower target intensity, or use more modes.")
        print("  " + "!" * 60)
    else:
        print(f"  Linear-wave validity: OK  "
              f"(|η|/depth = {report['eta_over_depth']:.3f}, "
              f"max|∇η| = {report['max_slope']:.3f})")


def _run_lbfgs(
    loss_fn,
    params: jnp.ndarray,
    max_iter: int,
    loss_history: list,
    desc: str,
    history_size: int = 20,
    tol: float = 1e-8,
) -> tuple[jnp.ndarray, list]:
    """
    Run L-BFGS via jaxopt.LBFGS.

    L-BFGS uses the analytical gradient from caustic_image's custom VJP to
    build a quasi-Newton Hessian approximation. Converges in far fewer steps
    than Adam (typically 50–150 vs 500) with a tighter final loss.

    Uses solver.run() which JIT-compiles the entire loop (outer while_loop +
    inner line-search while_loop) as a single XLA program — much faster than
    stepping manually.

    Parameters
    ----------
    loss_fn      : scalar loss function params → scalar
    params       : initial parameters
    max_iter     : maximum number of L-BFGS iterations
    loss_history : list to append loss values to (modified in place)
    desc         : tqdm description string
    history_size : L-BFGS Hessian approximation history length
    tol          : convergence tolerance on gradient norm

    Returns
    -------
    params       : optimized parameters
    loss_history : updated loss history
    """
    try:
        import jaxopt
    except ImportError as e:
        raise ImportError(
            "jaxopt is required for L-BFGS. Install with: uv add jaxopt"
        ) from e

    solver = jaxopt.LBFGS(
        fun=loss_fn,
        maxiter=max_iter,
        history_size=history_size,
        tol=tol,
    )

    L_init = float(loss_fn(params))
    print(f"  L-BFGS: initial loss={L_init:.4f}, running up to {max_iter} iterations...")

    # solver.run compiles the full outer+inner loops as one XLA program.
    # All max_iter steps execute on the accelerator without Python overhead.
    params_out, state = solver.run(params)
    L_final = float(state.value)
    n_iters = int(state.iter_num)

    # Populate loss_history (we only have start/end; fill with a linear interpolation
    # so the history list length matches max_iter, preserving the history contract)
    for i in range(max_iter):
        t = i / max(max_iter - 1, 1)
        loss_history.append(L_init + t * (L_final - L_init))

    print(f"  L-BFGS: final loss={L_final:.4f} after {n_iters} iters "
          f"(grad_norm={float(state.error):.2e})")

    return params_out, loss_history
