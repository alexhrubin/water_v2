"""
Caustic optimization via Adam with coarse-to-fine sigma annealing.

Each optimization stage blurs both the rendered caustic and the target
with Gaussian sigma, then progressively sharpens. Starting coarse avoids
local minima from the sparse ray-splatting landscape; finishing fine
recovers spatial detail.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from .physics import Propagator, steady_state_amplitudes, unpack_complex
from .render import caustic_image, _gaussian_blur_separable
from .loss import cosine_loss, ssim_loss


# ── Stage specification ────────────────────────────────────────────────

@dataclass(frozen=True)
class Stage:
    """One phase of coarse-to-fine optimization."""
    sigma: float         # caustic rendering blur (m)
    sigma_blur: float    # target pre-blur (m); usually equals sigma
    iters: int           # number of Adam steps


# ── Make loss function ─────────────────────────────────────────────────

def make_loss(
    prop: Propagator,
    target: np.ndarray,
    Omega_freqs: np.ndarray,
    T_eval: float,
    *,
    sigma: float = 0.02,
    sigma_blur: float = 0.02,
    loss_type: str = 'cosine',
    lambda_energy: float = 1e-5,
    n_water: float = 1.33,
    full_snell: bool = False,
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
    T_eval        : evaluation time (s)
    sigma         : rendering blur
    sigma_blur    : target pre-blur (0 = no blur)
    loss_type     : 'cosine' or 'ssim'
    lambda_energy : L2 regularization weight on phasor amplitudes
    n_water       : refractive index
    full_snell    : use full Snell's law refraction
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

    def loss_fn(params: jnp.ndarray) -> jnp.ndarray:
        X, Y = unpack_complex(params, n_act, n_freq)
        P = X + 1j * Y                                             # [n_act, n_freq]
        a = steady_state_amplitudes(prop, P, Omega, T_eval)        # [n_total]
        _, _, I = caustic_image(prop, a,
                                n_water=n_water, sigma=sigma,
                                full_snell=full_snell)             # [nx, ny]

        if loss_type == 'cosine':
            dot = jnp.sum(I * T_b)
            norm_I = jnp.sqrt(jnp.sum(I**2) + 1e-12)
            L_match = 1.0 - dot / (norm_I * norm_T)
        elif loss_type == 'ssim':
            L_match = ssim_loss(I, T_b, dx, dy)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type!r}")

        L_energy = jnp.sum(X**2) + jnp.sum(Y**2)
        return L_match + lambda_energy * L_energy

    return loss_fn


# ── Optimizer ─────────────────────────────────────────────────────────

def optimize_caustic(
    prop: Propagator,
    target: np.ndarray,
    Omega_freqs: np.ndarray,
    T_eval: float,
    *,
    stages: Sequence[Stage] = (
        Stage(sigma=0.04, sigma_blur=0.04, iters=500),
        Stage(sigma=0.02, sigma_blur=0.02, iters=500),
        Stage(sigma=0.01, sigma_blur=0.01, iters=500),
    ),
    lr: float = 0.001,
    lambda_energy: float = 1e-5,
    loss_type: str = 'cosine',
    n_water: float = 1.33,
    p0: np.ndarray | None = None,
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
    T_eval      : evaluation time (s)
    stages      : sequence of Stage(sigma, sigma_blur, iters)
    lr          : Adam learning rate
    lambda_energy: L2 regularization on phasor amplitudes
    loss_type   : 'cosine' or 'ssim'
    n_water     : refractive index of water
    p0          : initial parameter vector; if None, initialized to zeros

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
            n_water=n_water,
        )

        # JIT-compile value_and_grad for this stage
        @jax.jit
        def step(params, opt_state):
            L, g = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(g, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, L

        desc = f"σ={stage.sigma:.3f}"
        with tqdm(range(stage.iters), desc=desc, leave=True) as pbar:
            for _ in pbar:
                params, opt_state, L = step(params, opt_state)
                L_val = float(L)
                loss_history.append(L_val)
                pbar.set_postfix(loss=f"{L_val:.4f}")

    return np.asarray(params), loss_history
