"""Tests for wavetank.optimize: make_loss in single-frame and movie mode."""

import numpy as np
import jax
import jax.numpy as jnp

from wavetank import make_loss
from wavetank.nonideal import sample_random_phasors


def _gaussian_target(nx, ny):
    xx, yy = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing='ij')
    t = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.05).astype(np.float32)
    return t / t.max()


def test_make_loss_single_frame_returns_scalar(prop, Omega):
    """Single-frame loss returns a finite scalar with valid gradient."""
    target = _gaussian_target(prop.nx, prop.ny)
    loss = make_loss(prop, target, np.asarray(Omega), T_eval=1.0,
                     sigma=0.05, sigma_blur=0.05)

    params = sample_random_phasors(jax.random.PRNGKey(0), prop.n_act, len(Omega), scale=0.3)
    L = float(loss(params))
    assert np.isfinite(L)

    g = jax.grad(loss)(params)
    assert g.shape == params.shape
    assert bool(jnp.all(jnp.isfinite(g)))


def test_make_loss_movie_mode(prop, Omega):
    """Movie mode averages loss over multiple T values; gradient still flows."""
    target = _gaussian_target(prop.nx, prop.ny)
    period = 2 * np.pi / float(Omega[0])
    T_frames = np.linspace(0.0, period, 6, endpoint=False)

    movie_loss = make_loss(prop, target, np.asarray(Omega), T_eval=T_frames,
                           sigma=0.05, sigma_blur=0.05)
    params = sample_random_phasors(jax.random.PRNGKey(0), prop.n_act, len(Omega), scale=0.3)

    L = float(movie_loss(params))
    assert np.isfinite(L)

    g = jax.grad(movie_loss)(params)
    assert g.shape == params.shape
    assert bool(jnp.all(jnp.isfinite(g)))


def test_movie_loss_equals_mean_of_single_frame_losses(prop, Omega):
    """Movie loss should equal the mean of single-frame losses (modulo regulariser)."""
    target = _gaussian_target(prop.nx, prop.ny)
    period = 2 * np.pi / float(Omega[0])
    T_frames = np.linspace(0.0, period, 4, endpoint=False)
    params = sample_random_phasors(jax.random.PRNGKey(2), prop.n_act, len(Omega), scale=0.3)

    movie_loss = make_loss(prop, target, np.asarray(Omega), T_eval=T_frames,
                           sigma=0.05, sigma_blur=0.05, lambda_energy=0.0)
    L_movie = float(movie_loss(params))

    L_singles = []
    for t in T_frames:
        single = make_loss(prop, target, np.asarray(Omega), T_eval=float(t),
                           sigma=0.05, sigma_blur=0.05, lambda_energy=0.0)
        L_singles.append(float(single(params)))
    L_mean = sum(L_singles) / len(L_singles)

    np.testing.assert_allclose(L_movie, L_mean, rtol=1e-10)
