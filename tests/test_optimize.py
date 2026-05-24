"""Tests for wavetank.optimize: make_loss in single-frame and movie mode."""

import numpy as np
import jax
import jax.numpy as jnp

from wavetank import make_loss, surface_validity_report
from wavetank.physics import steady_state_amplitudes, unpack_complex
from wavetank.render import reconstruct_surface
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


def test_lambda_eta_adds_expected_penalty(prop, Omega):
    """The η penalty should equal lambda_eta * mean(η²) on top of the no-penalty loss."""
    target = _gaussian_target(prop.nx, prop.ny)
    params = sample_random_phasors(jax.random.PRNGKey(7), prop.n_act, len(Omega), scale=0.5)

    # Same loss config, only lambda_eta differs.
    common = dict(sigma=0.05, sigma_blur=0.05, lambda_energy=0.0)
    loss_no_eta = make_loss(prop, target, np.asarray(Omega), T_eval=1.0,
                             lambda_eta=0.0, **common)
    loss_with_eta = make_loss(prop, target, np.asarray(Omega), T_eval=1.0,
                               lambda_eta=42.0, **common)

    L0 = float(loss_no_eta(params))
    L1 = float(loss_with_eta(params))

    # Compute the expected penalty independently.
    X, Y = unpack_complex(params, prop.n_act, len(Omega))
    P = X + 1j * Y
    a = steady_state_amplitudes(prop, P, Omega, 1.0)
    eta, _, _ = reconstruct_surface(prop, a)
    expected_penalty = 42.0 * float(jnp.mean(eta ** 2))

    np.testing.assert_allclose(L1 - L0, expected_penalty, rtol=1e-10, atol=1e-12)


def test_lambda_eta_gradient_flows(prop, Omega):
    """The η-penalty term must contribute a finite gradient w.r.t. params."""
    target = _gaussian_target(prop.nx, prop.ny)
    loss = make_loss(prop, target, np.asarray(Omega), T_eval=1.0,
                     sigma=0.05, sigma_blur=0.05,
                     lambda_energy=0.0, lambda_eta=1000.0)
    params = sample_random_phasors(jax.random.PRNGKey(11), prop.n_act, len(Omega), scale=0.3)
    g = jax.grad(loss)(params)
    assert g.shape == params.shape
    assert bool(jnp.all(jnp.isfinite(g)))
    # With a non-trivial penalty and non-zero phasors, the gradient must be non-zero.
    assert float(jnp.linalg.norm(g)) > 0.0


def test_lambda_slope_adds_expected_penalty(prop, Omega):
    """The slope penalty should equal lambda_slope * mean(|∇η|²) on top of the no-penalty loss."""
    target = _gaussian_target(prop.nx, prop.ny)
    params = sample_random_phasors(jax.random.PRNGKey(7), prop.n_act, len(Omega), scale=0.5)

    common = dict(sigma=0.05, sigma_blur=0.05, lambda_energy=0.0, lambda_eta=0.0)
    loss_no_slope = make_loss(prop, target, np.asarray(Omega), T_eval=1.0,
                               lambda_slope=0.0, **common)
    loss_with_slope = make_loss(prop, target, np.asarray(Omega), T_eval=1.0,
                                 lambda_slope=37.0, **common)

    L0 = float(loss_no_slope(params))
    L1 = float(loss_with_slope(params))

    # Compute the expected penalty independently.
    X, Y = unpack_complex(params, prop.n_act, len(Omega))
    P = X + 1j * Y
    a = steady_state_amplitudes(prop, P, Omega, 1.0)
    _, deta_dx, deta_dy = reconstruct_surface(prop, a)
    expected_penalty = 37.0 * float(jnp.mean(deta_dx ** 2 + deta_dy ** 2))

    np.testing.assert_allclose(L1 - L0, expected_penalty, rtol=1e-10, atol=1e-12)


def test_full_snell_loss_finite_and_differentiable(prop, Omega):
    """make_loss(full_snell=True) → finite scalar with finite gradient."""
    target = _gaussian_target(prop.nx, prop.ny)
    loss = make_loss(prop, target, np.asarray(Omega), T_eval=1.0,
                     sigma=0.05, sigma_blur=0.05,
                     lambda_energy=0.0, lambda_eta=0.0, lambda_slope=0.0,
                     full_snell=True)
    params = sample_random_phasors(jax.random.PRNGKey(13), prop.n_act, len(Omega), scale=0.3)

    L = float(loss(params))
    assert np.isfinite(L)

    g = jax.grad(loss)(params)
    assert g.shape == params.shape
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.linalg.norm(g)) > 0.0


def test_surface_validity_report_flags_extreme_height(prop, Omega):
    """A grossly oversized phasor solution should fail the linear-wave check."""
    # Tiny phasors → safe
    small = sample_random_phasors(jax.random.PRNGKey(0), prop.n_act, len(Omega), scale=1e-4)
    rep_small = surface_validity_report(prop, small, np.asarray(Omega), T_eval=1.0)
    assert rep_small['eta_ok']
    assert rep_small['slope_ok']

    # Massively scaled phasors → should violate the |η|/depth threshold.
    huge = small * 1e6
    rep_big = surface_validity_report(prop, huge, np.asarray(Omega), T_eval=1.0)
    assert not rep_big['eta_ok']
    assert rep_big['max_abs_eta_m'] > rep_big['depth_m'] * 0.1


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
