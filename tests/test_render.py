"""Tests for wavetank.render: surface reconstruction, caustic image, custom VJP."""

import numpy as np
import jax
import jax.numpy as jnp

from wavetank import (
    reconstruct_surface, caustic_image,
    steady_state_amplitudes, unpack_complex,
)
from wavetank.nonideal import sample_random_phasors


def test_reconstruct_surface_zero_amplitude(prop):
    """Zero amplitudes → flat surface."""
    a = jnp.zeros(prop.omega.shape[0])
    eta, dx, dy = reconstruct_surface(prop, a)
    np.testing.assert_allclose(np.asarray(eta), 0.0)
    np.testing.assert_allclose(np.asarray(dx), 0.0)
    np.testing.assert_allclose(np.asarray(dy), 0.0)


def test_reconstruct_surface_single_mode(prop):
    """Activating one mode reproduces that cosine basis function."""
    j = 5  # arbitrary mode
    m, n = int(prop.mode_m[j]), int(prop.mode_n[j])
    a = jnp.zeros(prop.omega.shape[0]).at[j].set(2.0)
    eta, _, _ = reconstruct_surface(prop, a)

    Lx, Ly = prop.tank.Lx, prop.tank.Ly
    expected = 2.0 * np.outer(
        np.cos(m * np.pi * prop.xs / Lx),
        np.cos(n * np.pi * prop.ys / Ly),
    )
    np.testing.assert_allclose(np.asarray(eta), expected, atol=1e-12)


def test_caustic_image_shapes(prop):
    """caustic_image returns the expected (xs, ys, I) shapes."""
    a = jnp.zeros(prop.omega.shape[0])
    xs, ys, I = caustic_image(prop, a)
    assert xs.shape == (prop.nx,)
    assert ys.shape == (prop.ny,)
    assert I.shape == (prop.nx, prop.ny)


def test_caustic_image_flat_surface_uniform(prop):
    """A flat water surface should produce a (nearly) uniform caustic image
    away from the edges (boundary blur loses some mass)."""
    a = jnp.zeros(prop.omega.shape[0])
    _, _, I = caustic_image(prop, a, sigma=0.05)
    interior = np.asarray(I)[6:-6, 6:-6]
    rel_std = float(np.std(interior)) / max(float(np.mean(interior)), 1e-9)
    assert rel_std < 1e-3


def _grad_check_through_params(prop, Omega, full_snell, sigma):
    """Central-difference check on the parameter-space loss.

    Goes through steady_state → caustic_image so it exercises the custom VJP
    composed with the rest of the autodiff graph. A large sigma is used so the
    loss landscape is smooth enough that FD doesn't trip on splat discontinuities.
    """
    n_freq = len(Omega)
    params = sample_random_phasors(jax.random.PRNGKey(7), prop.n_act, n_freq, scale=0.3)
    target = jax.random.normal(jax.random.PRNGKey(8), shape=(prop.nx, prop.ny))

    def loss_fn(p):
        X, Y = unpack_complex(p, prop.n_act, n_freq)
        P = X + 1j * Y
        a = steady_state_amplitudes(prop, P, Omega, T=1.0)
        _, _, I = caustic_image(prop, a, sigma=sigma, full_snell=full_snell)
        return jnp.sum(I * target)

    g_analytic = jax.grad(loss_fn)(params)
    v = jax.random.normal(jax.random.PRNGKey(9), shape=params.shape)
    # eps=1e-5 needed because the paraxial gradient is small (the deflection
    # coefficient is (1-1/n)≈0.25), so larger eps gets swamped by bilinear-splat
    # discretization noise. Below 1e-5 FD matches analytic to ~1e-8.
    eps = 1e-5
    fd = (float(loss_fn(params + eps * v)) - float(loss_fn(params - eps * v))) / (2 * eps)
    analytic = float(jnp.sum(g_analytic * v))
    rel = abs(analytic - fd) / max(abs(fd), 1e-8)
    assert rel < 5e-3, f"rel error {rel:.2e} (analytic={analytic}, fd={fd})"


def test_caustic_vjp_paraxial(prop, Omega):
    """Custom VJP gradient matches FD for paraxial refraction (through params)."""
    _grad_check_through_params(prop, Omega, full_snell=False, sigma=0.08)


def test_caustic_vjp_full_snell(prop, Omega):
    """Custom VJP gradient matches FD for full vector Snell's law (through params)."""
    _grad_check_through_params(prop, Omega, full_snell=True, sigma=0.08)


def test_caustic_jit_compiles(prop):
    """caustic_image survives jax.jit (cosmetic check that custom_vjp signatures hold)."""
    @jax.jit
    def render(a):
        _, _, I = caustic_image(prop, a, sigma=0.03)
        return jnp.sum(I)
    a = jnp.zeros(prop.omega.shape[0])
    val = float(render(a))
    assert np.isfinite(val)
