"""Tests for wavetank.nonideal: zero-perturbation identity, gradient flow."""

import numpy as np
import jax
import jax.numpy as jnp

from wavetank import (
    NonIdealHyperparams, sample_nonideal_config,
    caustic_image_nonideal, sample_random_phasors,
    caustic_image, steady_state_amplitudes, unpack_complex,
)
from wavetank.nonideal import (
    make_nonideal_propagator,
    nonideal_steady_state_amplitudes,
    radial_distort,
)


def _zero_hyper():
    return NonIdealHyperparams(
        damping_alpha=0.0,
        omega_sigma=0.0,
        coupling_sigma=0.0,
        distortion_k1_range=0.0,
        distortion_k2_range=0.0,
        distortion_center_sigma=0.0,
    )


def test_zero_perturbation_identity(prop, Omega):
    """All-zero hyperparams: non-ideal output matches ideal exactly."""
    config = sample_nonideal_config(prop, _zero_hyper(), jax.random.PRNGKey(0))

    n_freq = len(Omega)
    params = sample_random_phasors(jax.random.PRNGKey(1), prop.n_act, n_freq, scale=0.3)
    X, Y = unpack_complex(params, prop.n_act, n_freq)
    P = X + 1j * Y

    # Ideal path
    a_id = steady_state_amplitudes(prop, P, Omega, T=0.5)
    _, _, I_id = caustic_image(prop, a_id, sigma=0.03)

    # Non-ideal path with zero perturbations
    _, _, I_ni = caustic_image_nonideal(prop, config, P, Omega, T=0.5, sigma=0.03)

    np.testing.assert_allclose(np.asarray(I_ni), np.asarray(I_id), rtol=1e-12, atol=1e-12)


def test_nonideal_propagator_swaps_arrays(prop):
    """make_nonideal_propagator replaces omega/C while preserving structure."""
    hyper = NonIdealHyperparams(omega_sigma=0.05, coupling_sigma=0.05)
    config = sample_nonideal_config(prop, hyper, jax.random.PRNGKey(2))
    ni_prop = make_nonideal_propagator(prop, config)

    # omega/C are perturbed
    assert not np.allclose(ni_prop.omega, prop.omega)
    assert not np.allclose(ni_prop.C, prop.C)
    # Structural fields remain identical
    assert ni_prop.tank is prop.tank
    assert ni_prop.nx == prop.nx
    np.testing.assert_array_equal(ni_prop.cos_x, prop.cos_x)


def test_nonideal_perturbs_output(prop, Omega):
    """A non-trivial config produces a visibly different caustic from the ideal."""
    hyper = NonIdealHyperparams(
        damping_alpha=0.05,
        omega_sigma=0.02,
        coupling_sigma=0.05,
        distortion_k1_range=0.1,
        distortion_k2_range=0.0,
        distortion_center_sigma=0.0,
    )
    config = sample_nonideal_config(prop, hyper, jax.random.PRNGKey(11))

    n_freq = len(Omega)
    params = sample_random_phasors(jax.random.PRNGKey(12), prop.n_act, n_freq, scale=0.3)
    X, Y = unpack_complex(params, prop.n_act, n_freq)
    P = X + 1j * Y

    a_id = steady_state_amplitudes(prop, P, Omega, T=0.5)
    _, _, I_id = caustic_image(prop, a_id, sigma=0.03)
    _, _, I_ni = caustic_image_nonideal(prop, config, P, Omega, T=0.5, sigma=0.03)

    diff = float(jnp.mean(jnp.abs(I_ni - I_id)))
    base = float(jnp.mean(jnp.abs(I_id)))
    # Visibly different from the ideal (but still finite/sane).
    assert diff / base > 5e-3
    assert np.all(np.isfinite(np.asarray(I_ni)))


def test_radial_distort_zero_is_identity(prop):
    """k1 = k2 = 0 → radial_distort returns the input unchanged."""
    hyper = _zero_hyper()
    config = sample_nonideal_config(prop, hyper, jax.random.PRNGKey(3))
    I = jnp.ones((prop.nx, prop.ny)) * 0.7
    out = radial_distort(I, prop.xs, prop.ys, config)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(I))


def test_caustic_image_nonideal_differentiable(prop, Omega):
    """Gradient flows through the full non-ideal pipeline."""
    hyper = NonIdealHyperparams(
        damping_alpha=0.02, omega_sigma=0.01, coupling_sigma=0.03,
        distortion_k1_range=0.05, distortion_k2_range=0.0,
        distortion_center_sigma=0.0,
    )
    config = sample_nonideal_config(prop, hyper, jax.random.PRNGKey(20))

    n_freq = len(Omega)
    params = sample_random_phasors(jax.random.PRNGKey(21), prop.n_act, n_freq, scale=0.3)

    def loss_fn(p):
        X, Y = unpack_complex(p, prop.n_act, n_freq)
        P = X + 1j * Y
        _, _, I = caustic_image_nonideal(prop, config, P, Omega, T=0.5, sigma=0.03)
        return jnp.sum(I ** 2)

    g = jax.grad(loss_fn)(params)
    assert g.shape == params.shape
    assert float(jnp.sum(jnp.abs(g))) > 0.0
    assert bool(jnp.all(jnp.isfinite(g)))


def test_nonideal_steady_state_uses_perturbed_config(prop, Omega):
    """nonideal_steady_state_amplitudes really reads omega/gamma/C from config."""
    hyper = NonIdealHyperparams(
        damping_alpha=0.0, omega_sigma=0.1, coupling_sigma=0.1,
        distortion_k1_range=0.0, distortion_k2_range=0.0,
        distortion_center_sigma=0.0,
    )
    config = sample_nonideal_config(prop, hyper, jax.random.PRNGKey(33))

    n_freq = len(Omega)
    P = jax.random.normal(jax.random.PRNGKey(34), shape=(prop.n_act, n_freq)) + 1j * \
        jax.random.normal(jax.random.PRNGKey(35), shape=(prop.n_act, n_freq))

    a_ideal    = steady_state_amplitudes(prop, P, Omega, T=0.5)
    a_nonideal = nonideal_steady_state_amplitudes(prop, config, P, Omega, T=0.5)
    assert not np.allclose(np.asarray(a_ideal), np.asarray(a_nonideal))
