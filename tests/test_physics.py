"""Tests for wavetank.physics: tank, propagator, transfer matrix, steady state."""

import numpy as np
import jax
import jax.numpy as jnp

from wavetank import (
    Tank, Actuator, build_propagator,
    transfer_matrix, steady_state_amplitudes,
    pack_complex, unpack_complex,
)


def test_propagator_shapes(prop, actuators):
    """Propagator arrays have the expected shapes."""
    n_modes = prop.n_modes
    n_total = n_modes * n_modes - 1   # we skip (0, 0)
    n_act = len(actuators)

    assert prop.n_act == n_act
    assert prop.omega.shape == (n_total,)
    assert prop.C.shape == (n_total, n_act)
    assert prop.cos_x.shape == (prop.nx, n_modes)
    assert prop.cos_y.shape == (prop.ny, n_modes)
    assert prop.X_src.shape == (prop.nx, prop.ny)
    assert prop.lin_2d.shape == (n_total,)


def test_eigenfrequencies_increasing_with_k(prop):
    """ω(k) should be monotonic in |k|: higher mode index → higher frequency."""
    k_sq = (prop.mode_m * np.pi / prop.tank.Lx) ** 2 + \
           (prop.mode_n * np.pi / prop.tank.Ly) ** 2
    order = np.argsort(k_sq)
    sorted_omega = prop.omega[order]
    assert np.all(np.diff(sorted_omega) >= -1e-12)


def test_transfer_matrix_matches_formula(prop):
    """H[j,k] = 1 / (ω_j² - Ω_k² + 2iγω_jΩ_k)."""
    Omega = jnp.array([0.5, 5.0])
    gamma = prop.tank.damping
    H = np.asarray(transfer_matrix(prop.omega, Omega, gamma))
    omega = np.asarray(prop.omega)[:, None]
    Om = np.asarray(Omega)[None, :]
    expected = 1.0 / (omega ** 2 - Om ** 2 + 2j * gamma * omega * Om)
    np.testing.assert_allclose(H, expected, rtol=1e-12)


def test_steady_state_amplitudes_zero_phasor(prop, Omega):
    """Zero forcing → zero amplitudes."""
    P = jnp.zeros((prop.n_act, len(Omega)), dtype=complex)
    a = steady_state_amplitudes(prop, P, Omega, T=0.0)
    np.testing.assert_allclose(np.asarray(a), 0.0)


def test_steady_state_linearity(prop, Omega):
    """a is linear in P: a(αP) = α a(P)."""
    key = jax.random.PRNGKey(0)
    Pr = jax.random.normal(key, shape=(prop.n_act, len(Omega)))
    Pi = jax.random.normal(key, shape=(prop.n_act, len(Omega)))
    P = Pr + 1j * Pi
    a1 = steady_state_amplitudes(prop, P, Omega, T=0.5)
    a2 = steady_state_amplitudes(prop, 3.0 * P, Omega, T=0.5)
    np.testing.assert_allclose(np.asarray(a2), 3.0 * np.asarray(a1), rtol=1e-12)


def test_steady_state_periodicity(prop, Omega):
    """a should oscillate with the driving period for a single-frequency input."""
    Omega_single = jnp.array([Omega[0]])
    P = jnp.ones((prop.n_act, 1), dtype=complex)
    period = 2 * np.pi / float(Omega_single[0])
    a0 = steady_state_amplitudes(prop, P, Omega_single, T=0.123)
    a1 = steady_state_amplitudes(prop, P, Omega_single, T=0.123 + period)
    np.testing.assert_allclose(np.asarray(a0), np.asarray(a1), atol=1e-10)


def test_pack_unpack_roundtrip(prop, n_freq):
    """pack_complex and unpack_complex are inverses."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((prop.n_act, n_freq))
    Y = rng.standard_normal((prop.n_act, n_freq))
    p = pack_complex(X, Y)
    X2, Y2 = unpack_complex(jnp.asarray(p), prop.n_act, n_freq)
    np.testing.assert_allclose(np.asarray(X2), X)
    np.testing.assert_allclose(np.asarray(Y2), Y)
