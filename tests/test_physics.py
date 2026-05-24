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


# ── Tank.throw decoupling ─────────────────────────────────────────────

def test_tank_throw_defaults_to_depth():
    """Tank.throw == Tank.depth when projection_distance is left as None."""
    tank = Tank(Lx=1.0, Ly=1.0, depth=0.12, damping=0.02)
    assert tank.throw == tank.depth
    assert tank.projection_distance is None


def test_tank_throw_independent_of_depth():
    """Setting projection_distance overrides throw without touching depth."""
    tank = Tank(Lx=1.0, Ly=1.0, depth=0.12, damping=0.02,
                projection_distance=0.5)
    assert tank.depth == 0.12
    assert tank.throw == 0.5


def test_throw_does_not_affect_omega_or_C(actuators):
    """
    The hydrodynamic state (omega, C, cos_x, cos_y) must be invariant
    under changes to projection_distance — only optical stages are
    allowed to depend on throw.
    """
    tank_a = Tank(Lx=1.0, Ly=1.0, depth=0.12, damping=0.02)
    tank_b = Tank(Lx=1.0, Ly=1.0, depth=0.12, damping=0.02,
                  projection_distance=0.5)
    pa = build_propagator(tank_a, actuators, n_modes=6, nx=24, ny=24)
    pb = build_propagator(tank_b, actuators, n_modes=6, nx=24, ny=24)

    np.testing.assert_array_equal(pa.omega, pb.omega)
    np.testing.assert_array_equal(pa.C,     pb.C)
    np.testing.assert_array_equal(pa.cos_x, pb.cos_x)
    np.testing.assert_array_equal(pa.cos_y, pb.cos_y)


def test_throw_changes_paraxial_landing_linearly(actuators):
    """
    The paraxial landing displacement (x_land - x) is linear in (throw - η),
    so doubling the throw (with η small) should very nearly double the
    displacement. Verifies that throw is plumbed into the optical pipeline.
    """
    from wavetank.render import reconstruct_surface, _paraxial_landing

    tank_a = Tank(Lx=1.0, Ly=1.0, depth=0.05, damping=0.02,
                  projection_distance=0.10)
    tank_b = Tank(Lx=1.0, Ly=1.0, depth=0.05, damping=0.02,
                  projection_distance=0.20)
    pa = build_propagator(tank_a, actuators, n_modes=6, nx=24, ny=24)
    pb = build_propagator(tank_b, actuators, n_modes=6, nx=24, ny=24)

    # Same modal field for both tanks (just to drive a non-trivial slope)
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal(pa.omega.shape) * 1e-4)

    eta_a, dxa, dya = reconstruct_surface(pa, a)
    eta_b, dxb, dyb = reconstruct_surface(pb, a)
    np.testing.assert_allclose(np.asarray(eta_a), np.asarray(eta_b))

    Xs = jnp.asarray(pa.X_src)
    Ys = jnp.asarray(pa.Y_src)
    xa, ya = _paraxial_landing(Xs, Ys, eta_a, dxa, dya, pa.tank.throw, 1.33)
    xb, yb = _paraxial_landing(Xs, Ys, eta_b, dxb, dyb, pb.tank.throw, 1.33)

    disp_a = np.asarray(xa - Xs)
    disp_b = np.asarray(xb - Xs)
    # disp = (throw - η) · ∂η/∂x / n_water, so the per-pixel ratio is
    # exactly (throw_b - η)/(throw_a - η).
    eta_np = np.asarray(eta_a)
    expected = (pb.tank.throw - eta_np) / (pa.tank.throw - eta_np)
    nz = np.abs(disp_a) > 1e-12
    np.testing.assert_allclose(disp_b[nz] / disp_a[nz], expected[nz], rtol=1e-10)
