"""Shared fixtures for the wavetank test suite."""

import jax

# Use float64 throughout — gradient checks need it.
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from wavetank import Tank, Actuator, build_propagator


@pytest.fixture(scope="session")
def tank():
    """Small canonical tank used by every test."""
    return Tank(Lx=1.0, Ly=1.0, depth=0.12, damping=0.02)


@pytest.fixture(scope="session")
def actuators(tank):
    """Eight actuators on the tank perimeter (2 per side)."""
    acts = []
    for i in range(2):
        t = (i + 1) / 3
        acts += [
            Actuator(x=0.0,    y=t * tank.Ly),
            Actuator(x=tank.Lx, y=t * tank.Ly),
            Actuator(x=t * tank.Lx, y=0.0),
            Actuator(x=t * tank.Lx, y=tank.Ly),
        ]
    return acts


@pytest.fixture(scope="session")
def prop(tank, actuators):
    """Small propagator: 6 modes per side, 24x24 grid — fast enough for autodiff."""
    return build_propagator(tank, actuators, n_modes=6, nx=24, ny=24)


@pytest.fixture(scope="session")
def Omega():
    """Three driving frequencies in rad/s."""
    import jax.numpy as jnp
    return jnp.array([6.28, 9.42, 12.57])


@pytest.fixture(scope="session")
def n_freq(Omega):
    return len(Omega)
