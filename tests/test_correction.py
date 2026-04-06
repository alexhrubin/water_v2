"""Tests for wavetank.correction: U-Net forward, save/load, training loop."""

import os
import tempfile

import numpy as np
import jax
import jax.numpy as jnp

from wavetank import (
    CorrectionUNet, apply_correction, save_model, load_model,
    train_correction, generate_training_data,
    NonIdealHyperparams, sample_nonideal_config,
)


def test_unet_forward_shape():
    """U-Net produces an output the same shape as the input."""
    model = CorrectionUNet(ch=8, key=jax.random.PRNGKey(0))
    x = jnp.zeros((1, 32, 32))
    y = model(x)
    assert y.shape == (1, 32, 32)


def test_apply_correction_returns_2d():
    """apply_correction strips the channel dim and adds residual."""
    model = CorrectionUNet(ch=8, key=jax.random.PRNGKey(1))
    I = jax.random.normal(jax.random.PRNGKey(2), shape=(32, 32))
    out = apply_correction(model, I)
    assert out.shape == I.shape
    # The residual should be small at init: model output ≈ random small values
    assert bool(jnp.all(jnp.isfinite(out)))


def test_save_load_roundtrip():
    """Saved → loaded model produces identical output."""
    model = CorrectionUNet(ch=8, key=jax.random.PRNGKey(3))
    x = jax.random.normal(jax.random.PRNGKey(4), shape=(1, 16, 16))
    y_before = model(x)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.eqx")
        save_model(path, model)
        model_loaded = load_model(path, ch=8)

    y_after = model_loaded(x)
    np.testing.assert_array_equal(np.asarray(y_before), np.asarray(y_after))


def test_generate_training_data_vmap(prop, Omega):
    """Vectorized generator returns the right shapes and finite values."""
    config = sample_nonideal_config(prop, NonIdealHyperparams(),
                                     jax.random.PRNGKey(0))
    I, delta = generate_training_data(
        prop, config, Omega, T=1.0, n_samples=4,
        key=jax.random.PRNGKey(1),
        phasor_scale=0.3, sigma=0.03,
    )
    assert I.shape == (4, prop.nx, prop.ny)
    assert delta.shape == (4, prop.nx, prop.ny)
    assert bool(jnp.all(jnp.isfinite(I)))
    assert bool(jnp.all(jnp.isfinite(delta)))
    # Distinct random keys → distinct samples
    assert not np.allclose(np.asarray(I[0]), np.asarray(I[1]))


def test_train_correction_reduces_loss():
    """A few epochs on a tiny synthetic dataset should drive loss down."""
    model = CorrectionUNet(ch=8, key=jax.random.PRNGKey(5))

    # Small synthetic dataset: target residual = 0.3 * I (linear, easy to learn)
    key = jax.random.PRNGKey(6)
    I = jax.random.normal(key, shape=(8, 16, 16))
    delta = 0.3 * I

    trained, history = train_correction(
        model, I, delta,
        lr=5e-3, n_epochs=20, batch_size=4,
        key=jax.random.PRNGKey(7),
        verbose=False,
    )

    assert len(history) == 20
    assert history[-1] < history[0], \
        f"loss did not decrease: {history[0]:.4f} → {history[-1]:.4f}"
