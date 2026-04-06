"""
Learned image-space correction: small U-Net predicts the residual between
ideal and non-ideal caustic images.

Architecture
------------
    I_ideal → UNet → ΔI
    I_corrected = I_ideal + ΔI  ≈  I_nonideal

The U-Net is a standard encoder-decoder with skip connections.  Downsampling
compresses the image through two bottleneck layers; upsampling restores the
resolution.  Skip connections carry high-frequency detail across.  The final
layer predicts a residual (ΔI), so the output is added to the input.

Training
--------
Generate (I_ideal, I_nonideal) pairs from random phasors using the non-ideal
simulator.  Train the network to minimise MSE(UNet(I_ideal), I_nonideal - I_ideal).

Usage in optimization (Step 3)
------------------------------
    def corrected_forward(params):
        a = steady_state_amplitudes(prop, P, Omega, T)
        _, _, I = caustic_image(prop, a, sigma=sigma)
        return I + model(I)

    loss = cosine_loss(corrected_forward(params), target)

Gradients flow through both the analytical VJP (ideal caustic) and the NN
(standard JAX autodiff).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from .physics import Propagator, steady_state_amplitudes, unpack_complex
from .render import caustic_image
from .nonideal import (
    NonIdealConfig,
    caustic_image_nonideal,
    sample_random_phasors,
)


# ── U-Net ─────────────────────────────────────────────────────────────

class ConvBlock(eqx.Module):
    """Two 3x3 convolutions with ReLU activation."""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d

    def __init__(self, in_ch: int, out_ch: int, *, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, key=k1)
        self.conv2 = eqx.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, key=k2)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        return x


class DownBlock(eqx.Module):
    """Downsample by 2x (average pool) then apply ConvBlock."""
    conv_block: ConvBlock

    def __init__(self, in_ch: int, out_ch: int, *, key: jax.Array):
        self.conv_block = ConvBlock(in_ch, out_ch, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Average pool 2x2 — works on (C, H, W)
        x = eqx.nn.AvgPool2d(kernel_size=2, stride=2)(x)
        return self.conv_block(x)


class UpBlock(eqx.Module):
    """Upsample by 2x (nearest neighbor), concatenate skip, then ConvBlock."""
    conv_block: ConvBlock

    def __init__(self, in_ch: int, out_ch: int, *, key: jax.Array):
        # in_ch = skip_ch + upsampled_ch (after concatenation)
        self.conv_block = ConvBlock(in_ch, out_ch, key=key)

    def __call__(self, x: jnp.ndarray, skip: jnp.ndarray) -> jnp.ndarray:
        # Nearest-neighbor upsample 2x — x is (C, H, W)
        x = jax.image.resize(x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2),
                             method='nearest')
        # Crop skip to match x if sizes differ by 1 due to odd input dimensions
        x = x[:, :skip.shape[1], :skip.shape[2]]
        x = jnp.concatenate([x, skip], axis=0)
        return self.conv_block(x)


class CorrectionUNet(eqx.Module):
    """Small U-Net for image-space correction.

    Input: (1, H, W) image  →  Output: (1, H, W) residual ΔI

    Architecture (default ch=16):
        enc0: 1 → 16    (H, W)
        enc1: 16 → 32   (H/2, W/2)
        bottleneck: 32 → 64  (H/4, W/4)
        dec1: 64+32 → 32   (H/2, W/2)
        dec0: 32+16 → 16   (H, W)
        head: 16 → 1        (H, W)
    """
    enc0: ConvBlock
    enc1: DownBlock
    bottleneck: DownBlock
    dec1: UpBlock
    dec0: UpBlock
    head: eqx.nn.Conv2d

    def __init__(self, ch: int = 16, *, key: jax.Array):
        k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
        self.enc0 = ConvBlock(1, ch, key=k0)
        self.enc1 = DownBlock(ch, ch * 2, key=k1)
        self.bottleneck = DownBlock(ch * 2, ch * 4, key=k2)
        self.dec1 = UpBlock(ch * 4 + ch * 2, ch * 2, key=k3)
        self.dec0 = UpBlock(ch * 2 + ch, ch, key=k4)
        self.head = eqx.nn.Conv2d(ch, 1, kernel_size=1, key=k5)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass. x is (1, H, W). Returns (1, H, W) residual."""
        s0 = self.enc0(x)           # (ch, H, W)
        s1 = self.enc1(s0)          # (ch*2, H/2, W/2)
        b = self.bottleneck(s1)     # (ch*4, H/4, W/4)
        d1 = self.dec1(b, s1)       # (ch*2, H/2, W/2)
        d0 = self.dec0(d1, s0)      # (ch, H, W)
        return self.head(d0)        # (1, H, W)


def apply_correction(model: CorrectionUNet, I: jnp.ndarray) -> jnp.ndarray:
    """Apply the correction model to a 2D caustic image.

    Parameters
    ----------
    model : trained CorrectionUNet
    I     : caustic image [H, W]

    Returns
    -------
    I_corrected : I + ΔI  [H, W]
    """
    x = I[None, :, :]            # (1, H, W)
    delta = model(x)             # (1, H, W)
    return I + delta[0]          # (H, W)


# ── Serialization ────────────────────────────────────────────────────

def save_model(path: str, model: CorrectionUNet) -> None:
    """Save a trained model to disk via Equinox's leaf serialization.

    Stores only the array leaves; the architecture (channel widths, etc.)
    must be reconstructed at load time. Pair with load_model.
    """
    eqx.tree_serialise_leaves(path, model)


def load_model(path: str, ch: int = 16) -> CorrectionUNet:
    """Load a trained model from disk.

    Parameters
    ----------
    path : file written by save_model
    ch   : base channel count, must match what the saved model was trained with

    Returns
    -------
    Trained CorrectionUNet with parameters loaded from disk.
    """
    # Build a skeleton with the right architecture, then overwrite leaves
    skeleton = CorrectionUNet(ch=ch, key=jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(path, skeleton)


# ── Training ──────────────────────────────────────────────────────────

def generate_training_data(
    prop: Propagator,
    config: NonIdealConfig,
    Omega_freqs: jnp.ndarray,
    T: float,
    n_samples: int,
    key: jax.Array,
    *,
    phasor_scale: float = 0.5,
    sigma: float = 0.02,
    n_water: float = 1.33,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a batch of (I_ideal, delta_I) training pairs.

    Parameters
    ----------
    prop        : Propagator
    config      : NonIdealConfig (fixed non-ideal realization)
    Omega_freqs : driving frequencies
    T           : evaluation time
    n_samples   : number of training pairs
    key         : JAX PRNG key
    phasor_scale: std of random phasor amplitudes
    sigma       : rendering blur
    n_water     : refractive index

    Returns
    -------
    I_ideals  : (n_samples, H, W)
    delta_Is  : (n_samples, H, W)  where delta = I_nonideal - I_ideal
    """
    n_freq = len(Omega_freqs)
    Omega_j = jnp.asarray(Omega_freqs)

    def _one_pair(k: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        params = sample_random_phasors(k, prop.n_act, n_freq, scale=phasor_scale)
        X, Y = unpack_complex(params, prop.n_act, n_freq)
        P = X + 1j * Y
        a_ideal = steady_state_amplitudes(prop, P, Omega_j, T)
        _, _, I_ideal = caustic_image(prop, a_ideal, n_water=n_water, sigma=sigma)
        _, _, I_ni = caustic_image_nonideal(
            prop, config, P, Omega_j, T, n_water=n_water, sigma=sigma)
        return I_ideal, I_ni - I_ideal

    keys = jax.random.split(key, n_samples)
    I_ideals, delta_Is = jax.vmap(_one_pair)(keys)
    return I_ideals, delta_Is


def train_correction(
    model: CorrectionUNet,
    I_ideals: jnp.ndarray,
    delta_Is: jnp.ndarray,
    *,
    lr: float = 1e-3,
    n_epochs: int = 200,
    batch_size: int = 16,
    key: jax.Array,
    verbose: bool = True,
) -> tuple[CorrectionUNet, list[float]]:
    """Train the correction U-Net on precomputed training pairs.

    Parameters
    ----------
    model     : CorrectionUNet (randomly initialized)
    I_ideals  : (N, H, W) ideal caustic images
    delta_Is  : (N, H, W) target residuals (I_nonideal - I_ideal)
    lr        : learning rate
    n_epochs  : number of passes over the dataset
    batch_size: mini-batch size
    key       : JAX PRNG key for shuffling
    verbose   : print progress

    Returns
    -------
    model       : trained CorrectionUNet
    loss_history: average MSE loss per epoch
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    n_samples = I_ideals.shape[0]

    @eqx.filter_jit
    def step(model, opt_state, I_batch, delta_batch):
        def loss_fn(model):
            def predict_one(I_in, delta_target):
                x = I_in[None, :, :]        # (1, H, W)
                delta_pred = model(x)[0]    # (H, W)
                return jnp.mean((delta_pred - delta_target) ** 2)
            return jnp.mean(jax.vmap(predict_one)(I_batch, delta_batch))

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state_new = optimizer.update(grads, opt_state, model)
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss

    loss_history = []

    for epoch in range(n_epochs):
        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, n_samples)
        I_shuf = I_ideals[perm]
        d_shuf = delta_Is[perm]

        epoch_losses = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            I_batch = I_shuf[start:end]
            d_batch = d_shuf[start:end]
            model, opt_state, loss = step(model, opt_state, I_batch, d_batch)
            epoch_losses.append(float(loss))

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_loss)

        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:4d}/{n_epochs}: MSE = {avg_loss:.6f}")

    return model, loss_history


# ── Corrected forward model (for optimization, Step 3) ───────────────

def make_corrected_loss(
    prop: Propagator,
    model: CorrectionUNet,
    target: np.ndarray,
    Omega_freqs: np.ndarray,
    T_eval: float,
    *,
    sigma: float = 0.02,
    n_water: float = 1.33,
    lambda_energy: float = 1e-5,
) -> callable:
    """Build a loss function that uses the ideal simulator + learned correction.

    The returned loss is differentiable w.r.t. params. Gradients flow through
    both the analytical VJP (ideal caustic) and the NN (standard autodiff).

    Parameters
    ----------
    prop        : Propagator
    model       : trained CorrectionUNet
    target      : target image [H, W]
    Omega_freqs : driving frequencies
    T_eval      : evaluation time
    sigma       : rendering blur
    n_water     : refractive index
    lambda_energy: L2 regularization on phasors
    """
    n_act = prop.n_act
    n_freq = len(Omega_freqs)
    Omega = jnp.asarray(Omega_freqs)
    T_jnp = jnp.asarray(target)
    norm_T = float(jnp.sqrt(jnp.sum(T_jnp ** 2) + 1e-12))

    def loss_fn(params: jnp.ndarray) -> jnp.ndarray:
        X, Y = unpack_complex(params, n_act, n_freq)
        P = X + 1j * Y
        a = steady_state_amplitudes(prop, P, Omega, T_eval)
        _, _, I = caustic_image(prop, a, n_water=n_water, sigma=sigma)
        I_corrected = apply_correction(model, I)

        # Cosine loss
        dot = jnp.sum(I_corrected * T_jnp)
        norm_I = jnp.sqrt(jnp.sum(I_corrected ** 2) + 1e-12)
        L_match = 1.0 - dot / (norm_I * norm_T)

        L_energy = jnp.sum(X ** 2) + jnp.sum(Y ** 2)
        return L_match + lambda_energy * L_energy

    return loss_fn
