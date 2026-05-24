"""Train a CNN-with-flatten inverse model on the random-phasor dataset.

Variant of train_naive_inverse.py with a CNN encoder + flatten + MLP head.
The first CNN attempt used global average pool, which collapsed the
spatial information needed to invert the caustic. This version keeps the
conv stack (so we get the locality inductive bias) but flattens instead
of pooling — preserving every spatial feature.

Param count (~2.4M) is intentionally close to the pure-MLP version (~2.7M)
so the head-to-head comparison is about inductive bias, not capacity.

Run as a script:
    python notebooks/train_cnn_inverse.py
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

# ── Config ───────────────────────────────────────────────────────────
DATA_DIR     = Path("data/naive_inverse")
DATASET_PATH = DATA_DIR / "dataset.npz"
CKPT_PATH    = DATA_DIR / "model_cnn.eqx"
HISTORY_PATH = DATA_DIR / "history_cnn.json"

VAL_FRAC     = 0.10
BATCH_SIZE   = 128
N_EPOCHS     = 30
LR_INITIAL   = 3e-4         # was 1e-3 — too aggressive, caused divergence at e6
LR_FINAL     = 1e-5
WARMUP_STEPS = 1000         # linear warmup from 0 → LR_INITIAL
GRAD_CLIP    = 0.5          # was 1.0 — tighter to catch bad batches
WEIGHT_DECAY = 1e-4
SEED         = 0
SANITY_N     = None


# ── Network ──────────────────────────────────────────────────────────
class InversionCNN(eqx.Module):
    """CNN encoder + flatten + small MLP head.

    The conv stack extracts local features cheaply (locality inductive bias).
    Flatten preserves spatial structure (what an inverse problem needs).
    Small MLP head maps spatial features → phasor vector.
    """
    encoder: tuple
    mlp: eqx.nn.MLP

    def __init__(self, n_params: int, *, key: jax.Array):
        keys = jax.random.split(key, 7)
        self.encoder = (
            eqx.nn.Conv2d(1,   32,  kernel_size=3, padding=1, key=keys[0]),
            eqx.nn.Conv2d(32,  32,  kernel_size=3, stride=2, padding=1, key=keys[1]),
            eqx.nn.Conv2d(32,  64,  kernel_size=3, padding=1, key=keys[2]),
            eqx.nn.Conv2d(64,  64,  kernel_size=3, stride=2, padding=1, key=keys[3]),
            eqx.nn.Conv2d(64,  128, kernel_size=3, padding=1, key=keys[4]),
            eqx.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, key=keys[5]),
        )
        # After 3 stride-2 downsamples on 64x64 input: spatial 8x8, 128 ch → 8192 features
        self.mlp = eqx.nn.MLP(
            in_size=128 * 8 * 8, out_size=n_params,
            width_size=256, depth=1,
            activation=jax.nn.gelu,
            key=keys[6],
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (1, 64, 64) per example; vmap'd at the call site
        for layer in self.encoder:
            x = jax.nn.gelu(layer(x))
        x = x.flatten()              # (128, 8, 8) → (8192,)  — no GAP
        return self.mlp(x)


# ── Data ─────────────────────────────────────────────────────────────
def load_data():
    print(f"Loading {DATASET_PATH}...")
    data = np.load(DATASET_PATH)
    caustics = data["caustics"]
    phasors  = data["phasors"]

    n_sample = min(10_000, caustics.shape[0])
    caustic_scale = float(np.percentile(caustics[:n_sample], 99))
    phasor_scale  = float(np.max(np.abs(phasors)))
    caustics /= caustic_scale
    phasors  /= phasor_scale

    print(f"  caustics {caustics.shape}  normalized by {caustic_scale:.3f}")
    print(f"  phasors  {phasors.shape}  normalized by {phasor_scale:.5f}")
    return caustics, phasors, caustic_scale, phasor_scale


def train_val_split(caustics, phasors, val_frac):
    n = caustics.shape[0]
    n_val = int(n * val_frac)
    return (
        caustics[n_val:], phasors[n_val:],
        caustics[:n_val], phasors[:n_val],
    )


# ── Training step ────────────────────────────────────────────────────
def make_train_step(optimizer):
    @eqx.filter_jit
    def step(model, opt_state, x_batch, y_batch):
        def loss_fn(m):
            y_pred = jax.vmap(m)(x_batch)
            return jnp.mean((y_pred - y_batch) ** 2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    return step


@eqx.filter_jit
def val_loss(model, x_batch, y_batch):
    y_pred = jax.vmap(model)(x_batch)
    return jnp.mean((y_pred - y_batch) ** 2)


def debug_model(model, x, y, label):
    y_pred = jax.vmap(model)(x[:32])
    err = y_pred - y[:32]
    # last_w from MLP head's final linear layer
    last_w = model.mlp.layers[-1].weight
    print(f"  [{label:>10}]  "
          f"y_pred std={float(y_pred.std()):.4e}  "
          f"y_true std={float(y[:32].std()):.4e}  "
          f"mse={float(jnp.mean(err**2)):.4e}  "
          f"last_w[0,0]={float(last_w[0,0]):+.4e}")


def main():
    print(f"JAX {jax.__version__} on {jax.default_backend()}\n")

    caustics, phasors, c_scale, p_scale = load_data()

    if SANITY_N is not None:
        caustics = caustics[:SANITY_N]
        phasors  = phasors[:SANITY_N]
        print(f"  SANITY CHECK: subset to {len(caustics)} samples")

    n_params = phasors.shape[1]

    key = jax.random.PRNGKey(SEED)
    key, _, k_init = jax.random.split(key, 3)

    x_train, y_train, x_val, y_val = train_val_split(caustics, phasors, VAL_FRAC)
    x_train = x_train[:, None, :, :]
    x_val   = x_val  [:, None, :, :]
    print(f"\n  train: {x_train.shape[0]} samples  val: {x_val.shape[0]}")

    model = InversionCNN(n_params=n_params, key=k_init)
    n_model_params = sum(
        x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))
    )
    print(f"  model: {n_model_params:,} parameters")

    n_train_batches = x_train.shape[0] // BATCH_SIZE
    total_steps = N_EPOCHS * n_train_batches
    # Warmup-cosine: linear ramp 0 → LR_INITIAL over WARMUP_STEPS, then
    # cosine decay to LR_FINAL over the rest. Warmup is the canonical fix
    # for early-training divergence in deep networks.
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LR_INITIAL,
        warmup_steps=WARMUP_STEPS,
        decay_steps=total_steps,
        end_value=LR_FINAL,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adam(lr_schedule),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    train_step = make_train_step(optimizer)

    history = {"train_loss": [], "val_loss": [], "time_s": []}
    t0 = time.perf_counter()

    print(f"\n  training {N_EPOCHS} epochs × {n_train_batches} batches "
          f"(batch_size={BATCH_SIZE})\n")

    debug_model(model, x_train, y_train, "init")
    rng_np = np.random.default_rng(SEED)

    for epoch in range(N_EPOCHS):
        perm = rng_np.permutation(x_train.shape[0])

        train_losses = []
        for b in range(n_train_batches):
            idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            model, opt_state, loss = train_step(
                model, opt_state, x_train[idx], y_train[idx],
            )
            train_losses.append(float(loss))

        if epoch in (0, 4, 19, 29):
            debug_model(model, x_train, y_train, f"epoch {epoch+1}")

        val_losses = []
        n_val_batches = max(1, x_val.shape[0] // BATCH_SIZE)
        for b in range(n_val_batches):
            s = b * BATCH_SIZE
            e = min(s + BATCH_SIZE, x_val.shape[0])
            val_losses.append(float(val_loss(model, x_val[s:e], y_val[s:e])))
        v_loss = float(np.mean(val_losses))
        t_loss = float(np.mean(train_losses))
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["time_s"].append(elapsed)

        print(f"  epoch {epoch+1:>3}/{N_EPOCHS}  "
              f"train={t_loss:.4e}  val={v_loss:.4e}  "
              f"elapsed={elapsed:.1f}s")

    eqx.tree_serialise_leaves(CKPT_PATH, model)
    print(f"\nSaved model to {CKPT_PATH}")

    history["caustic_scale"]   = c_scale
    history["phasor_scale"]    = p_scale
    history["n_train_samples"] = int(x_train.shape[0])
    history["n_val_samples"]   = int(x_val.shape[0])
    history["n_model_params"]  = int(n_model_params)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
