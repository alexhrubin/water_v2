"""Train naive image→phasor regression CNN on the random-phasor dataset.

Loads (caustic, phasor) pairs from data/naive_inverse/dataset.npz, trains a
small CNN encoder + MLP head to predict phasors from caustic images, saves the
trained checkpoint and the loss history.

Run as a script:
    python notebooks/train_naive_inverse.py
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
CKPT_PATH    = DATA_DIR / "model.eqx"
HISTORY_PATH = DATA_DIR / "history.json"

VAL_FRAC     = 0.10
BATCH_SIZE   = 128
N_EPOCHS     = 30
LR_INITIAL   = 1e-3
LR_FINAL     = 1e-5
WEIGHT_DECAY = 1e-4
SEED         = 0
SANITY_N     = None     # None = full dataset; set to int for subset sanity check


# ── Network ──────────────────────────────────────────────────────────
class InversionCNN(eqx.Module):
    """Image → phasor regression: small CNN encoder + MLP head."""
    encoder: list
    mlp: eqx.nn.MLP

    def __init__(self, n_params: int, *, key: jax.Array):
        # SANITY CHECK: pure MLP, no convs. Tests whether the architecture
        # (conv stack + GAP) was the issue.
        self.encoder = []  # disabled
        self.mlp = eqx.nn.MLP(
            in_size=64 * 64, out_size=n_params,
            width_size=512, depth=3,
            activation=jax.nn.gelu,
            key=key,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (1, H, W) → flatten → MLP
        return self.mlp(x.flatten())


# ── Data ─────────────────────────────────────────────────────────────
def load_data():
    print(f"Loading {DATASET_PATH}...")
    data = np.load(DATASET_PATH)
    caustics = data["caustics"]   # (N, H, W) float32
    phasors  = data["phasors"]    # (N, P) float32

    # Normalize: caustics by 99th percentile (robust to bright-spot tail),
    # phasors by max-abs so targets land in [-1, 1].
    # Percentile on a subsample (cheap and statistically identical) — sorting
    # 16 GB of float32 needs 30+ GB peak memory.
    n_sample = min(10_000, caustics.shape[0])
    caustic_scale = float(np.percentile(caustics[:n_sample], 99))
    phasor_scale  = float(np.max(np.abs(phasors)))
    # In-place division to avoid allocating a second 16 GB buffer
    caustics /= caustic_scale
    phasors  /= phasor_scale

    print(f"  caustics {caustics.shape}  normalized by {caustic_scale:.3f}")
    print(f"  phasors  {phasors.shape}  normalized by {phasor_scale:.5f}")
    return caustics, phasors, caustic_scale, phasor_scale


def train_val_split(caustics, phasors, val_frac, key):
    # Plain slicing returns views — no copy, no extra memory.
    # Data was already random-sampled at generation, so no shuffle needed.
    n = caustics.shape[0]
    n_val = int(n * val_frac)
    return (
        caustics[n_val:], phasors[n_val:],
        caustics[:n_val], phasors[:n_val],
    )


# ── Training step ────────────────────────────────────────────────────
def make_train_step(optimizer):
    """Close over the optimizer (canonical equinox pattern)."""
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
    """Print output stats and a sample weight to diagnose if model is learning."""
    y_pred = jax.vmap(model)(x[:32])
    err = y_pred - y[:32]
    # Sample weight: last MLP layer
    last_w = model.mlp.layers[-1].weight
    print(f"  [{label:>10}]  "
          f"y_pred std={float(y_pred.std()):.4e}  "
          f"y_true std={float(y[:32].std()):.4e}  "
          f"mse={float(jnp.mean(err**2)):.4e}  "
          f"last_w[0,0]={float(last_w[0,0]):+.4e}")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print(f"JAX {jax.__version__} on {jax.default_backend()}\n")

    caustics, phasors, c_scale, p_scale = load_data()

    if SANITY_N is not None:
        caustics = caustics[:SANITY_N]
        phasors  = phasors[:SANITY_N]
        print(f"  SANITY CHECK: subset to {len(caustics)} samples")

    n_params = phasors.shape[1]

    key = jax.random.PRNGKey(SEED)
    key, k_split, k_init, k_shuffle = jax.random.split(key, 4)

    x_train, y_train, x_val, y_val = train_val_split(
        caustics, phasors, VAL_FRAC, k_split,
    )
    # Add channel dim for CNN input
    x_train = x_train[:, None, :, :]
    x_val   = x_val  [:, None, :, :]
    print(f"\n  train: {x_train.shape[0]} samples  val: {x_val.shape[0]}")

    model = InversionCNN(n_params=n_params, key=k_init)
    n_model_params = sum(
        x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))
    )
    print(f"  model: {n_model_params:,} parameters")

    n_train_batches = x_train.shape[0] // BATCH_SIZE
    # SANITY CHECK: plain adam with constant LR, no weight decay
    optimizer = optax.adam(LR_INITIAL)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    train_step = make_train_step(optimizer)

    # Keep arrays as numpy — JAX will convert per-batch (cheap).
    # Pre-converting to jnp duplicates the dataset in memory, which OOMs
    # at 1M samples on a 24 GB machine.

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

        if epoch in (0, 4, 19, 49, 99, 199):
            debug_model(model, x_train, y_train, f"epoch {epoch+1}")

        # Validation in batches to avoid materialising x_val all at once
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

    # Save model + history + normalization scales
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
