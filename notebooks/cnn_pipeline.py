"""Combined data-gen + CNN training pipeline, all in memory.

Useful on Colab T4 where session storage is slow (4+ min to write 8 GB
via np.savez). This script generates the dataset and immediately trains
on it without ever touching disk for the data. Only the trained model
checkpoint and a metadata JSON are written.

Re-generation per training run costs ~15s — much cheaper than the disk
roundtrip. To iterate on training hyperparameters: edit this file,
`git push`, then on Colab `git pull` and re-run.

Run:
    python notebooks/cnn_pipeline.py
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from wavetank import (
    Tank, Actuator, build_propagator,
    steady_state_amplitudes, caustic_image, reconstruct_surface,
    unpack_complex, sample_random_phasors,
)

# ── Apparatus + data-gen config ──────────────────────────────────────
LX, LY, DEPTH, DAMPING = 1.0, 1.0, 3.0, 0.02
N_MODES        = 12
NX, NY         = 64, 64
N_ACT_PER_SIDE = 5
FREQS_HZ       = (1.0, 1.5, 2.0, 2.5)
T_EVAL         = 1.0
SIGMA_RENDER   = 0.02
PHASOR_SCALE_MIN = 5e-5
PHASOR_SCALE_MAX = 5e-4
ETA_CAP        = 0.10
SLOPE_CAP      = 0.10

N_SAMPLES   = 500_000
BATCH_SIZE  = 1024
SEED_GEN    = 0

# ── Training config ──────────────────────────────────────────────────
VAL_FRAC     = 0.10
TRAIN_BATCH  = 128
N_EPOCHS     = 30
LR_INITIAL   = 3e-4
LR_FINAL     = 1e-5
WARMUP_STEPS = 1000
GRAD_CLIP    = 0.5
SEED_TRAIN   = 0

# ── Output ───────────────────────────────────────────────────────────
DATA_DIR     = Path("data/naive_inverse")
CKPT_PATH    = DATA_DIR / "model_cnn.eqx"
HISTORY_PATH = DATA_DIR / "history_cnn.json"
META_PATH    = DATA_DIR / "metadata.json"


# ── Apparatus setup ──────────────────────────────────────────────────
def build_setup():
    tank = Tank(Lx=LX, Ly=LY, depth=DEPTH, damping=DAMPING)
    acts = []
    for i in range(N_ACT_PER_SIDE):
        t = (i + 1) / (N_ACT_PER_SIDE + 1)
        acts += [
            Actuator(x=0.0,    y=t * LY),
            Actuator(x=LX,     y=t * LY),
            Actuator(x=t * LX, y=0.0),
            Actuator(x=t * LX, y=LY),
        ]
    prop  = build_propagator(tank, acts, n_modes=N_MODES, nx=NX, ny=NY)
    Omega = jnp.asarray([2 * np.pi * f for f in FREQS_HZ])
    return prop, Omega


# ── In-memory data generation ────────────────────────────────────────
def make_batch_fn(prop, Omega):
    n_act, n_freq = prop.n_act, Omega.shape[0]

    def _one(key):
        k_scale, k_phasor = jax.random.split(key)
        log_scale = jax.random.uniform(
            k_scale,
            minval=jnp.log(PHASOR_SCALE_MIN),
            maxval=jnp.log(PHASOR_SCALE_MAX),
        )
        scale = jnp.exp(log_scale)
        p = sample_random_phasors(k_phasor, n_act, n_freq, scale=scale)
        X, Y = unpack_complex(p, n_act, n_freq)
        P = X + 1j * Y
        a = steady_state_amplitudes(prop, P, Omega, T_EVAL)

        eta, deta_dx, deta_dy = reconstruct_surface(prop, a)
        max_eta_norm = jnp.max(jnp.abs(eta)) / DEPTH
        max_slope    = jnp.max(jnp.sqrt(deta_dx ** 2 + deta_dy ** 2))

        _, _, I = caustic_image(prop, a, sigma=SIGMA_RENDER)
        return p, I, max_eta_norm, max_slope

    return jax.jit(jax.vmap(_one))


def generate_data(prop, Omega):
    n_params = 2 * prop.n_act * Omega.shape[0]
    batch_fn = make_batch_fn(prop, Omega)
    rng = jax.random.PRNGKey(SEED_GEN)

    print("JIT compiling generator...", flush=True)
    t_c = time.perf_counter()
    _p, _I, _e, _s = batch_fn(jax.random.split(rng, BATCH_SIZE))
    _I.block_until_ready()
    print(f"  compile: {time.perf_counter() - t_c:.1f}s")

    phasors_all  = np.zeros((N_SAMPLES, n_params), dtype=np.float32)
    caustics_all = np.zeros((N_SAMPLES, NX, NY),   dtype=np.float32)

    filled = 0
    batch_idx = 0
    t0 = time.perf_counter()
    while filled < N_SAMPLES:
        keys = jax.random.split(jax.random.fold_in(rng, batch_idx + 1), BATCH_SIZE)
        p_b, I_b, eta_b, slope_b = batch_fn(keys)
        I_b.block_until_ready()

        mask = (np.asarray(eta_b) < ETA_CAP) & (np.asarray(slope_b) < SLOPE_CAP)
        n_keep = min(int(mask.sum()), N_SAMPLES - filled)
        if n_keep > 0:
            phasors_all [filled:filled + n_keep] = np.asarray(p_b)[mask][:n_keep]
            caustics_all[filled:filled + n_keep] = np.asarray(I_b)[mask][:n_keep]
            filled += n_keep
        batch_idx += 1

    t = time.perf_counter() - t0
    print(f"Generated {filled} valid samples in {t:.1f}s ({filled / t:.0f}/s)")
    return phasors_all, caustics_all


# ── CNN model ────────────────────────────────────────────────────────
class InversionCNN(eqx.Module):
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
        self.mlp = eqx.nn.MLP(
            in_size=128 * 8 * 8, out_size=n_params,
            width_size=256, depth=1,
            activation=jax.nn.gelu,
            key=keys[6],
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.encoder:
            x = jax.nn.gelu(layer(x))
        x = x.flatten()
        return self.mlp(x)


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


def train(prop, Omega, phasors_all, caustics_all):
    n_params = phasors_all.shape[1]

    # Normalize (in-place, no extra memory)
    n_sample = min(10_000, caustics_all.shape[0])
    caustic_scale = float(np.percentile(caustics_all[:n_sample], 99))
    phasor_scale  = float(np.max(np.abs(phasors_all)))
    caustics_all /= caustic_scale
    phasors_all  /= phasor_scale
    print(f"  caustic_scale={caustic_scale:.3f}  phasor_scale={phasor_scale:.5f}")

    # Split via slicing (views, no copy)
    n_val = int(N_SAMPLES * VAL_FRAC)
    x_train, y_train = caustics_all[n_val:], phasors_all[n_val:]
    x_val,   y_val   = caustics_all[:n_val], phasors_all[:n_val]
    x_train = x_train[:, None, :, :]
    x_val   = x_val  [:, None, :, :]
    print(f"  train: {x_train.shape[0]} samples  val: {x_val.shape[0]}")

    key = jax.random.PRNGKey(SEED_TRAIN)
    model = InversionCNN(n_params=n_params, key=key)
    n_model_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"  model: {n_model_params:,} parameters")

    n_train_batches = x_train.shape[0] // TRAIN_BATCH
    total_steps = N_EPOCHS * n_train_batches
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
    rng_np = np.random.default_rng(SEED_TRAIN)
    t0 = time.perf_counter()
    print(f"\n  training {N_EPOCHS} epochs × {n_train_batches} batches "
          f"(batch_size={TRAIN_BATCH})\n", flush=True)

    for epoch in range(N_EPOCHS):
        perm = rng_np.permutation(x_train.shape[0])
        train_losses = []
        for b in range(n_train_batches):
            idx = perm[b * TRAIN_BATCH:(b + 1) * TRAIN_BATCH]
            model, opt_state, loss = train_step(
                model, opt_state, x_train[idx], y_train[idx],
            )
            train_losses.append(float(loss))

        val_losses = []
        n_val_batches = max(1, x_val.shape[0] // TRAIN_BATCH)
        for b in range(n_val_batches):
            s = b * TRAIN_BATCH
            e = min(s + TRAIN_BATCH, x_val.shape[0])
            val_losses.append(float(val_loss(model, x_val[s:e], y_val[s:e])))

        t_loss = float(np.mean(train_losses))
        v_loss = float(np.mean(val_losses))
        elapsed = time.perf_counter() - t0
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["time_s"].append(elapsed)
        print(f"  epoch {epoch+1:>3}/{N_EPOCHS}  "
              f"train={t_loss:.4e}  val={v_loss:.4e}  "
              f"elapsed={elapsed:.1f}s", flush=True)

    return model, history, caustic_scale, phasor_scale


def save_outputs(model, history, caustic_scale, phasor_scale, prop, Omega):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(CKPT_PATH, model)

    history["caustic_scale"]   = caustic_scale
    history["phasor_scale"]    = phasor_scale
    history["n_train_samples"] = int((1 - VAL_FRAC) * N_SAMPLES)
    history["n_val_samples"]   = int(VAL_FRAC * N_SAMPLES)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    # Metadata needed by eval_cnn_inverse.py
    meta = dict(
        Lx=LX, Ly=LY, depth=DEPTH, damping=DAMPING,
        n_modes=N_MODES, nx=NX, ny=NY,
        n_act_per_side=N_ACT_PER_SIDE, n_act=int(prop.n_act),
        freqs_hz=list(FREQS_HZ), T_eval=T_EVAL,
        sigma_render=SIGMA_RENDER,
        n_samples=int(N_SAMPLES), seed=SEED_GEN,
    )
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {CKPT_PATH}")
    print(f"Saved {HISTORY_PATH}")
    print(f"Saved {META_PATH}")


def main():
    print(f"JAX {jax.__version__} on {jax.default_backend()}\n", flush=True)
    prop, Omega = build_setup()
    print(f"  actuators={prop.n_act}  freqs={Omega.shape[0]}  "
          f"modes/axis={N_MODES}  grid={NX}x{NY}\n", flush=True)

    phasors_all, caustics_all = generate_data(prop, Omega)
    model, history, c_scale, p_scale = train(prop, Omega, phasors_all, caustics_all)
    save_outputs(model, history, c_scale, p_scale, prop, Omega)


if __name__ == "__main__":
    main()
