"""Evaluate the CNN-with-flatten inverse model on out-of-distribution targets.

Mirror of eval_naive_inverse.py but uses the CNN architecture from
train_cnn_inverse.py. Loads the CNN checkpoint, runs each target through
the model, pushes predicted phasors through the simulator, and plots the
target vs. predicted caustic side-by-side.

Run as a script:
    python notebooks/eval_cnn_inverse.py
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt

from wavetank import (
    Tank, Actuator, build_propagator,
    steady_state_amplitudes, caustic_image,
    unpack_complex, load_target_image,
)

# ── Config ───────────────────────────────────────────────────────────
DATA_DIR     = Path("data/naive_inverse")
CKPT_PATH    = DATA_DIR / "model_cnn.eqx"
HISTORY_PATH = DATA_DIR / "history_cnn.json"
META_PATH    = DATA_DIR / "metadata.json"
TARGETS_DIR  = Path("targets")
OUT_PATH     = DATA_DIR / "eval_cnn_ood.png"

TARGET_FILES = [
    "dog_square.jpg",
    "head.jpg",
    "ANNA.jpg",
    "HELLO.jpg",
]


# ── Model definition (must mirror train_cnn_inverse.py) ──────────────
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


# ── Simulator setup from metadata ────────────────────────────────────
def build_setup(meta):
    tank = Tank(
        Lx=meta["Lx"], Ly=meta["Ly"],
        depth=meta["depth"], damping=meta["damping"],
    )
    acts = []
    n_per_side = meta["n_act_per_side"]
    for i in range(n_per_side):
        t = (i + 1) / (n_per_side + 1)
        acts += [
            Actuator(x=0.0,            y=t * meta["Ly"]),
            Actuator(x=meta["Lx"],     y=t * meta["Ly"]),
            Actuator(x=t * meta["Lx"], y=0.0),
            Actuator(x=t * meta["Lx"], y=meta["Ly"]),
        ]
    prop  = build_propagator(tank, acts, n_modes=meta["n_modes"],
                             nx=meta["nx"], ny=meta["ny"])
    Omega = jnp.asarray([2 * np.pi * f for f in meta["freqs_hz"]])
    return prop, Omega


def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print(f"JAX {jax.__version__} on {jax.default_backend()}")
    with open(META_PATH) as f:    meta    = json.load(f)
    with open(HISTORY_PATH) as f: history = json.load(f)

    prop, Omega = build_setup(meta)
    n_params    = 2 * prop.n_act * len(Omega)
    c_scale     = history["caustic_scale"]
    p_scale     = history["phasor_scale"]
    print(f"  n_params={n_params}  caustic_scale={c_scale:.3f}  phasor_scale={p_scale:.5f}")

    model = InversionCNN(n_params=n_params, key=jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(CKPT_PATH, model)

    xs = np.linspace(0, meta["Lx"], meta["nx"])
    ys = np.linspace(0, meta["Ly"], meta["ny"])

    def predict_caustic(target_norm):
        pred_norm = model(target_norm[None])
        p = pred_norm * p_scale
        X, Y = unpack_complex(p, prop.n_act, len(Omega))
        P = X + 1j * Y
        a = steady_state_amplitudes(prop, P, Omega, meta["T_eval"])
        _, _, I = caustic_image(prop, a, sigma=meta["sigma_render"])
        return I, p

    n = len(TARGET_FILES)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.5 * n))
    if n == 1:
        axes = axes[None, :]

    print(f"\n  {'target':<20}  {'‖p‖':>10}  {'I_pred max':>10}  {'cos(target, pred)':>18}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*18}")

    for i, fname in enumerate(TARGET_FILES):
        target = load_target_image(TARGETS_DIR / fname, xs, ys)
        target = target[::-1].T
        target = target / max(target.max(), 1e-9)

        p99 = max(float(np.percentile(target, 99)), 1e-6)
        target_in = jnp.asarray((target / p99).astype(np.float32))

        I_pred, p_pred = predict_caustic(target_in)
        I_pred = np.asarray(I_pred)
        p_pred = np.asarray(p_pred)

        I_pred_show = I_pred / max(I_pred.max(), 1e-9)
        cs = cosine_sim(target, I_pred_show)

        axes[i, 0].imshow(target,      cmap="inferno"); axes[i, 0].set_title(f"target: {fname}")
        axes[i, 1].imshow(target_in,   cmap="inferno"); axes[i, 1].set_title("model input (norm)")
        axes[i, 2].imshow(I_pred_show, cmap="inferno"); axes[i, 2].set_title(f"CNN net → simulator   cos={cs:.3f}")
        for ax in axes[i]:
            ax.axis("off")

        print(f"  {fname:<20}  {np.linalg.norm(p_pred):>10.3e}  "
              f"{I_pred.max():>10.3f}  {cs:>18.3f}")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
