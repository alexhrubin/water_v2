"""Exact replica of example.ipynb cells 23-29 (Gaussian spots).

Goal: reproduce example.ipynb's visual quality in python by matching its setup
as closely as possible. If this works → we have a baseline that we can then
vary one knob at a time to understand what matters. If it doesn't work →
there's a python/Julia simulator-level difference to find.

Apparatus matches cell 6 exactly:
  depth=0.1m, 24 freqs (0.5-5Hz), 48 actuators (12/side), 15 modes/axis, 200×200 grid

Optimization matches cells 24, 29:
  Adam (Julia's only choice — no L-BFGS in that version of optimize_caustic)
  Random initialization (no analytical warm-start in cell 24)
  RAW target (cell 24 passes target_gauss directly, NOT a band-limited version)
  Paraxial Snell (Julia default)
  NO cap penalties (Julia optimize_caustic has only λ_energy=1e-5)
  3-stage σ-annealing 0.04→0.02→0.01, 500 iters each
  T_eval=1.5
  Temporal window n=5, σ=0.04
"""

import time
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from wavetank import (
    Tank, Actuator, build_propagator,
    steady_state_amplitudes, caustic_image, unpack_complex,
    Stage, optimize_caustic,
)

# ── Apparatus (cell 6) ──────────────────────────────────────────────
LX, LY         = 1.0, 1.0
DEPTH          = 0.1
DAMPING        = 0.02
NX, NY         = 200, 200
N_ACT_PER_SIDE = 12
N_MODES        = 15
N_FREQ         = 24
FREQ_MIN_HZ    = 0.5
FREQ_MAX_HZ    = 5.0
T_EVAL         = 1.5
SIGMA_RENDER   = 0.005

# ── Target (cell 28) ────────────────────────────────────────────────
SIGMA_SPOT = 0.04
SPOTS = [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)]

# ── Temporal window (cell 29) ───────────────────────────────────────
N_TEMPORAL     = 5
SIGMA_TEMPORAL = 0.04

# ── Optimization (cell 29) ──────────────────────────────────────────
STAGES = (
    Stage(sigma=0.04, sigma_blur=0.04, iters=500, method='adam'),
    Stage(sigma=0.02, sigma_blur=0.02, iters=500, method='adam'),
    Stage(sigma=0.01, sigma_blur=0.01, iters=500, method='adam'),
)
LR             = 1e-3
LAMBDA_ETA     = 0.0       # Julia has NO eta penalty
LAMBDA_SLOPE   = 0.0       # Julia has NO slope penalty
LAMBDA_ENERGY  = 1e-5
FULL_SNELL     = False     # Julia uses paraxial

OUT_DIR = Path("data/example_replica")


def build_setup():
    tank = Tank(Lx=LX, Ly=LY, depth=DEPTH, damping=DAMPING)
    acts = []
    for i in range(N_ACT_PER_SIDE):
        t = (i + 1) / (N_ACT_PER_SIDE + 1)
        acts += [
            Actuator(x=0.0,        y=t * LY),
            Actuator(x=LX,         y=t * LY),
            Actuator(x=t * LX,     y=0.0),
            Actuator(x=t * LX,     y=LY),
        ]
    prop  = build_propagator(tank, acts, n_modes=N_MODES, nx=NX, ny=NY)
    Omega = jnp.asarray([2 * np.pi * f for f in np.linspace(FREQ_MIN_HZ, FREQ_MAX_HZ, N_FREQ)])
    return prop, Omega


def make_target():
    xs = np.linspace(0, LX, NX)
    ys = np.linspace(0, LY, NY)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    target = np.zeros((NX, NY))
    for cx, cy in SPOTS:
        target += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * SIGMA_SPOT ** 2))
    target /= target.max()
    return target


def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"JAX {jax.__version__} on {jax.default_backend()}\n", flush=True)

    prop, Omega = build_setup()
    n_act, n_freq = prop.n_act, len(Omega)
    print(f"Apparatus: depth={DEPTH}m, actuators={n_act}, freqs={n_freq}")
    print(f"           modes={N_MODES}², grid={NX}×{NY}, phasor DOFs={2 * n_act * n_freq}")
    print(f"Optimizer: Adam, no caps, RANDOM init (matches Julia)")
    print(f"T_eval={T_EVAL}, temporal window n={N_TEMPORAL} σ={SIGMA_TEMPORAL}\n", flush=True)

    target = make_target().astype(np.float32)
    T_array = list(T_EVAL + np.linspace(-SIGMA_TEMPORAL, SIGMA_TEMPORAL, N_TEMPORAL))

    print("Optimizing 3-spot Gaussian (mimics example.ipynb cell 29)...\n", flush=True)
    t0 = time.perf_counter()
    params, history = optimize_caustic(
        prop, target, np.asarray(Omega), T_array,
        stages=STAGES,
        lr=LR,
        lambda_eta=LAMBDA_ETA,
        lambda_slope=LAMBDA_SLOPE,
        lambda_energy=LAMBDA_ENERGY,
        loss_type='cosine',
        full_snell=FULL_SNELL,
        p0=None,
        check_validity=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nOptimization elapsed: {elapsed:.1f}s   final loss: {history[-1]:.4e}")

    # Render final caustic at T_eval (single phase)
    X, Y = unpack_complex(jnp.asarray(params), n_act, n_freq)
    P = X + 1j * Y
    a = steady_state_amplitudes(prop, P, Omega, T_EVAL)
    _, _, I_final = caustic_image(prop, a, sigma=SIGMA_RENDER)
    I_final = np.asarray(I_final)
    I_show = I_final / max(I_final.max(), 1e-9)
    cs = cosine_sim(target, I_show)
    print(f"cos(target, render): {cs:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(target, cmap="inferno"); axes[0].set_title("3-spot Gaussian target")
    axes[1].imshow(I_show, cmap="inferno"); axes[1].set_title(f"Optimized caustic   cos={cs:.3f}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_png = OUT_DIR / "3spot_gaussian.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
