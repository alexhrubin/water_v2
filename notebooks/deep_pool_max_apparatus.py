"""Reproduce example.ipynb-quality caustics WITHIN the linear regime.

example.ipynb's striking results come from (a) a much richer apparatus
(24 freqs × 48 actuators, 200×200 grid) and (b) running optimization
without amplitude/slope penalties — almost certainly past the linear-wave
validity envelope (|η|/d, |∇η| > 0.1).

Hypothesis: (a) is what drives the visual quality. (b) is just how Julia
got the surface amplitudes high enough to make the small throw budget
(d = 0.1 m → 2.5 mm displacement budget) work. If we deepen the water
to 3 m, the same surface amplitudes that produced example's caustics now
fit comfortably under the linearity cap — so we should be able to match
example's quality physically, with all the safety regularizers on.

Apparatus matched to example.ipynb:
  24 freqs, 48 actuators, 15 modes/axis, 200×200 grid,
  3-stage σ-annealing (0.04 → 0.02 → 0.01), 5-frame temporal window.

Difference from example.ipynb:
  depth = 3 m (vs 0.1 m), lambda_eta = lambda_slope = 100 (vs 0).
  Validity report enabled — flags any cap violations.

Run:
    python notebooks/deep_pool_max_apparatus.py
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from wavetank import (
    Tank, Actuator, build_propagator,
    steady_state_amplitudes, caustic_image, unpack_complex,
    load_target_image,
    Stage, optimize_caustic,
)

# ── Apparatus (matched to example.ipynb except for depth) ────────────
LX, LY         = 1.0, 1.0
DEPTH          = 3.0           # vs example's 0.1 m — gives 30× more throw budget
DAMPING        = 0.02
NX, NY         = 200, 200
N_ACT_PER_SIDE = 12            # 48 total — matches example
N_MODES        = 15            # per axis — matches example
N_FREQ         = 24
FREQ_MIN_HZ    = 0.5
FREQ_MAX_HZ    = 5.0
T_EVAL         = 1.0
SIGMA_RENDER   = 0.005

# Temporal window: 5 sample times centered on T_EVAL ± σ_temporal
N_TEMPORAL     = 5
SIGMA_TEMPORAL = 0.033

# Cap enforcement (NOT in example.ipynb)
LAMBDA_ETA     = 100.0
LAMBDA_SLOPE   = 100.0
LAMBDA_ENERGY  = 1e-5

# Optimization
LR             = 1e-3
STAGES = (
    Stage(sigma=0.04, sigma_blur=0.04, iters=500),
    Stage(sigma=0.02, sigma_blur=0.02, iters=500),
    Stage(sigma=0.01, sigma_blur=0.01, iters=500),
)

TARGETS_DIR = Path("targets")
OUT_DIR     = Path("data/deep_pool_max")
TARGETS = ["dog_square.jpg", "head.jpg", "ANNA.jpg", "HELLO.jpg"]


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


def temporal_window():
    if N_TEMPORAL == 1:
        return T_EVAL
    return list(T_EVAL + np.linspace(-SIGMA_TEMPORAL, SIGMA_TEMPORAL, N_TEMPORAL))


def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"JAX {jax.__version__} on {jax.default_backend()}\n", flush=True)

    prop, Omega = build_setup()
    n_act, n_freq = prop.n_act, len(Omega)
    n_params = 2 * n_act * n_freq
    print(f"Apparatus: depth={DEPTH}m, actuators={n_act} ({N_ACT_PER_SIDE}/side), "
          f"freqs={n_freq} [{FREQ_MIN_HZ}-{FREQ_MAX_HZ}Hz]")
    print(f"           modes={N_MODES}², grid={NX}×{NY}, phasor DOFs={n_params}")
    print(f"Regularizers: λ_eta={LAMBDA_ETA}, λ_slope={LAMBDA_SLOPE} (linear regime enforced)")
    print(f"Optimizer: Adam lr={LR}, {sum(s.iters for s in STAGES)} total iters across "
          f"{len(STAGES)} stages, temporal window n={N_TEMPORAL}\n", flush=True)

    xs = np.linspace(0, LX, NX)
    ys = np.linspace(0, LY, NY)
    T_array = temporal_window()

    results = {}
    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(8, 4 * len(TARGETS)))

    for i, fname in enumerate(TARGETS):
        print(f"\n{'='*60}\n  Target: {fname}\n{'='*60}", flush=True)
        target = load_target_image(TARGETS_DIR / fname, xs, ys)
        target = target[::-1].T
        target = target / max(target.max(), 1e-9)

        t0 = time.perf_counter()
        params, history = optimize_caustic(
            prop, target.astype(np.float32), np.asarray(Omega), T_array,
            stages=STAGES,
            lr=LR,
            lambda_eta=LAMBDA_ETA,
            lambda_slope=LAMBDA_SLOPE,
            lambda_energy=LAMBDA_ENERGY,
            loss_type='cosine',
            check_validity=True,
        )
        elapsed = time.perf_counter() - t0

        # Render final caustic at T_EVAL (single phase, for visualisation)
        X, Y = unpack_complex(jnp.asarray(params), n_act, n_freq)
        P = X + 1j * Y
        a = steady_state_amplitudes(prop, P, Omega, T_EVAL)
        _, _, I_final = caustic_image(prop, a, sigma=SIGMA_RENDER)
        I_final = np.asarray(I_final)
        I_show = I_final / max(I_final.max(), 1e-9)

        cs = cosine_sim(target, I_show)
        print(f"  elapsed: {elapsed:.1f}s   final loss: {history[-1]:.4e}   "
              f"cos(target, render): {cs:.3f}", flush=True)
        results[fname] = {
            "cos": cs, "elapsed_s": elapsed,
            "final_loss": float(history[-1]),
        }

        axes[i, 0].imshow(target, cmap="inferno")
        axes[i, 0].set_title(f"target: {fname}")
        axes[i, 1].imshow(I_show, cmap="inferno")
        axes[i, 1].set_title(f"optimized   cos={cs:.3f}")
        for ax in axes[i]:
            ax.axis("off")

    fig.tight_layout()
    out_png = OUT_DIR / "comparison.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out_png}")

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_DIR / 'results.json'}")

    print(f"\n{'='*60}\n  Summary\n{'='*60}")
    print(f"  {'target':<20} {'cos':>8} {'time (s)':>10}")
    for fname, r in results.items():
        print(f"  {fname:<20} {r['cos']:>8.3f} {r['elapsed_s']:>10.1f}")


if __name__ == "__main__":
    main()
