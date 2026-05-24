"""Physical 3-spot Gaussian: cap-enforced optimization at target-matched depth.

The 3-spot Gaussian has σ=0.04m, so feature scale ~4cm. The throw rule
`d ≥ 30·L_feature` gives d_min ≈ 1.2m; we use d=2m for headroom.

Everything else matches example.ipynb's setup (same apparatus, same target,
same 5-frame temporal window) — but with the linear-regime caps ENFORCED
and the optimization given proper tools (L-BFGS, full Snell, analytical
warm-start) so it can find a good cap-respecting solution.

Reads off the final cos and validity report. The cos number — and the
fact that the validity report shows |η|/d and |∇η| both inside 0.1 —
is the honest answer to "what visual quality is physically achievable
with this target on a tabletop-scale apparatus?"
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
    analytical_solve,
)

# ── Apparatus — match example.ipynb except for depth ────────────────
LX, LY         = 1.0, 1.0
DEPTH          = 2.0           # target-matched: feature scale ~4cm, d ≥ 1.2m
DAMPING        = 0.02
NX, NY         = 200, 200
N_ACT_PER_SIDE = 12
N_MODES        = 15
N_FREQ         = 24
FREQ_MIN_HZ    = 0.5
FREQ_MAX_HZ    = 5.0
T_EVAL         = 1.5
SIGMA_RENDER   = 0.005

# Target — same as example.ipynb cell 28
SIGMA_SPOT = 0.04
SPOTS = [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)]

# Temporal window (same as example.ipynb cell 29)
N_TEMPORAL     = 5
SIGMA_TEMPORAL = 0.04

# Optimization — caps ON, deep_pool-style L-BFGS + full Snell
STAGES = (
    Stage(sigma=0.04, sigma_blur=0.04, iters=500, method='lbfgs'),
    Stage(sigma=0.02, sigma_blur=0.02, iters=500, method='lbfgs'),
    Stage(sigma=0.01, sigma_blur=0.01, iters=500, method='lbfgs'),
)
LR             = 1e-3
LAMBDA_ETA     = 100.0         # ENFORCE linear-wave cap
LAMBDA_SLOPE   = 100.0         # ENFORCE paraxial cap
LAMBDA_ENERGY  = 1e-6
FULL_SNELL     = True

OUT_DIR = Path("data/physical_3spot")


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
    return target / target.max()


def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"JAX {jax.__version__} on {jax.default_backend()}\n", flush=True)

    prop, Omega = build_setup()
    n_act, n_freq = prop.n_act, len(Omega)
    print(f"Apparatus: depth={DEPTH}m (matched to {SIGMA_SPOT*100:.0f}cm features),")
    print(f"           actuators={n_act} ({N_ACT_PER_SIDE}/side), freqs={n_freq}, "
          f"modes={N_MODES}², grid={NX}×{NY}")
    print(f"Optimizer: L-BFGS, full Snell, λ_eta=λ_slope={LAMBDA_ETA} (caps ENFORCED)")
    print(f"Throw budget: ~{0.025*DEPTH*1000:.0f}mm  (feature σ = {SIGMA_SPOT*1000:.0f}mm)\n",
          flush=True)

    target = make_target().astype(np.float32)
    T_array = list(T_EVAL + np.linspace(-SIGMA_TEMPORAL, SIGMA_TEMPORAL, N_TEMPORAL))

    # Analytical warm-start
    ana = analytical_solve(prop, target, np.asarray(Omega), T_EVAL)
    p0 = ana['p0']
    X, Y = unpack_complex(jnp.asarray(p0), n_act, n_freq)
    P = X + 1j * Y
    a_ws = steady_state_amplitudes(prop, P, Omega, T_EVAL)
    _, _, I_ws = caustic_image(prop, a_ws, sigma=SIGMA_RENDER)
    I_ws_show = np.asarray(I_ws) / max(np.asarray(I_ws).max(), 1e-9)
    ws_cos = cosine_sim(target, I_ws_show)
    print(f"Analytical warm-start: ‖p0‖={np.linalg.norm(p0):.3e}  "
          f"cos(target, ws)={ws_cos:.3f}\n", flush=True)

    print("Optimizing (caps enforced)...\n", flush=True)
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
        p0=p0,
        check_validity=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nOptimization elapsed: {elapsed:.1f}s   final loss: {history[-1]:.4e}")

    # Render final
    X, Y = unpack_complex(jnp.asarray(params), n_act, n_freq)
    P = X + 1j * Y
    a = steady_state_amplitudes(prop, P, Omega, T_EVAL)
    _, _, I_final = caustic_image(prop, a, sigma=SIGMA_RENDER, full_snell=FULL_SNELL)
    I_final = np.asarray(I_final)
    I_show = I_final / max(I_final.max(), 1e-9)
    cs = cosine_sim(target, I_show)
    print(f"cos(target, render): {cs:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(target,     cmap="inferno"); axes[0].set_title("target: 3-spot Gaussian")
    axes[1].imshow(I_ws_show,  cmap="inferno"); axes[1].set_title(f"analytical warm-start  cos={ws_cos:.3f}")
    axes[2].imshow(I_show,     cmap="inferno"); axes[2].set_title(f"optimized (caps ON)    cos={cs:.3f}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_png = OUT_DIR / "physical_3spot.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
