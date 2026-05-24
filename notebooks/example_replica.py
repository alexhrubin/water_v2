"""Multi-target replica of example.ipynb optimization runs.

Reproduces example.ipynb's setup across four targets:
  1. 3-spot Gaussian   (cells 28-29)
  2. Sine wave ridge   (cells 33-34)
  3. Recidiviz logo    (cell 55, optimizer from cell 41)
  4. Head photo        (cell 40 with head.jpg, optimizer from cell 41)

The apparatus is shared across all four (cell 6). Per-target optimization
recipes differ slightly — synthetic targets (Gaussian, sine) use the
3-stage Adam recipe from cell 24; image targets use the finer 5-stage
Adam recipe with lr=0.0001 from cell 41.

All other knobs match Julia exactly: random init, raw target (no band-limit),
paraxial Snell, no cap penalties.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from wavetank import (
    Tank, Actuator, build_propagator,
    steady_state_amplitudes, caustic_image, unpack_complex,
    load_target_image,
    Stage, optimize_caustic,
)

# ── Apparatus (cell 6) — shared across all targets ──────────────────
LX, LY         = 1.0, 1.0
DEPTH          = 0.1
DAMPING        = 0.02
NX, NY         = 200, 200
N_ACT_PER_SIDE = 12
N_MODES        = 15
N_FREQ         = 24
FREQ_MIN_HZ    = 0.5
FREQ_MAX_HZ    = 5.0
SIGMA_RENDER   = 0.005

# Optimization knobs that match Julia for ALL runs
LAMBDA_ETA    = 0.0
LAMBDA_SLOPE  = 0.0
LAMBDA_ENERGY = 1e-5
FULL_SNELL    = False

# Two optimization recipes:
# (A) Synthetic targets: 3-stage Adam, lr=1e-3 (cell 24)
STAGES_SYNTHETIC = (
    Stage(sigma=0.04, sigma_blur=0.04, iters=500, method='adam'),
    Stage(sigma=0.02, sigma_blur=0.02, iters=500, method='adam'),
    Stage(sigma=0.01, sigma_blur=0.01, iters=500, method='adam'),
)
LR_SYNTHETIC = 1e-3

# (B) Image targets: 5-stage Adam, lr=1e-4 (cell 41)
STAGES_IMAGE = (
    Stage(sigma=0.05,  sigma_blur=0.05,  iters=100, method='adam'),
    Stage(sigma=0.03,  sigma_blur=0.03,  iters=200, method='adam'),
    Stage(sigma=0.015, sigma_blur=0.015, iters=300, method='adam'),
    Stage(sigma=0.008, sigma_blur=0.008, iters=400, method='adam'),
    Stage(sigma=0.003, sigma_blur=0.003, iters=500, method='adam'),
)
LR_IMAGE = 1e-4

TARGETS_DIR = Path("targets")
OUT_DIR     = Path("data/example_replica")


@dataclass
class TargetConfig:
    name: str
    make: Callable[[np.ndarray, np.ndarray], np.ndarray]
    t_eval: float
    stages: tuple
    lr: float
    n_temporal: int = 1
    sigma_temporal: float = 0.0


# ── Target generators ────────────────────────────────────────────────
def make_3spot_gaussian(xs, ys):
    """3 spots: center + (0.7,0.3) + (0.3,0.7), σ=0.04 (cell 28)."""
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    sigma = 0.04
    spots = [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)]
    target = np.zeros((len(xs), len(ys)))
    for cx, cy in spots:
        target += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    return target / target.max()


def make_sine_wave(xs, ys):
    """Gaussian ridge along y = 0.5 + 0.2·sin(4π x) (cell 33)."""
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    A_wave, f_wave, sigma_wave = 0.2, 2.0, 0.03
    ridge_y = 0.5 + A_wave * np.sin(2 * np.pi * f_wave * X)
    target = np.exp(-(Y - ridge_y) ** 2 / (2 * sigma_wave ** 2))
    return target / target.max()


def make_image_loader(filename):
    def _load(xs, ys):
        target = load_target_image(TARGETS_DIR / filename, xs, ys)
        # Match example.ipynb's `reverse(target, dims=2)` axis flip,
        # adapted to our orientation convention (same as deep_pool_examples).
        target = target[::-1].T
        return target / max(target.max(), 1e-9)
    return _load


# Match example.ipynb cells 28-29, 33-34, 40-41, 55 setups
TARGETS = [
    TargetConfig(
        name="3spot_gaussian",
        make=make_3spot_gaussian,
        t_eval=1.5,
        stages=STAGES_SYNTHETIC, lr=LR_SYNTHETIC,
        n_temporal=5, sigma_temporal=0.04,        # cell 29
    ),
    TargetConfig(
        name="sine_wave",
        make=make_sine_wave,
        t_eval=1.5,
        stages=STAGES_SYNTHETIC, lr=LR_SYNTHETIC,
        n_temporal=1, sigma_temporal=0.0,         # cell 34 has no temporal
    ),
    TargetConfig(
        name="recidiviz_logo",
        make=make_image_loader("recidiviz_logo.jpg"),
        t_eval=1.5,
        stages=STAGES_IMAGE, lr=LR_IMAGE,
        n_temporal=5, sigma_temporal=0.033,       # cell 41
    ),
    TargetConfig(
        name="head",
        make=make_image_loader("head.jpg"),
        t_eval=1.0,
        stages=STAGES_IMAGE, lr=LR_IMAGE,
        n_temporal=5, sigma_temporal=0.033,       # cell 41
    ),
]


# ── Apparatus + helpers ─────────────────────────────────────────────
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


def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def temporal_window(t_eval, n_temporal, sigma_temporal):
    if n_temporal == 1:
        return t_eval
    return list(t_eval + np.linspace(-sigma_temporal, sigma_temporal, n_temporal))


def run_one(prop, Omega, cfg, xs, ys):
    print(f"\n{'='*60}\n  Target: {cfg.name}\n{'='*60}", flush=True)
    target = cfg.make(xs, ys).astype(np.float32)
    n_act, n_freq = prop.n_act, len(Omega)

    T_array = temporal_window(cfg.t_eval, cfg.n_temporal, cfg.sigma_temporal)
    print(f"  T_eval={cfg.t_eval}, temporal n={cfg.n_temporal} σ={cfg.sigma_temporal}", flush=True)
    print(f"  stages: {[(s.sigma, s.iters, s.method) for s in cfg.stages]}", flush=True)
    print(f"  lr={cfg.lr}", flush=True)

    t0 = time.perf_counter()
    params, history = optimize_caustic(
        prop, target, np.asarray(Omega), T_array,
        stages=cfg.stages,
        lr=cfg.lr,
        lambda_eta=LAMBDA_ETA,
        lambda_slope=LAMBDA_SLOPE,
        lambda_energy=LAMBDA_ENERGY,
        loss_type='cosine',
        full_snell=FULL_SNELL,
        p0=None,
        check_validity=True,
    )
    elapsed = time.perf_counter() - t0

    X, Y = unpack_complex(jnp.asarray(params), n_act, n_freq)
    P = X + 1j * Y
    a = steady_state_amplitudes(prop, P, Omega, cfg.t_eval)
    _, _, I_final = caustic_image(prop, a, sigma=SIGMA_RENDER)
    I_final = np.asarray(I_final)
    I_show = I_final / max(I_final.max(), 1e-9)
    cs = cosine_sim(target, I_show)
    print(f"  elapsed: {elapsed:.1f}s   final loss: {history[-1]:.4e}   "
          f"cos: {cs:.3f}", flush=True)

    return target, I_show, cs, elapsed


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"JAX {jax.__version__} on {jax.default_backend()}\n", flush=True)

    prop, Omega = build_setup()
    n_act, n_freq = prop.n_act, len(Omega)
    print(f"Apparatus: depth={DEPTH}m, actuators={n_act}, freqs={n_freq}, "
          f"modes={N_MODES}², grid={NX}×{NY}, phasor DOFs={2 * n_act * n_freq}")
    print(f"Optimizer: Adam, NO caps, RANDOM init, paraxial Snell (matches Julia)\n",
          flush=True)

    xs = np.linspace(0, LX, NX)
    ys = np.linspace(0, LY, NY)

    n = len(TARGETS)
    fig, axes = plt.subplots(n, 2, figsize=(11, 4.5 * n))
    if n == 1:
        axes = axes[None, :]

    summary = []
    for i, cfg in enumerate(TARGETS):
        target, I_show, cs, elapsed = run_one(prop, Omega, cfg, xs, ys)
        axes[i, 0].imshow(target, cmap="inferno"); axes[i, 0].set_title(f"target: {cfg.name}")
        axes[i, 1].imshow(I_show, cmap="inferno"); axes[i, 1].set_title(f"optimized   cos={cs:.3f}")
        for ax in axes[i]:
            ax.axis("off")
        summary.append((cfg.name, cs, elapsed))

    fig.tight_layout()
    out_png = OUT_DIR / "comparison.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out_png}")

    print(f"\n{'='*60}\n  Summary\n{'='*60}")
    print(f"  {'target':<20} {'cos':>8} {'time (s)':>10}")
    for name, cs, elapsed in summary:
        print(f"  {name:<20} {cs:>8.3f} {elapsed:>10.1f}")


if __name__ == "__main__":
    main()
