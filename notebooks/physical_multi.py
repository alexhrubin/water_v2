"""Physical multi-target: cap-enforced optimization with target-matched depth.

Same four targets as example_replica.py (3-spot Gaussian, sine wave, logo, head),
but with caps ENFORCED and depth chosen per-target via the rule
`d ≥ 30 · L_feature_min`.

This is the honest physical-regime answer to "what visual quality is achievable
with a real apparatus on these targets?" Compare against example_replica.py
(no caps, cos 0.90-0.97) to see the cost of staying physical.

Apparatus shared (matches example.ipynb except for depth): 48 actuators, 24 freqs,
15 modes/axis, 200×200 grid. Per-target: depth, optimization recipe.
"""

import argparse
import time
from dataclasses import dataclass
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
    analytical_solve,
)

# ── Apparatus shared across runs (matches example.ipynb cell 6) ─────
LX, LY         = 1.0, 1.0
DAMPING        = 0.02
NX, NY         = 200, 200
N_ACT_PER_SIDE = 12
N_MODES        = 15
N_FREQ         = 24
FREQ_MIN_HZ    = 0.5
FREQ_MAX_HZ    = 5.0
SIGMA_RENDER   = 0.005

# Cap enforcement + optimizer settings (deep_pool-style)
LAMBDA_ETA     = 100.0
LAMBDA_SLOPE   = 100.0
LAMBDA_ENERGY  = 1e-6
FULL_SNELL     = True

# Per-target stages — synthetic targets use 3-stage, image targets use 5-stage
STAGES_SYNTHETIC = (
    Stage(sigma=0.04, sigma_blur=0.04, iters=500, method='lbfgs'),
    Stage(sigma=0.02, sigma_blur=0.02, iters=500, method='lbfgs'),
    Stage(sigma=0.01, sigma_blur=0.01, iters=500, method='lbfgs'),
)
STAGES_IMAGE = (
    Stage(sigma=0.05,  sigma_blur=0.05,  iters=200, method='lbfgs'),
    Stage(sigma=0.03,  sigma_blur=0.03,  iters=300, method='lbfgs'),
    Stage(sigma=0.015, sigma_blur=0.015, iters=400, method='lbfgs'),
    Stage(sigma=0.008, sigma_blur=0.008, iters=400, method='lbfgs'),
    Stage(sigma=0.003, sigma_blur=0.003, iters=200, method='lbfgs'),
)
LR = 1e-3  # L-BFGS uses its own line search; lr is unused

TARGETS_DIR = Path("targets")
OUT_DIR     = Path("data/physical_multi")


@dataclass
class TargetConfig:
    name: str
    make: Callable[[np.ndarray, np.ndarray], np.ndarray]
    depth: float
    t_eval: float
    stages: tuple
    n_temporal: int = 1
    sigma_temporal: float = 0.0


def make_3spot_gaussian(xs, ys):
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    sigma = 0.04
    target = np.zeros((len(xs), len(ys)))
    for cx, cy in [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)]:
        target += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    return target / target.max()


def make_sine_wave(xs, ys):
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    A_wave, f_wave, sigma_wave = 0.2, 2.0, 0.03
    ridge_y = 0.5 + A_wave * np.sin(2 * np.pi * f_wave * X)
    target = np.exp(-(Y - ridge_y) ** 2 / (2 * sigma_wave ** 2))
    return target / target.max()


def make_image_loader(filename):
    def _load(xs, ys):
        target = load_target_image(TARGETS_DIR / filename, xs, ys)
        target = target[::-1].T
        return target / max(target.max(), 1e-9)
    return _load


# Per-target depth: rule of thumb d ≥ 30 · L_feature_min
# (3-spot σ=4cm → 1.2m min; sine ridge σ=3cm → 0.9m min; image features ~5cm → 1.5m min)
# Use generous headroom — fixing depth at the rule-of-thumb minimum leaves no
# slope budget for high-mode detail.
TARGETS = [
    TargetConfig(
        name="3spot_gaussian",
        make=make_3spot_gaussian,
        depth=2.0,                              # σ=4cm → d_min=1.2m, use 2m
        t_eval=1.5,
        stages=STAGES_SYNTHETIC,
        n_temporal=5, sigma_temporal=0.04,
    ),
    TargetConfig(
        name="sine_wave",
        make=make_sine_wave,
        depth=2.0,                              # ridge σ=3cm → d_min=0.9m, use 2m
        t_eval=1.5,
        stages=STAGES_SYNTHETIC,
        n_temporal=1, sigma_temporal=0.0,
    ),
    TargetConfig(
        name="recidiviz_logo",
        make=make_image_loader("recidiviz_logo.jpg"),
        depth=5.0,                              # image features ~5cm, use 5m
        t_eval=1.5,
        stages=STAGES_IMAGE,
        n_temporal=5, sigma_temporal=0.033,
    ),
    TargetConfig(
        name="head",
        make=make_image_loader("head.jpg"),
        depth=5.0,                              # facial detail ~5cm, use 5m (deep_pool used 5m)
        t_eval=1.0,
        stages=STAGES_IMAGE,
        n_temporal=5, sigma_temporal=0.033,
    ),
]


def build_setup(depth):
    tank = Tank(Lx=LX, Ly=LY, depth=depth, damping=DAMPING)
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


def run_one(cfg, xs, ys, loss_type='cosine'):
    print(f"\n{'='*60}\n  Target: {cfg.name} (depth={cfg.depth}m, loss={loss_type})\n{'='*60}", flush=True)
    prop, Omega = build_setup(cfg.depth)
    n_act, n_freq = prop.n_act, len(Omega)
    target = cfg.make(xs, ys).astype(np.float32)

    T_array = temporal_window(cfg.t_eval, cfg.n_temporal, cfg.sigma_temporal)

    # Warm-start
    ana = analytical_solve(prop, target, np.asarray(Omega), cfg.t_eval)
    p0 = ana['p0']
    Xw, Yw = unpack_complex(jnp.asarray(p0), n_act, n_freq)
    Pw = Xw + 1j * Yw
    a_ws = steady_state_amplitudes(prop, Pw, Omega, cfg.t_eval)
    _, _, I_ws = caustic_image(prop, a_ws, sigma=SIGMA_RENDER, full_snell=FULL_SNELL)
    I_ws = np.asarray(I_ws) / max(np.asarray(I_ws).max(), 1e-9)
    ws_cos = cosine_sim(target, I_ws)
    print(f"  warm-start cos = {ws_cos:.3f}  (throw budget ≈ {25*cfg.depth:.0f}mm)", flush=True)

    t0 = time.perf_counter()
    params, history = optimize_caustic(
        prop, target, np.asarray(Omega), T_array,
        stages=cfg.stages,
        lr=LR,
        lambda_eta=LAMBDA_ETA,
        lambda_slope=LAMBDA_SLOPE,
        lambda_energy=LAMBDA_ENERGY,
        loss_type=loss_type,
        full_snell=FULL_SNELL,
        p0=p0,
        check_validity=True,
    )
    elapsed = time.perf_counter() - t0

    X, Y = unpack_complex(jnp.asarray(params), n_act, n_freq)
    P = X + 1j * Y
    a = steady_state_amplitudes(prop, P, Omega, cfg.t_eval)
    _, _, I_final = caustic_image(prop, a, sigma=SIGMA_RENDER, full_snell=FULL_SNELL)
    I_final = np.asarray(I_final)
    I_show = I_final / max(I_final.max(), 1e-9)
    cs = cosine_sim(target, I_show)
    print(f"  elapsed: {elapsed:.1f}s   final cos: {cs:.3f}", flush=True)

    return target, I_show, ws_cos, cs, elapsed


def main(loss_type='cosine'):
    out_dir = OUT_DIR / loss_type
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"JAX {jax.__version__} on {jax.default_backend()}", flush=True)
    print(f"Apparatus (per target): actuators={4 * N_ACT_PER_SIDE}, freqs={N_FREQ}, "
          f"modes={N_MODES}², grid={NX}×{NY}")
    print(f"Optimizer: L-BFGS, full Snell, λ_eta=λ_slope={LAMBDA_ETA} (caps ENFORCED)")
    print(f"Loss: {loss_type}\n", flush=True)

    xs = np.linspace(0, LX, NX)
    ys = np.linspace(0, LY, NY)

    n = len(TARGETS)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.5 * n))
    if n == 1:
        axes = axes[None, :]

    summary = []
    for i, cfg in enumerate(TARGETS):
        target, I_show, ws_cos, cs, elapsed = run_one(cfg, xs, ys, loss_type=loss_type)
        axes[i, 0].imshow(target,  cmap="inferno")
        axes[i, 0].set_title(f"target: {cfg.name}\n(depth={cfg.depth}m)")
        axes[i, 1].imshow(I_show,  cmap="inferno")
        axes[i, 1].set_title(f"optimized (loss={loss_type})   cos={cs:.3f}")
        for ax in axes[i]:
            ax.axis("off")
        summary.append((cfg.name, cfg.depth, ws_cos, cs, elapsed))

    fig.tight_layout()
    out_png = out_dir / "comparison.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out_png}")

    print(f"\n{'='*60}\n  Summary (loss={loss_type}, caps enforced)\n{'='*60}")
    print(f"  {'target':<20} {'depth (m)':>10} {'ws cos':>8} {'final cos':>10} {'time (s)':>10}")
    for name, d, wcos, cs, elapsed in summary:
        print(f"  {name:<20} {d:>10.1f} {wcos:>8.3f} {cs:>10.3f} {elapsed:>10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loss', choices=['cosine', 'pearson', 'ssim'],
                        default='cosine',
                        help="Loss function. 'pearson' is offset-invariant, "
                             "matches the achievable-contrast story.")
    args = parser.parse_args()
    main(loss_type=args.loss)
