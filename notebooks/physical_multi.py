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
    steady_state_amplitudes, caustic_image, caustic_image_jacobian,
    reconstruct_surface,
    unpack_complex,
    load_target_image,
    Stage, optimize_caustic, make_hos_forward,
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
    TargetConfig(
        name="dog",
        make=make_image_loader("dog_square.jpg"),
        depth=6.0,                              # broad continuous tone (rule says ~6m)
        t_eval=1.0,
        stages=STAGES_IMAGE,
        n_temporal=5, sigma_temporal=0.033,
    ),
]


def build_setup(depth, n_modes=N_MODES, n_act_per_side=N_ACT_PER_SIDE,
                freq_max=FREQ_MAX_HZ, n_freq=N_FREQ, surface_tension=0.0,
                nx=NX, ny=NY):
    tank = Tank(Lx=LX, Ly=LY, depth=depth, damping=DAMPING,
                surface_tension=surface_tension)
    acts = []
    for i in range(n_act_per_side):
        t = (i + 1) / (n_act_per_side + 1)
        acts += [
            Actuator(x=0.0,        y=t * LY),
            Actuator(x=LX,         y=t * LY),
            Actuator(x=t * LX,     y=0.0),
            Actuator(x=t * LX,     y=LY),
        ]
    prop  = build_propagator(tank, acts, n_modes=n_modes, nx=nx, ny=ny)
    Omega = jnp.asarray([2 * np.pi * f for f in np.linspace(FREQ_MIN_HZ, freq_max, n_freq)])
    return prop, Omega


def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def temporal_window(t_eval, n_temporal, sigma_temporal):
    if n_temporal == 1:
        return t_eval
    return list(t_eval + np.linspace(-sigma_temporal, sigma_temporal, n_temporal))


def run_one(cfg, xs, ys, loss_type='cosine',
            depth_override=None, lambda_caps=LAMBDA_ETA, n_modes=N_MODES,
            n_act_per_side=N_ACT_PER_SIDE, freq_max=FREQ_MAX_HZ, n_freq=N_FREQ,
            hos_M=None, iters_scale=1.0, surface_tension=0.0,
            renderer='splat', nx=NX, ny=NY):
    depth = depth_override if depth_override is not None else cfg.depth
    caps_str = "ON" if lambda_caps > 0 else "OFF"
    forward_label = f"HOS(M={hos_M})" if hos_M else "linear"
    st_label = f", σ/ρ={surface_tension:g}" if surface_tension > 0 else ""
    print(f"\n{'='*60}\n  Target: {cfg.name} (depth={depth}m, modes={n_modes}², "
          f"act={n_act_per_side}/side, freq=[{FREQ_MIN_HZ}-{freq_max}Hz]×{n_freq}, "
          f"loss={loss_type}, caps={caps_str}, forward={forward_label}{st_label})\n{'='*60}",
          flush=True)
    prop, Omega = build_setup(depth, n_modes=n_modes, n_act_per_side=n_act_per_side,
                              freq_max=freq_max, n_freq=n_freq,
                              surface_tension=surface_tension,
                              nx=nx, ny=ny)
    n_act, n_freq = prop.n_act, len(Omega)
    target = cfg.make(xs, ys).astype(np.float32)

    # HOS forward only supports scalar T_eval (no multi-frame averaging).
    # Fall back to single-frame when HOS is on.
    if hos_M is not None and cfg.n_temporal > 1:
        T_array = cfg.t_eval
        print(f"  HOS mode: dropping temporal window (n={cfg.n_temporal}), "
              f"using scalar T_eval={cfg.t_eval}", flush=True)
    else:
        T_array = temporal_window(cfg.t_eval, cfg.n_temporal, cfg.sigma_temporal)

    # Warm-start
    ana = analytical_solve(prop, target, np.asarray(Omega), cfg.t_eval)
    p0 = ana['p0']
    Xw, Yw = unpack_complex(jnp.asarray(p0), n_act, n_freq)
    Pw = Xw + 1j * Yw
    a_ws = steady_state_amplitudes(prop, Pw, Omega, cfg.t_eval)
    if renderer == 'jacobian':
        _, _, I_ws = caustic_image_jacobian(prop, a_ws, full_snell=FULL_SNELL)
    else:
        _, _, I_ws = caustic_image(prop, a_ws, sigma=SIGMA_RENDER, full_snell=FULL_SNELL)
    I_ws = np.asarray(I_ws) / max(np.asarray(I_ws).max(), 1e-9)
    ws_cos = cosine_sim(target, I_ws)
    print(f"  warm-start cos = {ws_cos:.3f}  (throw budget ≈ {25*depth:.0f}mm)", flush=True)

    # Optionally scale iters down for the HOS plumbing test (~50x slower per iter)
    stages_used = cfg.stages
    if iters_scale != 1.0:
        stages_used = tuple(
            Stage(sigma=s.sigma, sigma_blur=s.sigma_blur,
                  iters=max(1, int(s.iters * iters_scale)), method=s.method)
            for s in cfg.stages
        )
        total_iters = sum(s.iters for s in stages_used)
        print(f"  iters scaled by {iters_scale}: {total_iters} total", flush=True)

    # Build HOS forward if requested. Disable linear-regime validity check
    # (its 0.1 threshold is misleading for HOS — M=2 is valid up to ~0.3).
    forward_fn = None
    check_validity = True
    if hos_M is not None:
        forward_fn = make_hos_forward(M=hos_M, dealias_max_modes=n_modes,
                                       steps_per_period=20, initial='steady')
        check_validity = False
        print(f"  using HOS M={hos_M} forward (initial='steady', "
              f"steps_per_period=20)", flush=True)

    t0 = time.perf_counter()
    params, history = optimize_caustic(
        prop, target, np.asarray(Omega), T_array,
        stages=stages_used,
        lr=LR,
        lambda_eta=lambda_caps,
        lambda_slope=lambda_caps,
        lambda_energy=LAMBDA_ENERGY,
        loss_type=loss_type,
        full_snell=FULL_SNELL,
        p0=p0,
        check_validity=check_validity,
        forward_fn=forward_fn,
        renderer=renderer,
    )
    elapsed = time.perf_counter() - t0

    X, Y = unpack_complex(jnp.asarray(params), n_act, n_freq)
    P = X + 1j * Y
    # Render the final result with the SAME forward we optimized against,
    # so what we plot is what the optimizer was actually scoring.
    if hos_M is not None:
        a = make_hos_forward(M=hos_M, dealias_max_modes=n_modes,
                              steps_per_period=20, initial='steady')(
            prop, P, Omega, cfg.t_eval)
    else:
        a = steady_state_amplitudes(prop, P, Omega, cfg.t_eval)
    if renderer == 'jacobian':
        _, _, I_final = caustic_image_jacobian(prop, a, full_snell=FULL_SNELL)
    else:
        _, _, I_final = caustic_image(prop, a, sigma=SIGMA_RENDER, full_snell=FULL_SNELL)
    I_final = np.asarray(I_final)
    I_show = I_final / max(I_final.max(), 1e-9)
    cs = cosine_sim(target, I_show)

    # Always report peak surface stats so we know what regime we ended in.
    # (When check_validity is False — e.g. under HOS — optimize_caustic
    # doesn't print this, but we still want to see it.)
    eta_grid, eta_x, eta_y = reconstruct_surface(prop, a)
    peak_eta = float(jnp.abs(eta_grid).max())
    peak_slope = float(jnp.sqrt(eta_x**2 + eta_y**2).max())
    eta_over_d = peak_eta / depth
    # Validity regime markers
    eta_status = "OK" if eta_over_d < 0.1 else "PAST LINEAR"
    if hos_M is not None:
        slope_status = ("OK (HOS M=2 valid)" if peak_slope < 0.30
                        else "PAST HOS M=2" if peak_slope < 0.45
                        else "PAST WAVE BREAKING")
    else:
        slope_status = "OK" if peak_slope < 0.10 else "PAST LINEAR"
    print(f"  elapsed: {elapsed:.1f}s   final cos: {cs:.3f}", flush=True)
    print(f"  peak |η|/depth = {eta_over_d:.3f}  [{eta_status}]    "
          f"peak |∇η| = {peak_slope:.3f}  [{slope_status}]", flush=True)

    # Apparatus config — JSON-serializable, sufficient for the export script
    # in notebooks/export_3d_animation.py to rebuild the apparatus.
    apparatus_config = dict(
        Lx=LX, depth=depth, damping=DAMPING,
        n_modes=n_modes, n_act_per_side=n_act_per_side,
        actuator_width=0.05,            # built-into Actuator default
        nx=nx, ny=ny,
        n_freq=n_freq, freq_min_hz=FREQ_MIN_HZ, freq_max_hz=freq_max,
        surface_tension=surface_tension,
        t_eval=cfg.t_eval, hos_M=hos_M,
    )

    return dict(
        target=target, I_show=I_show, ws_cos=ws_cos, cs=cs, elapsed=elapsed,
        params=np.asarray(params), eta=np.asarray(eta_grid),
        peak_eta=peak_eta, peak_slope=peak_slope,
        config=apparatus_config,
    )


def main(loss_type='cosine', depth_override=None, no_caps=False,
         n_modes=N_MODES, n_act_per_side=N_ACT_PER_SIDE,
         freq_max=FREQ_MAX_HZ, n_freq=N_FREQ,
         hos_M=None, iters_scale=1.0, targets_filter=None,
         lambda_override=None, surface_tension=0.0,
         renderer='splat', nx=NX, ny=NY):
    if lambda_override is not None:
        lambda_caps = lambda_override
    else:
        lambda_caps = 0.0 if no_caps else LAMBDA_ETA
    caps_tag = ("nocaps" if lambda_caps == 0
                else f"caps{lambda_caps:g}")
    depth_tag = f"d{depth_override}" if depth_override is not None else "dauto"
    parts = [loss_type, depth_tag, caps_tag]
    if n_modes != N_MODES:
        parts.append(f"m{n_modes}")
    if n_act_per_side != N_ACT_PER_SIDE:
        parts.append(f"a{n_act_per_side}")
    if freq_max != FREQ_MAX_HZ or n_freq != N_FREQ:
        parts.append(f"f{freq_max:g}x{n_freq}")
    if hos_M is not None:
        parts.append(f"hosM{hos_M}")
    if iters_scale != 1.0:
        parts.append(f"its{iters_scale:g}")
    if surface_tension > 0:
        parts.append(f"st{surface_tension:g}")
    if renderer != 'splat':
        parts.append(f"r{renderer}")
    if nx != NX or ny != NY:
        parts.append(f"g{nx}x{ny}")
    tag = "_".join(parts)
    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"JAX {jax.__version__} on {jax.default_backend()}", flush=True)
    print(f"Apparatus: actuators={4 * n_act_per_side} ({n_act_per_side}/side), "
          f"freqs={n_freq} [{FREQ_MIN_HZ}-{freq_max}Hz], "
          f"modes={n_modes}², grid={nx}×{ny}")
    print(f"Phasor DOFs: {2 * 4 * n_act_per_side * n_freq}")
    print(f"Optimizer: L-BFGS, full Snell, "
          f"λ_eta=λ_slope={lambda_caps} ({'caps OFF' if no_caps else 'caps ENFORCED'})")
    print(f"Loss: {loss_type}")
    if depth_override is not None:
        print(f"Depth: {depth_override}m (overriding per-target defaults)")
    print(f"Output: {out_dir}\n", flush=True)

    xs = np.linspace(0, LX, nx)
    ys = np.linspace(0, LY, ny)

    targets_to_run = TARGETS
    if targets_filter is not None:
        targets_to_run = [t for t in TARGETS if t.name in targets_filter]
        if not targets_to_run:
            raise ValueError(f"No targets matched filter {targets_filter}; "
                             f"available: {[t.name for t in TARGETS]}")

    n = len(targets_to_run)
    # 3 columns: target | optimized (honest scale) | optimized (contrast-stretched)
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 4.5 * n))
    if n == 1:
        axes = axes[None, :]

    summary = []
    for i, cfg in enumerate(targets_to_run):
        result = run_one(
            cfg, xs, ys, loss_type=loss_type,
            depth_override=depth_override, lambda_caps=lambda_caps,
            n_modes=n_modes, n_act_per_side=n_act_per_side,
            freq_max=freq_max, n_freq=n_freq,
            hos_M=hos_M, iters_scale=iters_scale,
            surface_tension=surface_tension,
            renderer=renderer,
            nx=nx, ny=ny,
        )
        target  = result['target']
        I_show  = result['I_show']
        ws_cos  = result['ws_cos']
        cs      = result['cs']
        elapsed = result['elapsed']
        depth_used = depth_override if depth_override is not None else cfg.depth

        # Save per-target npz so it can be fed into notebooks/export_3d_animation.py.
        # Includes params (steady-state phasors), config (apparatus dict), and
        # the final η on the optimization grid.
        import json
        npz_path = out_dir / f"{cfg.name}.npz"
        np.savez(
            npz_path,
            target=target,
            I_show=I_show,
            eta=result['eta'],
            params=result['params'],
            config=json.dumps(result['config']),
            ws_cos=ws_cos, final_cos=cs,
            peak_eta=result['peak_eta'],
            peak_slope=result['peak_slope'],
            elapsed=elapsed,
        )
        print(f"  Saved {npz_path}", flush=True)

        # Contrast-stretched view: per-image min-max → [0, 1]. Makes tiny
        # contrast variations visible (relevant for cap-on shallow runs
        # where physically-achievable contrast is sub-1%).
        I_stretched = (I_show - I_show.min()) / max(I_show.max() - I_show.min(), 1e-9)
        contrast_pct = 100.0 * (I_show.max() - I_show.min())

        axes[i, 0].imshow(target,      cmap="inferno")
        axes[i, 0].set_title(f"target: {cfg.name}\n(depth={depth_used}m)")
        axes[i, 1].imshow(I_show,      cmap="inferno", vmin=0, vmax=1)
        axes[i, 1].set_title(f"honest scale [0,1]\ncos={cs:.3f}")
        axes[i, 2].imshow(I_stretched, cmap="inferno")
        axes[i, 2].set_title(f"contrast-stretched\n(actual range = {contrast_pct:.1f}% of full)")
        for ax in axes[i]:
            ax.axis("off")
        summary.append((cfg.name, depth_used, ws_cos, cs, elapsed))

    fig.suptitle(f"loss={loss_type}, depth_override={depth_override}, "
                 f"{'NO caps' if no_caps else 'caps ENFORCED'}", fontsize=11)
    fig.tight_layout()
    out_png = out_dir / "comparison.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out_png}")

    print(f"\n{'='*60}\n  Summary ({tag})\n{'='*60}")
    print(f"  {'target':<20} {'depth (m)':>10} {'ws cos':>8} {'final cos':>10} {'time (s)':>10}")
    for name, d, wcos, cs, elapsed in summary:
        print(f"  {name:<20} {d:>10.1f} {wcos:>8.3f} {cs:>10.3f} {elapsed:>10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loss', choices=['cosine', 'pearson', 'ssim'],
                        default='cosine',
                        help="Loss function. 'pearson' is offset-invariant.")
    parser.add_argument('--depth', type=float, default=None,
                        help="Override per-target depth (m) — same depth for all targets.")
    parser.add_argument('--no_caps', action='store_true',
                        help="Disable linearity-cap penalties (matches example.ipynb).")
    parser.add_argument('--n_modes', type=int, default=N_MODES,
                        help=f"Modes per axis (default {N_MODES}). "
                             "Raises curvature budget at fixed slope cap.")
    parser.add_argument('--n_act_per_side', type=int, default=N_ACT_PER_SIDE,
                        help=f"Actuators per side (default {N_ACT_PER_SIDE}). "
                             "Raises rank of actuator coupling matrix.")
    parser.add_argument('--freq_max', type=float, default=FREQ_MAX_HZ,
                        help=f"Max driving frequency Hz (default {FREQ_MAX_HZ}). "
                             "Higher freqs resonantly excite higher-k modes.")
    parser.add_argument('--n_freq', type=int, default=N_FREQ,
                        help=f"Number of driving frequencies (default {N_FREQ}).")
    parser.add_argument('--hos_M', type=int, default=None, choices=[1, 2],
                        help="If set, use HOS forward at this order (1 or 2). "
                             "Disables linear-regime validity warnings. ~50× "
                             "slower per L-BFGS iter than linear forward.")
    parser.add_argument('--iters_scale', type=float, default=1.0,
                        help="Multiply each stage's iter count by this. Use "
                             "e.g. 0.2 for a fast plumbing test (~5x fewer iters).")
    parser.add_argument('--targets', nargs='*', default=None,
                        help="Subset of targets to run (e.g. --targets 3spot_gaussian). "
                             "Default: all four.")
    parser.add_argument('--lambda_slope', type=float, default=None,
                        help=f"Override λ_eta and λ_slope (default {LAMBDA_ETA}, "
                             "0 with --no_caps). Use ~10 for HOS-friendly soft cap.")
    parser.add_argument('--surface_tension', type=float, default=0.0,
                        help="Surface tension σ/ρ in m³/s² (default 0.0 = pure gravity). "
                             "For water at 20°C use 7.28e-5. Adds capillary contribution "
                             "to the dispersion ω² = (gk + σ/ρ·k³)·tanh(kd); matters at "
                             "λ ≲ 1.7 cm (shallow / high-mode regimes).")
    parser.add_argument('--renderer', choices=['splat', 'jacobian'], default='splat',
                        help="Caustic renderer used inside the optimization loop. "
                             "'splat' = bilinear-splat + Gaussian blur (default). "
                             "'jacobian' = Wallace-style anisotropic Gaussian splat "
                             "with per-source-point covariance from the local "
                             "Jacobian of the refraction map.")
    parser.add_argument('--nx', type=int, default=NX,
                        help=f"Source/floor grid x-resolution (default {NX}). "
                             "Higher = finer caustic detail in our renderer "
                             "(closer to Wallace's 1024×1024 caustic texture), "
                             "at O(nx²) optimization cost.")
    parser.add_argument('--ny', type=int, default=NY,
                        help=f"Source/floor grid y-resolution (default {NY}).")
    args = parser.parse_args()
    main(loss_type=args.loss, depth_override=args.depth, no_caps=args.no_caps,
         n_modes=args.n_modes, n_act_per_side=args.n_act_per_side,
         freq_max=args.freq_max, n_freq=args.n_freq,
         hos_M=args.hos_M, iters_scale=args.iters_scale,
         targets_filter=args.targets, lambda_override=args.lambda_slope,
         surface_tension=args.surface_tension,
         renderer=args.renderer,
         nx=args.nx, ny=args.ny)
