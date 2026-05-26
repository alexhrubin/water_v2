"""Transient HOS optimization driver.

Time-varying piecewise-constant actuator drive, integrated forward from
rest under HOS M=2, with the caustic at T_eval matched to a target.

Pipeline:
    1. Build apparatus (tank, actuators, propagator).
    2. Run steady-state analytical solve + (optional) Adam steady polish
       to obtain a phasor solution P. This is the baseline AND the warm
       start for the transient run.
    3. Convert P → theta[n_act, n_bins] at bin centers.
    4. Optimize theta via Adam through ``hos_forward_transient`` with
       sigma-annealed stages.
    5. Compare: steady-state caustic vs. transient caustic vs. target.

Designed for Colab A100. Local dev uses small basis (n_modes ≤ 12,
n_bins ≤ 20) to fit in CPU memory.

Usage:
    python notebooks/optimize_transient.py --target targets/ANNA.jpg
    python notebooks/optimize_transient.py --target targets/dog_square.jpg \\
        --depth 0.5 --n_bins 30 --T_eval 1.5 --iters 800
"""

import argparse
import time
import json
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from wavetank import (
    Tank, Actuator, build_propagator,
    steady_state_amplitudes, caustic_image, reconstruct_surface,
    unpack_complex,
    analytical_solve,
    Stage, optimize_caustic, make_hos_forward,
    hos_forward_transient, warm_start_from_steady,
    optimize_transient, TransientStage,
    HOSConfig,
)


def make_apparatus(args):
    """Build tank, actuators, propagator."""
    tank = Tank(Lx=args.Lx, Ly=args.Lx, depth=args.depth, damping=args.damping)
    positions = []
    for t in np.linspace(0, 1, args.n_act_per_side + 2)[1:-1]:
        positions += [(t*args.Lx, 0), (t*args.Lx, args.Lx),
                      (0, t*args.Lx), (args.Lx, t*args.Lx)]
    acts = [Actuator(x, y, width=args.actuator_width) for x, y in positions]
    prop = build_propagator(tank, acts,
                             n_modes=args.n_modes, nx=args.nx, ny=args.ny)
    print(f"Apparatus: L={args.Lx}m, d={args.depth}m, throw={tank.throw}m, "
          f"n_modes={args.n_modes}, n_act={prop.n_act} "
          f"({args.n_act_per_side}/side), grid={args.nx}×{args.ny}")
    return tank, prop


def load_target(path, nx, ny):
    """Load and resize a target image (or .npy) to (nx, ny), values in [0, 1]."""
    from PIL import Image
    p = Path(path)
    if p.suffix == '.npy':
        target = np.load(p).astype(np.float32)
        if target.shape != (nx, ny):
            img = Image.fromarray((np.clip(target, 0, 1) * 255).astype(np.uint8))
            img = img.resize((ny, nx), Image.BICUBIC)
            target = np.array(img, dtype=np.float32) / 255.0
    else:
        img = Image.open(p).convert('L')
        img = img.resize((ny, nx), Image.BICUBIC)              # PIL: (width, height)
        target = np.array(img, dtype=np.float32) / 255.0
    return np.clip(target.astype(np.float64), 0.0, 1.0)


def cos_sim(I, T):
    I = np.asarray(I).flatten(); T = np.asarray(T).flatten()
    Ic = I - I.mean(); Tc = T - T.mean()
    return float(np.dot(Ic, Tc) / (np.linalg.norm(Ic) * np.linalg.norm(Tc) + 1e-12))


def run_steady_baseline(prop, target, Omega_freqs, args):
    """Steady-state pipeline: analytical solve + (optional) Adam polish.

    Returns the steady-state phasor matrix P [n_act, n_freq] and the
    rendered steady-state caustic for comparison.
    """
    print("\n── Steady-state baseline ──")
    t0 = time.time()
    out = analytical_solve(
        prop, target, Omega_freqs, T_eval=args.T_eval_steady,
        n_water=args.n_water,
    )
    p0 = out['p0']
    print(f"  analytical_solve done ({time.time()-t0:.1f}s), ‖p0‖={np.linalg.norm(p0):.3e}")

    if args.steady_iters > 0:
        print(f"  Adam polish: {args.steady_iters} iters")
        stages = (Stage(sigma=args.sigma_render, sigma_blur=args.sigma_render,
                        iters=args.steady_iters),)
        hos_fwd = make_hos_forward(M=args.M, dealias_max_modes=args.dealias_max_modes,
                                   steps_per_period=args.steps_per_period,
                                   initial="steady")
        params, _ = optimize_caustic(
            prop, target, Omega_freqs, args.T_eval_steady,
            stages=stages, lr=args.steady_lr,
            lambda_eta=args.lambda_eta, lambda_slope=args.lambda_slope,
            lambda_energy=args.lambda_energy,
            loss_type=args.loss, full_snell=True, p0=p0,
            forward_fn=hos_fwd, check_validity=False,
        )
    else:
        params = p0

    X, Y = unpack_complex(jnp.asarray(params), prop.n_act, len(Omega_freqs))
    P = X + 1j * Y

    # Render steady-state result (using HOS for honest comparison)
    cfg = HOSConfig(M=args.M, dealias_max_modes=args.dealias_max_modes,
                    steps_per_period=args.steps_per_period)
    from wavetank.hos import hos_forward as hos_fwd_fn
    a_steady = hos_fwd_fn(prop, P, jnp.asarray(Omega_freqs),
                           T_eval=args.T_eval_steady,
                           config=cfg, initial="steady")
    _, _, I_steady = caustic_image(prop, a_steady, sigma=args.sigma_render,
                                    full_snell=True, n_water=args.n_water)
    eta_steady, dx_s, dy_s = reconstruct_surface(prop, a_steady)
    slope_steady = float(jnp.max(jnp.sqrt(dx_s**2 + dy_s**2)))

    return P, np.asarray(a_steady), np.asarray(I_steady), np.asarray(eta_steady), slope_steady


def run_transient(prop, target, P_warm, Omega_freqs, args):
    """Run the transient optimization."""
    dt_bin = args.T_eval / args.n_bins
    print(f"\n── Transient HOS optimization ──")
    print(f"  dt_bin = {dt_bin:.4f}s, n_bins = {args.n_bins}, T_eval = {args.T_eval}s")
    print(f"  Parameter count: {prop.n_act * args.n_bins}")

    theta0 = warm_start_from_steady(P_warm, Omega_freqs, dt_bin, args.n_bins)
    print(f"  Warm-start ‖θ0‖ = {float(jnp.linalg.norm(theta0)):.3e}")

    cfg = HOSConfig(M=args.M, dealias_max_modes=args.dealias_max_modes,
                    steps_per_period=args.steps_per_period)

    stages = (
        TransientStage(sigma=args.sigma_render*2, sigma_blur=args.sigma_render*2,
                       iters=args.iters // 3, lr=args.lr),
        TransientStage(sigma=args.sigma_render*1.4, sigma_blur=args.sigma_render*1.4,
                       iters=args.iters // 3, lr=args.lr*0.5),
        TransientStage(sigma=args.sigma_render, sigma_blur=args.sigma_render,
                       iters=args.iters // 3, lr=args.lr*0.2),
    )

    theta_opt, loss_history = optimize_transient(
        prop, target, dt_bin, args.n_bins,
        M=args.M, config=cfg,
        stages=stages, theta0=theta0,
        lambda_eta=args.lambda_eta,
        lambda_slope=args.lambda_slope,
        lambda_energy=args.lambda_energy,
        loss_type=args.loss,
        n_water=args.n_water, full_snell=True,
    )

    # Final render at T_eval
    theta_j = jnp.asarray(theta_opt)
    a_trans = hos_forward_transient(prop, theta_j, dt_bin, M=args.M, config=cfg)
    _, _, I_trans = caustic_image(prop, a_trans, sigma=args.sigma_render,
                                   full_snell=True, n_water=args.n_water)
    eta_trans, dx_t, dy_t = reconstruct_surface(prop, a_trans)
    slope_trans = float(jnp.max(jnp.sqrt(dx_t**2 + dy_t**2)))

    return (theta_opt, loss_history,
            np.asarray(a_trans), np.asarray(I_trans),
            np.asarray(eta_trans), slope_trans)


def make_comparison_figure(target, I_steady, I_trans, eta_steady, eta_trans,
                            loss_history, args, c_steady, c_trans,
                            slope_steady, slope_trans, out_path):
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.7])

    # Row 1: target | steady caustic | transient caustic | loss curve
    ax_t = fig.add_subplot(gs[0, 0])
    ax_t.imshow(target, cmap='gray', origin='lower', vmin=0, vmax=1)
    ax_t.set_title("Target")
    ax_t.axis('off')

    Lx = args.Lx
    extent = [0, Lx, 0, Lx]
    vmax = max(I_steady.max(), I_trans.max())
    ax_s = fig.add_subplot(gs[0, 1])
    ax_s.imshow(I_steady.T, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=vmax)
    ax_s.set_title(f"Steady HOS M={args.M}\ncos={c_steady:.3f}, slope={slope_steady:.3f}")
    ax_s.axis('off')

    ax_r = fig.add_subplot(gs[0, 2])
    ax_r.imshow(I_trans.T, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=vmax)
    ax_r.set_title(f"Transient HOS M={args.M}\ncos={c_trans:.3f}, slope={slope_trans:.3f}")
    ax_r.axis('off')

    ax_l = fig.add_subplot(gs[0, 3])
    ax_l.plot(loss_history)
    ax_l.set_xlabel("iter"); ax_l.set_ylabel("loss")
    ax_l.set_title(f"Loss (transient)\nfinal={loss_history[-1]:.4f}")

    # Row 2: surfaces η + actuator drive heatmap
    eta_lim = max(np.abs(eta_steady).max(), np.abs(eta_trans).max())
    ax_es = fig.add_subplot(gs[1, 0])
    ax_es.imshow(eta_steady.T, cmap='RdBu_r', origin='lower', extent=extent,
                  vmin=-eta_lim, vmax=eta_lim)
    ax_es.set_title("η steady")
    ax_es.axis('off')

    ax_et = fig.add_subplot(gs[1, 1])
    ax_et.imshow(eta_trans.T, cmap='RdBu_r', origin='lower', extent=extent,
                  vmin=-eta_lim, vmax=eta_lim)
    ax_et.set_title("η transient (t=T_eval)")
    ax_et.axis('off')

    fig.suptitle(
        f"Transient HOS optimization: target={Path(args.target).stem}, "
        f"d={args.depth}m, n_bins={args.n_bins}, T_eval={args.T_eval}s, M={args.M}",
        fontsize=11, y=0.99,
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--Lx', type=float, default=1.0)
    parser.add_argument('--depth', type=float, default=2.0)
    parser.add_argument('--damping', type=float, default=0.02)
    parser.add_argument('--n_modes', type=int, default=15)
    parser.add_argument('--n_act_per_side', type=int, default=12)
    parser.add_argument('--actuator_width', type=float, default=0.05)
    parser.add_argument('--nx', type=int, default=200)
    parser.add_argument('--ny', type=int, default=200)
    parser.add_argument('--n_freq', type=int, default=8)
    parser.add_argument('--freq_min_hz', type=float, default=0.5)
    parser.add_argument('--freq_max_hz', type=float, default=5.0)
    parser.add_argument('--n_water', type=float, default=1.33)
    # Transient-specific
    parser.add_argument('--T_eval', type=float, default=1.0,
                        help="Total simulation time (s)")
    parser.add_argument('--n_bins', type=int, default=20)
    parser.add_argument('--M', type=int, default=2)
    parser.add_argument('--dealias_max_modes', type=int, default=None)
    parser.add_argument('--steps_per_period', type=int, default=20)
    parser.add_argument('--iters', type=int, default=600)
    parser.add_argument('--lr', type=float, default=1e-3)
    # Steady baseline
    parser.add_argument('--T_eval_steady', type=float, default=1.0)
    parser.add_argument('--steady_iters', type=int, default=200,
                        help="Polish steps on the steady-state baseline (0 = skip polish)")
    parser.add_argument('--steady_lr', type=float, default=1e-3)
    # Loss + caps
    parser.add_argument('--loss', type=str, default='cosine', choices=['cosine', 'pearson'])
    parser.add_argument('--lambda_eta', type=float, default=100.0)
    parser.add_argument('--lambda_slope', type=float, default=100.0)
    parser.add_argument('--lambda_energy', type=float, default=1e-5)
    parser.add_argument('--sigma_render', type=float, default=0.01)
    # Output
    parser.add_argument('--out_dir', type=str, default='notebooks/transient_out')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.target).stem

    tank, prop = make_apparatus(args)
    target = load_target(args.target, args.nx, args.ny)
    print(f"Target: {args.target}  (mean={target.mean():.3f})")

    freqs = np.linspace(args.freq_min_hz, args.freq_max_hz, args.n_freq)
    Omega_freqs = 2 * np.pi * freqs

    # Baseline
    P, a_steady, I_steady, eta_steady, slope_steady = run_steady_baseline(
        prop, target, Omega_freqs, args
    )
    c_steady = cos_sim(I_steady, target)
    print(f"\nSteady-state cos = {c_steady:.4f}, peak slope = {slope_steady:.3f}")

    # Transient
    t0 = time.time()
    theta_opt, loss_history, a_trans, I_trans, eta_trans, slope_trans = run_transient(
        prop, target, P, Omega_freqs, args
    )
    elapsed = time.time() - t0
    c_trans = cos_sim(I_trans, target)
    print(f"\nTransient cos = {c_trans:.4f}, peak slope = {slope_trans:.3f} "
          f"({elapsed:.1f}s wall)")

    # Save artifacts
    np.savez(out_dir / f"{stem}_transient.npz",
             theta=theta_opt, loss=np.asarray(loss_history),
             a_steady=a_steady, a_trans=a_trans,
             I_steady=I_steady, I_trans=I_trans,
             eta_steady=eta_steady, eta_trans=eta_trans,
             c_steady=c_steady, c_trans=c_trans,
             slope_steady=slope_steady, slope_trans=slope_trans,
             config=json.dumps(vars(args)))

    make_comparison_figure(
        target, I_steady, I_trans, eta_steady, eta_trans,
        loss_history, args,
        c_steady, c_trans, slope_steady, slope_trans,
        out_dir / f"{stem}_transient.png",
    )

    print(f"\n── Summary ──")
    print(f"  Steady-state cos:  {c_steady:.4f}  (slope={slope_steady:.3f})")
    print(f"  Transient cos:     {c_trans:.4f}  (slope={slope_trans:.3f})")
    print(f"  Δcos = {c_trans - c_steady:+.4f}")


if __name__ == "__main__":
    main()
