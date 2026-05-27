"""Shallow-tank diagnostic: optimize an arbitrary .npy/.png target at chosen depth.

Tests the hypothesis that the shallow-tank optimization works fine when given
targets matched to the apparatus's natural output class (concentric arcs, fine
lines, sparse dots) and only fails on broad-blob targets (3-spot Gaussian)
because those are unreachable per the flux-transport geometric bound.

Pipeline:
    1. Load target .npy or image file, resample to grid.
    2. Build apparatus (defaults: shallow d=0.1m, n_modes=30, K=20/side).
    3. Analytical Poisson warm start.
    4. Sigma-annealed L-BFGS optimization with full Snell rendering.
    5. Save comparison figure: target | warm start | optimized | stretched.

Usage:
    python notebooks/generate_pulse_targets.py     # first, generate targets
    python notebooks/optimize_shallow_target.py --target targets/ring_arcs.npy
    python notebooks/optimize_shallow_target.py --target targets/dots_on_ring.npy \\
        --depth 0.1 --n_modes 30 --surface_tension 7.28e-5
"""

import argparse
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from wavetank import (
    Tank, Actuator, build_propagator,
    caustic_image, reconstruct_surface,
    unpack_complex,
    analytical_solve,
    Stage, optimize_caustic, make_hos_forward,
)


def load_target(path, nx, ny):
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
        img = img.resize((ny, nx), Image.BICUBIC)
        target = np.array(img, dtype=np.float32) / 255.0
    return np.clip(target.astype(np.float64), 0.0, 1.0)


def cos_sim(I, T):
    I = np.asarray(I).flatten(); T = np.asarray(T).flatten()
    Ic = I - I.mean(); Tc = T - T.mean()
    return float(np.dot(Ic, Tc) / (np.linalg.norm(Ic) * np.linalg.norm(Tc) + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--depth', type=float, default=0.1)
    parser.add_argument('--Lx', type=float, default=1.0)
    parser.add_argument('--damping', type=float, default=0.02)
    parser.add_argument('--n_modes', type=int, default=30)
    parser.add_argument('--n_act_per_side', type=int, default=20)
    parser.add_argument('--actuator_width', type=float, default=0.005,
                        help="Small actuators (5mm) — shallow regime needs sub-cm footprint")
    parser.add_argument('--nx', type=int, default=200)
    parser.add_argument('--ny', type=int, default=200)
    parser.add_argument('--n_freq', type=int, default=12)
    parser.add_argument('--freq_min_hz', type=float, default=1.0)
    parser.add_argument('--freq_max_hz', type=float, default=15.0,
                        help="Shallow tanks need higher drive (default 15 Hz vs the "
                             "5 Hz used at d=2m). Scale up with n_modes/depth ratio.")
    parser.add_argument('--surface_tension', type=float, default=0.0,
                        help="σ/ρ (m³/s²). Use 7.28e-5 for water at 20°C; matters at "
                             "λ ≲ 1.7cm which is exactly this regime.")
    parser.add_argument('--T_eval', type=float, default=1.0)
    parser.add_argument('--hos_M', type=int, default=None, choices=[1, 2],
                        help="If set, optimize through HOS forward. Default linear.")
    parser.add_argument('--n_water', type=float, default=1.33)
    parser.add_argument('--loss', type=str, default='cosine', choices=['cosine', 'pearson'])
    parser.add_argument('--lambda_eta', type=float, default=100.0)
    parser.add_argument('--lambda_slope', type=float, default=100.0)
    parser.add_argument('--iters_scale', type=float, default=1.0)
    parser.add_argument('--out_dir', type=str, default='notebooks/shallow_diag')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.target).stem

    # ── Apparatus ────────────────────────────────────────────────────────
    tank = Tank(Lx=args.Lx, Ly=args.Lx, depth=args.depth, damping=args.damping,
                surface_tension=args.surface_tension)
    positions = []
    for t in np.linspace(0, 1, args.n_act_per_side + 2)[1:-1]:
        positions += [(t*args.Lx, 0), (t*args.Lx, args.Lx),
                      (0, t*args.Lx), (args.Lx, t*args.Lx)]
    acts = [Actuator(x, y, width=args.actuator_width) for x, y in positions]
    prop = build_propagator(tank, acts, n_modes=args.n_modes, nx=args.nx, ny=args.ny)
    print(f"Apparatus: L={args.Lx}m, d={args.depth}m, throw={tank.throw}m, "
          f"σ/ρ={args.surface_tension:.2e}")
    print(f"  n_modes={args.n_modes}, n_act={prop.n_act} ({args.n_act_per_side}/side), "
          f"σ_act={args.actuator_width}m, grid={args.nx}×{args.ny}")

    freqs = np.linspace(args.freq_min_hz, args.freq_max_hz, args.n_freq)
    Omega = np.array([2 * np.pi * f for f in freqs])
    print(f"  Drive: {args.n_freq} freqs in [{args.freq_min_hz}, {args.freq_max_hz}] Hz")

    target = load_target(args.target, args.nx, args.ny)
    print(f"Target: {args.target} (range [{target.min():.3f}, {target.max():.3f}])")

    # ── Analytical warm start ────────────────────────────────────────────
    t0 = time.time()
    out = analytical_solve(prop, target, Omega, T_eval=args.T_eval,
                            n_water=args.n_water)
    p0 = out['p0']
    print(f"  analytical_solve: {time.time()-t0:.1f}s, ‖p0‖={np.linalg.norm(p0):.3e}")

    # Warm-start render
    X, Y = unpack_complex(jnp.asarray(p0), prop.n_act, args.n_freq)
    P_ws = X + 1j * Y
    from wavetank import steady_state_amplitudes
    a_ws = steady_state_amplitudes(prop, P_ws, jnp.asarray(Omega), args.T_eval)
    _, _, I_ws = caustic_image(prop, a_ws, sigma=0.005, full_snell=True,
                                n_water=args.n_water)
    ws_cos = cos_sim(I_ws, target)
    print(f"  warm-start cos = {ws_cos:.4f}")

    # ── Optimize ─────────────────────────────────────────────────────────
    forward_fn = None
    if args.hos_M is not None:
        forward_fn = make_hos_forward(M=args.hos_M, steps_per_period=20,
                                       initial="steady")
        print(f"  Forward: HOS M={args.hos_M}")
    else:
        print("  Forward: linear")

    base_iters = [200, 300, 400, 400, 200]
    sigmas = [0.05, 0.03, 0.015, 0.008, 0.003]
    stages = tuple(
        Stage(sigma=s, sigma_blur=s,
              iters=max(1, int(it * args.iters_scale)),
              method='lbfgs')
        for s, it in zip(sigmas, base_iters)
    )

    t0 = time.time()
    params, _ = optimize_caustic(
        prop, target, Omega, args.T_eval,
        stages=stages, lr=1e-3,
        lambda_eta=args.lambda_eta, lambda_slope=args.lambda_slope,
        lambda_energy=1e-6,
        loss_type=args.loss, full_snell=True, n_water=args.n_water,
        p0=p0, forward_fn=forward_fn, check_validity=False,
    )
    elapsed = time.time() - t0

    # Final render (always with full Snell)
    X, Y = unpack_complex(jnp.asarray(params), prop.n_act, args.n_freq)
    P = X + 1j * Y
    if forward_fn is not None:
        a_final = forward_fn(prop, P, jnp.asarray(Omega), args.T_eval)
    else:
        a_final = steady_state_amplitudes(prop, P, jnp.asarray(Omega), args.T_eval)
    _, _, I_final = caustic_image(prop, a_final, sigma=0.005, full_snell=True,
                                   n_water=args.n_water)
    eta, dx_eta, dy_eta = reconstruct_surface(prop, a_final)
    slope_peak = float(jnp.max(jnp.sqrt(dx_eta**2 + dy_eta**2)))
    eta_peak = float(jnp.max(jnp.abs(eta)))

    final_cos = cos_sim(I_final, target)
    print(f"\n  optimized cos = {final_cos:.4f} ({elapsed:.1f}s)")
    print(f"  peak |η| = {eta_peak:.3e} (|η|/d = {eta_peak/args.depth:.3f})")
    print(f"  peak slope = {slope_peak:.3f}")

    # Contrast-stretched view of final caustic
    I_final_arr = np.asarray(I_final)
    I_stretched = (I_final_arr - I_final_arr.min()) / max(
        I_final_arr.max() - I_final_arr.min(), 1e-9
    )
    contrast_pct = 100.0 * (I_final_arr.max() - I_final_arr.min())

    # ── Figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(target.T, origin='lower', extent=[0, args.Lx, 0, args.Lx],
                    cmap='inferno', vmin=0, vmax=1)
    axes[0].set_title(f"Target: {stem}\nd={args.depth}m")
    axes[0].axis('off')

    axes[1].imshow(np.asarray(I_ws).T, origin='lower',
                    extent=[0, args.Lx, 0, args.Lx], cmap='inferno', vmin=0, vmax=1)
    axes[1].set_title(f"Warm-start (Poisson)\ncos = {ws_cos:.3f}")
    axes[1].axis('off')

    axes[2].imshow(I_final_arr.T, origin='lower',
                    extent=[0, args.Lx, 0, args.Lx], cmap='inferno', vmin=0, vmax=1)
    axes[2].set_title(f"Optimized (honest [0,1])\ncos = {final_cos:.3f}")
    axes[2].axis('off')

    axes[3].imshow(I_stretched.T, origin='lower',
                    extent=[0, args.Lx, 0, args.Lx], cmap='inferno')
    axes[3].set_title(f"Stretched (range = {contrast_pct:.1f}%)\n"
                       f"slope={slope_peak:.3f}, |η|/d={eta_peak/args.depth:.3f}")
    axes[3].axis('off')

    fig.suptitle(
        f"Shallow-tank diagnostic — d={args.depth}m, "
        f"n_modes={args.n_modes}, K={args.n_act_per_side}/side, "
        f"σ_act={args.actuator_width}m, σ/ρ={args.surface_tension:.2e}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out_path = out_dir / f"{stem}_d{args.depth}.png"
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Save raw outputs for inspection. config is JSON-encoded so the
    # 3D-export script can rebuild the apparatus from this file alone.
    import json
    np.savez(out_dir / f"{stem}_d{args.depth}.npz",
             target=target, I_ws=np.asarray(I_ws), I_final=I_final_arr,
             eta=np.asarray(eta), params=np.asarray(params),
             ws_cos=ws_cos, final_cos=final_cos,
             slope_peak=slope_peak, eta_peak=eta_peak,
             contrast_pct=contrast_pct,
             config=json.dumps(vars(args)))


if __name__ == "__main__":
    main()
