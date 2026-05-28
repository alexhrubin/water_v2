"""Export an optimized surface (η) as a binary animation for the WebGL demo.

Takes either:
  (a) an npz with `eta_steady`, `eta_trans`, or `eta` — uses that directly
      (one static snapshot, repeated for the demo's frame buffer), OR
  (b) an npz with `params` (steady-state phasors) + a config dict —
      computes η(x,y,t) at N times over one driving period for a real
      animation.

Output: a binary file the WebGL demo (external/webgl-water-demo) can load
via the "Load animation" file picker. Format:

    [u32 magic = 0xDEADBEEF]
    [u32 n_frames]
    [u32 nx]
    [u32 ny]
    [f32 period_s]
    [f32 Lx]
    [f32 depth]
    [f32 eta_max]                  (peak |η|, for normalization on JS side)
    [f32 * n_frames * nx * ny]    (heights in m, row-major)

Run:
    python notebooks/export_3d_animation.py \\
        --npz notebooks/transient_test/3spot_gaussian_128_transient.npz \\
        --output notebooks/anim_3spot.bin

If the npz has a `config` field (saved by optimize_transient.py),
apparatus parameters are read from it automatically. Otherwise pass them
as CLI args.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from wavetank import (
    Tank, Actuator, build_propagator,
    steady_state_amplitudes, reconstruct_surface,
    unpack_complex,
)


def resample_eta_to_grid(eta_src, target_nx, target_ny):
    """Bilinearly resample eta from its source grid to (target_nx, target_ny)."""
    if eta_src.shape == (target_nx, target_ny):
        return eta_src.astype(np.float32)
    src_nx, src_ny = eta_src.shape
    xs_src = np.linspace(0, 1, src_nx)
    ys_src = np.linspace(0, 1, src_ny)
    xs_dst = np.linspace(0, 1, target_nx)
    ys_dst = np.linspace(0, 1, target_ny)
    # Simple bilinear via numpy
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((xs_src, ys_src), eta_src,
                                      method='linear', bounds_error=False,
                                      fill_value=0.0)
    Xd, Yd = np.meshgrid(xs_dst, ys_dst, indexing='ij')
    pts = np.stack([Xd.ravel(), Yd.ravel()], axis=-1)
    return interp(pts).reshape(target_nx, target_ny).astype(np.float32)


def build_apparatus(cfg):
    """Build Propagator from a config dict (the JSON saved by optimize_*)."""
    tank = Tank(Lx=cfg['Lx'], Ly=cfg['Lx'],
                depth=cfg['depth'], damping=cfg['damping'],
                surface_tension=cfg.get('surface_tension', 0.0))
    positions = []
    n_act_per_side = cfg['n_act_per_side']
    Lx = cfg['Lx']
    for t in np.linspace(0, 1, n_act_per_side + 2)[1:-1]:
        positions += [(t*Lx, 0), (t*Lx, Lx), (0, t*Lx), (Lx, t*Lx)]
    acts = [Actuator(x, y, width=cfg.get('actuator_width', 0.05))
            for x, y in positions]
    prop = build_propagator(tank, acts, n_modes=cfg['n_modes'],
                             nx=cfg['nx'], ny=cfg['ny'])
    return prop


def build_render_prop(cfg, render_nx, render_ny):
    """Build a propagator at the desired render resolution (same physics,
    same modal basis, same actuators — just a different evaluation grid).
    Lets us evaluate the cos basis analytically at the high-res grid
    instead of bilinearly resampling a lower-res reconstruction."""
    tank = Tank(Lx=cfg['Lx'], Ly=cfg['Lx'],
                depth=cfg['depth'], damping=cfg['damping'],
                surface_tension=cfg.get('surface_tension', 0.0))
    n_act_per_side = cfg['n_act_per_side']
    Lx = cfg['Lx']
    positions = []
    for t in np.linspace(0, 1, n_act_per_side + 2)[1:-1]:
        positions += [(t*Lx, 0), (t*Lx, Lx), (0, t*Lx), (Lx, t*Lx)]
    acts = [Actuator(x, y, width=cfg.get('actuator_width', 0.05))
            for x, y in positions]
    return build_propagator(tank, acts, n_modes=cfg['n_modes'],
                             nx=render_nx, ny=render_ny)


def render_animation_from_params(cfg, prop, params, Omega_freqs, period_s,
                                  n_frames, render_nx, render_ny,
                                  with_normals=True):
    """Compute η(x,y,t), and optionally analytical gradients, at n_frames times.

    When with_normals=True, also computes the surface normals in Wallace's
    world coordinates (his pool spans 2 world units laterally, so slope
    in his coords = slope_meters · Lx/2). Avoids the bilinear resample +
    4-neighbor finite-diff chain that Wallace's normalShader would
    otherwise apply, which low-pass-filters high-k gradient content.
    """
    # Render propagator: same physics as the optimization prop, but at
    # the higher render resolution so we can evaluate analytically.
    render_prop = build_render_prop(cfg, render_nx, render_ny)

    n_freq = len(Omega_freqs)
    X, Y = unpack_complex(jnp.asarray(params), prop.n_act, n_freq)
    P = X + 1j * Y
    times = np.linspace(0, period_s, n_frames + 1)[:-1]
    frames = np.zeros((n_frames, render_nx, render_ny), dtype=np.float32)
    normals = np.zeros((n_frames, render_nx, render_ny, 2),
                        dtype=np.float32) if with_normals else None
    Lx = cfg['Lx']
    for i, t in enumerate(times):
        # Modal amplitudes at this time — uses the render_prop's omega
        # (same modes, so same amplitudes).
        a = steady_state_amplitudes(render_prop, P, jnp.asarray(Omega_freqs),
                                     float(t))
        eta, deta_dx, deta_dy = reconstruct_surface(render_prop, a)
        frames[i] = np.asarray(eta).astype(np.float32)
        if with_normals:
            # Convert slopes from our (meters) coords to Wallace's world
            # coords (pool spans 2 units laterally, so dx_wallace =
            # 2/Lx · dx_meters → slope_wallace = slope_meters · Lx/2).
            slope_x_w = np.asarray(deta_dx) * (Lx / 2)
            slope_y_w = np.asarray(deta_dy) * (Lx / 2)
            norm = np.sqrt(1.0 + slope_x_w**2 + slope_y_w**2)
            # info.b = n_x, info.a = n_z (Wallace's channel convention)
            normals[i, ..., 0] = (-slope_x_w / norm).astype(np.float32)
            normals[i, ..., 1] = (-slope_y_w / norm).astype(np.float32)
        if i % 10 == 0 or i == n_frames - 1:
            slope_peak = float(np.max(np.sqrt(
                np.asarray(deta_dx)**2 + np.asarray(deta_dy)**2)))
            print(f"  frame {i:3d}: t={t:.4f}s, peak |η|={np.abs(frames[i]).max():.3e}, "
                  f"peak |∇η|={slope_peak:.4f}")
    return frames, normals, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True)
    parser.add_argument('--use_field', type=str, default='auto',
                        help="Which field in npz to use: 'eta', 'eta_steady', "
                             "'eta_trans', 'params', or 'auto'")
    parser.add_argument('--render_nx', type=int, default=256)
    parser.add_argument('--render_ny', type=int, default=256)
    parser.add_argument('--n_frames', type=int, default=1,
                        help="Number of animation frames (default 1 = static).")
    parser.add_argument('--period_s', type=float, default=None,
                        help="Animation period (only used with params). "
                             "Default: 2π/Ω_min.")
    parser.add_argument('--output', type=str, default='surface_animation.bin')
    parser.add_argument('--no_normals', action='store_true',
                        help="Don't include analytical surface normals in the "
                             "output binary. By default normals ARE included when "
                             "the npz contains `params` (lets the demo bypass its "
                             "finite-difference normal computation, which low-pass-"
                             "filters high-k gradient content).")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    print(f"Loaded {args.npz}, fields: {list(data.keys())}")

    # Read config if present
    cfg = None
    if 'config' in data:
        cfg_raw = str(data['config'])
        cfg = json.loads(cfg_raw)
        print(f"Found config: depth={cfg.get('depth')}, n_modes={cfg.get('n_modes')}, "
              f"n_act_per_side={cfg.get('n_act_per_side')}, Lx={cfg.get('Lx')}")
    else:
        raise SystemExit("npz has no `config` field. Re-run the optimization "
                         "with a script that saves apparatus config, or "
                         "manually patch this exporter to take CLI args.")

    # Decide which path: animation from params, or static frame from eta
    if args.use_field == 'auto':
        if 'params' in data and args.n_frames > 1:
            field = 'params'
        elif 'eta' in data:
            field = 'eta'
        elif 'eta_steady' in data:
            field = 'eta_steady'
        elif 'eta_trans' in data:
            field = 'eta_trans'
        elif 'params' in data:
            field = 'params'
        else:
            raise SystemExit("npz contains none of: params, eta, eta_steady, eta_trans")
    else:
        field = args.use_field
    print(f"Using field: {field}")

    Lx = cfg['Lx']
    depth = cfg['depth']

    normals = None
    if field == 'params':
        # Build apparatus, run steady_state at multiple times.
        prop = build_apparatus(cfg)
        freqs = np.linspace(cfg['freq_min_hz'], cfg['freq_max_hz'], cfg['n_freq'])
        Omega = 2 * np.pi * freqs
        period_s = args.period_s
        if period_s is None:
            period_s = 2 * np.pi / Omega[0]
        print(f"Computing {args.n_frames} frames over period {period_s:.4f}s"
              f"{' (with analytical normals)' if not args.no_normals else ''}")
        frames, normals, times = render_animation_from_params(
            cfg, prop, data['params'], Omega, period_s,
            args.n_frames, args.render_nx, args.render_ny,
            with_normals=not args.no_normals,
        )
    else:
        eta = np.asarray(data[field])
        print(f"Static frame from `{field}`, source shape={eta.shape}, "
              f"peak |η|={np.abs(eta).max():.3e}")
        single = resample_eta_to_grid(eta, args.render_nx, args.render_ny)
        # Repeat to fill animation buffer
        frames = np.tile(single[None, ...], (args.n_frames, 1, 1))
        period_s = 1.0  # placeholder
        if not args.no_normals:
            print("  (normals omitted: only `params` path supports analytical "
                  "normals; static eta-field falls back to Wallace's "
                  "finite-difference normalShader.)")

    eta_max = float(np.abs(frames).max())
    print(f"\nGlobal peak |η| = {eta_max:.4e} m  "
          f"(|η|/depth = {eta_max/depth:.4f})")

    # Write binary. The 32-byte header is unchanged from v1 (magic
    # 0xDEADBEEF). After the heights block we optionally append two
    # additional Float32 channels per voxel — (info.b = n_x, info.a = n_z)
    # in Wallace's world-coord convention. The JS loader detects the
    # extension by comparing arrayBuffer.byteLength to the heights-only
    # size; no magic-number change needed.
    out_path = Path(args.output)
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<I', 0xDEADBEEF))
        f.write(struct.pack('<I', args.n_frames))
        f.write(struct.pack('<I', args.render_nx))
        f.write(struct.pack('<I', args.render_ny))
        f.write(struct.pack('<f', float(period_s)))
        f.write(struct.pack('<f', float(Lx)))
        f.write(struct.pack('<f', float(depth)))
        f.write(struct.pack('<f', eta_max))
        f.write(frames.astype('<f4').tobytes())
        if normals is not None:
            f.write(normals.astype('<f4').tobytes())

    file_size = out_path.stat().st_size
    has_normals_tag = " [with normals]" if normals is not None else ""
    print(f"\nSaved: {out_path} ({file_size:,} bytes, "
          f"{file_size / 1024 / 1024:.2f} MB){has_normals_tag}")
    print(f"\nDrop this file onto the Load Animation control in "
          f"external/webgl-water-demo/")


if __name__ == "__main__":
    main()
