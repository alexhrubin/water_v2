"""
Animate a saved physical_multi.py result as a time sweep around t_eval.

Loads the apparatus and optimized phasors from the .npz, sweeps a range of
times around the optimization target time, renders the caustic at each, and
saves a side-by-side (target | rendered caustic at time t) gif.

Usage:
    python notebooks/animate_from_npz.py <path/to/result.npz> [--out path.gif]
                                          [--t_min t0] [--t_max t1]
                                          [--n_frames N] [--sigma_render s]

If --t_min / --t_max are omitted, the sweep covers
    [t_eval - 1/f_min, t_eval + 1/f_min]
where f_min is the lowest drive frequency in the optimization band.
"""

import argparse, json, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wavetank import (
    Tank, Actuator, build_propagator,
    steady_state_amplitudes, caustic_image, unpack_complex,
)


def rebuild_apparatus(config):
    """Reconstruct (tank, actuators, prop, Omega) from the saved config dict."""
    Lx, Ly = config['Lx'], config['Ly']
    depth = config['depth']
    n_per_side = config['n_act_per_side']
    actuator_width = config['actuator_width']
    n_modes = config['n_modes']
    nx, ny = config['nx'], config['ny']

    tank = Tank(Lx=Lx, Ly=Ly, depth=depth, damping=config['damping'],
                projection_distance=depth)

    actuators = []
    for i in range(n_per_side):
        t = (i + 1) / (n_per_side + 1)
        actuators += [
            Actuator(x=0.0,    y=t * Ly, width=actuator_width),
            Actuator(x=Lx,     y=t * Ly, width=actuator_width),
            Actuator(x=t * Lx, y=0.0,    width=actuator_width),
            Actuator(x=t * Lx, y=Ly,     width=actuator_width),
        ]
    prop = build_propagator(tank, actuators, n_modes=n_modes, nx=nx, ny=ny)

    freqs = np.linspace(config['freq_min_hz'], config['freq_max_hz'],
                        config['n_freq'])
    Omega = jnp.asarray(2.0 * np.pi * freqs)
    return tank, actuators, prop, Omega, freqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz', help='Path to saved physical_multi.py result.')
    ap.add_argument('--out', default=None,
                    help='Output gif path (default: alongside npz).')
    ap.add_argument('--t_min', type=float, default=None)
    ap.add_argument('--t_max', type=float, default=None)
    ap.add_argument('--n_frames', type=int, default=80)
    ap.add_argument('--sigma_render', type=float, default=0.005,
                    help='Render blur sigma in meters (default 0.005, '
                         'matches SIGMA_RENDER in physical_multi.py).')
    ap.add_argument('--fps', type=int, default=15)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    config = json.loads(str(d['config']))
    target = np.asarray(d['target'])
    params = np.asarray(d['params'])

    print(f"Loaded {args.npz}")
    print(f"  apparatus: {config['Lx']:.1f}x{config['Ly']:.1f} m, "
          f"depth {config['depth']:.1f} m, "
          f"{4 * config['n_act_per_side']} actuators, "
          f"{config['n_freq']} freqs in "
          f"[{config['freq_min_hz']:.2f}, {config['freq_max_hz']:.2f}] Hz, "
          f"{config['n_modes']}² modes")
    t_eval = float(config.get('t_eval', 1.0))
    print(f"  optimization target time: t_eval = {t_eval:.3f} s")

    tank, actuators, prop, Omega, freqs = rebuild_apparatus(config)
    n_act = prop.n_act
    n_freq = config['n_freq']
    X, Y = unpack_complex(jnp.asarray(params), n_act, n_freq)
    P = X + 1j * Y

    T_low = 1.0 / float(freqs.min())
    t_min = args.t_min if args.t_min is not None else t_eval - T_low
    t_max = args.t_max if args.t_max is not None else t_eval + T_low
    times = np.linspace(t_min, t_max, args.n_frames)
    sigma_render = args.sigma_render

    print(f"  rendering {args.n_frames} frames from t={t_min:.3f} to "
          f"t={t_max:.3f} s (sweep = {t_max - t_min:.3f} s, "
          f"slowest period = {T_low:.3f} s)")

    frames = []
    for k, t in enumerate(times):
        a_t = steady_state_amplitudes(prop, P, Omega, T=float(t))
        _, _, I_t = caustic_image(prop, a_t, sigma=sigma_render,
                                  full_snell=True)
        # 90 deg CCW so the rendered caustic and target read upright.
        frames.append(np.rot90(np.asarray(I_t)))
        if (k + 1) % 10 == 0:
            print(f"    frame {k+1}/{args.n_frames}")

    target_disp = np.rot90(target)
    vmax = float(np.percentile(np.stack(frames), 99.5))
    frame_dt = (t_max - t_min) / args.n_frames
    base = args.out and os.path.splitext(args.out)[0]
    base = base or os.path.splitext(args.npz)[0]

    # 1. Side-by-side: target | caustic(t)
    fig, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(8.5, 4.4))
    ax_t.imshow(target_disp, cmap='inferno')
    ax_t.set_title('target')
    ax_t.set_xticks([]); ax_t.set_yticks([])
    im_side = ax_c.imshow(frames[0], cmap='inferno', vmin=0, vmax=vmax)
    title_side = ax_c.set_title(f't = {times[0]:.3f} s')
    ax_c.set_xticks([]); ax_c.set_yticks([])

    def update_side(k):
        im_side.set_data(frames[k])
        marker = ' ← t_eval' if abs(times[k] - t_eval) < frame_dt else ''
        title_side.set_text(f't = {times[k]:.3f} s{marker}')
        return [im_side, title_side]

    ani_side = animation.FuncAnimation(fig, update_side, frames=args.n_frames,
                                        blit=False, interval=1000 / args.fps)
    out_side = base + '.gif'
    ani_side.save(out_side, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"saved {out_side}")

    # 2. Caustic only.
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    im_solo = ax.imshow(frames[0], cmap='inferno', vmin=0, vmax=vmax)
    title_solo = ax.set_title(f't = {times[0]:.3f} s')
    ax.set_xticks([]); ax.set_yticks([])

    def update_solo(k):
        im_solo.set_data(frames[k])
        marker = ' ← t_eval' if abs(times[k] - t_eval) < frame_dt else ''
        title_solo.set_text(f't = {times[k]:.3f} s{marker}')
        return [im_solo, title_solo]

    ani_solo = animation.FuncAnimation(fig, update_solo, frames=args.n_frames,
                                        blit=False, interval=1000 / args.fps)
    out_solo = base + '_caustic.gif'
    ani_solo.save(out_solo, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"saved {out_solo}")


if __name__ == '__main__':
    main()
