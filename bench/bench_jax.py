"""JAX benchmark — companion to bench_julia.jl.

Times the four hot paths used during caustic optimization on a fixed,
deterministic problem so the numbers can be compared 1:1 against Julia.

Run with:  uv run python bench/bench_jax.py
"""

import time
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from wavetank import (
    Tank, Actuator, build_propagator,
    steady_state_amplitudes, caustic_image,
    pack_complex, unpack_complex,
    make_loss,
)


# ── Fixed configuration (must match bench_julia.jl exactly) ──────────

LX, LY, DEPTH, DAMPING = 1.0, 1.0, 0.12, 0.02
N_MODES = 12
NX, NY = 64, 64
N_ACT_PER_SIDE = 4
FREQS_HZ = (1.0, 2.0, 3.0, 4.0)
T_EVAL = 1.0
SIGMA_RENDER = 0.02
SIGMA_BLUR = 0.02
SEED = 0
N_REPEAT = 50      # timed iterations per benchmark
N_WARMUP = 3       # untimed warm-up calls (JAX needs them; Julia does too)


def build_setup():
    tank = Tank(Lx=LX, Ly=LY, depth=DEPTH, damping=DAMPING)
    actuators = []
    for i in range(N_ACT_PER_SIDE):
        t = (i + 1) / (N_ACT_PER_SIDE + 1)
        actuators += [
            Actuator(x=0.0,    y=t * LY, width=0.05),
            Actuator(x=LX,     y=t * LY, width=0.05),
            Actuator(x=t * LX, y=0.0,    width=0.05),
            Actuator(x=t * LX, y=LY,     width=0.05),
        ]
    prop = build_propagator(tank, actuators, n_modes=N_MODES, nx=NX, ny=NY)

    Omega = jnp.array([2 * np.pi * f for f in FREQS_HZ])
    n_freq = len(FREQS_HZ)

    # Deterministic phasors via a closed-form so the values match Julia exactly
    # (numpy and Julia RNGs produce different streams from the same seed).
    i_idx = np.arange(prop.n_act)[:, None].astype(np.float64)
    k_idx = np.arange(n_freq)[None, :].astype(np.float64)
    X = 0.3 * np.sin(0.7 * i_idx + 1.3 * k_idx)
    Y = 0.3 * np.cos(0.4 * i_idx + 0.9 * k_idx + 0.2)
    params = jnp.asarray(pack_complex(X, Y))

    # Gaussian-ring target (same closed form Julia builds)
    xs = np.linspace(0, LX, NX)
    ys = np.linspace(0, LY, NY)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    r = np.sqrt((XX - 0.5) ** 2 + (YY - 0.5) ** 2)
    target = np.exp(-((r - 0.28) ** 2) / (2 * 0.08 ** 2)).astype(np.float64)
    target /= target.max()

    return prop, Omega, params, target


def bench(label, fn, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
    """Time fn() — assumes fn block_until_ready's its result."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    median_ms = times[len(times) // 2] * 1e3
    best_ms   = times[0]                * 1e3
    print(f"  {label:32s}  median = {median_ms:8.3f} ms   best = {best_ms:8.3f} ms")
    return median_ms


def main():
    print("=" * 64)
    print("  JAX benchmark — wavetank caustic pipeline")
    print("=" * 64)
    print(f"  JAX {jax.__version__} on {jax.default_backend()}")

    prop, Omega, params, target = build_setup()
    n_freq = len(FREQS_HZ)

    print(f"  Modes={len(prop.omega)}  Grid={NX}x{NY}  "
          f"Actuators={prop.n_act}  Frequencies={n_freq}")
    print(f"  Parameter dim = {params.size}")
    print(f"  Repeats={N_REPEAT}, warmup={N_WARMUP}\n")

    # ── 1) steady_state_amplitudes (jit-compiled) ────────────────────
    @jax.jit
    def ss_fn(p):
        X, Y = unpack_complex(p, prop.n_act, n_freq)
        P = X + 1j * Y
        return steady_state_amplitudes(prop, P, Omega, T_EVAL)

    a_ref = ss_fn(params); a_ref.block_until_ready()
    bench("steady_state (jit)",
          lambda: ss_fn(params).block_until_ready())

    # ── 2) caustic_image forward (jit-compiled) ──────────────────────
    @jax.jit
    def render_fn(p):
        X, Y = unpack_complex(p, prop.n_act, n_freq)
        P = X + 1j * Y
        a = steady_state_amplitudes(prop, P, Omega, T_EVAL)
        _, _, I = caustic_image(prop, a, sigma=SIGMA_RENDER)
        return I

    I_ref = render_fn(params); I_ref.block_until_ready()
    bench("caustic_image forward (jit)",
          lambda: render_fn(params).block_until_ready())

    # ── 3) Loss only (jit-compiled) ──────────────────────────────────
    loss_fn = make_loss(prop, target, np.asarray(Omega), T_EVAL,
                        sigma=SIGMA_RENDER, sigma_blur=SIGMA_BLUR,
                        loss_type='cosine', lambda_energy=0.0)
    loss_jit = jax.jit(loss_fn)
    L_ref = float(loss_jit(params))
    bench("loss (jit)",
          lambda: loss_jit(params).block_until_ready())

    # ── 4) Loss + gradient (jit-compiled) ────────────────────────────
    vg_fn = jax.jit(jax.value_and_grad(loss_fn))
    L_vg, g_ref = vg_fn(params); g_ref.block_until_ready()
    bench("value_and_grad (jit)",
          lambda: vg_fn(params)[1].block_until_ready())

    print()
    print(f"  Reference values (for cross-language sanity)")
    print(f"    a[0]            = {float(a_ref[0]):+.10e}")
    print(f"    sum(I)          = {float(jnp.sum(I_ref)):+.10e}")
    print(f"    loss            = {float(L_ref):+.10e}")
    print(f"    ‖∇loss‖         = {float(jnp.linalg.norm(g_ref)):+.10e}")
    print("=" * 64)


if __name__ == "__main__":
    main()
