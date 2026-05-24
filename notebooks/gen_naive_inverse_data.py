"""Generate (phasor, caustic) training pairs on GPU via vmap+jit.

Random phasors are sampled, run through the steady-state simulator, and the
resulting caustic image is saved. Samples that violate the linear-wave caps
(|η|/depth or |∇η| > 0.10) are post-selected out so the training set stays
in the physically valid regime.

Run as a script:
    python notebooks/gen_naive_inverse_data.py

Or paste into a Colab cell after `pip install -e .` of the repo.
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from wavetank import (
    Tank, Actuator, build_propagator,
    steady_state_amplitudes, caustic_image, reconstruct_surface,
    unpack_complex, sample_random_phasors,
)

# ── Config ───────────────────────────────────────────────────────────
LX, LY, DEPTH, DAMPING = 1.0, 1.0, 3.0, 0.02
N_MODES        = 12
NX, NY         = 64, 64
N_ACT_PER_SIDE = 5
FREQS_HZ       = (1.0, 1.5, 2.0, 2.5)
T_EVAL         = 1.0
SIGMA_RENDER   = 0.02
PHASOR_SCALE_MIN = 5e-5         # gentle / soft caustics
PHASOR_SCALE_MAX = 5e-4         # near linearity cap / sharp caustics
                                # per-sample scale drawn log-uniform in [min, max]

ETA_CAP        = 0.10           # post-select: |η|/depth threshold
SLOPE_CAP      = 0.10           # post-select: |∇η|     threshold

N_SAMPLES   = 1_000_000
BATCH_SIZE  = 1024
SEED        = 0
OUT_DIR     = Path("data/naive_inverse")

    
# ── Setup ────────────────────────────────────────────────────────────
def build_setup():
    tank = Tank(Lx=LX, Ly=LY, depth=DEPTH, damping=DAMPING)
    acts = []
    for i in range(N_ACT_PER_SIDE):
        t = (i + 1) / (N_ACT_PER_SIDE + 1)
        acts += [
            Actuator(x=0.0,    y=t * LY),
            Actuator(x=LX,     y=t * LY),
            Actuator(x=t * LX, y=0.0),
            Actuator(x=t * LX, y=LY),
        ]
    prop  = build_propagator(tank, acts, n_modes=N_MODES, nx=NX, ny=NY)
    Omega = jnp.asarray([2 * np.pi * f for f in FREQS_HZ])
    return prop, Omega


# ── Batched generator ────────────────────────────────────────────────
def make_batch_fn(prop, Omega):
    n_act, n_freq = prop.n_act, Omega.shape[0]

    def _one(key):
        k_scale, k_phasor = jax.random.split(key)
        log_scale = jax.random.uniform(
            k_scale,
            minval=jnp.log(PHASOR_SCALE_MIN),
            maxval=jnp.log(PHASOR_SCALE_MAX),
        )
        scale = jnp.exp(log_scale)
        p = sample_random_phasors(k_phasor, n_act, n_freq, scale=scale)
        X, Y = unpack_complex(p, n_act, n_freq)
        P = X + 1j * Y
        a = steady_state_amplitudes(prop, P, Omega, T_EVAL)

        eta, deta_dx, deta_dy = reconstruct_surface(prop, a)
        max_eta_norm = jnp.max(jnp.abs(eta)) / DEPTH
        max_slope    = jnp.max(jnp.sqrt(deta_dx ** 2 + deta_dy ** 2))

        _, _, I = caustic_image(prop, a, sigma=SIGMA_RENDER)
        return p, I, max_eta_norm, max_slope

    return jax.jit(jax.vmap(_one))


# ── Main ─────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"JAX {jax.__version__} on {jax.default_backend()}")

    prop, Omega = build_setup()
    n_params = 2 * prop.n_act * Omega.shape[0]
    print(f"  actuators={prop.n_act}  freqs={Omega.shape[0]}  "
          f"modes/axis={N_MODES}  grid={NX}x{NY}")
    print(f"  param dim={n_params}  image dim={NX * NY}")
    print(f"  caps: |η|/d < {ETA_CAP}, |∇η| < {SLOPE_CAP}\n")

    batch_fn = make_batch_fn(prop, Omega)
    rng = jax.random.PRNGKey(SEED)

    # Warm-up + diagnostic: pay JIT compile cost once and report what the
    # phasor scale is actually producing, so the user can pick a sensible
    # value before launching the full run.
    t_compile = time.perf_counter()
    _p, _I, eta_diag, slope_diag = batch_fn(jax.random.split(rng, BATCH_SIZE))
    _I.block_until_ready()
    print(f"  JIT compile: {time.perf_counter() - t_compile:.1f}s")

    eta_arr   = np.asarray(eta_diag)
    slope_arr = np.asarray(slope_diag)
    pct = (50, 90, 99, 100)
    eta_q   = np.percentile(eta_arr,   pct)
    slope_q = np.percentile(slope_arr, pct)
    diag_mask = (eta_arr < ETA_CAP) & (slope_arr < SLOPE_CAP)
    diag_pass = diag_mask.mean()
    print(f"  diagnostic batch (n={BATCH_SIZE}, "
          f"PHASOR_SCALE ∈ [{PHASOR_SCALE_MIN:.1e}, {PHASOR_SCALE_MAX:.1e}]):")
    print(f"    |η|/d   percentiles {pct} = "
          f"{eta_q[0]:.3f}, {eta_q[1]:.3f}, {eta_q[2]:.3f}, {eta_q[3]:.3f} "
          f"(cap {ETA_CAP})")
    print(f"    |∇η|    percentiles {pct} = "
          f"{slope_q[0]:.3f}, {slope_q[1]:.3f}, {slope_q[2]:.3f}, {slope_q[3]:.3f} "
          f"(cap {SLOPE_CAP})")
    print(f"    pass rate = {diag_pass:.1%}")
    if diag_pass < 0.5:
        # Both caps scale linearly with the per-sample scale, so the largest
        # binding ratio at the 99th percentile tells us how much to back off MAX.
        binding = max(eta_q[2] / ETA_CAP, slope_q[2] / SLOPE_CAP)
        suggested = PHASOR_SCALE_MAX / binding
        print(f"\n  pass rate too low. Try PHASOR_SCALE_MAX ≈ {suggested:.3g} "
              f"(current is {PHASOR_SCALE_MAX / suggested:.1f}× too high)")
        print(f"  aborting full run. Adjust PHASOR_SCALE_MAX and re-run.")
        return
    print()

    # Allocate to target N_SAMPLES; oversample by looping until filled
    phasors_all  = np.zeros((N_SAMPLES, n_params), dtype=np.float32)
    caustics_all = np.zeros((N_SAMPLES, NX, NY),   dtype=np.float32)

    filled = 0
    batch_idx = 0
    log_every = 10
    t0 = time.perf_counter()
    while filled < N_SAMPLES:
        keys = jax.random.split(jax.random.fold_in(rng, batch_idx + 1), BATCH_SIZE)
        p_b, I_b, eta_b, slope_b = batch_fn(keys)
        I_b.block_until_ready()

        mask = (np.asarray(eta_b) < ETA_CAP) & (np.asarray(slope_b) < SLOPE_CAP)
        n_keep = min(int(mask.sum()), N_SAMPLES - filled)
        if n_keep > 0:
            phasors_all [filled:filled + n_keep] = np.asarray(p_b)[mask][:n_keep]
            caustics_all[filled:filled + n_keep] = np.asarray(I_b)[mask][:n_keep]
            filled += n_keep

        batch_idx += 1
        if batch_idx % log_every == 0:
            elapsed = time.perf_counter() - t0
            pass_rate = filled / (batch_idx * BATCH_SIZE)
            print(f"  batch {batch_idx:>4}  "
                  f"filled {filled:>6}/{N_SAMPLES}  "
                  f"pass_rate={pass_rate:.1%}  "
                  f"{filled / elapsed:6.0f} valid/s")

    total = time.perf_counter() - t0
    final_pass_rate = filled / (batch_idx * BATCH_SIZE)
    print(f"\nGenerated {filled} valid samples in {total:.1f}s "
          f"({filled / total:.0f} valid/s, "
          f"overall pass rate {final_pass_rate:.1%})")

    # Save
    out_path = OUT_DIR / "dataset.npz"
    np.savez(out_path, phasors=phasors_all, caustics=caustics_all)
    print(f"Saved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    meta = dict(
        Lx=LX, Ly=LY, depth=DEPTH, damping=DAMPING,
        n_modes=N_MODES, nx=NX, ny=NY,
        n_act_per_side=N_ACT_PER_SIDE, n_act=int(prop.n_act),
        freqs_hz=list(FREQS_HZ), T_eval=T_EVAL,
        sigma_render=SIGMA_RENDER,
        phasor_scale_min=PHASOR_SCALE_MIN, phasor_scale_max=PHASOR_SCALE_MAX,
        eta_cap=ETA_CAP, slope_cap=SLOPE_CAP,
        n_samples=filled, seed=SEED,
        pass_rate=final_pass_rate,
        backend=jax.default_backend(),
        wall_clock_s=total,
    )
    meta_path = OUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
