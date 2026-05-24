"""
Pre-flight feasibility analysis for caustic targets.

Predicts whether a given target is achievable in the linear-wave regime
*before* burning compute on a 1000-iteration optimization. Composes
existing pieces (analytical_solve + reconstruct_surface + caustic_image)
into a single verdict the user can act on.

A target can fail feasibility for two distinct reasons:

1. **Target itself violates linearity.** The analytical Poisson inversion
   produces ideal modal amplitudes a_desired. If the surface implied by
   a_desired already has |η|/depth > 0.1 or |∇η| > 0.1, no actuator
   improvement helps — the target intrinsically requires nonlinear waves.

2. **Actuator subspace cannot reach the target.** a_desired is fine, but
   the realized phasors p0 (after least-squares projection onto the
   actuator-driven subspace) produce a much smaller surface. The tank
   doesn't have enough actuators / modes / driving frequencies to express
   what the target asks for.

The report distinguishes these cases and prints actionable suggestions.
"""

import numpy as np
import jax.numpy as jnp

from .physics import Propagator, steady_state_amplitudes, unpack_complex
from .render import reconstruct_surface, caustic_image
from .loss import cosine_loss
from .analytical import analytical_solve


# ── Linear-wave thresholds ────────────────────────────────────────────

ETA_OVER_DEPTH_LIMIT = 0.1   # |η|/depth: linear free-surface assumption
SLOPE_LIMIT = 0.1            # |∇η|: paraxial refraction assumption


# ── Report ────────────────────────────────────────────────────────────

def feasibility_report(
    prop: Propagator,
    target: np.ndarray,
    Omega_freqs: np.ndarray,
    T_eval: float = 1.0,
    *,
    n_water: float = 1.33,
    sigma: float = 0.02,
) -> dict:
    """
    Predict whether `target` is reproducible in the linear-wave regime.

    Runs the analytical Poisson warm-start, then evaluates the implied
    surface and the realized warm-start caustic. Returns a dict suitable
    for printing or for programmatic gating before launching an optimizer.

    Parameters
    ----------
    prop        : configured Propagator (typically from setup_from_target)
    target      : target caustic [nx, ny] in [0, 1]
    Omega_freqs : driving angular frequencies [n_freq]
    T_eval      : evaluation time (s)
    n_water     : refractive index
    sigma       : Gaussian blur for the realized caustic preview

    Returns
    -------
    dict with two nested 'desired'/'realized' blocks (max_abs_eta_m,
    eta_over_depth, max_slope, a_norm), plus:
        cos_sim_warmstart
        water_depth_m              : tank.depth (sets the linearity caps)
        projection_distance_m      : tank.throw (optical throw)
        resolution_limit_m         : smallest feature size in linear regime
                                       = throw · SLOPE_LIMIT / n_water
        recommended_projection_distance_m
            : optical throw at which the desired surface fits inside the
              linear-wave caps, computed from the a ∝ 1/throw scaling of
              the analytical Poisson inversion
        p0, a_desired, a_realized
    and three boolean verdict keys:
        target_achievable_in_linear_regime
        actuator_subspace_sufficient
        ready_to_optimize
    """
    sol = analytical_solve(prop, target, Omega_freqs, T_eval, n_water=n_water)
    p0 = jnp.asarray(sol['p0'])
    a_desired = jnp.asarray(sol['a_desired'])

    # Realized amplitudes after least-squares projection onto reachable subspace
    n_act = prop.n_act
    n_freq = len(Omega_freqs)
    X, Y = unpack_complex(p0, n_act, n_freq)
    P = X + 1j * Y
    a_realized = steady_state_amplitudes(prop, P, jnp.asarray(Omega_freqs), T_eval)

    # Surface stats from BOTH a_desired (the ideal) and a_realized (what
    # actuators can drive). The gap diagnoses tank expressivity.
    eta_d, dx_d, dy_d = reconstruct_surface(prop, a_desired)
    eta_r, dx_r, dy_r = reconstruct_surface(prop, a_realized)

    # Two physically distinct length scales — see Tank docstring.
    water_depth = float(prop.tank.depth)   # hydrodynamic; sets |η|/depth cap
    throw       = float(prop.tank.throw)   # optical projection distance

    def _stats(eta, dx_, dy_, a):
        max_abs_eta = float(jnp.max(jnp.abs(eta)))
        return {
            'max_abs_eta_m':  max_abs_eta,
            'eta_over_depth': max_abs_eta / water_depth,
            'max_slope':      float(jnp.max(jnp.sqrt(dx_ ** 2 + dy_ ** 2))),
            'a_norm':         float(jnp.linalg.norm(a)),
        }

    desired = _stats(eta_d, dx_d, dy_d, a_desired)
    realized = _stats(eta_r, dx_r, dy_r, a_realized)

    # Realized caustic similarity (the warm-start the optimizer would start from)
    _, _, I0 = caustic_image(prop, a_realized, sigma=sigma)
    target_j = jnp.asarray(target)
    cos_sim = float(1.0 - cosine_loss(I0, target_j))

    target_ok = (desired['eta_over_depth'] < ETA_OVER_DEPTH_LIMIT
                 and desired['max_slope'] < SLOPE_LIMIT)
    realized_ok = (realized['eta_over_depth'] < ETA_OVER_DEPTH_LIMIT
                   and realized['max_slope'] < SLOPE_LIMIT)

    # ── Resolution limit ──────────────────────────────────────────────
    # Max ray displacement = throw · slope_max = (throw/n_water) · SLOPE_LIMIT.
    # Sets the smallest feature the apparatus can produce in the linear regime.
    resolution_limit = throw * SLOPE_LIMIT / n_water

    # ── Recommended projection distance ───────────────────────────────
    # The Poisson inversion gives a ∝ 1/throw, so desired-surface stats
    # also scale as throw₀/throw_new. Imposing both linearity caps with
    # water depth held fixed:
    #   (throw₀/throw_new) · max|η_d| / water_depth ≤ ETA_OVER_DEPTH_LIMIT
    #   (throw₀/throw_new) · max|∇η_d|              ≤ SLOPE_LIMIT
    # The slope cap scales linearly in throw; the height cap scales
    # linearly because water_depth is held fixed under this rescaling
    # (the height-cap denominator does not move with throw).
    t_h = float(throw * desired['eta_over_depth'] / ETA_OVER_DEPTH_LIMIT)
    t_s = float(throw * desired['max_slope']      / SLOPE_LIMIT)
    recommended_throw = max(t_h, t_s)

    return {
        'cos_sim_warmstart':   cos_sim,
        'water_depth_m':       water_depth,
        'projection_distance_m': throw,
        'resolution_limit_m':  resolution_limit,
        'recommended_projection_distance_m': recommended_throw,
        'desired':  desired,
        'realized': realized,
        'target_achievable_in_linear_regime': target_ok,
        'actuator_subspace_sufficient':       realized_ok,
        'ready_to_optimize':                  target_ok and realized_ok,
        'p0':         np.asarray(p0),
        'a_desired':  np.asarray(a_desired),
        'a_realized': np.asarray(a_realized),
    }


def print_feasibility_report(report: dict) -> None:
    """Pretty-print a feasibility report with verdict and suggestions."""
    d = report['desired']
    r = report['realized']
    water_depth_mm = report['water_depth_m'] * 1000
    throw_mm       = report['projection_distance_m'] * 1000

    bar = "=" * 60
    print(bar)
    print("  Feasibility report")
    print(bar)
    print(f"  Warm-start cosine similarity:  {report['cos_sim_warmstart']:.4f}")
    print(f"  Water depth:           {water_depth_mm:7.1f} mm")
    print(f"  Projection distance:   {throw_mm:7.1f} mm")
    res_mm = report['resolution_limit_m'] * 1000
    ratio = report['projection_distance_m'] / max(report['resolution_limit_m'], 1e-12)
    print(f"  Resolution limit:      {res_mm:7.1f} mm "
          f"(smallest feature in linear regime ≈ throw/{ratio:.1f})")
    print(f"  Recommended throw:     {report['recommended_projection_distance_m']*1000:7.1f} mm "
          f"(for this target to fit the linear regime)")
    print()
    print(f"                       desired    realized")
    print(f"  max|η|     (mm)    {d['max_abs_eta_m']*1000:8.2f}    {r['max_abs_eta_m']*1000:8.2f}")
    print(f"  |η|/depth          {d['eta_over_depth']:8.3f}    {r['eta_over_depth']:8.3f}")
    print(f"  max|∇η|            {d['max_slope']:8.3f}    {r['max_slope']:8.3f}")
    print(f"  ‖a‖                {d['a_norm']:8.4f}    {r['a_norm']:8.4f}")
    print()
    print("  Verdict:")

    # ── Failure mode 1: target itself violates linearity ──────────────
    target_fail = []
    if d['eta_over_depth'] >= ETA_OVER_DEPTH_LIMIT:
        target_fail.append(
            f"desired |η|/depth = {d['eta_over_depth']:.3f}  > "
            f"{ETA_OVER_DEPTH_LIMIT}  (linear-wave limit)")
    if d['max_slope'] >= SLOPE_LIMIT:
        target_fail.append(
            f"desired |∇η|      = {d['max_slope']:.3f}  > "
            f"{SLOPE_LIMIT}  (paraxial limit)")
    for msg in target_fail:
        print(f"    [FAIL] {msg}")

    # ── Failure mode 2: actuator subspace cannot reach target ─────────
    if not target_fail:
        # Only meaningful if the target itself was OK
        if (r['eta_over_depth'] < d['eta_over_depth'] * 0.9
                or r['max_slope'] < d['max_slope'] * 0.9):
            print(f"    [WARN] realized < desired "
                  f"(slope ratio {r['max_slope']/max(d['max_slope'], 1e-12):.2f}, "
                  f"height ratio {r['max_abs_eta_m']/max(d['max_abs_eta_m'], 1e-12):.2f})")
            print("           → actuator subspace cannot fully express the target")

    if report['ready_to_optimize']:
        print("    [PASS] target and actuator subspace both inside the linear regime")
        print(f"           — ready to optimize")
        print(bar)
        return

    # ── Suggestions ──────────────────────────────────────────────────
    print()
    print("  Suggestions:")
    if target_fail:
        rec_mm = report['recommended_projection_distance_m'] * 1000
        print("    • The TARGET itself requires nonlinear waves — even a perfect")
        print("      tank cannot reproduce it under linear/paraxial physics.")
        print("    • Try one or more of:")
        print(f"        - Increase the projection distance to ≥ {rec_mm:.0f} mm "
              f"(currently {throw_mm:.0f} mm)")
        print("        - Blur the target (sharp features need short-wavelength modes)")
        print("        - Pass full_snell=True to optimize_caustic")
        print("          (handles steep slopes — may rescue the slope failure)")
        print("        - Increase n_modes via setup_from_target(n_modes_max=80,")
        print("          energy_fraction=0.99) — finer modes lower amplitudes")
    elif not report['actuator_subspace_sufficient']:
        print("    • The actuator subspace is the bottleneck. Add more actuators")
        print("      or driving frequencies, or place actuators closer to the modes")
        print("      the target needs (see analyze_target output for top modes).")
    print(bar)
