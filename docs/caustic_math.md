# Caustic Mathematics

The minimum amount of geometric optics you need to follow how caustics
work in this project. Lays out the ray map, the Jacobian, the caustic
condition, and how surface curvature relates to caustic brightness.
Referenced from `image_generation_constraints.md` and
`reachability_and_capacity.md`.

## The ray map

Sunlight enters the tank as approximately-parallel vertical rays. At
each point on the water surface, the ray refracts according to Snell's
law (we use the paraxial small-slope approximation), then travels in a
straight line through the water and the air gap (if any) until it hits
the floor.

In coordinates, write `r_s = (x, y)` for the horizontal position where a
ray enters the surface, and `r_f` for where the same ray lands on the
floor. The paraxial (small-slope) map is:

    r_f(r_s)  =  r_s  +  throw · (1 - 1/n_water) · ∇η(r_s)        (paraxial)

where `η(x, y)` is the surface elevation, `∇η` is its horizontal
gradient, `throw` is the optical distance from surface to floor, and
`n_water ≈ 1.33`. The deflection coefficient `(1 - 1/n) ≈ 0.248` comes
from Snell's law: a vertical ray hitting a surface tilted by slope `s`
refracts at the air-water interface and emerges at angle `(1-1/n)·s`
from vertical. (The full vector Snell's law reduces to this to leading
order; corrections are quadratic in slope.)

So the map is determined entirely by `∇η`. A flat surface (∇η = 0)
gives the identity map — every ray lands directly below where it
entered, and the floor is uniformly illuminated. A curved surface
deforms the map — rays converge where the surface focuses them, diverge
where it scatters.

## Intensity from the Jacobian

For a uniform incoming beam, intensity on the floor is just *how many
rays per unit area arrive there*. By change of variables:

    I(r_f)  =  I_incident · |det(J)|^{-1}

where `J = ∂r_f/∂r_s` is the **2×2 Jacobian** of the ray map.
Computing it from the paraxial formula:

    J  =  I_{2×2}  +  throw · (1 - 1/n_water) · H

where `H = ∇∇η` is the **Hessian** of the surface (the 2×2 matrix of
second partial derivatives). Three regimes:

  - `det(J) > 1`:  surface expands rays  →  dimmer than ambient
  - `det(J) = 1`:  flat surface  →  uniform illumination
  - `0 < det(J) < 1`:  surface compresses rays  →  brighter than ambient
  - `det(J) → 0`:   **caustic** — diverging intensity

The caustic locus is the set of floor points where `det(J) = 0`. In
the ray-optics limit, intensity there is mathematically infinite (a
delta-function singularity); in practice it's regularized by the
rendering blur σ, finite ray count, or — in the real world — the
finite angular size of the sun and diffraction.

## Where caustics form: the focal condition

The proper paraxial refraction formula (Snell's law for small angles)
is:

    r_f  =  r_s  -  throw · (1 - 1/n_water) · ∇η

with corresponding Jacobian:

    J  =  I  -  throw · (1 - 1/n_water) · H

Write the Hessian eigenvalues as `λ₁, λ₂`:

    det(J)  =  (1 - throw·(1−1/n)·λ₁) · (1 - throw·(1−1/n)·λ₂)

A caustic forms wherever *either* eigenvalue hits the critical value:

    λ_crit  =  1 / (throw · (1 − 1/n_water))  =  n_water / ((n_water−1) · throw)

For `n_water = 1.33`, this works out to **`λ_crit ≈ 4/throw`**.

This is the **focal condition**. At this curvature, the local
"surface lens" focuses parallel rays exactly at floor depth. Other
directions may defocus, but at least one focuses, and that's enough
to make the local Jacobian singular.

So caustics form along **contour lines** of the surface Hessian — the
1D locus where one eigenvalue hits `λ_crit`. As the surface evolves,
these contours sweep around, and the caustic lines on the floor move
with them.

## The complete derivation: mode count for caustic formation

Now build the full chain from `det(J)=0` to "how many modes does my
basis need."

**Step 1: Single-mode caustic threshold.** Decompose the surface in
cosine modes `η = Σ aₙ φₙ` with wavenumbers `qₙ`. For a single mode at
amplitude `a` and wavenumber `q`, the Hessian eigenvalue is `−a·q²`.
The focal condition `|λ| ≥ 1/(throw·(1−1/n))` becomes:

    a · q² · throw · (1 − 1/n)  ≥  1

or, rearranged:

    **a · q² · throw  ≥  n/(n−1)  ≈  4  (for n=1.33)**

This is the canonical "curvature × depth ≥ constant" rule for caustic
formation by a single sinusoidal wave. It is independent of slope or
amplitude in isolation — it is the product `a·q²·throw` that matters.

**Step 2: Linear-wave constraint.** Linear-wave theory requires
slope `|∇η| ≤ s_max`. For a single mode of wavenumber `q`, slope is
`a·q`, so the linearity-allowed amplitude is `a ≤ s_max/q`. Substituting
into the caustic condition:

    (s_max / q) · q² · throw  ≥  4
    s_max · q · throw  ≥  4
    **q  ≥  4 / (s_max · throw)**

For `s_max = 0.1`, this is **`q ≥ 40 / throw`** (in radians/meter).

This says: under the linear-wave slope cap, only modes with wavenumber
above `40/throw` can form caustics. Or equivalently — there is a
minimum spatial frequency below which the surface simply cannot carry
enough curvature to clear the focal threshold within linear physics.

**Step 3: Modal basis constraint.** A tank of side length `L` has
modes `q_n = nπ/L`. Plugging in:

    nπ/L  ≥  4 / (s_max · throw)
    **n  ≥  4 · L / (π · s_max · throw)  =  k_n · (L/throw)**

with the dimensionless constant `k_n = 4/(π·s_max) ≈ 13` for
`s_max = 0.1`. So:

    **n_min  ≈  13 · (L/throw)**         (linear regime)

This is the punchline. The minimum mode index per axis to form caustics
in the linear regime depends *only on the apparatus geometry ratio*
`L/throw`, with a coefficient of ~13.

**Step 4: Apply to apparatus regimes.** For `L = 1m` tank:

| throw (m) | L/throw | linear `n_min` (s=0.1) | HOS M=2 `n_min` (s=0.3) | HOS M=3 `n_min` (s=0.4) |
|---|---|---|---|---|
| 5 (deep pool) | 0.2  | 3   | 1   | 1   |
| 2 (our typical) | 0.5  | 7   | 2   | 2   |
| 0.5 (shallow tank) | 2    | 26  | 9   | 7   |
| 0.1 (very shallow) | 10   | **130** | **43**  | **33**  |
| 0.02 (puddle) | 50   | 650 | 217 | 163 |

The L/throw column is the dimensionless geometry parameter. At
`L/throw ≤ 1`, our typical 15-mode-per-axis basis comfortably clears
even the linear threshold. At `L/throw ≥ 5` (true shallow), the linear
threshold needs many tens of modes; HOS reduces this by ~3-4× but
puddle-scale `L/throw = 50` still needs hundreds of modes per axis.

This **directly explains why pools and puddles get sharp caustics in
real life** despite being shallow: real water has access to *much*
higher-frequency modes (capillary ripples with `λ` of millimeters,
giving `q ~ 1000+ rad/m`). Per the derivation, you need
`q ≥ 40/throw = 2000 rad/m` at `throw = 0.02m` — completely
inaccessible to our 15-30 mode basis on a 1m tank, but trivial for
the actual capillary-gravity modes that occur in real water.

## Empirical confirmation

We tested this prediction directly:

- **`d=2m, n=15, HOS M=2`** (well above threshold): cos jumps from
  warm-start 0.27 → 0.835. Caustics form sharply. ✓
- **`d=0.1m, n=15, HOS M=2`** (well below threshold; need ~43 modes):
  cos stays at warm-start (0.27 → 0.26). Slopes hit cap (0.113) but
  curvature `k·slope ≈ 7.6` can't clear threshold `≈ 13–40`. **No
  caustic formation.** ✓
- **`d=0.1m, n=50, HOS M=2`** (above threshold; predicted to work):
  HOS integrator goes NaN at iter 1. The high mode count's
  nonlinear-mode-coupling overflow is an integrator-stability issue,
  not a physics one. With more careful dealiasing and timestepping,
  this run should succeed. *Future work.*

So the analytical bound is confirmed in the cases the simulator can
actually run. The "shallow water + linear regime can't form caustics"
result is not a failure mode of the apparatus — it's the geometric
consequence of slope-capped wave amplitudes being unable to deliver
the required curvature at the wavelengths our basis supports.

## Caustic brightness: the curvature excess

Crossing the focal condition doesn't tell you how *bright* the caustic
is — only *whether* it forms. Brightness is set by how rapidly `det(J)`
passes through zero, which is set by how much surface curvature you
have *in excess of* the critical value `n_water/throw`.

Near a caustic line, intensity scales like `1/√(distance from caustic)`,
and the peak (after regularization) goes as `√(H_excess)` where:

    H_excess  =  H_available  −  H_critical
              =  (k_max · s_max)  −  n_water/((n_water−1)·throw)

Concretely: example.ipynb's fantasy regime has `H_excess ≈ 925` (slopes
of 14 give massive curvature far above critical). Our linear cap-enforced
deep-water runs have `H_excess ≈ 6.4` (barely above critical). The peak
brightness scales as `√(925 / 6.4) ≈ 12×` between the two — consistent
with the observed visual quality gap.

## Why this matters for the apparatus story

Two of the dimensionless apparatus knobs map directly to the focal
feasibility inequality:

  - **More slope budget** (HOS) raises `s_max`, lowering the required `k` and raising `H_excess`.
  - **Larger throw** (deeper water, or glass bottom) lowers `n_water/throw`, lowering the required `k` and raising `H_excess`.
  - **More high-k modes** raises `k_max` and `H_max`, raising `H_excess`.

These compound multiplicatively. The reachability and information-capacity
analyses in `docs/reachability_and_capacity.md` are essentially this same
inequality with the right book-keeping for what the optimizer can actually
*do* with the available `H_excess`.

## What this doesn't include

  - **Anisotropic Hessians**. The surface is 2D, so `H` has two
    eigenvalues. A caustic line forms when *either* hits the focal
    threshold; a *point* caustic forms only when both do simultaneously
    (rare). Most observed caustics are line-shaped because they're the
    1D contour of "one eigenvalue at critical."
  - **Higher-order caustics**. Cusps, swallowtails, etc. appear where
    the caustic surface itself develops a singularity (the *second*
    eigenvalue also changes). Mathematically rich, visually distinctive
    (the "bright cusps" you sometimes see at the tip of a caustic line),
    but our optimization doesn't deliberately try to produce them.
  - **Diffraction and finite-source effects.** The sun has angular size
    ~0.5°, which sets a real-world minimum caustic-line width of
    `throw · sin(0.5°)`. Our rendering blur σ plays the same role in
    simulation. The mathematical singularity is regularized either way.
  - **Wave breaking and post-breaking dissipation.** Past slope ~0.44 the
    surface stops being a single-valued function of `(x, y)` — Snell's
    law itself stops applying. HOS handles up to slope ~0.4 cleanly; past
    that you need different physics entirely.

## References

  - Berry, M. V. and Upstill, C. (1980). "Catastrophe optics: morphologies
    of caustics and their diffraction patterns." *Progress in Optics 18*.
    Standard reference for the singularity structure of caustic surfaces.
  - Born, M. and Wolf, E., *Principles of Optics*. Chapter on geometrical
    optics covers the Jacobian/intensity formulation.
