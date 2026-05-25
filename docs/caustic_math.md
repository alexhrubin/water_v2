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
floor. The map is:

    r_f(r_s)  =  r_s  -  (throw / n_water) · ∇η(r_s)         (paraxial)

where `η(x, y)` is the surface elevation, `∇η` is its horizontal
gradient, `throw` is the optical distance from surface to floor, and
`n_water ≈ 1.33`. (The full vector Snell's law is the same to leading
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

    J  =  I_{2×2}  -  (throw / n_water) · H

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

Write the Hessian eigenvalues as `λ₁, λ₂`. Then:

    det(J)  =  (1 - (throw/n_water) · λ₁) · (1 - (throw/n_water) · λ₂)

A caustic forms wherever *either* eigenvalue equals `n_water/throw`:

    λ_caustic  =  n_water / throw

This is the **focal condition**. It says: at this point on the surface,
the local curvature in some direction is exactly what's needed to
focus parallel rays at the floor depth. Other directions may
defocus, but at least one focuses, and that's enough to make the
local Jacobian singular.

So caustics form along **contour lines** of the surface Hessian — the
1D locus where one eigenvalue hits `n_water/throw`. As surface
shape evolves (in periodic driving or random wave motion), these
contours sweep around, and the caustic lines on the floor move with
them.

## Connecting to surface modes

Decompose the surface in cosine modes:

    η(x, y, t) = Σ a_{m,n}(t) · cos(mπx/Lx) cos(nπy/Ly)

with wavenumber `k_{m,n} = π·√((m/Lx)² + (n/Ly)²)`. For a single mode,

    |Hessian| ~ k²·|a|       (curvature is amplitude × wavenumber-squared)
    |slope|   ~ k·|a|        (slope is amplitude × wavenumber)

So `Hessian = k · slope`. At the slope cap `s_max`, the *maximum
possible* curvature from a single mode of wavenumber `k` is:

    H_max(k)  =  k · s_max

Plugging into the focal condition: a mode of wavenumber `k` can form
caustics only if `H_max(k) ≥ n_water / throw`, i.e.:

    k  ≥  n_water / (throw · s_max)   (focal feasibility)

This is the cleanest one-line summary of when caustics are possible
at all: you need enough mode wavenumber, slope budget, and throw,
multiplied together, to clear the refractive threshold.

Examples for `n_water = 1.33`, `s_max = 0.1` (linear cap):

| throw  | focal feasibility (min k) | min mode index (1m tank) |
|--------|---------------------------|--------------------------|
| 0.1 m  | k ≥ 133                   | m ≈ 42 (need 40+ modes!) |
| 1 m    | k ≥ 13.3                  | m ≈ 4                    |
| 5 m    | k ≥ 2.66                  | m = 1 already qualifies  |

So in the linear regime at shallow throw, the high-`k` modes needed to
form caustics may be outside your basis entirely. Deeper water (more
throw) lowers the bar; nonlinear physics (HOS, with `s_max` up to ~0.3)
lowers it independently. Doing both lowers it a lot.

## Caustic brightness: the curvature excess

Crossing the focal condition doesn't tell you how *bright* the caustic
is — only *whether* it forms. Brightness is set by how rapidly `det(J)`
passes through zero, which is set by how much surface curvature you
have *in excess of* the critical value `n_water/throw`.

Near a caustic line, intensity scales like `1/√(distance from caustic)`,
and the peak (after regularization) goes as `√(H_excess)` where:

    H_excess  =  H_available  −  H_critical
              =  (k_max · s_max)  −  (n_water / throw)

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
