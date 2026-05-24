# Constraints on Image Generation

Design notes on the relationships between apparatus parameters and
caustic image quality. Written to support apparatus sizing decisions —
not a tutorial on using the code.

## The two physical length scales

The system has two distinct length scales that are easy to conflate but
must be kept separate:

- **Water depth** (`Tank.depth`): a hydrodynamic parameter. It enters
  the dispersion relation `ω² = g·k·tanh(k·depth)` and sets which wave
  frequencies are available for a given spatial mode. It also sets the
  denominator in the linearity cap `|η|/depth < 0.1`. It does not
  appear in the optics.

- **Optical throw** (`Tank.projection_distance`): the vertical distance
  from the water surface to the projection plane. It sets how strongly
  surface slopes deflect rays: a slope `∂η/∂x` deflects a ray by
  `throw · (∂η/∂x) / n_water` in the transverse direction. It does
  not appear in the wave physics.

These two scales are fully decoupled and must be chosen on separate
grounds. Confusing them is the most common source of incorrect
intuition about the system.

---

## Spatial resolution: modes and feature size

The wave surface is expressed in the cosine eigenmode basis:

    φ_{m,n}(x,y) = cos(m·π·x/Lx) · cos(n·π·y/Ly)

Mode `(m,n)` has spatial period `2Lx/m` in x and `2Ly/n` in y. With
`n_modes` modes kept in each direction, the finest representable
feature has size roughly `L/n_modes`.

The target analysis step (`analyze_target`) identifies which modes
carry 95 % (or user-specified fraction) of the target's spatial energy
and recommends an `n_modes` accordingly. This is the right starting
point for mode selection: using too few modes means the optimizer
cannot express the target; using too many wastes actuators and
parameters on modes that carry no energy.

**The mode count is set by the target, not by the apparatus.** If the
target has features of size `d`, you need at least `n_modes ≈ L/d`
modes to represent them. The apparatus must then be designed to support
that mode count — not the other way around.

---

## Actuator layout

The coupling matrix `C[j,i]` evaluates mode `j` at actuator `i`'s
position. The system can only excite modes that are in the column span
of `C`: if an actuator is at a node of mode `(m,n)`, it cannot drive
that mode.

### What the validation experiment showed

A layout sweep over 13 random perimeter layouts with `n_act ∈
{6,8,10,12,16,20}` — varying both count and arclength positions — and
two target types found:

- **Final optimized quality is essentially the same across all
  reasonable layouts.** For easy targets (smooth Gaussian), the
  optimizer compensates for layout differences entirely: warm-start
  scores varied by 0.27 across layouts, but final scores collapsed to a
  ~0.009-wide band. For harder targets (4 asymmetric tight spots), all
  layouts started from a similar warm-start and the optimizer
  improved each roughly equally.

- **The warm-start cosine similarity is not a reliable proxy for final
  quality.** It is layout-invariant for hard targets (all layouts look
  equally good at the warm-start step) and anti-correlated with final
  quality for easy targets (the optimizer preferentially rescues
  lower-scoring starting points).

The practical conclusion: **any reasonable perimeter layout works**,
where "reasonable" means actuators are spread around all four sides
with `n_act_per_side ≈ n_modes`. There is no benefit to optimizing
actuator positions within that constraint in the linear regime.

### Rules of thumb

- Minimum `n_act_per_side ≈ max(m_max, n_max) + 1`, where `m_max` and
  `n_max` are the highest mode indices the target needs. `analyze_target`
  computes these directly.
- Uniform perimeter spacing is sufficient. Non-uniform layouts add
  complexity without measurable benefit.
- More actuators than the minimum is never harmful, only adds
  parameters. The optimizer handles the redundancy via the lstsq
  projection.

---

## Driving frequencies

Each mode `(m,n)` has a natural frequency set by the dispersion
relation. To excite mode `(m,n)` in steady state, the actuators must
be driven at (or near) its natural frequency. The resonance condition
is what makes the linear transfer matrix `H[j,k] = 1/(ω_j² - Ω_k² +
2iγω_jΩ_k)` large near `Ω_k ≈ ω_j`.

`analyze_target` identifies the natural frequencies of the important
modes and recommends a frequency range `[f_min, f_max]` and count
`n_freq`. The rule of thumb is `n_freq ≈ 2(f_max - f_min) + 1`.

Key constraint: **water depth sets the available frequency range**.
Deep water compresses the dispersion curve (all modes have similar
frequencies); shallow water spreads it. If the target requires modes
with very different natural frequencies, a deeper tank may make them
harder to drive simultaneously (the frequency spread narrows). For
typical caustic targets the effect is small, but it matters when
designing for a specific driving frequency range.

---

## Throw and the linearity constraint

### Why throw cancels out of caustic quality

The Poisson inversion maps target contrast to modal amplitudes:

    a_{m,n} = c_{m,n} · n_water / (throw · k²_{m,n})

so `a ∝ 1/throw`. The caustic intensity deviation is:

    δI = (throw / n_water) · ∇²η  ∝  throw · (1/throw)  =  constant

Throw cancels exactly. The caustic image produced by the Poisson
inversion is throw-invariant: any throw above the minimum feasible
value reproduces the target equally well. This means throw is **not**
a quality parameter — it is purely a feasibility parameter.

### The linearity caps

The linear-wave forward model is valid only when:

    |η| / depth  <  0.1          (free-surface linearisation)
    |∇η|         <  0.1          (paraxial refraction)

Since `a ∝ 1/throw`, both surface amplitude and slope scale as
`1/throw`. The minimum feasible throw is therefore:

    throw_min = max(
        throw_current · desired_|η|/depth / 0.1,
        throw_current · desired_max|∇η|  / 0.1,
    )

This is a closed-form calculation from the target's modal content —
no search required. It is what `feasibility_report` returns as
`recommended_projection_distance_m`.

### Apparatus sizing

Because the minimum feasible throw is target-dependent, apparatus
sizing follows directly from the hardest target in the intended library:

    throw_apparatus = max over target library of throw_min(target)

The targets that drive throw up are those with the finest features
and highest contrast: tight caustic spots or sharp edges. A rough
scaling: halving the minimum feature size roughly doubles the required
throw (more modes → steeper slopes → more throw needed to stay linear).

The physical design tradeoff is therefore:

    **finer caustic features  ↔  taller apparatus**

There is no way around this tradeoff; it is a direct consequence of the
paraxial approximation and the linearity caps. The only knobs are the
contrast of the target (lower contrast reduces required amplitudes,
reduces required throw) and the tank size (larger tank spreads modes
out, allowing finer features at the same mode index and slope).

### Glass-bottom alternative for tabletop apparatus

For demanding targets (text, fine detail), `throw_apparatus` can run to
several meters — impractical for a tabletop demo. A glass-bottom design
decouples the optical throw from the water depth: the tank holds a
shallow layer of water above a transparent floor, with an air gap below
the floor extending the optical path to the screen.

Apparatus geometry:

    water depth d_water   ────  shallow (e.g. 10 cm)
    glass bottom          ────  rigid transparent boundary
    air gap d_air         ────  optical throw extension
    screen / floor        ────  caustic display surface

Optical throw is then `d_water + (n_water/n_air)·d_air ≈ d_water + 1.33·d_air`.

**What this gains.** The lever arm for caustic feature formation now
scales with `d_air` instead of `d_water`. A 10 cm water layer with a
5 m air gap behaves *optically* like a ~6.7 m deep tank, while
containing 50× less water. The hydrodynamic actuator problem stays at
the much easier 10 cm scale: lower mode frequencies, less wave inertia,
smaller actuators.

**What it doesn't gain.** The water depth still sets the linear-wave
amplitude cap `|η|/depth < 0.1`. At `d_water = 10 cm`, this is
`|η| < 1 cm`, vs `|η| < 50 cm` at `d_water = 5 m`. This binds the
*low-spatial-frequency* modes (those with `k < 1/d_water`), whose
slope-cap amplitudes `s_max/k` would otherwise exceed the η-cap.
Concretely, modes with k ≲ 10 lose roughly half their available
amplitude at d_water=10 cm vs at d_water=5 m. Low-k modes carry the
smooth/broad spatial information in the surface — they're what makes a
caustic look like a face rather than noise. The effect:

- **Sharp targets** (text, edges, fine spots): glass-bottom shallow
  matches deep-water quality at the same throw to within a few percent.
  The dominant modes are high-k, which aren't η-cap-limited.
- **Smooth targets** (face contours, blob silhouettes): glass-bottom
  shallow loses ~10-20% on cosine vs equivalent deep water. Low-k
  modes are amplitude-starved.

A secondary effect: shallow water changes the dispersion relation for
low-k modes (`ω² = gk·tanh(kd)` deviates from deep-water `ω² = gk` when
`kd < 1`). This shifts which driving frequencies resonantly excite
those modes. Manageable as long as the driving-frequency grid is dense
enough to cover both regimes — but it's a real consideration if you
were targeting a narrow band of driving frequencies tuned to deep-water
resonances.

**When to use which.** Two clean recipes:

| Target class           | Recommended apparatus                    |
|------------------------|------------------------------------------|
| Sharp / high-frequency | Glass-bottom shallow + long air gap     |
| Smooth / low-frequency | Deep water at matched depth (no glass)  |
| Mixed                  | Deep water for honesty, or glass bottom and accept the smooth-target degradation in exchange for portability |

In every case, the fully clean reading of "what's achievable" is given
by the joint constraint `s_max · k_max · throw / n_water` (see
`docs/reachability_and_capacity.md`) — glass bottom buys you throw at
the cost of low-k amplitude, deep water buys you both at the cost of
apparatus size.

### Using `feasibility_report` for design

Running `feasibility_report` on a candidate target tells you:

- `recommended_projection_distance_m`: minimum throw your apparatus
  needs to reproduce this target in the linear regime.
- `resolution_limit_m`: finest feature reproducible at the current
  throw (`throw · SLOPE_LIMIT / n_water`).
- `target_achievable_in_linear_regime`: whether the target is
  intrinsically feasible (independent of actuators).
- `actuator_subspace_sufficient`: whether the current actuator layout
  can reach the required modes.

For apparatus design, the first two are the key numbers. For a given
target library, iterate through the hardest examples and read off the
maximum `recommended_projection_distance_m` — that is your minimum
apparatus height.

---

## Summary: the constraint flow

Starting from the target image, every apparatus parameter is
determined by a chain of one-way dependencies:

```
Target features (size d, contrast c)
    ↓
n_modes = L / d                           (mode count from resolution)
    ↓
n_act_per_side ≈ n_modes                  (actuators from modes)
    ↓
f_min, f_max from dispersion relation     (frequencies from modes + depth)
    ↓
throw_min from linearity caps             (throw from amplitude of modes)
```

None of these steps requires numerical optimisation — they are all
analytical calculations from the target's spatial content. The
`analyze_target` and `feasibility_report` functions implement this
chain. `setup_from_target` runs it end-to-end and returns a
ready-to-optimize configuration.

The optimizer (`optimize_caustic`) then refines the actuator phasors
within the fixed apparatus defined by this chain. Its job is to
improve the match between the achieved caustic and the target; the
apparatus design choices above are all made before the optimizer runs.

---

## The linear regime boundary

Everything above assumes validity of the linear wave model. The
0.1 caps are conservative; the regime starts breaking down visibly
around slopes of 0.3–0.4. Near the boundary, nonlinear effects
(harmonic generation, Stokes-like crest steepening) produce sharper,
brighter caustic features than the linear model can achieve — the most
visually interesting caustic structure lives precisely in this regime.

See `docs/nonlinear_extension.md` for a sketch of how the HOS
(Higher-Order Spectral) method extends this framework to finite-
amplitude nonlinear waves, and why the linear warm start remains
applicable there as the M=1 base of the perturbation expansion.
