# Reachability and Information Capacity of the Apparatus

A sketch — not a plan, not a TODO. Captures the analysis arc we developed
during the naive-ML / CNN-pipeline experiment, so we can return to it as
a writeup section. Models its style on `nonlinear_extension.md`.

The core idea: characterize the apparatus's information-theoretic
capabilities by decomposing the forward map into a **linear "reachability"
part** and a **nonlinear "expressivity" part**, and computing them
separately. Together they bound what any algorithm can extract from this
apparatus class.

## The decomposition

The forward map:
```
phasors p  →  modal amplitudes a  →  surface (η, ∇η)  →  caustic image I
       |←————— linear part —————→|   |←——— nonlinear part ———→|
```

Specifically:
- `p ∈ ℝ^{2·n_act·n_freq}` — real/imag parts of complex phasors
- `a = M · p` — modal amplitudes, where M packs the H ⊙ C product with
  the `exp(iΩT)` phase into a real-valued matrix
- `(η, ∇η) = DCT(a)` — linear reconstruction in the cosine eigenbasis
- `I = render(η, ∇η)` — paraxial Snell + bilinear splat + Gaussian blur;
  nonlinear

The first three steps are linear (composition of linear maps). The
caustic step is where the nonlinearity enters.

---

## The linear part: reachability via rank(M)

Two distinct subspaces:

- **Representable**: the full mode space `ℝ^{n_modes}` (12² = 144 modes
  in current setup). Any modal-amplitude vector in here can be expressed
  as a surface.
- **Reachable**: the column space of M ⊂ ℝ^{n_modes}. This is what the
  actuator × frequency setup can actually drive.

These differ when M is rank-deficient. The effective rank depends on:

- `n_act` — caps the rank contribution per frequency
- `n_freq` — number of frequency blocks
- H's spectral profile — how broadly each frequency excites modes around
  its resonance (width set by damping γ)

**Algebraic minimum**: `2·n_act·n_freq ≥ n_modes`, giving
`n_freq ≥ n_modes / (2·n_act)` (≈ 3.6 in current apparatus). This is
necessary but not sufficient — it gets you full algebraic rank but not
necessarily full *effective* rank, because nearby frequencies produce
nearly-collinear columns.

**Heuristic saturation point**: `n_freq ≈ n_modes / n_act ≈ 7-8`. Each
frequency contributes ~n_act independent directions concentrated near its
resonance; H profiles overlap when frequencies get denser than this.

**Correct answer**: compute the SVD of M for candidate n_freq values and
find where the singular-value spectrum saturates empirically.

```python
M = build_M_matrix(prop, Omega, T_eval, gamma)
s = np.linalg.svd(M, compute_uv=False)
rank_eff = (s > 1e-3 * s.max()).sum()
```

---

## The nonlinear part: expressivity via the Jacobian

The nonlinear render step maps the reachable surface manifold into
4096-pixel image space. Key properties:

1. **Doesn't expand dimensionality.** A smooth nonlinear map from a
   d-dim manifold gives back a d-dim manifold (generically), just curved
   and embedded in a higher-dim ambient space. The caustic-image
   manifold has the same intrinsic dim as the reachable surface
   manifold (~80 currently, ≤144 saturated). The 4096 pixels are mostly
   correlated noise once you condition on "it's a caustic from this
   apparatus."

2. **Distorts the geometry.** Distances in surface space don't
   correspond linearly to distances in image space. Two close surfaces
   can produce very different caustics, and vice versa.

3. **Redistributes sensitivity non-uniformly across directions.** The
   local Jacobian `J = ∂I/∂a` has large singular values near
   caustic-forming configurations (small `δa` swings rays between focal
   positions) and small ones in flat-flow regions. The same surface
   amplitude carries more bits in caustic-y directions than in smooth
   ones.

4. **Introduces non-uniqueness via folds.** Fold singularities mean
   distinct surfaces can produce nearly-identical caustics. This is the
   *structural* source of the MSE-on-phasors training pathology — the
   model is asked to discriminate between phasor patterns whose
   downstream consequence is identical.

---

## Information capacity bound

Combining linear + nonlinear:

```
I_capacity  ≈  Σ_i log₂( σ_i(J) · dynamic_range / noise_floor )
```

summed over Jacobian singular values, integrated over the reachable
surface manifold.

- **Linear rank** bounds the *count* of non-zero σ_i.
- **Nonlinearity** controls the *distribution* of their magnitudes.
- **Slope-cap and amplitude-cap** from `image_generation_constraints.md`
  bound the dynamic range per direction.
- **Noise floor** comes from rendering blur σ, pixel discretization,
  and (on real hardware) photon shot noise + camera dynamic range.

Total = number of independent directions × bits per direction (set by
Jacobian magnitude × dynamic range).

---

## Geometric contrast bound: the reach-disk argument

The slope cap and throw together set a **geometric** constraint on what
fraction of the water surface can contribute to a given target pixel —
which in turn bounds the achievable contrast at that pixel. This is the
*physical* shadow of the information-capacity bound above, expressed in
length scales rather than bits.

### The reach disk

At slope cap `s_max`, the maximum displacement of a single ray on the
floor is

    R_reach  =  (throw / n_water) · s_max  ≈  0.075 · throw

Reading this backward: a bright target pixel at floor position `r_t` can
only receive rays from a disk of radius `R_reach` around `r_t` on the
surface. Surface points outside this reach disk *cannot* deliver rays to
this pixel within the linear regime — their refracted rays would have to
land further away than the cap allows.

Equivalently: each bright target pixel "draws from" a fixed surface area
`A_reach = π · R_reach²`. Targets with multiple bright spots draw from
the union of their reach disks.

### The achievable-contrast bound

A bright spot of area `A_spot` on the floor can be made to receive at
most all the rays from its reach disk (ideal redistribution), plus the
rays that would already have landed there from a flat surface:

    max_contrast(spot)  ≈  (A_reach / A_spot)  +  1

where contrast = peak intensity / background intensity. The first term
is the ratio of "extra rays donated by the reach disk" to "rays a flat
surface delivers to the spot's own area."

Numerically, for a 4 cm bright spot (area ~13 cm²):

| throw | `R_reach` | `A_reach` | max contrast |
|-------|-----------|-----------|--------------|
| 0.1 m | 7.5 mm   | 24 mm²   | **1.02×** (invisible) |
| 1.0 m | 75 mm    | 0.018 m² | ~14×                  |
| 2.0 m | 150 mm   | 0.071 m² | ~55×                  |
| 5.0 m | 375 mm   | 0.44 m²  | ~340× (saturates apparatus) |

A target asking for 20× contrast on 4 cm bright spots is **not
physically reproducible** at throw < ~1.5 m, regardless of optimizer,
architecture, or training. Past throw ~5 m the bound exceeds typical
target requirements and other constraints (rank, basis match) bind
instead.

### Why the *un*-reachable area matters

Surface points outside any bright spot's reach disk are "wasted" for
contrast purposes — if deformed at all, they redirect rays to dark
regions of the target, *reducing* the rendered match. In the ideal
solution, those points should stay flat.

But the cosine loss doesn't know this. Cosine is scale-invariant
(insensitive to overall image brightness) but **not offset-invariant**:
if the target has bright spots at intensity 20 on background 0 and the
physically-achievable rendering has bright spots at intensity 1.02 on
background 1.0, cosine penalizes the "elevated background" heavily —
even though the *spatial structure* matches perfectly. The optimizer
chases this gradient by deforming unreachable surface regions too,
trying to reduce the rendered background, which it cannot do without
moving rays *somewhere* (and there is nowhere physical for them to go).

The signature of this in our cap-enforced runs: slopes are at-cap
roughly uniformly across the entire surface (e.g., max|∇η| = 0.11 on
the 3-spot Gaussian run), not concentrated in reach disks. The
optimizer has spent its slope budget chasing impossible contrast
instead of concentrating it where it would actually help.

### Implication: a contrast-invariant loss

The geometric bound implies that for any throw + target combination,
there is an **achievable contrast ceiling**. Below that ceiling, the
optimizer should be free to find the right *shape* without being asked
to also match contrast levels that are physically out of reach.

Switching to **Pearson correlation** (mean-subtracted cosine) makes
the loss offset-invariant:

    pearson(T, I)  =  cos(T - mean(T), I - mean(I))

Now "bright spots in the right places, dim above background" scores
the same as "bright spots in the right places, very bright above
background." The optimizer no longer fights the unreachable contrast
gap; it concentrates its slope budget on shape.

Predicted effect: in cap-enforced physical runs, the slope budget
moves out of the unreachable regions and into the reach disks where
it actually contributes. The Pearson score should improve, while the
cosine score may or may not change (cosine measures something the
optimizer is no longer minimizing). Visual quality should improve at
the locations of bright features in the target.

### Open question: per-target throw-vs-feature-size analysis

Given a target, we can compute the achievable-contrast bound by:

1. Identify bright features in the target (threshold + connected
   components).
2. For each bright feature, compute its area `A_spot` and reach disk
   area `A_reach` at the current throw.
3. Compute predicted max contrast per feature.
4. Compare to target's contrast at each feature.
5. Features where target > prediction are "unreachable in contrast";
   their fidelity is structurally bounded.

This would give us a closed-form per-target prediction of "what
fraction of the target's bright features can be achieved at maximum
intensity vs. clipped to physical ceiling." Plotting this against
the achieved cos (cosine vs Pearson) on the physical run would
empirically validate the bound.

---

## Practical analysis plan

Three plots that together characterize the apparatus's information limit:

1. **Effective rank vs n_freq** — SVD of M, sweep n_freq. Picks the
   principled `n_freq*` where adding more frequencies stops adding
   independent directions.

2. **Jacobian singular-value distribution across the manifold** —
   compute `J = ∂I/∂a` at sample points on the reachable manifold via
   `jax.jacrev`. Show histogram of singular values; show how non-
   uniformly information is allocated across directions; identify the
   "caustic-y" vs "smooth" regions.

3. **OOD eval metric vs n_freq** — empirical confirmation that learning
   quality saturates with reachable rank. Plot CLIP cosine (or some
   recognizability metric) on the same target set as n_freq sweeps.

Together these turn the apparatus characterization from "we got cos=X
on these targets" into "the apparatus has ~Y bits of information
capacity, distributed roughly Z across reachable directions, and
learning saturates the capacity at n_freq* ≈ W."

---

## Open questions

- **Integration measure on the reachable manifold**. To go from local
  Jacobian to global capacity number, we need to integrate over the
  reachable manifold. With random-phasor sampling, this is automatically
  weighted by the surface-amplitude distribution induced by sampling. Is
  that the right weighting? (Probably yes for the experiment, maybe not
  for "intrinsic" apparatus capacity.)

- **Noise floor specification**. Rendering blur σ contributes
  blur-pixel-equivalent noise; in real hardware the camera adds more.
  Need to pin this down to get an absolute bit number rather than a
  relative one.

- **Basis match vs rank**. Even at saturated reachable rank, the
  apparatus uses a *cosine* basis. Targets whose natural representation
  is *not* cosine-aligned (e.g., images with sharp edges off the wall
  axes) will be under-represented even with full rank. This deserves
  its own analysis — possibly via projection of target images onto the
  reachable subspace and looking at the residual.

- **The recognizability vs fidelity gap**. Cosine similarity captures
  fidelity (pixel overlap). Human recognition is much more fault-
  tolerant for text-like and silhouette-like content. CLIP cosine or
  similar perceptual metrics would capture this; pixel cosine doesn't.
  The capacity bound is about fidelity; recognition has a different
  bound (entropy of the human's image prior).

---

## How this connects to existing writeup

- `image_generation_constraints.md` — the slope-cap and displacement-
  budget analyses are the *physics-level* expressions of the same
  information bounds. Connecting them gives a unified story.
- `nonlinear_extension.md` — HOS lifts the slope cap from 0.1 to ~0.3,
  raising dynamic range per direction → more bits per direction. The
  reachability analysis here would directly quantify the HOS gain in
  information-theoretic terms.
- `my_understanding.md` — the depth-from-target rule (`d ≈ 30·L_target_min`)
  is a length-scale version of the basis-match question above.

---

## Why this is worth writing up properly

Most ML-for-physics work treats the apparatus as a fixed forward
function and reports "our learned model achieves cosine X on this set
of targets." This is unsatisfying because it doesn't separate "the
algorithm is limited" from "the apparatus is limited." The reachability
+ capacity analysis lets us distinguish these cleanly:

- *Inside* the information bound: choice of architecture, loss, and
  training data matters; better algorithms close the gap to optimal.
- *Outside* the bound: no algorithm can succeed without changing the
  apparatus.

A reader gets a much stronger sense of "this person understands the
physics, the math, AND the machine learning, and knows where each one
binds." Few project writeups in this space have that.

---

## Status

Discussed and parked on 2026-05-24 during the naive-ML / CNN-pipeline
experiment. Revisit when:

- The CNN-with-flatten run has finished and we have OOD numbers in hand
  to compare against the bound
- We're ready to write up the apparatus-design section of the main
  writeup (this analysis is the spine of that section)
- We want to characterize what changes when apparatus parameters change
  (more frequencies, more actuators, deeper water, full Snell, HOS)
