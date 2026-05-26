# Physics to Pictures: the constraint chain

This document threads together the physics-first narrative for the writeup.
Four sections, each more concrete than the last:

1. **What a caustic is** — the geometric-optics object we're trying to control
2. **What a target demands** — translating an image into a curvature requirement
3. **What an apparatus delivers** — translating physics + geometry into a curvature budget
4. **Linear vs HOS vs fantasy** — empirical confirmation that the budget predicts the result

Each section grounds the next. Together they form an end-to-end account where
the apparatus's empirical performance is a *prediction* of the physics, not
just an observation.

---

## 1. What a caustic is

A caustic is **the locus on the floor where the ray map's Jacobian is
singular**. Sunlight enters the tank as approximately-parallel vertical
rays; at each surface point a ray refracts (Snell) and travels through
the water to land on the floor. The map `r_surface → r_floor` has a
2×2 Jacobian `J`. By change of variables, intensity on the floor is
`1/|det(J)|`. Caustics are the set `{ det(J) = 0 }` — the geometric
limit where rays collapse onto a degenerate set, mathematically giving
infinite intensity (regularized in practice by render blur, sun's
angular size, etc.).

**Caustics form via a focal condition.** For the paraxial Snell map
`J = I - (throw/n) · H` where `H` is the surface Hessian, caustics
appear wherever a Hessian eigenvalue equals `n/throw`. Since the
Hessian varies across the surface, caustics appear along **1D contour
lines** rather than at isolated points — the contour where local
curvature happens to match the focal threshold.

**Brightness is set by how strongly `det(J)` passes through zero.**
For a surface with available curvature `H_max` above the focal
threshold `H_crit = n/throw`, caustic peak intensity scales as
`√(H_max - H_crit)`. Big curvature excess → sharp bright lines;
just-above-critical → soft glowing regions.

Full math: `docs/caustic_math.md`.

---

## 2. What a target demands

A target image makes two distinct demands on the caustic:

**(a) Spatial-frequency content.** The image's brightness pattern
decomposes into cosine modes. The smallest representable spatial
feature corresponds to the highest wavenumber `k` with significant
energy in the target's cosine expansion. Targets vary widely:

| Target type                | Energy bands     | Required `k_max` |
|---------------------------|------------------|------------------|
| Single smooth Gaussian spot| k < 10           | small            |
| 3-spot Gaussian (our test) | k < 20           | moderate         |
| Face / silhouette photo    | k < 30           | moderate         |
| Sharp text (ANNA, HELLO)   | k > 50, broad    | high             |

Per `docs/image_generation_constraints.md`, you can't represent
features smaller than `Lx / n_modes` — so if the target needs `k = 60`
on a 1m tank, you need at least `n_modes ≈ 20` *just to represent*
the target, before you even ask whether you can produce a caustic
matching it.

**(b) Brightness contrast.** The target specifies how bright the
bright spots should be relative to the background. A target with a
20× brightness ratio demands a corresponding `H_excess` from the
apparatus (Section 3) — the apparatus has to *exceed* the focal
threshold by enough that the caustic peak brightness matches the
target's bright-spot intensity.

Quantitatively, with peak intensity ~ √(H_excess), and the
geometric reach-disk argument (`docs/reachability_and_capacity.md`,
"Geometric contrast bound") setting an upper limit on contrast at
given throw, a target with bright spots of width `w` and contrast
ratio `C` is reproducible only if:

    H_excess  ≥  C² · H_crit            (rough)

So the target's contrast requirement maps directly to required
curvature excess. High-contrast targets are harder.

**The two demands compound.** A high-frequency target *also* tends
to be a high-contrast target (sharp edges = big brightness jumps
over small spatial distances). So in practice the harder targets
(text, fine details) demand high `k_max_needed` AND high contrast.
Both routes load up the curvature requirements.

The `notebooks/predict_target.py` script automates step (a) for any
target — it returns the `k_max` needed to capture a chosen fraction
(default 95%) of the target's energy.

---

## 3. What an apparatus delivers

The apparatus contributes three numbers:

  - `k_max_available` from the modal basis: `π·√2·n_modes/Lx`
  - `s_max` from the physics regime: linear caps at 0.10; HOS M=2 at
    ~0.30; HOS M=3 at ~0.40; fantasy (no model) unbounded
  - `throw` from depth (no glass) or depth + air gap (with glass)

These combine into the **maximum available curvature** at any single mode:

    H_available  =  k_max_available · s_max

and the **focal threshold** that must be cleared for caustics to form
at all (from full paraxial Snell, derived in `caustic_math.md`):

    H_required   =  n_water / ((n_water − 1) · throw)  ≈  4/throw

The cleanest one-line summary of "can this apparatus produce caustics?":

    H_excess  =  H_available  −  H_required

with three regimes:

  - **`H_excess ≤ 0`**: caustics cannot form. Surface curvature
    insufficient at any reachable mode. Reachable at this throw only
    by extending physics (HOS) or extending geometry (deeper / glass
    bottom + longer air gap).
  - **`H_excess > 0` but small**: caustics form, but softly. Peak
    brightness ~ √H_excess, so soft glowing regions rather than
    sharp lines.
  - **`H_excess >> H_required`**: caustics form sharply. Bright tight
    filaments.

The same formula tells us how each lever moves the result:

| Lever | Effect on `H_excess` |
|---|---|
| Deeper water (with no glass) | raises `throw` → lowers `H_required` → more excess |
| Glass bottom + air gap       | same, with smaller water volume |
| More modes (higher n_modes)  | raises `k_max_available` → more excess |
| More actuators / freqs       | raises *reachable* `k_max` if not already saturated |
| HOS M=2 (vs linear)          | raises `s_max` from 0.1 to 0.3 → 3× more excess |
| HOS M=3                      | raises `s_max` to 0.4 → 4× more excess |

The empirical work in this project has tested most of these:
  - Mode count: 15 → 30 (tiny gain; saturated apparatus rank)
  - Actuator count / freq range: 12-side / 5Hz → 20-side / 10Hz (tiny gain)
  - Depth: 0.12m → 2-5m (large gain — this was the main physical lever)
  - HOS M=2: in progress (substantial gain — see Section 4)

### The mode-count threshold

Substituting `a ≤ s_max/q` (linearity constraint) and `q_n = nπ/L`
(modal basis) into the focal condition `a·q²·throw ≥ n/(n−1)` gives
the **minimum mode index per axis to form any caustic at all**:

    n_min  ≈  k_phys · (L / throw)

with `k_phys = 4/(π·s_max·(1−1/n_water))`. For water (`n=1.33`):

  - Linear regime (`s_max = 0.1`): `k_phys ≈ 13` → `n_min ≈ 13·(L/throw)`
  - HOS M=2  (`s_max = 0.3`):     `k_phys ≈ 4.3` → `n_min ≈ 4·(L/throw)`
  - HOS M=3  (`s_max = 0.4`):     `k_phys ≈ 3.2` → `n_min ≈ 3·(L/throw)`

A clean dimensionless one-liner: caustic formation requires
`n_modes ≥ k_phys · (L/throw)`. The deeper the apparatus relative
to its width (small L/throw), the fewer modes you need. The shallower
the apparatus (large L/throw), the more high-`q` modes you need to
clear the threshold. HOS reduces the requirement by 3-4×.

This is why pools and puddles (`L/throw = 10–50`) need real high-`q`
content to produce caustics, why our `d=2m` tank (`L/throw = 0.5`)
works easily even in linear regime with 15 modes, and why our `d=0.1m`
tank (`L/throw = 10`) needs ≥130 modes in linear or ≥43 in HOS M=2 —
which we don't currently have in our basis.

See `docs/caustic_math.md` ("The complete derivation") for the full
chain from `det(J)=0` to the mode-count threshold, including
empirical confirmation on our actual runs.

---

## 4. Linear vs HOS vs fantasy: prediction and measurement

The constraint chain in Sections 1-3 makes a quantitative prediction
for every target × apparatus × physics-regime combination: compute
`H_excess`, compute `brightness_factor = √(H_excess/H_required)`,
map to predicted cos via a calibrated relationship.

Empirically (from this project's sweeps at depth=2m for synthetic
targets, depth=5m for image targets):

| Target | predicted (linear) | measured (linear) | predicted (HOS M=2) | measured (HOS M=2) | measured (fantasy) |
|---|---|---|---|---|---|
| 3spot_gaussian | 0.65 | **0.625** | 0.86 | **0.835** | 0.954 |
| sine_wave      | 0.65 | **0.672** | 0.86 | *pending* | 0.901 |
| recidiviz_logo | 0.74 | **0.696** | 0.91 | *pending* | 0.968 |
| head           | 0.74 | **0.737** | 0.91 | *pending* | 0.907 |
| dog            | 0.78 | *pending* | 0.92 | *pending* | n/a |

Predictions are based purely on apparatus + physics, no optimization.
Empirical values land within ~5% of predictions on the 3spot case;
similar agreement expected on the rest pending the HOS sweep.

**The arc that this constraint chain establishes:**

1. **Linear cap-respecting** is the honest physical baseline. Soft
   recognizable caustics. ~0.6-0.7 cos.
2. **HOS M=2** is the natural physics extension. Lifts `s_max` 3×,
   produces ~0.85 cos *while staying in physically valid surface
   configurations*.
3. **Fantasy** (linear math with no validity enforcement, as in
   `example.ipynb`) gives ~0.95 cos but at slopes of 14 — water
   configurations that violate the model's own assumptions by 100×.
   Visually beautiful, physically impossible.

The middle row is the most important contribution: it shows that
recognizable, sharp, physically-realistic caustics ARE possible on a
sensibly-sized tabletop apparatus *if the physics model is honest
about its own validity envelope*. Most ML-for-physics work in this
space conflates rows 2 and 3 by silently optimizing past the model's
validity (because the optimizer doesn't know better and the model
doesn't refuse). Our framework makes the distinction explicit and
shows the cost.

---

## What this writeup section enables

A reader who follows the four sections above can:

  - Pick up any new target and predict, from first principles, what
    apparatus is needed to reproduce it
  - Understand *why* some targets are easy and some are hard, not as
    a hand-wavy intuition but as a derivable budget
  - Compare apparatus design choices (depth vs glass bottom vs HOS
    vs more modes) on the same footing — the curvature-excess
    framework subsumes them all
  - Predict the impact of any future improvement (e.g. "what would
    HOS M=3 give us?") without re-running the optimization

That's a substantially stronger position than "we trained a model
and got this number on this dataset." It's the difference between
demonstrating a result and explaining one.

---

## How the existing docs slot in

| Doc                                    | Role in this narrative                |
|----------------------------------------|---------------------------------------|
| `caustic_math.md`                      | Section 1 — full math reference       |
| (the `predict_target.py` script)       | Section 2 — automates the per-target k_max derivation |
| `image_generation_constraints.md`      | Section 3 — apparatus design knobs (depth, throw, modes, actuators) |
| `reachability_and_capacity.md`         | Section 3 — info-theoretic frame for the same constraints, plus the reach-disk contrast bound |
| `nonlinear_extension.md` + `hos_math.md` | Section 3-4 — HOS as the next physics layer |
| `my_understanding.md`                  | Glue document — the original narrative arc this is sharpening |

This document is the writeup spine; the others are appendices.
