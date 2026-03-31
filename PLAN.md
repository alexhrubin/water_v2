# Plan: Sharper Caustics via Frequency-Domain Physics

## Context

The caustic images are blurry and the optimization often stalls. Three root causes:

1. **The time-domain convolution is slow and unnecessary.** Actuators are sinusoidal, so the steady-state modal response has a closed-form solution. The current approach discretizes time (2001 steps), builds a huge G matrix, and convolves — ~80M FLOPs per evaluation. The exact steady-state answer costs ~600K FLOPs.

2. **The dispersion relation is wrong.** The code uses shallow-water `ω = c|k|` but kd reaches 7.5 for high modes. The correct `ω² = gk tanh(kd)` changes modal frequencies significantly, meaning the optimizer is driving modes at the wrong resonances.

3. **Dense Φ matrices block scaling.** At 50 modes/axis with a 200×200 grid, the three basis matrices (Φ, dΦ_dx, dΦ_dy) cost 2.4 GB. But the modes are separable cosines — a 2D DCT replaces all three matrices with O(N log N) transforms and zero storage.

Fixing these three things enables: higher resolution, more modes, more optimizer iterations, and correct physics — all of which produce sharper images. The steady-state formulation also naturally gives **stable repeating patterns** (single-frequency mode).

## Key Physics Insight: Linearity of the Steady-State

With complex phasor parameterization P_ik = X_ik + i Y_ik (real/imag parts as free parameters), the steady-state modal amplitudes are:

```
a_j(T) = Im[ Σ_k  H_j(Ω_k) · (Σ_i C_ji P_ik) · e^{iΩ_k T} ]
```

where H_j(Ω) = 1/(ω_j² - Ω² + 2iγω_jΩ) is the transfer function.

This is **linear in (X, Y)**. So a(T) = M · p for a precomputable matrix M. The entire nonlinearity of the optimization lives in the caustic optics (ray tracing), not the wave physics. This dramatically simplifies the gradient and improves the optimization landscape.

---

## Phase 1: Frequency-Domain Wave Physics

**Files:** `src/WaveTank.jl`

### 1a. Correct dispersion relation
One-line fix in `build_propagator`:
```julia
# Before:
c = sqrt(g * depth)
ω = [c * sqrt((m*π/Lx)^2 + (n*π/Ly)^2) for ...]
# After:
k = [sqrt((m*π/Lx)^2 + (n*π/Ly)^2) for ...]
ω = [sqrt(g * k_j * tanh(k_j * depth)) for ...]
```

### 1b. Transfer function and steady-state amplitudes
New functions:
- `transfer_matrix(ω_modes, Ω_drive, γ)` → complex `H` [n_modes × n_freq]
- `steady_state_amplitudes(prop, P_complex, Ω_freqs, T)` → real `a` [n_modes]

Core computation (~5 lines):
```julia
H = 1 ./ (ω.^2 .- Ω'.^2 .+ 2im .* γ .* ω .* Ω')   # [n_modes × n_freq]
α = H .* (C * P)                                        # [n_modes × n_freq]
E = exp.(im .* Ω .* T)                                  # [n_freq]
a = imag(α * E)                                          # [n_modes]
```

### 1c. Complex phasor parameterization
Replace (A, φ) with (Re(P), Im(P)). New helpers:
- `pack_complex(X, Y)` → flat real vector
- `unpack_complex(params, n_act, n_freq)` → X, Y matrices
- `complex_phasors(X, Y)` → P = X + im*Y

This makes the full pipeline `params → a` linear, eliminating sin/cos nonlinearity from the phase.

### 1d. Updated loss factory
New `make_caustic_loss_ss` (steady-state variant):
```julia
function make_caustic_loss_ss(prop, T, target, Ω_freqs; ...)
    H = transfer_matrix(prop.ω, Ω_freqs, γ)
    # ... closure using steady_state_amplitudes
end
```

The `_refining` variant with Ref{Float64} sigmas also gets a steady-state version.

### 1e. Single-frequency mode
When Ω_freqs has one entry, the caustic repeats every 2π/Ω seconds.
- Optimize for the pattern at a chosen phase T₀
- Visualize: the caustic oscillates, showing the target image twice per cycle
- Physical realization: strobe light at frequency Ω to freeze the pattern

No special code needed — just use `Ω_freqs = [Ω]` with the same API.

### Validation
Run both time-domain (at T >> 1/(γ·ω_min) so transients decay) and steady-state on identical parameters. Modal amplitudes should agree within ~1%.

---

## Phase 2: DCT-Based Spatial Reconstruction

**Files:** `src/WaveTank.jl`, add `FFTW` to Project.toml

Replace dense Φ matrices with DCT operations. The eigenmodes cos(mπx/Lx)cos(nπy/Ly) on the endpoint-inclusive grid are exactly the DCT-I (FFTW REDFT00) basis.

### 2a. Surface and derivatives via DCT
```julia
function surface_from_modes(a_2d, nx, ny)
    # a_2d[m+1, n+1] are modal amplitudes, zero-padded to (nx, ny)
    # FFTW.r2r(padded, FFTW.REDFT00) with normalization
end
```

Derivatives: multiply modal coefficients by wavenumber before transform.
- dη/dx: coefficients × (-mπ/Lx), then DST-I in x, DCT-I in y
- dη/dy: coefficients × (-nπ/Ly), then DCT-I in x, DST-I in y

### 2b. Second derivatives (Hessian) for Phase 3
- d²η/dx²: coefficients × (mπ/Lx)², then DCT-I in both (same basis, just scaled)
- d²η/dy²: coefficients × (nπ/Ly)², DCT-I in both
- d²η/dxdy: coefficients × (mπ/Lx)(nπ/Ly), DST-I in both

These are needed for the Jacobian-based rendering in Phase 3.

### 2c. Refactored Propagator
Add a `use_dct::Bool` flag (defaulting to `true` for large problems). When true, skip precomputing dense Φ/dΦ_dx/dΦ_dy. The `caustic_image` function dispatches to DCT-based reconstruction.

Keep the dense-matrix path for small problems and backward compatibility.

### 2d. Zygote adjoints for DCT
DCT is linear, so its adjoint is the transpose transform (DCT-I is self-adjoint up to scaling). Define:
```julia
Zygote.@adjoint function my_dct(x)
    y = dct_forward(x)
    back(Δ) = (dct_adjoint(Δ),)
    return y, back
end
```

### Impact
- Memory: O(n_modes) instead of O(n_modes × grid_points)
- 100 modes/axis + 400×400 grid: ~0 MB for basis (vs 38 GB dense)
- Enables sharp images via fine grids and many modes

---

## Phase 3: Enhanced Caustic Optics

**Files:** `src/WaveTank.jl`

### 3a. Full Snell's law refraction
Replace paraxial approximation:
```julia
# Before (paraxial):
x_land = x + (d - η) * ∂η/∂x / n_water

# After (full vector Snell's law):
# Surface normal: n̂ = normalize(-∂η/∂x, -∂η/∂y, 1)
# Incident ray: d̂_i = (0, 0, -1)
# Snell: sin(θ_t) = sin(θ_i) / n_water
# Trace refracted ray to z = 0
```

For small slopes these agree; for large slopes (strong caustics) the full version gives stronger deflection and sharper focusing.

### 3b. Jacobian-based intensity (alternative renderer)
With surface Hessian available from Phase 2b, compute:
```
det(J) = (1 + r(d-η)η_xx - rη_x²)(1 + r(d-η)η_yy - rη_y²)
       - (r(d-η)η_xy - rη_xη_y)²
```
where r = 1/n_water.

Intensity: I(x,y) = 1 / (|det(J)| + ε)

This gives analytically sharp caustic lines wherever det(J) → 0 — no splatting or blur needed. Use for:
- High-quality final visualization
- Diagnostic: overlay caustic line locations
- Optional loss term

Keep splatting+blur for optimization (smoother loss landscape), but render final results with Jacobian.

---

## Phase 4: Analytical Gradient (if Zygote struggles with DCT)

Since a(T) = M·p (linear), the gradient through wave physics is just Mᵀ.
The full VJP chain:
```
dL/dp = Mᵀ · (DCTᵀ · (dI/d∇η · dL/dI))
```

Each step:
1. dL/dI: cosine similarity gradient — closed form
2. dI → d(landing positions): adjoint of splat + blur
3. d(landing) → d(η, ∇η): elementwise chain rule through Snell's law
4. d(η, ∇η) → d(a): transpose DCT (= forward DCT, self-adjoint)
5. d(a) → d(params): Mᵀ (precomputed)

Implement only if Zygote can't differentiate through FFTW's r2r transforms, or if profiling shows AD overhead is significant.

---

## Phase 5: Updated Example Notebook

1. **Stable caustic demo**: single-frequency, optimize, show repeating pattern
2. **Sharp caustic comparison**: old (blurry splat) vs new (Jacobian) rendering
3. **High-resolution**: 200×200+ grid, 50+ modes/axis, recognizable face image
4. **Animation**: caustic evolving through one period of the driving frequency

---

## Implementation Order

```
1a (dispersion fix)  ──── trivial, do first
1b (steady-state)    ──── core change
1c (complex params)  ──── depends on 1b
1d (loss factory)    ──── depends on 1b, 1c
                          ** TEST & VALIDATE against time-domain **
1e (single-freq)     ──── falls out of 1b-1d, just notebook
2a-2d (DCT)          ──── independent module, do after 1
                          ** TEST: compare DCT vs dense Φ **
3a (full Snell)      ──── independent, can do anytime
3b (Jacobian)        ──── depends on 2b (Hessian)
4 (analytical grad)  ──── depends on 1+2, do if needed
5 (notebook)         ──── incremental after each phase
```

Start with Phase 1 (biggest bang: correct physics + 100x speed). Then Phase 2 (unlock resolution). Then Phase 3 (sharp rendering). Phase 4 only if needed.

## Verification

1. **Steady-state vs transient**: identical modal amplitudes at large T (within ~1%)
2. **DCT vs dense Φ**: surface values match to machine precision
3. **Full Snell vs paraxial**: agree at small slopes, diverge at large slopes
4. **Gradient**: finite-difference check on small problem (as in test_gradient.jl)
5. **Visual**: optimize same target with old vs new pipeline, compare image quality



<!-- ❯ So far we've discussed improvements to the existing sinusoidal actuator definitions. In the plan you pointed out we can naturally achieve repeating caustic
images, which makes sense because the actuators are periodic so the image is too. Improving the caustic image sharpness is great, but I am curious about the
plausibility of achieving a sequence of images -- in other words, playing a short movie in the caustic image. Intuitively it seems like an overconstrained problem,
 but is there any possibility this could be achieved?

⏺ Your intuition is right that it's overconstrained, but it's worth being precise about how overconstrained, because there's a feasibility window.

  The temporal DOF of the system

  In steady state, each modal amplitude is itself a sum of sinusoids at the driving frequencies:

  a_j(t) = Σ_k |β_jk| sin(Ω_k t + ψ_jk)

  So the surface shape over time is a bandlimited periodic signal. With K driving frequencies (harmonics of a base Ω₀), the surface traces a fixed loop in "shape
  space" with period 2π/Ω₀. The number of temporally independent surface configurations is ~2K.

  The total degrees of freedom in the system are:

  DOF = n_controllable_modes × 2K

  where n_controllable_modes is limited by actuator count (~a few hundred with 64 actuators). For a movie with F frames at an effective spatial resolution of R
  pixels each:

  Constraints = F × R

  Running the numbers:

  ┌────────────────────────────────┬───────────────────────┬──────────────────┬───────────┐
  │            Scenario            │          DOF          │   Constraints    │ Feasible? │
  ├────────────────────────────────┼───────────────────────┼──────────────────┼───────────┤
  │ 1 sharp frame, 30×30 effective │ 300 modes × 30 = 9000 │ 1 × 900 = 900    │ Yes, very │
  ├────────────────────────────────┼───────────────────────┼──────────────────┼───────────┤
  │ 5 frames, 20×20 each           │ 300 × 30 = 9000       │ 5 × 400 = 2000   │ Yes       │
  ├────────────────────────────────┼───────────────────────┼──────────────────┼───────────┤
  │ 15 frames, 20×20 each          │ 300 × 30 = 9000       │ 15 × 400 = 6000  │ Marginal  │
  ├────────────────────────────────┼───────────────────────┼──────────────────┼───────────┤
  │ 5 frames, 50×50 each           │ 300 × 30 = 9000       │ 5 × 2500 = 12500 │ No        │
  └────────────────────────────────┴───────────────────────┴──────────────────┴───────────┘

  So there's a spatial-resolution / temporal-variety tradeoff. The total information capacity of the actuator array is fixed. You can spend it on one sharp image,
  or several blurrier ones. This is actually a nice result — it's a clean information-theoretic constraint, not a hard impossibility.

  What a "movie" would actually look like

  With harmonic frequencies Ω₀, 2Ω₀, ..., KΩ₀, the caustic loops with period 2π/Ω₀. Within one loop, the optimizer could target F different images at times T₁, ...,
   T_F. The loss would just be:

  L = Σ_f  loss(I(T_f), target_f)

  Between target times, the caustic smoothly interpolates (it's a bandlimited signal). So you don't get arbitrary frame-to-frame jumps — the movie must be "smooth"
  in some sense. Simple motions work well: a shape that translates, rotates, scales, or morphs between two similar images. A face that blinks or a shape that pulses
   would be achievable. A sequence of unrelated images would not.

  The implementation is trivial once Phase 1 is done. The steady-state formulation evaluates the caustic at any time T cheaply. A multi-frame loss is just a loop
  over target times. The only question is whether the optimizer can find good solutions in the more constrained landscape.

  Bottom line: A short loop (2-5 frames) of moderate-resolution images depicting smooth motion is plausible and would be very compelling. It's worth trying once
  Phase 1 is working — the code change is minimal (loop over T values in the loss), and even partial success (recognizable but imperfect frames) would be striking. -->
