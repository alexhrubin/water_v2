# Water Caustic Optimizer — Roadmap

## Overview

The existing Julia/Zygote system works and lives on `main`. This branch (`python-rewrite`) develops a JAX-based rewrite that enables:

1. **Analytical gradient** — eliminate Zygote tape overhead, enable L-BFGS
2. **GPU acceleration** — JAX compiles to any XLA backend with no code changes
3. **Learned simulator correction** — NN trained to close the gap between the ideal physics model and a realistic "non-ideal" simulator, as a proxy for real-world deployment on a physical tank

These three directions converge naturally in JAX and were impossible/impractical to pursue cleanly in Julia.

---

## Why JAX

| Need | JAX solution |
|------|-------------|
| Analytical gradient | `jax.custom_vjp` — clean VJP implementation |
| GPU (NVIDIA/Apple Metal) | `jax.jit` compiles to XLA on any backend |
| Batch training | `jax.vmap` — vectorize over targets with no loops |
| Research portfolio | JAX is standard in physics simulation research |
| Replace Zygote | `jax.grad` / `jax.value_and_grad` |

**Development strategy:** Work on CPU JAX locally. Use Google Colab (NVIDIA T4/A100) for neural surrogate training.

---

## Phase 1: JAX Port of Core Physics Simulator

Port the Julia physics pipeline to a clean Python/JAX package.

### Repo structure
```
wavetank/
├── __init__.py
├── physics.py      # Tank dataclass, eigenmodes, transfer_matrix, steady_state_amplitudes
├── render.py       # reconstruct_surface, caustic_image (with custom_vjp in Phase 2)
├── loss.py         # cosine_loss, ssim_loss
├── optimize.py     # optimize_caustic (Adam via optax, L-BFGS via jaxopt)
├── analytical.py   # analytical_solve, setup_from_target
├── nonideal.py     # Non-ideal simulator with physically motivated perturbations (Phase 4)
└── correction.py   # Learned correction NN + corrected forward model (Phase 4)
notebooks/
├── validate.ipynb  # Side-by-side Julia vs JAX validation
└── optimize.ipynb  # Main optimization notebook
```

### Key translations
| Julia | JAX |
|-------|-----|
| `Zygote.gradient` | `jax.grad` / `jax.value_and_grad` |
| `Zygote.@adjoint` | `jax.custom_vjp` |
| `Zygote.ignore()` | not needed (`jit` handles static ops) |
| `scatter_add` + custom adjoint | `jax.ops.segment_sum` (GPU-native, gradient built-in) |
| `Optim.jl` L-BFGS | `jaxopt.LBFGS` |
| Complex arrays | JAX supports natively (`jnp.complex64`) |

### Validation
- Run Julia and JAX side-by-side on identical inputs, compare caustic outputs
- `jax.test_util.check_grads` for automatic gradient validation

---

## Phase 2: Analytical Gradient

The forward map splits into a **linear part** (parameters → modal amplitudes) and a **nonlinear rendering** (modal amplitudes → caustic image):

```
∂L/∂θ = Mᵀ · ∂L/∂a
```

M is the same linear map used in `analytical_solve`. `∂L/∂a` is the rendering adjoint, derived manually.

### Workflow
1. **Write `docs/gradient_derivation.md`** with full math:
   - ∂L/∂I: cosine loss gradient
   - ∂L/∂D: Gaussian blur adjoint (blur is self-adjoint)
   - ∂L/∂(x_land, y_land): bilinear splatting adjoint
   - ∂L/∂(dηdx, dηdy): from paraxial refraction formula
   - ∂L/∂a: linear adjoint of surface reconstruction

2. **Implement** `@jax.custom_vjp` on `caustic_image` in `render.py`

3. **Validate** with `jax.test_util.check_grads` against JAX's own AD

4. **Switch optimizer** to L-BFGS (`jaxopt`) — converges in ~100 steps vs 1500 Adam

### Expected speedup
- 3–5× faster gradient (no tape)
- 10–15× fewer iterations (L-BFGS vs Adam)
- **Combined: ~15–25× total** over current Julia/Adam

---

## Phase 3: GPU Acceleration

Once Phase 1 is working on CPU, GPU requires no code changes:

```python
# Move arrays to GPU device
with jax.default_device(jax.devices("cuda")[0]):  # or "METAL"
    result = optimize_caustic(prop, target, ...)
```

### What GPU unlocks
- **Larger grids** (500×500, 1000×1000) at current wall-clock times
- **Batched neural surrogate training**: `jax.vmap` over 16–64 targets simultaneously
- Local Apple Metal GPU when jax-metal matures; Colab in the meantime

---

## Phase 4: Learned Simulator Correction

### Motivation

A real physical tank deviates from the ideal model in several concrete ways:
- Mode-dependent damping (higher frequencies damp faster than the uniform γ model assumes)
- Actuator nonlinearity and positional jitter (real motors aren't point sources at exact coordinates)
- Slightly perturbed eigenfrequencies (manufacturing tolerances in tank geometry)
- Non-uniform illumination and camera distortion in the rendering

If you optimized phasors against the ideal simulator and drove a real tank with them, the caustic would be wrong. A learned residual model corrects for this.

### Step 1: Build a non-ideal simulator (`nonideal.py`)

Add physically motivated perturbations to create `caustic_image_nonideal(p)`:
- Mode-dependent damping: `γ_j = γ₀ · (1 + α · ω_j)`
- Eigenfrequency jitter: `ω_j → ω_j · (1 + εⱼ)`, εⱼ ~ N(0, σ²)
- Coupling noise: `C[j,i] → C[j,i] · (1 + δⱼᵢ)`, δ ~ N(0, σ²)
- Radial camera distortion applied to the final caustic image

These perturbations are named and physically justified — this isn't noise for noise's sake.

### Step 2: Train an image-correction NN (`correction.py`)

The NN lives entirely in **image space**. Given the ideal caustic, it predicts the correction needed to match the non-ideal (real) output:

```
p → caustic_image_ideal(p) → I_ideal  [differentiable, analytical gradient]
                              I_ideal → NN → ΔI        [learned correction]
                              I_corrected = I_ideal + ΔI ≈ I_real(p)
```

Training data requires no target images — just random phasors:
```python
# Generate pairs: (ideal caustic, non-ideal caustic)
p_batch = sample_random_phasors(batch_size)                  # random actuator configs
I_ideal = jax.vmap(caustic_image_ideal)(p_batch)             # ideal simulator
I_noisy = jax.vmap(caustic_image_nonideal)(p_batch)          # non-ideal simulator
delta_I = I_noisy - I_ideal                                  # residual to learn

# Train NN: I_ideal → ΔI
loss = mse(model.apply(params, I_ideal), delta_I)
```

Architecture: a small U-Net (image → image), since the correction is spatially structured.

### Step 3: Use corrected simulator for optimization

The optimization loop is unchanged except the forward model is now corrected:

```python
def corrected_loss(p, target):
    I_ideal = caustic_image(steady_state_amplitudes(prop, p, Ω_freqs, T_eval))
    I_corrected = I_ideal + correction_model(I_ideal)
    return cosine_loss(I_corrected, target)

# Gradients flow through both: analytical VJP (ideal part) + NN autograd (correction)
p_opt = lbfgs_optimize(corrected_loss, p0, target)
```

### Why this matters for a real tank

On actual hardware, replace `caustic_image_nonideal` with camera frames: feed the real tank a known set of phasors, photograph the resulting caustic, and fine-tune the correction NN on those pairs. The architecture is identical — only the training data source changes.

---

## Stretch Goal: Interactive Demo

A Gradio web app where you upload a target image and watch the optimized caustic form. Enabled by the analytical warm start + L-BFGS convergence (~100 steps). Makes the project accessible to non-technical viewers and is a strong portfolio artifact.

---

## Implementation Order

```
Phase 1: JAX port        → correctness baseline, ecosystem unlocked
    ↓
Phase 2: Analytical grad → math doc first, then custom_vjp, then L-BFGS
    ↓
Phase 3: GPU             → nearly free once Phase 1 is done (Colab for training)
    ↓
Phase 4: Learned correction → non-ideal simulator first, then NN, then corrected optimization
    ↓
Stretch: Gradio demo     → fast enough after Phase 2; can be done anytime after
```

Phases 1 & 2 are local CPU work. Phase 4 training runs on Colab.
