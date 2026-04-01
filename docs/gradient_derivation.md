# Analytical Gradient for Caustic Optimization

## Overview

The optimization minimizes a loss `L(I, T)` between the rendered caustic `I` and a target `T`,
over the real parameter vector `θ = [vec(X); vec(Y)]` where `P = X + iY` is the `[n_act, n_freq]`
complex phasor matrix.

The forward pipeline is:

```
θ  →  P  →  a  →  (η, ηx, ηy)  →  (xl, yl)  →  D  →  I  →  L
     unpack  ssa    reconstruct     refract       splat  blur  loss
```

Each arrow is a function; the adjoint (backward pass) traverses in reverse, propagating
`∂L/∂(·)` from right to left.

---

## Forward pass in detail

### 1. Unpack: `θ → P`

```
θ = [X.ravel(); Y.ravel()]   ∈ ℝ^{2·n_act·n_freq}
X = θ[:n_act·n_freq].reshape(n_act, n_freq)
Y = θ[n_act·n_freq:].reshape(n_act, n_freq)
P = X + iY                   ∈ ℂ^{n_act × n_freq}
```

### 2. Steady-state amplitudes: `P → a`

```
H[j,k]   = 1 / (ω_j² - Ω_k² + 2iγ ω_j Ω_k)       [n_total, n_freq]  complex
β[j,k]   = H[j,k] · exp(iΩ_k T)                     [n_total, n_freq]  complex
α[j,k]   = (C @ P)[j,k]                              [n_total, n_freq]  complex
a[j]     = Im(Σ_k α[j,k] · β[j,k])                  [n_total]          real
```

Expanding `P = X + iY` and `α = C·X + i·C·Y`:

```
a[j] = Σ_k [ (C·X)[j,k] · Im β[j,k]  +  (C·Y)[j,k] · Re β[j,k] ]
```

This is **linear** in `(X, Y)` — the map `θ → a` equals the matrix `M` from `analytical_solve`:

```
M_{j, k·n_act + i}         = C[j,i] · Im β[j,k]
M_{j, n_act·n_freq + k·n_act + i} = C[j,i] · Re β[j,k]

a = M θ
```

### 3. Surface reconstruction: `a → (η, ηx, ηy)`

Modes are indexed by pairs `(m,n)` with `lin_2d[j] = m·n_modes + n`. Scatter `a` into a 2D grid:

```
a_2d[m,n] = a[j]   where j is the index of mode (m,n)
```

Then apply separable matmuls:

```
η[i,j]   = (cos_x  @ a_2d @ cos_y.T)[i,j]     cos_x[i,m]  = cos(mπ·xs[i]/Lx)
ηx[i,j]  = (dcos_x @ a_2d @ cos_y.T)[i,j]     dcos_x[i,m] = −(mπ/Lx)·sin(mπ·xs[i]/Lx)
ηy[i,j]  = (cos_x  @ a_2d @ dcos_y.T)[i,j]    dcos_y[j,n] = −(nπ/Ly)·sin(nπ·ys[j]/Ly)
```

This is **linear** in `a`.

### 4. Paraxial refraction: `(η, ηx, ηy) → (xl, yl)`

```
xl[i,j] = X_src[i,j] + (depth − η[i,j]) · ηx[i,j] / n_water
yl[i,j] = Y_src[i,j] + (depth − η[i,j]) · ηy[i,j] / n_water
```

This is **nonlinear** (product `η · ηx`).

### 5. Bilinear splatting: `(xl, yl) → D`

Each source pixel `(i,j)` contributes to the 4 nearest destination pixels.
Let `fi = (xl − xs[0])/dx`, `fj = (yl − ys[0])/dy`:

```
ix0 = floor(fi),   iy0 = floor(fj)       (integer bin indices)
wx  = fi − ix0,    wy  = fj − iy0        (fractional offsets ∈ [0,1))

D[ix0,   iy0  ] += (1−wx)(1−wy)  · mask[i,j]
D[ix0+1, iy0  ] +=  wx  (1−wy)  · mask[i,j]
D[ix0,   iy0+1] += (1−wx) wy    · mask[i,j]
D[ix0+1, iy0+1] +=  wx   wy    · mask[i,j]
```

where `mask = 1` if the ray lands in bounds, `0` otherwise.

### 6. Gaussian blur: `D → I`

```
I = K * D    (2D separable convolution with Gaussian kernel, zero-padded)
K[p,q] = exp(−(p·dx)²/(2σ²)) · exp(−(q·dy)²/(2σ²))
```

The kernel is **not** normalized (intentional: normalization is left to the loss function).

---

## Backward pass (analytical adjoint)

We are given `∂L/∂I[i,j]` (from the loss function) and want `∂L/∂θ`.

### Step 6 adjoint: `∂L/∂I → ∂L/∂D`

Convolution is self-adjoint under zero-padding (the transpose of a convolution with kernel `K` is
convolution with the flipped kernel; for a symmetric Gaussian, flipping is a no-op):

```
∂L/∂D = K * (∂L/∂I)         (same blur, applied to the loss gradient)
```

### Step 5 adjoint: `∂L/∂D → ∂L/∂(xl), ∂L/∂(yl)`

For source pixel `(i,j)` with bins `(ix0, iy0)` and weights `(wx, wy)`:

```
∂L/∂(xl[i,j]) = (1/dx) · [
    −(∂L/∂D)[ix0,  iy0  ] · (1−wy)
    +(∂L/∂D)[ix0+1,iy0  ] · (1−wy)
    −(∂L/∂D)[ix0,  iy0+1] · wy
    +(∂L/∂D)[ix0+1,iy0+1] · wy
] · mask[i,j]

∂L/∂(yl[i,j]) = (1/dy) · [
    −(∂L/∂D)[ix0,  iy0  ] · (1−wx)
    −(∂L/∂D)[ix0+1,iy0  ] · wx
    +(∂L/∂D)[ix0,  iy0+1] · (1−wx)
    +(∂L/∂D)[ix0+1,iy0+1] · wx
] · mask[i,j]
```

This is a **bilinear gather** of `∂L/∂D` at positions `(fi+1, fj)` vs `(fi, fj)` for the x
component — equivalent to gathering the finite difference of `∂L/∂D` across bins.

More compactly: define `G = ∂L/∂D` gathered bilinearly:

```
∂L/∂(xl) = (1/dx) · mask · bilinear_gather_dx(G, ix0, iy0, wy)
∂L/∂(yl) = (1/dy) · mask · bilinear_gather_dy(G, ix0, iy0, wx)
```

where `bilinear_gather_dx` gathers `G[ix0+1,·] − G[ix0,·]` weighted by `(1−wy, wy)`, and
similarly for `dy`.

### Step 4 adjoint: `∂L/∂(xl), ∂L/∂(yl) → ∂L/∂η, ∂L/∂ηx, ∂L/∂ηy`

From the paraxial formulas:

```
∂xl/∂ηx = (depth − η) / n_water       ∂xl/∂η = −ηx / n_water
∂yl/∂ηy = (depth − η) / n_water       ∂yl/∂η = −ηy / n_water
```

(Cross-terms `∂xl/∂ηy = 0` and `∂yl/∂ηx = 0` under the paraxial approximation.)

Therefore:

```
∂L/∂(ηx[i,j]) = ∂L/∂(xl[i,j]) · (depth − η[i,j]) / n_water
∂L/∂(ηy[i,j]) = ∂L/∂(yl[i,j]) · (depth − η[i,j]) / n_water
∂L/∂(η[i,j])  = −∂L/∂(xl[i,j]) · ηx[i,j] / n_water
               − ∂L/∂(yl[i,j]) · ηy[i,j] / n_water
```

### Step 3 adjoint: `∂L/∂η, ∂L/∂ηx, ∂L/∂ηy → ∂L/∂a`

From the separable matmul structure:

```
∂L/∂a_2d = cos_x.T  @ ∂L/∂η  @ cos_y   (from η  = cos_x  @ a_2d @ cos_y.T)
          + dcos_x.T @ ∂L/∂ηx @ cos_y   (from ηx = dcos_x @ a_2d @ cos_y.T)
          + cos_x.T  @ ∂L/∂ηy @ dcos_y  (from ηy = cos_x  @ a_2d @ dcos_y.T)
```

Then gather back to the flat modal index:

```
∂L/∂a[j] = ∂L/∂a_2d[mode_m[j], mode_n[j]]
```

### Steps 2–1 adjoint: `∂L/∂a → ∂L/∂θ`

Since `a = Mθ` (linear), the adjoint is simply:

```
∂L/∂θ = Mᵀ · ∂L/∂a
```

Expanding:

```
∂L/∂X[i,k] = Σ_j C[j,i] · Im β[j,k] · ∂L/∂a[j]  =  (Cᵀ @ (∂L/∂a · Im β))[i,k]
∂L/∂Y[i,k] = Σ_j C[j,i] · Re β[j,k] · ∂L/∂a[j]  =  (Cᵀ @ (∂L/∂a · Re β))[i,k]
```

where `β[j,k] = H[j,k] · exp(iΩ_k T)` is precomputed in the forward pass.

---

## Implementation plan (`caustic_image_vjp`)

```python
@jax.custom_vjp
def caustic_image(prop, a, *, n_water, sigma, ...):
    # same forward computation

def caustic_image_fwd(prop, a, *, n_water, sigma, ...):
    eta, ηx, ηy = reconstruct_surface(prop, a)
    xl, yl       = _paraxial_landing(X_src, Y_src, eta, ηx, ηy, depth, n_water)
    fi, fj       = (xl - xs[0])/dx, (yl - ys[0])/dy
    ix0, iy0     = floor(fi), floor(fj)
    wx, wy       = fi - ix0, fj - iy0
    mask         = in_bounds(fi, fj)
    D            = scatter_bilinear(ix0, iy0, wx, wy, mask)
    I            = gaussian_blur(D)
    residuals    = (eta, ηx, ηy, xl, yl, ix0, iy0, wx, wy, mask, D)
    return (xs, ys, I), residuals

def caustic_image_bwd(prop, n_water, sigma, dx, dy, residuals, g):
    dL_dI = g[2]                             # gradient w.r.t. I
    # Step 6: blur adjoint
    dL_dD = gaussian_blur(dL_dI)
    # Step 5: splat adjoint
    dL_dxl, dL_dyl = splat_adjoint(dL_dD, ix0, iy0, wx, wy, mask, dx, dy)
    # Step 4: refraction adjoint
    dL_dηx = dL_dxl * (depth - eta) / n_water
    dL_dηy = dL_dyl * (depth - eta) / n_water
    dL_dη  = -dL_dxl * ηx / n_water - dL_dyl * ηy / n_water
    # Step 3: reconstruct adjoint
    dL_da  = reconstruct_adjoint(prop, dL_dη, dL_dηx, dL_dηy)
    return (None, dL_da)   # gradient w.r.t. (prop, a)

caustic_image.defvjp(caustic_image_fwd, caustic_image_bwd)
```

The full `θ → a → I → L` gradient is obtained by composing this custom VJP with JAX's automatic
differentiation through `steady_state_amplitudes` (which is a pure linear function of `P`, and
therefore of `θ`).

---

## Speedup rationale

JAX's standard autodiff stores every intermediate array on the backward tape (O(n_ops) memory,
~2–5× wall time vs forward pass). The custom VJP stores only the 11 arrays listed in `residuals`
and replaces the backward tape with ~10 matrix multiplications. For a 100×100 grid with 100 modes:

| Step | Forward FLOP | Backward FLOP |
|------|-------------|---------------|
| reconstruct_surface | O(nx·ny·n_modes) | O(nx·ny·n_modes) via 3 matmuls |
| paraxial refraction | O(nx·ny) | O(nx·ny) |
| bilinear splat | O(nx·ny) | O(nx·ny) (gather at saved indices) |
| Gaussian blur | O(nx·ny·w) | O(nx·ny·w) (same kernel) |

The backward pass is approximately the same cost as the forward pass, matching the ideal
`2× forward` bound for reverse-mode AD of smooth functions.

---

## Validation

After implementing `caustic_image_vjp`, verify with:

```python
# Should agree to ~1e-5 (limited by float32 FD precision)
jax.test_util.check_grads(caustic_image, (prop, a0), order=1, modes=['rev'])
```

And confirm the custom VJP matches JAX's autodiff on the same input:

```python
g_custom = jax.grad(lambda a: jnp.sum(caustic_image(prop, a)[2]))(a0)
g_auto   = jax.grad(lambda a: jnp.sum(caustic_image_no_vjp(prop, a)[2]))(a0)
assert jnp.allclose(g_custom, g_auto, rtol=1e-4)
```
