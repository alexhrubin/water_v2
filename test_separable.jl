# test_separable.jl — Validate separable reconstruction against dense Φ
#
# Run with:  julia --project=. test_separable.jl

using LinearAlgebra
using Zygote

include("src/WaveTank.jl")
using .WaveTank

println("=" ^ 60)
println("  Separable vs Dense Reconstruction Validation")
println("=" ^ 60)

# ── 1. Build propagator with both dense and separable bases ─────────

tank = Tank(1.0, 0.5, 0.1; damping=0.02)
n_freq = 3
freqs = [2.0, 5.0, 8.0]

actuators = [
    Actuator(0.0, y, SineSum(; freqs=freqs, A=0.001 .* randn(n_freq), φ=randn(n_freq)); width=0.05)
    for y in range(0.05, 0.45, length=6)
]

sim = WaveSim(tank, actuators, (0.0, 2.0), 0.01; n_modes=15)

# Build with dense (default)
prop_dense = build_propagator(sim; nx=60, ny=30, dense_basis=true)
# Build without dense
prop_sep = build_propagator(sim; nx=60, ny=30, dense_basis=false)

println("\nModes: $(length(prop_dense.ω)), Grid: $(prop_dense.nx)×$(prop_dense.ny)")
println("Dense Φ size: $(size(prop_dense.Φ)) = $(sizeof(prop_dense.Φ) ÷ 1024) KB")
println("Separable cos_x size: $(size(prop_sep.cos_x)) = $(sizeof(prop_sep.cos_x) ÷ 1024) KB")
println("Separable Φ size: $(size(prop_sep.Φ)) (empty)")

# ── 2. Compare surface reconstruction ───────────────────────────────

a = evaluate_modal_amplitudes(prop_dense, 1.5)

# Dense path
η_dense = reshape(prop_dense.Φ' * a, prop_dense.nx, prop_dense.ny)
dηdx_dense = reshape(prop_dense.dΦ_dx' * a, prop_dense.nx, prop_dense.ny)
dηdy_dense = reshape(prop_dense.dΦ_dy' * a, prop_dense.nx, prop_dense.ny)

# Separable path
η_sep, dηdx_sep, dηdy_sep = reconstruct_surface(prop_dense, a)

println("\nSurface reconstruction comparison:")
println("  η    max diff: $(maximum(abs, η_dense .- η_sep))")
println("  dη/dx max diff: $(maximum(abs, dηdx_dense .- dηdx_sep))")
println("  dη/dy max diff: $(maximum(abs, dηdy_dense .- dηdy_sep))")

@assert maximum(abs, η_dense .- η_sep) < 1e-12 "η mismatch!"
@assert maximum(abs, dηdx_dense .- dηdx_sep) < 1e-12 "dη/dx mismatch!"
@assert maximum(abs, dηdy_dense .- dηdy_sep) < 1e-12 "dη/dy mismatch!"
println("  ✓ All fields match to machine precision")

# ── 3. Compare caustic images ───────────────────────────────────────

println("\nCaustic image comparison:")
_, _, I_dense = caustic_image(prop_dense, a; use_separable=false)
_, _, I_sep   = caustic_image(prop_dense, a; use_separable=true)

max_diff = maximum(abs, I_dense .- I_sep)
rel_diff = max_diff / max(maximum(abs, I_dense), 1e-12)
println("  Max absolute diff: $max_diff")
println("  Max relative diff: $(round(rel_diff * 100, digits=8))%")

@assert max_diff < 1e-10 "Caustic image mismatch!"
println("  ✓ Caustic images match")

# ── 4. Test dense_basis=false propagator works end-to-end ───────────

println("\nTesting dense_basis=false end-to-end:")
a_sep = evaluate_modal_amplitudes(prop_sep, 1.5)
@assert a_sep ≈ a "Modal amplitudes differ between propagators!"
println("  ✓ Modal amplitudes match")

_, _, I_from_sep = caustic_image(prop_sep, a_sep)
@assert maximum(abs, I_from_sep .- I_dense) < 1e-10 "Caustic from sep propagator differs!"
println("  ✓ Caustic from dense_basis=false propagator matches")

# ── 5. Zygote gradient through separable path ──────────────────────

println("\nZygote gradient through separable path:")
Ω_freqs = 2π .* freqs
n_act = length(actuators)

loss = make_caustic_loss_ss(prop_sep, 1.5,
    [exp(-((x-0.5)^2 + (y-0.25)^2) / 0.01) for x in prop_sep.xs, y in prop_sep.ys],
    Ω_freqs; σ_blur=0.02, λ_energy=1e-4)

p0 = 0.001 .* randn(2 * n_act * n_freq)
L0 = loss(p0)
g = Zygote.gradient(loss, p0)[1]

@assert isfinite(L0) "Loss not finite!"
@assert all(isfinite, g) "Gradient not finite!"
println("  Loss: $L0")
println("  Gradient norm: $(norm(g))")
println("  ✓ Zygote works through separable path")

# ── 6. Finite-difference check on separable gradient ────────────────

println("\nFinite-difference check (separable):")
idx = argmax(abs.(g))
ε = 1e-5
p_plus = copy(p0); p_plus[idx] += ε
p_minus = copy(p0); p_minus[idx] -= ε
fd = (loss(p_plus) - loss(p_minus)) / (2ε)
ad = g[idx]
rel_err = abs(fd - ad) / max(abs(fd), abs(ad), 1e-12)
println("  AD:  $ad")
println("  FD:  $fd")
println("  Rel error: $(round(rel_err * 100, digits=4))%")
@assert rel_err < 0.05 "Gradient error too large!"
println("  ✓ Gradient matches finite differences")

println("\n" * "=" ^ 60)
println("  ALL TESTS PASSED")
println("=" ^ 60)
