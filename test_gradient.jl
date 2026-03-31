# test_gradient.jl — Verify Zygote differentiability of the caustic loss pipeline
#
# Run with:  julia test_gradient.jl

using LinearAlgebra
using Zygote

# Load WaveTank module
include("src/WaveTank.jl")
using .WaveTank

println("=" ^ 60)
println("  Zygote Gradient Test for Caustic Loss Pipeline")
println("=" ^ 60)

# ── 1. Build a small propagator ──────────────────────────────────────

n_act = 4
n_freq = 3
n_modes = 5   # small for speed

tank = Tank(1.0, 0.5, 0.1; damping=0.02)

# Create actuators with SineSum forcing (needed for the propagator)
freqs = [2.0, 4.0, 6.0]
ω_freqs = 2π .* freqs

actuators = [
    Actuator(0.25, 0.125, SineSum(; freqs=freqs, A=0.001 .* randn(n_freq), φ=randn(n_freq)); width=0.05),
    Actuator(0.75, 0.125, SineSum(; freqs=freqs, A=0.001 .* randn(n_freq), φ=randn(n_freq)); width=0.05),
    Actuator(0.25, 0.375, SineSum(; freqs=freqs, A=0.001 .* randn(n_freq), φ=randn(n_freq)); width=0.05),
    Actuator(0.75, 0.375, SineSum(; freqs=freqs, A=0.001 .* randn(n_freq), φ=randn(n_freq)); width=0.05),
]

sim = WaveSim(tank, actuators, (0.0, 0.5), 0.01; n_modes=n_modes)
prop = build_propagator(sim; nx=30, ny=15)

println("\nPropagator built: $(length(prop.ω)) modes, $(prop.nx)×$(prop.ny) grid, $(length(prop.t_grid)) time steps")

# ── 2. Create synthetic target ───────────────────────────────────────

# Use a simple Gaussian blob as target
xs, ys = prop.xs, prop.ys
nx, ny = prop.nx, prop.ny
cx, cy = 0.5, 0.25
target = [exp(-((x - cx)^2 + (y - cy)^2) / (2 * 0.05^2)) for x in xs, y in ys]
target ./= maximum(target)

println("Target created: $(size(target)) with max=$(maximum(target))")

# ── 3. Build loss closure ────────────────────────────────────────────

T_eval = 0.4
loss = make_caustic_loss(prop, T_eval, target, ω_freqs;
                         σ_blur=0.0, λ_energy=1e-4, λ_smooth=1e-6)

# ── 4. Initial parameters ───────────────────────────────────────────

A0 = 0.001 .* randn(n_act, n_freq)
φ0 = randn(n_act, n_freq)
p0 = pack_params(A0, φ0)

println("\nParameter vector length: $(length(p0))")

# ── 5. Evaluate loss — check finite ─────────────────────────────────

L0 = loss(p0)
println("\nLoss at p0: $L0")
@assert isfinite(L0) "Loss is not finite!"
println("  ✓ Loss is finite")

# ── 6. Zygote gradient — check finite and correct length ────────────

println("\nComputing Zygote gradient...")
g = Zygote.gradient(loss, p0)[1]

@assert length(g) == length(p0) "Gradient length mismatch: $(length(g)) vs $(length(p0))"
println("  ✓ Gradient length matches: $(length(g))")

@assert all(isfinite, g) "Gradient contains non-finite values!"
println("  ✓ All gradient entries are finite")
println("  Gradient norm: $(norm(g))")
println("  Gradient range: [$(minimum(g)), $(maximum(g))]")

# ── 7. Finite-difference check on one direction ─────────────────────

println("\nFinite-difference check...")
idx = argmax(abs.(g))  # pick the direction with largest gradient
ε = 1e-5

p_plus = copy(p0);  p_plus[idx]  += ε
p_minus = copy(p0); p_minus[idx] -= ε

fd_grad = (loss(p_plus) - loss(p_minus)) / (2ε)
ad_grad = g[idx]

rel_err = abs(fd_grad - ad_grad) / max(abs(fd_grad), abs(ad_grad), 1e-12)
println("  Parameter index: $idx")
println("  AD gradient:     $ad_grad")
println("  FD gradient:     $fd_grad")
println("  Relative error:  $(round(rel_err * 100, digits=4))%")

@assert rel_err < 0.01 "Relative error $(rel_err) exceeds 1%!"
println("  ✓ Relative error < 1%")

# ── 8. One gradient descent step — confirm loss decreases ───────────

println("\nGradient descent step...")
α = 1e-3 / max(norm(g), 1e-10)  # normalized step size
p1 = p0 .- α .* g
L1 = loss(p1)

println("  Loss before: $L0")
println("  Loss after:  $L1")
println("  Decrease:    $(L0 - L1)")

@assert L1 < L0 "Loss did not decrease after gradient step!"
println("  ✓ Loss decreased")

println("\n" * "=" ^ 60)
println("  ALL TESTS PASSED")
println("=" ^ 60)
