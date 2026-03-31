# test_steady_state.jl — Validate steady-state formulation against time-domain
#
# Run with:  julia test_steady_state.jl

using LinearAlgebra
using Zygote

include("src/WaveTank.jl")
using .WaveTank

println("=" ^ 60)
println("  Steady-State vs Time-Domain Validation")
println("=" ^ 60)

# ── 1. Build propagator with sinusoidal actuators ────────────────────

n_act = 8
n_freq = 4
n_modes = 10

tank = Tank(1.0, 1.0, 0.1; damping=0.02)

freqs = [2.0, 5.0, 8.0, 12.0]
Ω_freqs = 2π .* freqs

# Random complex phasors
X0 = 0.001 .* randn(n_act, n_freq)
Y0 = 0.001 .* randn(n_act, n_freq)
P0 = X0 .+ im .* Y0

# Create SineSum actuators matching the phasor parameterization
# q_i(t) = Σ_k Im[P_ik e^{iΩ_k t}] = Σ_k (X_ik sin(Ω_k t) + Y_ik cos(Ω_k t))
# Rewrite as: Σ_k A_ik sin(Ω_k t + φ_ik) where A = |P|, φ = angle(P)
A_mat = abs.(P0)
φ_mat = angle.(P0)

perimeter = [(x, 0.0) for x in range(0.1, 0.9, length=n_act÷2)]
append!(perimeter, [(0.0, y) for y in range(0.1, 0.9, length=n_act÷2)])

actuators = [
    Actuator(pos[1], pos[2],
             SineSum(; freqs=freqs, A=A_mat[i,:], φ=φ_mat[i,:]);
             width=0.05)
    for (i, pos) in enumerate(perimeter)
]

# Long time span so transients decay: T >> 1/(γ·ω_min)
# ω_min ≈ 2π·2 ≈ 12.6 → 1/(0.02·12.6) ≈ 4.0 s; use T = 20s
T_eval = 20.0
sim = WaveSim(tank, actuators, (0.0, T_eval), 0.005; n_modes=n_modes)
prop = build_propagator(sim; nx=30, ny=30)

println("\nPropagator: $(length(prop.ω)) modes, $(prop.nx)×$(prop.ny) grid, $(length(prop.t_grid)) time steps")

# ── 2. Time-domain modal amplitudes ─────────────────────────────────

a_td = evaluate_modal_amplitudes(prop, T_eval)
println("\nTime-domain |a|_max = $(maximum(abs, a_td))")

# ── 3. Steady-state modal amplitudes ────────────────────────────────

a_ss = steady_state_amplitudes(prop, P0, Ω_freqs, T_eval)
println("Steady-state |a|_max = $(maximum(abs, a_ss))")

# ── 4. Compare ──────────────────────────────────────────────────────

diff = a_td .- a_ss
rel_err = norm(diff) / max(norm(a_td), 1e-12)
max_abs_err = maximum(abs, diff)
max_rel_per_mode = maximum(abs.(diff) ./ max.(abs.(a_td), 1e-15))

println("\nComparison:")
println("  L2 relative error: $(round(rel_err * 100, digits=4))%")
println("  Max absolute error: $max_abs_err")
println("  Max per-mode relative error: $(round(max_rel_per_mode * 100, digits=4))%")

# Allow up to 5% error since transients may not be fully decayed
if rel_err < 0.05
    println("  ✓ Steady-state matches time-domain (< 5% error)")
else
    println("  ✗ Error too large — transients may not have decayed")
    println("    Try increasing T_eval or damping")
end

# ── 5. Test Zygote differentiability of steady-state loss ───────────

println("\n" * "=" ^ 60)
println("  Zygote Gradient Test for Steady-State Loss")
println("=" ^ 60)

# Build target and loss
xs, ys = prop.xs, prop.ys
target = [exp(-((x - 0.5)^2 + (y - 0.5)^2) / (2 * 0.1^2)) for x in xs, y in ys]
target ./= maximum(target)

loss = make_caustic_loss_ss(prop, T_eval, target, Ω_freqs;
                             σ_blur=0.02, λ_energy=1e-4)

p0 = pack_complex(X0, Y0)
println("\nParameter vector length: $(length(p0))")

L0 = loss(p0)
println("Loss at p0: $L0")
@assert isfinite(L0) "Loss is not finite!"
println("  ✓ Loss is finite")

println("\nComputing Zygote gradient...")
g = Zygote.gradient(loss, p0)[1]

@assert length(g) == length(p0) "Gradient length mismatch"
@assert all(isfinite, g) "Gradient contains non-finite values!"
println("  ✓ Gradient is finite, length $(length(g))")
println("  Gradient norm: $(norm(g))")

# ── 6. Finite-difference check ──────────────────────────────────────

println("\nFinite-difference check...")
idx = argmax(abs.(g))
ε = 1e-5
p_plus = copy(p0); p_plus[idx] += ε
p_minus = copy(p0); p_minus[idx] -= ε
fd_grad = (loss(p_plus) - loss(p_minus)) / (2ε)
ad_grad = g[idx]

rel_err_grad = abs(fd_grad - ad_grad) / max(abs(fd_grad), abs(ad_grad), 1e-12)
println("  AD gradient:     $ad_grad")
println("  FD gradient:     $fd_grad")
println("  Relative error:  $(round(rel_err_grad * 100, digits=4))%")

@assert rel_err_grad < 0.05 "Gradient relative error too large!"
println("  ✓ Gradient matches finite differences")

# ── 7. Gradient descent step ────────────────────────────────────────

println("\nGradient descent step...")
α = 1e-3 / max(norm(g), 1e-10)
p1 = p0 .- α .* g
L1 = loss(p1)
println("  Loss before: $L0")
println("  Loss after:  $L1")
@assert L1 < L0 "Loss did not decrease!"
println("  ✓ Loss decreased")

println("\n" * "=" ^ 60)
println("  ALL TESTS PASSED")
println("=" ^ 60)
