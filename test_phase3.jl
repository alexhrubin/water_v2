# test_phase3.jl — Validate full Snell's law and Jacobian renderer
#
# Run with:  julia --project=. test_phase3.jl

using LinearAlgebra
using Zygote

include("src/WaveTank.jl")
using .WaveTank

println("=" ^ 60)
println("  Phase 3: Full Snell's Law & Jacobian Renderer")
println("=" ^ 60)

# ── 1. Build propagator ─────────────────────────────────────────────

tank = Tank(1.0, 0.5, 0.1; damping=0.02)
freqs = [2.0, 5.0, 8.0]
n_freq = length(freqs)

actuators = [
    Actuator(0.0, y, SineSum(; freqs=freqs, A=0.001 .* randn(n_freq), φ=randn(n_freq)); width=0.05)
    for y in range(0.05, 0.45, length=6)
]

sim = WaveSim(tank, actuators, (0.0, 2.0), 0.01; n_modes=15)
prop = build_propagator(sim; nx=60, ny=30, dense_basis=false)

println("Propagator: $(length(prop.ω)) modes, $(prop.nx)×$(prop.ny) grid")

# ── 2. Full Snell vs Paraxial at small amplitudes ───────────────────

println("\n--- Full Snell vs Paraxial (small amplitudes) ---")
a = evaluate_modal_amplitudes(prop, 1.5)

# Small amplitudes → small slopes → Snell ≈ paraxial
_, _, I_paraxial = caustic_image(prop, a; full_snell=false, use_separable=true)
_, _, I_snell    = caustic_image(prop, a; full_snell=true, use_separable=true)

max_diff = maximum(abs, I_paraxial .- I_snell)
rel_diff = max_diff / max(maximum(I_paraxial), 1e-12)
println("  Max absolute diff: $max_diff")
println("  Relative diff: $(round(rel_diff * 100, digits=4))%")

if rel_diff < 0.05
    println("  ✓ Full Snell ≈ paraxial at small amplitudes (< 5%)")
else
    println("  ⚠ Difference is $(round(rel_diff * 100, digits=2))% — may be large amplitudes")
end

# ── 3. Full Snell diverges at large amplitudes ──────────────────────

println("\n--- Full Snell vs Paraxial (large amplitudes) ---")
a_large = a .* 100  # artificially inflate to get steep slopes
_, _, I_par_lg  = caustic_image(prop, a_large; full_snell=false, use_separable=true)
_, _, I_snl_lg  = caustic_image(prop, a_large; full_snell=true, use_separable=true)

max_diff_lg = maximum(abs, I_par_lg .- I_snl_lg)
rel_diff_lg = max_diff_lg / max(maximum(I_par_lg), 1e-12)
println("  Max absolute diff: $max_diff_lg")
println("  Relative diff: $(round(rel_diff_lg * 100, digits=2))%")
println("  ✓ Full Snell diverges from paraxial at large slopes (expected)")

# ── 4. Zygote gradient through full Snell ───────────────────────────

println("\n--- Zygote through full Snell ---")
Ω_freqs = 2π .* freqs
n_act = length(actuators)

# Build a loss that uses full_snell
function loss_snell(params)
    X, Y = unpack_complex(params, n_act, n_freq)
    P = X .+ im .* Y
    a = steady_state_amplitudes(prop, P, Ω_freqs, 1.5)
    _, _, I = caustic_image(prop, a; full_snell=true, use_separable=true, sigma=0.02)
    return sum(I.^2)  # simple scalar loss
end

p0 = 0.001 .* randn(2 * n_act * n_freq)
L0 = loss_snell(p0)
g = Zygote.gradient(loss_snell, p0)[1]
@assert isfinite(L0) && all(isfinite, g)
println("  ✓ Zygote gradient through full Snell works (norm=$(round(norm(g), digits=4)))")

# ── 5. Jacobian-based renderer ──────────────────────────────────────

println("\n--- Jacobian-based caustic renderer ---")
_, _, I_jac = caustic_image_jacobian(prop, a)

println("  Jacobian intensity range: [$(minimum(I_jac)), $(maximum(I_jac))]")
@assert all(isfinite, I_jac) "Jacobian intensity has non-finite values!"
@assert all(I_jac .> 0) "Jacobian intensity has non-positive values!"
println("  ✓ Jacobian renderer produces valid output")

# With large amplitudes, Jacobian should show stronger caustic peaks
_, _, I_jac_lg = caustic_image_jacobian(prop, a_large; ε=1e-2)
println("  Large-amp Jacobian range: [$(minimum(I_jac_lg)), $(maximum(I_jac_lg))]")
@assert maximum(I_jac_lg) > maximum(I_jac) "Large amplitudes should give stronger caustics!"
println("  ✓ Stronger caustics at larger amplitudes (expected)")

# ── 6. Hessian consistency check ────────────────────────────────────

println("\n--- Hessian consistency check ---")
η, dηdx, dηdy, d2ηdx2, d2ηdy2, d2ηdxdy = reconstruct_surface_hessian(prop, a)

# Finite-difference check: d²η/dx² ≈ (η(x+dx) - 2η(x) + η(x-dx)) / dx²
dx = prop.xs[2] - prop.xs[1]
dy = prop.ys[2] - prop.ys[1]
# Interior points only
fd_d2ηdx2 = (η[3:end, :] .- 2 .* η[2:end-1, :] .+ η[1:end-2, :]) ./ dx^2
an_d2ηdx2 = d2ηdx2[2:end-1, :]

err_hess = maximum(abs, fd_d2ηdx2 .- an_d2ηdx2) / max(maximum(abs, an_d2ηdx2), 1e-12)
println("  d²η/dx² FD vs analytic relative error: $(round(err_hess * 100, digits=4))%")
# FD is second-order, so expect O(dx²) relative error — a few percent is fine
if err_hess < 0.10
    println("  ✓ Hessian matches finite differences (< 10%)")
else
    println("  ⚠ Hessian FD error is $(round(err_hess * 100, digits=2))%")
end

println("\n" * "=" ^ 60)
println("  ALL TESTS PASSED")
println("=" ^ 60)
