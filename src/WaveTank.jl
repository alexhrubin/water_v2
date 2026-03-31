module WaveTank

using LinearAlgebra
using Statistics
using SparseArrays
using Printf
using Zygote
using FileIO
using ColorTypes

export Tank, Actuator, SineSum, WaveSim, Propagator
export build_propagator, evaluate_surface, visualize
export evaluate_modal_amplitudes, caustic_image, caustic_loss, visualize_caustic
export params_to_Q, pack_params, unpack_params, make_caustic_loss, make_caustic_loss_refining
export transfer_matrix, steady_state_amplitudes
export reconstruct_surface, reconstruct_surface_hessian, snell_landing
export caustic_image_jacobian
export pack_complex, unpack_complex, make_caustic_loss_ss, make_caustic_loss_ss_refining
export make_caustic_loss_ss_keyframes
export load_target_image, analyze_target, setup_from_target, analytical_solve

# ── Data structures ──────────────────────────────────────────────────

struct Tank
    Lx::Float64
    Ly::Float64
    depth::Float64
    g::Float64
    damping::Float64  # modal damping ratio γ
end
Tank(Lx, Ly, depth; g=9.81, damping=0.01) = Tank(Lx, Ly, depth, g, damping)

struct SineSum
    A::Vector{Float64}      # amplitudes
    ω::Vector{Float64}      # angular frequencies
    φ::Vector{Float64}      # phases
end
(s::SineSum)(t) = sum(s.A[n] * sin(s.ω[n] * t + s.φ[n]) for n in eachindex(s.A))
SineSum(; freqs, A, φ=zeros(length(A))) = SineSum(A, 2π .* freqs, φ)

# ── Parameter helpers for AD ─────────────────────────────────────────

"""
    params_to_Q(A_mat, φ_mat, ω_freqs, t_grid)

Pure broadcast computation of actuator signals from Fourier coefficients.
- `A_mat` [n_act × n_freq], `φ_mat` [n_act × n_freq] — differentiable
- `ω_freqs` [n_freq], `t_grid` [n_steps] — fixed constants
Returns `Q` [n_act × n_steps] where Q[i,k] = Σ_n A[i,n] * sin(ω[n]*t[k] + φ[i,n])
"""
function params_to_Q(A_mat::AbstractMatrix, φ_mat::AbstractMatrix,
                     ω_freqs::AbstractVector, t_grid::AbstractVector)
    # A_mat:  [n_act × n_freq]
    # φ_mat:  [n_act × n_freq]
    # ω_freqs: [n_freq]
    # t_grid:  [n_steps]
    # 3D broadcast: phases[n_act, n_freq, n_steps] = ω[1,n,1]*t[1,1,k] + φ[i,n,1]
    n_act, n_freq = size(A_mat)
    n_steps = length(t_grid)
    ω_3d = reshape(ω_freqs, 1, n_freq, 1)        # [1, n_freq, 1]
    t_3d = reshape(t_grid, 1, 1, n_steps)          # [1, 1, n_steps]
    A_3d = reshape(A_mat, n_act, n_freq, 1)        # [n_act, n_freq, 1]
    φ_3d = reshape(φ_mat, n_act, n_freq, 1)        # [n_act, n_freq, 1]
    # sin_vals: [n_act, n_freq, n_steps]
    sin_vals = sin.(ω_3d .* t_3d .+ φ_3d)
    # weighted sum over freq dim → [n_act, 1, n_steps] → reshape to [n_act, n_steps]
    Q = dropdims(sum(A_3d .* sin_vals, dims=2), dims=2)
    return Q
end

"""
    pack_params(A_mat, φ_mat) → flat vector [vec(A); vec(φ)]
"""
pack_params(A_mat::AbstractMatrix, φ_mat::AbstractMatrix) = vcat(vec(A_mat), vec(φ_mat))

"""
    unpack_params(params, n_act, n_freq) → (A_mat, φ_mat)
"""
function unpack_params(params::AbstractVector, n_act::Int, n_freq::Int)
    n = n_act * n_freq
    A_mat = reshape(params[1:n], n_act, n_freq)
    φ_mat = reshape(params[n+1:2n], n_act, n_freq)
    return A_mat, φ_mat
end

# ── Complex phasor parameter helpers (for steady-state AD) ──────────

"""
    pack_complex(X, Y) → flat real vector [vec(X); vec(Y)]

Pack real and imaginary parts of complex phasors into a single parameter vector.
"""
pack_complex(X::AbstractMatrix, Y::AbstractMatrix) = vcat(vec(X), vec(Y))

"""
    unpack_complex(params, n_act, n_freq) → (X, Y) real matrices

Inverse of `pack_complex`. Returns Re and Im parts as separate matrices.
"""
function unpack_complex(params::AbstractVector, n_act::Int, n_freq::Int)
    n = n_act * n_freq
    X = reshape(params[1:n], n_act, n_freq)
    Y = reshape(params[n+1:2n], n_act, n_freq)
    return X, Y
end

struct Actuator{F}
    x::Float64
    y::Float64
    forcing::F             # t → amplitude (any callable)
    width::Float64         # Gaussian half-width σ (0 = point source)
end
Actuator(x, y, forcing; width=0.0) = Actuator(x, y, forcing, width)

struct WaveSim{A<:Vector{<:Actuator}}
    tank::Tank
    actuators::A
    tspan::Tuple{Float64,Float64}
    dt::Float64
    n_modes::Int  # per direction
end
WaveSim(tank, actuators, tspan, dt; n_modes=20) =
    WaveSim(tank, actuators, tspan, dt, n_modes)

struct Propagator
    sim::WaveSim
    # Mode indices (m, n) and frequencies
    mode_m::Vector{Int}
    mode_n::Vector{Int}
    ω::Vector{Float64}      # natural frequencies
    ω_d::Vector{Float64}    # damped frequencies
    # Coupling matrix C[j, i] = φ_j(x_i, y_i) / N_j
    C::Matrix{Float64}
    # Dense spatial basis Φ[j, nx*ny] — empty when dense_basis=false
    Φ::Matrix{Float64}
    dΦ_dx::Matrix{Float64}
    dΦ_dy::Matrix{Float64}
    # Separable 1D basis matrices (always populated, tiny memory)
    cos_x::Matrix{Float64}    # [nx × n_modes] cos(mπx/Lx)
    cos_y::Matrix{Float64}    # [ny × n_modes] cos(nπy/Ly)
    dcos_x::Matrix{Float64}   # [nx × n_modes] -mπ/Lx · sin(mπx/Lx)
    dcos_y::Matrix{Float64}   # [ny × n_modes] -nπ/Ly · sin(nπy/Ly)
    d2cos_x::Matrix{Float64}  # [nx × n_modes] -(mπ/Lx)² · cos(mπx/Lx)
    d2cos_y::Matrix{Float64}  # [ny × n_modes] -(nπ/Ly)² · cos(nπy/Ly)
    lin_2d::Vector{Int}       # flat mode j → linear index in [n_modes × n_modes] grid
    # Evaluation grid
    xs::Vector{Float64}
    ys::Vector{Float64}
    nx::Int
    ny::Int
    # Time grid
    t_grid::Vector{Float64}
    # Cached coordinate grids [nx × ny]
    X_src::Matrix{Float64}
    Y_src::Matrix{Float64}
end

# ── Build propagator ─────────────────────────────────────────────────

function build_propagator(sim::WaveSim; nx=100, ny=50, dense_basis=true)
    (; tank, actuators, tspan, dt, n_modes) = sim
    (; Lx, Ly, depth, g, damping) = tank

    # Collect mode indices, skipping (0,0)
    mode_m = Int[]
    mode_n = Int[]
    for m in 0:n_modes-1, n in 0:n_modes-1
        (m == 0 && n == 0) && continue
        push!(mode_m, m)
        push!(mode_n, n)
    end
    n_total = length(mode_m)

    # Mode wavenumbers and frequencies (full gravity-wave dispersion)
    k = [sqrt((mode_m[j]*π/Lx)^2 + (mode_n[j]*π/Ly)^2) for j in 1:n_total]
    ω = [sqrt(g * k[j] * tanh(k[j] * depth)) for j in 1:n_total]
    ω_d = ω .* sqrt(1 - damping^2)

    # Normalization factors N_j = ∫∫ φ_j² dx dy
    function norm_j(m, n)
        Ix = m == 0 ? Lx : Lx / 2
        Iy = n == 0 ? Ly : Ly / 2
        return Ix * Iy
    end

    # Eigenmode evaluation
    φ(m, n, x, y) = cos(m * π * x / Lx) * cos(n * π * y / Ly)

    # Coupling matrix C[j, i]
    n_act = length(actuators)
    C = zeros(n_total, n_act)
    for i in 1:n_act
        ax, ay = actuators[i].x, actuators[i].y
        σ_a = actuators[i].width
        for j in 1:n_total
            m, n = mode_m[j], mode_n[j]
            kx = m * π / Lx
            ky = n * π / Ly
            blob = σ_a > 0 ? exp(-0.5 * σ_a^2 * (kx^2 + ky^2)) : 1.0
            C[j, i] = φ(m, n, ax, ay) * blob / norm_j(m, n)
        end
    end

    # Evaluation grid
    xs = range(0, Lx, length=nx)
    ys = range(0, Ly, length=ny)

    # ── Separable 1D basis matrices (always computed, tiny memory) ──
    cos_x   = [cos(m * π * xs[i] / Lx) for i in 1:nx, m in 0:n_modes-1]
    cos_y   = [cos(n * π * ys[j] / Ly) for j in 1:ny, n in 0:n_modes-1]
    dcos_x  = [-m * π / Lx * sin(m * π * xs[i] / Lx) for i in 1:nx, m in 0:n_modes-1]
    dcos_y  = [-n * π / Ly * sin(n * π * ys[j] / Ly) for j in 1:ny, n in 0:n_modes-1]
    d2cos_x = [-(m * π / Lx)^2 * cos(m * π * xs[i] / Lx) for i in 1:nx, m in 0:n_modes-1]
    d2cos_y = [-(n * π / Ly)^2 * cos(n * π * ys[j] / Ly) for j in 1:ny, n in 0:n_modes-1]

    # Mapping: flat mode index j → linear index in [n_modes × n_modes] 2D grid
    lin_2d = [mode_m[j] + 1 + n_modes * mode_n[j] for j in 1:n_total]

    # ── Dense basis matrices (optional, for backward compatibility) ──
    if dense_basis
        Φ = zeros(n_total, nx * ny)
        dΦ_dx = zeros(n_total, nx * ny)
        dΦ_dy = zeros(n_total, nx * ny)
        idx = 0
        for iy in 1:ny, ix in 1:nx
            idx += 1
            for j in 1:n_total
                m, n = mode_m[j], mode_n[j]
                Φ[j, idx] = φ(m, n, xs[ix], ys[iy])
                dΦ_dx[j, idx] = -m * π / Lx * sin(m * π * xs[ix] / Lx) * cos(n * π * ys[iy] / Ly)
                dΦ_dy[j, idx] = -n * π / Ly * cos(m * π * xs[ix] / Lx) * sin(n * π * ys[iy] / Ly)
            end
        end
    else
        Φ = zeros(0, 0)
        dΦ_dx = zeros(0, 0)
        dΦ_dy = zeros(0, 0)
    end

    # Time grid
    t_grid = collect(tspan[1]:dt:tspan[2])

    xs_vec = collect(xs)
    ys_vec = collect(ys)
    X_src = repeat(xs_vec, 1, ny)
    Y_src = repeat(ys_vec', nx, 1)
    return Propagator(sim, mode_m, mode_n, ω, ω_d, C, Φ, dΦ_dx, dΦ_dy,
                      cos_x, cos_y, dcos_x, dcos_y, d2cos_x, d2cos_y, lin_2d,
                      xs_vec, ys_vec, nx, ny, t_grid, X_src, Y_src)
end

# ── Green's function kernel ──────────────────────────────────────────

function green_kernel(ω_j, ω_dj, γ, τ)
    τ <= 0 && return 0.0
    return exp(-γ * ω_j * τ) * sin(ω_dj * τ) / ω_dj
end

# ── Modal amplitudes ────────────────────────────────────────────────

function evaluate_modal_amplitudes(prop::Propagator, T::Real)
    (; sim, C, ω, ω_d, t_grid) = prop
    (; tank, actuators, dt) = sim
    γ = tank.damping

    n_steps = length(t_grid)

    # Sample actuator signals: Q[n_act × n_steps] — broadcast per actuator
    Q = reduce(vcat, [act.forcing.(t_grid)' for act in actuators])

    # Modal forcing: F = C · Q  → [n_modes × n_steps]
    F = C * Q

    # Vectorised temporal convolution via Green's kernel matrix
    τ_vec = T .- t_grid                                            # [n_steps]
    mask = τ_vec .> 0                                              # [n_steps] Bool
    G = (exp.((-γ) .* ω .* τ_vec') .* sin.(ω_d .* τ_vec') ./ ω_d) .* mask'
    #    [n_modes × n_steps]
    a = vec(sum(G .* F, dims=2)) .* dt                             # [n_modes]

    return a
end

"""
    evaluate_modal_amplitudes(prop, T, Q)

Like `evaluate_modal_amplitudes(prop, T)` but takes pre-computed actuator
signal matrix `Q` [n_act × n_steps] instead of sampling from actuators.
This method is Zygote-differentiable w.r.t. `Q`.
"""
function evaluate_modal_amplitudes(prop::Propagator, T::Real, Q::AbstractMatrix)
    (; C, ω, ω_d, t_grid) = prop
    γ = prop.sim.tank.damping
    dt = prop.sim.dt

    # Modal forcing: F = C · Q  → [n_modes × n_steps]
    F = C * Q

    # Vectorised temporal convolution via Green's kernel matrix
    τ_vec = T .- t_grid                                            # [n_steps]
    mask = τ_vec .> 0                                              # [n_steps] Bool
    G = (exp.((-γ) .* ω .* τ_vec') .* sin.(ω_d .* τ_vec') ./ ω_d) .* mask'
    #    [n_modes × n_steps]
    a = vec(sum(G .* F, dims=2)) .* dt                             # [n_modes]

    return a
end

# ── Steady-state frequency-domain formulation ───────────────────────

"""
    transfer_matrix(ω_modes, Ω_drive, γ) → Complex matrix [n_modes × n_freq]

Frequency-response of each mode to each driving frequency.
H_j(Ω) = 1 / (ω_j² - Ω² + 2i·γ·ω_j·Ω)
"""
function transfer_matrix(ω_modes::AbstractVector, Ω_drive::AbstractVector, γ::Real)
    return 1 ./ (ω_modes.^2 .- Ω_drive'.^2 .+ 2im .* γ .* ω_modes .* Ω_drive')
end

"""
    steady_state_amplitudes(prop, P, Ω_freqs, T) → real vector a [n_modes]

Compute modal amplitudes at time T from the steady-state response to
sinusoidal actuators with complex phasors P [n_act × n_freq].

P_ik = X_ik + i·Y_ik encodes amplitude and phase of actuator i at frequency k:
the physical signal is q_i(t) = Σ_k Im[P_ik · e^{iΩ_k t}].

The modal amplitudes are **linear** in P (and in the real parameters X, Y).
"""
function steady_state_amplitudes(prop::Propagator, P::AbstractMatrix{<:Complex},
                                  Ω_freqs::AbstractVector, T::Real)
    γ = prop.sim.tank.damping
    H = transfer_matrix(prop.ω, Ω_freqs, γ)       # [n_modes × n_freq]
    α = H .* (prop.C * P)                           # [n_modes × n_freq]
    E = exp.(im .* Ω_freqs .* T)                    # [n_freq]
    a = imag(α * E)                                  # [n_modes]
    return a
end

# ── Evaluate surface ─────────────────────────────────────────────────

function evaluate_surface(prop::Propagator, T::Real)
    (; xs, ys, nx, ny) = prop

    a = evaluate_modal_amplitudes(prop, T)

    if size(prop.Φ) == (0, 0)
        η, _, _ = reconstruct_surface(prop, a)
    else
        η = reshape(prop.Φ' * a, nx, ny)
    end

    return xs, ys, η
end

# ── Separable surface reconstruction ────────────────────────────────

"""
    reconstruct_surface(prop, a) → (η, dηdx, dηdy)  each [nx × ny]

Reconstruct surface height and gradients from flat modal amplitudes `a`
using separable 1D basis matrices. O(n_modes × ny × (n_modes + nx)) instead
of O(n_total × nx × ny) for the dense Φ approach.

Uses `scatter_add` (Zygote-compatible) to map flat modes → 2D mode grid,
then two matrix multiplies per field.
"""
function reconstruct_surface(prop::Propagator, a::AbstractVector)
    n_modes = prop.sim.n_modes
    # Scatter flat mode amplitudes to 2D grid [n_modes × n_modes]
    a_2d = reshape(scatter_add(a, prop.lin_2d, n_modes * n_modes), n_modes, n_modes)
    # Separable reconstruction: η[i,j] = Σ_m Σ_n a[m,n] cos_x[i,m] cos_y[j,n]
    η     = prop.cos_x  * a_2d * prop.cos_y'
    dηdx  = prop.dcos_x * a_2d * prop.cos_y'
    dηdy  = prop.cos_x  * a_2d * prop.dcos_y'
    return η, dηdx, dηdy
end

"""
    reconstruct_surface_hessian(prop, a) → (η, dηdx, dηdy, d²ηdx², d²ηdy², d²ηdxdy)

Like `reconstruct_surface` but also returns second derivatives.
"""
function reconstruct_surface_hessian(prop::Propagator, a::AbstractVector)
    n_modes = prop.sim.n_modes
    a_2d = reshape(scatter_add(a, prop.lin_2d, n_modes * n_modes), n_modes, n_modes)
    η      = prop.cos_x   * a_2d * prop.cos_y'
    dηdx   = prop.dcos_x  * a_2d * prop.cos_y'
    dηdy   = prop.cos_x   * a_2d * prop.dcos_y'
    d2ηdx2 = prop.d2cos_x * a_2d * prop.cos_y'
    d2ηdy2 = prop.cos_x   * a_2d * prop.d2cos_y'
    d2ηdxdy = prop.dcos_x * a_2d * prop.dcos_y'
    return η, dηdx, dηdy, d2ηdx2, d2ηdy2, d2ηdxdy
end

# ── AD utilities ─────────────────────────────────────────────────────

function scatter_add(vals::AbstractVector, indices::AbstractVector{<:Integer}, n::Int)
    dst = zeros(eltype(vals), n)
    for k in eachindex(vals)
        dst[indices[k]] += vals[k]
    end
    return dst
end

Zygote.@adjoint function scatter_add(vals, indices, n)
    dst = scatter_add(vals, indices, n)
    back(Δ) = (Δ[indices], nothing, nothing)
    return dst, back
end

# ── Refraction ──────────────────────────────────────────────────────

"""
    snell_landing(X_src, Y_src, η, dηdx, dηdy, depth, n_water)

Compute ray landing positions on the tank floor using full vector Snell's law.

Incident ray d̂_i = (0, 0, -1). Surface normal n̂ = normalize(-∂η/∂x, -∂η/∂y, 1).
Refracted ray is traced from the surface point (x, y, η) to z = 0.

All inputs/outputs are [nx × ny] arrays (pure broadcast, Zygote-compatible).
"""
function snell_landing(X_src, Y_src, η, dηdx, dηdy, depth, n_water)
    # Surface normal components (unnormalized): (-dηdx, -dηdy, 1)
    inv_norm = 1.0 ./ sqrt.(dηdx.^2 .+ dηdy.^2 .+ 1.0)
    nx_s = .-dηdx .* inv_norm
    ny_s = .-dηdy .* inv_norm
    nz_s = inv_norm

    # Incident ray: d_i = (0, 0, -1)
    # cos(θ_i) = -d_i · n̂ = nz_s
    cos_i = nz_s

    # Snell's law: n_air * sin(θ_i) = n_water * sin(θ_t)
    # sin²(θ_i) = 1 - cos²(θ_i)
    ratio = 1.0 / n_water   # n_air / n_water
    sin2_i = 1.0 .- cos_i.^2
    sin2_t = ratio^2 .* sin2_i

    # cos(θ_t) — clamp to avoid sqrt of negative (total internal reflection edge case)
    cos_t = sqrt.(max.(1.0 .- sin2_t, 0.0))

    # Refracted ray direction: d_t = ratio * d_i + (ratio * cos_i - cos_t) * n̂
    # d_i = (0, 0, -1), so:
    coeff = ratio .* cos_i .- cos_t
    dt_x = coeff .* nx_s
    dt_y = coeff .* ny_s
    dt_z = ratio .* (-1.0) .+ coeff .* nz_s

    # Trace from surface point (X_src, Y_src, η) along d_t to z = 0
    # z(t) = η + dt_z * t = 0  →  t = -η / dt_z
    t_hit = .-depth ./ dt_z

    x_land = X_src .+ dt_x .* t_hit
    y_land = Y_src .+ dt_y .* t_hit

    return x_land, y_land
end

# ── Caustic rendering ──────────────────────────────────────────────

function caustic_image(prop::Propagator, T::Real;
                       n_water=1.33, sigma=0.0, cutoff_sigmas=4.0, full_snell=false)
    a = evaluate_modal_amplitudes(prop, T)
    return caustic_image(prop, a; n_water=n_water, sigma=sigma,
                         cutoff_sigmas=cutoff_sigmas, full_snell=full_snell)
end

"""
    caustic_image(prop, a; n_water=1.33, sigma=0.0, cutoff_sigmas=4.0,
                  use_separable=nothing, full_snell=false)

Render caustic image from pre-computed modal amplitudes `a`.
This method is Zygote-differentiable w.r.t. `a`.

Set `full_snell=true` for physically correct vector Snell's law refraction
(matters for steep waves). Default is the paraxial approximation.
"""
function caustic_image(prop::Propagator, a::AbstractVector;
                       n_water=1.33, sigma=0.0, cutoff_sigmas=4.0,
                       use_separable=nothing, full_snell=false)
    (; sim, xs, ys, nx, ny) = prop
    depth = sim.tank.depth

    # Choose reconstruction method: separable if requested or if dense Φ is empty
    _use_sep = use_separable === nothing ? size(prop.Φ) == (0, 0) : use_separable

    if _use_sep
        η, dηdx, dηdy = reconstruct_surface(prop, a)
    else
        η     = reshape(prop.Φ' * a,      nx, ny)
        dηdx  = reshape(prop.dΦ_dx' * a,  nx, ny)
        dηdy  = reshape(prop.dΦ_dy' * a,  nx, ny)
    end

    # Default sigma: 1.5 × max grid spacing
    dx = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    σ = sigma > 0 ? sigma : 1.5 * max(dx, dy)

    σ2 = σ * σ
    inv_2σ2 = 1.0 / (2.0 * σ2)

    w = Zygote.ignore() do
        ceil(Int, cutoff_sigmas * σ / max(dx, dy))
    end

    # Phase A — Landing positions
    X_src = prop.X_src
    Y_src = prop.Y_src

    if full_snell
        x_land, y_land = snell_landing(X_src, Y_src, η, dηdx, dηdy, depth, n_water)
    else
        ratio = 1.0 / n_water
        x_land = X_src .+ (depth .- η) .* dηdx .* ratio
        y_land = Y_src .+ (depth .- η) .* dηdy .* ratio
    end

    # Phase B — Bilinear splatting via scatter_add (Zygote-compatible)
    n_pix = nx * ny
    fi_raw = (x_land .- xs[1]) ./ dx .+ 1.0
    fj_raw = (y_land .- ys[1]) ./ dy .+ 1.0

    # Rays landing outside the tank floor contribute zero (they hit the wall)
    fi = clamp.(fi_raw, 1.0, Float64(nx))   # clamp for safe indexing
    fj = clamp.(fj_raw, 1.0, Float64(ny))
    mask = Zygote.ignore() do
        Float64.((fi_raw .>= 1.0) .& (fi_raw .<= Float64(nx)) .&
                 (fj_raw .>= 1.0) .& (fj_raw .<= Float64(ny)))
    end

    # Integer indices — no gradient needed
    ix0, iy0, lin00, lin10, lin01, lin11 = Zygote.ignore() do
        ix0_ = clamp.(floor.(Int, fi), 1, nx - 1)
        iy0_ = clamp.(floor.(Int, fj), 1, ny - 1)
        lin00_ = ix0_      .+ (iy0_ .- 1) .* nx
        lin10_ = (ix0_.+1) .+ (iy0_ .- 1) .* nx
        lin01_ = ix0_      .+ iy0_        .* nx
        lin11_ = (ix0_.+1) .+ iy0_        .* nx
        (ix0_, iy0_, lin00_, lin10_, lin01_, lin11_)
    end

    wx = fi .- Float64.(ix0)
    wy = fj .- Float64.(iy0)

    w00 = mask .* (1.0 .- wx) .* (1.0 .- wy)
    w10 = mask .* wx           .* (1.0 .- wy)
    w01 = mask .* (1.0 .- wx) .* wy
    w11 = mask .* wx           .* wy

    all_vals    = vcat(vec(w00), vec(w10), vec(w01), vec(w11))
    all_indices = vcat(vec(lin00), vec(lin10), vec(lin01), vec(lin11))
    D = reshape(scatter_add(all_vals, all_indices, n_pix), nx, ny)

    # Phase C — Separable Gaussian convolution (2 × O(w) instead of O(w²))
    I = _gaussian_blur_separable(D, dx, dy, σ, w)

    return xs, ys, I
end

# ── Gaussian blur (mutation-free, separable) ────────────────────────

"""
    _gaussian_blur_separable(M, dx, dy, σ, w)

Separable 2D Gaussian blur: two 1D passes of width (2w+1) each.
O(2 × (2w+1) × n_pixels) instead of O((2w+1)² × n_pixels).
Zygote-compatible (no mutation).
"""
function _gaussian_blur_separable(M::AbstractMatrix, dx, dy, σ, w)
    inv_2σ2 = 1.0 / (2.0 * σ * σ)
    nx, ny = size(M)

    # Precompute 1D kernel weights (constants w.r.t. differentiated vars)
    wx, wy = Zygote.ignore() do
        ([exp(-(di * dx)^2 * inv_2σ2) for di in -w:w],
         [exp(-(dj * dy)^2 * inv_2σ2) for dj in -w:w])
    end

    # Pass 1: blur along x (rows)
    M_padx = vcat(zeros(w, ny), M, zeros(w, ny))
    tmp = sum(wx[i] .* M_padx[i:nx+i-1, :] for i in 1:2w+1)

    # Pass 2: blur along y (columns)
    tmp_pady = hcat(zeros(nx, w), tmp, zeros(nx, w))
    return sum(wy[j] .* tmp_pady[:, j:ny+j-1] for j in 1:2w+1)
end

function gaussian_blur(M::AbstractMatrix{<:Real}, dx, dy, σ; cutoff_sigmas=4.0)
    σ <= 0 && return Float64.(M)
    w = Zygote.ignore() do
        ceil(Int, cutoff_sigmas * σ / max(dx, dy))
    end
    return _gaussian_blur_separable(Float64.(M), dx, dy, σ, w)
end

# ── Loss helpers ─────────────────────────────────────────────────────

function _cosine_loss(I::AbstractMatrix, T_b::AbstractMatrix, norm_T::Real)
    dot_IT = sum(I .* T_b)
    norm_I = sqrt(sum(I .^ 2) + 1e-12)
    return 1.0 - dot_IT / (norm_I * norm_T)
end

function _ssim_loss(I::AbstractMatrix, T_b::AbstractMatrix, dx, dy; σ_ssim=nothing)
    σ_w = σ_ssim === nothing ? 1.5 * max(dx, dy) : σ_ssim

    # Scale-normalize I to match T_b's mean
    n = length(I)
    mean_T = sum(T_b) / n
    mean_I = sum(I) / n + 1e-12
    I_n = I .* (mean_T / mean_I)

    # Stability constants
    L = Zygote.ignore() do; maximum(T_b) end
    C1 = (0.01 * L)^2
    C2 = (0.03 * L)^2

    # Normalization: gaussian_blur computes weighted sums, not means.
    # Divide by W to get proper local averages (also handles boundary effects).
    W = Zygote.ignore() do
        gaussian_blur(ones(size(I_n)), dx, dy, σ_w)
    end

    # Local statistics via normalized Gaussian-weighted windows
    μ_x  = gaussian_blur(I_n, dx, dy, σ_w) ./ W
    μ_y  = Zygote.ignore() do; gaussian_blur(T_b, dx, dy, σ_w) ./ W end
    σ_x2 = max.(gaussian_blur(I_n .^ 2, dx, dy, σ_w) ./ W .- μ_x .^ 2, 0.0)
    σ_y2 = Zygote.ignore() do
        max.(gaussian_blur(T_b .^ 2, dx, dy, σ_w) ./ W .- μ_y .^ 2, 0.0)
    end
    σ_xy = gaussian_blur(I_n .* T_b, dx, dy, σ_w) ./ W .- μ_x .* μ_y

    ssim_map = ((2.0 .* μ_x .* μ_y .+ C1) .* (2.0 .* σ_xy .+ C2)) ./
               ((μ_x .^ 2 .+ μ_y .^ 2 .+ C1) .* (σ_x2 .+ σ_y2 .+ C2))
    return 1.0 - sum(ssim_map) / n
end

# ── Jacobian-based caustic intensity ─────────────────────────────────

"""
    caustic_image_jacobian(prop, a; n_water=1.33, ε=1e-2)

Render caustic intensity from the Jacobian determinant of the ray map.
Intensity I(x,y) = 1 / (|det(J)| + ε), giving analytically sharp caustic
lines wherever det(J) → 0.

Requires separable basis matrices (always available in Propagator).
Suitable for high-quality visualization; use splatting-based `caustic_image`
for optimization (smoother loss landscape).
"""
function caustic_image_jacobian(prop::Propagator, a::AbstractVector;
                                 n_water=1.33, ε=1e-2, sigma=0.0)
    (; xs, ys, nx, ny) = prop
    depth = prop.sim.tank.depth

    η, dηdx, dηdy, d2ηdx2, d2ηdy2, d2ηdxdy = reconstruct_surface_hessian(prop, a)

    r = 1.0 / n_water

    # Jacobian of the ray map (x,y) → (x_land, y_land):
    # J₁₁ = 1 + r·[(d-η)·η_xx - η_x²]
    # J₁₂ = r·[(d-η)·η_xy - η_x·η_y]
    # J₂₁ = J₁₂  (symmetric for paraxial)
    # J₂₂ = 1 + r·[(d-η)·η_yy - η_y²]
    h = depth .- η
    J11 = 1.0 .+ r .* (h .* d2ηdx2  .- dηdx.^2)
    J22 = 1.0 .+ r .* (h .* d2ηdy2  .- dηdy.^2)
    J12 = r .* (h .* d2ηdxdy .- dηdx .* dηdy)

    det_J = J11 .* J22 .- J12.^2

    I = 1.0 ./ (abs.(det_J) .+ ε)

    if sigma > 0
        dx = xs[2] - xs[1]
        dy = ys[2] - ys[1]
        I = gaussian_blur(I, dx, dy, sigma)
    end

    return xs, ys, I
end

# ── Differentiable loss factory ──────────────────────────────────────

"""
    make_caustic_loss(prop, T_time, target, ω_freqs; ...) → loss(params)

Returns a closure `loss(params::AbstractVector) -> scalar` that is
differentiable with Zygote. The full pipeline is:

    params → (A_mat, φ_mat) → Q → a → I → L

The image-match term uses cosine similarity (1 - cos_sim), making it
invariant to the absolute brightness of the simulated caustic.

Keyword arguments:
- `n_water=1.33`: refractive index
- `sigma=0.0`: caustic rendering blur
- `σ_blur=0.0`: additional blur for image-match term
- `λ_energy=0.0`: penalty on sum of squared amplitudes
- `λ_smooth=0.0`: penalty on high-frequency content
"""
function make_caustic_loss(prop::Propagator, T_time::Real,
                           target::Matrix{<:Real}, ω_freqs::AbstractVector;
                           n_water=1.33, sigma=0.0, σ_blur=0.0,
                           λ_energy=0.0, λ_smooth=0.0,
                           loss_type::Symbol=:cosine, σ_ssim=nothing)
    n_act = length(prop.sim.actuators)
    n_freq = length(ω_freqs)
    t_grid = prop.t_grid
    dx = prop.xs[2] - prop.xs[1]
    dy = prop.ys[2] - prop.ys[1]

    # Pre-blur target once (constant w.r.t. params)
    T_b = gaussian_blur(Float64.(target), dx, dy, σ_blur)
    norm_T = sqrt(sum(T_b .^ 2) + 1e-12)

    function loss(params::AbstractVector)
        A_mat, φ_mat = unpack_params(params, n_act, n_freq)

        # Forward pass
        Q = params_to_Q(A_mat, φ_mat, ω_freqs, t_grid)
        a = evaluate_modal_amplitudes(prop, T_time, Q)
        _, _, I = caustic_image(prop, a; n_water=n_water, sigma=sigma)

        # Term 1: image match
        L_match = loss_type == :ssim ? _ssim_loss(I, T_b, dx, dy; σ_ssim) :
                                       _cosine_loss(I, T_b, norm_T)

        # Term 2: energy penalty Σ A²
        L_energy = sum(A_mat .^ 2)

        # Term 3: smoothness penalty ½ Σ (A·ω)²
        L_smooth = 0.5 * sum((A_mat .* ω_freqs') .^ 2)

        return L_match + λ_energy * L_energy + λ_smooth * L_smooth
    end

    return loss
end

"""
    make_caustic_loss_refining(prop, T_time, target, ω_freqs,
                               sigma_ref, σ_blur_ref; ...) → loss(params)

Like `make_caustic_loss`, but takes `Ref{Float64}` wrappers for `sigma` and
`σ_blur` so the caller can mutate them between iterations for progressive
coarse-to-fine annealing.  The target is re-blurred inside the closure on
every call (since `σ_blur` changes over time).
"""
function make_caustic_loss_refining(prop::Propagator, T_time::Real,
                                    target::Matrix{<:Real}, ω_freqs::AbstractVector,
                                    sigma_ref::Ref{Float64}, σ_blur_ref::Ref{Float64};
                                    n_water=1.33, λ_energy=0.0, λ_smooth=0.0,
                                    loss_type::Symbol=:cosine, σ_ssim=nothing)
    n_act = length(prop.sim.actuators)
    n_freq = length(ω_freqs)
    t_grid = prop.t_grid
    dx = prop.xs[2] - prop.xs[1]
    dy = prop.ys[2] - prop.ys[1]
    target_f64 = Float64.(target)

    # Cache blurred target and its norm
    cached_σ_blur = Ref(-1.0)
    cached_T_b = Ref(target_f64)
    cached_norm_T = Ref(sqrt(sum(target_f64 .^ 2) + 1e-12))

    function loss(params::AbstractVector)
        A_mat, φ_mat = unpack_params(params, n_act, n_freq)

        # Read current sigma values from refs
        sigma = sigma_ref[]
        σ_blur = σ_blur_ref[]

        # Forward pass
        Q = params_to_Q(A_mat, φ_mat, ω_freqs, t_grid)
        a = evaluate_modal_amplitudes(prop, T_time, Q)
        _, _, I = caustic_image(prop, a; n_water=n_water, sigma=sigma)

        # Recompute blurred target only when σ_blur changes
        T_b = Zygote.ignore() do
            if σ_blur != cached_σ_blur[]
                cached_σ_blur[] = σ_blur
                cached_T_b[] = gaussian_blur(target_f64, dx, dy, σ_blur)
                cached_norm_T[] = sqrt(sum(cached_T_b[] .^ 2) + 1e-12)
            end
            cached_T_b[]
        end
        norm_T = Zygote.ignore() do; cached_norm_T[] end

        # Term 1: image match
        L_match = loss_type == :ssim ? _ssim_loss(I, T_b, dx, dy; σ_ssim) :
                                       _cosine_loss(I, T_b, norm_T)

        # Term 2: energy penalty Σ A²
        L_energy = sum(A_mat .^ 2)

        # Term 3: smoothness penalty ½ Σ (A·ω)²
        L_smooth = 0.5 * sum((A_mat .* ω_freqs') .^ 2)

        return L_match + λ_energy * L_energy + λ_smooth * L_smooth
    end

    return loss
end

# ── Steady-state differentiable loss factories ──────────────────────

"""
    make_caustic_loss_ss(prop, T, target, Ω_freqs; ...) → loss(params)

Steady-state version of `make_caustic_loss`. Parameters are complex phasors
packed as [vec(Re(P)); vec(Im(P))]. The pipeline is:

    params → P_complex → a (via transfer function) → I → L

The wave physics is **linear** in parameters — only the optics is nonlinear.
"""
function make_caustic_loss_ss(prop::Propagator, T_time::Real,
                               target::Matrix{<:Real}, Ω_freqs::AbstractVector;
                               n_water=1.33, sigma=0.0, σ_blur=0.0,
                               λ_energy=0.0,
                               loss_type::Symbol=:cosine, σ_ssim=nothing,
                               n_temporal::Int=1, σ_temporal::Float64=0.0,
                               λ_temporal::Float64=0.0)
    n_act = length(prop.sim.actuators)
    n_freq = length(Ω_freqs)
    dx = prop.xs[2] - prop.xs[1]
    dy = prop.ys[2] - prop.ys[1]

    # Pre-blur target once (constant w.r.t. params)
    T_b = gaussian_blur(Float64.(target), dx, dy, σ_blur)
    norm_T = sqrt(sum(T_b .^ 2) + 1e-12)

    # Pre-compute transfer matrix (constant w.r.t. params)
    γ = prop.sim.tank.damping
    H = transfer_matrix(prop.ω, Ω_freqs, γ)

    # Precompute temporal sample offsets and weights
    δs, ws = if n_temporal > 1
        δs_ = collect(range(-3σ_temporal, 3σ_temporal, length=n_temporal))
        ws_ = [exp(-δ^2 / (2 * σ_temporal^2 + 1e-30)) for δ in δs_]
        ws_ ./= sum(ws_)
        (δs_, ws_)
    else
        (Float64[0.0], Float64[1.0])
    end

    function loss(params::AbstractVector)
        X, Y = unpack_complex(params, n_act, n_freq)
        P = X .+ im .* Y

        # Steady-state modal amplitudes (linear in P)
        α = H .* (prop.C * P)                       # [n_modes × n_freq]

        # Image match: weighted sum over time samples
        L_match = zero(eltype(params))
        for k in 1:length(δs)
            E_k = exp.(im .* Ω_freqs .* (T_time + δs[k]))
            a_k = imag(α * E_k)
            _, _, I_k = caustic_image(prop, a_k; n_water=n_water, sigma=sigma)
            L_k = loss_type == :ssim ? _ssim_loss(I_k, T_b, dx, dy; σ_ssim) :
                                       _cosine_loss(I_k, T_b, norm_T)
            L_match = L_match + ws[k] * L_k
        end

        # Temporal gradient penalty: ‖da/dt‖² at T
        if λ_temporal > 0
            E_T = exp.(im .* Ω_freqs .* T_time)
            dadt = real(α * (im .* Ω_freqs .* E_T))
            L_match = L_match + λ_temporal * sum(dadt .^ 2)
        end

        # Energy penalty on phasor magnitudes
        L_energy = sum(X .^ 2) + sum(Y .^ 2)

        return L_match + λ_energy * L_energy
    end

    return loss
end

"""
    make_caustic_loss_ss_refining(prop, T, target, Ω_freqs,
                                  sigma_ref, σ_blur_ref; ...) → loss(params)

Steady-state version with mutable sigma refs for coarse-to-fine annealing.
"""
function make_caustic_loss_ss_refining(prop::Propagator, T_time::Real,
                                       target::Matrix{<:Real}, Ω_freqs::AbstractVector,
                                       sigma_ref::Ref{Float64}, σ_blur_ref::Ref{Float64};
                                       n_water=1.33, λ_energy=0.0,
                                       loss_type::Symbol=:cosine, σ_ssim=nothing,
                                       n_temporal::Int=1, σ_temporal::Float64=0.0,
                                       λ_temporal::Float64=0.0)
    n_act = length(prop.sim.actuators)
    n_freq = length(Ω_freqs)
    dx = prop.xs[2] - prop.xs[1]
    dy = prop.ys[2] - prop.ys[1]
    target_f64 = Float64.(target)

    γ = prop.sim.tank.damping
    H = transfer_matrix(prop.ω, Ω_freqs, γ)

    # Cache blurred target and its norm
    cached_σ_blur = Ref(-1.0)
    cached_T_b = Ref(target_f64)
    cached_norm_T = Ref(sqrt(sum(target_f64 .^ 2) + 1e-12))

    # Precompute temporal sample offsets and weights
    δs, ws = if n_temporal > 1
        δs_ = collect(range(-3σ_temporal, 3σ_temporal, length=n_temporal))
        ws_ = [exp(-δ^2 / (2 * σ_temporal^2 + 1e-30)) for δ in δs_]
        ws_ ./= sum(ws_)
        (δs_, ws_)
    else
        (Float64[0.0], Float64[1.0])
    end

    function loss(params::AbstractVector)
        X, Y = unpack_complex(params, n_act, n_freq)
        P = X .+ im .* Y

        sigma = sigma_ref[]
        σ_blur = σ_blur_ref[]

        # Steady-state modal amplitudes
        α = H .* (prop.C * P)

        # Recompute blurred target only when σ_blur changes
        T_b = Zygote.ignore() do
            if σ_blur != cached_σ_blur[]
                cached_σ_blur[] = σ_blur
                cached_T_b[] = gaussian_blur(target_f64, dx, dy, σ_blur)
                cached_norm_T[] = sqrt(sum(cached_T_b[] .^ 2) + 1e-12)
            end
            cached_T_b[]
        end
        norm_T = Zygote.ignore() do; cached_norm_T[] end

        # Image match: weighted sum over time samples
        L_match = zero(eltype(params))
        for k in 1:length(δs)
            E_k = exp.(im .* Ω_freqs .* (T_time + δs[k]))
            a_k = imag(α * E_k)
            _, _, I_k = caustic_image(prop, a_k; n_water=n_water, sigma=sigma)
            L_k = loss_type == :ssim ? _ssim_loss(I_k, T_b, dx, dy; σ_ssim) :
                                       _cosine_loss(I_k, T_b, norm_T)
            L_match = L_match + ws[k] * L_k
        end

        # Temporal gradient penalty: ‖da/dt‖² at T
        if λ_temporal > 0
            E_T = exp.(im .* Ω_freqs .* T_time)
            dadt = real(α * (im .* Ω_freqs .* E_T))
            L_match = L_match + λ_temporal * sum(dadt .^ 2)
        end

        L_energy = sum(X .^ 2) + sum(Y .^ 2)

        return L_match + λ_energy * L_energy
    end

    return loss
end

"""
    make_caustic_loss_ss_keyframes(prop, keyframes, Ω_freqs,
                                   sigma_ref, σ_blur_ref; ...) → loss(params)

Multi-keyframe steady-state loss. `keyframes` is a vector of NamedTuples
`(t=..., target=..., weight=...)` specifying target images at different times.
Supports temporal windowing and da/dt penalty per keyframe.
"""
function make_caustic_loss_ss_keyframes(
        prop::Propagator,
        keyframes::Vector{<:NamedTuple},
        Ω_freqs::AbstractVector,
        sigma_ref::Ref{Float64}, σ_blur_ref::Ref{Float64};
        n_water=1.33, λ_energy=0.0,
        loss_type::Symbol=:cosine, σ_ssim=nothing,
        n_temporal::Int=1, σ_temporal::Float64=0.0,
        λ_temporal::Float64=0.0)

    n_act = length(prop.sim.actuators)
    n_freq = length(Ω_freqs)
    dx = prop.xs[2] - prop.xs[1]
    dy = prop.ys[2] - prop.ys[1]
    n_kf = length(keyframes)

    γ = prop.sim.tank.damping
    H = transfer_matrix(prop.ω, Ω_freqs, γ)

    # Normalize keyframe weights
    kf_weights = [Float64(kf.weight) for kf in keyframes]
    kf_weights ./= sum(kf_weights)
    kf_times = [Float64(kf.t) for kf in keyframes]
    kf_targets = [Float64.(kf.target) for kf in keyframes]

    # Precompute temporal sample offsets and weights
    δs, ws = if n_temporal > 1
        δs_ = collect(range(-3σ_temporal, 3σ_temporal, length=n_temporal))
        ws_ = [exp(-δ^2 / (2 * σ_temporal^2 + 1e-30)) for δ in δs_]
        ws_ ./= sum(ws_)
        (δs_, ws_)
    else
        (Float64[0.0], Float64[1.0])
    end

    # Cache blurred targets — one per keyframe, recompute when σ_blur changes
    cached_σ_blur = Ref(-1.0)
    cached_T_bs = Ref(kf_targets)
    cached_norm_Ts = Ref([sqrt(sum(t .^ 2) + 1e-12) for t in kf_targets])

    function loss(params::AbstractVector)
        X, Y = unpack_complex(params, n_act, n_freq)
        P = X .+ im .* Y

        sigma = sigma_ref[]
        σ_blur = σ_blur_ref[]

        α = H .* (prop.C * P)

        # Recompute blurred targets when σ_blur changes
        T_bs = Zygote.ignore() do
            if σ_blur != cached_σ_blur[]
                cached_σ_blur[] = σ_blur
                blurred = [gaussian_blur(t, dx, dy, σ_blur) for t in kf_targets]
                cached_T_bs[] = blurred
                cached_norm_Ts[] = [sqrt(sum(b .^ 2) + 1e-12) for b in blurred]
            end
            cached_T_bs[]
        end
        norm_Ts = Zygote.ignore() do; cached_norm_Ts[] end

        # Sum weighted loss across keyframes
        L_match = zero(eltype(params))
        for j in 1:n_kf
            T_b_j = T_bs[j]
            norm_T_j = norm_Ts[j]
            t_j = kf_times[j]

            # Temporal window around this keyframe
            for k in 1:length(δs)
                E_k = exp.(im .* Ω_freqs .* (t_j + δs[k]))
                a_k = imag(α * E_k)
                _, _, I_k = caustic_image(prop, a_k; n_water=n_water, sigma=sigma)
                L_k = loss_type == :ssim ? _ssim_loss(I_k, T_b_j, dx, dy; σ_ssim) :
                                           _cosine_loss(I_k, T_b_j, norm_T_j)
                L_match = L_match + kf_weights[j] * ws[k] * L_k
            end

            # da/dt penalty at this keyframe's time
            if λ_temporal > 0
                E_T = exp.(im .* Ω_freqs .* t_j)
                dadt = real(α * (im .* Ω_freqs .* E_T))
                L_match = L_match + λ_temporal * sum(dadt .^ 2) / n_kf
            end
        end

        L_energy = sum(X .^ 2) + sum(Y .^ 2)

        return L_match + λ_energy * L_energy
    end

    return loss
end

# ── Loss function (original, non-AD) ────────────────────────────────
#
#   L = ImageMatch(I, T) + λ_energy Σ Aᵢ² + λ_smooth Smoothness(q)
#
# ImageMatch: Gaussian-blurred L2 distance between caustic and target.
# Energy:    Penalises large actuator amplitudes (prevents huge waves).
# Smoothness: Penalises high-frequency content in actuator signals.
#   - SineSum:  analytic  ½ Σ (Aₙ ωₙ)²  (time-averaged |dq/dt|²)
#   - Generic:  finite-difference  Σ (Δq/Δt)² Δt

function caustic_loss(prop::Propagator, T_time::Real, target::Matrix{<:Real};
                      n_water=1.33, sigma=0.0,
                      σ_blur=0.0, λ_energy=0.0, λ_smooth=0.0)

    _, _, I = caustic_image(prop, T_time; n_water=n_water, sigma=sigma)

    dx = prop.xs[2] - prop.xs[1]
    dy = prop.ys[2] - prop.ys[1]

    # ── Term 1: blurred image-match ──
    I_b = gaussian_blur(I, dx, dy, σ_blur)
    T_b = gaussian_blur(Float64.(target), dx, dy, σ_blur)
    L_match = sum((I_b .- T_b) .^ 2) * dx * dy      # integrate over area

    # ── Term 2: energy penalty  Σ Aᵢₙ² ──
    L_energy = 0.0
    for act in prop.sim.actuators
        if act.forcing isa SineSum
            L_energy += sum(act.forcing.A .^ 2)
        end
    end

    # ── Term 3: smoothness penalty ──
    L_smooth = 0.0
    for act in prop.sim.actuators
        f = act.forcing
        if f isa SineSum
            # Time-averaged (dq/dt)² = ½ Σ (Aₙ ωₙ)²
            L_smooth += 0.5 * sum((f.A .* f.ω) .^ 2)
        else
            # Finite-difference fallback for generic callables
            t_grid = prop.t_grid
            dt = prop.sim.dt
            q = f.(t_grid)
            dq = diff(q) ./ dt
            L_smooth += sum(dq .^ 2) * dt
        end
    end

    return L_match + λ_energy * L_energy + λ_smooth * L_smooth
end

# ── Visualization ────────────────────────────────────────────────────

function visualize(prop::Propagator; fps=30, clims=nothing)
    # Plots must be loaded by the caller
    Plots = Base.get_extension(Base, :Plots)  # won't work, use invokelatest pattern
    plt = try
        Main.Plots
    catch
        error("Plots.jl must be loaded before calling visualize(). Run `using Plots` first.")
    end

    (; sim, xs, ys) = prop
    t0, t1 = sim.tspan
    dt_frame = 1.0 / fps
    frames = t0:dt_frame:t1

    # Pre-evaluate to find good color limits if not provided
    if clims === nothing
        sample_times = range(t0 + 0.1, t1, length=min(10, length(frames)))
        maxval = 0.0
        for t in sample_times
            _, _, η = evaluate_surface(prop, t)
            maxval = max(maxval, maximum(abs, η))
        end
        maxval = max(maxval, 1e-10)
        clims = (-maxval, maxval)
    end

    anim = plt.Animation()
    for T in frames
        _, _, η = evaluate_surface(prop, T)
        p = plt.heatmap(xs, ys, η',
                xlabel="x (m)", ylabel="y (m)",
                title=@sprintf("Wave Tank  t = %5.2f s", T),
                color=:RdBu, clims=clims,
                aspect_ratio=:equal, size=(800, 400))
        for act in sim.actuators
            plt.scatter!(p, [act.x], [act.y], color=:black, markersize=6,
                     label=nothing, markershape=:diamond)
        end
        plt.frame(anim, p)
    end

    return plt.gif(anim, "wave_tank.gif", fps=fps)
end

function visualize_caustic(prop::Propagator;
                           fps=30, speed=1.0, clims=nothing, n_water=1.33,
                           sigma=0.0, filename="caustic.gif", Q=nothing)
    plt = try
        Main.Plots
    catch
        error("Plots.jl must be loaded before calling visualize_caustic(). Run `using Plots` first.")
    end

    (; sim, xs, ys) = prop
    t0, t1 = sim.tspan
    dt_frame = speed / fps
    frames = t0:dt_frame:t1

    # Pre-sample to auto-determine clims if not provided
    if clims === nothing
        sample_times = range(t0 + 0.1, t1, length=min(10, length(frames)))
        all_vals = Float64[]
        for t in sample_times
            if Q !== nothing
                a = evaluate_modal_amplitudes(prop, t, Q)
                _, _, I = caustic_image(prop, a; n_water=n_water, sigma=sigma)
            else
                _, _, I = caustic_image(prop, t; n_water=n_water, sigma=sigma)
            end
            append!(all_vals, vec(I))
        end
        sort!(all_vals)
        # Use 99.5th percentile so caustic lines are visible without
        # rare extreme peaks washing the colormap to black
        hi = all_vals[max(1, round(Int, 0.995 * length(all_vals)))]
        lo = all_vals[max(1, round(Int, 0.005 * length(all_vals)))]
        clims = (lo, max(hi, lo + 1e-10))
    end

    anim = plt.Animation()
    for T in frames
        if Q !== nothing
            a = evaluate_modal_amplitudes(prop, T, Q)
            _, _, I = caustic_image(prop, a; n_water=n_water, sigma=sigma)
        else
            _, _, I = caustic_image(prop, T; n_water=n_water, sigma=sigma)
        end
        p = plt.heatmap(xs, ys, I',
                xlabel="x (m)", ylabel="y (m)",
                title=@sprintf("Caustic Pattern  t = %5.2f s", T),
                color=:inferno, clims=clims,
                aspect_ratio=:equal, size=(800, 400))
        plt.frame(anim, p)
    end

    return plt.gif(anim, filename, fps=fps)
end

# ── Image target loading ──────────────────────────────────────────────

"""
    load_target_image(path, prop; invert=false) → Matrix{Float64}

Load an image file (JPG, PNG, etc.), convert to grayscale, and resample
to the propagator's `nx × ny` grid.  Returns a `Matrix{Float64}` in [0,1]
suitable for `make_caustic_loss`.

If `invert=true`, dark pixels become high target values (useful when the
subject is dark on a light background).
"""
function load_target_image(path::AbstractString, prop::Propagator; invert=false)
    img = FileIO.load(path)
    gray = Gray.(img)                           # convert to grayscale
    mat = Float64.(gray)                        # Matrix{Float64} in [0,1]
    if invert
        mat = 1.0 .- mat                        # dark regions → high target
    end
    # Resize to match propagator grid (nx × ny) using nearest-neighbor
    # img is (height, width) i.e. (ny_img, nx_img); we need (nx, ny)
    ny_img, nx_img = size(mat)
    nx, ny = length(prop.xs), length(prop.ys)
    target = zeros(nx, ny)
    for j in 1:ny
        for i in 1:nx
            xi = clamp(round(Int, (i - 1) / (nx - 1) * (nx_img - 1)) + 1, 1, nx_img)
            yi = clamp(round(Int, (j - 1) / (ny - 1) * (ny_img - 1)) + 1, 1, ny_img)
            target[i, j] = mat[yi, xi]
        end
    end
    # Normalize to [0, 1]
    target ./= max(maximum(target), 1e-10)
    return target
end

# ── Target analysis and auto-setup ────────────────────────────────────

"""
    analyze_target(target, tank; n_modes_max=50, energy_fraction=0.95) → NamedTuple

Analyze a target image's spatial frequency content relative to the tank's
modal structure. Returns suggested parameters for optimization setup.

The target is projected onto the 2D cosine eigenmode basis and each mode's
energy is mapped to the corresponding natural frequency via the dispersion
relation.
"""
function analyze_target(target::Matrix{<:Real}, tank::Tank;
                        n_modes_max::Int=50, energy_fraction::Float64=0.95)
    Lx, Ly, depth, g = tank.Lx, tank.Ly, tank.depth, tank.g
    nx, ny = size(target)

    # Build separable cosine basis on the target's grid
    xs = range(0, Lx, length=nx)
    ys = range(0, Ly, length=ny)
    # Indices 0:n_modes_max (include DC for completeness, exclude (0,0) mode later)
    ms = 0:n_modes_max
    ns = 0:n_modes_max
    cos_x = [cos(m * π * x / Lx) for x in xs, m in ms]  # [nx × (n_modes_max+1)]
    cos_y = [cos(n * π * y / Ly) for y in ys, n in ns]  # [ny × (n_modes_max+1)]

    # Project target onto eigenmode basis (unnormalized inner products)
    # coeffs[m+1, n+1] = Σ_ij target[i,j] * cos(mπx_i/Lx) * cos(nπy_j/Ly)
    raw_coeffs = cos_x' * Float64.(target) * cos_y  # [(n_modes_max+1) × (n_modes_max+1)]

    # Normalize by mode norm and grid size to get proper expansion coefficients
    # The inner product approximation: ∫∫ f·φ dx dy ≈ (Lx/nx)·(Ly/ny) · Σ f·φ
    dx_grid = Lx / nx
    dy_grid = Ly / ny
    coeffs = similar(raw_coeffs)
    for mi in 1:n_modes_max+1, ni in 1:n_modes_max+1
        m, n = mi - 1, ni - 1
        Ix = m == 0 ? Lx : Lx / 2
        Iy = n == 0 ? Ly : Ly / 2
        N_mn = Ix * Iy
        coeffs[mi, ni] = raw_coeffs[mi, ni] * dx_grid * dy_grid / N_mn
    end
    coeffs[1, 1] = 0.0  # exclude DC mode (0,0)

    # Energy per mode and natural frequencies
    energy = coeffs .^ 2
    E_total = sum(energy)

    # Compute natural frequency for each (m,n) mode
    freq_map = zeros(n_modes_max + 1, n_modes_max + 1)
    for mi in 1:n_modes_max+1, ni in 1:n_modes_max+1
        m, n = mi - 1, ni - 1
        (m == 0 && n == 0) && continue
        k = sqrt((m * π / Lx)^2 + (n * π / Ly)^2)
        freq_map[mi, ni] = sqrt(g * k * tanh(k * depth)) / (2π)  # Hz
    end

    # Sort modes by energy (descending), find set capturing energy_fraction
    mode_list = [(m=mi-1, n=ni-1, E=energy[mi,ni], f=freq_map[mi,ni])
                 for mi in 1:n_modes_max+1 for ni in 1:n_modes_max+1
                 if !(mi == 1 && ni == 1)]
    sort!(mode_list, by=x -> -x.E)

    cumE = cumsum([m.E for m in mode_list])
    n_needed = findfirst(>=(energy_fraction * E_total), cumE)
    n_needed = n_needed === nothing ? length(mode_list) : n_needed
    important_modes = mode_list[1:n_needed]

    # Suggested parameters
    m_max = maximum(m.m for m in important_modes)
    n_max = maximum(m.n for m in important_modes)
    freq_min = minimum(m.f for m in important_modes if m.f > 0)
    freq_max = maximum(m.f for m in important_modes)
    suggested_n_modes = max(m_max, n_max) + 1

    # Actuator count: ~2× highest mode index per wall side, all 4 sides
    n_act_per_side = max(m_max, n_max) + 1
    suggested_n_act = 4 * n_act_per_side

    # Suggested n_freq: roughly one per Hz in the range, at least 4
    suggested_n_freq = max(4, round(Int, 2 * (freq_max - freq_min) + 1))

    # Print summary
    println("=" ^ 60)
    println("  Target Analysis")
    println("=" ^ 60)
    println("  Grid: $(nx)×$(ny)")
    println("  Modes capturing $(round(energy_fraction*100))% energy: $n_needed / $(length(mode_list))")
    println("  Highest mode indices: m_max=$m_max, n_max=$n_max")
    println("  Frequency range: $(round(freq_min, digits=2)) – $(round(freq_max, digits=2)) Hz")
    println()
    println("  Suggested parameters:")
    println("    n_modes = $suggested_n_modes")
    println("    n_freq  = $suggested_n_freq  (range $(round(freq_min, digits=2))–$(round(freq_max, digits=2)) Hz)")
    println("    n_act   = $suggested_n_act  ($n_act_per_side per side)")
    println()
    println("  Top 10 modes by energy:")
    for (i, m) in enumerate(mode_list[1:min(10, end)])
        pct = m.E / E_total * 100
        println("    ($( m.m), $(m.n))  f=$(round(m.f, digits=2)) Hz  energy=$(round(pct, digits=1))%")
    end
    println("=" ^ 60)

    return (coeffs=coeffs, energy=energy, freq_map=freq_map,
            E_total=E_total, important_modes=important_modes,
            n_modes=suggested_n_modes, n_freq=suggested_n_freq,
            n_act=suggested_n_act, freq_min=freq_min, freq_max=freq_max,
            m_max=m_max, n_max=n_max, cos_x=cos_x, cos_y=cos_y)
end

"""
    setup_from_target(target, tank; energy_fraction=0.95, nx=100, ny=100,
                      n_modes_max=50, actuator_width=0.05,
                      T_eval=1.0, n_water=1.33) → NamedTuple

Automatically configure optimization setup from a target image. Returns:
- `prop`: Propagator ready for optimization
- `Ω_freqs`: driving frequencies (angular, rad/s)
- `target_bl`: band-limited target (unachievable spatial frequencies removed)
- `analysis`: full analysis results from `analyze_target`
- `p0`: analytical initial guess parameter vector from `analytical_solve`
"""
function setup_from_target(target::Matrix{<:Real}, tank::Tank;
                           energy_fraction::Float64=0.95, nx::Int=100, ny::Int=100,
                           n_modes_max::Int=50, actuator_width::Float64=0.05,
                           T_eval::Float64=1.0, n_water::Float64=1.33)
    Lx, Ly = tank.Lx, tank.Ly

    # Analyze on the target's own grid
    analysis = analyze_target(target, tank; n_modes_max, energy_fraction)

    n_modes = analysis.n_modes
    n_freq = analysis.n_freq
    freq_min = analysis.freq_min
    freq_max = analysis.freq_max
    n_act_per_side = max(analysis.m_max, analysis.n_max) + 1

    # Build driving frequencies (evenly spaced in the suggested range)
    freqs = collect(range(freq_min, freq_max, length=n_freq))
    Ω_freqs = 2π .* freqs

    # Place actuators evenly around perimeter
    positions = Tuple{Float64,Float64}[]
    for x in range(0, Lx, length=n_act_per_side + 2)[2:end-1]
        push!(positions, (x, 0.0))   # bottom wall
        push!(positions, (x, Ly))    # top wall
    end
    for y in range(0, Ly, length=n_act_per_side + 2)[2:end-1]
        push!(positions, (0.0, y))   # left wall
        push!(positions, (Lx, y))    # right wall
    end
    n_act = length(positions)

    actuators = [
        Actuator(pos[1], pos[2],
                 SineSum(; freqs=freqs, A=zeros(n_freq), φ=zeros(n_freq));
                 width=actuator_width)
        for pos in positions
    ]

    sim = WaveSim(tank, actuators, (0.0, 5.0), 0.01; n_modes=n_modes)
    prop = build_propagator(sim; nx=nx, ny=ny, dense_basis=false)

    # Band-limit the target: reconstruct using only achievable modes
    # Reproject onto the propagator's grid
    cos_x_out = [cos(m * π * x / Lx) for x in prop.xs, m in 0:n_modes-1]
    cos_y_out = [cos(n * π * y / Ly) for y in prop.ys, n in 0:n_modes-1]
    # Use coefficients from analysis (truncated to n_modes)
    c = analysis.coeffs[1:n_modes, 1:n_modes]
    target_bl = cos_x_out * c * cos_y_out'
    target_bl = max.(target_bl, 0.0)
    bl_max = maximum(target_bl)
    if bl_max > 0
        target_bl ./= bl_max
    end

    println("\nSetup: $(n_act) actuators, $(n_freq) frequencies ($(round(freq_min,digits=2))–$(round(freq_max,digits=2)) Hz), $(length(prop.ω)) modes")
    println("Grid: $(nx)×$(ny), Parameters: $(2 * n_act * n_freq)")

    # Analytical solve for initial guess
    sol = analytical_solve(prop, target_bl, Ω_freqs, T_eval; n_water=n_water)

    println("Analytical solve: ‖a_desired‖ = $(round(norm(sol.a_desired), sigdigits=4)), ‖p0‖ = $(round(norm(sol.p0), sigdigits=4))")

    return (prop=prop, Ω_freqs=Ω_freqs, target_bl=target_bl, analysis=analysis,
            freqs=freqs, actuators=actuators, p0=sol.p0)
end

# ── Analytical solve ──────────────────────────────────────────────────

"""
    analytical_solve(prop, target, Ω_freqs, T_eval; n_water=1.33, max_contrast=0.5) → (p0, a_desired)

Compute actuator phasors analytically from a target caustic image using the
paraxial approximation I ≈ 1 - (depth/n_water)·∇²η.

Pipeline:
1. Project (target - 1) onto cosine eigenmodes → coefficients c_{m,n}
2. Poisson inversion: a_{m,n} = c_{m,n} · n_water / (depth · k²_{m,n})
3. Linear solve: find phasors P such that imag(H .* (C·P) · exp(iΩT)) = a_desired

Returns a NamedTuple with:
- `p0`: parameter vector [2·n_act·n_freq] in pack_complex format
- `a_desired`: target modal amplitudes [n_total]
"""
function analytical_solve(prop::Propagator, target::Matrix{<:Real},
                          Ω_freqs::AbstractVector, T_eval::Real;
                          n_water::Float64=1.33, max_contrast::Float64=0.5)
    (; sim, mode_m, mode_n, ω, C, cos_x, cos_y, xs, ys, nx, ny) = prop
    (; Lx, Ly, depth, damping) = sim.tank
    n_modes = sim.n_modes
    n_total = length(ω)
    n_act = size(C, 2)
    n_freq = length(Ω_freqs)

    @assert size(target) == (nx, ny) "Target size $(size(target)) must match propagator grid ($nx, $ny)"

    # ── 1. Project (I_target - 1) onto cosine eigenmode basis ──────────
    # The paraxial caustic formula is I ≈ 1 - (depth/n_water)·∇²η, where
    # I is the physical intensity (= 1 for flat surface, >1 at bright caustic).
    # The input `target` is normalized to [0,1], so we rescale to mean=1 to
    # get the physical intensity: I_target = target / mean(target).
    # This ensures energy conservation and the correct sign (bright → I>1 → ∇²η<0).
    dx = Lx / nx
    dy = Ly / ny
    t_mean = max(mean(Float64.(target)), 1e-6)
    residual = Float64.(target) ./ t_mean .- 1.0   # = I_target - 1, mean=0
    raw = cos_x' * residual * cos_y   # [n_modes × n_modes]

    # Normalize by grid spacing and mode norms
    coeffs = similar(raw)
    for mi in 1:n_modes, ni in 1:n_modes
        m, n = mi - 1, ni - 1
        Ix = m == 0 ? Lx : Lx / 2
        Iy = n == 0 ? Ly : Ly / 2
        coeffs[mi, ni] = raw[mi, ni] * dx * dy / (Ix * Iy)
    end
    coeffs[1, 1] = 0.0  # DC mode excluded

    # ── 2. Poisson inversion: a_{m,n} = c_{m,n} · n_water / (depth · k²) ──
    a_desired = zeros(n_total)
    for j in 1:n_total
        m, n = mode_m[j], mode_n[j]
        k2 = (m * π / Lx)^2 + (n * π / Ly)^2
        a_desired[j] = coeffs[m + 1, n + 1] * n_water / (depth * k2)
    end

    # ── 2b. Mask out modes the actuators can't drive ─────────────────
    # High-k modes are suppressed by the Gaussian blob factor exp(-σ²k²/2).
    # Including them in a_desired forces pinv to invert near-zero singular
    # values, blowing up p0 without actually producing those amplitudes.
    max_coupling = vec(maximum(abs.(C), dims=2))
    coupling_threshold = 1e-3 * maximum(max_coupling)
    achievable = max_coupling .>= coupling_threshold
    n_achievable = sum(achievable)
    a_desired .*= achievable
    println("  Analytical solve: $(n_achievable)/$(n_total) modes achievable (coupling > 1e-3 * max)")

    # ── 2c. Scale so caustic contrast ≤ max_contrast ─────────────────
    # The caustic intensity deviation is δI = -(depth/n_water)·∇²η,
    # and ∇²φ_{m,n} = -k²·φ_{m,n}, so we compute the implied Laplacian field.
    a_2d = zeros(n_modes, n_modes)
    k2_2d = zeros(n_modes, n_modes)
    for j in 1:n_total
        m, n = mode_m[j], mode_n[j]
        a_2d[m + 1, n + 1] = a_desired[j]
        k2_2d[m + 1, n + 1] = (m * π / Lx)^2 + (n * π / Ly)^2
    end
    lap_field = cos_x * (-k2_2d .* a_2d) * cos_y'   # ∇²η field [nx × ny]
    caustic_dev = (depth / n_water) .* lap_field       # = I - 1 implied
    caustic_range = maximum(abs, caustic_dev)
    if caustic_range > max_contrast
        scale = max_contrast / caustic_range
        a_desired .*= scale
        println("  Analytical solve: scaled caustic contrast $(round(caustic_range,digits=3)) → $max_contrast (scale=$(round(scale,sigdigits=3)))")
    else
        println("  Analytical solve: caustic contrast $(round(caustic_range,digits=3)) (within target $max_contrast)")
    end

    # ── 3. Build linear system M·θ = a_desired ────────────────────────
    # a = imag(H .* (C * P) * exp(iΩT)) is linear in θ = [vec(X); vec(Y)]
    H = transfer_matrix(ω, Ω_freqs, damping)   # [n_total × n_freq]
    E = exp.(im .* Ω_freqs .* T_eval)           # [n_freq]
    β = H .* E'                                  # [n_total × n_freq], β_{j,k} = H_{j,k}·e^{iΩ_k T}

    # M = [M_X  M_Y] where M_X[:,block_k] = diag(imag(β[:,k])) * C
    #                       M_Y[:,block_k] = diag(real(β[:,k])) * C
    # Only build rows for achievable modes to keep conditioning tractable
    M = zeros(n_total, 2 * n_act * n_freq)
    for k in 1:n_freq
        col_x = (k - 1) * n_act
        col_y = n_act * n_freq + (k - 1) * n_act
        for j in 1:n_total
            achievable[j] || continue
            bI = imag(β[j, k])
            bR = real(β[j, k])
            for i in 1:n_act
                M[j, col_x + i] = C[j, i] * bI
                M[j, col_y + i] = C[j, i] * bR
            end
        end
    end

    # ── 4. Solve via least-squares ────────────────────────────────────
    # rtol truncates near-zero singular values → bounded, physical p0
    θ = pinv(M; rtol=1e-3) * a_desired

    return (p0=θ, a_desired=a_desired)
end

end # module
