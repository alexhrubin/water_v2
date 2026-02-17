module WaveTank

using LinearAlgebra
using SparseArrays
using Printf
using Zygote
using FileIO
using ColorTypes

export Tank, Actuator, SineSum, WaveSim, Propagator
export build_propagator, evaluate_surface, visualize
export evaluate_modal_amplitudes, caustic_image, caustic_loss, visualize_caustic
export params_to_Q, pack_params, unpack_params, make_caustic_loss
export load_target_image

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
    # Spatial basis Φ[j, nx*ny] on evaluation grid
    Φ::Matrix{Float64}
    # Analytic spatial derivatives of eigenmodes
    dΦ_dx::Matrix{Float64}  # dφ_j/dx evaluated on grid
    dΦ_dy::Matrix{Float64}  # dφ_j/dy evaluated on grid
    # Evaluation grid
    xs::Vector{Float64}
    ys::Vector{Float64}
    nx::Int
    ny::Int
    # Time grid
    t_grid::Vector{Float64}
end

# ── Build propagator ─────────────────────────────────────────────────

function build_propagator(sim::WaveSim; nx=100, ny=50)
    (; tank, actuators, tspan, dt, n_modes) = sim
    (; Lx, Ly, depth, g, damping) = tank

    c = sqrt(g * depth)  # wave speed

    # Collect mode indices, skipping (0,0)
    mode_m = Int[]
    mode_n = Int[]
    for m in 0:n_modes-1, n in 0:n_modes-1
        (m == 0 && n == 0) && continue
        push!(mode_m, m)
        push!(mode_n, n)
    end
    n_total = length(mode_m)

    # Mode frequencies
    ω = [c * sqrt((mode_m[j]*π/Lx)^2 + (mode_n[j]*π/Ly)^2) for j in 1:n_total]
    ω_d = ω .* sqrt(1 - damping^2)

    # Normalization factors N_j = ∫∫ φ_j² dx dy
    # For cos(mπx/Lx)·cos(nπy/Ly):
    #   ∫₀^Lx cos²(mπx/Lx) dx = Lx/2 if m>0, Lx if m=0
    #   similarly for y
    function norm_j(m, n)
        Ix = m == 0 ? Lx : Lx / 2
        Iy = n == 0 ? Ly : Ly / 2
        return Ix * Iy
    end

    # Eigenmode evaluation
    φ(m, n, x, y) = cos(m * π * x / Lx) * cos(n * π * y / Ly)

    # Coupling matrix C[j, i]
    # For finite-width actuators (Gaussian blob with half-width σ_a),
    # the coupling picks up exp(-½ σ_a² k²) per spatial dimension,
    # which suppresses high-k modes that a finite actuator can't excite.
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

    # Spatial basis Φ[j, nx*ny] and derivative matrices on evaluation grid
    xs = range(0, Lx, length=nx)
    ys = range(0, Ly, length=ny)
    Φ = zeros(n_total, nx * ny)
    dΦ_dx = zeros(n_total, nx * ny)
    dΦ_dy = zeros(n_total, nx * ny)
    idx = 0
    for iy in 1:ny, ix in 1:nx
        idx += 1
        for j in 1:n_total
            m, n = mode_m[j], mode_n[j]
            Φ[j, idx] = φ(m, n, xs[ix], ys[iy])
            # dφ/dx = -mπ/Lx · sin(mπx/Lx) · cos(nπy/Ly)
            dΦ_dx[j, idx] = -m * π / Lx * sin(m * π * xs[ix] / Lx) * cos(n * π * ys[iy] / Ly)
            # dφ/dy = -nπ/Ly · cos(mπx/Lx) · sin(nπy/Ly)
            dΦ_dy[j, idx] = -n * π / Ly * cos(m * π * xs[ix] / Lx) * sin(n * π * ys[iy] / Ly)
        end
    end

    # Time grid
    t_grid = collect(tspan[1]:dt:tspan[2])

    return Propagator(sim, mode_m, mode_n, ω, ω_d, C, Φ, dΦ_dx, dΦ_dy,
                      collect(xs), collect(ys), nx, ny, t_grid)
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

# ── Evaluate surface ─────────────────────────────────────────────────

function evaluate_surface(prop::Propagator, T::Real)
    (; Φ, xs, ys, nx, ny) = prop

    a = evaluate_modal_amplitudes(prop, T)

    # Spatial reconstruction: η = Φᵀ · a → [nx*ny]
    η_flat = Φ' * a
    η = reshape(η_flat, nx, ny)

    return xs, ys, η
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

# ── Caustic rendering ───────────────────────────────────────────────

function caustic_image(prop::Propagator, T::Real;
                       n_water=1.33, sigma=0.0, cutoff_sigmas=4.0)
    a = evaluate_modal_amplitudes(prop, T)
    return caustic_image(prop, a; n_water=n_water, sigma=sigma, cutoff_sigmas=cutoff_sigmas)
end

"""
    caustic_image(prop, a; n_water=1.33, sigma=0.0, cutoff_sigmas=4.0)

Render caustic image from pre-computed modal amplitudes `a`.
This method is Zygote-differentiable w.r.t. `a`.
"""
function caustic_image(prop::Propagator, a::AbstractVector;
                       n_water=1.33, sigma=0.0, cutoff_sigmas=4.0)
    (; sim, Φ, dΦ_dx, dΦ_dy, xs, ys, nx, ny) = prop
    depth = sim.tank.depth

    # Surface height and analytic gradients → reshaped to [nx, ny]
    η     = reshape(Φ' * a,      nx, ny)
    dηdx  = reshape(dΦ_dx' * a,  nx, ny)
    dηdy  = reshape(dΦ_dy' * a,  nx, ny)

    # Default sigma: 1.5 × max grid spacing
    dx = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    σ = sigma > 0 ? sigma : 1.5 * max(dx, dy)

    σ2 = σ * σ
    inv_2σ2 = 1.0 / (2.0 * σ2)

    w = Zygote.ignore() do
        ceil(Int, cutoff_sigmas * σ / max(dx, dy))
    end

    ratio = 1.0 / n_water

    # Phase A — Landing positions (pure broadcast)
    X_src = repeat(xs, 1, ny)           # [nx, ny]
    Y_src = repeat(ys', nx, 1)          # [nx, ny]

    x_land = X_src .+ (depth .- η) .* dηdx .* ratio
    y_land = Y_src .+ (depth .- η) .* dηdy .* ratio

    # Phase B — Bilinear splatting via scatter_add (Zygote-compatible)
    n_pix = nx * ny
    fi = clamp.((x_land .- xs[1]) ./ dx .+ 1.0, 1.0, Float64(nx))
    fj = clamp.((y_land .- ys[1]) ./ dy .+ 1.0, 1.0, Float64(ny))

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

    w00 = (1.0 .- wx) .* (1.0 .- wy)
    w10 = wx           .* (1.0 .- wy)
    w01 = (1.0 .- wx) .* wy
    w11 = wx           .* wy

    all_vals    = vcat(vec(w00), vec(w10), vec(w01), vec(w11))
    all_indices = vcat(vec(lin00), vec(lin10), vec(lin01), vec(lin11))
    D = reshape(scatter_add(all_vals, all_indices, n_pix), nx, ny)

    # Phase C — Gaussian convolution without mutation
    D_pad = vcat(zeros(w, ny + 2w),
                 hcat(zeros(nx, w), D, zeros(nx, w)),
                 zeros(w, ny + 2w))

    I = sum(
        exp(-((di * dx)^2 + (dj * dy)^2) * inv_2σ2) .*
            D_pad[w+1+di:w+nx+di, w+1+dj:w+ny+dj]
        for di in -w:w, dj in -w:w
    )

    return xs, ys, I
end

# ── Gaussian blur (mutation-free) ───────────────────────────────────

function gaussian_blur(M::AbstractMatrix{<:Real}, dx, dy, σ; cutoff_sigmas=4.0)
    σ <= 0 && return Float64.(M)
    inv_2σ2 = 1.0 / (2.0 * σ * σ)
    w = Zygote.ignore() do
        ceil(Int, cutoff_sigmas * σ / max(dx, dy))
    end
    nx, ny = size(M)
    M_pad = vcat(zeros(w, ny + 2w),
                 hcat(zeros(nx, w), Float64.(M), zeros(nx, w)),
                 zeros(w, ny + 2w))
    return sum(
        exp(-((di * dx)^2 + (dj * dy)^2) * inv_2σ2) .*
            M_pad[w+1+di:w+nx+di, w+1+dj:w+ny+dj]
        for di in -w:w, dj in -w:w
    )
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
                           λ_energy=0.0, λ_smooth=0.0)
    n_act = length(prop.sim.actuators)
    n_freq = length(ω_freqs)
    t_grid = prop.t_grid
    dx = prop.xs[2] - prop.xs[1]
    dy = prop.ys[2] - prop.ys[1]

    # Pre-blur target once (constant w.r.t. params)
    T_b = gaussian_blur(Float64.(target), dx, dy, σ_blur)

    function loss(params::AbstractVector)
        A_mat, φ_mat = unpack_params(params, n_act, n_freq)

        # Forward pass
        Q = params_to_Q(A_mat, φ_mat, ω_freqs, t_grid)
        a = evaluate_modal_amplitudes(prop, T_time, Q)
        _, _, I = caustic_image(prop, a; n_water=n_water, sigma=sigma)

        # Term 1: cosine similarity (scale-invariant pattern match)
        I_b = gaussian_blur(I, dx, dy, σ_blur)
        dot_IT = sum(I_b .* T_b)
        norm_I = sqrt(sum(I_b .^ 2) + 1e-12)
        norm_T = sqrt(sum(T_b .^ 2) + 1e-12)
        L_match = 1 - dot_IT / (norm_I * norm_T)

        # Term 2: energy penalty Σ A²
        L_energy = sum(A_mat .^ 2)

        # Term 3: smoothness penalty ½ Σ (A·ω)²
        L_smooth = 0.5 * sum((A_mat .* ω_freqs') .^ 2)

        return L_match + λ_energy * L_energy + λ_smooth * L_smooth
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

end # module
