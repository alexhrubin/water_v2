module WaveTank

using LinearAlgebra
using SparseArrays
using Printf

export Tank, Actuator, WaveSim, Propagator
export build_propagator, evaluate_surface, visualize
export evaluate_modal_amplitudes, caustic_image, visualize_caustic

# ── Data structures ──────────────────────────────────────────────────

struct Tank
    Lx::Float64
    Ly::Float64
    depth::Float64
    g::Float64
    damping::Float64  # modal damping ratio γ
end
Tank(Lx, Ly, depth; g=9.81, damping=0.01) = Tank(Lx, Ly, depth, g, damping)

struct Actuator
    x::Float64
    y::Float64
    forcing::Function  # t → amplitude
    width::Float64     # Gaussian half-width σ (0 = point source)
end
Actuator(x, y, forcing; width=0.0) = Actuator(x, y, forcing, width)

struct WaveSim
    tank::Tank
    actuators::Vector{Actuator}
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
    a = sum(G .* F, dims=2)[:] .* dt                               # [n_modes]

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

# ── Caustic rendering ───────────────────────────────────────────────

function caustic_image(prop::Propagator, T::Real;
                       n_water=1.33, sigma=0.0, cutoff_sigmas=4.0)
    (; sim, Φ, dΦ_dx, dΦ_dy, xs, ys, nx, ny) = prop
    depth = sim.tank.depth

    a = evaluate_modal_amplitudes(prop, T)

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
    w = ceil(Int, cutoff_sigmas * σ / max(dx, dy))

    ratio = 1.0 / n_water

    # Phase A — Landing positions (pure broadcast)
    X_src = repeat(xs, 1, ny)           # [nx, ny]
    Y_src = repeat(ys', nx, 1)          # [nx, ny]

    x_land = X_src .+ (depth .- η) .* dηdx .* ratio
    y_land = Y_src .+ (depth .- η) .* dηdy .* ratio

    # Phase B — Bilinear splatting via sparse() (no mutation)
    n_pix = nx * ny
    fi = clamp.((x_land .- xs[1]) ./ dx .+ 1.0, 1.0, Float64(nx))
    fj = clamp.((y_land .- ys[1]) ./ dy .+ 1.0, 1.0, Float64(ny))

    ix0 = clamp.(floor.(Int, fi), 1, nx - 1)
    iy0 = clamp.(floor.(Int, fj), 1, ny - 1)
    wx  = fi .- ix0
    wy  = fj .- iy0

    # Linear indices for the 4 bilinear corners (column-major: row = ix, col = iy)
    lin00 = ix0      .+ (iy0 .- 1) .* nx
    lin10 = (ix0.+1) .+ (iy0 .- 1) .* nx
    lin01 = ix0      .+ iy0        .* nx
    lin11 = (ix0.+1) .+ iy0        .* nx

    w00 = (1.0 .- wx) .* (1.0 .- wy)
    w10 = wx           .* (1.0 .- wy)
    w01 = (1.0 .- wx) .* wy
    w11 = wx           .* wy

    row_idx = vcat(vec(lin00), vec(lin10), vec(lin01), vec(lin11))
    vals    = vcat(vec(w00),   vec(w10),   vec(w01),   vec(w11))
    col_idx = ones(Int, 4 * n_pix)

    D = reshape(Array(sparse(row_idx, col_idx, vals, n_pix, 1))[:], nx, ny)

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
                           fps=30, clims=nothing, n_water=1.33,
                           sigma=0.0, filename="caustic.gif")
    plt = try
        Main.Plots
    catch
        error("Plots.jl must be loaded before calling visualize_caustic(). Run `using Plots` first.")
    end

    (; sim, xs, ys) = prop
    t0, t1 = sim.tspan
    dt_frame = 1.0 / fps
    frames = t0:dt_frame:t1

    # Pre-sample to auto-determine clims if not provided
    if clims === nothing
        sample_times = range(t0 + 0.1, t1, length=min(10, length(frames)))
        all_vals = Float64[]
        for t in sample_times
            _, _, I = caustic_image(prop, t; n_water=n_water, sigma=sigma)
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
        _, _, I = caustic_image(prop, T; n_water=n_water, sigma=sigma)
        p = plt.heatmap(xs, ys, I',
                xlabel="x (m)", ylabel="y (m)",
                title=@sprintf("Caustic Pattern  t = %5.2f s", T),
                color=:inferno, clims=clims,
                aspect_ratio=:equal, size=(800, 400))
        plt.frame(anim, p)
    end

    return plt.gif(anim, filename, fps=fps)
end

end # module
