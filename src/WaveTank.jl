module WaveTank

using LinearAlgebra

export Tank, Actuator, WaveSim, Propagator
export build_propagator, evaluate_surface, visualize

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
end

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
    n_act = length(actuators)
    C = zeros(n_total, n_act)
    for i in 1:n_act
        ax, ay = actuators[i].x, actuators[i].y
        for j in 1:n_total
            C[j, i] = φ(mode_m[j], mode_n[j], ax, ay) / norm_j(mode_m[j], mode_n[j])
        end
    end

    # Spatial basis Φ[j, nx*ny] on evaluation grid
    xs = range(0, Lx, length=nx)
    ys = range(0, Ly, length=ny)
    Φ = zeros(n_total, nx * ny)
    idx = 0
    for iy in 1:ny, ix in 1:nx
        idx += 1
        for j in 1:n_total
            Φ[j, idx] = φ(mode_m[j], mode_n[j], xs[ix], ys[iy])
        end
    end

    # Time grid
    t_grid = collect(tspan[1]:dt:tspan[2])

    return Propagator(sim, mode_m, mode_n, ω, ω_d, C, Φ,
                      collect(xs), collect(ys), nx, ny, t_grid)
end

# ── Green's function kernel ──────────────────────────────────────────

function green_kernel(ω_j, ω_dj, γ, τ)
    τ <= 0 && return 0.0
    return exp(-γ * ω_j * τ) * sin(ω_dj * τ) / ω_dj
end

# ── Evaluate surface ─────────────────────────────────────────────────

function evaluate_surface(prop::Propagator, T::Real)
    (; sim, C, Φ, ω, ω_d, t_grid, xs, ys, nx, ny) = prop
    (; tank, actuators, dt) = sim
    γ = tank.damping

    n_act = length(actuators)
    n_steps = length(t_grid)
    n_modes = length(ω)

    # Sample actuator signals: Q[i, k]
    Q = zeros(n_act, n_steps)
    for i in 1:n_act
        for k in 1:n_steps
            Q[i, k] = actuators[i].forcing(t_grid[k])
        end
    end

    # Modal forcing: F = C · Q  → [n_modes × n_steps]
    F = C * Q

    # Temporal convolution per mode: a_j = Σ_k g_j(T - t_k) · F[j,k] · Δt
    a = zeros(n_modes)
    for j in 1:n_modes
        s = 0.0
        for k in 1:n_steps
            τ = T - t_grid[k]
            τ <= 0 && continue
            s += green_kernel(ω[j], ω_d[j], γ, τ) * F[j, k]
        end
        a[j] = s * dt
    end

    # Spatial reconstruction: η = Φᵀ · a → [nx*ny]
    η_flat = Φ' * a
    η = reshape(η_flat, nx, ny)

    return xs, ys, η
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
                title="Wave Tank  t = $(round(T, digits=2)) s",
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

end # module
