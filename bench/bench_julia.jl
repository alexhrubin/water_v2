# Julia benchmark — companion to bench_jax.py.
#
# Times the four hot paths used during caustic optimization on a fixed,
# deterministic problem so the numbers can be compared 1:1 against JAX.
#
# Run with:  julia --project=. bench/bench_julia.jl

using LinearAlgebra
using Random
using Statistics
using Printf
using Zygote

include(joinpath(@__DIR__, "..", "src", "WaveTank.jl"))
using .WaveTank

# ── Fixed configuration (must match bench_jax.py exactly) ────────────

const LX, LY, DEPTH, DAMPING = 1.0, 1.0, 0.12, 0.02
const N_MODES = 12
const NX, NY = 64, 64
const N_ACT_PER_SIDE = 4
const FREQS_HZ = (1.0, 2.0, 3.0, 4.0)
const T_EVAL = 1.0
const SIGMA_RENDER = 0.02
const SIGMA_BLUR = 0.02
const SEED = 0
const N_REPEAT = 50
const N_WARMUP = 3

# ── Setup ────────────────────────────────────────────────────────────

function build_setup()
    tank = Tank(LX, LY, DEPTH; damping=DAMPING)
    Ω_freqs = collect(2π .* FREQS_HZ)
    n_freq = length(FREQS_HZ)

    # Trivial dummy forcing — we drive via phasors directly, not via Actuator.forcing.
    dummy = SineSum(zeros(n_freq), Ω_freqs, zeros(n_freq))

    actuators = WaveTank.Actuator[]
    for i in 1:N_ACT_PER_SIDE
        t = i / (N_ACT_PER_SIDE + 1)
        push!(actuators, WaveTank.Actuator(0.0,    t * LY, dummy; width=0.05))
        push!(actuators, WaveTank.Actuator(LX,     t * LY, dummy; width=0.05))
        push!(actuators, WaveTank.Actuator(t * LX, 0.0,    dummy; width=0.05))
        push!(actuators, WaveTank.Actuator(t * LX, LY,     dummy; width=0.05))
    end

    sim = WaveSim(tank, actuators, (0.0, T_EVAL), 0.01; n_modes=N_MODES)
    prop = build_propagator(sim; nx=NX, ny=NY, dense_basis=false)

    # Deterministic phasors via a closed form so the values match the JAX bench
    # exactly (numpy and Julia RNGs produce different streams from the same seed).
    n_act = length(actuators)
    X = [0.3 * sin(0.7 * (i - 1) + 1.3 * (k - 1)) for i in 1:n_act, k in 1:n_freq]
    Y = [0.3 * cos(0.4 * (i - 1) + 0.9 * (k - 1) + 0.2) for i in 1:n_act, k in 1:n_freq]
    params = vcat(vec(X), vec(Y))

    # Same Gaussian-ring target the JAX bench builds.
    xs = collect(range(0.0, LX, length=NX))
    ys = collect(range(0.0, LY, length=NY))
    target = [exp(-((sqrt((x - 0.5)^2 + (y - 0.5)^2) - 0.28)^2) / (2 * 0.08^2))
              for x in xs, y in ys]
    target ./= maximum(target)

    return prop, Ω_freqs, params, target
end

# ── Timing helper ────────────────────────────────────────────────────

function bench(label::String, fn; n_warmup=N_WARMUP, n_repeat=N_REPEAT)
    for _ in 1:n_warmup
        fn()
    end
    times = Float64[]
    for _ in 1:n_repeat
        push!(times, @elapsed fn())
    end
    sort!(times)
    median_ms = times[div(length(times), 2) + 1] * 1e3
    best_ms   = times[1] * 1e3
    @printf "  %-32s  median = %8.3f ms   best = %8.3f ms\n" label median_ms best_ms
    return median_ms
end

# ── Main ─────────────────────────────────────────────────────────────

function main()
    println("=" ^ 64)
    println("  Julia benchmark — wavetank caustic pipeline")
    println("=" ^ 64)
    println("  Julia $(VERSION)")

    prop, Ω_freqs, params, target = build_setup()
    n_freq = length(Ω_freqs)
    n_act = length(prop.sim.actuators)

    println("  Modes=$(length(prop.ω))  Grid=$(prop.nx)x$(prop.ny)  " *
            "Actuators=$n_act  Frequencies=$n_freq")
    println("  Parameter dim = $(length(params))")
    println("  Repeats=$N_REPEAT, warmup=$N_WARMUP\n")

    # ── 1) steady_state_amplitudes ───────────────────────────────────
    function ss_call()
        X, Y = unpack_complex(params, n_act, n_freq)
        P = X .+ im .* Y
        return steady_state_amplitudes(prop, P, Ω_freqs, T_EVAL)
    end
    a_ref = ss_call()
    bench("steady_state", ss_call)

    # ── 2) caustic_image forward ─────────────────────────────────────
    function render_call()
        X, Y = unpack_complex(params, n_act, n_freq)
        P = X .+ im .* Y
        a = steady_state_amplitudes(prop, P, Ω_freqs, T_EVAL)
        _, _, I = caustic_image(prop, a; sigma=SIGMA_RENDER)
        return I
    end
    I_ref = render_call()
    bench("caustic_image forward", render_call)

    # ── 3) Loss only ─────────────────────────────────────────────────
    loss_fn = make_caustic_loss_ss(prop, T_EVAL, target, Ω_freqs;
                                    sigma=SIGMA_RENDER, σ_blur=SIGMA_BLUR,
                                    loss_type=:cosine, λ_energy=0.0)
    L_ref = loss_fn(params)
    bench("loss", () -> loss_fn(params))

    # ── 4) Loss + gradient (Zygote) ──────────────────────────────────
    function vg_call()
        L, back = Zygote.pullback(loss_fn, params)
        g, = back(1.0)
        return L, g
    end
    L_vg, g_ref = vg_call()
    bench("value_and_grad (Zygote)", vg_call)

    println()
    println("  Reference values (for cross-language sanity)")
    @printf "    a[1]            = %+.10e\n"  a_ref[1]
    @printf "    sum(I)          = %+.10e\n"  sum(I_ref)
    @printf "    loss            = %+.10e\n"  L_ref
    @printf "    ‖∇loss‖         = %+.10e\n"  norm(g_ref)
    println("=" ^ 64)
end

main()
