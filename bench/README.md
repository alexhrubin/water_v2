# Cross-language benchmarks

`bench_jax.py` and `bench_julia.jl` time the four hot paths used during caustic
optimization on an *identical*, fully deterministic problem so the numbers can
be compared 1:1.

## Run

```sh
# JAX (CPU)
uv run python bench/bench_jax.py

# Julia
julia --project=. bench/bench_julia.jl
```

Both scripts use the same closed-form phasor matrix (no RNG), so the
"Reference values" block at the bottom of each output **must agree to all
printed digits**. If it doesn't, one of the implementations has drifted.

## What is timed

| Stage | What it covers |
|-------|----------------|
| `steady_state` | `H ⊙ (C·P) @ exp(iΩT)` — pure linear algebra |
| `caustic_image forward` | surface reconstruction → refraction → splat → blur |
| `loss` | forward + cosine similarity to a target image |
| `value_and_grad` | loss + gradient w.r.t. the parameter vector |

The first three exercise the forward pipeline; the last is what each L-BFGS
step actually calls.

## Caveats

- JAX is timed with JIT-compiled functions and 3 warm-up calls excluded.
  Julia is timed after 3 warm-up calls so its first-call latency is excluded.
- Timing is `median` and `best` of 50 runs. On a busy machine, prefer `best`.
- Both run on CPU. Re-running JAX on a GPU should make
  `caustic_image forward` and `value_and_grad` substantially faster while
  leaving Julia unchanged.
