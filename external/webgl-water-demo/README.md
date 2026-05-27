# Evan Wallace's WebGL Water — with a depth slider

This is a local copy of Evan Wallace's WebGL water demo (https://madebyevan.com/webgl-water/,
MIT-licensed), patched with one addition: a runtime slider for water depth (pool height).

## Why this is here

Comparison material for the Water_v2 project. Wallace's demo uses a fundamentally
different approach to caustic *rendering* than we do (forward mesh rasterization with
GPU `dFdx`/`dFdy` derivatives for the Jacobian; see Wallace 2014 on Medium), and a
much simpler *wave model* than we use:

```glsl
average = mean(4-neighbor heights);
velocity += (average - height) * 2.0;
velocity *= 0.995;
height += velocity;
```

That's the discrete linear wave equation `h_tt = c²·∇²h` with constant `c²=2` in
pixel units. No dispersion, no gravity, no surface tension, no depth. All wavelengths
propagate at the same speed.

## What the depth slider does

The added slider controls `poolHeight` — the geometric distance from the water
surface to the floor in his renderer. Since his wave model has no depth term, the
slider does *not* change the wave physics; it only changes how far the refracted
rays travel before hitting the floor. This maps to "throw" in our framework.

Visible effect of depth (throw):

- **Deeper** (slider → 3.0): rays travel further, lateral ray displacement is larger,
  caustic features grow larger and more sparse. Matches the `throw/20` rule.
- **Shallower** (slider → 0.1): rays travel less, lateral displacement is small,
  caustic features become very fine and tightly tied to surface curvature.

## How to run

WebGL won't load textures from `file://` URLs (CORS), so you need a local server:

```bash
cd external/webgl-water-demo
python3 -m http.server 8000
# then open http://localhost:8000/ in a browser
```

## Loading our optimized solutions

```bash
# 1. Generate a binary animation from any of our optimized .npz files:
python notebooks/export_3d_animation.py \
    --npz notebooks/transient_test/3spot_gaussian_128_transient.npz \
    --output /tmp/anim.bin

# 2. Start the demo server:
cd external/webgl-water-demo
python3 -m http.server 8000
# open http://localhost:8000/

# 3. In the demo, use the "Animation file" picker to load /tmp/anim.bin.
#    The surface jumps to our optimized solution, caustics on the floor.
#    Drag to orbit; the depth and height-scale sliders adjust the view.
```

The exporter handles two cases:
- npz contains a `params` field (steady-state phasors): can produce a real
  time-varying animation by re-running steady_state at multiple times. Use
  `--n_frames 60` for smooth motion.
- npz contains only `eta` / `eta_steady` / `eta_trans` (single snapshot):
  ships one static frame, rendered as a still surface.

## Modifications from upstream

**Depth slider (small):**
- `renderer.js` line 14: changed `const float poolHeight = 1.0;` to `uniform float poolHeight;`
- `renderer.js`: added `poolHeight: this.poolHeight` to each of the four `.uniforms()` blocks
- `renderer.js`: added `this.poolHeight = 1.0` default in the `Renderer` constructor
- `index.html`: added a `#controls` div with the depth slider
- `main.js`: added an `input` listener to set `renderer.poolHeight` live

**Animation playback (larger):**
- `water.js`: added `loadAnimation(arrayBuffer)` to parse our binary format,
  and `playFrame(idx, scale)` to upload a frame's η to textureA.r and
  re-compute normals
- `main.js`: replaced the simulation loop with animation playback
  (`stepSimulation`/`moveSphere` calls removed; `update()` now advances
  through animation frames). Drop-mode and sphere-move-mode removed from
  `startDrag` (only camera orbit remains). The 20 initial random drops are
  also removed so the demo starts on still water until an animation is loaded.
- `main.js`: sphere is hidden via `radius = 0` and a guard around
  `renderer.renderSphere()` (the sphere code remains in `renderer.js` but
  is never drawn)
- `index.html`: added file picker for `.bin` animation, plus a height-scale
  slider

Everything else is upstream Evan Wallace code (camera orbit, caustic
rendering shader, water-surface refraction, environment map, pool floor
tiles, light controls).
