"""Predict caustic feasibility for a given target + apparatus + physics regime.

Walks the chain from target image → required spatial content → required curvature
→ apparatus's available curvature → predicted cos similarity. Pure physics-based
prediction, no optimization. The empirical sweep validates these predictions.

Run:
    python notebooks/predict_target.py targets/dog_square.jpg
    python notebooks/predict_target.py targets/ANNA.jpg --depth 5
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

G        = 9.81
N_WATER  = 1.33

# Slope-cap envelopes by physics regime (max |∇η| trusted in the model)
S_MAX = {
    'linear':  0.10,    # paraxial Snell + linear free-surface BCs
    'hos_m2':  0.30,    # HOS at order 2 — Stokes-like steepening allowed
    'hos_m3':  0.40,    # HOS at order 3 — close to wave breaking
    'fantasy': 14.0,    # what example.ipynb's optimizer actually used
}

# Empirical mapping from "brightness factor" √(H_excess/H_required) to cos.
# Calibration NOTE: the original BF_SCALE = 3.0 was fit when both this script
# and the renderer used the buggy paraxial threshold H_req ≈ n/throw. With the
# corrected paraxial Snell (H_req ≈ n/((n−1)·throw), ~3× higher), brightness
# factors shrink by ~√3, so this scale needs recalibration against re-run
# empirical sweeps. Use predictions as relative ordering, not absolute cos,
# until then.
BF_SCALE = 3.0


def load_target(path, nx=200, ny=200):
    img = Image.open(path).convert('L')
    img = img.resize((ny, nx), Image.BICUBIC)
    return np.array(img, dtype=np.float32) / 255.0


def required_k_max(target, Lx, Ly, n_modes_max=50, energy_fraction=0.95):
    """Smallest k that captures `energy_fraction` of target cosine energy."""
    nx, ny = target.shape
    xs = np.linspace(0, Lx, nx)
    ys = np.linspace(0, Ly, ny)
    ms = np.arange(n_modes_max + 1)
    cos_x = np.cos(np.outer(xs, ms * np.pi / Lx))
    cos_y = np.cos(np.outer(ys, ms * np.pi / Ly))
    dx, dy = Lx / nx, Ly / ny
    raw = cos_x.T @ target @ cos_y                              # [M+1, M+1]
    coeffs = np.zeros_like(raw)
    for mi in range(n_modes_max + 1):
        for ni in range(n_modes_max + 1):
            Ix = Lx if mi == 0 else Lx / 2
            Iy = Ly if ni == 0 else Ly / 2
            coeffs[mi, ni] = raw[mi, ni] * dx * dy / (Ix * Iy)
    coeffs[0, 0] = 0.0
    energy = coeffs ** 2
    E_total = energy.sum()
    if E_total < 1e-12:
        return 0.0

    # Sort all modes by wavenumber, accumulate energy
    k_grid = np.zeros_like(raw)
    for mi in range(n_modes_max + 1):
        for ni in range(n_modes_max + 1):
            k_grid[mi, ni] = math.pi * math.sqrt((mi / Lx) ** 2 + (ni / Ly) ** 2)
    order = np.argsort(k_grid.flatten())
    cum = np.cumsum(energy.flatten()[order])
    threshold = energy_fraction * E_total
    idx = int(np.searchsorted(cum, threshold))
    idx = min(idx, len(cum) - 1)
    return float(k_grid.flatten()[order][idx])


def predict(target_path, Lx=1.0, depth=2.0, n_modes=15, energy_fraction=0.95):
    target = load_target(target_path)
    k_need = required_k_max(target, Lx, Lx, n_modes_max=max(50, 3 * n_modes),
                            energy_fraction=energy_fraction)
    k_avail = math.pi * math.sqrt(2 * n_modes ** 2) / Lx    # at mode (n_modes, n_modes)
    throw   = depth                                          # no glass bottom
    H_req   = N_WATER / ((N_WATER - 1.0) * throw)            # paraxial focal threshold

    print(f"\nTarget: {target_path}")
    print(f"Apparatus: Lx={Lx}m, depth={depth}m, n_modes={n_modes}² → k_max={k_avail:.1f}")
    print(f"Target needs k ≥ {k_need:.1f} for {energy_fraction:.0%} of its energy")
    print(f"Caustic-formation threshold: H_required = n/((n−1)·throw) = {H_req:.3f}/m\n")

    rows = []
    for regime in ['linear', 'hos_m2', 'fantasy']:
        s_max = S_MAX[regime]
        H_avail = k_avail * s_max
        H_excess = H_avail - H_req
        feasible_basis = k_avail >= k_need
        forms = H_excess > 0
        if forms:
            bf = math.sqrt(H_excess / H_req)
            cos_pred = 1.0 - math.exp(-bf / BF_SCALE)
        else:
            bf = 0.0
            cos_pred = 0.3       # warm-start floor — analytical solve can do something
        rows.append((regime, s_max, H_avail, H_excess, forms, feasible_basis, bf, cos_pred))

    print(f"  {'regime':<10}  {'s_max':>6}  {'H_avail':>8}  {'H_excess':>9}  "
          f"{'forms?':>7}  {'basis?':>7}  {'bf':>6}  {'pred cos':>9}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*9}")
    for regime, s, ha, hx, forms, basis, bf, cos in rows:
        print(f"  {regime:<10}  {s:>6.2f}  {ha:>8.3f}  {hx:>9.3f}  "
              f"{('YES' if forms else 'no'):>7}  "
              f"{('YES' if basis else 'no'):>7}  "
              f"{bf:>6.2f}  {cos:>9.3f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('target', help="Path to target image (jpg/png)")
    parser.add_argument('--Lx',       type=float, default=1.0)
    parser.add_argument('--depth',    type=float, default=2.0)
    parser.add_argument('--n_modes',  type=int,   default=15)
    parser.add_argument('--energy',   type=float, default=0.95,
                        help="Target energy fraction for k_max_needed (default 0.95)")
    args = parser.parse_args()
    predict(args.target, Lx=args.Lx, depth=args.depth,
            n_modes=args.n_modes, energy_fraction=args.energy)


if __name__ == "__main__":
    main()
