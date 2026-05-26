"""Generate target images matching what shallow tanks naturally produce.

Hypothesis under test: shallow tanks at d ≪ throw can produce sharp caustics,
but only of certain spatial classes — concentric arcs, fine lines, sparse
dots. The optimization fails at d=0.1m on broad-blob targets (3-spot
Gaussian, ANNA) because those targets are unreachable, not because the
apparatus can't produce sharp caustics.

These targets are designed to match the natural-output class:
  - ring_arcs.npy: concentric arcs from a wall point (single-speaker pulse)
  - dots_on_ring.npy: 8 small bright dots on a circle (sparse high-k pattern)
  - thin_arc.npy: single curved line (fine-feature line art)
  - sparse_grid.npy: 3×3 grid of small dots (sparse aperiodic dots)
  - fine_lines.npy: set of parallel thin vertical lines (high-k pattern)

Run:
    python notebooks/generate_pulse_targets.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def make_grid(nx, ny):
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X, Y


def gaussian_at(X, Y, cx, cy, sigma):
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))


def make_ring_arcs(nx=200, ny=200, n_arcs=5, center=(0.5, 0.0),
                    r0=0.08, dr=0.10, thickness=0.008):
    """Concentric arcs centered on (cx, cy). Each arc is a thin ring fragment.

    Matches the natural output of a single wall-mounted speaker driven
    impulsively — what the pulse experiment showed at d=0.1m.
    """
    X, Y = make_grid(nx, ny)
    R = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    img = np.zeros_like(X)
    for i in range(n_arcs):
        r = r0 + i * dr
        img += np.exp(-((R - r) ** 2) / (2 * thickness ** 2))
    return np.clip(img / max(img.max(), 1e-9), 0.0, 1.0)


def make_dots_on_ring(nx=200, ny=200, n_dots=8,
                       center=(0.5, 0.5), radius=0.3, sigma=0.012):
    """8 small bright dots arranged on a circle. Sparse high-k pattern."""
    X, Y = make_grid(nx, ny)
    img = np.zeros_like(X)
    for k in range(n_dots):
        theta = 2 * np.pi * k / n_dots
        cx = center[0] + radius * np.cos(theta)
        cy = center[1] + radius * np.sin(theta)
        img += gaussian_at(X, Y, cx, cy, sigma)
    return np.clip(img / max(img.max(), 1e-9), 0.0, 1.0)


def make_thin_arc(nx=200, ny=200,
                   center=(0.5, 0.5), radius=0.35,
                   theta_start=np.pi * 0.2, theta_end=np.pi * 0.8,
                   thickness=0.008):
    """Single curved arc — thin line caustic feature."""
    X, Y = make_grid(nx, ny)
    cx, cy = center
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    theta = np.arctan2(Y - cy, X - cx)
    # Wrap to [0, 2π]
    theta = np.mod(theta, 2 * np.pi)
    # Smooth mask for angle range
    in_range = (theta >= theta_start) & (theta <= theta_end)
    img = np.exp(-((R - radius) ** 2) / (2 * thickness ** 2)) * in_range
    return np.clip(img / max(img.max(), 1e-9), 0.0, 1.0)


def make_sparse_grid(nx=200, ny=200, n_per_side=3, sigma=0.012):
    """3×3 grid of small bright dots. Sparse pattern, no broad regions."""
    X, Y = make_grid(nx, ny)
    img = np.zeros_like(X)
    for i in range(n_per_side):
        for j in range(n_per_side):
            cx = 0.25 + 0.25 * i
            cy = 0.25 + 0.25 * j
            img += gaussian_at(X, Y, cx, cy, sigma)
    return np.clip(img / max(img.max(), 1e-9), 0.0, 1.0)


def make_fine_lines(nx=200, ny=200, n_lines=8, thickness=0.005):
    """Vertical parallel thin lines. High spatial frequency pattern."""
    X, Y = make_grid(nx, ny)
    img = np.zeros_like(X)
    for k in range(n_lines):
        x0 = (k + 0.5) / n_lines
        img += np.exp(-((X - x0) ** 2) / (2 * thickness ** 2))
    return np.clip(img / max(img.max(), 1e-9), 0.0, 1.0)


def main():
    out_dir = Path('targets')
    out_dir.mkdir(exist_ok=True)

    targets = {
        'ring_arcs':     make_ring_arcs(),
        'dots_on_ring':  make_dots_on_ring(),
        'thin_arc':      make_thin_arc(),
        'sparse_grid':   make_sparse_grid(),
        'fine_lines':    make_fine_lines(),
    }

    fig, axes = plt.subplots(1, len(targets), figsize=(3 * len(targets), 3.5))
    for ax, (name, img) in zip(axes, targets.items()):
        npy_path = out_dir / f"{name}.npy"
        np.save(npy_path, img.astype(np.float32))
        print(f"Saved {npy_path}  ({img.shape}, range [{img.min():.3f}, {img.max():.3f}])")
        ax.imshow(img.T, origin='lower', extent=[0, 1, 0, 1], cmap='inferno')
        ax.set_title(name, fontsize=10)
        ax.axis('off')

    fig.suptitle("Pulse-natural targets for shallow-tank diagnostic", fontsize=11)
    fig.tight_layout()
    preview = out_dir / 'pulse_targets_preview.png'
    fig.savefig(preview, dpi=110, bbox_inches='tight')
    print(f"\nPreview: {preview}")


if __name__ == "__main__":
    main()
