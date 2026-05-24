"""
Static optimal-transport surface solver for water caustics.

Three entry points:
  1. ``solve_target_surface``      — splat-based OT (the original Python port).
                                      Fast, but fails for high-contrast targets:
                                      rays at dark-target regions get pushed
                                      across the domain boundary, clip, pile
                                      up, and corrupt the Poisson update via a
                                      step-size collapse.
  2. ``solve_target_surface_mesh`` — mesh-based OT, the faithful port of the
                                      Julia causticsEngineering algorithm.
                                      Uses a deformable quadrilateral mesh and
                                      computes ray density from Jacobians
                                      (triangle areas), so it is exactly
                                      mass-conservative and stays stable on
                                      portrait-like targets.
  3. ``project_to_modes``          — projects η*(x,y) onto the wave
                                      cosine eigenspace.

Reference: Schwartzburg et al. "High-contrast computational caustic design"
(SIGGRAPH 2014). Julia implementation at ~/code/causticsEngineering.
"""

import numpy as np
import scipy.fft
import scipy.linalg

from .physics import Propagator


# ── Poisson solver ─────────────────────────────────────────────────────

def _solve_poisson_neumann(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Solve ∇²φ = f on a rectangular grid with Neumann (no-flux) BCs.

    Uses DCT-II to diagonalise the Neumann Laplacian exactly in O(N² log N).
    The Neumann problem is only defined up to an additive constant; this
    function returns the zero-mean solution.

    Parameters
    ----------
    f   : (nx, ny) right-hand side; must have zero mean (enforced internally).
    dx  : grid spacing in x (= xs[1] - xs[0]).
    dy  : grid spacing in y (= ys[1] - ys[0]).

    Returns
    -------
    phi : (nx, ny) solution with phi.mean() == 0.
    """
    f = f - f.mean()
    F = scipy.fft.dctn(f, type=2, norm='ortho')

    nx, ny = f.shape
    j = np.arange(nx, dtype=np.float64)
    k = np.arange(ny, dtype=np.float64)

    # Eigenvalues of the discrete Neumann Laplacian (DCT-II diagonalisation)
    lam = (- 4.0 * np.sin(np.pi * j / (2.0 * nx))[:, None] ** 2 / dx ** 2
           - 4.0 * np.sin(np.pi * k / (2.0 * ny))[None, :] ** 2 / dy ** 2)
    lam[0, 0] = 1.0       # avoid divide-by-zero; DC handled below

    Phi = F / lam
    Phi[0, 0] = 0.0        # enforce zero-mean solution

    phi = scipy.fft.idctn(Phi, type=2, norm='ortho')
    return phi - phi.mean()


# ── Bilinear splatting ─────────────────────────────────────────────────

def _bilinear_splat_density(
    Tx: np.ndarray,
    Ty: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """
    Bilinear-splat a transport map (Tx, Ty) onto the (xs, ys) grid to
    compute projected ray density.

    Each source pixel (i,j) contributes unit weight at position
    (Tx[i,j], Ty[i,j]) on the output grid via bilinear interpolation.
    Rays that land outside [xs[0], xs[-1]] × [ys[0], ys[-1]] are discarded.

    Returns
    -------
    density : (nx, ny) ndarray — summing to ≈ nx*ny when all rays land in
              bounds, so density.mean() ≈ 1 for a reasonable transport map.
    """
    nx, ny = len(xs), len(ys)
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])

    fi = (Tx - xs[0]) / dx    # fractional x index, shape (nx, ny)
    fj = (Ty - ys[0]) / dy

    in_bounds = ((fi >= 0) & (fi <= nx - 1) &
                 (fj >= 0) & (fj <= ny - 1))
    mask = in_bounds.astype(np.float64)

    fi = np.clip(fi, 0.0, nx - 1.0)
    fj = np.clip(fj, 0.0, ny - 1.0)
    ix0 = np.clip(np.floor(fi).astype(np.int64), 0, nx - 2)
    iy0 = np.clip(np.floor(fj).astype(np.int64), 0, ny - 2)
    wx = fi - ix0
    wy = fj - iy0

    density = np.zeros((nx, ny))
    np.add.at(density, (ix0,       iy0      ), mask * (1.0 - wx) * (1.0 - wy))
    np.add.at(density, (ix0 + 1,   iy0      ), mask * wx         * (1.0 - wy))
    np.add.at(density, (ix0,       iy0 + 1  ), mask * (1.0 - wx) * wy)
    np.add.at(density, (ix0 + 1,   iy0 + 1  ), mask * wx         * wy)

    return density


# ── Main OT solver ─────────────────────────────────────────────────────

def solve_target_surface(
    target: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    throw: float,
    n_water: float = 1.33,
    n_iter: int = 50,
) -> dict:
    """
    Find the optimal refractive surface η*(x,y) that produces a target caustic.

    Algorithm
    ---------
    Maintains a grid-based transport map (Tx, Ty) starting as the identity.
    Each iteration:
      1. Forward-project: bilinear-splat (Tx, Ty) → projected density ρ.
      2. Loss: D = ρ - target_normalised (zero-meaned for solvability).
      3. Poisson solve (Neumann BCs): ∇²φ = D.
      4. Adaptive gradient step: (Tx, Ty) -= step * ∇φ.
    After convergence, reconstruct η* via paraxial Snell's law:
      ∂η/∂x = (n_water/throw) · (Tx - x),  ∂η/∂y = (n_water/throw) · (Ty - y)
      ∇²η = ∂(∂η/∂x)/∂x + ∂(∂η/∂y)/∂y  →  solve another Poisson equation.

    Parameters
    ----------
    target  : (nx, ny) ndarray — target caustic intensity, non-negative.
              Does not need to be normalised; the algorithm handles this.
    xs, ys  : 1D coordinate arrays (use prop.xs and prop.ys).
    throw   : Optical projection distance in metres (use prop.tank.throw).
    n_water : Refractive index of water (default 1.33).
    n_iter  : Number of Poisson-iteration steps (default 50).
              Each step moves rays at most 0.5 pixels, so reaching a
              target whose nearest pixel is d away takes ≈ 2d/dx steps.

    Returns
    -------
    dict with keys:
      'eta'     : (nx, ny) ndarray — optimal surface height in metres (zero-mean).
      'Tx'      : (nx, ny) ndarray — converged transport map, x component.
      'Ty'      : (nx, ny) ndarray — converged transport map, y component.
      'caustic' : (nx, ny) ndarray — ray density produced by η* (mean ≈ 1),
                  for direct comparison with caustic_image() output.
    """
    nx, ny = len(xs), len(ys)
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])

    target = np.asarray(target, dtype=np.float64)
    tsum = target.sum()
    if tsum <= 0:
        raise ValueError("target must have positive total intensity")
    # Normalise so the total mass matches the number of source pixels
    target_norm = target * (nx * ny / tsum)

    # Transport map: starts as identity
    X_grid, Y_grid = np.meshgrid(xs, ys, indexing='ij')   # (nx, ny)
    Tx = X_grid.copy()
    Ty = Y_grid.copy()

    for _ in range(n_iter):
        rho = _bilinear_splat_density(Tx, Ty, xs, ys)
        D = rho - target_norm
        D -= D.mean()   # enforce zero-sum so the Neumann problem is solvable

        phi = _solve_poisson_neumann(D, dx, dy)

        grad_x = np.gradient(phi, dx, axis=0)
        grad_y = np.gradient(phi, dy, axis=1)

        # Two-criterion step size:
        #   (a) Collapse criterion: cell with D[i,j] > 0 collapses at t=1/D[i,j].
        #       Use half the minimum to keep all cells non-degenerate.
        #   (b) Displacement criterion: limit max absolute movement to 0.5 pixels
        #       per step.  Without this, the Poisson solution's gradient can be
        #       metres/step for peaked targets, flinging all rays outside the domain
        #       in a single iteration.
        d_pos = D[D > 0]
        step_collapse = 0.5 / d_pos.max() if d_pos.size > 0 else 1.0
        max_grad = max(float(np.abs(grad_x).max()), float(np.abs(grad_y).max()))
        step_disp = 0.5 * min(dx, dy) / max_grad if max_grad > 1e-12 else 1.0
        step = min(step_collapse, step_disp)

        Tx -= step * grad_x
        Ty -= step * grad_y

        # Clip to domain: rays must land within [xs[0], xs[-1]] × [ys[0], ys[-1]].
        # Boundary clipping is physically valid (out-of-domain rays are lost) and
        # prevents boundary instability where tiny gradients push edge rays outside.
        Tx = np.clip(Tx, xs[0], xs[-1])
        Ty = np.clip(Ty, ys[0], ys[-1])

    # Surface reconstruction via paraxial Snell's law
    # ∂η/∂x = (n_water / throw) · (Tx - x_grid)
    eta_x = (n_water / throw) * (Tx - X_grid)
    eta_y = (n_water / throw) * (Ty - Y_grid)

    divergence = (np.gradient(eta_x, dx, axis=0)
                  + np.gradient(eta_y, dy, axis=1))
    eta = _solve_poisson_neumann(divergence, dx, dy)

    # Forward caustic for verification (comparable to caustic_image output)
    rho_final = _bilinear_splat_density(Tx, Ty, xs, ys)
    mean_rho = rho_final.mean()
    caustic = rho_final / mean_rho if mean_rho > 0 else rho_final

    return {'eta': eta, 'Tx': Tx, 'Ty': Ty, 'caustic': caustic}


# ── Mesh-based OT solver (port of Julia causticsEngineering) ───────────

def _mesh_pixel_areas(Mx: np.ndarray, My: np.ndarray) -> np.ndarray:
    """
    Pixel areas from a deformable mesh of shape (W+1, H+1).

    Each pixel (i,j) — for i in 0..W-1, j in 0..H-1 — is the quadrilateral
    with corners at mesh nodes (i,j), (i+1,j), (i,j+1), (i+1,j+1). The
    quad is split into two triangles along the diagonal
    (lower-left, upper-right):

        upper-left ─── upper-right
            │  ╲           │
            │   ╲  tri 1   │     tri 1: (LL, UR, UL)
            │    ╲         │
            │ tri ╲        │     tri 2: (LL, LR, UR)
            │  2   ╲       │
        lower-left ─── lower-right

    Returns absolute pixel areas, shape (W, H).
    """
    UL_x, UL_y = Mx[:-1, :-1], My[:-1, :-1]
    UR_x, UR_y = Mx[1:,  :-1], My[1:,  :-1]
    LL_x, LL_y = Mx[:-1, 1:],  My[:-1, 1:]
    LR_x, LR_y = Mx[1:,  1:],  My[1:,  1:]

    # Signed area = 0.5 * cross-product of two edges
    a1 = 0.5 * ((UR_x - LL_x) * (UL_y - LL_y) - (UR_y - LL_y) * (UL_x - LL_x))
    a2 = 0.5 * ((LR_x - LL_x) * (UR_y - LL_y) - (LR_y - LL_y) * (UR_x - LL_x))
    return np.abs(a1) + np.abs(a2)


def _mesh_velocities_from_phi(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mesh-node velocities from a Poisson potential φ on the pixel grid.

    Follows the Julia ``marchMesh!`` / ``∇`` boundary convention:
      - φ has shape (W, H) (pixel-centered)
      - velocity has shape (W+1, H+1) (mesh-node-centered)
      - The rightmost column of velocity (i = W) has u = 0
      - The bottom row of velocity (j = H) has v = 0
      - For nodes on the bottom row (j = H) but interior (i < W), the
        x-velocity is computed using the row j = H-1 (forward diff in x).
        Symmetric for the rightmost column.

    Returns (u, v), each shape (W+1, H+1). Sign: motion direction is -∇φ
    (rays flow downhill on φ), so this returns -∇φ directly.
    """
    W, H = phi.shape
    u = np.zeros((W + 1, H + 1))
    v = np.zeros((W + 1, H + 1))

    # u = ∂φ/∂x via forward differences. Available only for x in 0..W-2
    # (φ[x+1] is out of bounds at x = W-1). Mesh nodes at x = W-1 and x = W
    # therefore stay 0 — the rightmost two columns of nodes don't move in x.
    # Same pattern for v on the bottom.
    u[:W-1, :H] = phi[1:, :] - phi[:-1, :]
    u[:W-1, H]  = phi[1:, H-1] - phi[:-1, H-1]   # bottom-row velocity uses row H-1

    v[:W, :H-1] = phi[:, 1:] - phi[:, :-1]
    v[W, :H-1]  = phi[W-1, 1:] - phi[W-1, :-1]   # right-column velocity uses col W-1

    return -u, -v


def _find_t_collapse(x1, y1, x2, y2, x3, y3, u1, v1, u2, v2, u3, v3):
    """
    Time at which a moving triangle becomes degenerate (signed area → 0).

    Signed area as function of t is at² + bt + c (vectorised). Derivation:
    edges from p1: e_a = (p2 + t·V2) - (p1 + t·V1), e_b similarly to p3.
    A(t) = (1/2) (e_a × e_b). Expand to get coefficients a, b, c.

    Returns (t1, t2), the two roots (np.inf where there is no positive
    collapse time).
    """
    # Edge vectors at t=0 and velocity differences
    ax = x2 - x1;   ay = y2 - y1
    bx = x3 - x1;   by = y3 - y1
    au = u2 - u1;   av = v2 - v1
    bu = u3 - u1;   bv = v3 - v1

    a = au * bv - av * bu                                    # t² coeff
    b = ax * bv + au * by - ay * bu - av * bx                # t¹ coeff
    c = ax * by - ay * bx                                    # t⁰ coeff (= 2·A₀)

    disc = b * b - 4.0 * a * c
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))

    abs_a = np.abs(a)
    abs_b = np.abs(b)

    # Quadratic branch (|a| not tiny)
    safe_a = np.where(abs_a > 1e-18, a, 1.0)
    quad_t1 = (-b - sqrt_disc) / (2.0 * safe_a)
    quad_t2 = (-b + sqrt_disc) / (2.0 * safe_a)

    # Linear branch (|a| tiny): t = -c / b
    safe_b = np.where(abs_b > 1e-18, b, 1.0)
    lin_t  = -c / safe_b

    use_linear  = abs_a <= 1e-18
    no_real_root = disc < 0

    t1 = np.where(use_linear, lin_t, np.where(no_real_root, np.inf, quad_t1))
    t2 = np.where(use_linear, lin_t, np.where(no_real_root, np.inf, quad_t2))

    # Where the linear branch has b ≈ 0 too, the triangle never collapses
    t1 = np.where(use_linear & (abs_b <= 1e-18), np.inf, t1)
    t2 = np.where(use_linear & (abs_b <= 1e-18), np.inf, t2)
    return t1, t2


def _mesh_step_size(Mx, My, u, v) -> float:
    """
    Largest safe time-step before any triangle in the mesh becomes
    degenerate, divided by 2 for safety. Matches Julia's δ = min_t / 2.
    """
    UL_x, UL_y = Mx[:-1, :-1], My[:-1, :-1]
    UR_x, UR_y = Mx[1:,  :-1], My[1:,  :-1]
    LL_x, LL_y = Mx[:-1, 1:],  My[:-1, 1:]
    LR_x, LR_y = Mx[1:,  1:],  My[1:,  1:]
    UL_u, UL_v = u[:-1, :-1],  v[:-1, :-1]
    UR_u, UR_v = u[1:,  :-1],  v[1:,  :-1]
    LL_u, LL_v = u[:-1, 1:],   v[:-1, 1:]
    LR_u, LR_v = u[1:,  1:],   v[1:,  1:]

    # Triangle 1: LL, UR, UL
    t1a, t1b = _find_t_collapse(LL_x, LL_y, UR_x, UR_y, UL_x, UL_y,
                                LL_u, LL_v, UR_u, UR_v, UL_u, UL_v)
    # Triangle 2: LL, LR, UR
    t2a, t2b = _find_t_collapse(LL_x, LL_y, LR_x, LR_y, UR_x, UR_y,
                                LL_u, LL_v, LR_u, LR_v, UR_u, UR_v)

    all_t = np.concatenate([t1a.ravel(), t1b.ravel(), t2a.ravel(), t2b.ravel()])
    pos = all_t[all_t > 0]
    if pos.size == 0:
        return 1.0
    return float(pos.min()) * 0.5


def solve_target_surface_mesh(
    target: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    throw: float,
    n_water: float = 1.33,
    n_iter: int = 4,
    verbose: bool = False,
) -> dict:
    """
    Mesh-based OT solver — Python port of the Julia causticsEngineering algorithm.

    Maintains a deformable (W+1, H+1) quadrilateral mesh over a (W, H) image.
    Each outer iteration:

      1. Compute pixel areas (Jacobian of the inverse transport) from the
         mesh triangulation.
      2. D = pixel_area − target_norm  (zero-mean for Neumann solvability).
      3. Solve Poisson ∇²φ = D via DCT-II (exact Neumann Laplacian inverter).
      4. Compute mesh-node velocities from forward-difference ∇φ, with
         u = 0 on the right boundary and v = 0 on the bottom boundary.
      5. Step size = (min triangle-collapse time) / 2 — guarantees that no
         triangle inverts in this iteration.
      6. Update mesh: nodes move by step · velocity.

    Density is exact (triangle areas) and the algorithm is mass-conservative,
    avoiding the splat-based version's pile-up-at-the-boundary failure mode
    on high-contrast targets.

    After ``n_iter`` outer steps, η is reconstructed from the converged
    transport map via paraxial Snell:

        ∂η/∂x = (n_water / throw) · (T_x − x_init)

    and a final Poisson solve on the divergence of the slope field.

    Parameters
    ----------
    target  : (W, H) ndarray — target caustic intensity, non-negative.
    xs, ys  : 1D coordinate arrays (length W, H respectively).
    throw   : optical projection distance (m).
    n_water : refractive index of water (default 1.33).
    n_iter  : number of outer mesh-march iterations (default 4 matches the
              Julia engineer_caustics). The Poisson solve inside each
              iteration converges exactly (DCT-II), so few outer iterations
              are needed.
    verbose : print per-iteration diagnostics.

    Returns
    -------
    dict with keys:
      'eta'     : (W, H) ndarray — optimal surface height in metres (zero-mean).
      'Tx'      : (W, H) ndarray — transport map x-component, in physical metres.
      'Ty'      : (W, H) ndarray — transport map y-component.
      'caustic' : (W, H) ndarray — ray density produced by the converged mesh
                  (mean ≈ 1), for direct comparison with the target.
    """
    W, H = target.shape
    if W != len(xs) or H != len(ys):
        raise ValueError(
            f"target shape {target.shape} must match (len(xs), len(ys)) "
            f"= ({len(xs)}, {len(ys)})"
        )

    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])

    # Mesh in PIXEL-INDEX coordinates: node (i,j) starts at (i, j).
    # Working in pixel-index space keeps the initial pixel-area uniformly 1.
    Mx_init = np.broadcast_to(np.arange(W + 1, dtype=np.float64)[:, None], (W + 1, H + 1)).copy()
    My_init = np.broadcast_to(np.arange(H + 1, dtype=np.float64)[None, :], (W + 1, H + 1)).copy()
    Mx, My = Mx_init.copy(), My_init.copy()

    # Boost target so its sum equals the initial total pixel area (= W·H).
    target = np.asarray(target, dtype=np.float64)
    tsum = float(target.sum())
    if tsum <= 0:
        raise ValueError("target must have positive total intensity")
    target_boost = target * (W * H / tsum)

    for outer in range(n_iter):
        area = _mesh_pixel_areas(Mx, My)            # (W, H)
        D = area - target_boost                     # (W, H)
        D -= D.mean()

        # ∇²φ = D on the (W, H) pixel grid with Neumann BCs
        phi = _solve_poisson_neumann(D, 1.0, 1.0)   # work in pixel-index units

        u, v = _mesh_velocities_from_phi(phi)       # (W+1, H+1)
        step = _mesh_step_size(Mx, My, u, v)

        Mx += step * u
        My += step * v

        if verbose:
            print(f"  iter {outer}: |D|max={np.abs(D).max():.4f}, "
                  f"step={step:.4f}, area∈[{area.min():.3f},{area.max():.3f}]")

    # ── Reconstruct η from converged transport map via paraxial Snell ──
    # In pixel coordinates: displacement of node (i, j) from identity.
    dMx_pix = Mx - Mx_init      # pixels in x
    dMy_pix = My - My_init      # pixels in y

    # Convert pixel displacements to physical metres
    dMx_m = dMx_pix * dx
    dMy_m = dMy_pix * dy

    # Slope fields at mesh nodes, Python convention: η_x = (n / throw) · ΔT
    eta_x_nodes = (n_water / throw) * dMx_m         # (W+1, H+1)
    eta_y_nodes = (n_water / throw) * dMy_m

    # Divergence at pixel centres: forward-difference between adjacent nodes,
    # average across the two diagonal pairs to keep it pixel-centred.
    div = (
        (eta_x_nodes[1:, :-1] - eta_x_nodes[:-1, :-1]) / dx +
        (eta_y_nodes[:-1, 1:] - eta_y_nodes[:-1, :-1]) / dy
    )                                               # (W, H)
    div -= div.mean()

    eta = _solve_poisson_neumann(div, dx, dy)       # (W, H), zero-mean

    # Transport map in physical metres on the image grid (use the (W, H)
    # interior of the mesh).
    Tx_phys = (Mx[:W, :H] - Mx_init[:W, :H]) * dx + xs[:, None]
    Ty_phys = (My[:W, :H] - My_init[:W, :H]) * dy + ys[None, :]

    final_area = _mesh_pixel_areas(Mx, My)
    caustic = final_area / final_area.mean()

    # Convert full mesh-node positions (W+1, H+1) to physical metres.
    Mx_full = Mx * dx + xs[0]
    My_full = My * dy + ys[0]

    return {
        'eta': eta,
        'Tx': Tx_phys, 'Ty': Ty_phys,
        'Mx_nodes': Mx_full, 'My_nodes': My_full,   # full (W+1, H+1) mesh, in metres
        'caustic': caustic,
    }


# ── Mode projection ────────────────────────────────────────────────────

def project_to_modes(
    eta: np.ndarray,
    prop: Propagator,
) -> np.ndarray:
    """
    Least-squares project a surface height map onto the cosine eigenspace.

    Solves  min ‖Σ_{m,n} a_{m,n} · cos_x[:,m] · cos_y[:,n] - η‖²
    in two separable lstsq steps, following the same matrix structure as
    reconstruct_surface (render.py:27-62).

    The result is the closest point in the wave eigenspace to η.  The
    residual ‖reconstruct_surface(prop, a) - η‖ / ‖η‖ measures how much
    quality the wave constraint costs relative to a free static surface.

    Parameters
    ----------
    eta  : (nx, ny) ndarray — surface height map (e.g. from solve_target_surface).
    prop : Propagator with precomputed cos_x, cos_y basis matrices.

    Returns
    -------
    a : (n_total,) ndarray compatible with reconstruct_surface(prop, a).
    """
    eta = np.asarray(eta, dtype=np.float64)
    cos_x = prop.cos_x   # (nx, n_modes)
    cos_y = prop.cos_y   # (ny, n_modes)

    # Step 1: A = pinv(cos_x) @ eta  →  shape (n_modes, ny)
    #   cos_x @ A ≈ eta
    A, _, _, _ = scipy.linalg.lstsq(cos_x, eta)

    # Step 2: a_2d.T = pinv(cos_y) @ A.T  →  a_2d shape (n_modes, n_modes)
    #   cos_y @ a_2d.T ≈ A.T
    a_2d_T, _, _, _ = scipy.linalg.lstsq(cos_y, A.T)
    a_2d = a_2d_T.T

    # Gather flat amplitudes: a[j] = a_2d[mode_m[j], mode_n[j]]
    # (mode (0,0) is implicitly excluded by prop.mode_m / prop.mode_n)
    return a_2d[prop.mode_m, prop.mode_n].copy()
