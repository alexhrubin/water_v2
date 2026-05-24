"""
Loss functions for caustic optimization and image utilities.
"""

import numpy as np
import jax.numpy as jnp
from PIL import Image

from .render import _gaussian_blur_separable


# ── Loss functions ─────────────────────────────────────────────────────

def cosine_loss(I: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """
    1 - cosine_similarity(I, target).

    Invariant to absolute brightness scale: only the spatial pattern
    matters. Minimum is 0 (perfect match), maximum is 2 (anti-correlated).

    Caveat: cosine is *not* offset-invariant. If the physically-achievable
    rendering has elevated background (e.g. bright spots barely above a
    bright background) while the target has zero background, cosine loss
    will penalize the offset heavily even when the spatial structure is
    correct. Use ``pearson_loss`` in that regime — see the reach-disk
    discussion in docs/reachability_and_capacity.md.
    """
    dot = jnp.sum(I * target)
    norm_I = jnp.sqrt(jnp.sum(I**2) + 1e-12)
    norm_T = jnp.sqrt(jnp.sum(target**2) + 1e-12)
    return 1.0 - dot / (norm_I * norm_T)


def pearson_loss(I: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """
    1 - pearson_correlation(I, target) = mean-subtracted cosine loss.

    Invariant to *both* brightness scale and brightness offset: only the
    relative spatial pattern matters. Equivalent to cosine_loss after
    subtracting the mean from both inputs.

    Use this when the physical contrast ceiling is below what the target
    image asks for — e.g. shallow throw + small bright spots. Cosine
    will penalize the unreachable contrast gap; Pearson won't, freeing
    the optimizer's slope budget for shape-matching in reach-disk regions.
    """
    I_c = I - jnp.mean(I)
    T_c = target - jnp.mean(target)
    dot = jnp.sum(I_c * T_c)
    norm_I = jnp.sqrt(jnp.sum(I_c**2) + 1e-12)
    norm_T = jnp.sqrt(jnp.sum(T_c**2) + 1e-12)
    return 1.0 - dot / (norm_I * norm_T)


def ssim_loss(
    I: jnp.ndarray,
    target: jnp.ndarray,
    dx: float,
    dy: float,
    sigma_w: float | None = None,
) -> jnp.ndarray:
    """
    1 - mean SSIM (Structural Similarity Index).

    Measures local luminance, contrast, and structure similarity in
    Gaussian-windowed patches. More sensitive to spatial structure than
    cosine similarity.

    Parameters
    ----------
    I, target : images [nx, ny]
    dx, dy    : grid spacing (m)
    sigma_w   : Gaussian window std (default = 1.5 * max(dx, dy))
    """
    sigma_w = sigma_w if sigma_w is not None else 1.5 * max(dx, dy)

    n = I.size
    # Scale I to match target's mean brightness
    mean_T = jnp.sum(target) / n
    mean_I = jnp.sum(I) / n + 1e-12
    I_n = I * (mean_T / mean_I)

    # SSIM stability constants (Wang et al. 2004).  Keep as jnp scalars
    # so the function is jit-safe (float() on a traced array would error).
    L = jnp.max(target)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    import math
    w = int(math.ceil(4.0 * sigma_w / max(dx, dy)))

    # Local statistics via normalized Gaussian windows
    W    = _gaussian_blur_separable(jnp.ones_like(I_n), dx, dy, sigma_w, w)
    mu_x = _gaussian_blur_separable(I_n, dx, dy, sigma_w, w) / W
    mu_y = _gaussian_blur_separable(target, dx, dy, sigma_w, w) / W

    sig_x2 = jnp.clip(
        _gaussian_blur_separable(I_n**2, dx, dy, sigma_w, w) / W - mu_x**2, 0.0)
    # sig_y2 is constant w.r.t. I — keep as array [nx, ny], not a scalar
    sig_y2 = jnp.clip(
        _gaussian_blur_separable(target**2, dx, dy, sigma_w, w) / W - mu_y**2, 0.0)
    sig_xy = (_gaussian_blur_separable(I_n * target, dx, dy, sigma_w, w) / W
               - mu_x * mu_y)

    ssim_map = ((2.0 * mu_x * mu_y + C1) * (2.0 * sig_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sig_x2 + sig_y2 + C2))
    return 1.0 - jnp.mean(ssim_map)


# ── Image utilities ────────────────────────────────────────────────────

def load_target_image(
    path: str,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    invert: bool = False,
) -> np.ndarray:
    """
    Load an image file and resample it onto the propagator grid.

    The image is converted to grayscale, resized to (nx, ny) using
    bicubic resampling, and normalized to [0, 1].

    Parameters
    ----------
    path   : path to image file (JPEG, PNG, etc.)
    xs, ys : grid coordinate arrays from the Propagator
    invert : if True, invert intensity (dark → bright, for dark-on-white images)

    Returns
    -------
    target : float32 array [nx, ny] in [0, 1]
    """
    nx, ny = len(xs), len(ys)
    img = Image.open(path).convert('L')                     # grayscale
    img = img.resize((ny, nx), Image.BICUBIC)               # PIL: (width, height)
    arr = np.array(img, dtype=np.float32) / 255.0           # [nx, ny], [0, 1]
    if invert:
        arr = 1.0 - arr
    return arr
