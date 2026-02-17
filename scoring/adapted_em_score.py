"""
JAX-based EM density scoring adapted from Arthur Zalevsky's accc_jax.py.

This module combines:
  - Model-to-map density projection and CCC computation from accc_jax.py
  - Bayesian probabilistic scoring model from em_score.py

It is a **drop-in replacement** for ``scoring.em_score``.  Change one import
line and the rest of your pipeline (SMC, diagnostics, IO) works unchanged.

Mass convention
===============
Each coarse-grained particle carries weight = radius**3 (proportional to
sphere volume).  This matches the IMP model builder::

    mass_value = radius_value ** 3          # in create_imp_model_and_save_to_rmf
    IMP.atom.Mass.setup_particle(p, mass_value)

The same convention is used inside ``calc_projection_jax`` and
``calculate_ccc_jax`` here, so the model map generated during scoring is
identical to the target map generated for the same coordinates.

Density projection
==================
Adapted from ``accc_jax.py`` (Arthur Zalevsky, MIT licence, v0.0.2):

  1. Bin particle centres into a 3D histogram weighted by mass = r**3.
  2. Reshape (x, y, z) → ``swapaxes(0, 2)`` → MRC (z, y, x) convention.
  3. Separable Gaussian blur at the nominal map resolution via FFT
     convolution.

The Gaussian sigma is derived purely from the stated map resolution::

    sigma = resolution / (4 * sqrt(2 * ln(2))) / voxel_size

No additional sigma for finite particle size is added, matching
the reference ``accc_jax.py`` implementation.

Probabilistic model
===================

**Likelihood (Gaussian on CCC mismatch):**

    log L(x | data) = -(1 - CCC(x))^2 / (2 * sigma_ccc^2)

**Spatial prior (exponential attraction to density centre):**

    log pi_attract(x) = -lambda_attract * sum_i ||r_i - r_COM||

**Soft box prior (for HMC / bounded sampling):**

    log pi_box(x) = -sum_j max(|x_j| - box_size, 0)^2 / (2 * steepness^2)

**Tempered posterior for SMC:**

    pi_t(x)  ∝  prior(x) * L(x)^{lambda_t}

Implementation notes
====================
- All functions are JAX-differentiable (no ``np``, no ``-inf``,
  no Python control flow on traced values).
- Safe norm ``sqrt(||·||^2 + eps)`` used for HMC gradient stability.
- GPU-compatible: all heavy operations use ``jnp`` / ``jax.scipy``.
"""

import math
import numpy as np
from typing import Tuple, NamedTuple, Callable

import jax
import jax.numpy as jnp
from jax.scipy import signal

jit = jax.jit

# Safe epsilon for norm gradients (avoid 0/0 in safe norm)
_NORM_EPS = 1e-8


# =====================================================================
# Core density projection (adapted from accc_jax.py)
# =====================================================================

_SIGMA_FACTOR = 4.0 * math.sqrt(2.0 * math.log(2.0))


def resolution_to_sigma(resolution, pixel_size):
    """
    Convert map resolution (≈ FWHM) to Gaussian sigma in pixel units.

        sigma = resolution / (4 * sqrt(2 * ln 2)) / pixel_size

    This matches IMP.em2d and accc_jax.py conventions.
    Works with both Python floats and JAX traced values.
    """
    return resolution / _SIGMA_FACTOR / pixel_size


@jit
def pairwise_correlation_jax(A: jnp.ndarray, B: jnp.ndarray) -> float:
    """
    Pearson correlation coefficient between two flattened arrays.

    Uses float32 internally and returns 0.0 when the denominator is
    zero (matching ``accc_jax.py``'s safe-division convention).
    """
    A = A.astype(jnp.float32)
    B = B.astype(jnp.float32)
    A_centered = A - jnp.mean(A)
    B_centered = B - jnp.mean(B)
    numerator = jnp.sum(A_centered * B_centered)
    denominator = jnp.sqrt(
        jnp.sum(A_centered ** 2) * jnp.sum(B_centered ** 2)
    )
    return jnp.where(denominator > 0, numerator / denominator, 0.0)


@jit
def calc_projection_jax(
    coords: jnp.ndarray,
    weights: jnp.ndarray,
    bins: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    resolution: float,
) -> jnp.ndarray:
    """
    Compute mass-weighted 3D density from particle positions.

    Adapted from ``accc_jax.py  _calc_projection_jax_impl`` (float32 path).

    Steps
    -----
    1. Scatter-add particle weights into a uniform 3-D histogram.
    2. Swap axes  (x, y, z) → (z, y, x)  to match MRC convention.
    3. Separable Gaussian blur at map resolution via FFT convolution.

    Parameters
    ----------
    coords : (N, 3) particle positions in Ångström
    weights : (N,) per-particle weights  (= radii**3 for CG spheres)
    bins : tuple of 3 arrays, each (n+1,) bin edges for x, y, z
    resolution : map resolution in Ångström

    Returns
    -------
    density : (nz, ny, nx) float32 array in MRC convention
    """
    nx = bins[0].shape[0] - 1
    ny = bins[1].shape[0] - 1
    nz = bins[2].shape[0] - 1

    # --- step 1: 3-D weighted histogram via scatter-add -----------------
    voxel_size_x = bins[0][1] - bins[0][0]
    voxel_size_y = bins[1][1] - bins[1][0]
    voxel_size_z = bins[2][1] - bins[2][0]
    origin_x, origin_y, origin_z = bins[0][0], bins[1][0], bins[2][0]

    x_idx = jnp.clip(
        ((coords[:, 0] - origin_x) / voxel_size_x).astype(jnp.int32),
        0, nx - 1,
    )
    y_idx = jnp.clip(
        ((coords[:, 1] - origin_y) / voxel_size_y).astype(jnp.int32),
        0, ny - 1,
    )
    z_idx = jnp.clip(
        ((coords[:, 2] - origin_z) / voxel_size_z).astype(jnp.int32),
        0, nz - 1,
    )

    linear_idx = x_idx * (ny * nz) + y_idx * nz + z_idx

    # Sort indices for better scatter performance (matches accc_jax.py)
    order = jnp.argsort(linear_idx)
    linear_idx = linear_idx[order]
    sorted_weights = weights[order]

    histogram_flat = (
        jnp.zeros(nx * ny * nz, dtype=jnp.float32)
        .at[linear_idx]
        .add(sorted_weights)
    )
    img = histogram_flat.reshape((nx, ny, nz))

    # --- step 2: (x, y, z) → (z, y, x) for MRC convention ---------------
    img = jnp.swapaxes(img, 0, 2)

    # --- step 3: separable Gaussian blur at map resolution ----------------
    sigma = resolution_to_sigma(resolution, voxel_size_x)

    max_radius = 30
    x = jnp.arange(-max_radius, max_radius + 1, dtype=jnp.float32)
    kernel_1d = jnp.exp(-0.5 * (x / sigma) ** 2)
    # Truncate at 4*sigma (matches accc_jax.py)
    kernel_1d = kernel_1d * (jnp.abs(x) <= 4.0 * sigma)
    kernel_1d = kernel_1d / jnp.sum(kernel_1d)

    # Separable FFT convolution along each axis
    img = signal.fftconvolve(img, kernel_1d[jnp.newaxis, jnp.newaxis, :], mode="same")
    img = signal.fftconvolve(img, kernel_1d[jnp.newaxis, :, jnp.newaxis], mode="same")
    img = signal.fftconvolve(img, kernel_1d[:, jnp.newaxis, jnp.newaxis], mode="same")

    return img.astype(jnp.float32)


# =====================================================================
# Grid / bin utilities (adapted from accc_jax.py)
# =====================================================================

def bins_from_density(density) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct bin edges from an mrcfile density header.

    Handles both ``header.origin`` and ``nxstart`` conventions.
    Adapted from ``accc_jax.py``.
    """
    ox = float(density.header.origin.x)
    oy = float(density.header.origin.y)
    oz = float(density.header.origin.z)

    # Fall back to nxstart if origin is zero
    if ox == 0 and density.header.nxstart != 0:
        ox = float(density.header.nxstart) * float(density.voxel_size.x)
        oy = float(density.header.nystart) * float(density.voxel_size.y)
        oz = float(density.header.nzstart) * float(density.voxel_size.z)

    nx = int(density.header.nx)
    ny = int(density.header.ny)
    nz = int(density.header.nz)
    vx = float(density.voxel_size.x)
    vy = float(density.voxel_size.y)
    vz = float(density.voxel_size.z)

    binsx = np.linspace(ox, ox + nx * vx, nx + 1)
    binsy = np.linspace(oy, oy + ny * vy, ny + 1)
    binsz = np.linspace(oz, oz + nz * vz, nz + 1)
    return (binsx, binsy, binsz)


def _compute_bins_centered(nx, ny, nz, voxel_size):
    """Bin edges centred at the origin (for synthetic maps)."""
    return tuple(
        jnp.linspace(-d * voxel_size / 2, d * voxel_size / 2, d + 1)
        for d in (nx, ny, nz)
    )


# =====================================================================
# Configuration container
# =====================================================================

class EMConfig(NamedTuple):
    """All static data needed for EM density scoring."""

    target_data: jnp.ndarray   # (nz, ny, nx) experimental density
    bins_x: jnp.ndarray        # (nx+1,) bin edges
    bins_y: jnp.ndarray        # (ny+1,)
    bins_z: jnp.ndarray        # (nz+1,)
    resolution: float          # in Ångström
    voxel_size: float          # in Ångström
    density_com: jnp.ndarray   # (3,) centre of mass of target density


@jit
def _calculate_density_com(target_data, bins_x, bins_y, bins_z):
    """Centre of mass of target density (positive voxels only)."""
    cx = (bins_x[:-1] + bins_x[1:]) / 2
    cy = (bins_y[:-1] + bins_y[1:]) / 2
    cz = (bins_z[:-1] + bins_z[1:]) / 2
    Z, Y, X = jnp.meshgrid(cz, cy, cx, indexing="ij")
    pos = jnp.maximum(target_data, 0)
    total = jnp.sum(pos) + _NORM_EPS
    return jnp.array([
        jnp.sum(X * pos),
        jnp.sum(Y * pos),
        jnp.sum(Z * pos),
    ]) / total


# =====================================================================
# Config factories
# =====================================================================

def create_em_config_from_mrcfile(
    density, resolution: float, center_at_origin: bool = False,
) -> EMConfig:
    """
    Create EMConfig from an mrcfile density object.

    Parameters
    ----------
    density : mrcfile object (opened with ``mrcfile.open``)
    resolution : map resolution in Ångström
    center_at_origin : If True, override header and centre bins at the
        origin.  If False (default), reconstruct bins from the MRC
        header via ``bins_from_density`` (matches ``accc_jax.py``).
    """
    nx = int(density.header.nx)
    ny = int(density.header.ny)
    nz = int(density.header.nz)
    vx = float(density.voxel_size.x)
    target_data = jnp.array(density.data, dtype=jnp.float32)

    if center_at_origin:
        bins_x, bins_y, bins_z = _compute_bins_centered(nx, ny, nz, vx)
    else:
        bx, by, bz = bins_from_density(density)
        bins_x = jnp.array(bx, dtype=jnp.float32)
        bins_y = jnp.array(by, dtype=jnp.float32)
        bins_z = jnp.array(bz, dtype=jnp.float32)

    density_com = _calculate_density_com(target_data, bins_x, bins_y, bins_z)
    return EMConfig(target_data, bins_x, bins_y, bins_z, resolution, vx, density_com)


def create_em_config_from_arrays(
    target_data: np.ndarray,
    voxel_size: float,
    resolution: float,
    center_at_origin: bool = True,
) -> EMConfig:
    """Create EMConfig from raw numpy arrays."""
    nz, ny, nx = target_data.shape
    target_jax = jnp.array(target_data, dtype=jnp.float32)

    if center_at_origin:
        bins_x, bins_y, bins_z = _compute_bins_centered(nx, ny, nz, voxel_size)
    else:
        bins_x = jnp.linspace(0, nx * voxel_size, nx + 1)
        bins_y = jnp.linspace(0, ny * voxel_size, ny + 1)
        bins_z = jnp.linspace(0, nz * voxel_size, nz + 1)

    density_com = _calculate_density_com(target_jax, bins_x, bins_y, bins_z)
    return EMConfig(
        target_jax, bins_x, bins_y, bins_z,
        resolution, voxel_size, density_com,
    )


# =====================================================================
# Density generation (same code path as CCC → self-CCC = 1.0)
# =====================================================================

def generate_density_map(
    coords: np.ndarray,
    radii: np.ndarray,
    resolution: float,
    voxel_size: float,
    box_size: float,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate a 3D density map from CG particle positions.

    Uses the *same* ``calc_projection_jax`` pipeline as CCC scoring,
    guaranteeing perfect self-CCC = 1.0 (up to float32 precision).

    Mass convention:  weights = radii**3   (sphere-volume proportional).

    Parameters
    ----------
    coords : (N, 3) positions in Ångström
    radii : (N,) radii in Ångström
    resolution : map resolution (Å)
    voxel_size : grid spacing (Å)
    box_size : total side-length of cubic grid (Å), centred at origin

    Returns
    -------
    density : (nz, ny, nx) numpy array
    bins : tuple of 3 numpy arrays (x, y, z bin edges)
    """
    grid_dim = int(math.ceil(box_size / voxel_size))
    half = (grid_dim * voxel_size) / 2.0
    bins_1d = jnp.linspace(-half, half, grid_dim + 1)
    bins = (bins_1d, bins_1d, bins_1d)

    coords_jax = jnp.array(coords, dtype=jnp.float32)
    weights = jnp.array(radii, dtype=jnp.float32) ** 3

    density = calc_projection_jax(coords_jax, weights, bins, float(resolution))

    bins_np = np.array(bins_1d)
    return np.array(density), (bins_np, bins_np.copy(), bins_np.copy())


def create_em_config_from_coords(
    coords: np.ndarray,
    radii: np.ndarray,
    resolution: float,
    voxel_size: float,
    box_size: float,
) -> EMConfig:
    """
    Generate a synthetic target density and return an EMConfig.

    Because target and model maps share the same ``calc_projection_jax``
    code path, self-CCC is exactly 1.0.
    """
    density_np, bins_np = generate_density_map(
        coords, radii, resolution, voxel_size, box_size,
    )
    target = jnp.array(density_np, dtype=jnp.float32)
    bx = jnp.array(bins_np[0])
    by = jnp.array(bins_np[1])
    bz = jnp.array(bins_np[2])
    com = _calculate_density_com(target, bx, by, bz)
    return EMConfig(target, bx, by, bz, float(resolution), float(voxel_size), com)


def save_density_as_mrc(
    density: np.ndarray,
    voxel_size: float,
    bins: Tuple[np.ndarray, np.ndarray, np.ndarray],
    filename: str,
    resolution: float = 0.0,
) -> None:
    """Save a density array to an MRC file with correct header metadata."""
    import mrcfile

    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(density.astype(np.float32))
        mrc.voxel_size = voxel_size
        mrc.header.origin.x = float(bins[0][0])
        mrc.header.origin.y = float(bins[1][0])
        mrc.header.origin.z = float(bins[2][0])
        mrc.header.nxstart = int(round(float(bins[0][0]) / voxel_size))
        mrc.header.nystart = int(round(float(bins[1][0]) / voxel_size))
        mrc.header.nzstart = int(round(float(bins[2][0]) / voxel_size))
        if resolution > 0:
            mrc.header.label[0] = (
                f"Res: {resolution:.2f}A".ljust(80).encode("utf-8")
            )
        mrc.update_header_stats()


# =====================================================================
# CCC computation (single source of truth)
# =====================================================================

@jit
def calculate_ccc_jax(
    coords: jnp.ndarray,
    radii: jnp.ndarray,
    config: EMConfig,
) -> float:
    """
    Raw cross-correlation coefficient for CG spheres.

    Parameters
    ----------
    coords : (N, 3) positions
    radii : (N,) radii  (weights = radii**3 applied internally)
    config : EMConfig

    Returns
    -------
    ccc : float in [-1, 1]
    """
    weights = radii ** 3
    bins = (config.bins_x, config.bins_y, config.bins_z)
    projection = calc_projection_jax(coords, weights, bins, config.resolution)
    return pairwise_correlation_jax(
        projection.flatten(), config.target_data.flatten()
    )


def calculate_ccc_score(
    coords: np.ndarray, radii: np.ndarray, config: EMConfig,
) -> float:
    """Convenience wrapper returning a Python float."""
    return float(
        calculate_ccc_jax(jnp.array(coords), jnp.array(radii), config)
    )


# =====================================================================
# Safe distance computation
# =====================================================================

@jit
def _safe_distances_to_com(
    coords: jnp.ndarray, com: jnp.ndarray,
) -> jnp.ndarray:
    """
    ||r_i - r_COM||_safe  =  sqrt(||·||^2 + eps)

    Avoids gradient singularity of Euclidean norm at r = 0.
    """
    diff = coords - com[None, :]
    return jnp.sqrt(jnp.sum(diff ** 2, axis=1) + _NORM_EPS)


# =====================================================================
# Probabilistic model components
# =====================================================================

def create_gaussian_ccc_log_likelihood(
    config: EMConfig,
    radii: np.ndarray,
    sigma_ccc: float = 0.3,
) -> Callable:
    """
    Create a Gaussian log-likelihood based on (1 - CCC).

    Model::

        log L(x) = -(1 - CCC(x))^2 / (2 * sigma_ccc^2)

    Maximised when CCC → 1.  ``sigma_ccc`` controls sharpness:

    ============  ===============================================
    sigma_ccc     behaviour
    ============  ===============================================
    0.1           only CCC > ~0.9 gets significant probability
    0.3           moderate discrimination (good default)
    1.0           very flat, weak CCC contribution
    ============  ===============================================

    Parameters
    ----------
    config : EMConfig
    radii : array-like  (particle radii, weights = radii**3)
    sigma_ccc : float

    Returns
    -------
    log_likelihood_fn : flat_coords (N*3,) → scalar
    """
    radii_jax = jnp.array(radii, dtype=jnp.float32)
    inv_2s2 = 1.0 / (2.0 * sigma_ccc ** 2)

    @jit
    def log_likelihood_fn(flat_coords: jnp.ndarray) -> float:
        coords = flat_coords.reshape(-1, 3)
        ccc = calculate_ccc_jax(coords, radii_jax, config)
        mismatch = 1.0 - ccc
        return -(mismatch ** 2) * inv_2s2

    return log_likelihood_fn


def create_exponential_distance_log_prior(
    config: EMConfig,
    lambda_attract: float = 0.001,
) -> Callable:
    """
    Exponential (Laplace-type) prior pulling particles toward the
    density centre of mass.

    Model::

        log pi_attract(x) = -lambda * sum_i ||r_i - r_COM||

    Parameters
    ----------
    config : EMConfig (provides r_COM)
    lambda_attract : float

    Returns
    -------
    log_prior_fn : flat_coords (N*3,) → scalar
    """
    com = config.density_com

    @jit
    def log_prior_fn(flat_coords: jnp.ndarray) -> float:
        coords = flat_coords.reshape(-1, 3)
        return -lambda_attract * jnp.sum(
            _safe_distances_to_com(coords, com)
        )

    return log_prior_fn


def create_soft_box_log_prior(
    box_size: float,
    steepness: float = 1.0,
) -> Callable:
    """
    Differentiable soft-wall box prior.

    Model::

        log pi_box(x) = -sum_j max(|x_j| - box_size, 0)^2 / (2 * steepness^2)

    Parameters
    ----------
    box_size : float  (half-width of the cubic box)
    steepness : float (wall softness; smaller = harder wall)

    Returns
    -------
    log_prior_fn : flat_coords (N*3,) → scalar
    """
    inv_2s2 = 1.0 / (2.0 * steepness ** 2)

    @jit
    def log_prior_fn(flat_coords: jnp.ndarray) -> float:
        excess = jnp.maximum(jnp.abs(flat_coords) - box_size, 0.0)
        return -jnp.sum(excess ** 2) * inv_2s2

    return log_prior_fn


# =====================================================================
# Combined scoring model factory for SMC
# =====================================================================

def create_em_scoring_model(
    config: EMConfig,
    radii: np.ndarray,
    sigma_ccc: float = 0.3,
    lambda_attract: float = 0.001,
    box_size: float = 500.0,
    box_steepness: float = 1.0,
) -> Tuple[Callable, Callable, Callable]:
    """
    Build ``(log_prior, log_likelihood, log_prob)`` for BlackJAX SMC.

    SMC tempers::

        pi_t(x)  ∝  prior(x) * likelihood(x)^{lambda_t}

    The prior (box + attraction) is always on; the CCC likelihood is
    gradually brought in by the adaptive tempering schedule.

    Parameters
    ----------
    config : EMConfig
    radii : array-like
    sigma_ccc, lambda_attract, box_size, box_steepness : floats

    Returns
    -------
    log_prior_fn : flat_coords → scalar
    log_likelihood_fn : flat_coords → scalar
    log_prob_fn : flat_coords → scalar  (= prior + likelihood)
    """
    box = create_soft_box_log_prior(box_size, box_steepness)
    attract = create_exponential_distance_log_prior(config, lambda_attract)
    ccc_lik = create_gaussian_ccc_log_likelihood(config, radii, sigma_ccc)

    @jit
    def log_prior_fn(flat_coords: jnp.ndarray) -> float:
        return box(flat_coords) + attract(flat_coords)

    @jit
    def log_likelihood_fn(flat_coords: jnp.ndarray) -> float:
        return ccc_lik(flat_coords)

    @jit
    def log_prob_fn(flat_coords: jnp.ndarray) -> float:
        return log_prior_fn(flat_coords) + log_likelihood_fn(flat_coords)

    return log_prior_fn, log_likelihood_fn, log_prob_fn


# =====================================================================
# Diagnostics
# =====================================================================

def diagnose_model(
    flat_coords: jnp.ndarray,
    config: EMConfig,
    radii: np.ndarray,
    sigma_ccc: float = 0.3,
    lambda_attract: float = 0.001,
    box_size: float = 500.0,
) -> dict:
    """
    Compute all model components for a given configuration.

    Useful for understanding the relative contributions of each term.

    Returns
    -------
    dict with keys:
        ccc, mismatch, log_lik_ccc, mean_distance_to_com,
        log_prior_attract, log_prior_box, log_prior_total, log_posterior
    """
    radii_jax = jnp.array(radii, dtype=jnp.float32)
    coords = flat_coords.reshape(-1, 3)

    # CCC
    ccc = float(calculate_ccc_jax(coords, radii_jax, config))
    mismatch = 1.0 - ccc

    # Likelihood
    inv_2s2 = 1.0 / (2.0 * sigma_ccc ** 2)
    log_lik = -(mismatch ** 2) * inv_2s2

    # Distances
    distances = _safe_distances_to_com(coords, config.density_com)
    mean_dist = float(jnp.mean(distances))
    log_attract = -lambda_attract * float(jnp.sum(distances))

    # Box
    excess = jnp.maximum(jnp.abs(flat_coords) - box_size, 0.0)
    log_box = -float(jnp.sum(excess ** 2) / 2.0)

    return {
        "ccc": ccc,
        "mismatch": mismatch,
        "log_lik_ccc": log_lik,
        "mean_distance_to_com": mean_dist,
        "log_prior_attract": log_attract,
        "log_prior_box": log_box,
        "log_prior_total": log_attract + log_box,
        "log_posterior": log_lik + log_attract + log_box,
    }


# =====================================================================
# Legacy API (backward compatibility with em_score.py)
# =====================================================================

def create_em_log_prob_fn(
    config: EMConfig,
    radii: np.ndarray,
    scale: float = 100.0,
    slope: float = 0.0,
) -> Callable:
    """
    Legacy log probability function  ``scale * CCC - slope * distance``.

    DEPRECATED — use ``create_em_scoring_model()`` for proper Bayesian model.
    Kept for backward compatibility with existing RMH scripts.
    """
    radii_jax = jnp.array(radii, dtype=jnp.float32)

    @jit
    def log_prob_fn(flat_coords: jnp.ndarray) -> float:
        coords = flat_coords.reshape(-1, 3)
        ccc = calculate_ccc_jax(coords, radii_jax, config)
        log_prob = scale * ccc

        weights = radii_jax ** 3
        dists = _safe_distances_to_com(coords, config.density_com)
        penalty = slope * jnp.sum(dists * weights) / (jnp.sum(weights) + _NORM_EPS)
        return log_prob - penalty

    return log_prob_fn


def create_em_energy_fn(
    config: EMConfig,
    radii: np.ndarray,
    scale: float = 100.0,
    slope: float = 0.0,
) -> Callable:
    """Legacy energy function for minimisation. DEPRECATED."""
    radii_jax = jnp.array(radii, dtype=jnp.float32)

    @jit
    def energy_fn(flat_coords: jnp.ndarray) -> float:
        coords = flat_coords.reshape(-1, 3)
        ccc = calculate_ccc_jax(coords, radii_jax, config)
        return scale * (1.0 - ccc)

    return energy_fn
