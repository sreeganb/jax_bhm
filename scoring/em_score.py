"""
JAX-based EM density scoring with a proper probabilistic model.

Probabilistic Model
===================

We define a Bayesian model for fitting coarse-grained particle positions
to an experimental cryo-EM density map.

**Likelihood (CCC-based Gaussian model):**

    The cross-correlation coefficient (CCC) between the simulated density
    (from current particle positions) and the experimental map quantifies
    fit quality. We define a Gaussian likelihood on the mismatch:

        log L(x | data) = -(1 - CCC(x))^2 / (2 * sigma_ccc^2)

    where:
        - CCC(x) in [-1, 1], typically [0, 1] for density maps
        - 1 - CCC(x) in [0, 2], with 0 = perfect match
        - sigma_ccc controls sensitivity:
            * sigma_ccc = 0.1 -> sharp: strong discrimination (CCC > 0.9 matters)
            * sigma_ccc = 0.3 -> moderate: good general-purpose setting
            * sigma_ccc = 1.0 -> flat: weak CCC contribution

    This is maximized when CCC -> 1 and smoothly differentiable everywhere,
    making it compatible with HMC.

**Spatial prior (exponential attraction to density center):**

    An exponential (Laplace-type) prior pulls particles toward the center
    of mass of the experimental density:

        log pi_attract(x) = -lambda_attract * sum_i ||r_i - r_COM||

    where:
        - r_i is the position of particle i
        - r_COM is the center of mass of the experimental density
        - lambda_attract controls attraction strength:
            * 0.001 -> very gentle nudge (recommended starting point)
            * 0.01  -> moderate pull
            * 0.1   -> strong (may collapse particles to center)

    We use a safe norm: sqrt(||r||^2 + eps) to ensure differentiability
    at r = r_COM (the Euclidean norm gradient is undefined at zero).

**Soft box prior (for HMC):**

    Instead of hard walls (which produce NaN gradients for HMC), we use
    a quadratic penalty outside the box:

        log pi_box(x) = -sum_j max(|x_j| - box_size, 0)^2 / (2 * steepness^2)

**In SMC, the tempered posterior at step t is:**

    pi_t(x) ∝ [pi_box(x) * pi_attract(x)] * L(x | data)^{lambda_t}
               \_________________________/   \____________________/
                      prior (always on)        likelihood (tempered)

    lambda_t goes from 0 to 1 adaptively. The prior guides particles toward
    the density region from the start; the CCC likelihood is gradually turned
    on to refine the fit.

Implementation Notes
====================
- All functions are JAX-differentiable (no np, no -inf, no Python control flow)
- Safe norm used everywhere for HMC gradient stability
- CCC computation uses FFT convolution (differentiable through JAX)
- GPU-compatible: all operations are jnp
"""
import math
import numpy as np
from typing import Tuple, NamedTuple, Callable, Optional

import jax
import jax.numpy as jnp
from jax.scipy import signal

jit = jax.jit

# Safe epsilon for norm gradients
_NORM_EPS = 1e-8


# =============================================================================
# Core CCC computation (unchanged, differentiable)
# =============================================================================

def resolution_to_sigma(resolution: float, pixel_size: float) -> float:
    """Estimate sigma for Gaussian smoothing from resolution."""
    return resolution / (4 * math.sqrt(2.0 * math.log(2.0))) / pixel_size


@jit
def pairwise_correlation_jax(A: jnp.ndarray, B: jnp.ndarray) -> float:
    """Pearson correlation coefficient between two arrays."""
    A_centered = A - jnp.mean(A)
    B_centered = B - jnp.mean(B)
    numerator = jnp.sum(A_centered * B_centered)
    denominator = jnp.sqrt(
        jnp.sum(A_centered ** 2) * jnp.sum(B_centered ** 2) + _NORM_EPS
    )
    return numerator / denominator


@jit
def calc_projection_jax(coords, weights, bins, resolution):
    """Compute weighted histogram + Gaussian blur (differentiable)."""
    nx = bins[0].shape[0] - 1
    ny = bins[1].shape[0] - 1
    nz = bins[2].shape[0] - 1

    voxel_size_x = bins[0][1] - bins[0][0]
    origin_x, origin_y, origin_z = bins[0][0], bins[1][0], bins[2][0]

    x_indices = jnp.clip(
        ((coords[:, 0] - origin_x) / voxel_size_x).astype(jnp.int32), 0, nx - 1
    )
    y_indices = jnp.clip(
        ((coords[:, 1] - origin_y) / (bins[1][1] - bins[1][0])).astype(jnp.int32),
        0, ny - 1,
    )
    z_indices = jnp.clip(
        ((coords[:, 2] - origin_z) / (bins[2][1] - bins[2][0])).astype(jnp.int32),
        0, nz - 1,
    )

    linear_indices = x_indices * (ny * nz) + y_indices * nz + z_indices
    histogram_flat = (
        jnp.zeros(nx * ny * nz, dtype=jnp.float32)
        .at[linear_indices]
        .add(weights)
    )
    img = jnp.swapaxes(histogram_flat.reshape((nx, ny, nz)), 0, 2)

    # Gaussian blur (separable, 3 passes)
    sigma = resolution_to_sigma(resolution, voxel_size_x)
    x = jnp.arange(-30, 31, dtype=jnp.float32)
    kernel_1d = jnp.exp(-0.5 * (x / sigma) ** 2) * (jnp.abs(x) <= 4.0 * sigma)
    kernel_1d = kernel_1d / jnp.sum(kernel_1d)

    for axis, shape in [(2, (1, 1, -1)), (1, (1, -1, 1)), (0, (-1, 1, 1))]:
        img = signal.fftconvolve(img, kernel_1d.reshape(shape), mode="same")

    return img.astype(jnp.float32)


# =============================================================================
# Configuration
# =============================================================================


class EMConfig(NamedTuple):
    """Configuration for EM density scoring."""

    target_data: jnp.ndarray       # (nz, ny, nx) experimental density
    bins_x: jnp.ndarray            # (nx+1,) bin edges
    bins_y: jnp.ndarray            # (ny+1,)
    bins_z: jnp.ndarray            # (nz+1,)
    resolution: float               # in Angstrom
    voxel_size: float               # in Angstrom
    density_com: jnp.ndarray        # (3,) center of mass of experimental density


def _compute_bins(nx, ny, nz, voxel_size, center_at_origin):
    if center_at_origin:
        return tuple(
            jnp.linspace(-d * voxel_size / 2, d * voxel_size / 2, d + 1)
            for d in (nx, ny, nz)
        )
    return tuple(jnp.linspace(0, d * voxel_size, d + 1) for d in (nx, ny, nz))


@jit
def _calculate_density_com(target_data, bins_x, bins_y, bins_z):
    """Center of mass of target density map."""
    cx = (bins_x[:-1] + bins_x[1:]) / 2
    cy = (bins_y[:-1] + bins_y[1:]) / 2
    cz = (bins_z[:-1] + bins_z[1:]) / 2
    Z, Y, X = jnp.meshgrid(cz, cy, cx, indexing="ij")
    density_pos = jnp.maximum(target_data, 0)
    total_mass = jnp.sum(density_pos) + _NORM_EPS
    return jnp.array([
        jnp.sum(X * density_pos),
        jnp.sum(Y * density_pos),
        jnp.sum(Z * density_pos),
    ]) / total_mass


def create_em_config_from_mrcfile(
    density, resolution: float, center_at_origin: bool = True
) -> EMConfig:
    """Create EMConfig from an mrcfile density object."""
    nx, ny, nz = density.header.nx, density.header.ny, density.header.nz
    vx = float(density.voxel_size.x)
    target_data = jnp.array(density.data, dtype=jnp.float32)

    if center_at_origin:
        bins_x, bins_y, bins_z = _compute_bins(nx, ny, nz, vx, True)
    else:
        bins_x = (jnp.linspace(0, nx, nx + 1) + density.nstart.x) * vx
        bins_y = (jnp.linspace(0, ny, ny + 1) + density.nstart.y) * vx
        bins_z = (jnp.linspace(0, nz, nz + 1) + density.nstart.z) * vx

    density_com = _calculate_density_com(target_data, bins_x, bins_y, bins_z)
    return EMConfig(target_data, bins_x, bins_y, bins_z, resolution, vx, density_com)


def create_em_config_from_arrays(
    target_data: np.ndarray,
    voxel_size: float,
    resolution: float,
    center_at_origin: bool = True,
) -> EMConfig:
    """Create EMConfig from numpy arrays."""
    nz, ny, nx = target_data.shape
    target_jax = jnp.array(target_data, dtype=jnp.float32)
    bins_x, bins_y, bins_z = _compute_bins(nx, ny, nz, voxel_size, center_at_origin)
    density_com = _calculate_density_com(target_jax, bins_x, bins_y, bins_z)
    return EMConfig(target_jax, bins_x, bins_y, bins_z, resolution, voxel_size, density_com)


# =============================================================================
# CCC computation (single source of truth)
# =============================================================================


@jit
def calculate_ccc_jax(
    coords: jnp.ndarray,
    radii: jnp.ndarray,
    config: EMConfig,
) -> float:
    """
    Calculate cross-correlation coefficient for coarse-grained spheres.

    This is the RAW CCC with no penalties or transformations.
    Use this for diagnostics and reporting.

    Parameters
    ----------
    coords : jnp.ndarray, shape (n_particles, 3)
    radii : jnp.ndarray, shape (n_particles,)
    config : EMConfig

    Returns
    -------
    ccc : float in [-1, 1]
    """
    weights = radii ** 3
    bins = (config.bins_x, config.bins_y, config.bins_z)
    projection = calc_projection_jax(coords, weights, bins, config.resolution)
    return pairwise_correlation_jax(projection.flatten(), config.target_data.flatten())


def calculate_ccc_score(
    coords: np.ndarray, radii: np.ndarray, config: EMConfig
) -> float:
    """Calculate raw CCC (convenience wrapper, returns Python float)."""
    return float(calculate_ccc_jax(jnp.array(coords), jnp.array(radii), config))


# =============================================================================
# Safe distance computation
# =============================================================================


@jit
def _safe_distances_to_com(
    coords: jnp.ndarray, com: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute distances from each particle to a reference point,
    using a safe norm that is differentiable at zero.

    ||r||_safe = sqrt(||r||^2 + eps)

    This avoids the gradient singularity of the Euclidean norm at r = 0.

    Parameters
    ----------
    coords : (n_particles, 3)
    com : (3,)

    Returns
    -------
    distances : (n_particles,)
    """
    diff = coords - com[None, :]  # (n_particles, 3)
    return jnp.sqrt(jnp.sum(diff ** 2, axis=1) + _NORM_EPS)


# =============================================================================
# Probabilistic model components
# =============================================================================


def create_gaussian_ccc_log_likelihood(
    config: EMConfig,
    radii: np.ndarray,
    sigma_ccc: float = 0.3,
) -> Callable:
    """
    Create a Gaussian log-likelihood based on (1 - CCC).

    Model:
        log L(x) = -(1 - CCC(x))^2 / (2 * sigma_ccc^2)

    This is maximized when CCC = 1 (perfect fit) and smoothly penalizes
    deviations. sigma_ccc controls how sharply:

        sigma_ccc = 0.1 : only CCC > ~0.9 gets significant probability
        sigma_ccc = 0.3 : moderate discrimination (good default)
        sigma_ccc = 1.0 : very flat, weak CCC contribution

    Parameters
    ----------
    config : EMConfig
        Density map configuration.
    radii : array-like
        Particle radii (for computing simulated density weights).
    sigma_ccc : float
        Standard deviation of the Gaussian on (1 - CCC).

    Returns
    -------
    log_likelihood_fn : Callable
        Takes flat_coords (n_particles*3,) -> scalar log-likelihood.
    """
    radii_jax = jnp.array(radii, dtype=jnp.float32)
    inv_two_sigma_sq = 1.0 / (2.0 * sigma_ccc ** 2)

    @jit
    def log_likelihood_fn(flat_coords: jnp.ndarray) -> float:
        coords = flat_coords.reshape(-1, 3)
        ccc = calculate_ccc_jax(coords, radii_jax, config)
        mismatch = 1.0 - ccc  # in [0, 2]
        return -(mismatch ** 2) * inv_two_sigma_sq

    return log_likelihood_fn


def create_exponential_distance_log_prior(
    config: EMConfig,
    lambda_attract: float = 0.001,
) -> Callable:
    """
    Create an exponential (Laplace-type) spatial prior that pulls particles
    toward the center of mass of the experimental density.

    Model:
        log pi_attract(x) = -lambda_attract * sum_i ||r_i - r_COM||

    This creates a gentle funnel toward the density center without
    collapsing particles to a point.

    Parameters
    ----------
    config : EMConfig
        Density map configuration (provides r_COM).
    lambda_attract : float
        Attraction strength. Guidelines:
            0.0001 - 0.001 : very gentle (particles wander freely)
            0.001  - 0.01  : moderate (recommended starting range)
            0.01   - 0.1   : strong (may over-constrain)

    Returns
    -------
    log_prior_fn : Callable
        Takes flat_coords (n_particles*3,) -> scalar log-prior.
    """
    density_com = config.density_com

    @jit
    def log_prior_fn(flat_coords: jnp.ndarray) -> float:
        coords = flat_coords.reshape(-1, 3)
        distances = _safe_distances_to_com(coords, density_com)
        return -lambda_attract * jnp.sum(distances)

    return log_prior_fn


def create_soft_box_log_prior(
    box_size: float,
    steepness: float = 1.0,
) -> Callable:
    """
    Create a differentiable soft-wall box prior.

    Model:
        log pi_box(x) = -sum_j max(|x_j| - box_size, 0)^2 / (2 * steepness^2)

    Parameters
    ----------
    box_size : float
        Half-width of the cubic box.
    steepness : float
        Wall softness. Smaller = harder wall.

    Returns
    -------
    log_prior_fn : Callable
        Takes flat_coords (n_particles*3,) -> scalar log-prior.
    """
    inv_two_s_sq = 1.0 / (2.0 * steepness ** 2)

    @jit
    def log_prior_fn(flat_coords: jnp.ndarray) -> float:
        excess = jnp.maximum(jnp.abs(flat_coords) - box_size, 0.0)
        return -jnp.sum(excess ** 2) * inv_two_s_sq

    return log_prior_fn


# =============================================================================
# Combined model factory for SMC
# =============================================================================


def create_em_scoring_model(
    config: EMConfig,
    radii: np.ndarray,
    sigma_ccc: float = 0.3,
    lambda_attract: float = 0.001,
    box_size: float = 500.0,
    box_steepness: float = 1.0,
) -> Tuple[Callable, Callable, Callable]:
    """
    Create the complete probabilistic model for SMC sampling.

    Returns three functions matching BlackJAX's SMC interface:
        - log_prior_fn:      soft box + exponential distance prior
        - log_likelihood_fn:  Gaussian CCC likelihood
        - log_prob_fn:        prior + likelihood (for scoring/diagnostics)

    In SMC, the tempering schedule applies only to the likelihood:
        pi_t(x) ∝ prior(x) * likelihood(x)^{lambda_t}

    So the attraction prior guides particles from the start, while the
    CCC data fit is gradually turned on.

    Parameters
    ----------
    config : EMConfig
    radii : array-like
        Particle radii.
    sigma_ccc : float
        Gaussian width on (1 - CCC). Smaller = sharper CCC discrimination.
    lambda_attract : float
        Exponential attraction strength to density COM.
    box_size : float
        Half-width of the soft box.
    box_steepness : float
        Softness of box walls.

    Returns
    -------
    log_prior_fn : Callable
        flat_coords -> scalar. Box + attraction prior.
    log_likelihood_fn : Callable
        flat_coords -> scalar. Gaussian CCC likelihood.
    log_prob_fn : Callable
        flat_coords -> scalar. Total log-posterior for scoring.

    Example
    -------
    >>> log_prior, log_likelihood, log_prob = create_em_scoring_model(
    ...     config, radii, sigma_ccc=0.3, lambda_attract=0.005
    ... )
    >>> # Use with BlackJAX SMC:
    >>> state, info, best_pos, best_scores = run_tempered_smc(
    ...     log_prior_fn=log_prior,
    ...     log_likelihood_fn=log_likelihood,
    ...     log_prob_fn=log_prob,
    ...     ...
    ... )
    """
    # Individual components
    box_prior = create_soft_box_log_prior(box_size, box_steepness)
    attract_prior = create_exponential_distance_log_prior(config, lambda_attract)
    ccc_likelihood = create_gaussian_ccc_log_likelihood(config, radii, sigma_ccc)

    @jit
    def log_prior_fn(flat_coords: jnp.ndarray) -> float:
        return box_prior(flat_coords) + attract_prior(flat_coords)

    @jit
    def log_likelihood_fn(flat_coords: jnp.ndarray) -> float:
        return ccc_likelihood(flat_coords)

    @jit
    def log_prob_fn(flat_coords: jnp.ndarray) -> float:
        return log_prior_fn(flat_coords) + log_likelihood_fn(flat_coords)

    return log_prior_fn, log_likelihood_fn, log_prob_fn


# =============================================================================
# Diagnostic utilities
# =============================================================================


def diagnose_model(
    flat_coords: jnp.ndarray,
    config: EMConfig,
    radii: np.ndarray,
    sigma_ccc: float = 0.3,
    lambda_attract: float = 0.001,
    box_size: float = 500.0,
) -> dict:
    """
    Compute all model components for a given configuration. Useful for
    understanding the relative contributions of each term.

    Returns a dict with:
        ccc:              raw CCC value
        mismatch:         1 - CCC
        log_lik_ccc:      Gaussian CCC log-likelihood
        mean_distance:    mean particle distance to density COM
        log_prior_attract: exponential distance log-prior
        log_prior_box:    soft box log-prior
        log_prior_total:  combined prior
        log_posterior:    total log-posterior
    """
    radii_jax = jnp.array(radii, dtype=jnp.float32)
    coords = flat_coords.reshape(-1, 3)

    # CCC
    ccc = float(calculate_ccc_jax(coords, radii_jax, config))
    mismatch = 1.0 - ccc

    # Likelihood
    inv_two_sigma_sq = 1.0 / (2.0 * sigma_ccc ** 2)
    log_lik = -(mismatch ** 2) * inv_two_sigma_sq

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


# =============================================================================
# Legacy API (backward compatibility)
# =============================================================================


def create_em_log_prob_fn(
    config: EMConfig,
    radii: np.ndarray,
    scale: float = 100.0,
    slope: float = 0.0,
) -> Callable:
    """
    Legacy log probability function (scale * CCC - slope * distance).

    DEPRECATED: Use create_em_scoring_model() for proper probabilistic model.
    Kept for backward compatibility with existing RMH scripts.
    """
    radii_jax = jnp.array(radii, dtype=jnp.float32)

    @jit
    def log_prob_fn(flat_coords: jnp.ndarray) -> float:
        coords = flat_coords.reshape(-1, 3)
        ccc = calculate_ccc_jax(coords, radii_jax, config)
        log_prob = scale * ccc

        # Optional slope penalty (legacy behavior)
        weights = radii_jax ** 3
        distances = _safe_distances_to_com(coords, config.density_com)
        penalty = slope * jnp.sum(distances * weights) / (jnp.sum(weights) + _NORM_EPS)
        return log_prob - penalty

    return log_prob_fn


def create_em_energy_fn(
    config: EMConfig,
    radii: np.ndarray,
    scale: float = 100.0,
    slope: float = 0.0,
) -> Callable:
    """Legacy energy function for minimization. DEPRECATED."""
    radii_jax = jnp.array(radii, dtype=jnp.float32)

    @jit
    def energy_fn(flat_coords: jnp.ndarray) -> float:
        coords = flat_coords.reshape(-1, 3)
        ccc = calculate_ccc_jax(coords, radii_jax, config)
        return scale * (1.0 - ccc)

    return energy_fn