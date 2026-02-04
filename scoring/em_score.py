"""
JAX-based EM density scoring for coarse-grained particle systems.
"""
import math
import numpy as np
from typing import Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy import signal

jit = jax.jit

# =============================================================================
# Core functions
# =============================================================================

def resolution_to_sigma(resolution: float, pixel_size: float) -> float:
    """Estimate sigma for gaussian smoothing from resolution."""
    return resolution / (4 * math.sqrt(2.0 * math.log(2.0))) / pixel_size


@jit
def pairwise_correlation_jax(A: jnp.ndarray, B: jnp.ndarray) -> float:
    """Calculate Pearson correlation coefficient between two arrays."""
    A_centered = A - jnp.mean(A)
    B_centered = B - jnp.mean(B)
    numerator = jnp.sum(A_centered * B_centered)
    denominator = jnp.sqrt(jnp.sum(A_centered ** 2) * jnp.sum(B_centered ** 2))
    return jnp.where(denominator > 0, numerator / denominator, 0.0)


@jit
def calc_projection_jax(coords, weights, bins, resolution):
    """Compute weighted histogram + Gaussian blur."""
    nx, ny, nz = bins[0].shape[0] - 1, bins[1].shape[0] - 1, bins[2].shape[0] - 1
    voxel_size_x = bins[0][1] - bins[0][0]
    origin_x, origin_y, origin_z = bins[0][0], bins[1][0], bins[2][0]
    
    x_indices = jnp.clip(((coords[:, 0] - origin_x) / voxel_size_x).astype(jnp.int32), 0, nx - 1)
    y_indices = jnp.clip(((coords[:, 1] - origin_y) / (bins[1][1] - bins[1][0])).astype(jnp.int32), 0, ny - 1)
    z_indices = jnp.clip(((coords[:, 2] - origin_z) / (bins[2][1] - bins[2][0])).astype(jnp.int32), 0, nz - 1)
    
    linear_indices = x_indices * (ny * nz) + y_indices * nz + z_indices
    histogram_flat = jnp.zeros(nx * ny * nz, dtype=jnp.float32).at[linear_indices].add(weights)
    img = jnp.swapaxes(histogram_flat.reshape((nx, ny, nz)), 0, 2)
    
    # Gaussian blur
    sigma = resolution_to_sigma(resolution, voxel_size_x)
    x = jnp.arange(-30, 31, dtype=jnp.float32)
    kernel_1d = jnp.exp(-0.5 * (x / sigma) ** 2) * (jnp.abs(x) <= 4.0 * sigma)
    kernel_1d = kernel_1d / jnp.sum(kernel_1d)
    
    for axis, shape in [(2, (1, 1, -1)), (1, (1, -1, 1)), (0, (-1, 1, 1))]:
        img = signal.fftconvolve(img, kernel_1d.reshape(shape), mode='same')
    
    return img.astype(jnp.float32)


# =============================================================================
# Configuration
# =============================================================================

class EMConfig(NamedTuple):
    """Configuration for EM density scoring."""
    target_data: jnp.ndarray
    bins_x: jnp.ndarray
    bins_y: jnp.ndarray
    bins_z: jnp.ndarray
    resolution: float
    voxel_size: float
    density_com: jnp.ndarray = None  # Added: precomputed COM


def _compute_bins(nx, ny, nz, voxel_size, center_at_origin):
    """Helper to compute bin edges."""
    if center_at_origin:
        return tuple(jnp.linspace(-d * voxel_size / 2, d * voxel_size / 2, d + 1) 
                     for d in (nx, ny, nz))
    return tuple(jnp.linspace(0, d * voxel_size, d + 1) for d in (nx, ny, nz))


@jit
def _calculate_density_com(target_data, bins_x, bins_y, bins_z):
    """Calculate center of mass of target density map."""
    cx, cy, cz = [(b[:-1] + b[1:]) / 2 for b in (bins_x, bins_y, bins_z)]
    Z, Y, X = jnp.meshgrid(cz, cy, cx, indexing='ij')
    density_pos = jnp.maximum(target_data, 0)
    total_mass = jnp.sum(density_pos) + 1e-10
    return jnp.array([jnp.sum(X * density_pos), jnp.sum(Y * density_pos), 
                      jnp.sum(Z * density_pos)]) / total_mass


def create_em_config_from_mrcfile(density, resolution: float, center_at_origin: bool = True) -> EMConfig:
    """Create EMConfig from an mrcfile density object."""
    nx, ny, nz = density.header.nx, density.header.ny, density.header.nz
    vx = float(density.voxel_size.x)
    target_data = jnp.array(density.data, dtype=jnp.float32)
    bins_x, bins_y, bins_z = _compute_bins(nx, ny, nz, vx, center_at_origin) if center_at_origin else (
        (jnp.linspace(0, nx, nx + 1) + density.nstart.x) * vx,
        (jnp.linspace(0, ny, ny + 1) + density.nstart.y) * vx,
        (jnp.linspace(0, nz, nz + 1) + density.nstart.z) * vx,
    )
    density_com = _calculate_density_com(target_data, bins_x, bins_y, bins_z)
    return EMConfig(target_data, bins_x, bins_y, bins_z, resolution, vx, density_com)


def create_em_config_from_arrays(target_data: np.ndarray, voxel_size: float, 
                                  resolution: float, center_at_origin: bool = True) -> EMConfig:
    """Create EMConfig from numpy arrays."""
    nz, ny, nx = target_data.shape
    target_jax = jnp.array(target_data, dtype=jnp.float32)
    bins_x, bins_y, bins_z = _compute_bins(nx, ny, nz, voxel_size, center_at_origin)
    density_com = _calculate_density_com(target_jax, bins_x, bins_y, bins_z)
    return EMConfig(target_jax, bins_x, bins_y, bins_z, resolution, voxel_size, density_com)


# =============================================================================
# CCC Calculation (single source of truth)
# =============================================================================

@jit
def calculate_ccc_jax(coords, radii, config: EMConfig, slope: float = 0.0) -> float:
    """Calculate cross-correlation coefficient for coarse-grained spheres."""
    weights = radii ** 3
    bins = (config.bins_x, config.bins_y, config.bins_z)
    projection = calc_projection_jax(coords, weights, bins, config.resolution)
    ccc = pairwise_correlation_jax(projection.flatten(), config.target_data.flatten())
    
    slope_penalty = jax.lax.cond(
        slope > 0.0,
        lambda _: slope * jnp.sum(jnp.linalg.norm(coords - config.density_com, axis=1) * weights) / jnp.sum(weights),
        lambda _: 0.0, operand=None
    )
    return ccc - slope_penalty


def calculate_ccc_score(coords: np.ndarray, radii: np.ndarray, config: EMConfig, slope: float = 0.0) -> float:
    """Calculate CCC score (convenience wrapper)."""
    return float(calculate_ccc_jax(jnp.array(coords), jnp.array(radii), config, slope))


# =============================================================================
# Log probability / Energy functions
# =============================================================================

def create_em_log_prob_fn(config: EMConfig, radii: np.ndarray, scale: float = 100.0, slope: float = 0.0):
    """Create a log probability function compatible with MCMC samplers."""
    radii_jax = jnp.array(radii, dtype=jnp.float32)
    
    @jit
    def log_prob_fn(flat_coords: jnp.ndarray) -> float:
        return scale * calculate_ccc_jax(flat_coords.reshape(-1, 3), radii_jax, config, slope)
    
    @jit
    def log_prob_with_ccc_fn(flat_coords: jnp.ndarray) -> Tuple[float, float]:
        raw_ccc = calculate_ccc_jax(flat_coords.reshape(-1, 3), radii_jax, config, 0.0)
        ccc_with_penalty = calculate_ccc_jax(flat_coords.reshape(-1, 3), radii_jax, config, slope)
        return scale * ccc_with_penalty, raw_ccc
    
    log_prob_fn.with_ccc = log_prob_with_ccc_fn
    return log_prob_fn


def create_em_energy_fn(config: EMConfig, radii: np.ndarray, scale: float = 100.0, slope: float = 0.0):
    """Create an energy function for minimization."""
    radii_jax = jnp.array(radii, dtype=jnp.float32)
    
    @jit
    def energy_fn(flat_coords: jnp.ndarray) -> float:
        return scale * (1.0 - calculate_ccc_jax(flat_coords.reshape(-1, 3), radii_jax, config, slope))
    
    return energy_fn