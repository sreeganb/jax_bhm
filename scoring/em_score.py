"""
JAX-based EM density scoring for coarse-grained particle systems.
Adapted from Arthur Zalevsky's accc_jax.py for sphere-based models.

This module extracts the core JAX functions from accc_jax.py and adapts them
to work with coarse-grained representations where particles have radii
and weights are computed as radius^3 (volume).
"""
import math
import numpy as np
from typing import Tuple, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax.scipy import signal

jit = jax.jit
device_put = jax.device_put


# =============================================================================
# Core functions from accc_jax.py (Arthur Zalevsky)
# =============================================================================

def resolution_to_sigma(resolution: float, pixel_size: float) -> float:
    """Estimate sigma for gaussian smoothing from resolution (from IMP.em2d)."""
    return resolution / (4 * math.sqrt(2.0 * math.log(2.0))) / pixel_size


@jit
def pairwise_correlation_jax(A: jnp.ndarray, B: jnp.ndarray) -> float:
    """Calculate Pearson correlation coefficient between two arrays."""
    A = A.astype(jnp.float32)
    B = B.astype(jnp.float32)
    
    A_centered = A - jnp.mean(A)
    B_centered = B - jnp.mean(B)
    
    numerator = jnp.sum(A_centered * B_centered)
    denominator = jnp.sqrt(jnp.sum(A_centered ** 2) * jnp.sum(B_centered ** 2))
    
    return jnp.where(denominator > 0, numerator / denominator, 0.0)


def _calc_projection_impl(coords, weights, bins, resolution):
    """
    Core projection calculation from accc_jax.py.
    Computes weighted histogram + Gaussian blur.
    """
    nx = bins[0].shape[0] - 1
    ny = bins[1].shape[0] - 1
    nz = bins[2].shape[0] - 1
    
    # Voxel sizes and origins
    voxel_size_x = bins[0][1] - bins[0][0]
    voxel_size_y = bins[1][1] - bins[1][0]
    voxel_size_z = bins[2][1] - bins[2][0]
    origin_x, origin_y, origin_z = bins[0][0], bins[1][0], bins[2][0]
    
    # Bin coordinates (faster than digitize for uniform grids)
    x_indices = jnp.clip(((coords[:, 0] - origin_x) / voxel_size_x).astype(jnp.int32), 0, nx - 1)
    y_indices = jnp.clip(((coords[:, 1] - origin_y) / voxel_size_y).astype(jnp.int32), 0, ny - 1)
    z_indices = jnp.clip(((coords[:, 2] - origin_z) / voxel_size_z).astype(jnp.int32), 0, nz - 1)
    
    # Linear indices and scatter-add
    linear_indices = x_indices * (ny * nz) + y_indices * nz + z_indices
    histogram_flat = jnp.zeros(nx * ny * nz, dtype=jnp.float32)
    histogram_flat = histogram_flat.at[linear_indices].add(weights)
    
    # Reshape to (x, y, z) then swap to (z, y, x) for mrcfile convention
    img = histogram_flat.reshape((nx, ny, nz))
    img = jnp.swapaxes(img, 0, 2)
    
    # Gaussian blur
    sigma = resolution_to_sigma(resolution, voxel_size_x)
    max_radius = 30
    x = jnp.arange(-max_radius, max_radius + 1, dtype=jnp.float32)
    kernel_1d = jnp.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d * (jnp.abs(x) <= 4.0 * sigma)
    kernel_1d = kernel_1d / jnp.sum(kernel_1d)
    
    # Separable FFT convolution (fast for large arrays)
    img = signal.fftconvolve(img, kernel_1d[jnp.newaxis, jnp.newaxis, :], mode='same')
    img = signal.fftconvolve(img, kernel_1d[jnp.newaxis, :, jnp.newaxis], mode='same')
    img = signal.fftconvolve(img, kernel_1d[:, jnp.newaxis, jnp.newaxis], mode='same')
    
    return img.astype(jnp.float32)


@jit
def calc_projection_jax(coords, weights, bins, resolution):
    """JIT-compiled density projection."""
    return _calc_projection_impl(coords, weights, bins, resolution)


@jit
def compare_data_jax_full(projection, density_data):
    """Compare projection with density (full map, all voxels)."""
    return pairwise_correlation_jax(projection.flatten(), density_data.flatten())


# =============================================================================
# Adaptation layer for coarse-grained sphere systems
# =============================================================================

class EMConfig(NamedTuple):
    """Configuration for EM density scoring."""
    target_data: jnp.ndarray   # Target density map data (z, y, x)
    bins_x: jnp.ndarray        # Bin edges x
    bins_y: jnp.ndarray        # Bin edges y
    bins_z: jnp.ndarray        # Bin edges z
    resolution: float          # Map resolution in Angstroms
    voxel_size: float          # Voxel size in Angstroms


def create_em_config_from_mrcfile(density, resolution: float, center_at_origin: bool = True) -> EMConfig:
    """
    Create EMConfig from an mrcfile density object.
    
    Args:
        density: mrcfile object (from mrcfile.open())
        resolution: Map resolution in Angstroms
        center_at_origin: If True, center bins at origin (recommended for most cases)
    
    Returns:
        EMConfig ready for use with scoring functions
    """
    nx, ny, nz = density.header.nx, density.header.ny, density.header.nz
    vx = float(density.voxel_size.x)
    
    if center_at_origin:
        # Center map at origin (matches FullSampler.bins_from_density)
        x_ext, y_ext, z_ext = nx * vx / 2, ny * vx / 2, nz * vx / 2
        bins_x = jnp.linspace(-x_ext, x_ext, nx + 1)
        bins_y = jnp.linspace(-y_ext, y_ext, ny + 1)
        bins_z = jnp.linspace(-z_ext, z_ext, nz + 1)
    else:
        # Use mrcfile header info
        bins_x = (jnp.linspace(0, nx, nx + 1) + density.nstart.x) * vx
        bins_y = (jnp.linspace(0, ny, ny + 1) + density.nstart.y) * vx
        bins_z = (jnp.linspace(0, nz, nz + 1) + density.nstart.z) * vx
    
    return EMConfig(
        target_data=jnp.array(density.data, dtype=jnp.float32),
        bins_x=bins_x,
        bins_y=bins_y,
        bins_z=bins_z,
        resolution=resolution,
        voxel_size=vx
    )


def create_em_config_from_arrays(
    target_data: np.ndarray,
    voxel_size: float,
    resolution: float,
    center_at_origin: bool = True
) -> EMConfig:
    """
    Create EMConfig from numpy arrays.
    
    Args:
        target_data: 3D numpy array (z, y, x) of density values
        voxel_size: Voxel size in Angstroms
        resolution: Map resolution in Angstroms
        center_at_origin: If True, center bins at origin
    """
    nz, ny, nx = target_data.shape
    
    if center_at_origin:
        x_ext = nx * voxel_size / 2
        y_ext = ny * voxel_size / 2
        z_ext = nz * voxel_size / 2
        bins_x = jnp.linspace(-x_ext, x_ext, nx + 1)
        bins_y = jnp.linspace(-y_ext, y_ext, ny + 1)
        bins_z = jnp.linspace(-z_ext, z_ext, nz + 1)
    else:
        bins_x = jnp.linspace(0, nx * voxel_size, nx + 1)
        bins_y = jnp.linspace(0, ny * voxel_size, ny + 1)
        bins_z = jnp.linspace(0, nz * voxel_size, nz + 1)
    
    return EMConfig(
        target_data=jnp.array(target_data, dtype=jnp.float32),
        bins_x=bins_x,
        bins_y=bins_y,
        bins_z=bins_z,
        resolution=resolution,
        voxel_size=voxel_size
    )

@jit
def calculate_density_com(
    target_data: jnp.ndarray,
    bins_x: jnp.ndarray,
    bins_y: jnp.ndarray,
    bins_z: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate center of mass of target density map.
    Returns (3,) array of COM coordinates.
    """
    # Get voxel centers
    cx = (bins_x[:-1] + bins_x[1:]) / 2
    cy = (bins_y[:-1] + bins_y[1:]) / 2
    cz = (bins_z[:-1] + bins_z[1:]) / 2
    
    # Create meshgrid (note: target_data is (z, y, x) order)
    Z, Y, X = jnp.meshgrid(cz, cy, cx, indexing='ij')
    
    # Use positive density only
    density_pos = jnp.maximum(target_data, 0)
    total_mass = jnp.sum(density_pos)
    
    com_x = jnp.sum(X * density_pos) / (total_mass + 1e-10)
    com_y = jnp.sum(Y * density_pos) / (total_mass + 1e-10)
    com_z = jnp.sum(Z * density_pos) / (total_mass + 1e-10)
    
    return jnp.array([com_x, com_y, com_z])

@jit
def calculate_ccc_jax(
    coords: jnp.ndarray,
    radii: jnp.ndarray,
    target_data: jnp.ndarray,
    bins_x: jnp.ndarray,
    bins_y: jnp.ndarray,
    bins_z: jnp.ndarray,
    resolution: float,
    density_com: jnp.ndarray,
    slope: float = 0.0
) -> float:
    """
    Calculate cross-correlation coefficient for coarse-grained spheres.
    
    Args:
        coords: (N, 3) particle coordinates
        radii: (N,) particle radii
        target_data: Target density map (z, y, x)
        bins_x/y/z: Bin edges for each dimension
        resolution: Map resolution
    
    Returns:
        Cross-correlation coefficient (higher = better fit)
    """
    # Use volume (radius^3) as weight, matching FullSampler behavior
    weights = radii ** 3
    
    # Create bins tuple for projection function
    bins = (bins_x, bins_y, bins_z)
    
    # Calculate simulated projection
    projection = calc_projection_jax(coords, weights, bins, resolution)
    
    # calculate CCC
    ccc = compare_data_jax_full(projection, target_data)
    
    # Calculate slope penalty (mean distance to density COM)
    if slope > 0:
        distances = jnp.linalg.norm(coords - density_com, axis=1)
        # Weight by particle volume for consistency
        weighted_dist = jnp.sum(distances * weights) / jnp.sum(weights)
        slope_penalty = slope * weighted_dist
    else:
        slope_penalty = 0.0
    
    # CCC is in [-1, 1], slope_penalty is in [0, inf)
    # Return combined score (higher = better)
    return ccc - slope_penalty


def calculate_ccc_score(
    coords: np.ndarray,
    radii: np.ndarray,
    config: EMConfig
) -> float:
    """
    Calculate CCC score (convenience wrapper).
    
    Args:
        coords: (N, 3) particle coordinates (numpy or jax array)
        radii: (N,) particle radii (numpy or jax array)
        config: EMConfig with target density and grid info
    
    Returns:
        Cross-correlation coefficient as Python float
    """
    coords_jax = jnp.array(coords, dtype=jnp.float32)
    radii_jax = jnp.array(radii, dtype=jnp.float32)
    
    ccc = calculate_ccc_jax(
        coords_jax, radii_jax,
        config.target_data,
        config.bins_x, config.bins_y, config.bins_z,
        config.resolution
    )
    return float(ccc)


# =============================================================================
# Log probability functions for MCMC samplers
# =============================================================================

@jit
def em_log_probability(
    coords: jnp.ndarray,
    radii: jnp.ndarray,
    target_data: jnp.ndarray,
    bins_x: jnp.ndarray,
    bins_y: jnp.ndarray,
    bins_z: jnp.ndarray,
    resolution: float,
    scale: float = 100.0
) -> float:
    """
    Log probability based on EM density cross-correlation.
    
    Higher CCC = higher log probability (for MCMC maximization).
    log_prob = scale * CCC, so range is [-scale, +scale].
    
    Args:
        coords: (N, 3) particle coordinates
        radii: (N,) particle radii
        target_data: Target density map
        bins_x/y/z: Bin edges
        resolution: Map resolution
        scale: Scaling factor for CCC -> log_prob
    
    Returns:
        Log probability (higher = better fit)
    """
    ccc = calculate_ccc_jax(coords, radii, target_data, bins_x, bins_y, bins_z, resolution)
    return scale * ccc


def create_em_log_prob_fn(config: EMConfig, 
                          radii: np.ndarray, 
                          scale: float = 100.0,
                          slope: float = 0.0):
    """
    Create a log probability function compatible with MCMC samplers.
    
    The returned function takes flat coordinates (N*3,) and returns log probability.
    Radii are fixed (baked into the closure).
    
    Args:
        config: EMConfig with target density and grid info
        radii: (N,) particle radii (fixed during sampling)
        scale: CCC to log-prob scaling factor
        slope: Slope penalty factor
    Returns:
        Function: flat_coords (N*3,) -> log_probability
    
    Example:
        >>> config = create_em_config_from_mrcfile(density, resolution=50.0)
        >>> radii = np.array([24.0]*8 + [14.0]*8 + [16.0]*16)
        >>> log_prob_fn = create_em_log_prob_fn(config, radii)
        >>> log_prob = log_prob_fn(coords.flatten())
    """
    radii_jax = jnp.array(radii, dtype=jnp.float32)
    
    # Pre-compute density COM (only once!)
    density_com = calculate_density_com(
        config.target_data,
        config.bins_x, config.bins_y, config.bins_z
    )
    
    @jit
    def log_prob_fn(flat_coords: jnp.ndarray) -> float:
        coords = flat_coords.reshape(-1, 3)
        
        # CCC with slope
        score = calculate_ccc_jax(
            coords, radii_jax,
            config.target_data,
            config.bins_x, config.bins_y, config.bins_z,
            config.resolution,
            density_com,
            slope
        )
        
        # Convert to log probability
        # score is CCC - slope_penalty, range approximately [-1 - penalty, 1]
        # We want higher score = higher log prob
        return scale * score
    
    return log_prob_fn


def create_em_energy_fn(config: EMConfig, radii: np.ndarray, scale: float = 100.0):
    """
    Create an energy function (negative log probability) for minimization.
    
    Energy = scale * (1 - CCC), so lower energy = better fit.
    
    Args:
        config: EMConfig with target density and grid info
        radii: (N,) particle radii (fixed during sampling)
        scale: Scaling factor
    
    Returns:
        Function: flat_coords (N*3,) -> energy (lower = better)
    """
    radii_jax = jnp.array(radii, dtype=jnp.float32)
    
    @jit
    def energy_fn(flat_coords: jnp.ndarray) -> float:
        coords = flat_coords.reshape(-1, 3)
        ccc = calculate_ccc_jax(
            coords, radii_jax,
            config.target_data,
            config.bins_x, config.bins_y, config.bins_z,
            config.resolution
        )
        return scale * (1.0 - ccc)
    
    return energy_fn