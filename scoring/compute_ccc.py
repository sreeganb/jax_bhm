#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JAX-based density map generation and cross-correlation for coarse-grained bead models.

Adapted from accc.py (Arthur Zalevsky) for coarse-grained particle systems
where each bead represents a protein or group of residues as a sphere
with defined radius, mass, and coordinates.

Each bead is represented as a 3D Gaussian blob whose sigma is derived from
its physical radius (radius of gyration equivalence). The density map is
the sum of all bead Gaussian contributions, optionally smoothed to a target
resolution.

Usage:
    # As a library
    from cg_density import setup_grid, calc_cg_density, compare_densities, write_map

    # Generate density from beads
    bins, grid_shape = setup_grid(box_size=(200, 200, 200), voxel_size=3.0)
    density = calc_cg_density(coords, masses, radii, bins, resolution=10.0)

    # Compare two densities
    ccc = compare_densities(density_a, density_b, mask_mode='full')
"""

__author__ = "Sree Ganesh Balasubramani (adapted from Arthur Zalevsky's accc.py)"
__version__ = "0.1.0"

import sys
import logging
import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
import mrcfile

import jax
import jax.numpy as jnp
from jax.scipy import signal

jit = jax.jit
device_put = jax.device_put

# ---------------------------------------------------------------------------
# Grid and sigma utilities
# ---------------------------------------------------------------------------

def resolution_to_sigma(resolution: float, voxel_size: float) -> float:
    """Convert map resolution (Å) to Gaussian sigma in **voxel** units.

    Uses the same convention as IMP.em / accc.py:
        sigma = resolution / (4 * sqrt(2 * ln2)) / voxel_size
    """
    return resolution / (4.0 * math.sqrt(2.0 * math.log(2.0))) / voxel_size


def radius_to_sigma(radius: float) -> float:
    """Convert bead radius (Å) to Gaussian sigma in **Å**.

    For a uniform-density sphere of radius R the equivalent Gaussian
    (matching the radius of gyration) has sigma = R * sqrt(3/5) ≈ 0.7746 R.
    """
    return radius * math.sqrt(3.0 / 5.0)


def setup_grid(
    box_size: Tuple[float, float, float],
    voxel_size: float,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[int, int, int]]:
    """Create grid bin edges for density map generation.

    Args:
        box_size:   (Lx, Ly, Lz) extent of the box in Å.
        voxel_size: uniform voxel size in Å.
        center:     (cx, cy, cz) centre of the box in Å (default: origin).

    Returns:
        bins:       (binsx, binsy, binsz) – JAX arrays of bin **edges**
                    with shape (nx+1,), (ny+1,), (nz+1,).
        grid_shape: (nx, ny, nz) number of voxels along each axis.
    """
    cx, cy, cz = center
    lx, ly, lz = box_size

    nx = int(round(lx / voxel_size))
    ny = int(round(ly / voxel_size))
    nz = int(round(lz / voxel_size))

    # Bin edges centred on `center`
    binsx = jnp.linspace(cx - lx / 2.0, cx + lx / 2.0, nx + 1, dtype=jnp.float32)
    binsy = jnp.linspace(cy - ly / 2.0, cy + ly / 2.0, ny + 1, dtype=jnp.float32)
    binsz = jnp.linspace(cz - lz / 2.0, cz + lz / 2.0, nz + 1, dtype=jnp.float32)

    return (binsx, binsy, binsz), (nx, ny, nz)


def grid_centers_from_bins(bins):
    """Return 1-D grid-centre arrays from bin-edge arrays.

    Args:
        bins: (binsx, binsy, binsz)

    Returns:
        (cx, cy, cz) each of shape (n,) – centres of the voxels.
    """
    return tuple((b[:-1] + b[1:]) / 2.0 for b in bins)


# ---------------------------------------------------------------------------
# Bead density generation (Gaussian blobs)
# ---------------------------------------------------------------------------

@jit
def _gaussian_blob_density_same_sigma(
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    grid_z: jnp.ndarray,
    centers: jnp.ndarray,
    masses: jnp.ndarray,
    sigma: float,
) -> jnp.ndarray:
    """Compute density for a group of beads that share the same sigma.

    Uses the separable property of Gaussians for efficiency:
        G(x,y,z) = Gx(x) · Gy(y) · Gz(z)

    Args:
        grid_x, grid_y, grid_z: 1-D arrays of grid centres (Å).
        centers: (N, 3) bead centres in Å.
        masses:  (N,) bead masses in Da.
        sigma:   Gaussian sigma in Å (same for all beads in this call).

    Returns:
        density: (nz, ny, nx) density array (mrcfile z-y-x convention).
    """
    # 1-D Gaussian evaluations: shape (N, n_i)
    gx = jnp.exp(-0.5 * ((grid_x[None, :] - centers[:, 0:1]) / sigma) ** 2)
    gy = jnp.exp(-0.5 * ((grid_y[None, :] - centers[:, 1:2]) / sigma) ** 2)
    gz = jnp.exp(-0.5 * ((grid_z[None, :] - centers[:, 2:3]) / sigma) ** 2)

    norm = (2.0 * jnp.pi * sigma ** 2) ** 1.5
    weights = masses / norm  # (N,)

    # Accumulate: density[z, y, x] = Σ_i  w_i · gz_i[z] · gy_i[y] · gx_i[x]
    density = jnp.einsum("i,iz,iy,ix->zyx", weights, gz, gy, gx)
    return density


def _gaussian_blob_density_general(
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    grid_z: jnp.ndarray,
    centers: jnp.ndarray,
    masses: jnp.ndarray,
    sigmas: jnp.ndarray,
) -> jnp.ndarray:
    """Compute density for beads with per-bead sigma using lax.scan.

    Accumulates one bead at a time to avoid allocating (N, nz, ny, nx).
    Suitable when beads have heterogeneous radii and grouping is inconvenient.
    """

    def _add_one_bead(density, bead):
        cx, cy, cz, mass, sigma = bead
        gx = jnp.exp(-0.5 * ((grid_x - cx) / sigma) ** 2)
        gy = jnp.exp(-0.5 * ((grid_y - cy) / sigma) ** 2)
        gz = jnp.exp(-0.5 * ((grid_z - cz) / sigma) ** 2)
        norm = (2.0 * jnp.pi * sigma ** 2) ** 1.5
        blob = (mass / norm) * gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
        return density + blob, None

    nz = grid_z.shape[0]
    ny = grid_y.shape[0]
    nx = grid_x.shape[0]
    init = jnp.zeros((nz, ny, nx), dtype=jnp.float32)

    bead_data = jnp.column_stack([centers, masses[:, None], sigmas[:, None]])  # (N, 5)
    density, _ = jax.lax.scan(_add_one_bead, init, bead_data)
    return density


# ---------------------------------------------------------------------------
# Resolution smoothing (separable FFT convolution, from accc.py)
# ---------------------------------------------------------------------------

@jit
def _apply_resolution_smoothing(density: jnp.ndarray, sigma_voxels: float) -> jnp.ndarray:
    """Apply 3-D Gaussian blur for target resolution (separable FFT convolution).

    Args:
        density:      (nz, ny, nx) density in z-y-x order.
        sigma_voxels: sigma of the resolution kernel **in voxel units**.

    Returns:
        Smoothed density of the same shape and dtype.
    """
    truncate = 4.0
    max_radius = 30
    x = jnp.arange(-max_radius, max_radius + 1, dtype=jnp.float32)
    kernel = jnp.exp(-0.5 * (x / sigma_voxels) ** 2)
    mask = jnp.abs(x) <= truncate * sigma_voxels
    kernel = kernel * mask
    kernel = kernel / jnp.sum(kernel)

    # Convolve along each axis (z=0, y=1, x=2)
    out = signal.fftconvolve(density, kernel[jnp.newaxis, jnp.newaxis, :], mode="same")
    out = signal.fftconvolve(out,     kernel[jnp.newaxis, :, jnp.newaxis], mode="same")
    out = signal.fftconvolve(out,     kernel[:, jnp.newaxis, jnp.newaxis], mode="same")
    return out


# ---------------------------------------------------------------------------
# High-level density computation
# ---------------------------------------------------------------------------

def calc_cg_density(
    coords: jnp.ndarray,
    masses: jnp.ndarray,
    radii: jnp.ndarray,
    bins: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    resolution: Optional[float] = None,
    group_by_radius: bool = True,
) -> jnp.ndarray:
    """Generate a 3-D density map from coarse-grained beads.

    Each bead is painted as a normalised 3-D Gaussian whose sigma is
    derived from its radius (radius-of-gyration equivalence).  An
    optional resolution kernel is convolved on top.

    Args:
        coords:     (N, 3) bead centres in Å (JAX array).
        masses:     (N,) bead masses in Da (JAX array).
        radii:      (N,) bead radii in Å (JAX array).
        bins:       (binsx, binsy, binsz) bin-edge arrays from setup_grid.
        resolution: target map resolution in Å.  If None, no resolution
                    smoothing is applied (you get the intrinsic bead density).
        group_by_radius: if True (default), group beads by unique radius and
                    use the efficient einsum kernel.  Set False to use the
                    lax.scan fallback (handles continuous radius distributions).

    Returns:
        density: (nz, ny, nx) JAX array in mrcfile z-y-x convention.
    """
    grid_x, grid_y, grid_z = grid_centers_from_bins(bins)
    nz, ny, nx = grid_z.shape[0], grid_y.shape[0], grid_x.shape[0]

    # Convert radii → Gaussian sigmas in Å
    sigmas = jnp.array([radius_to_sigma(float(r)) for r in np.asarray(radii)])

    if group_by_radius:
        # Group beads by unique sigma for efficient batched computation
        unique_sigmas = jnp.unique(sigmas)
        density = jnp.zeros((nz, ny, nx), dtype=jnp.float32)

        for s in unique_sigmas:
            s_val = float(s)
            mask = jnp.isclose(sigmas, s)
            idx = jnp.where(mask, size=int(jnp.sum(mask)))[0]
            grp_centers = coords[idx]
            grp_masses = masses[idx]
            density = density + _gaussian_blob_density_same_sigma(
                grid_x, grid_y, grid_z, grp_centers, grp_masses, s_val
            )
    else:
        density = _gaussian_blob_density_general(
            grid_x, grid_y, grid_z, coords, masses, sigmas
        )

    # Optional resolution smoothing
    if resolution is not None:
        voxel_size = float(bins[0][1] - bins[0][0])
        sigma_res = resolution_to_sigma(resolution, voxel_size)
        density = _apply_resolution_smoothing(density, sigma_res)

    return density


def calc_cg_density_from_dict(
    coords_dict: Dict[str, jnp.ndarray],
    mass_dict: Dict[str, float],
    radius_dict: Dict[str, float],
    bins: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    resolution: Optional[float] = None,
) -> jnp.ndarray:
    """Convenience wrapper: build density from per-type dictionaries.

    This matches the ParticleSystem convention where coords, masses,
    and radii are keyed by particle-type name.

    Args:
        coords_dict:  {'A': (n_A, 3), 'B': (n_B, 3), ...}
        mass_dict:    {'A': mass_per_bead, ...}  in Da.
        radius_dict:  {'A': radius, ...}  in Å.
        bins:         bin-edge tuple from setup_grid.
        resolution:   optional resolution in Å.

    Returns:
        density: (nz, ny, nx) JAX array.
    """
    all_coords = []
    all_masses = []
    all_radii = []

    for key in sorted(coords_dict.keys()):
        c = coords_dict[key]
        n = c.shape[0]
        all_coords.append(c)
        all_masses.append(jnp.full((n,), mass_dict[key], dtype=jnp.float32))
        all_radii.append(jnp.full((n,), radius_dict[key], dtype=jnp.float32))

    coords = jnp.concatenate(all_coords, axis=0)
    masses = jnp.concatenate(all_masses, axis=0)
    radii = jnp.concatenate(all_radii, axis=0)

    return calc_cg_density(coords, masses, radii, bins, resolution=resolution)


# ---------------------------------------------------------------------------
# Pearson cross-correlation (from accc.py, unchanged)
# ---------------------------------------------------------------------------

@jit
def pairwise_correlation_jax(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Pearson correlation coefficient between two flat arrays."""
    A = A.astype(jnp.float32)
    B = B.astype(jnp.float32)
    A_c = A - jnp.mean(A)
    B_c = B - jnp.mean(B)
    num = jnp.sum(A_c * B_c)
    den = jnp.sqrt(jnp.sum(A_c ** 2) * jnp.sum(B_c ** 2))
    return jnp.where(den > 0, num / den, 0.0)


# ---------------------------------------------------------------------------
# Masked correlation variants (from accc.py)
# ---------------------------------------------------------------------------

@jit
def _masked_correlation(proj_flat, density_flat, mask):
    """Correlation over masked voxels (shared helper, JIT-compiled)."""
    proj_m = jnp.where(mask, proj_flat, 0.0)
    dens_m = jnp.where(mask, density_flat, 0.0)
    weights = mask.astype(jnp.float32)
    n_valid = jnp.sum(weights)
    n_safe = jnp.maximum(n_valid, 1.0)

    p_mean = jnp.sum(proj_m) / n_safe
    d_mean = jnp.sum(dens_m) / n_safe

    pc = (proj_m - p_mean) * weights
    dc = (dens_m - d_mean) * weights

    num = jnp.sum(pc * dc)
    den = jnp.sqrt(jnp.sum(pc ** 2) * jnp.sum(dc ** 2))
    return jnp.where((den > 0) & (n_valid >= 2), num / den, 0.0)


@jit
def compare_data_jax_full(projection, density_data, contour_level=None):
    if contour_level is not None:
        density_data = jnp.clip(density_data, contour_level, None)
    return pairwise_correlation_jax(projection.flatten(), density_data.flatten())


@jit
def compare_data_jax_original_positive(projection, density_data, contour_level=None):
    if contour_level is not None:
        density_data = jnp.clip(density_data, contour_level, None)
    pf = projection.flatten()
    df = density_data.flatten()
    return _masked_correlation(pf, df, df > 0)


@jit
def compare_data_jax_simulated_positive(projection, density_data, contour_level=None):
    if contour_level is not None:
        density_data = jnp.clip(density_data, contour_level, None)
    pf = projection.flatten()
    df = density_data.flatten()
    return _masked_correlation(pf, df, pf > 0)


@jit
def compare_data_jax_intersection(projection, density_data, contour_level=None):
    if contour_level is not None:
        density_data = jnp.clip(density_data, contour_level, None)
    pf = projection.flatten()
    df = density_data.flatten()
    return _masked_correlation(pf, df, (pf > 0) & (df > 0))


_CCC_DISPATCH = {
    "full": compare_data_jax_full,
    "original_positive": compare_data_jax_original_positive,
    "simulated_positive": compare_data_jax_simulated_positive,
    "intersection": compare_data_jax_intersection,
}


# ---------------------------------------------------------------------------
# High-level comparison API
# ---------------------------------------------------------------------------

def compare_densities(
    density_a: jnp.ndarray,
    density_b: jnp.ndarray,
    contour_level: Optional[float] = None,
    mask_mode: Union[str, List[str], None] = None,
) -> Union[float, Dict[str, float]]:
    """Compute cross-correlation coefficient between two density maps.

    Args:
        density_a: simulated / model density (nz, ny, nx).
        density_b: reference / target density  (nz, ny, nx).
        contour_level: optional contour threshold applied to density_b.
        mask_mode: 'full', 'original_positive', 'simulated_positive',
                   'intersection', or a list of these.
                   None → all four modes.

    Returns:
        Single float if one mode, else dict {mode: ccc}.
    """
    if mask_mode is None:
        modes = list(_CCC_DISPATCH.keys())
    elif isinstance(mask_mode, str):
        modes = [mask_mode]
    else:
        modes = list(mask_mode)

    results = {}
    for m in modes:
        fn = _CCC_DISPATCH[m]
        results[m] = float(fn(density_a, density_b, contour_level))

    if len(results) == 1:
        return next(iter(results.values()))
    return results


# ---------------------------------------------------------------------------
# MRC file I/O
# ---------------------------------------------------------------------------

def write_map(
    density: jnp.ndarray,
    voxel_size: float,
    output_path: str,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    bins: Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = None,
) -> None:
    """Write a density map to MRC format.

    The map is written in the standard mrcfile z-y-x convention.

    Args:
        density:     (nz, ny, nx) JAX or numpy array.
        voxel_size:  uniform voxel size in Å.
        output_path: file path for the output .map/.mrc file.
        origin:      (ox, oy, oz) map origin in Å (used only if bins is None).
        bins:        optional bin-edge tuple – if provided, origin and
                     nxstart/nystart/nzstart are derived from the bin edges.
    """
    # To numpy
    try:
        density_np = np.array(jax.device_get(density), dtype=np.float32)
    except Exception:
        density_np = np.array(density, dtype=np.float32)

    nz, ny, nx = density_np.shape

    with mrcfile.new(output_path, data=density_np, overwrite=True) as mrc:
        mrc.voxel_size = voxel_size

        # Cell dimensions (Å)
        mrc.header.cella.x = nx * voxel_size
        mrc.header.cella.y = ny * voxel_size
        mrc.header.cella.z = nz * voxel_size
        mrc.header.cellb.alpha = 90.0
        mrc.header.cellb.beta = 90.0
        mrc.header.cellb.gamma = 90.0

        # Grid sampling = number of voxels
        mrc.header.mx = nx
        mrc.header.my = ny
        mrc.header.mz = nz

        # Axis mapping (standard: X=1, Y=2, Z=3)
        mrc.header.mapc = 1
        mrc.header.mapr = 2
        mrc.header.maps = 3

        if bins is not None:
            # Derive nstart from bin edges
            # bins[0] = x-edges, bins[1] = y-edges, bins[2] = z-edges
            # nxstart = floor(first_x_edge / voxel_size)
            mrc.header.nxstart = int(round(float(bins[0][0]) / voxel_size))
            mrc.header.nystart = int(round(float(bins[1][0]) / voxel_size))
            mrc.header.nzstart = int(round(float(bins[2][0]) / voxel_size))

            # Origin in Å (column-start positions)
            mrc.header.origin.x = float(bins[0][0])
            mrc.header.origin.y = float(bins[1][0])
            mrc.header.origin.z = float(bins[2][0])
        else:
            ox, oy, oz = origin
            mrc.header.nxstart = int(round(ox / voxel_size))
            mrc.header.nystart = int(round(oy / voxel_size))
            mrc.header.nzstart = int(round(oz / voxel_size))
            mrc.header.origin.x = ox
            mrc.header.origin.y = oy
            mrc.header.origin.z = oz

        mrc.update_header_from_data()

    logging.info(f"CG density map written to: {output_path}")


def read_map(fname: str):
    """Open an MRC density map (thin wrapper around mrcfile.open)."""
    density = mrcfile.open(fname)
    try:
        assert density.voxel_size.x == density.voxel_size.y == density.voxel_size.z
    except AssertionError:
        logging.error("Non-uniform grids are not supported")
        density.close()
        sys.exit(1)
    return density


def bins_from_mrc(density, nonzero: bool = False, contour_level=None):
    """Derive bin-edge arrays from an mrcfile density object (mirrors accc.py)."""
    if nonzero:
        subgrid, (z0, y0, x0) = extract_nonzero_subgrid(
            density.data, return_offset=True, contour_level=contour_level
        )
        nz_s, ny_s, nx_s = subgrid.shape
        binsx = (np.linspace(0, nx_s, nx_s + 1) + density.nstart.x + x0) * density.voxel_size.x
        binsy = (np.linspace(0, ny_s, ny_s + 1) + density.nstart.y + y0) * density.voxel_size.y
        binsz = (np.linspace(0, nz_s, nz_s + 1) + density.nstart.z + z0) * density.voxel_size.z
    else:
        binsx = (np.linspace(0, density.header.nx, density.header.nx + 1) + density.nstart.x) * density.voxel_size.x
        binsy = (np.linspace(0, density.header.ny, density.header.ny + 1) + density.nstart.y) * density.voxel_size.y
        binsz = (np.linspace(0, density.header.nz, density.header.nz + 1) + density.nstart.z) * density.voxel_size.z

    return (
        jnp.array(binsx, dtype=jnp.float32),
        jnp.array(binsy, dtype=jnp.float32),
        jnp.array(binsz, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Sub-grid extraction (from accc.py)
# ---------------------------------------------------------------------------

def extract_nonzero_subgrid(data, return_offset=False, contour_level=None):
    """Extract bounding-box sub-grid of non-zero voxels.

    Args:
        data:          3-D array (z, y, x).
        return_offset: also return (z0, y0, x0) start indices.
        contour_level: clip below this value to 0 before finding bbox.

    Returns:
        subgrid or (subgrid, (z0, y0, x0)).
    """
    try:
        data_np = np.array(jax.device_get(data)).copy()
    except Exception:
        data_np = np.array(data).copy()

    if contour_level is not None:
        data_np[data_np < contour_level] = 0.0

    nz_idx = np.nonzero(data_np)
    if len(nz_idx[0]) == 0:
        empty = np.array([[[0.0]]])
        return (empty, (0, 0, 0)) if return_offset else empty

    z0, z1 = int(nz_idx[0].min()), int(nz_idx[0].max()) + 1
    y0, y1 = int(nz_idx[1].min()), int(nz_idx[1].max()) + 1
    x0, x1 = int(nz_idx[2].min()), int(nz_idx[2].max()) + 1

    sub = data_np[z0:z1, y0:y1, x0:x1].copy()
    return (sub, (z0, y0, x0)) if return_offset else sub


# ---------------------------------------------------------------------------
# Utility: quick self-test / demo
# ---------------------------------------------------------------------------

def _demo():
    """Quick smoke test with the A₈B₈C₁₆ toy system."""
    from particle_system import get_ideal_coords

    ideal = get_ideal_coords()

    # Particle definitions: mass in Da (approximate MW of representative
    # PSD proteins – placeholder values), radius in Å.
    type_info = {
        "A": {"mass": 100_000.0, "radius": 30.0, "copy": 8},
        "B": {"mass": 80_000.0,  "radius": 25.0, "copy": 8},
        "C": {"mass": 40_000.0,  "radius": 15.0, "copy": 16},
    }

    # Flatten into arrays
    all_coords, all_masses, all_radii = [], [], []
    for k in sorted(type_info):
        c = ideal[k]
        n = c.shape[0]
        all_coords.append(c)
        all_masses.append(jnp.full((n,), type_info[k]["mass"], dtype=jnp.float32))
        all_radii.append(jnp.full((n,), type_info[k]["radius"], dtype=jnp.float32))

    coords = jnp.concatenate(all_coords, axis=0)
    masses = jnp.concatenate(all_masses, axis=0)
    radii = jnp.concatenate(all_radii, axis=0)

    # Setup grid – box must enclose all beads plus padding
    pad = 60.0  # Å of padding around extreme coordinates
    cmin = np.array(coords).min(axis=0) - pad
    cmax = np.array(coords).max(axis=0) + pad
    box_size = tuple(float(cmax[i] - cmin[i]) for i in range(3))
    center = tuple(float((cmax[i] + cmin[i]) / 2.0) for i in range(3))
    voxel_size = 3.0  # Å

    bins, grid_shape = setup_grid(box_size, voxel_size, center=center)
    print(f"Grid shape: {grid_shape}")

    # Generate density (bead Gaussians only)
    density_intrinsic = calc_cg_density(coords, masses, radii, bins, resolution=None)
    print(f"Intrinsic density: min={float(density_intrinsic.min()):.4g}, "
          f"max={float(density_intrinsic.max()):.4g}")

    # Generate density at 10 Å resolution
    density_10A = calc_cg_density(coords, masses, radii, bins, resolution=10.0)
    print(f"10 Å density: min={float(density_10A.min()):.4g}, "
          f"max={float(density_10A.max()):.4g}")

    # Self-CCC (should be 1.0)
    ccc_self = compare_densities(density_10A, density_10A, mask_mode="full")
    print(f"Self-CCC: {ccc_self:.6f}")

    # CCC between intrinsic and 10 Å
    ccc_res = compare_densities(density_intrinsic, density_10A, mask_mode="full")
    print(f"Intrinsic vs 10 Å CCC: {ccc_res:.6f}")

    # Write maps
    write_map(density_intrinsic, voxel_size, "cg_intrinsic.mrc", bins=bins)
    write_map(density_10A, voxel_size, "cg_10A.mrc", bins=bins)
    print("Maps written: cg_intrinsic.mrc, cg_10A.mrc")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()