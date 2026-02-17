#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging
import math
import numpy as np
import scipy.ndimage
import mrcfile
import argparse
from types import SimpleNamespace

# Optional: Import IMP if available, but allow the script to run without it for the mrcfile parts
try:
    import IMP
    import IMP.em
    HAS_IMP = True
except ImportError:
    HAS_IMP = False
    logging.warning("IMP module not found. IMP-specific map writing will be skipped.")

# --- Backend Selection (GPU/CPU) ---
BACKEND = 'cpu'
try:
    import cupy as cp
    import cupyx.scipy.ndimage
    BACKEND = 'gpu'
    print('>> CuPy is available, using GPU backend.')
except ImportError:
    print('>> CuPy is unavailable, using NumPy backend for CPU.')

# =====================================================================
# CORE MATH FUNCTIONS
# =====================================================================

def resolution_to_sigma(resolution: float, pixel_size: float) -> float:
    """
    Converts resolution (FWHM) to Gaussian sigma in pixels.
    Formula: Sigma = Resolution / (2 * sqrt(2 * ln(2)))
    """
    fwhm_factor = 2 * math.sqrt(2. * math.log(2.)) # ~2.355
    sigma_pixels = (resolution / fwhm_factor) / pixel_size
    return sigma_pixels

def generate_density_grid(coords, weights, resolution, voxel_size, box_size, particle_radius_approx=0.0):
    """
    Generates a 3D density grid from coordinates using histogramming + gaussian blur.
    
    Args:
        coords: (N,3) array of coordinates
        weights: (N,) array of weights (e.g. radius^3)
        resolution: Target resolution in Angstroms
        voxel_size: Angstroms per pixel
        box_size: Total box side length in Angstroms
        particle_radius_approx: (Optional) If > 0, adds this value to the Gaussian width
                                to simulate solid spheres rather than point sources.
    
    Returns:
        density: 3D numpy array
        bins: list of bin edges (x, y, z)
    """
    # 1. Define the grid
    # We center the box at (0,0,0)
    grid_dim = int(math.ceil(box_size / voxel_size))
    half_box = (grid_dim * voxel_size) / 2.0
    
    # Create bin edges: from -half_box to +half_box
    bins_1d = np.linspace(-half_box, half_box, grid_dim + 1)
    bins = [bins_1d, bins_1d, bins_1d]
    
    # 2. Histogram (Binning particles into voxels)
    # This treats every particle as a point source at its center.
    if BACKEND == 'gpu':
        coords_gpu = cp.asarray(coords)
        weights_gpu = cp.asarray(weights)
        bins_gpu = cp.asarray(bins)
        
        # histogramdd on GPU
        img_, _ = cp.histogramdd(coords_gpu, weights=weights_gpu, bins=bins_gpu)
        img_ = cp.swapaxes(img_, 0, 2) # Swap to match standard EM (z, y, x) ordering conventions often used
        
        # 3. Gaussian Blur (Simulating Resolution)
        # Standard sigma for resolution
        sigma_res = resolution_to_sigma(resolution, voxel_size)
        
        # Optional: Add particle radius to sigma to reduce "disconnected balls" look
        # Sigma_total = sqrt(sigma_res^2 + sigma_particle^2)
        sigma_total = sigma_res
        if particle_radius_approx > 0:
            sigma_part = particle_radius_approx / voxel_size
            sigma_total = math.sqrt(sigma_res**2 + sigma_part**2)
            
        density = cupyx.scipy.ndimage.gaussian_filter(img_, sigma_total, truncate=4).astype(cp.float32)
        return cp.asnumpy(density), bins
        
    else:
        # CPU Version
        img_, _ = np.histogramdd(coords, weights=weights, bins=bins)
        img_ = np.swapaxes(img_, 0, 2)
        
        sigma_res = resolution_to_sigma(resolution, voxel_size)
        
        # Logic to fix "disconnected balls":
        sigma_total = sigma_res
        if particle_radius_approx > 0:
            sigma_part = particle_radius_approx / voxel_size
            sigma_total = math.sqrt(sigma_res**2 + sigma_part**2)
            
        density = scipy.ndimage.gaussian_filter(img_, sigma_total, truncate=4).astype(np.float32)
        return density, bins

# =====================================================================
# SCORING FUNCTION
# =====================================================================

def calculate_ccc_score(sphere_coords, sphere_radii, target_density_map, resolution):
    """
    Calculates CCC score.
    """
    # 1. Setup Mock Model
    mock_model = SimpleNamespace()
    mock_model.getCoords = lambda: sphere_coords
    mock_model.getMasses = lambda: sphere_radii**3 # Valid mass approximation

    # 2. Extract Target Data
    # Assuming target_density_map is an mrcfile object
    target_data = target_density_map.data
    
    # We must generate the model density ON THE SAME GRID as the target.
    # The target map defines the bins/box.
    
    # Extract grid info from MRC header
    nx, ny, nz = target_density_map.header.nx, target_density_map.header.ny, target_density_map.header.nz
    voxel_size = target_density_map.voxel_size.x
    
    # Calculate physical origin (MRC usually stores origin in nxstart/nystart or origin field)
    # For correlation, we just need the grid definitions to match.
    # We assume the target map is centered at 0,0,0 if created by this script.
    # If loading an experimental map, we need strict origin matching.
    
    # Construct bins from the target map header
    # origin + i * voxel_size
    start_x = target_density_map.header.origin.x
    start_y = target_density_map.header.origin.y
    start_z = target_density_map.header.origin.z
    
    # Note: If origin is 0, check nxstart. 
    # For this script's specific generated maps, we know how bins are made.
    # We will use the generic 'compare_data' logic which rebuilds bins from header.
    
    if BACKEND == 'gpu':
        return compare_data_gpu(target_density_map, mock_model, resolution)
    else:
        return compare_data_cpu(target_density_map, mock_model, resolution)

def compare_data_cpu(density, model, resolution: float) -> float:
    # Re-create bins matching the density object
    bins = bins_from_density(density)
    
    coords = model.getCoords()
    weights = model.getMasses()
    
    # Generate projection on the exact same bins
    img_, _ = np.histogramdd(coords, weights=weights, bins=bins)
    img_ = np.swapaxes(img_, 0, 2)
    
    voxel_size = bins[0][1] - bins[0][0]
    sigma = resolution_to_sigma(resolution, voxel_size)
    
    # Note: We do NOT use particle_radius_approx here strictly, 
    # because CCC is usually calculated against the "pure" resolution projection 
    # unless you want to score against a "filled" map.
    projection = scipy.ndimage.gaussian_filter(img_, sigma, truncate=4).astype(np.float32)
    
    # Normalize and Correlate
    ccc = pairwise_correlation_cpu(projection.flatten(), density.data.flatten())
    return ccc

def compare_data_gpu(density, model, resolution: float) -> float:
    bins = bins_from_density(density)
    coords = cp.asarray(model.getCoords())
    weights = cp.asarray(model.getMasses())
    bins_gpu = cp.asarray(bins)
    
    img_, _ = cp.histogramdd(coords, weights=weights, bins=bins_gpu)
    img_ = cp.swapaxes(img_, 0, 2)
    
    voxel_size = bins[0][1] - bins[0][0]
    sigma = resolution_to_sigma(resolution, float(voxel_size))
    projection = cupyx.scipy.ndimage.gaussian_filter(img_, sigma, truncate=4).astype(cp.float32)
    
    density_data_gpu = cp.asarray(density.data)
    ccc = pairwise_correlation_gpu(projection.flatten(), density_data_gpu.flatten())
    return float(cp.asnumpy(ccc))

def pairwise_correlation_cpu(A, B):
    am = A - np.mean(A)
    bm = B - np.mean(B)
    return np.sum(am * bm) / (np.sqrt(np.sum(am**2)) * np.sqrt(np.sum(bm**2)))

def pairwise_correlation_gpu(A, B):
    am = A - cp.mean(A)
    bm = B - cp.mean(B)
    return cp.sum(am * bm) / (cp.sqrt(cp.sum(am**2)) * cp.sqrt(cp.sum(bm**2)))

# =====================================================================
# UTILITIES
# =====================================================================

def bins_from_density(density) -> list:
    """Calculates bin edges based on MRC header info."""
    # Note: This assumes the map origin is correctly set in header.origin 
    # or that the map is centered (standard for IMP/Chimera workflows)
    
    # Check if origin is set, otherwise calculate from nxstart
    ox = density.header.origin.x
    oy = density.header.origin.y
    oz = density.header.origin.z
    
    # If origin is 0 but nxstart is negative, recalculate physical start
    if ox == 0 and density.header.nxstart != 0:
        ox = density.header.nxstart * density.voxel_size.x
        oy = density.header.nystart * density.voxel_size.y
        oz = density.header.nzstart * density.voxel_size.z

    binsx = np.linspace(ox, ox + density.header.nx * density.voxel_size.x, density.header.nx + 1)
    binsy = np.linspace(oy, oy + density.header.ny * density.voxel_size.y, density.header.ny + 1)
    binsz = np.linspace(oz, oz + density.header.nz * density.voxel_size.z, density.header.nz + 1)
    return (binsx, binsy, binsz)

def save_mrc_file(density, voxel_size, origin, filename="map.mrc", resolution_label=0.0):
    """Saves numpy density array to MRC file with correct metadata."""
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(density.astype(np.float32))
        mrc.voxel_size = voxel_size
        
        # Set Origin (critical for fitting)
        # origin is (x_start, y_start, z_start)
        mrc.header.origin.x = origin[0]
        mrc.header.origin.y = origin[1]
        mrc.header.origin.z = origin[2]
        
        # Set nxstart (grid indices) relative to (0,0,0)
        # standard is to set origin and let software handle it, 
        # or set nxstart = origin / voxel_size
        mrc.header.nxstart = int(round(origin[0] / voxel_size))
        mrc.header.nystart = int(round(origin[1] / voxel_size))
        mrc.header.nzstart = int(round(origin[2] / voxel_size))
        
        # Labels
        mrc.header.label[0] = f"Res: {resolution_label:.2f}A".ljust(80).encode('utf-8')
        mrc.update_header_stats()
    print(f">> Saved MRC: {filename}")

def save_imp_map(density, bins, voxel_size, resolution, filename="imp_map.mrc"):
    """Saves density using IMP (if available)"""
    if not HAS_IMP:
        return
        
    nx, ny, nz = density.shape
    
    # Bounding Box from bins
    bbox = IMP.algebra.BoundingBox3D(
        IMP.algebra.Vector3D(bins[0][0], bins[1][0], bins[2][0]),
        IMP.algebra.Vector3D(bins[0][-1], bins[1][-1], bins[2][-1])
    )
    
    model_map = IMP.em.create_density_map(bbox, voxel_size)
    
    # Fast update of grid (avoid set_value loop)
    # We can access the raw grid if supported, otherwise loop is needed 
    # but we can check if IMP exposes a buffer interface (often tricky).
    # For safety/compatibility with older IMP, we use the loop but only if small
    # OR we trust mrcfile to do the writing usually. 
    # Here is the loop, but note it is slow for large maps.
    
    # Optimization: IMP map usually iterates x,y,z
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Check indices - IMP might use different ordering vs numpy
                # Usually IMP is (x,y,z) accessing
                val = float(density[i, j, k])
                if val > 1e-6: # Sparse optimization
                    model_map.set_value(i, j, k, val)
                    
    model_map.get_header_writable().set_resolution(resolution)
    model_map.calcRMS()
    IMP.em.write_map(model_map, filename)
    print(f">> Saved IMP map: {filename}")

# =====================================================================
# MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    # 1. SETUP MODEL
    print("--- Setting up the toy model ---")
    
    # Cleaned up array definitions
    array_A = np.array([
        [63.,0.,0.],[44.55,44.55,0.],[0.,63.,0.],[-44.55,44.55,0.],
        [-63.,0.,0.],[-44.55,-44.55,0.],[-0.,-63.,0.],[44.55,-44.55,0.]
    ])
    array_B = np.array([
        [63.,0.,-38.5],[44.55,44.55,-38.5],[0.,63.,-38.5],[-44.55,44.55,-38.5],
        [-63.,0.,-38.5],[-44.55,-44.55,-38.5],[-0.,-63.,-38.5],[44.55,-44.55,-38.5]
    ])
    array_C = np.array([
        [47.,0.,-68.5],[79.,0.,-68.5],[55.86,55.86,-68.5],[33.23,33.23,-68.5],
        [0.,47.,-68.5],[0.,79.,-68.5],[-55.86,55.86,-68.5],[-33.23,33.23,-68.5],
        [-47.,0.,-68.5],[-79.,0.,-68.5],[-55.86,-55.86,-68.5],[-33.23,-33.23,-68.5],
        [0.,-47.,-68.5],[0.,-79.,-68.5],[55.86,-55.86,-68.5],[33.23,-33.23,-68.5]
    ])

    radius_A, radius_B, radius_C = 24.0, 14.0, 16.0
    
    ideal_coords = np.vstack([array_A, array_B, array_C])
    ideal_radii = np.concatenate([
        np.full(len(array_A), radius_A),
        np.full(len(array_B), radius_B),
        np.full(len(array_C), radius_C)
    ])
    
    # 2. GENERATE MAP
    RESOLUTION = 50.0  
    VOXEL_SIZE = 3.0   # 3.0 is very fine for 50A resolution (Nyquist=10). 5.0 is sufficient.
    BOX_SIZE = 300.0   # Increased slightly to ensure padding
    
    # Weights = Volume = r^3
    weights = ideal_radii**3
    
    print(f"Generating map at {RESOLUTION}A resolution...")
    
    # TRICK FOR "DISCONNECTED BALLS":
    # Pass mean radius to particle_radius_approx to widen the blobs
    mean_radius = np.mean(ideal_radii)
    
    density, bins = generate_density_grid(
        ideal_coords, weights, 
        RESOLUTION, VOXEL_SIZE, BOX_SIZE,
        particle_radius_approx=mean_radius # <--- Try setting this to 0 to see the difference!
    )
    
    # Save using mrcfile (Faster, standard)
    origin = (bins[0][0], bins[1][0], bins[2][0])
    save_mrc_file(density, VOXEL_SIZE, origin, "target_map_filled.mrc", RESOLUTION)
    
    # Save using IMP (if you need it for IMP internal formats)
    if HAS_IMP:
        save_imp_map(density, bins, VOXEL_SIZE, RESOLUTION, "target_map_imp.mrc")
        
    # 3. SCORE
    print("\n--- Scoring ---")
    # Load the map back to prove scoring works with file IO
    target_map_obj = mrcfile.open("target_map_filled.mrc")
    
    score = calculate_ccc_score(ideal_coords, ideal_radii, target_map_obj, RESOLUTION)
    print(f"CCC Score (Self): {score:.6f}")
    
    target_map_obj.close()