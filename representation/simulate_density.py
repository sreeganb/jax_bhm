#!/usr/bin/env python3
"""
Create simulated density map from ideal coordinates.
"""

import numpy as np
import mrcfile
import scipy.ndimage
import math
from pathlib import Path


def get_ideal_coords():
    """Return ideal coordinates for particle types."""
    array_A = np.array([
        [63., 0., 0.],
        [44.55, 44.55, 0.],
        [0., 63., 0.],
        [-44.55, 44.55, 0.],
        [-63., 0., 0.],
        [-44.55, -44.55, 0.],
        [0., -63., 0.],
        [44.55, -44.55, 0.]
    ])
    array_B = np.array([
        [63., 0., -38.5],
        [44.55, 44.55, -38.5],
        [0., 63., -38.5],
        [-44.55, 44.55, -38.5],
        [-63., 0., -38.5],
        [-44.55, -44.55, -38.5],
        [0., -63., -38.5],
        [44.55, -44.55, -38.5]
    ])
    array_C = np.array([
        [47.00, 0.00, -68.50],
        [79.00, 0.00, -68.50],
        [55.86, 55.86, -68.50],
        [33.23, 33.23, -68.50],
        [0.00, 47.00, -68.50],
        [0.00, 79.00, -68.50],
        [-55.86, 55.86, -68.50],
        [-33.23, 33.23, -68.50],
        [-47.00, 0.00, -68.50],
        [-79.00, 0.00, -68.50],
        [-55.86, -55.86, -68.50],
        [-33.23, -33.23, -68.50],
        [0.00, -47.00, -68.50],
        [0.00, -79.00, -68.50],
        [55.86, -55.86, -68.50],
        [33.23, -33.23, -68.50],
    ])
    return {'A': array_A, 'B': array_B, 'C': array_C}


def get_types_config():
    """Particle types with radii."""
    return {
        'A': {'radius': 24.0, 'copy': 8},
        'B': {'radius': 14.0, 'copy': 8},
        'C': {'radius': 16.0, 'copy': 16},
    }


def resolution_to_sigma(resolution: float, pixel_size: float) -> float:
    """Estimate sigma for gaussian smoothing from resolution."""
    return resolution / (4 * math.sqrt(2.0 * math.log(2.0))) / pixel_size


def create_simulated_density(
    coords_dict: dict,
    types_config: dict,
    resolution: float,
    grid_spacing: float,
    padding: float = 50.0,
    output_file: str = "simulated_density.mrc"
):
    """
    Create simulated density map from particle coordinates.
    
    Args:
        coords_dict: Dict mapping particle type to (N, 3) coordinates
        types_config: Dict with 'radius' for each particle type
        resolution: Map resolution in Angstroms
        grid_spacing: Voxel size in Angstroms
        padding: Padding around structure in Angstroms
        output_file: Output MRC filename
    """
    # Collect all coordinates and weights (radius^3 for volume)
    all_coords = []
    all_weights = []
    
    for ptype in sorted(coords_dict.keys()):
        coords = coords_dict[ptype]
        radius = types_config[ptype]['radius']
        weight = radius ** 3  # Volume-based weight
        
        all_coords.append(coords)
        all_weights.extend([weight] * len(coords))
        print(f"  Type {ptype}: {len(coords)} particles, radius={radius:.1f}, weight={weight:.1f}")
    
    coords = np.concatenate(all_coords, axis=0)
    weights = np.array(all_weights, dtype=np.float32)
    
    print(f"\nTotal particles: {len(coords)}")
    print(f"Coordinate range:")
    print(f"  X: [{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}]")
    print(f"  Y: [{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]")
    print(f"  Z: [{coords[:, 2].min():.1f}, {coords[:, 2].max():.1f}]")
    
    # Create grid centered at origin with padding
    min_coords = coords.min(axis=0) - padding
    max_coords = coords.max(axis=0) + padding
    
    # Make grid symmetric around origin for cleaner handling
    extent = max(abs(min_coords).max(), abs(max_coords).max())
    
    # Create bin edges (centered at origin)
    n_bins = int(2 * extent / grid_spacing) + 1
    half_extent = (n_bins * grid_spacing) / 2
    
    bins_x = np.linspace(-half_extent, half_extent, n_bins + 1)
    bins_y = np.linspace(-half_extent, half_extent, n_bins + 1)
    bins_z = np.linspace(-half_extent, half_extent, n_bins + 1)
    
    print(f"\nGrid parameters:")
    print(f"  Resolution: {resolution:.1f} Å")
    print(f"  Grid spacing: {grid_spacing:.1f} Å")
    print(f"  Grid size: {n_bins} x {n_bins} x {n_bins}")
    print(f"  Grid extent: [{-half_extent:.1f}, {half_extent:.1f}] Å")
    
    # Create weighted histogram
    histogram, _ = np.histogramdd(coords, bins=[bins_x, bins_y, bins_z], weights=weights)
    
    # Swap axes to match MRC convention (z, y, x)
    histogram = np.swapaxes(histogram, 0, 2)
    
    # Apply Gaussian smoothing
    sigma = resolution_to_sigma(resolution, grid_spacing)
    print(f"  Gaussian sigma: {sigma:.2f} voxels")
    
    density = scipy.ndimage.gaussian_filter(histogram, sigma, truncate=4.0)
    density = density.astype(np.float32)
    
    print(f"  Density range: [{density.min():.4f}, {density.max():.4f}]")
    
    # Save as MRC
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(density)
        mrc.voxel_size = grid_spacing
        
        # Set origin (corner of the grid)
        mrc.header.origin.x = -half_extent
        mrc.header.origin.y = -half_extent
        mrc.header.origin.z = -half_extent
        
        # nstart values for grid offset
        mrc.header.nxstart = int(-half_extent / grid_spacing)
        mrc.header.nystart = int(-half_extent / grid_spacing)
        mrc.header.nzstart = int(-half_extent / grid_spacing)
    
    print(f"\nSaved density map: {output_file}")
    return density


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create simulated density from ideal coordinates")
    parser.add_argument("--resolution", type=float, default=40.0,
                        help="Map resolution in Angstroms (default: 40)")
    parser.add_argument("--grid-spacing", type=float, default=4.0,
                        help="Voxel size in Angstroms (default: 4)")
    parser.add_argument("--padding", type=float, default=50.0,
                        help="Padding around structure in Angstroms (default: 50)")
    parser.add_argument("--output", type=str, default="output/simulated_target_density.mrc",
                        help="Output MRC file path")
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Creating Simulated Density Map")
    print("=" * 60)
    
    coords_dict = get_ideal_coords()
    types_config = get_types_config()
    
    print("\nParticle configuration:")
    create_simulated_density(
        coords_dict=coords_dict,
        types_config=types_config,
        resolution=args.resolution,
        grid_spacing=args.grid_spacing,
        padding=args.padding,
        output_file=args.output
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()