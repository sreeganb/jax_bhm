#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic density map from particle coordinates.

This script provides full control over density map generation for any given
coordinates. By default, it generates a map from the ideal ground state
coordinates of the particle system.

Usage:
    # Generate default ideal coordinate density
    python generate_synthetic_density.py
    
    # Custom parameters
    python generate_synthetic_density.py \
        --resolution 50.0 \
        --voxel_size 5.0 \
        --box_size 300.0 \
        --output output/synthetic_ideal_density.mrc
"""

import numpy as np
import sys
import os
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import jax.numpy as jnp
import jax

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representation.particle_system import ParticleSystem, get_ideal_coords
from scoring.compute_ccc import (
    setup_grid,
    calc_cg_density,
    calc_cg_density_from_dict,
    write_map,
    grid_centers_from_bins,
)


@dataclass
class DensityConfig:
    """Configuration for density map generation."""
    # Resolution and voxel size
    resolution: float = 50.0  # Target resolution in Angstroms
    voxel_size: float = 5.0   # Voxel size in Angstroms
    
    # Box dimensions
    box_size: Tuple[float, float, float] = (300.0, 300.0, 300.0)
    box_center: Tuple[float, float, float] = (0.0, 0.0, -30.0)  # Center around ideal coords
    
    # Output
    output_path: str = "output/synthetic_ideal_density.mrc"


def get_types_config() -> Dict[str, Dict[str, float]]:
    """Get particle type configuration with radii and masses."""
    return {
        'A': {'radius': 24.0, 'copy': 8, 'mass': 50000.0},
        'B': {'radius': 14.0, 'copy': 8, 'mass': 25000.0},
        'C': {'radius': 16.0, 'copy': 16, 'mass': 30000.0},
    }


def generate_density_from_coords(
    coords: Dict[str, jnp.ndarray],
    types_config: Dict[str, Dict[str, float]],
    config: DensityConfig,
    verbose: bool = True,
) -> Tuple[jnp.ndarray, tuple, tuple]:
    """
    Generate a density map from particle coordinates.
    
    Args:
        coords: Dictionary of coordinates per particle type
        types_config: Particle configuration with radius/mass/copy info
        config: Density generation configuration
        verbose: Print progress information
        
    Returns:
        density: (nz, ny, nx) density array
        bins: Grid bin edges (binsx, binsy, binsz)
        grid_shape: (nx, ny, nz) grid dimensions
    """
    if verbose:
        print("=" * 60)
        print("Generating Density Map")
        print("=" * 60)
        print(f"  Resolution:  {config.resolution:.1f} Å")
        print(f"  Voxel size:  {config.voxel_size:.1f} Å")
        print(f"  Box size:    {config.box_size}")
        print(f"  Box center:  {config.box_center}")
        print(f"  Output:      {config.output_path}")
        print("")
    
    # Setup grid
    bins, grid_shape = setup_grid(
        box_size=config.box_size,
        voxel_size=config.voxel_size,
        center=config.box_center,
    )
    
    if verbose:
        print(f"Grid setup:")
        print(f"  Grid shape:  {grid_shape}")
        print(f"  X range:     [{float(bins[0][0]):.1f}, {float(bins[0][-1]):.1f}] Å")
        print(f"  Y range:     [{float(bins[1][0]):.1f}, {float(bins[1][-1]):.1f}] Å")
        print(f"  Z range:     [{float(bins[2][0]):.1f}, {float(bins[2][-1]):.1f}] Å")
        print("")
    
    # Build mass and radius dictionaries
    mass_dict = {k: types_config[k]['mass'] for k in types_config}
    radius_dict = {k: types_config[k]['radius'] for k in types_config}
    
    # Generate density
    if verbose:
        print("Computing density...")
        for k, c in coords.items():
            print(f"  {k}: {c.shape[0]} particles, radius={types_config[k]['radius']:.1f} Å")
    
    density = calc_cg_density_from_dict(
        coords_dict=coords,
        mass_dict=mass_dict,
        radius_dict=radius_dict,
        bins=bins,
        resolution=config.resolution,
    )
    
    if verbose:
        print(f"\nDensity statistics:")
        print(f"  Shape:       {density.shape}")
        print(f"  Min:         {float(jnp.min(density)):.6f}")
        print(f"  Max:         {float(jnp.max(density)):.6f}")
        print(f"  Mean:        {float(jnp.mean(density)):.6f}")
        print(f"  Sum:         {float(jnp.sum(density)):.2f}")
        print(f"  Non-zero:    {int(jnp.sum(density > 0))} voxels")
    
    return density, bins, grid_shape


def save_density(
    density: jnp.ndarray,
    bins: tuple,
    config: DensityConfig,
    verbose: bool = True,
) -> None:
    """Save density map to MRC file."""
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    write_map(
        density=density,
        voxel_size=config.voxel_size,
        output_path=str(output_path),
        bins=bins,
    )
    
    if verbose:
        print(f"\nDensity map saved to: {output_path}")
        print("=" * 60)


def main(
    resolution: float = 50.0,
    voxel_size: float = 5.0,
    box_size: float = 300.0,
    output_path: str = "output/synthetic_ideal_density.mrc",
    verbose: bool = True,
):
    """
    Generate synthetic density map from ideal coordinates.
    
    Args:
        resolution: Target resolution in Angstroms
        voxel_size: Voxel size in Angstroms
        box_size: Cubic box size in Angstroms
        output_path: Output MRC file path
        verbose: Print progress information
    """
    # Get ideal coordinates
    ideal_coords = get_ideal_coords()
    types_config = get_types_config()
    
    if verbose:
        print("\nIdeal coordinate statistics:")
        all_coords = []
        for k, c in ideal_coords.items():
            all_coords.append(np.array(c))
            print(f"  {k}: {c.shape[0]} particles")
            print(f"      X range: [{float(jnp.min(c[:, 0])):.1f}, {float(jnp.max(c[:, 0])):.1f}]")
            print(f"      Y range: [{float(jnp.min(c[:, 1])):.1f}, {float(jnp.max(c[:, 1])):.1f}]")
            print(f"      Z range: [{float(jnp.min(c[:, 2])):.1f}, {float(jnp.max(c[:, 2])):.1f}]")
        
        all_coords = np.vstack(all_coords)
        com = np.mean(all_coords, axis=0)
        print(f"\n  Overall COM: [{com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f}]")
        print(f"  Overall range X: [{np.min(all_coords[:, 0]):.1f}, {np.max(all_coords[:, 0]):.1f}]")
        print(f"  Overall range Y: [{np.min(all_coords[:, 1]):.1f}, {np.max(all_coords[:, 1]):.1f}]")
        print(f"  Overall range Z: [{np.min(all_coords[:, 2]):.1f}, {np.max(all_coords[:, 2]):.1f}]")
    
    # Compute center of mass for box centering
    all_coords_list = []
    for k in sorted(ideal_coords.keys()):
        all_coords_list.append(np.array(ideal_coords[k]))
    all_coords_np = np.vstack(all_coords_list)
    com = np.mean(all_coords_np, axis=0)
    
    # Create configuration
    config = DensityConfig(
        resolution=resolution,
        voxel_size=voxel_size,
        box_size=(box_size, box_size, box_size),
        box_center=(float(com[0]), float(com[1]), float(com[2])),
        output_path=output_path,
    )
    
    # Generate density
    density, bins, grid_shape = generate_density_from_coords(
        coords=ideal_coords,
        types_config=types_config,
        config=config,
        verbose=verbose,
    )
    
    # Save to MRC
    save_density(density, bins, config, verbose=verbose)
    
    return density, bins, grid_shape


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic density map from particle coordinates."
    )
    parser.add_argument('--resolution', type=float, default=50.0,
                       help='Target resolution in Angstroms (default: 50.0)')
    parser.add_argument('--voxel_size', type=float, default=5.0,
                       help='Voxel size in Angstroms (default: 5.0)')
    parser.add_argument('--box_size', type=float, default=300.0,
                       help='Cubic box size in Angstroms (default: 300.0)')
    parser.add_argument('--output', type=str, default='output/synthetic_ideal_density.mrc',
                       help='Output MRC file path')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print(f"JAX backend: {jax.default_backend()}")
    
    main(
        resolution=args.resolution,
        voxel_size=args.voxel_size,
        box_size=args.box_size,
        output_path=args.output,
        verbose=not args.quiet,
    )
