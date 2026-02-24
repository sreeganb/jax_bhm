#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a synthetic EM density map using the SAME density pipeline
as the scoring function (scoring/compute_ccc.py).

This guarantees that CCC(ideal_coords, target_map) ≈ 1.0.

The previous generate_em_map.py used histogramdd + scipy.gaussian_filter,
which produces a different density than the JAX-based Gaussian blob
representation in compute_ccc.calc_cg_density. That mismatch caused
ideal CCC ~ 0.81 instead of ~1.0.

Usage:
    python generate_em_map_consistent.py
    python generate_em_map_consistent.py --resolution 40 --voxel_size 4.0
    python generate_em_map_consistent.py --output output/my_target.mrc
"""

import sys
import os
import argparse
import math
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp

from representation.particle_system import get_ideal_coords
from scoring.compute_ccc import (
    setup_grid,
    calc_cg_density,
    write_map,
    compare_densities,
)


# =============================================================================
# Particle system definition (must match what the test/scoring code uses)
# =============================================================================

TYPES_CONFIG = {
    'A': {'radius': 24.0, 'copy': 8, 'mass': 50000.0},
    'B': {'radius': 14.0, 'copy': 8, 'mass': 25000.0},
    'C': {'radius': 16.0, 'copy': 16, 'mass': 30000.0},
}


def build_flat_arrays(types_config, ideal_coords_dict):
    """Build flattened coordinate, mass, and radii arrays in sorted type order."""
    order = sorted(types_config.keys())
    all_coords = []
    all_masses = []
    all_radii = []

    for k in order:
        c = ideal_coords_dict[k]
        n = c.shape[0]
        all_coords.append(c)
        all_masses.append(jnp.full((n,), types_config[k]['mass'], dtype=jnp.float32))
        all_radii.append(jnp.full((n,), types_config[k]['radius'], dtype=jnp.float32))

    coords = jnp.concatenate(all_coords, axis=0)
    masses = jnp.concatenate(all_masses, axis=0)
    radii = jnp.concatenate(all_radii, axis=0)

    return coords, masses, radii, order


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic EM density using the same pipeline as the scorer."
    )
    parser.add_argument("--resolution", type=float, default=50.0,
                        help="Target resolution in Angstroms (default: 50.0)")
    parser.add_argument("--voxel_size", type=float, default=3.0,
                        help="Voxel size in Angstroms (default: 3.0)")
    parser.add_argument("--box_size", type=float, default=300.0,
                        help="Box size in Angstroms (default: 300.0)")
    parser.add_argument("--padding", type=float, default=60.0,
                        help="Padding around coordinates in Angstroms (default: 60.0)")
    parser.add_argument("--auto_box", action="store_true", default=False,
                        help="Auto-compute box from coordinate extent + padding")
    parser.add_argument("--output", "-o", type=str,
                        default=None,
                        help="Output MRC file path")
    parser.add_argument("--no_resolution_smooth", action="store_true",
                        help="Skip resolution smoothing (intrinsic bead density only)")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Verify self-CCC after generation (default: True)")
    args = parser.parse_args()

    print("=" * 70)
    print("Consistent Synthetic EM Map Generator")
    print("Uses scoring/compute_ccc.calc_cg_density for density generation")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")

    # -------------------------------------------------------------------------
    # 1. Load ideal coordinates
    # -------------------------------------------------------------------------
    print("\n[1] Loading ideal coordinates...")
    ideal_dict = get_ideal_coords()
    coords, masses, radii, order = build_flat_arrays(TYPES_CONFIG, ideal_dict)

    n_particles = coords.shape[0]
    print(f"  Particles: {n_particles}")
    print(f"  Type order: {order}")
    for k in order:
        n = TYPES_CONFIG[k]['copy']
        r = TYPES_CONFIG[k]['radius']
        m = TYPES_CONFIG[k]['mass']
        print(f"    {k}: {n} copies, radius={r}Å, mass={m}")

    # Coordinate statistics
    coords_np = np.array(coords)
    coord_min = coords_np.min(axis=0)
    coord_max = coords_np.max(axis=0)
    coord_center = (coord_min + coord_max) / 2.0
    coord_extent = coord_max - coord_min

    print(f"  Coord range: [{coord_min[0]:.1f}, {coord_max[0]:.1f}] x "
          f"[{coord_min[1]:.1f}, {coord_max[1]:.1f}] x "
          f"[{coord_min[2]:.1f}, {coord_max[2]:.1f}]")
    print(f"  Coord center: ({coord_center[0]:.1f}, {coord_center[1]:.1f}, {coord_center[2]:.1f})")
    print(f"  Coord extent: ({coord_extent[0]:.1f}, {coord_extent[1]:.1f}, {coord_extent[2]:.1f})")

    # -------------------------------------------------------------------------
    # 2. Setup grid
    # -------------------------------------------------------------------------
    print(f"\n[2] Setting up grid...")

    if args.auto_box:
        # Compute box size from coordinate extent + padding + max radius
        max_radius = float(radii.max())
        pad = args.padding + max_radius
        box_size = tuple(float(coord_extent[i] + 2 * pad) for i in range(3))
        center = tuple(float(coord_center[i]) for i in range(3))
        print(f"  Auto box: padding={args.padding}Å + max_radius={max_radius}Å")
    else:
        box_size = (args.box_size, args.box_size, args.box_size)
        center = (0.0, 0.0, 0.0)

    bins, grid_shape = setup_grid(box_size, args.voxel_size, center=center)

    print(f"  Box size: {box_size}")
    print(f"  Center: {center}")
    print(f"  Voxel size: {args.voxel_size} Å")
    print(f"  Grid shape: {grid_shape}")
    print(f"  X range: [{float(bins[0][0]):.1f}, {float(bins[0][-1]):.1f}]")
    print(f"  Y range: [{float(bins[1][0]):.1f}, {float(bins[1][-1]):.1f}]")
    print(f"  Z range: [{float(bins[2][0]):.1f}, {float(bins[2][-1]):.1f}]")

    # Verify all coordinates are inside grid
    inside_x = (coord_min[0] >= float(bins[0][0])) and (coord_max[0] <= float(bins[0][-1]))
    inside_y = (coord_min[1] >= float(bins[1][0])) and (coord_max[1] <= float(bins[1][-1]))
    inside_z = (coord_min[2] >= float(bins[2][0])) and (coord_max[2] <= float(bins[2][-1]))

    if inside_x and inside_y and inside_z:
        print("  ✓ All coordinates are inside the grid")
    else:
        print("  ✗ WARNING: Some coordinates are OUTSIDE the grid!")
        print("    Increase --box_size or use --auto_box")

    # -------------------------------------------------------------------------
    # 3. Generate density using calc_cg_density
    # -------------------------------------------------------------------------
    resolution = None if args.no_resolution_smooth else args.resolution

    print(f"\n[3] Generating density...")
    print(f"  Method: calc_cg_density (Gaussian blobs)")
    print(f"  Resolution smoothing: {resolution if resolution else 'None (intrinsic only)'}")

    density = calc_cg_density(
        coords, masses, radii, bins,
        resolution=resolution,
        group_by_radius=True,
    )
    jax.block_until_ready(density)

    density_np = np.array(density)
    print(f"  Density shape: {density_np.shape}")
    print(f"  Density range: [{density_np.min():.6g}, {density_np.max():.6g}]")
    print(f"  Density mean:  {density_np.mean():.6g}")
    print(f"  Non-zero voxels: {np.count_nonzero(density_np > 1e-10)} / {density_np.size}")

    # -------------------------------------------------------------------------
    # 4. Verify self-CCC
    # -------------------------------------------------------------------------
    if args.verify:
        print(f"\n[4] Verifying self-CCC...")

        # Regenerate density from ideal coords (should be identical)
        density_check = calc_cg_density(
            coords, masses, radii, bins,
            resolution=resolution,
            group_by_radius=True,
        )

        ccc_self = compare_densities(density_check, density, mask_mode='full')
        print(f"  Self-CCC (regenerated vs target): {ccc_self:.8f}")

        if ccc_self > 0.999:
            print(f"  ✓ Self-CCC is {ccc_self:.6f} — pipeline is self-consistent!")
        else:
            print(f"  ✗ WARNING: Self-CCC is only {ccc_self:.6f}")
            print(f"    This should not happen — check for numerical issues.")

        # Also test with grouped=False (lax.scan) to make sure both paths match
        density_scan = calc_cg_density(
            coords, masses, radii, bins,
            resolution=resolution,
            group_by_radius=False,
        )
        ccc_scan = compare_densities(density_scan, density, mask_mode='full')
        print(f"  CCC (lax.scan vs grouped): {ccc_scan:.8f}")

    # -------------------------------------------------------------------------
    # 5. Save MRC file
    # -------------------------------------------------------------------------
    if args.output is None:
        res_str = f"{int(args.resolution)}ang" if resolution else "intrinsic"
        vox_str = f"{args.voxel_size:.0f}vox"
        output_path = str(
            project_root / "output" / f"synthetic_ideal_density_{res_str}_{vox_str}.mrc"
        )
    else:
        output_path = args.output

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[5] Saving MRC file...")
    print(f"  Output: {output_path}")

    write_map(density, args.voxel_size, output_path, bins=bins)

    # Verify the saved file can be loaded and scores correctly
    print(f"\n[6] Verification: loading saved MRC and scoring...")

    import mrcfile
    with mrcfile.open(output_path, mode='r') as mrc:
        loaded_density = jnp.array(mrc.data, dtype=jnp.float32)
        loaded_voxel = float(mrc.voxel_size.x)
        loaded_origin = (
            float(mrc.header.origin.x),
            float(mrc.header.origin.y),
            float(mrc.header.origin.z),
        )
        loaded_shape = mrc.data.shape

    print(f"  Loaded: shape={loaded_shape}, voxel={loaded_voxel}Å, "
          f"origin=({loaded_origin[0]:.1f}, {loaded_origin[1]:.1f}, {loaded_origin[2]:.1f})")

    # Reconstruct bins from loaded MRC
    nz, ny, nx = loaded_shape
    loaded_bins = (
        jnp.linspace(loaded_origin[0], loaded_origin[0] + nx * loaded_voxel, nx + 1, dtype=jnp.float32),
        jnp.linspace(loaded_origin[1], loaded_origin[1] + ny * loaded_voxel, ny + 1, dtype=jnp.float32),
        jnp.linspace(loaded_origin[2], loaded_origin[2] + nz * loaded_voxel, nz + 1, dtype=jnp.float32),
    )

    # Score ideal coords against loaded map
    density_from_ideal = calc_cg_density(
        coords, masses, radii, loaded_bins,
        resolution=resolution,
        group_by_radius=True,
    )

    from scoring.em_log_likelihood import differentiable_ccc
    final_ccc = float(differentiable_ccc(density_from_ideal, loaded_density))
    print(f"  CCC(ideal_coords, loaded_map): {final_ccc:.8f}")

    if final_ccc > 0.999:
        print(f"\n  ✓✓✓ SUCCESS: Ideal CCC = {final_ccc:.6f}")
        print(f"  The map is fully consistent with the scoring function.")
    elif final_ccc > 0.99:
        print(f"\n  ✓ GOOD: Ideal CCC = {final_ccc:.6f} (minor float32 rounding)")
    else:
        print(f"\n  ✗ PROBLEM: Ideal CCC = {final_ccc:.6f}")
        print(f"  There is still a mismatch between generation and scoring.")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Output:       {output_path}")
    print(f"  Resolution:   {args.resolution} Å")
    print(f"  Voxel size:   {args.voxel_size} Å")
    print(f"  Grid shape:   {grid_shape}")
    print(f"  Box size:     {box_size}")
    print(f"  Ideal CCC:    {final_ccc:.6f}")
    print(f"{'=' * 70}")

    return output_path, final_ccc


if __name__ == "__main__":
    main()