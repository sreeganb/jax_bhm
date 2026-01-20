"""
EM density-based sampling using BlackJAX.

This script:
1. Loads a target density from an MRC file (or generates one from ideal coords)
2. Verifies CCC at ideal coordinates
3. Runs annealed RMH sampling starting from random positions
4. Saves trajectory to HDF5

Score: -(1 - CCC) so that 0 = perfect match, -2 = anticorrelated

Usage:
    python run_em_sampling.py                    # Use existing MRC
    python run_em_sampling.py --generate-mrc     # Generate new MRC from ideal coords
    python run_em_sampling.py --n-steps 1000     # Fewer steps for testing
"""
import jax.numpy as jnp
import jax
import numpy as np
import sys
import os
from pathlib import Path
import argparse
import mrcfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representation.particle_system import ParticleSystem, get_ideal_coords
from scoring.energy import log_excluded_volume_kernel, log_probability
from scoring.em_score import (
    create_em_config_from_mrcfile,
    create_em_config_from_arrays,
    calc_projection_jax,
    calculate_ccc_score,
    create_em_log_prob_fn,
)
from sampling.rmh import run_annealed_rmh
from io_utils.io_handlers import save_mcmc_to_hdf5


def generate_density_from_coords(system, coords, resolution, voxel_size, grid_size):
    """Generate density map from coordinates."""
    flat_coords = system.flatten(coords)
    flat_radii = system.get_flat_radii()
    
    half_extent = grid_size * voxel_size / 2
    bins_x = jnp.linspace(-half_extent, half_extent, grid_size + 1)
    bins_y = jnp.linspace(-half_extent, half_extent, grid_size + 1)
    bins_z = jnp.linspace(-half_extent, half_extent, grid_size + 1)
    bins = (bins_x, bins_y, bins_z)
    
    weights = jnp.array(flat_radii) ** 3
    coords_3d = jnp.array(flat_coords).reshape(-1, 3)
    density = calc_projection_jax(coords_3d, weights, bins, resolution)
    
    return np.array(density), voxel_size


def compute_density_center_of_mass(density: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Compute center of mass of density map.
    
    Returns coordinates in Angstroms, centered at origin if map is centered.
    """
    # Get grid dimensions
    nz, ny, nx = density.shape
    
    # Create coordinate grids (centered at origin)
    half_x = nx * voxel_size / 2
    half_y = ny * voxel_size / 2
    half_z = nz * voxel_size / 2
    
    x = np.linspace(-half_x + voxel_size/2, half_x - voxel_size/2, nx)
    y = np.linspace(-half_y + voxel_size/2, half_y - voxel_size/2, ny)
    z = np.linspace(-half_z + voxel_size/2, half_z - voxel_size/2, nz)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Normalize density (use only positive values)
    density_pos = np.maximum(density, 0)
    total_mass = np.sum(density_pos)
    
    if total_mass < 1e-10:
        return np.array([0.0, 0.0, 0.0])
    
    # Compute center of mass
    com_x = np.sum(X * density_pos) / total_mass
    com_y = np.sum(Y * density_pos) / total_mass
    com_z = np.sum(Z * density_pos) / total_mass
    
    return np.array([com_x, com_y, com_z])


def main():
    parser = argparse.ArgumentParser(description="EM density sampling")
    parser.add_argument("--n-steps", type=int, default=2000, help="MCMC steps")
    parser.add_argument("--generate-mrc", action="store_true", help="Generate MRC from ideal coords")
    parser.add_argument("--resolution", type=float, default=50.0, help="Map resolution (Å)")
    parser.add_argument("--grid-size", type=int, default=64, help="Grid size for generated MRC")
    args = parser.parse_args()
    
    print("=" * 60)
    print("EM Density Sampling with BlackJAX")
    print("=" * 60)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # 1. Setup particle system
    # =========================================================================
    types_config = {
        'A': {'radius': 24.0, 'copy': 8},
        'B': {'radius': 14.0, 'copy': 8},
        'C': {'radius': 16.0, 'copy': 16},
    }
    
    target_dists = {'AA': 48.5, 'AB': 38.5, 'BC': 31.0}
    nuisance_params = {'AA': 1.5, 'AB': 1.2, 'BC': 1.0}
    
    ideal_coords = get_ideal_coords()
    system = ParticleSystem(types_config, {}, ideal_coords)
    flat_radii = system.get_flat_radii()
    n_dims = system.total_particles * 3
    
    print(f"\nSystem: {system.total_particles} particles, {n_dims} dimensions")
    
    # =========================================================================
    # 2. Load or generate target density
    # =========================================================================
    mrc_path = output_dir / "simulated_target_density.mrc"
    resolution = args.resolution
    
    if args.generate_mrc or not mrc_path.exists():
        print(f"\nGenerating target density from ideal coordinates...")
        voxel_size = 5.0
        grid_size = args.grid_size
        
        density, voxel_size = generate_density_from_coords(
            system, ideal_coords, resolution, voxel_size, grid_size
        )
        
        print(f"  Resolution: {resolution} Å, Grid: {grid_size}³, Voxel: {voxel_size} Å")
        print(f"  Density range: [{density.min():.4f}, {density.max():.4f}]")
        
        # Save to MRC
        with mrcfile.new(str(mrc_path), overwrite=True) as mrc:
            mrc.set_data(density.astype(np.float32))
            mrc.voxel_size = voxel_size
        print(f"  Saved to: {mrc_path}")
        
        em_config = create_em_config_from_arrays(density, voxel_size, resolution, center_at_origin=True)
        density_data = density
        density_voxel_size = voxel_size
    else:
        print(f"\nLoading target density: {mrc_path}")
        with mrcfile.open(str(mrc_path), mode='r') as mrc:
            em_config = create_em_config_from_mrcfile(mrc, resolution)
            density_data = mrc.data.copy()
            density_voxel_size = float(mrc.voxel_size.x)
            print(f"  Shape: {mrc.data.shape}, Voxel: {density_voxel_size:.2f} Å")
    
    # Compute density center of mass
    density_com = compute_density_center_of_mass(density_data, density_voxel_size)
    print(f"  Density COM: [{density_com[0]:.2f}, {density_com[1]:.2f}, {density_com[2]:.2f}] Å")
    
    # =========================================================================
    # 3. Verify CCC = 1.0 at ideal coordinates
    # =========================================================================
    print("\n--- Verification ---")
    
    flat_ideal = system.flatten(ideal_coords)
    coords_3d = np.array(flat_ideal).reshape(-1, 3)
    ccc_ideal = calculate_ccc_score(coords_3d, flat_radii, em_config)
    
    print(f"CCC at ideal coords: {ccc_ideal:.6f}")
    if ccc_ideal > 0.99:
        print("✓ PASSED: CCC ≈ 1.0")
    else:
        print(f"✗ WARNING: Expected CCC ≈ 1.0, got {ccc_ideal:.4f}")
    
    # =========================================================================
    # 4. Define log probability: -(1 - CCC) + excluded volume
    # =========================================================================
    em_log_prob = create_em_log_prob_fn(em_config, flat_radii, scale=1.0)
    radii_jax = jnp.array(flat_radii, dtype=jnp.float32)
    
    @jax.jit
    def log_prob_fn(flat_coords: jnp.ndarray) -> float:
        ccc = em_log_prob(flat_coords)  # CCC in [-1, 1]
        score = -(1.0 - ccc)  # 0 = perfect, -2 = anticorrelated
                
        log_lik = log_probability(
            flat_coords, system, flat_radii,
            target_dists, nuisance_params,
            exclusion_weight=1.0,  # Softer exclusion
            pair_weight=1.0, 
            exvol_sigma=0.1  # Larger sigma = softer
        )

        # Add pair score to total
        score += log_lik
        
        return score
    
    # Verify score at ideal
    score_ideal = float(log_prob_fn(jnp.array(flat_ideal)))
    print(f"Score at ideal coords: {score_ideal:.4f} (should be ≈ 0)")
    
    # =========================================================================
    # 5. Start from random coordinates, aligned to density COM
    # =========================================================================
    print("\n--- Sampling Setup ---")
    
    rng_key = jax.random.PRNGKey(42)
    
    # Get random starting position in a box centered at origin
    box_size = [300.0, 300.0, 300.0]
    random_coords = system.get_random_coords(rng_key, box_size=box_size)
    flat_random = system.flatten(random_coords)
    
    # Compute particle center of mass
    coords_3d_random = np.array(flat_random).reshape(-1, 3)
    particle_com = np.mean(coords_3d_random, axis=0)
    
    print(f"Random coords COM: [{particle_com[0]:.2f}, {particle_com[1]:.2f}, {particle_com[2]:.2f}] Å")
    print(f"Density COM:       [{density_com[0]:.2f}, {density_com[1]:.2f}, {density_com[2]:.2f}] Å")
    
    # Shift particles so their COM matches density COM
    shift_vector = density_com - particle_com
    print(f"Shifting by:       [{shift_vector[0]:.2f}, {shift_vector[1]:.2f}, {shift_vector[2]:.2f}] Å")
    
    # Apply shift to all particles
    coords_3d_shifted = coords_3d_random + shift_vector
    initial_position = jnp.array(coords_3d_shifted.flatten(), dtype=jnp.float32)
    
    # Verify shift
    new_com = np.mean(coords_3d_shifted, axis=0)
    print(f"After shift COM:   [{new_com[0]:.2f}, {new_com[1]:.2f}, {new_com[2]:.2f}] Å")
    
    # Check initial score
    score_random = float(log_prob_fn(initial_position))
    ccc_random = calculate_ccc_score(coords_3d_shifted, flat_radii, em_config)
    
    print(f"\nInitial state - CCC: {ccc_random:.4f}, Score: {score_random:.4f}")
    
    # =========================================================================
    # 6. Run annealed RMH sampling
    # =========================================================================
    print("\n--- Running Annealed RMH ---")
    
    rng_key, sample_key = jax.random.split(rng_key)
    
    positions, log_probs, acceptance_rate = run_annealed_rmh(
        rng_key=sample_key,
        log_prob_fn=log_prob_fn,
        initial_position=initial_position,
        n_steps=args.n_steps,
        sigma=1.0,  # Slightly larger steps for exploration
        temp_start=10.0,
        temp_end=1.0,
        save_every=10,
        verbose=True,
    )
    
    # =========================================================================
    # 7. Results
    # =========================================================================
    print("\n--- Results ---")
    
    best_idx = np.argmax(log_probs)
    best_coords = positions[best_idx].reshape(-1, 3)
    best_ccc = calculate_ccc_score(best_coords, flat_radii, em_config)
    
    final_coords = positions[-1].reshape(-1, 3)
    final_ccc = calculate_ccc_score(final_coords, flat_radii, em_config)
    
    print(f"Samples: {len(log_probs)}")
    print(f"Acceptance: {acceptance_rate:.1%}")
    print(f"Best score: {log_probs[best_idx]:.4f} (CCC: {best_ccc:.4f})")
    print(f"Final score: {log_probs[-1]:.4f} (CCC: {final_ccc:.4f})")
    
    # =========================================================================
    # 8. Save trajectory
    # =========================================================================
    output_file = output_dir / "em_trajectory.h5"
    
    save_mcmc_to_hdf5(
        positions=positions,
        log_probs=log_probs,
        acceptance_rate=acceptance_rate,
        filename=str(output_file),
        system_template=system,
        params={'method': 'Annealed_RMH_EM', 'resolution': resolution},
        convert_to_rmf3=True,
        color_map={'A': (0.2, 0.6, 1.0), 'B': (0.9, 0.4, 0.2), 'C': (0.3, 0.8, 0.4)}
    )
    
    print(f"\nSaved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()