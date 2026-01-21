"""
Example script to run SMC simulation using BlackJAX.
"""
import numpy as np
import sys
import os
from pathlib import Path
import mrcfile

# comment the next line to use GPU/TPU if available
#os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import jax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representation.particle_system import ParticleSystem, get_ideal_coords
from scoring.energy import log_probability
from sampling.smc import run_tempered_smc, get_smc_samples, get_best_sample
from io_utils.io_handlers import save_mcmc_to_hdf5
from scoring.em_score import (
    create_em_config_from_mrcfile,
    create_em_config_from_arrays,
    calc_projection_jax,
    calculate_ccc_score,
    create_em_log_prob_fn,
)

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
    print("Setting up system...")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Define particle system configuration
    types_config = {
        'A': {'radius': 24.0, 'copy': 8},
        'B': {'radius': 14.0, 'copy': 8},
        'C': {'radius': 16.0, 'copy': 16},
    }
    
    ideal_coords = get_ideal_coords()
    
    # 2. Load target density for EM scoring
    mrc_path = output_dir / "simulated_target_density.mrc"
    resolution = 50.0  # Å
    
    print(f"\nLoading target density: {mrc_path}")
    with mrcfile.open(str(mrc_path), mode='r') as mrc:
        em_config = create_em_config_from_mrcfile(mrc, resolution)
        density_data = mrc.data.copy()
        density_voxel_size = float(mrc.voxel_size.x)
        print(f"  Shape: {mrc.data.shape}, Voxel: {density_voxel_size:.2f} Å")
    
    # Compute density center of mass
    density_com = compute_density_center_of_mass(density_data, density_voxel_size)
    print(f"  Density COM: [{density_com[0]:.2f}, {density_com[1]:.2f}, {density_com[2]:.2f}] Å")
    
    # 3. Generate random coordinates
    temp_system = ParticleSystem(types_config, {}, ideal_coords)
    coords = temp_system.get_random_coords(
        jax.random.PRNGKey(42), box_size=[500.0, 500.0, 500.0]
    )
    
    # 4. Shift coordinates COM to match density COM BEFORE creating system
    identity_order = sorted(types_config.keys())
    coords_array = jnp.concatenate([coords[k] for k in identity_order], axis=0)
    coords_com = jnp.mean(coords_array, axis=0)
    shift_vector = density_com - np.array(coords_com)
    
    print(f"  Coords COM before shift: [{coords_com[0]:.2f}, {coords_com[1]:.2f}, {coords_com[2]:.2f}] Å")
    print(f"  Shifting by: [{shift_vector[0]:.2f}, {shift_vector[1]:.2f}, {shift_vector[2]:.2f}] Å")
    
    # Apply shift to coordinates dictionary
    shifted_coords = {}
    for k in coords:
        shifted_coords[k] = coords[k] + jnp.array(shift_vector)
    
    # Verify shift
    shifted_array = jnp.concatenate([shifted_coords[k] for k in identity_order], axis=0)
    shifted_com = jnp.mean(shifted_array, axis=0)
    print(f"  Coords COM after shift: [{shifted_com[0]:.2f}, {shifted_com[1]:.2f}, {shifted_com[2]:.2f}] Å")
    
    # 5. NOW create the system with shifted coordinates
    system = ParticleSystem(types_config, shifted_coords, ideal_coords)
    flat_radii = system.get_flat_radii()
    n_dims = system.total_particles * 3
    print(f"\nSystem: {system.total_particles} particles, {n_dims} dimensions")
    
    # 6. Setup EM scoring
    em_log_prob = create_em_log_prob_fn(em_config, flat_radii, scale=1.0)
    radii_jax = jnp.array(flat_radii, dtype=jnp.float32)
    
    # 7. Restraints
    target_dists = {'AA': 48.2, 'AB': 38.5, 'BC': 34.0}
    nuisance_params = {'AA': 1.6, 'AB': 1.4, 'BC': 1.0}
    
    # 8. Define prior and likelihood separately (required by SMC)
    box_size = 500.0
    
    def log_prior_fn(flat_coords):
        """Uniform prior in box."""
        coords = flat_coords.reshape(-1, 3)
        in_box = jnp.all((coords >= -box_size) & (coords <= box_size))
        return jnp.where(in_box, 0.0, -jnp.inf)
    
    @jax.jit
    def log_likelihood_fn(flat_coords):
        """Combined likelihood: EM score + exclusion volume + pairwise restraints."""
        # EM score (CCC-based)
        ccc = em_log_prob(flat_coords)  # CCC in [-1, 1]
        em_score = -(1.0 - ccc)  # 0 = perfect, -2 = anticorrelated
        
        # Exclusion volume + pairwise restraints
        log_lik = log_probability(
            flat_coords, system, flat_radii,
            target_dists, nuisance_params,
            exclusion_weight=1.0,
            pair_weight=1.0,
            exvol_sigma=0.1
        )
        
        return em_score + log_lik
    
    def log_prob_fn(flat_coords):
        """Combined log probability."""
        return log_prior_fn(flat_coords) + log_likelihood_fn(flat_coords)

    # 9. Initialize particles (n_particles, n_dims) - start from shifted coords with noise
    n_particles = 50
    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)
    
    # Start from shifted coordinates with small Gaussian noise
    flat_shifted = system.flatten(shifted_coords)
    initial_positions = flat_shifted + jax.random.normal(init_key, (n_particles, n_dims)) * 10.0
    
    # Check initial scores
    init_scores = jax.vmap(log_prob_fn)(initial_positions)
    print(f"\nInitial Score (mean): {jnp.mean(init_scores):.2f}")
    
    # 10. Run SMC
    rng_key, smc_key = jax.random.split(rng_key)
    final_state, info_history, best_positions, best_scores = run_tempered_smc(
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_prob_fn=log_prob_fn,
        initial_positions=initial_positions,
        rng_key=smc_key,
        n_mcmc_steps=20,
        rmh_sigma=5.0,
        target_ess=0.7,
        record_best=True,
    )
    
    # 11. Get results
    final_positions = get_smc_samples(final_state)
    best_pos, best_score = get_best_sample(final_state, log_prob_fn)
    
    final_scores = jax.vmap(log_prob_fn)(final_positions)
    print(f"\nFinal Score (mean): {jnp.mean(final_scores):.2f}")
    print(f"Best Score: {best_score:.2f}")
    
    # 12. Save
    if best_positions is not None and best_scores is not None:
        # Report motion between steps
        diffs = np.linalg.norm(np.diff(np.array(best_positions), axis=0), axis=1)
        print(f"Best-step motion: min={diffs.min():.4f}, max={diffs.max():.4f}, mean={diffs.mean():.4f}")

        output_file = output_dir / "smc_trajectory.h5"

        save_mcmc_to_hdf5(
            np.array(best_positions),
            np.array(best_scores),
            1.0,
            str(output_file),
            system,
            params={'method': 'BlackJAX_SMC', 'trajectory': 'best_per_step'},
            convert_to_rmf3=True,
            color_map={'A': (0.2, 0.6, 1.0), 'B': (0.9, 0.4, 0.2), 'C': (0.3, 0.8, 0.4)}
        )

        print(f"\nSaved to {output_file}")
    else:
        output_file = output_dir / "smc_samples.h5"
        save_mcmc_to_hdf5(
            np.array(final_positions),
            np.array(final_scores),
            1.0,
            str(output_file),
            system,
            params={'method': 'BlackJAX_SMC', 'trajectory': 'final_population'},
            convert_to_rmf3=True,
            color_map={'A': (0.2, 0.6, 1.0), 'B': (0.9, 0.4, 0.2), 'C': (0.3, 0.8, 0.4)}
        )
        print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()