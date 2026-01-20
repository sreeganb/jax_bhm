"""
Example script to run a simulation.
"""
import jax.numpy as jnp
import jax
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representation.particle_system import ParticleSystem, get_ideal_coords
from scoring.energy import log_probability
from sampling.mcmc import setup_rmh, run_chain
from io_utils.io_handlers import save_mcmc_to_hdf5

def main():
    print("Setting up system...")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Define particle system
    types_config = {
        'A': {'radius': 24.0, 'copy': 8},
        'B': {'radius': 14.0, 'copy': 8},   # Changed to 8 to match ideal coords
        'C': {'radius': 16.0, 'copy': 16},  # Changed to 16 to match ideal coords
    }
    
    # 2. Get ideal coordinates
    ideal_coords = get_ideal_coords()
    
    # 3. Initialize from random coordinates sampled from inside a box
    box_size = [500.0, 500.0, 500.0]  # Angstrom
    init_key = jax.random.PRNGKey(42)
    coords = ParticleSystem(types_config, {}, ideal_coords).get_random_coords(init_key, box_size)
    
    # 4. Create particle system
    system = ParticleSystem(types_config, coords, ideal_coords)
    print(f"System created with {system.total_particles} particles.")
    
    # 5. Set target distances (restraints) - use minimum observed distances
    # from ideal configuration
    target_dists = {
        'AA': 48.2,  # Minimum distance between A particles
        'AB': 38.5,  # Minimum distance A-B layers  
        'BC': 34.0,  # Minimum distance B-C layers
    }
    
    # 6. Nuisance parameters (sigma values) - larger to allow flexibility
    nuisance_params = {
        'AA': 5.0,  # Allow +/- 5 Angstrom variation
        'AB': 5.0,
        'BC': 5.0,
    }
    
    # 7. Get flat arrays for JAX
    flat_coords = system.flatten(coords)
    flat_radii = system.get_flat_radii()
    
    # 8. Define log probability function
    def log_prob_fn(flat_coords):
        return log_probability(
            flat_coords,
            system,
            flat_radii,
            target_dists,
            nuisance_params,
            exclusion_weight=1.0,  
            pair_weight=1.0
        )
    
    # Check initial score
    init_score = log_prob_fn(flat_coords)
    print(f"Initial Score: {init_score}")
    
    # 9. Setup MCMC with properly scaled step size
    rng_key = jax.random.PRNGKey(0)
    
    # Scale step size to coordinate scale (particles move ~50-100 Angstroms)
    # Want to move ~1-2 Angstroms per coordinate per step
    step_size = 1.0
    
    rmh_kernel = setup_rmh(log_prob_fn, step_size)
    
    print(f"Step size: {step_size} Angstrom")
    
    # 10. Run sampling
    n_steps = 5000
    positions, log_probs, acc_rate = run_chain(
        rng_key,
        rmh_kernel,
        flat_coords,
        n_steps,
        log_prob_fn=log_prob_fn
    )
    
    print("\nDone!")
    print(f"Final acceptance rate: {acc_rate:.2%}")
    print(f"Best log prob: {np.max(log_probs):.2f}")
    
    # 11. Save results
    output_file = output_dir / "trajectory.h5"
    
    # Color map for visualization (optional)
    color_map = {
        'A': (0.2, 0.6, 1.0),  # blue
        'B': (0.9, 0.4, 0.2),  # orange
        'C': (0.3, 0.8, 0.4),  # green
    }
    
    save_mcmc_to_hdf5(
        positions,
        log_probs,
        acc_rate,
        str(output_file),
        system,
        params={
            'target_dists': target_dists,
            'nuisance_params': nuisance_params,
            'step_size': step_size,
            'n_steps': n_steps
        },
        convert_to_rmf3=True,  # Enable RMF3 conversion
        color_map=color_map
    )
    
    print(f"\nResults saved to {output_dir}/")
    print(f"  - trajectory.h5 (HDF5 format)")
    print(f"  - trajectory.rmf3 (RMF3 format for visualization)")
    print("\nTo visualize in ChimeraX:")
    print(f"  open {output_dir}/trajectory.rmf3")

if __name__ == "__main__":
    main()