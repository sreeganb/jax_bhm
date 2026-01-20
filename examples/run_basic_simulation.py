"""
Example script to run a simulation.
"""
import jax.numpy as jnp
import jax
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representation.particle_system import ParticleSystem, get_ideal_coords
from scoring.energy import log_probability
from sampling.mcmc import setup_rmh, run_chain
from io_utils.io_handlers import save_mcmc_to_hdf5

def main():
    print("Setting up system...")
    
    # 1. Define System
    # Using the structure from ParticleSystem definition
    types = {
        'A': {'radius': 1.0, 'copy': 8},
        'B': {'radius': 1.2, 'copy': 8},
        'C': {'radius': 0.8, 'copy': 16}
    }
    
    # Initialize with ideal coords + some noise
    ideal = get_ideal_coords()
    k_order = sorted(types.keys())
    
    # Flatten ideal coords
    init_coords_list = []
    for k in k_order:
        init_coords_list.append(ideal[k])
    init_coords_flat = jnp.concatenate(init_coords_list, axis=0)
    
    # Add noise
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, init_coords_flat.shape) * 2.0
    start_coords = init_coords_flat + noise
    
    system = ParticleSystem.create(types, start_coords)
    flat_radii = system.get_flat_radii()
    
    print(f"System created with {start_coords.shape[0]} particles.")
    
    # 2. Define Scoring
    # Target distances (simulating experimental restraints)
    target_dists = {
        'AA': 48.22,
        'AB': 38.5, 
        'BC': 34.0
    }
    nuisance = {
        'AA': 2.0,
        'AB': 2.0,
        'BC': 2.0
    }
    
    # Define log_prob function closure
    # Note: blackjax expects func(position) -> float
    def log_prob_fn(pos):
        return log_probability(
            pos, 
            system, 
            flat_radii, 
            target_dists, 
            nuisance
        )
    
    # Test score
    init_score = log_prob_fn(start_coords)
    print(f"Initial Score: {init_score}")
    
    # 3. Setup Sampler
    step_size = 0.5 
    kernel = setup_rmh(log_prob_fn, step_size)
    
    # 4. Run Sampling
    n_steps = 5000
    positions, log_probs, acc_rate = run_chain(
        jax.random.PRNGKey(0),
        kernel,
        start_coords,
        n_steps,
        log_prob_fn=log_prob_fn
    )
    
    # 5. Save Output
    output_file = "simulation_output.h5"
    save_mcmc_to_hdf5(
        positions,
        log_probs,
        acc_rate,
        output_file,
        system,
        params={'target_dists': target_dists, 'nuisance': nuisance}
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
