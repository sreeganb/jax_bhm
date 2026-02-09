"""
Run script for HMC sampler
1) Define particle system
2) Define scoring function / log probability
3) Initialize starting position
4) Run HMC sampling
5) Report results
6) Save trajectory

NOTE: in this code, the sigmas are not sampled, but fixed. So there is no prior on the sigmas 
"""
import numpy as np
import sys
import os
from pathlib import Path

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import jax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representation.particle_system import ParticleSystem, get_ideal_coords
from scoring.energy import log_probability
from sampling.hmc import run_hmc_sampling
from io_utils.io_handlers import save_mcmc_to_hdf5
from scoring.log_priors import Priors

def main():
    print("=" * 60)
    print("HMC Sampling with BlackJAX")
    print("=" * 60)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # 1. Define particle system
    # =========================================================================
    types_config = {
        'A': {'radius': 24.0, 'copy': 8},
        'B': {'radius': 14.0, 'copy': 8},
        'C': {'radius': 16.0, 'copy': 16},
    }
    
    ideal_coords = get_ideal_coords()
    
    box_size = 500.0
    
    coords = ParticleSystem(types_config, {}, ideal_coords).get_random_coords(
        jax.random.PRNGKey(2387), box_size=[box_size, box_size, box_size], center_at_origin=True
    )
    
    system = ParticleSystem(types_config, {}, coords)
    flat_radii = system.get_flat_radii()
    n_dims = system.total_particles * 3
    
    print(f"\nSystem: {system.total_particles} particles, {n_dims} dimensions")
    
    # =========================================================================
    # 2. Define scoring function / log probability
    # =========================================================================
    target_dists = {'AA': 48.5, 'AB': 38.5, 'BC': 31.0}
    nuisance_params = {'AA': 1.5, 'AB': 1.2, 'BC': 1.0}  # Softer constraints
    # Sample the nuisance parameters in a full Bayesian treatment, give them inverse gamma priors.
    # Define them as variables and pick them randomly from an interval
#    nuisance_intervals = {'AA': (0.1, 5.0), 'AB': (0.1, 5.0), 'BC': (0.1, 5.0)}
#    # Pick initial nuisance params randomly within intervals
#    for key in nuisance_params:
#        low, high = nuisance_intervals[key]
#        nuisance_params[key] = np.random.uniform(low, high)
    
    @jax.jit
    def log_prob_fn(flat_coords):
        """Combined prior + likelihood with softer penalties."""
        # Prior: Uniform within box
        lower = -box_size
        upper = box_size
        log_prior = jnp.sum(Priors.log_uniform_prior(
            flat_coords, lower_bound=lower, upper_bound=upper
        ))
        
        # Assign inverse gamma priors to nuisance parameters
#        for key in nuisance_params:
#            a = 3.0  # shape
#            scale = 1.0  # scale
#            param = nuisance_params[key]
#            log_prior += Priors.log_inverse_gamma_prior(param, a, scale)
        
        # Likelihood from scoring
        log_lik = log_probability(
            flat_coords, system, flat_radii,
            target_dists, nuisance_params,
            exclusion_weight=1.0,
            pair_weight=4.0, 
            exvol_sigma=0.1
        )
        
        return log_prior + log_lik    
    # =========================================================================
    # 3. Initialize starting position
    # =========================================================================
    initial_position = system.flatten(coords)  
    rng_key = jax.random.PRNGKey(123)
      
    # =========================================================================
    # 4. Run HMC Sampling
    # =========================================================================
    print("\n" + "-" * 60)
    print("Running HMC Sampling...")
    print("-" * 60)
    
    rng_key, sample_key = jax.random.split(rng_key)
    
    # HMC parameters
    # NOTE: Step size must be tuned based on gradient magnitude
    # Rule of thumb: step_size * gradient_norm ≈ 1 for stable integration
    # For this problem, gradient_norm ≈ 4000, so step_size ≈ 0.001-0.01
    n_steps = 500000  # Reduced since HMC is more efficient per step
    step_size = 0.01  # Much smaller step size for stable integration
    save_every = 1000       # Save every 10 steps
    inv_mass = jnp.ones(n_dims)  # Identity mass matrix
    num_int_steps = 50  # More leapfrog steps for better proposals
    
    positions, log_probs, acceptance_rate = run_hmc_sampling(
        rng_key=sample_key,
        log_prob_fn=log_prob_fn,
        initial_position=initial_position,
        n_steps=n_steps,
        step_size=step_size,
        inverse_mass_matrix=inv_mass,
        num_integration_steps=num_int_steps,
        burnin=1000,
        thin=10,
        save_interval=save_every,
        verbose=True,
    )
    
    # =========================================================================
    # 5. Report results
    # =========================================================================
    print("\n" + "-" * 60)
    print("Results Summary")
    print("-" * 60)
    
    best_idx = np.argmax(log_probs)
    print(f"Saved samples: {len(log_probs)}")
    print(f"Acceptance rate: {acceptance_rate:.1%}")
    print(f"Best log probability: {log_probs[best_idx]:.2f}")
    print(f"Final log probability: {log_probs[-1]:.2f}")
    print(f"Mean log probability: {np.mean(log_probs):.2f}")
    
    # Motion between frames
    if len(positions) > 1:
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        print(f"\nFrame motion: min={diffs.min():.2f}, max={diffs.max():.2f}, mean={diffs.mean():.2f}")
    
    # =========================================================================
    # 6. Save trajectory
    # =========================================================================
    output_file = output_dir / "hmc_trajectory.h5"
    
    save_mcmc_to_hdf5(
        positions=positions,
        log_probs=log_probs,
        acceptance_rate=acceptance_rate,
        filename=str(output_file),
        system_template=system,
        params={
            'method': 'BlackJAX_HMC',
            'n_steps': n_steps,
            'step_size': step_size,
            'num_integration_steps': num_int_steps,
        },
        convert_to_rmf3=True,
        color_map={'A': (0.2, 0.6, 1.0), 'B': (0.9, 0.4, 0.2), 'C': (0.3, 0.8, 0.4)}
    )
    
    print(f"\nTrajectory saved to: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run HMC sampling with BlackJAX")
    args = parser.parse_args()
    
    main()
